from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Optional, Self, Sequence, TypedDict, cast
from typing_extensions import Literal

from emcie.server.core.contextual_correlator import ContextualCorrelator
from emcie.server.core.emissions import EventEmitterFactory
from emcie.server.core.services.tools.openapi import OpenAPIClient
from emcie.server.core.services.tools.plugins import PluginClient
from emcie.server.core.tools import ToolService
from emcie.server.core.common import ItemNotFoundError, Version, UniqueId
from emcie.server.core.persistence.document_database import (
    DocumentDatabase,
    ObjectId,
)

ToolServiceKind = Literal["openapi", "sdk"]


class ServiceRegistry(ABC):
    @abstractmethod
    async def update_tool_service(
        self,
        name: str,
        kind: ToolServiceKind,
        url: str,
        openapi_json: Optional[str] = None,
    ) -> ToolService: ...

    @abstractmethod
    async def read_tool_service(
        self,
        name: str,
    ) -> ToolService: ...

    @abstractmethod
    async def list_tool_services(
        self,
    ) -> Sequence[tuple[str, ToolService]]: ...

    @abstractmethod
    async def delete_service(
        self,
        name: str,
    ) -> None: ...


class _ToolServiceDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    name: str
    kind: ToolServiceKind
    url: str
    openapi_json: Optional[str]


class ServiceDocumentRegistry(ServiceRegistry):
    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        database: DocumentDatabase,
        event_emitter_factory: EventEmitterFactory,
        correlator: ContextualCorrelator,
    ):
        self._tool_services_collection = database.get_or_create_collection(
            name="tool_services",
            schema=_ToolServiceDocument,
        )

        self._event_emitter_factory = event_emitter_factory
        self._correlator = correlator
        self._exit_stack: AsyncExitStack
        self._running_services: dict[str, ToolService] = {}

    def _cast_to_specific_tool_service_class(
        self,
        service: ToolService,
    ) -> OpenAPIClient | PluginClient:
        if isinstance(service, OpenAPIClient):
            return service
        else:
            return cast(PluginClient, service)

    async def __aenter__(self) -> Self:
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        documents = await self._tool_services_collection.find({})

        for document in documents:
            service = self._deserialize_tool_service(document)
            await self._exit_stack.enter_async_context(
                self._cast_to_specific_tool_service_class(service)
            )
            self._running_services[document["name"]] = service

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        if self._exit_stack:
            await self._exit_stack.__aexit__(exc_type, exc_value, traceback)
            self._running_services.clear()
        return False

    def _serialize_tool_service(
        self,
        name: str,
        service: ToolService,
    ) -> _ToolServiceDocument:
        return _ToolServiceDocument(
            id=ObjectId(name),
            version=self.VERSION.to_string(),
            name=name,
            kind="openapi" if isinstance(service, OpenAPIClient) else "sdk",
            url=service.server_url
            if isinstance(service, OpenAPIClient)
            else cast(PluginClient, service).url,
            openapi_json=service.openapi_json if isinstance(service, OpenAPIClient) else None,
        )

    def _deserialize_tool_service(self, document: _ToolServiceDocument) -> ToolService:
        if document["kind"] == "openapi":
            return OpenAPIClient(
                server_url=document["url"],
                openapi_json=cast(str, document["openapi_json"]),
            )
        elif document["kind"] == "sdk":
            return PluginClient(
                url=document["url"],
                event_emitter_factory=self._event_emitter_factory,
                correlator=self._correlator,
            )
        else:
            raise ValueError("Unsupported ToolService kind.")

    async def update_tool_service(
        self,
        name: str,
        kind: ToolServiceKind,
        url: str,
        openapi_json: Optional[str] = None,
    ) -> ToolService:
        service: ToolService

        if kind == "openapi":
            assert openapi_json
            service = OpenAPIClient(server_url=url, openapi_json=openapi_json)
        else:
            service = PluginClient(
                url=url,
                event_emitter_factory=self._event_emitter_factory,
                correlator=self._correlator,
            )

        if name in self._running_services:
            await (
                self._cast_to_specific_tool_service_class(self._running_services[name])
            ).__aexit__(None, None, None)

        await self._exit_stack.enter_async_context(
            self._cast_to_specific_tool_service_class(service)
        )

        self._running_services[name] = service

        await self._tool_services_collection.update_one(
            filters={"name": {"$eq": name}},
            params=self._serialize_tool_service(name, service),
            upsert=True,
        )

        return service

    async def read_tool_service(
        self,
        name: str,
    ) -> ToolService:
        if name not in self._running_services:
            raise ItemNotFoundError(item_id=UniqueId(name))
        return self._running_services[name]

    async def list_tool_services(
        self,
    ) -> Sequence[tuple[str, ToolService]]:
        return list(self._running_services.items())

    async def delete_service(self, name: str) -> None:
        if name in self._running_services:
            service = self._running_services[name]
            await (self._cast_to_specific_tool_service_class(service)).__aexit__(None, None, None)
            del self._running_services[name]

        result = await self._tool_services_collection.delete_one({"name": {"$eq": name}})
        if not result.deleted_count:
            raise ItemNotFoundError(item_id=UniqueId(name))
