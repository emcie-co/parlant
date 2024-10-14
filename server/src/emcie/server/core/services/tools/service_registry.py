from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from typing import Optional, Sequence, TypeAlias, TypedDict, cast
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
ServiceName: TypeAlias = str


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
    ) -> Sequence[tuple[ServiceName, ToolService]]: ...

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


class ServiceRegistryDocument(ServiceRegistry):
    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        database: DocumentDatabase,
        event_emitter_factory: EventEmitterFactory,
        correlator: ContextualCorrelator,
        exit_stack: AsyncExitStack,
    ):
        self._tool_services_collection = database.get_or_create_collection(
            name="tool_services",
            schema=_ToolServiceDocument,
        )

        self._event_emitter_factory = event_emitter_factory
        self._correlator = correlator
        self._exit_stack = exit_stack

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
        # TODO: In case a service is running and we override it, the service that is already running needs to exit.
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
            await self._exit_stack.enter_async_context(service)

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
        document = await self._tool_services_collection.find_one({"name": {"$eq": name}})
        if not document:
            raise ItemNotFoundError(item_id=UniqueId(name))

        service = self._deserialize_tool_service(document)

        return service

    async def list_tool_services(
        self,
    ) -> Sequence[tuple[ServiceName, ToolService]]:
        documents = await self._tool_services_collection.find({})

        return [(d["name"], self._deserialize_tool_service(d)) for d in documents]

    async def delete_service(self, name: str) -> None:
        result = await self._tool_services_collection.delete_one({"name": {"$eq": name}})
        if not result.deleted_count:
            raise ItemNotFoundError(item_id=UniqueId(name))
