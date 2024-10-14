from abc import ABC, abstractmethod
from typing import Sequence, TypedDict, cast
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


class ServiceRegistry(ABC):
    @abstractmethod
    async def create_tool_service(
        self,
        service_name: str,
        service: ToolService,
    ) -> None: ...

    @abstractmethod
    async def read_tool_service(
        self,
        service_name: str,
    ) -> ToolService: ...

    @abstractmethod
    async def list_tool_services(
        self,
    ) -> Sequence[ToolService]: ...

    @abstractmethod
    async def delete_service(
        self,
        service_name: str,
    ) -> None: ...


class _BaseToolServiceDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    service_name: str
    service_type: Literal["openapi", "sdk"]


class _OpenAPIClientDocument(_BaseToolServiceDocument, total=False):
    server_url: str
    openapi_json: str


class _PluginClientDocument(_BaseToolServiceDocument, total=False):
    url: str


class ServiceRegistryDocumentStore(ServiceRegistry):
    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        database: DocumentDatabase,
        event_emitter: EventEmitterFactory,
        correlator: ContextualCorrelator,
    ):
        self._tool_services_collection = database.get_or_create_collection(
            name="tool_services",
            schema=_BaseToolServiceDocument,
        )

        self._event_emitter = event_emitter
        self._correlator = correlator

    def _serialize_tool_service(
        self,
        service_name: str,
        service: ToolService,
    ) -> _BaseToolServiceDocument:
        if isinstance(service, OpenAPIClient):
            return _OpenAPIClientDocument(
                id=ObjectId(service_name),
                version=self.VERSION.to_string(),
                service_name=service_name,
                service_type="openapi",
                server_url=service.server_url,
                openapi_json=service.openapi_json,
            )
        elif isinstance(service, PluginClient):
            return _PluginClientDocument(
                id=ObjectId(service_name),
                version=self.VERSION.to_string(),
                service_name=service_name,
                service_type="sdk",
                url=service.url,
            )
        else:
            raise ValueError("Unsupported ToolService type.")

    def _deserialize_tool_service(self, document: _BaseToolServiceDocument) -> ToolService:
        if document["service_type"] == "openapi":
            return OpenAPIClient(
                server_url=cast(_OpenAPIClientDocument, document)["server_url"],
                openapi_json=cast(_OpenAPIClientDocument, document)["openapi_json"],
            )
        elif document["service_type"] == "sdk":
            return PluginClient(
                url=cast(_PluginClientDocument, document)["url"],
                event_emitter_factory=self._event_emitter,
                correlator=self._correlator,
            )
        else:
            raise ValueError("Unsupported ToolService type.")

    async def create_tool_service(
        self,
        service_name: str,
        service: ToolService,
    ) -> None:
        existing_service = await self._tool_services_collection.find_one(
            {"service_name": {"$eq": service_name}}
        )
        if existing_service:
            raise ValueError(f"Service with name '{service_name}' already exists.")

        document = self._serialize_tool_service(service_name, service)

        await self._tool_services_collection.insert_one(document)

    async def read_tool_service(
        self,
        service_name: str,
    ) -> ToolService:
        document = await self._tool_services_collection.find_one(
            {"service_name": {"$eq": service_name}}
        )
        if not document:
            raise ItemNotFoundError(item_id=UniqueId(service_name))

        service = self._deserialize_tool_service(document)

        return service

    async def list_tool_services(
        self,
    ) -> Sequence[ToolService]:
        documents = await self._tool_services_collection.find({})

        return [self._deserialize_tool_service(doc) for doc in documents]

    async def delete_service(self, service_name: str) -> None:
        result = await self._tool_services_collection.delete_one(
            {"service_name": {"$eq": service_name}}
        )
        if not result.deleted_count:
            raise ItemNotFoundError(item_id=UniqueId(service_name))
