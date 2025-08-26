# Copyright 2025 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Callable, Mapping, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self

import aiofiles
import httpx
from typing_extensions import Literal

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.emissions import EventEmitterFactory
from parlant.core.loggers import Logger
from parlant.core.nlp.moderation import ModerationService
from parlant.core.nlp.service import NLPService
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper
from parlant.core.services.tools.openapi import OpenAPIClient
from parlant.core.services.tools.plugins import PluginClient
from parlant.core.services.tools.mcp_service import MCPToolClient
from parlant.core.tools import LocalToolService, ToolService
from parlant.core.common import ItemNotFoundError, Version, UniqueId
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
)


ToolServiceKind = Literal["openapi", "sdk", "local", "mcp"]


class ServiceRegistry(ABC):
    """An interface for managing tool services in the engine."""

    @abstractmethod
    async def update_tool_service(
        self,
        name: str,
        kind: ToolServiceKind,
        url: str,
        source: Optional[str] = None,
        transient: bool = False,
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
    async def read_moderation_service(
        self,
        name: str,
    ) -> ModerationService: ...

    @abstractmethod
    async def list_moderation_services(
        self,
    ) -> Sequence[tuple[str, ModerationService]]: ...

    @abstractmethod
    async def read_nlp_service(
        self,
        name: str,
    ) -> NLPService: ...

    @abstractmethod
    async def list_nlp_services(
        self,
    ) -> Sequence[tuple[str, NLPService]]: ...

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
    source: Optional[str]


class ServiceDocumentRegistry(ServiceRegistry):
    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        database: DocumentDatabase,
        event_emitter_factory: EventEmitterFactory,
        logger: Logger,
        correlator: ContextualCorrelator,
        nlp_services_provider: Callable[[], Mapping[str, NLPService]],
        allow_migration: bool = False,
    ):
        self._database = database
        self._tool_services_collection: DocumentCollection[_ToolServiceDocument]

        self._event_emitter_factory = event_emitter_factory
        self._logger = logger
        self._correlator = correlator

        self._nlp_services_provider = nlp_services_provider
        self._nlp_services: Mapping[str, NLPService]

        self._moderation_services: Mapping[str, ModerationService]
        self._exit_stack: AsyncExitStack
        self._running_services: dict[str, ToolService] = {}
        self._service_sources: dict[str, str] = {}

        self._allow_migration = allow_migration
        self._lock = ReaderWriterLock()

    def _cast_to_specific_tool_service_class(
        self,
        service: ToolService,
    ) -> OpenAPIClient | PluginClient | MCPToolClient:
        if not (
            isinstance(service, OpenAPIClient)
            or isinstance(service, PluginClient)
            or isinstance(service, MCPToolClient)
        ):
            raise ValueError("Unsupported ToolService class.")

        return service

    async def _document_loader(self, doc: BaseDocument) -> Optional[_ToolServiceDocument]:
        if doc["version"] == "0.1.0":
            return cast(_ToolServiceDocument, doc)
        return None

    async def __aenter__(self) -> Self:
        self._nlp_services = self._nlp_services_provider()

        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
        ):
            self._tool_services_collection = await self._database.get_or_create_collection(
                name="tool_services",
                schema=_ToolServiceDocument,
                document_loader=self._document_loader,
            )

        self._moderation_services = {
            name: await nlp_service.get_moderation_service()
            for name, nlp_service in self._nlp_services.items()
        }

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        documents = await self._tool_services_collection.find({})

        for document in documents:
            service = await self._deserialize_tool_service(document)
            await self._exit_stack.enter_async_context(
                self._cast_to_specific_tool_service_class(service)
            )
            self._running_services[document["name"]] = service
            if document["source"]:
                self._service_sources[document["name"]] = document["source"]

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
            self._service_sources.clear()
        return False

    async def _get_openapi_json_from_source(self, source: str) -> str:
        if source.startswith("http://") or source.startswith("https://"):
            async with httpx.AsyncClient() as client:
                response = await client.get(source)
                response.raise_for_status()
                return response.text
        else:
            async with aiofiles.open(source, "r") as f:
                return await f.read()

    def _serialize_tool_service(
        self,
        name: str,
        service: ToolService,
    ) -> _ToolServiceDocument:
        kind: ToolServiceKind

        if isinstance(service, OpenAPIClient):
            kind = "openapi"
            url = service.server_url
        elif isinstance(service, PluginClient):
            kind = "sdk"
            url = service.url
        elif isinstance(service, MCPToolClient):
            kind = "mcp"
            url = service.url
        else:
            raise ValueError("Unsupported ToolService class.")

        return _ToolServiceDocument(
            id=ObjectId(name),
            version=self.VERSION.to_string(),
            name=name,
            kind=kind,
            url=url,
            source=self._service_sources.get(name) if isinstance(service, OpenAPIClient) else None,
        )

    async def _deserialize_tool_service(self, document: _ToolServiceDocument) -> ToolService:
        if document["kind"] == "openapi":
            openapi_json = await self._get_openapi_json_from_source(cast(str, document["source"]))

            return OpenAPIClient(
                server_url=document["url"],
                openapi_json=openapi_json,
            )
        elif document["kind"] == "sdk":
            return PluginClient(
                url=document["url"],
                event_emitter_factory=self._event_emitter_factory,
                logger=self._logger,
                correlator=self._correlator,
            )
        elif document["kind"] == "mcp":
            return MCPToolClient(
                url=document["url"],
                event_emitter_factory=self._event_emitter_factory,
                logger=self._logger,
                correlator=self._correlator,
            )
        else:
            raise ValueError("Unsupported ToolService kind.")

    @override
    async def update_tool_service(
        self,
        name: str,
        kind: ToolServiceKind,
        url: str,
        source: Optional[str] = None,
        transient: bool = False,
    ) -> ToolService:
        async with self._lock.writer_lock:
            service: ToolService

            if kind == "local":
                self._running_services[name] = LocalToolService()
                return self._running_services[name]
            elif kind == "openapi":
                assert source
                openapi_json = await self._get_openapi_json_from_source(source)
                service = OpenAPIClient(server_url=url, openapi_json=openapi_json)
                self._service_sources[name] = source
            elif kind == "mcp":
                service = MCPToolClient(
                    url=url,
                    event_emitter_factory=self._event_emitter_factory,
                    logger=self._logger,
                    correlator=self._correlator,
                )
            elif kind == "sdk":
                service = PluginClient(
                    url=url,
                    event_emitter_factory=self._event_emitter_factory,
                    logger=self._logger,
                    correlator=self._correlator,
                )
            else:
                raise ValueError(f"Unsupported ToolService kind: {kind}")

            if name in self._running_services:
                await (
                    self._cast_to_specific_tool_service_class(self._running_services[name])
                ).__aexit__(None, None, None)

            await self._exit_stack.enter_async_context(
                self._cast_to_specific_tool_service_class(service)
            )

            self._running_services[name] = service

        await self._exit_stack.enter_async_context(
            self._cast_to_specific_tool_service_class(service)
        )

        self._running_services[name] = service

        if not transient:
            await self._tool_services_collection.update_one(
                filters={"name": {"$eq": name}},
                params=self._serialize_tool_service(name, service),
                upsert=True,
            )

        return service

    @override
    async def read_tool_service(
        self,
        name: str,
    ) -> ToolService:
        async with self._lock.reader_lock:
            if name not in self._running_services:
                raise ItemNotFoundError(item_id=UniqueId(name))

            return self._running_services[name]

    @override
    async def list_tool_services(
        self,
    ) -> Sequence[tuple[str, ToolService]]:
        async with self._lock.reader_lock:
            return list(self._running_services.items())

    @override
    async def read_moderation_service(
        self,
        name: str,
    ) -> ModerationService:
        if name not in self._moderation_services:
            raise ItemNotFoundError(item_id=UniqueId(name))

        return self._moderation_services[name]

    @override
    async def list_moderation_services(
        self,
    ) -> Sequence[tuple[str, ModerationService]]:
        async with self._lock.reader_lock:
            return list(self._moderation_services.items())

    @override
    async def read_nlp_service(
        self,
        name: str,
    ) -> NLPService:
        async with self._lock.reader_lock:
            if name not in self._nlp_services:
                raise ItemNotFoundError(item_id=UniqueId(name))

            return self._nlp_services[name]

    @override
    async def list_nlp_services(
        self,
    ) -> Sequence[tuple[str, NLPService]]:
        async with self._lock.reader_lock:
            return list(self._nlp_services.items())

    @override
    async def delete_service(self, name: str) -> None:
        async with self._lock.writer_lock:
            if name in self._running_services:
                if isinstance(self._running_services[name], LocalToolService):
                    del self._running_services[name]
                    return

                service = self._running_services[name]
                await (self._cast_to_specific_tool_service_class(service)).__aexit__(
                    None, None, None
                )
                del self._running_services[name]
                if name in self._service_sources:
                    del self._service_sources[name]

            result = await self._tool_services_collection.delete_one({"name": {"$eq": name}})

        if not result.deleted_count:
            raise ItemNotFoundError(item_id=UniqueId(name))
