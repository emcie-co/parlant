# Copyright 2026 Emcie Co Ltd.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from itertools import chain
from typing import NewType, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self

from parlant.core.async_utils import ReaderWriterLock, safe_gather
from parlant.core.common import (
    ItemNotFoundError,
    try_or_none,
    UniqueId,
    Version,
    IdGenerator,
    xxh3_checksum,
    to_json_dict,
)
from parlant.core.persistence.common import (
    ObjectId,
)
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
)
from parlant.core.persistence.document_database_helper import (
    DocumentMigrationHelper,
    DocumentStoreMigrationHelper,
)
from parlant.core.tags import TagId

AgentId = NewType("AgentId", str)


class CompositionMode(Enum):
    FLUID = "fluid"
    CANNED_FLUID = "canned_fluid"
    CANNED_COMPOSITED = "canned_composited"
    CANNED_STRICT = "canned_strict"


class MessageOutputMode(Enum):
    """Defines how the agent outputs messages."""

    BLOCK = "block"
    """Full message is sent at once (default behavior)."""

    STREAM = "stream"
    """Message is streamed token by token."""


class Effort(Enum):
    """Defines how much effort the agent invests in processing."""

    MIN = "min"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


class AgentUpdateParams(TypedDict, total=False):
    name: str
    description: Optional[str]
    max_engine_iterations: int
    composition_mode: CompositionMode
    message_output_mode: MessageOutputMode
    engine: str
    effort: Effort


@dataclass(frozen=True)
class Agent:
    id: AgentId
    name: str
    description: Optional[str]
    creation_utc: datetime
    last_modified_utc: datetime
    max_engine_iterations: int
    tags: Sequence[TagId]
    engine: str
    composition_mode: CompositionMode = CompositionMode.FLUID
    message_output_mode: MessageOutputMode = MessageOutputMode.BLOCK
    effort: Effort = Effort.MEDIUM


class AgentStore(ABC):
    @abstractmethod
    async def create_agent(
        self,
        name: str,
        description: Optional[str] = None,
        creation_utc: Optional[datetime] = None,
        max_engine_iterations: Optional[int] = None,
        composition_mode: Optional[CompositionMode] = None,
        message_output_mode: Optional[MessageOutputMode] = None,
        engine: Optional[str] = None,
        effort: Optional[Effort] = None,
        tags: Optional[Sequence[TagId]] = None,
        id: Optional[AgentId] = None,
    ) -> Agent: ...

    @abstractmethod
    async def list_agents(
        self,
    ) -> Sequence[Agent]: ...

    @abstractmethod
    async def read_agent(
        self,
        agent_id: AgentId,
    ) -> Agent: ...

    @abstractmethod
    async def update_agent(
        self,
        agent_id: AgentId,
        params: AgentUpdateParams,
    ) -> Agent: ...

    @abstractmethod
    async def delete_agent(
        self,
        agent_id: AgentId,
    ) -> None: ...

    @abstractmethod
    async def upsert_tag(
        self,
        agent_id: AgentId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_tag(
        self,
        agent_id: AgentId,
        tag_id: TagId,
    ) -> None: ...


class _AgentDocument_v0_5_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    name: str
    description: Optional[str]
    max_engine_iterations: int
    composition_mode: str
    message_output_mode: str


class _AgentDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    last_modified: str
    name: str
    description: Optional[str]
    max_engine_iterations: int
    composition_mode: str
    message_output_mode: str
    engine: str
    effort: str


class _AgentTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    agent_id: AgentId
    tag_id: TagId


class AgentDocumentStore(AgentStore):
    VERSION = Version.from_string("0.6.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        database: DocumentDatabase,
        allow_migration: bool = False,
        collections_prefix: str | None = None,
    ):
        self._id_generator = id_generator

        self._database = database

        self._agents_collection: DocumentCollection[_AgentDocument]
        self._tag_association_collection: DocumentCollection[_AgentTagAssociationDocument]
        self._collections_prefix = collections_prefix

        self._allow_migration = allow_migration

        self._lock = ReaderWriterLock()

    async def _document_loader(self, doc: BaseDocument) -> Optional[_AgentDocument]:
        async def v0_1_0_to_v0_2_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        async def v0_2_0_to_v0_3_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        async def v0_3_0_to_v0_4_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(_AgentDocument, doc)

            if doc["version"] == "0.3.0":
                utterance_to_canned_response_composition_mode = {
                    "fluid": CompositionMode.FLUID.value,
                    "fluid_utterance": CompositionMode.CANNED_FLUID.value,
                    "composited_utterance": CompositionMode.CANNED_COMPOSITED.value,
                    "strict_utterance": CompositionMode.CANNED_STRICT.value,
                }

                return _AgentDocument(
                    id=ObjectId(doc["id"]),
                    version=Version.String("0.4.0"),
                    creation_utc=doc["creation_utc"],
                    name=doc["name"],
                    description=doc.get("description"),
                    max_engine_iterations=doc["max_engine_iterations"],
                    composition_mode=utterance_to_canned_response_composition_mode.get(
                        doc["composition_mode"], CompositionMode.FLUID.value
                    ),
                )

            if doc["version"] == "0.4.0":
                return doc

            return None

        async def v0_4_0_to_v0_5_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(_AgentDocument_v0_5_0, doc)

            if doc["version"] == "0.4.0":
                return _AgentDocument_v0_5_0(
                    id=ObjectId(doc["id"]),
                    version=Version.String("0.5.0"),
                    creation_utc=doc["creation_utc"],
                    name=doc["name"],
                    description=doc.get("description"),
                    max_engine_iterations=doc["max_engine_iterations"],
                    composition_mode=doc.get("composition_mode", CompositionMode.FLUID.value),
                    message_output_mode=MessageOutputMode.BLOCK.value,
                )

            if doc["version"] == "0.5.0":
                return doc

            return None

        async def v0_5_0_to_v0_6_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(_AgentDocument_v0_5_0, doc)
            return _AgentDocument(
                id=d["id"],
                version=Version.String("0.6.0"),
                creation_utc=d["creation_utc"],
                last_modified=d["creation_utc"],
                name=d["name"],
                description=d.get("description"),
                max_engine_iterations=d["max_engine_iterations"],
                composition_mode=d.get("composition_mode", CompositionMode.FLUID.value),
                message_output_mode=d.get("message_output_mode", MessageOutputMode.BLOCK.value),
                engine="alpha",
                effort=Effort.MEDIUM.value,
            )

        return await DocumentMigrationHelper[_AgentDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_2_0,
                "0.2.0": v0_2_0_to_v0_3_0,
                "0.3.0": v0_3_0_to_v0_4_0,
                "0.4.0": v0_4_0_to_v0_5_0,
                "0.5.0": v0_5_0_to_v0_6_0,
            },
        ).migrate(doc)

    async def _association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[_AgentTagAssociationDocument]:
        doc = cast(_AgentTagAssociationDocument, doc)

        if doc["version"] in ("0.3.0", "0.4.0", "0.5.0"):
            return _AgentTagAssociationDocument(
                id=ObjectId(doc["id"]),
                version=Version.String("0.6.0"),
                creation_utc=doc["creation_utc"],
                agent_id=AgentId(doc["agent_id"]),
                tag_id=TagId(doc["tag_id"]),
            )

        if doc["version"] == "0.6.0":
            return doc

        return None

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
            collections_prefix=self._collections_prefix,
        ):
            self._agents_collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_agents" if self._collections_prefix else "agents",
                schema=_AgentDocument,
                document_loader=self._document_loader,
            )

            self._tag_association_collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_agent_tags"
                if self._collections_prefix
                else "agent_tags",
                schema=_AgentTagAssociationDocument,
                document_loader=self._association_document_loader,
            )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        return False

    def _serialize_agent(self, agent: Agent) -> _AgentDocument:
        return _AgentDocument(
            id=ObjectId(agent.id),
            version=self.VERSION.to_string(),
            creation_utc=agent.creation_utc.isoformat(),
            last_modified=agent.last_modified_utc.isoformat(),
            name=agent.name,
            description=agent.description,
            max_engine_iterations=agent.max_engine_iterations,
            composition_mode=agent.composition_mode.value,
            message_output_mode=agent.message_output_mode.value,
            engine=agent.engine,
            effort=agent.effort.value,
        )

    async def _deserialize_agent(self, agent_document: _AgentDocument) -> Agent:
        tags = [
            d["tag_id"]
            for d in await self._tag_association_collection.find(
                {"agent_id": {"$eq": agent_document["id"]}}
            )
        ]

        return Agent(
            id=AgentId(agent_document["id"]),
            creation_utc=datetime.fromisoformat(agent_document["creation_utc"]),
            last_modified_utc=datetime.fromisoformat(agent_document["last_modified"]),
            name=agent_document["name"],
            description=agent_document["description"],
            max_engine_iterations=agent_document["max_engine_iterations"],
            tags=tags,
            composition_mode=CompositionMode(agent_document.get("composition_mode", "fluid")),
            message_output_mode=MessageOutputMode(
                agent_document.get("message_output_mode", "block")
            ),
            engine=agent_document.get("engine", "alpha"),
            effort=Effort(agent_document.get("effort", Effort.MEDIUM.value)),
        )

    @override
    async def create_agent(
        self,
        name: str,
        description: Optional[str] = None,
        creation_utc: Optional[datetime] = None,
        max_engine_iterations: Optional[int] = None,
        composition_mode: Optional[CompositionMode] = None,
        message_output_mode: Optional[MessageOutputMode] = None,
        engine: Optional[str] = None,
        effort: Optional[Effort] = None,
        tags: Optional[Sequence[TagId]] = None,
        id: Optional[AgentId] = None,
    ) -> Agent:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)
            max_engine_iterations = max_engine_iterations or 3

            # Use provided ID or generate one
            if id is not None:
                agent_id = id

                # Check if agent with this ID already exists
                existing = await self._agents_collection.find_one(filters={"id": {"$eq": agent_id}})
                if existing:
                    raise ValueError(f"Agent with id '{agent_id}' already exists")
            else:
                agent_checksum = xxh3_checksum(f"{name}{description}{max_engine_iterations}{tags}")
                agent_id = AgentId(self._id_generator.generate(agent_checksum))

            agent = Agent(
                id=agent_id,
                name=name,
                description=description,
                creation_utc=creation_utc,
                last_modified_utc=creation_utc,
                max_engine_iterations=max_engine_iterations,
                tags=tags or [],
                composition_mode=composition_mode or CompositionMode.FLUID,
                message_output_mode=message_output_mode or MessageOutputMode.BLOCK,
                engine=engine or "alpha",
                effort=effort or Effort.MEDIUM,
            )

            await self._agents_collection.insert_one(document=self._serialize_agent(agent=agent))

            for tag_id in tags or []:
                tag_checksum = xxh3_checksum(f"{agent.id}{tag_id}")

                await self._tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(self._id_generator.generate(tag_checksum)),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "agent_id": agent.id,
                        "tag_id": tag_id,
                    }
                )

        return agent

    @override
    async def list_agents(
        self,
    ) -> Sequence[Agent]:
        async with self._lock.reader_lock:
            return [
                await self._deserialize_agent(d)
                for d in await self._agents_collection.find(filters={})
            ]

    @override
    async def read_agent(self, agent_id: AgentId) -> Agent:
        async with self._lock.reader_lock:
            agent_document = await self._agents_collection.find_one(
                filters={
                    "id": {"$eq": agent_id},
                }
            )

        if not agent_document:
            raise ItemNotFoundError(item_id=UniqueId(agent_id))

        return await self._deserialize_agent(agent_document=agent_document)

    @override
    async def update_agent(
        self,
        agent_id: AgentId,
        params: AgentUpdateParams,
    ) -> Agent:
        async with self._lock.writer_lock:
            agent_document = await self._agents_collection.find_one(
                filters={
                    "id": {"$eq": agent_id},
                }
            )

            if not agent_document:
                raise ItemNotFoundError(item_id=UniqueId(agent_id))

            update_payload = cast(_AgentDocument, to_json_dict(params))
            update_payload["last_modified"] = datetime.now(timezone.utc).isoformat()

            result = await self._agents_collection.update_one(
                filters={"id": {"$eq": agent_id}},
                params=update_payload,
            )

        assert result.updated_document

        return await self._deserialize_agent(agent_document=result.updated_document)

    @override
    async def delete_agent(
        self,
        agent_id: AgentId,
    ) -> None:
        async with self._lock.writer_lock:
            result = await self._agents_collection.delete_one({"id": {"$eq": agent_id}})

            for doc in await self._tag_association_collection.find(
                filters={
                    "agent_id": {"$eq": agent_id},
                }
            ):
                await self._tag_association_collection.delete_one(
                    filters={"id": {"$eq": doc["id"]}}
                )

        if result.deleted_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(agent_id))

    @override
    async def upsert_tag(
        self,
        agent_id: AgentId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            agent = await self.read_agent(agent_id)

            if tag_id in agent.tags:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_checksum = xxh3_checksum(f"{agent_id}{tag_id}")

            association_document: _AgentTagAssociationDocument = {
                "id": ObjectId(self._id_generator.generate(association_checksum)),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "agent_id": agent_id,
                "tag_id": tag_id,
            }

            _ = await self._tag_association_collection.insert_one(document=association_document)

            agent_document = await self._agents_collection.find_one({"id": {"$eq": agent_id}})

            if agent_document:
                await self._agents_collection.update_one(
                    filters={"id": {"$eq": agent_id}},
                    params={"last_modified": datetime.now(timezone.utc).isoformat()},
                )

        if not agent_document:
            raise ItemNotFoundError(item_id=UniqueId(agent_id))

        return True

    @override
    async def remove_tag(
        self,
        agent_id: AgentId,
        tag_id: TagId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._tag_association_collection.delete_one(
                {
                    "agent_id": {"$eq": agent_id},
                    "tag_id": {"$eq": tag_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(tag_id))

            agent_document = await self._agents_collection.find_one({"id": {"$eq": agent_id}})

            if agent_document:
                await self._agents_collection.update_one(
                    filters={"id": {"$eq": agent_id}},
                    params={"last_modified": datetime.now(timezone.utc).isoformat()},
                )

        if not agent_document:
            raise ItemNotFoundError(item_id=UniqueId(agent_id))


class CompositeAgentStore(AgentStore):
    def __init__(
        self,
        writable_store: AgentStore,
        readable_stores: Sequence[AgentStore],
    ) -> None:
        self._writable_store = writable_store
        self._readable_stores = readable_stores
        self._all_stores: Sequence[AgentStore] = [writable_store, *readable_stores]

    @override
    async def create_agent(
        self,
        name: str,
        description: Optional[str] = None,
        creation_utc: Optional[datetime] = None,
        max_engine_iterations: Optional[int] = None,
        composition_mode: Optional[CompositionMode] = None,
        message_output_mode: Optional[MessageOutputMode] = None,
        tags: Optional[Sequence[TagId]] = None,
        id: Optional[AgentId] = None,
    ) -> Agent:
        return await self._writable_store.create_agent(
            name=name,
            description=description,
            creation_utc=creation_utc,
            max_engine_iterations=max_engine_iterations,
            composition_mode=composition_mode,
            message_output_mode=message_output_mode,
            tags=tags,
            id=id,
        )

    @override
    async def list_agents(
        self,
    ) -> Sequence[Agent]:
        results = await safe_gather(*[store.list_agents() for store in self._all_stores])
        return list(chain.from_iterable(results))

    @override
    async def read_agent(
        self,
        agent_id: AgentId,
    ) -> Agent:
        results = await safe_gather(
            *[try_or_none(store.read_agent(agent_id)) for store in self._all_stores]
        )
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise ItemNotFoundError(item_id=UniqueId(agent_id))
        return result

    @override
    async def update_agent(
        self,
        agent_id: AgentId,
        params: AgentUpdateParams,
    ) -> Agent:
        return await self._writable_store.update_agent(agent_id, params)

    @override
    async def delete_agent(
        self,
        agent_id: AgentId,
    ) -> None:
        return await self._writable_store.delete_agent(agent_id)

    @override
    async def upsert_tag(
        self,
        agent_id: AgentId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        return await self._writable_store.upsert_tag(agent_id, tag_id, creation_utc)

    @override
    async def remove_tag(
        self,
        agent_id: AgentId,
        tag_id: TagId,
    ) -> None:
        return await self._writable_store.remove_tag(agent_id, tag_id)
