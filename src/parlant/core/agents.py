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
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import NewType, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import (
    ItemNotFoundError,
    UniqueId,
    Version,
    IdGenerator,
    md5_checksum,
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


class AgentUpdateParams(TypedDict, total=False):
    name: str
    description: Optional[str]
    max_engine_iterations: int
    composition_mode: CompositionMode


@dataclass(frozen=True)
class Agent:
    id: AgentId
    name: str
    description: Optional[str]
    creation_utc: datetime
    max_engine_iterations: int
    tags: Sequence[TagId]
    composition_mode: CompositionMode = CompositionMode.FLUID


class AgentStore(ABC):
    @abstractmethod
    async def create_agent(
        self,
        name: str,
        description: Optional[str] = None,
        creation_utc: Optional[datetime] = None,
        max_engine_iterations: Optional[int] = None,
        composition_mode: Optional[CompositionMode] = None,
        tags: Optional[Sequence[TagId]] = None,
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


class _AgentDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    name: str
    description: Optional[str]
    max_engine_iterations: int
    composition_mode: str


class _AgentTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    agent_id: AgentId
    tag_id: TagId


class AgentDocumentStore(AgentStore):
    VERSION = Version.from_string("0.4.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        database: DocumentDatabase,
        allow_migration: bool = False,
    ):
        self._id_generator = id_generator

        self._database = database
        self._agents_collection: DocumentCollection[_AgentDocument]
        self._tag_association_collection: DocumentCollection[_AgentTagAssociationDocument]
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

        return await DocumentMigrationHelper[_AgentDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_2_0,
                "0.2.0": v0_2_0_to_v0_3_0,
                "0.3.0": v0_3_0_to_v0_4_0,
            },
        ).migrate(doc)

    async def _association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[_AgentTagAssociationDocument]:
        doc = cast(_AgentTagAssociationDocument, doc)

        if doc["version"] == "0.3.0":
            return _AgentTagAssociationDocument(
                id=ObjectId(doc["id"]),
                version=Version.String("0.4.0"),
                creation_utc=doc["creation_utc"],
                agent_id=AgentId(doc["agent_id"]),
                tag_id=TagId(doc["tag_id"]),
            )

        if doc["version"] == "0.4.0":
            return doc

        return None

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
        ):
            self._agents_collection = await self._database.get_or_create_collection(
                name="agents",
                schema=_AgentDocument,
                document_loader=self._document_loader,
            )

            self._tag_association_collection = await self._database.get_or_create_collection(
                name="agent_tags",
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
            name=agent.name,
            description=agent.description,
            max_engine_iterations=agent.max_engine_iterations,
            composition_mode=agent.composition_mode.value,
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
            name=agent_document["name"],
            description=agent_document["description"],
            max_engine_iterations=agent_document["max_engine_iterations"],
            tags=tags,
            composition_mode=CompositionMode(agent_document.get("composition_mode", "fluid")),
        )

    @override
    async def create_agent(
        self,
        name: str,
        description: Optional[str] = None,
        creation_utc: Optional[datetime] = None,
        max_engine_iterations: Optional[int] = None,
        composition_mode: Optional[CompositionMode] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Agent:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)
            max_engine_iterations = max_engine_iterations or 3

            agent_checksum = md5_checksum(f"{name}{description}{max_engine_iterations}{tags}")

            agent_checksum = md5_checksum(f"{name}{description}{max_engine_iterations}{tags}")

            agent = Agent(
                id=AgentId(self._id_generator.generate(agent_checksum)),
                name=name,
                description=description,
                creation_utc=creation_utc,
                max_engine_iterations=max_engine_iterations,
                tags=tags or [],
                composition_mode=composition_mode or CompositionMode.FLUID,
            )

            await self._agents_collection.insert_one(document=self._serialize_agent(agent=agent))

            for tag_id in tags or []:
                tag_checksum = md5_checksum(f"{agent.id}{tag_id}")

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

            result = await self._agents_collection.update_one(
                filters={"id": {"$eq": agent_id}},
                params=cast(_AgentDocument, to_json_dict(params)),
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

            association_checksum = md5_checksum(f"{agent_id}{tag_id}")

            association_document: _AgentTagAssociationDocument = {
                "id": ObjectId(self._id_generator.generate(association_checksum)),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "agent_id": agent_id,
                "tag_id": tag_id,
            }

            _ = await self._tag_association_collection.insert_one(document=association_document)

            agent_document = await self._agents_collection.find_one({"id": {"$eq": agent_id}})

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

        if not agent_document:
            raise ItemNotFoundError(item_id=UniqueId(agent_id))
