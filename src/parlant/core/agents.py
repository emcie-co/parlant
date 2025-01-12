# Copyright 2024 Emcie Co Ltd.
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
from pathlib import Path
from typing import Optional, Sequence, cast
from typing_extensions import Self, TypedDict, override

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import (
    SCHEMA_VERSION_UNKNOWN,
    ItemNotFoundError,
    SchemaVersion,
    UniqueId,
    Version,
    generate_id,
    AgentId,
)
from parlant.core.documents.agents import _AgentDocument_v1
from parlant.core.persistence.common import VersionedStore, ObjectId
from parlant.core.persistence.document_database import DocumentCollection, DocumentDatabase


class AgentUpdateParams(TypedDict, total=False):
    name: str
    description: Optional[str]
    max_engine_iterations: int


@dataclass(frozen=True)
class Agent:
    id: AgentId
    name: str
    description: Optional[str]
    creation_utc: datetime
    max_engine_iterations: int


class AgentStore(ABC):
    @abstractmethod
    async def create_agent(
        self,
        name: str,
        description: Optional[str] = None,
        creation_utc: Optional[datetime] = None,
        max_engine_iterations: Optional[int] = None,
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


class _AgentDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    name: str
    description: Optional[str]
    max_engine_iterations: int


class AgentDocumentStore(AgentStore, VersionedStore):
    VERSION = SchemaVersion(1)

    def __init__(
        self,
        database: DocumentDatabase,
    ):
        if database.version == SCHEMA_VERSION_UNKNOWN:
            database.version = self.VERSION
        self._database = database
        self._collection: DocumentCollection[_AgentDocument_v1]

        self._lock = ReaderWriterLock()

    async def __aenter__(self) -> Self:
        self._collection = await self._database.get_or_create_collection(
            name="agents",
            schema=_AgentDocument_v1,
        )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    def _serialize(self, agent: Agent) -> _AgentDocument_v1:
        return _AgentDocument_v1(
            id=ObjectId(agent.id),
            creation_utc=agent.creation_utc.isoformat(),
            name=agent.name,
            description=agent.description,
            max_engine_iterations=agent.max_engine_iterations,
        )

    def _deserialize(self, agent_document: _AgentDocument_v1) -> Agent:
        return Agent(
            id=AgentId(agent_document["id"]),
            creation_utc=datetime.fromisoformat(agent_document["creation_utc"]),
            name=agent_document["name"],
            description=agent_document["description"],
            max_engine_iterations=agent_document["max_engine_iterations"],
        )

    @property
    @override
    def database_version(self) -> SchemaVersion:
        return self._database.version

    @property
    @override
    def database_name(self) -> str:
        if "file_path" in self._database.__dict__ and isinstance(
            db_path := getattr(self._database, "file_path"), Path
        ):
            return db_path.name
        return "unknown"

    @override
    async def create_agent(
        self,
        name: str,
        description: Optional[str] = None,
        creation_utc: Optional[datetime] = None,
        max_engine_iterations: Optional[int] = None,
    ) -> Agent:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)
            max_engine_iterations = max_engine_iterations or 3

            agent = Agent(
                id=AgentId(generate_id()),
                name=name,
                description=description,
                creation_utc=creation_utc,
                max_engine_iterations=max_engine_iterations,
            )

            await self._collection.insert_one(document=self._serialize(agent=agent))

        return agent

    @override
    async def list_agents(
        self,
    ) -> Sequence[Agent]:
        async with self._lock.reader_lock:
            return [self._deserialize(d) for d in await self._collection.find(filters={})]

    @override
    async def read_agent(self, agent_id: AgentId) -> Agent:
        async with self._lock.reader_lock:
            agent_document = await self._collection.find_one(
                filters={
                    "id": {"$eq": agent_id},
                }
            )

        if not agent_document:
            raise ItemNotFoundError(item_id=UniqueId(agent_id))

        return self._deserialize(agent_document)

    @override
    async def update_agent(
        self,
        agent_id: AgentId,
        params: AgentUpdateParams,
    ) -> Agent:
        async with self._lock.writer_lock:
            agent_document = await self._collection.find_one(
                filters={
                    "id": {"$eq": agent_id},
                }
            )

            if not agent_document:
                raise ItemNotFoundError(item_id=UniqueId(agent_id))

            result = await self._collection.update_one(
                filters={"id": {"$eq": agent_id}},
                params=cast(_AgentDocument_v1, params),
            )

        assert result.updated_document

        return self._deserialize(agent_document=result.updated_document)

    @override
    async def delete_agent(
        self,
        agent_id: AgentId,
    ) -> None:
        async with self._lock.writer_lock:
            result = await self._collection.delete_one({"id": {"$eq": agent_id}})

        if result.deleted_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(agent_id))
