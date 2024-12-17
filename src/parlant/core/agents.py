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
from typing import NewType, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import ItemNotFoundError, UniqueId, Version, generate_id
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import DocumentDatabase, DocumentCollection

AgentId = NewType("AgentId", str)


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


class AgentDocumentStore(AgentStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        database: DocumentDatabase,
    ):
        self._database = database
        self._collection: DocumentCollection[_AgentDocument]

        self._lock = ReaderWriterLock()

    async def __aenter__(self) -> Self:
        self._collection = await self._database.get_or_create_collection(
            name="agents",
            schema=_AgentDocument,
        )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    def _serialize(self, agent: Agent) -> _AgentDocument:
        return _AgentDocument(
            id=ObjectId(agent.id),
            version=self.VERSION.to_string(),
            creation_utc=agent.creation_utc.isoformat(),
            name=agent.name,
            description=agent.description,
            max_engine_iterations=agent.max_engine_iterations,
        )

    def _deserialize(self, agent_document: _AgentDocument) -> Agent:
        return Agent(
            id=AgentId(agent_document["id"]),
            creation_utc=datetime.fromisoformat(agent_document["creation_utc"]),
            name=agent_document["name"],
            description=agent_document["description"],
            max_engine_iterations=agent_document["max_engine_iterations"],
        )

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
                params=cast(_AgentDocument, params),
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
