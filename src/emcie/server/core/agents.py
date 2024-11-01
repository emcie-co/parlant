from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import NewType, Optional, Sequence, TypedDict, cast

from emcie.server.core.common import ItemNotFoundError, UniqueId, Version, generate_id
from emcie.server.core.persistence.document_database import (
    DocumentDatabase,
    ObjectId,
)

AgentId = NewType("AgentId", str)


class AgentUpdateParams(TypedDict, total=False):
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
    async def list_agents(self) -> Sequence[Agent]: ...

    @abstractmethod
    async def read_agent(self, agent_id: AgentId) -> Agent: ...

    @abstractmethod
    async def update_agent(
        self,
        agent_id: AgentId,
        params: AgentUpdateParams,
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
        self._collection = database.get_or_create_collection(
            name="agents",
            schema=_AgentDocument,
        )

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

    async def create_agent(
        self,
        name: str,
        description: Optional[str] = None,
        creation_utc: Optional[datetime] = None,
        max_engine_iterations: Optional[int] = None,
    ) -> Agent:
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

    async def list_agents(
        self,
    ) -> Sequence[Agent]:
        return [self._deserialize(d) for d in await self._collection.find(filters={})]

    async def read_agent(self, agent_id: AgentId) -> Agent:
        agent_document = await self._collection.find_one(
            filters={
                "id": {"$eq": agent_id},
            }
        )

        if not agent_document:
            raise ItemNotFoundError(item_id=UniqueId(agent_id))

        return self._deserialize(agent_document)

    async def update_agent(
        self,
        agent_id: AgentId,
        params: AgentUpdateParams,
    ) -> None:
        await self._collection.update_one(
            filters={"id": {"$eq": agent_id}},
            params=cast(_AgentDocument, params),
        )
