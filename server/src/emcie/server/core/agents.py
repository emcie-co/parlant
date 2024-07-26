from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import NewType, Optional, Sequence

from emcie.server.base_models import DefaultBaseModel
from emcie.server.core import common
from emcie.server.core.persistence.document_database import DocumentDatabase

AgentId = NewType("AgentId", str)


@dataclass(frozen=True)
class Agent:
    id: AgentId
    name: str
    description: Optional[str]
    creation_utc: datetime


class AgentStore(ABC):
    @abstractmethod
    async def create_agent(
        self,
        name: str,
        description: Optional[str] = None,
        creation_utc: Optional[datetime] = None,
    ) -> Agent: ...

    @abstractmethod
    async def list_agents(self) -> Sequence[Agent]: ...

    @abstractmethod
    async def read_agent(
        self,
        agent_id: AgentId,
    ) -> Agent: ...


class AgentDocumentStore(AgentStore):
    class AgentDocument(DefaultBaseModel):
        id: AgentId
        creation_utc: datetime
        name: str
        description: Optional[str]

    def __init__(
        self,
        database: DocumentDatabase,
    ):
        self._collection = database.get_or_create_collection(
            name="agents",
            schema=self.AgentDocument,
        )

    async def create_agent(
        self,
        name: str,
        description: Optional[str] = None,
        creation_utc: Optional[datetime] = None,
    ) -> Agent:
        creation_utc = creation_utc or datetime.now(timezone.utc)
        agent_id = await self._collection.insert_one(
            document={
                "id": common.generate_id(),
                "name": name,
                "description": description,
                "creation_utc": creation_utc,
            },
        )
        return Agent(
            id=AgentId(agent_id),
            name=name,
            description=description,
            creation_utc=creation_utc,
        )

    async def list_agents(
        self,
    ) -> Sequence[Agent]:
        return [
            Agent(
                id=a["id"],
                name=a["name"],
                description=a.get("description"),
                creation_utc=a["creation_utc"],
            )
            for a in await self._collection.find(filters={})
        ]

    async def read_agent(
        self,
        agent_id: AgentId,
    ) -> Agent:
        agent_document = await self._collection.find_one(
            filters={
                "id": {"$eq": agent_id},
            }
        )
        return Agent(
            id=agent_document["id"],
            name=agent_document["name"],
            description=agent_document["description"],
            creation_utc=agent_document["creation_utc"],
        )
