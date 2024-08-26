from datetime import datetime
from typing import Optional
from fastapi import APIRouter

from emcie.server.base_models import DefaultBaseModel
from emcie.server.core.agents import AgentId, AgentStore


class AgentDTO(DefaultBaseModel):
    id: AgentId
    name: str
    description: Optional[str]
    creation_utc: datetime


class CreateAgentRequest(DefaultBaseModel):
    agent_name: str
    agent_description: Optional[str] = None


class CreateAgentResponse(DefaultBaseModel):
    agent_id: AgentId


class ListAgentsResponse(DefaultBaseModel):
    agents: list[AgentDTO]


def create_router(
    agent_store: AgentStore,
) -> APIRouter:
    router = APIRouter()

    @router.post("/")
    async def create_agent(
        request: Optional[CreateAgentRequest] = None,
    ) -> CreateAgentResponse:
        agent = await agent_store.create_agent(
            name=request and request.agent_name or "Unnamed Agent",
            description=request and request.agent_description or None,
        )

        return CreateAgentResponse(agent_id=agent.id)

    @router.get("/")
    async def list_agents() -> ListAgentsResponse:
        agents = await agent_store.list_agents()

        return ListAgentsResponse(
            agents=[
                AgentDTO(
                    id=a.id,
                    name=a.name,
                    description=a.description,
                    creation_utc=a.creation_utc,
                )
                for a in agents
            ]
        )

    return router
