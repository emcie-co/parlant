from dataclasses import dataclass
from datetime import datetime
import os
from typing import Any, Dict, Iterable, NewType, Optional
import importlib

from emcie.server import common
from emcie.server.models import ModelId

AgentId = NewType("AgentId", str)


@dataclass(frozen=True)
class Agent:
    id: AgentId
    model_id: ModelId
    creation_utc: datetime


class AgentStore:
    def __init__(
        self,
    ) -> None:
        self._agents: Dict[AgentId, Agent] = {}
        self._skills: Dict[str, Any] = {}

    async def create_skill(
        self,
        skill_id: str,
        module_path: str,
        spec: Dict[Any, Any],
    ) -> None:
        module = importlib.import_module(module_path)
        func = getattr(module, "skill")

        self._skills[skill_id] = {
            "id": skill_id,
            "spec": spec,
            "func": func,
        }

    async def list_skills(self) -> Iterable[Any]:
        return self._skills.values()

    async def create_agent(
        self,
        model_id: Optional[ModelId] = None,
        creation_utc: Optional[datetime] = None,
    ) -> Agent:
        agent = Agent(
            id=AgentId(common.generate_id()),
            model_id=model_id or ModelId(os.environ["DEFAULT_AGENT_MODEL"]),
            creation_utc=creation_utc or datetime.utcnow(),
        )

        self._agents[agent.id] = agent

        return agent

    async def list_agents(self) -> Iterable[Agent]:
        return self._agents.values()

    async def read_agent(self, agent_id: AgentId) -> Agent:
        return self._agents[agent_id]
