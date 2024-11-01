from abc import ABC, abstractmethod
from dataclasses import dataclass

from emcie.server.core.agents import AgentId
from emcie.server.core.sessions import SessionId
from emcie.server.core.emissions import EventEmitter


@dataclass(frozen=True)
class Context:
    session_id: SessionId
    agent_id: AgentId


class Engine(ABC):
    @abstractmethod
    async def process(
        self,
        context: Context,
        event_emitter: EventEmitter,
    ) -> bool: ...
