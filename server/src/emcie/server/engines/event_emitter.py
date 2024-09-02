from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from emcie.server.core.common import JSONSerializable
from emcie.server.core.sessions import EventSource, ToolCallResult


@dataclass(frozen=True)
class EmittedEvent:
    source: EventSource
    kind: str
    data: JSONSerializable


class EventEmitter(ABC):
    @abstractmethod
    async def emit_message(self, message: str) -> EmittedEvent: ...

    @abstractmethod
    async def emit_tool_results(self, results: Sequence[ToolCallResult]) -> EmittedEvent: ...
