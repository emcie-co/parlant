from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from emcie.server.core.common import JSONSerializable
from emcie.server.core.sessions import EventKind, EventSource, ToolCall


@dataclass(frozen=True)
class EmittedEvent:
    source: EventSource
    kind: EventKind
    correlation_id: str
    data: JSONSerializable


class EventEmitter(ABC):
    @abstractmethod
    async def emit_message(
        self,
        correlation_id: str,
        message: str,
    ) -> EmittedEvent: ...

    @abstractmethod
    async def emit_tool_results(
        self,
        correlation_id: str,
        results: Sequence[ToolCall],
    ) -> EmittedEvent: ...
