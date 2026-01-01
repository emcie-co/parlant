# Copyright 2026 Emcie Co Ltd.
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

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping
from typing_extensions import deprecated

from parlant.core.agents import AgentId
from parlant.core.common import JSONSerializable
from parlant.core.sessions import (
    EventKind,
    EventSource,
    MessageEventData,
    SessionId,
    StatusEventData,
    ToolEventData,
)


@dataclass(frozen=True)
class EmittedEvent:
    """An event that has been emitted, but not yet persisted, by the system."""

    source: EventSource
    kind: EventKind
    trace_id: str
    data: JSONSerializable
    metadata: Mapping[str, JSONSerializable] | None

    @property
    @deprecated("Use 'trace_id' instead")
    def correlation_id(self) -> str:
        return self.trace_id


@dataclass(frozen=True)
class MessageEventHandle:
    """A handle to an emitted message event that allows updating it."""

    event: EmittedEvent
    update: Callable[[MessageEventData], Awaitable[MessageEventHandle]]


class EventEmitter(ABC):
    """An interface for emitting events in the system."""

    @abstractmethod
    async def emit_status_event(
        self,
        trace_id: str | None = None,
        data: StatusEventData | None = None,
        metadata: Mapping[str, JSONSerializable] | None = None,
        **kwargs: Any,
    ) -> EmittedEvent:
        """Emit a status event with the given trace ID and data."""
        ...

    @abstractmethod
    async def emit_message_event(
        self,
        trace_id: str | None = None,
        data: str | MessageEventData | None = None,
        metadata: Mapping[str, JSONSerializable] | None = None,
        **kwargs: Any,
    ) -> MessageEventHandle:
        """Emit a message event with the given trace ID and data."""
        ...

    @abstractmethod
    async def emit_tool_event(
        self,
        trace_id: str | None,
        data: ToolEventData | None = None,
        metadata: Mapping[str, JSONSerializable] | None = None,
    ) -> EmittedEvent:
        """Emit a tool event with the given trace ID and data."""
        ...

    @abstractmethod
    async def emit_custom_event(
        self,
        trace_id: str | None,
        data: JSONSerializable | None = None,
        metadata: Mapping[str, JSONSerializable] | None = None,
        **kwargs: Any,
    ) -> EmittedEvent:
        """Emit a custom event with the given trace ID and data."""
        ...


class EventEmitterFactory(ABC):
    """An interface for creating event emitters."""

    @abstractmethod
    async def create_event_emitter(
        self,
        emitting_agent_id: AgentId,
        session_id: SessionId,
    ) -> EventEmitter:
        """Create an event emitter for the given agent and session."""
        ...


def ensure_new_usage_params_and_get_trace_id(trace_id: str | None, data: Any, **kwargs: Any) -> str:
    if "correlation_id" in kwargs:
        import warnings

        warnings.warn(
            "The 'correlation_id' parameter is deprecated. Use 'trace_id' instead.",
            DeprecationWarning,
            stacklevel=3,
        )

        if trace_id is None:
            return str(kwargs["correlation_id"])

    if trace_id is None:
        raise ValueError("trace_id must be provided and cannot be None")

    if data is None:
        raise ValueError("data must be provided and cannot be None")

    return trace_id
