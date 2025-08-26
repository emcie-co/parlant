# Copyright 2025 Emcie Co Ltd.
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
    correlation_id: str
    data: JSONSerializable


class EventEmitter(ABC):
    """An interface for emitting events in the system."""

    @abstractmethod
    async def emit_status_event(
        self,
        correlation_id: str,
        data: StatusEventData,
    ) -> EmittedEvent:
        """Emit a status event with the given correlation ID and data."""
        ...

    @abstractmethod
    async def emit_message_event(
        self,
        correlation_id: str,
        data: str | MessageEventData,
    ) -> EmittedEvent:
        """Emit a message event with the given correlation ID and data."""
        ...

    @abstractmethod
    async def emit_tool_event(
        self,
        correlation_id: str,
        data: ToolEventData,
    ) -> EmittedEvent:
        """Emit a tool event with the given correlation ID and data."""
        ...

    @abstractmethod
    async def emit_custom_event(
        self,
        correlation_id: str,
        data: JSONSerializable,
    ) -> EmittedEvent:
        """Emit a custom event with the given correlation ID and data."""
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
