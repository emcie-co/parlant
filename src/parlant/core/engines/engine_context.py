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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, Optional, Sequence, cast

from typing_extensions import TypeVar

from parlant.core.agents import Agent
from parlant.core.async_utils import Stopwatch
from parlant.core.common import JSONSerializable
from parlant.core.tracer import Tracer
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent, EventEmitter
from parlant.core.engines.types import Context
from parlant.core.engines.session_metadata_workspace import SessionMetadataWorkspace
from parlant.core.loggers import Logger
from parlant.core.sessions import (
    AttachmentRef,
    delivered_message_text,
    Event,
    EventKind,
    EventSource,
    MessageEventData,
    Participant,
    Session,
    ToolEventData,
)
from parlant.core.tools import ToolId, ToolResult


# Parameterize the mutable response state and the immutable, engine-specific
# dependency bundle independently. Both default to Any so a bare EngineContext
# remains convenient in engine-agnostic hooks, retrievers, and model providers.
TState = TypeVar("TState", default=Any)
TDependencies = TypeVar("TDependencies", default=Any)


@dataclass(frozen=True)
class InteractionMessage:
    """A message in the interaction history"""

    source: EventSource
    """The source type of the message (e.g., customer, AI agent, etc.)"""

    participant: Participant
    """The participant who sent the message (includes display name and ID)"""

    trace_id: str
    """The trace ID of the message"""

    content: str
    """The content of the message"""

    creation_utc: datetime
    """The timestamp when the message was created"""

    attachments: Sequence[AttachmentRef] = ()
    """File references attached to the message (empty when there are none)"""

    def __str__(self) -> str:
        """Returns a string representation of the message"""
        return f"{self.participant['display_name']} ({self.source}): {self.content}"

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class Interaction:
    """Helper class to access a session's interaction state"""

    @staticmethod
    def empty() -> Interaction:
        """Returns an empty interaction state"""
        return Interaction(events=[])

    @property
    def messages(self) -> Sequence[InteractionMessage]:
        """Returns the messages in the interaction session"""
        return [
            InteractionMessage(
                source=event.source,
                participant=cast(MessageEventData, event.data)["participant"],
                trace_id=event.trace_id,
                content=delivered_message_text(cast(MessageEventData, event.data), event.metadata),
                creation_utc=event.creation_utc,
                attachments=cast(MessageEventData, event.data).get("attachments", ()),
            )
            for event in self.events
            if event.kind == EventKind.MESSAGE
        ]

    @property
    def last_customer_message(self) -> Optional[InteractionMessage]:
        """Returns the last customer message in the interaction session, if it exists"""
        if event := self.last_customer_message_event:
            message_data = cast(MessageEventData, event.data)

            return InteractionMessage(
                source=event.source,
                participant=message_data["participant"],
                trace_id=event.trace_id,
                content=delivered_message_text(message_data, event.metadata),
                creation_utc=event.creation_utc,
                attachments=message_data.get("attachments", ()),
            )

        return None

    @property
    def last_customer_message_event(self) -> Optional[Event]:
        """Returns the last customer message in the interaction session, if it exists"""
        for event in reversed(self.events):
            if event.kind == EventKind.MESSAGE and event.source == EventSource.CUSTOMER:
                return event

        return None

    events: Sequence[Event]
    """An sequenced event-by-event representation of the interaction"""


@dataclass
class EngineContext(Generic[TState, TDependencies]):
    """Helper class to access loaded values that are relevant for responding in a particular context"""

    info: Context
    """The raw call context which is here represented in its loaded form"""

    logger: Logger
    """The logger used to log messages in the current context"""

    tracer: Tracer
    """The tracer used to track the trace ID and properties in the current context"""

    agent: Agent
    """The agent which is currently requested to respond"""

    customer: Customer
    """The customer to which the agent is responding"""

    session: Session
    """The session being processed"""

    session_event_emitter: EventEmitter
    """Emits new events into the loaded session"""

    response_event_emitter: EventEmitter
    """Emits new events that are scoped to the current response"""

    interaction: Interaction
    """A snapshot of the interaction history in the loaded session"""

    state: TState
    """The current state of the response being processed"""

    dependencies: TDependencies
    """Immutable engine-specific services resolved for this context"""

    creation: Stopwatch = field(default_factory=Stopwatch.start)
    """A stopwatch that was started when the context was created"""

    session_metadata: SessionMetadataWorkspace = field(kw_only=True)
    """Required turn-scoped metadata workspace."""

    async def add_tool_event(
        self,
        tool_id: ToolId,
        arguments: dict[str, JSONSerializable],
        result: ToolResult,
        rationale: str = "",
    ) -> None:
        """Adds a staged tool event to the loaded context"""
        # state is generic (TState); every concrete ResponseState carries
        # tool_events, but that contract isn't expressible on the bare TypeVar.
        cast(Any, self.state).tool_events.append(
            EmittedEvent(
                source=EventSource.SYSTEM,
                kind=EventKind.TOOL,
                trace_id=self.tracer.trace_id,
                data=cast(
                    JSONSerializable,
                    ToolEventData(
                        # TODO: Add a common method to create a session-store compatible ToolCall from ToolResult
                        tool_calls=[
                            {
                                "tool_id": tool_id.to_string(),
                                "arguments": arguments,
                                "rationale": rationale,
                                "result": {
                                    "data": result.data,
                                    "metadata": result.metadata,
                                    "control": result.control,
                                    "canned_responses": result.canned_responses,
                                    "canned_response_fields": result.canned_response_fields,
                                },
                            }
                        ]
                    ),
                ),
                metadata=None,
            )
        )
