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
from typing import Any, Optional, Sequence, cast
from typing_extensions import deprecated

from parlant.core.agents import Agent
from parlant.core.async_utils import Stopwatch
from parlant.core.capabilities import Capability
from parlant.core.common import JSONSerializable
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.tracer import Tracer
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent, EventEmitter
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.types import Context
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.glossary import Term
from parlant.core.guidelines import Guideline
from parlant.core.journeys import Journey, JourneyId
from parlant.core.loggers import Logger
from parlant.core.sessions import (
    Event,
    EventKind,
    EventSource,
    MessageEventData,
    Participant,
    Session,
    ToolEventData,
)
from parlant.core.tools import ToolId, ToolResult


@dataclass(frozen=True)
class IterationState:
    """State of a single iteration in the response process"""

    matched_guidelines: list[GuidelineMatch]
    resolved_guidelines: list[GuidelineMatch]
    tool_insights: ToolInsights
    executed_tools: list[ToolId]


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
                content=cast(MessageEventData, event.data)["message"],
                creation_utc=event.creation_utc,
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
                content=message_data["message"],
                creation_utc=event.creation_utc,
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

    @property
    @deprecated("Use the events property instead")
    def history(self) -> Sequence[Event]:
        """Returns a string representation of the interaction history"""
        return self.events


@dataclass(frozen=False)
class ResponseState:
    """Used to access and update the state needed for responding properly"""

    context_variables: list[tuple[ContextVariable, ContextVariableValue]]
    glossary_terms: set[Term]
    capabilities: list[Capability]
    iterations: list[IterationState]
    ordinary_guideline_matches: list[GuidelineMatch]
    tool_enabled_guideline_matches: dict[GuidelineMatch, list[ToolId]]
    journeys: list[Journey]
    journey_paths: dict[JourneyId, list[Optional[str]]]
    tool_events: list[EmittedEvent]
    tool_insights: ToolInsights
    prepared_to_respond: bool
    message_events: list[EmittedEvent]
    additional_canned_response_fields: dict[str, Any] = field(default_factory=dict)

    @property
    def ordinary_guidelines(self) -> list[Guideline]:
        return [gp.guideline for gp in self.ordinary_guideline_matches]

    @property
    def tool_enabled_guidelines(self) -> list[Guideline]:
        return [gp.guideline for gp in self.tool_enabled_guideline_matches.keys()]

    @property
    def guidelines(self) -> list[Guideline]:
        return self.ordinary_guidelines + self.tool_enabled_guidelines

    @property
    def all_events(self) -> list[EmittedEvent]:
        return self.tool_events + self.message_events


@dataclass
class EngineContext:
    """Helper class to access loaded values that are relevant for responding in a particular context"""

    info: Context
    """The raw call context which is here represented in its loaded form"""

    logger: Logger
    """The logger used to log messages in the current context"""

    tracer: Tracer
    """The tracer used to track the trace ID and properties in the current context"""

    @property
    @deprecated("Use the tracer property instead")
    def correlator(self) -> Tracer:
        return self.tracer

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

    state: ResponseState
    """The current state of the response being processed"""

    creation: Stopwatch = field(default_factory=Stopwatch.start)
    """A stopwatch that was started when the context was created"""

    async def add_tool_event(
        self,
        tool_id: ToolId,
        arguments: dict[str, JSONSerializable],
        result: ToolResult,
    ) -> None:
        """Adds a staged tool event to the loaded context"""
        self.state.tool_events.append(
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


@deprecated("Please use the EngineContext class instead of LoadedContext")
class LoadedContext(EngineContext):
    pass
