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

from typing import Any, Mapping, cast
from typing_extensions import override

from parlant.core.common import JSONSerializable
from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.emissions import (
    EmittedEvent,
    EventEmitter,
    EventEmitterFactory,
    MessageEventHandle,
    ensure_new_usage_params_and_get_trace_id,
)
from parlant.core.sessions import (
    EventKind,
    EventSource,
    MessageEventData,
    SessionId,
    StatusEventData,
    ToolEventData,
)


class EventBufferMessageUpdater:
    """MessageEventUpdater implementation that updates events in an EventBuffer."""

    def __init__(self, buffer: "EventBuffer", event_index: int) -> None:
        self._buffer = buffer
        self._event_index = event_index

    async def __call__(self, data: MessageEventData) -> MessageEventHandle:
        # EmittedEvent is frozen, so we need to replace with a new event
        old_event = self._buffer.events[self._event_index]
        new_event = EmittedEvent(
            source=old_event.source,
            kind=old_event.kind,
            trace_id=old_event.trace_id,
            data=cast(JSONSerializable, data),
            metadata=old_event.metadata,
        )
        self._buffer.events[self._event_index] = new_event

        return MessageEventHandle(event=new_event, update=self)


class EventBuffer(EventEmitter):
    def __init__(self, emitting_agent: Agent) -> None:
        self.agent = emitting_agent
        self.events: list[EmittedEvent] = []

    @override
    async def emit_status_event(
        self,
        trace_id: str | None = None,
        data: StatusEventData | None = None,
        metadata: Mapping[str, JSONSerializable] | None = None,
        **kwargs: Any,
    ) -> EmittedEvent:
        trace_id = ensure_new_usage_params_and_get_trace_id(trace_id, data, **kwargs)

        event = EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.STATUS,
            trace_id=trace_id,
            data=cast(JSONSerializable, data),
            metadata=metadata,
        )

        self.events.append(event)

        return event

    @override
    async def emit_message_event(
        self,
        trace_id: str | None = None,
        data: str | MessageEventData | None = None,
        metadata: Mapping[str, JSONSerializable] | None = None,
        **kwargs: Any,
    ) -> MessageEventHandle:
        trace_id = ensure_new_usage_params_and_get_trace_id(trace_id, data, **kwargs)

        if isinstance(data, str):
            message_data = cast(
                JSONSerializable,
                MessageEventData(
                    message=data,
                    participant={
                        "id": self.agent.id,
                        "display_name": self.agent.name,
                    },
                ),
            )
        else:
            message_data = cast(JSONSerializable, data)

        event = EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.MESSAGE,
            trace_id=trace_id,
            data=message_data,
            metadata=metadata,
        )

        event_index = len(self.events)
        self.events.append(event)

        updater = EventBufferMessageUpdater(buffer=self, event_index=event_index)
        return MessageEventHandle(event=event, update=updater)

    @override
    async def emit_tool_event(
        self,
        trace_id: str | None = None,
        data: ToolEventData | None = None,
        metadata: Mapping[str, JSONSerializable] | None = None,
        **kwargs: Any,
    ) -> EmittedEvent:
        trace_id = ensure_new_usage_params_and_get_trace_id(trace_id, data, **kwargs)

        event = EmittedEvent(
            source=EventSource.SYSTEM,
            kind=EventKind.TOOL,
            trace_id=trace_id,
            data=cast(JSONSerializable, data),
            metadata=metadata,
        )

        self.events.append(event)

        return event

    @override
    async def emit_custom_event(
        self,
        trace_id: str | None = None,
        data: JSONSerializable | None = None,
        metadata: Mapping[str, JSONSerializable] | None = None,
        **kwargs: Any,
    ) -> EmittedEvent:
        trace_id = ensure_new_usage_params_and_get_trace_id(trace_id, data, **kwargs)

        event = EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.CUSTOM,
            trace_id=trace_id,
            data=data,
            metadata=metadata,
        )

        self.events.append(event)

        return event


class EventBufferFactory(EventEmitterFactory):
    def __init__(self, agent_store: AgentStore) -> None:
        self._agent_store = agent_store

    @override
    async def create_event_emitter(
        self,
        emitting_agent_id: AgentId,
        session_id: SessionId,
    ) -> EventEmitter:
        _ = session_id
        agent = await self._agent_store.read_agent(emitting_agent_id)
        return EventBuffer(emitting_agent=agent)
