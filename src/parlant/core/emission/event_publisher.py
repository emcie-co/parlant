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

from dataclasses import replace
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
    Event,
    EventId,
    EventKind,
    EventSource,
    EventUpdateParams,
    MessageEventData,
    SessionId,
    SessionStore,
    StatusEventData,
    ToolEventData,
)


class EventPublisherMessageUpdater:
    """MessageEventUpdater implementation that updates events in the SessionStore."""

    def __init__(
        self,
        session_store: SessionStore,
        session_id: SessionId,
        event: EmittedEvent,
        persisted_event_id: EventId,
    ) -> None:
        self._store = session_store
        self._session_id = session_id
        self._event = event
        self._event_id = persisted_event_id

    async def __call__(self, data: MessageEventData) -> MessageEventHandle:
        await self._store.update_event(
            session_id=self._session_id,
            event_id=self._event_id,
            params=EventUpdateParams(data=cast(JSONSerializable, data)),
        )

        updated_event = replace(self._event, data=cast(JSONSerializable, data))

        return MessageEventHandle(event=updated_event, update=self)


class EventPublisher(EventEmitter):
    def __init__(
        self,
        emitting_agent: Agent,
        session_store: SessionStore,
        session_id: SessionId,
    ) -> None:
        self.agent = emitting_agent
        self._store = session_store
        self._session_id = session_id

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

        await self._publish_event(event)

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

        emitted_event = EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.MESSAGE,
            trace_id=trace_id,
            data=message_data,
            metadata=metadata,
        )

        persisted_event = await self._publish_event(emitted_event)

        updater = EventPublisherMessageUpdater(
            session_store=self._store,
            session_id=self._session_id,
            event=emitted_event,
            persisted_event_id=persisted_event.id,
        )

        return MessageEventHandle(event=emitted_event, update=updater)

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

        await self._publish_event(event)

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

        await self._publish_event(event)

        return event

    async def _publish_event(
        self,
        event: EmittedEvent,
    ) -> Event:
        return await self._store.create_event(
            session_id=self._session_id,
            source=EventSource.AI_AGENT,
            kind=event.kind,
            trace_id=event.trace_id,
            data=event.data,
            metadata=event.metadata or {},
        )


class EventPublisherFactory(EventEmitterFactory):
    def __init__(
        self,
        agent_store: AgentStore,
        session_store: SessionStore,
    ) -> None:
        self._agent_store = agent_store
        self._session_store = session_store

    @override
    async def create_event_emitter(
        self,
        emitting_agent_id: AgentId,
        session_id: SessionId,
    ) -> EventEmitter:
        agent = await self._agent_store.read_agent(emitting_agent_id)
        return EventPublisher(agent, self._session_store, session_id)
