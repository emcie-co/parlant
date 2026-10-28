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
from typing import Callable, Mapping, cast
from typing_extensions import override

from parlant.core.common import JSONSerializable
from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.store_provider import StoreProvider, StoreProviderHints
from parlant.core.emission.session_tap import SessionEventTap, TapEventKind, TapItem
from parlant.core.emissions import (
    EmittedEvent,
    EventEmitter,
    EventEmitterFactory,
    MessageEventHandle,
    StatusEventHandle,
)
from parlant.core.sessions import (
    Event,
    EventId,
    EventKind,
    EventSource,
    EventUpdateParams,
    MessageEventData,
    Participant,
    SessionId,
    SessionStore,
    StatusEventData,
    ToolEventData,
    patch_internal_event_metadata,
)
from parlant.core import metadata as metadata_utils
from parlant.core.tracer import Tracer, TURN_TRACE_ID_ATTRIBUTE


class EventPublisherMessageUpdater:
    """MessageEventUpdater implementation that updates events in the SessionStore."""

    def __init__(
        self,
        session_store: SessionStore,
        session_id: SessionId,
        event: EmittedEvent,
        persisted_event_id: EventId,
        tap: SessionEventTap | None = None,
    ) -> None:
        self._store = session_store
        self._session_id = session_id
        self._event = event
        self._event_id = persisted_event_id
        self._tap = tap

    async def __call__(
        self,
        data: MessageEventData,
        *,
        internal_metadata: Mapping[str, JSONSerializable] | None = None,
    ) -> MessageEventHandle:
        saved = await self._store.update_event(
            session_id=self._session_id,
            event_id=self._event_id,
            params=EventUpdateParams(data=cast(JSONSerializable, data)),
        )

        # Patch only our paths against the latest metadata revision; public
        # annotations and independent internal updates must survive completion.
        for key, value in (internal_metadata or {}).items():
            saved = await patch_internal_event_metadata(
                self._store, self._session_id, self._event_id, (key,), value
            )
        updated_event = replace(
            self._event, data=cast(JSONSerializable, data), metadata=saved.metadata
        )
        self._event = updated_event

        if self._tap:
            self._tap.publish(
                TapItem(
                    session_id=self._session_id,
                    trace_id=self._event.trace_id,
                    kind=TapEventKind.EVENT_UPDATED,
                    event_id=self._event_id,
                    event_kind=self._event.kind,
                    data=cast(JSONSerializable, data),
                )
            )

        return MessageEventHandle(event=updated_event, update=self)


class EventPublisherStatusUpdater:
    """StatusEventUpdater implementation that updates events in the SessionStore."""

    def __init__(
        self,
        session_store: SessionStore,
        session_id: SessionId,
        event: EmittedEvent,
        persisted_event_id: EventId,
        tap: SessionEventTap | None = None,
    ) -> None:
        self._store = session_store
        self._session_id = session_id
        self._event = event
        self._event_id = persisted_event_id
        self._tap = tap

    async def __call__(self, data: StatusEventData) -> StatusEventHandle:
        await self._store.update_event(
            session_id=self._session_id,
            event_id=self._event_id,
            params=EventUpdateParams(data=cast(JSONSerializable, data)),
        )

        updated_event = replace(self._event, data=cast(JSONSerializable, data))

        if self._tap:
            self._tap.publish(
                TapItem(
                    session_id=self._session_id,
                    trace_id=self._event.trace_id,
                    kind=TapEventKind.EVENT_UPDATED,
                    event_id=self._event_id,
                    event_kind=self._event.kind,
                    data=cast(JSONSerializable, data),
                )
            )

        return StatusEventHandle(event=updated_event, update=self)


class EventPublisher(EventEmitter):
    def __init__(
        self,
        emitting_agent: Agent,
        session_store: SessionStore,
        session_id: SessionId,
        tap: SessionEventTap | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.agent = emitting_agent
        self._store = session_store
        self._session_id = session_id
        self._tap = tap
        self._tracer = tracer

    def _event_metadata(
        self,
        trace_id: str,
        metadata: Mapping[str, JSONSerializable] | None,
    ) -> Mapping[str, JSONSerializable] | None:
        if self._tracer is None:
            return metadata
        origin = self._tracer.get_attribute(TURN_TRACE_ID_ATTRIBUTE)
        if (
            isinstance(origin, str)
            and origin.strip()
            and self._tracer.trace_id == trace_id
            and self._tracer.get_attribute("session_id") == self._session_id
        ):
            return metadata_utils.set_internal_value(metadata or {}, ("turn_trace_id",), origin)
        return metadata

    @override
    async def emit_status_event(
        self,
        trace_id: str,
        data: StatusEventData,
        metadata: Mapping[str, JSONSerializable] | None = None,
    ) -> StatusEventHandle:
        emitted_event = EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.STATUS,
            trace_id=trace_id,
            data=cast(JSONSerializable, data),
            metadata=self._event_metadata(trace_id, metadata),
        )

        persisted_event = await self._publish_event(emitted_event)

        updater = EventPublisherStatusUpdater(
            session_store=self._store,
            session_id=self._session_id,
            event=emitted_event,
            persisted_event_id=persisted_event.id,
            tap=self._tap,
        )

        return StatusEventHandle(event=emitted_event, update=updater)

    @override
    async def emit_message_event(
        self,
        trace_id: str,
        data: str | MessageEventData,
        metadata: Mapping[str, JSONSerializable] | None = None,
        *,
        source: EventSource = EventSource.AI_AGENT,
    ) -> MessageEventHandle:
        if source not in (EventSource.AI_AGENT, EventSource.SYSTEM):
            raise ValueError(f"Unsupported message event source: {source}")

        if isinstance(data, str):
            participant: Participant = (
                {
                    "id": self.agent.id,
                    "display_name": self.agent.name,
                }
                if source == EventSource.AI_AGENT
                else {
                    "id": None,
                    "display_name": "System",
                }
            )
            message_data = cast(
                JSONSerializable,
                MessageEventData(
                    message=data,
                    participant=participant,
                ),
            )
        else:
            message_data = cast(JSONSerializable, data)

        emitted_event = EmittedEvent(
            source=source,
            kind=EventKind.MESSAGE,
            trace_id=trace_id,
            data=message_data,
            metadata=self._event_metadata(trace_id, metadata),
        )

        persisted_event = await self._publish_event(emitted_event)

        updater = EventPublisherMessageUpdater(
            session_store=self._store,
            session_id=self._session_id,
            event=emitted_event,
            persisted_event_id=persisted_event.id,
            tap=self._tap,
        )

        return MessageEventHandle(event=emitted_event, update=updater)

    @override
    async def emit_tool_event(
        self,
        trace_id: str,
        data: ToolEventData,
        metadata: Mapping[str, JSONSerializable] | None = None,
    ) -> EmittedEvent:
        event = EmittedEvent(
            source=EventSource.SYSTEM,
            kind=EventKind.TOOL,
            trace_id=trace_id,
            data=cast(JSONSerializable, data),
            metadata=self._event_metadata(trace_id, metadata),
        )

        await self._publish_event(event)

        return event

    @override
    async def emit_custom_event(
        self,
        trace_id: str,
        data: JSONSerializable,
        metadata: Mapping[str, JSONSerializable] | None = None,
    ) -> EmittedEvent:
        event = EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.CUSTOM,
            trace_id=trace_id,
            data=data,
            metadata=self._event_metadata(trace_id, metadata),
        )

        await self._publish_event(event)

        return event

    async def _publish_event(
        self,
        event: EmittedEvent,
    ) -> Event:
        persisted = await self._store.create_event(
            session_id=self._session_id,
            source=event.source,
            kind=event.kind,
            trace_id=event.trace_id,
            data=event.data,
            metadata=metadata_utils.initialize_internal_metadata(event.metadata or {}),
        )

        if self._tap:
            self._tap.publish(
                TapItem(
                    session_id=self._session_id,
                    trace_id=event.trace_id,
                    kind=TapEventKind.EVENT_PERSISTED,
                    event_id=persisted.id,
                    event_kind=event.kind,
                    data=event.data,
                )
            )

        return persisted


class EventPublisherFactory(EventEmitterFactory):
    def __init__(
        self,
        store_provider_factory: Callable[[], StoreProvider],
        tap: SessionEventTap | None = None,
        tracer_factory: Callable[[], Tracer] | None = None,
    ) -> None:
        self._store_provider_factory = store_provider_factory
        self._tap = tap
        self._tracer_factory = tracer_factory

    @override
    async def create_event_emitter(
        self,
        emitting_agent_id: AgentId,
        session_id: SessionId,
    ) -> EventEmitter:
        agent_store = self._store_provider_factory().get_store(
            AgentStore, StoreProviderHints(call_site="engine")
        )
        session_store = self._store_provider_factory().get_store(
            SessionStore, StoreProviderHints(call_site="engine")
        )
        agent = await agent_store.read_agent(emitting_agent_id)
        return EventPublisher(
            agent,
            session_store,
            session_id,
            tap=self._tap,
            tracer=self._tracer_factory() if self._tracer_factory else None,
        )
