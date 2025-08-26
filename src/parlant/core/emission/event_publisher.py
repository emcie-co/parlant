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

from typing import cast
from typing_extensions import override

from parlant.core.common import JSONSerializable
from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.emissions import EmittedEvent, EventEmitter, EventEmitterFactory
from parlant.core.sessions import (
    EventKind,
    EventSource,
    MessageEventData,
    SessionId,
    SessionStore,
    StatusEventData,
    ToolEventData,
)


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
        correlation_id: str,
        data: StatusEventData,
    ) -> EmittedEvent:
        event = EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.STATUS,
            correlation_id=correlation_id,
            data=cast(JSONSerializable, data),
        )

        await self._publish_event(event)

        return event

    @override
    async def emit_message_event(
        self,
        correlation_id: str,
        data: str | MessageEventData,
    ) -> EmittedEvent:
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
            correlation_id=correlation_id,
            data=message_data,
        )

        await self._publish_event(event)

        return event

    @override
    async def emit_tool_event(
        self,
        correlation_id: str,
        data: ToolEventData,
    ) -> EmittedEvent:
        event = EmittedEvent(
            source=EventSource.SYSTEM,
            kind=EventKind.TOOL,
            correlation_id=correlation_id,
            data=cast(JSONSerializable, data),
        )

        await self._publish_event(event)

        return event

    @override
    async def emit_custom_event(
        self,
        correlation_id: str,
        data: JSONSerializable,
    ) -> EmittedEvent:
        event = EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.CUSTOM,
            correlation_id=correlation_id,
            data=data,
        )

        await self._publish_event(event)

        return event

    async def _publish_event(
        self,
        event: EmittedEvent,
    ) -> None:
        await self._store.create_event(
            session_id=self._session_id,
            source=EventSource.AI_AGENT,
            kind=event.kind,
            correlation_id=event.correlation_id,
            data=event.data,
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
