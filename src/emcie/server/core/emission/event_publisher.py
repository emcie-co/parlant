from typing import cast

from emcie.server.core.common import JSONSerializable
from emcie.server.core.agents import Agent, AgentId, AgentStore
from emcie.server.core.emissions import EmittedEvent, EventEmitter, EventEmitterFactory
from emcie.server.core.sessions import (
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

    async def emit_status_event(
        self,
        correlation_id: str,
        data: StatusEventData,
    ) -> EmittedEvent:
        event = EmittedEvent(
            source="ai_agent",
            kind="status",
            correlation_id=correlation_id,
            data=cast(JSONSerializable, data),
        )

        await self._publish_event(event)

        return event

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
            source="ai_agent",
            kind="message",
            correlation_id=correlation_id,
            data=message_data,
        )

        await self._publish_event(event)

        return event

    async def emit_tool_event(
        self,
        correlation_id: str,
        data: ToolEventData,
    ) -> EmittedEvent:
        event = EmittedEvent(
            source="ai_agent",
            kind="tool",
            correlation_id=correlation_id,
            data=cast(JSONSerializable, data),
        )

        await self._publish_event(event)

        return event

    async def _publish_event(
        self,
        event: EmittedEvent,
    ) -> None:
        await self._store.create_event(
            session_id=self._session_id,
            source="ai_agent",
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

    async def create_event_emitter(
        self,
        emitting_agent_id: AgentId,
        session_id: SessionId,
    ) -> EventEmitter:
        agent = await self._agent_store.read_agent(emitting_agent_id)
        return EventPublisher(agent, self._session_store, session_id)
