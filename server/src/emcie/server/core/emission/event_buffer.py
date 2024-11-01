from typing import cast

from emcie.server.core.common import JSONSerializable
from emcie.server.core.agents import Agent, AgentId, AgentStore
from emcie.server.core.emissions import EmittedEvent, EventEmitter, EventEmitterFactory
from emcie.server.core.sessions import (
    MessageEventData,
    SessionId,
    StatusEventData,
    ToolEventData,
)


class EventBuffer(EventEmitter):
    def __init__(self, emitting_agent: Agent) -> None:
        self.agent = emitting_agent
        self.events: list[EmittedEvent] = []

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

        self.events.append(event)

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

        self.events.append(event)

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

        self.events.append(event)

        return event


class EventBufferFactory(EventEmitterFactory):
    def __init__(self, agent_store: AgentStore) -> None:
        self._agent_store = agent_store

    async def create_event_emitter(
        self,
        emitting_agent_id: AgentId,
        session_id: SessionId,
    ) -> EventEmitter:
        _ = session_id
        agent = await self._agent_store.read_agent(emitting_agent_id)
        return EventBuffer(emitting_agent=agent)
