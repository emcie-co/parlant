import asyncio
from pytest_bdd import given, when
from unittest.mock import AsyncMock

from emcie.server.core.agents import Agent, AgentId, AgentStore
from emcie.server.core.engines.alpha.engine import AlphaEngine

from emcie.server.core.engines.alpha.message_event_producer import MessageEventProducer
from emcie.server.core.emissions import EmittedEvent
from emcie.server.core.engines.types import Context
from emcie.server.core.emission.event_buffer import EventBuffer
from emcie.server.core.sessions import SessionId
from tests.core.engines.alpha.utils import ContextOfTest, step


@step(given, "the alpha engine", target_fixture="engine")
def given_the_alpha_engine(
    context: ContextOfTest,
) -> AlphaEngine:
    return context.container[AlphaEngine]


@step(given, "a faulty message production mechanism")
def given_a_faulty_message_production_mechanism(
    context: ContextOfTest,
) -> None:
    producer = context.container[MessageEventProducer]
    producer.produce_events = AsyncMock(side_effect=Exception())  # type: ignore


@step(when, "processing is triggered", target_fixture="emitted_events")
def when_processing_is_triggered(
    context: ContextOfTest,
    engine: AlphaEngine,
    session_id: SessionId,
    agent_id: AgentId,
) -> list[EmittedEvent]:
    buffer = EventBuffer(
        context.sync_await(
            context.container[AgentStore].read_agent(agent_id),
        )
    )

    context.sync_await(
        engine.process(
            Context(
                session_id=session_id,
                agent_id=agent_id,
            ),
            buffer,
        )
    )

    return buffer.events


@step(when, "processing is triggered and cancelled in the middle", target_fixture="emitted_events")
def when_processing_is_triggered_and_cancelled_in_the_middle(
    context: ContextOfTest,
    engine: AlphaEngine,
    agent_id: AgentId,
    session_id: SessionId,
) -> list[EmittedEvent]:
    event_buffer = EventBuffer(
        context.sync_await(
            context.container[AgentStore].read_agent(agent_id),
        )
    )

    processing_task = context.sync_await.event_loop.create_task(
        engine.process(
            Context(
                session_id=session_id,
                agent_id=agent_id,
            ),
            event_buffer,
        )
    )

    context.sync_await(asyncio.sleep(1))

    processing_task.cancel()

    assert not context.sync_await(processing_task)

    return event_buffer.events


@step(when, "messages are emitted", target_fixture="emitted_events")
def when_messages_are_emitted(
    context: ContextOfTest,
    agent: Agent,
) -> list[EmittedEvent]:
    event_buffer = EventBuffer(
        context.sync_await(
            context.container[AgentStore].read_agent(agent.id),
        )
    )

    message_event_producer = context.container[MessageEventProducer]

    message_events = context.sync_await(
        message_event_producer.produce_events(
            event_emitter=event_buffer,
            agents=[agent],
            context_variables=[],
            interaction_history=context.events,
            terms=[],
            ordinary_guideline_propositions=list(context.guideline_propositions.values()),
            tool_enabled_guideline_propositions={},
            staged_events=[],
        )
    )

    return list(message_events)
