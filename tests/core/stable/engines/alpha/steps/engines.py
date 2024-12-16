# Copyright 2024 Emcie Co Ltd.
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

import asyncio
from typing import cast
from pytest_bdd import given, when
from unittest.mock import AsyncMock

from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.customers import CustomerStore
from parlant.core.engines.alpha.engine import AlphaEngine

from parlant.core.engines.alpha.message_event_generator import MessageEventGenerator
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.types import Context
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.sessions import SessionId, SessionStore
from tests.core.stable.engines.alpha.utils import ContextOfTest, step


@step(given, "the alpha engine", target_fixture="engine")
def given_the_alpha_engine(
    context: ContextOfTest,
) -> AlphaEngine:
    return context.container[AlphaEngine]


@step(given, "a faulty message production mechanism")
def given_a_faulty_message_production_mechanism(
    context: ContextOfTest,
) -> None:
    generator = context.container[MessageEventGenerator]
    generator.generate_events = AsyncMock(side_effect=Exception())  # type: ignore


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
    session_id: SessionId,
) -> list[EmittedEvent]:
    session = context.sync_await(context.container[SessionStore].read_session(session_id))
    customer = context.sync_await(
        context.container[CustomerStore].read_customer(session.customer_id)
    )

    event_buffer = EventBuffer(
        context.sync_await(
            context.container[AgentStore].read_agent(agent.id),
        )
    )

    message_event_generator = context.container[MessageEventGenerator]

    message_event_generator_results = context.sync_await(
        message_event_generator.generate_events(
            event_emitter=event_buffer,
            agents=[agent],
            customer=customer,
            context_variables=[],
            interaction_history=context.events,
            terms=[],
            ordinary_guideline_propositions=list(context.guideline_propositions.values()),
            tool_enabled_guideline_propositions={},
            staged_events=[],
        )
    )

    assert len(message_event_generator_results) > 0
    assert all(e is not None for e in message_event_generator_results[0].events)

    return list(cast(list[EmittedEvent], message_event_generator_results[0].events))
