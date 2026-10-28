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

"""Engine driver steps for the compass BDD suites.

Ported from the alpha suites with these translations:

- "the engine" resolves the CompassEngine from the test container.
- Alpha's "detection and processing" step (which ran its response-analysis
  batch by hand to mark previously-applied guidelines) is unnecessary:
  compass tracks applied rules itself, so the step is plain processing.
- "messages are emitted" drives the responder directly with a seeded
  response state (the scenario's declared matches), mirroring how alpha
  drove its message composer with pre-built guideline matches.
"""

import asyncio
from collections.abc import Awaitable, Callable

from pytest_bdd import given, when, parsers

from parlant.core.agents import AgentId, AgentStore
from parlant.core.container import Singleton
from parlant.core.customers import CustomerStore
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.compass.context import (
    CompassDependencies,
    EngineContext as CompassEngineContext,
    ResponseState,
)
from parlant.core.engines.compass.engine import CompassEngine
from parlant.core.engines.compass.loop.loop import LoopResult
from parlant.core.engines.compass.responder.responder import Responder
from parlant.core.engines.engine_context import (
    EngineContext as CoreEngineContext,
    Interaction,
)
from parlant.core.engines.session_metadata_workspace import SessionMetadataWorkspace
from parlant.core.engines.types import Context, UtteranceRationale, UtteranceRequest
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.loggers import Logger
from parlant.core.sessions import SessionId, SessionStore
from parlant.core.tracer import Tracer

from tests.core.common.engines.compass.utils import step
from tests.core.common.utils import ContextOfTest


@step(given, "the engine", target_fixture="engine")
def given_the_engine(
    context: ContextOfTest,
) -> CompassEngine:
    return context.container[CompassEngine]


class _FailingResponder(Responder):
    """The container-resolved responder with ``respond`` replaced by a failure.

    Deliberately skips ``Responder.__init__``: the base responder's state is
    adopted wholesale, and ``respond`` fails before touching any of it anyway.
    """

    def __init__(self, base: Responder) -> None:
        self.__dict__ = dict(base.__dict__)

    async def respond(
        self,
        context: CompassEngineContext,
        refresh_state: Callable[[CompassEngineContext], Awaitable[None]] | None = None,
    ) -> LoopResult:
        raise Exception()


@step(given, "a faulty message production mechanism", target_fixture="engine")
def given_a_faulty_message_production_mechanism(
    context: ContextOfTest,
) -> CompassEngine:
    # Rebind the responder in the container rather than patching the resolved
    # instance. The engine singleton already resolved the healthy responder, so
    # rebind it too (on a clone — the container forbids in-place redefinition)
    # and hand the scenario the freshly composed engine.
    container = context.container.clone()
    container[Responder] = _FailingResponder(context.container[Responder])
    container[CompassEngine] = Singleton(CompassEngine)
    context.container = container

    return container[CompassEngine]


@step(
    given,
    parsers.parse('an utterance request "{action}", to {do_something}'),
)
def given_a_follow_up_utterance_request(
    context: ContextOfTest, action: str, do_something: str
) -> UtteranceRequest:
    utterance_request = UtteranceRequest(
        action=action,
        rationale={
            "follow up with the customer": UtteranceRationale.FOLLOW_UP,
            "buy time": UtteranceRationale.BUY_TIME,
        }[do_something],
    )

    context.actions.append(utterance_request)

    return utterance_request


def _run_processing(
    context: ContextOfTest,
    engine: CompassEngine,
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


@step(when, "processing is triggered", target_fixture="emitted_events")
def when_processing_is_triggered(
    context: ContextOfTest,
    engine: CompassEngine,
    session_id: SessionId,
    agent_id: AgentId,
) -> list[EmittedEvent]:
    return _run_processing(context, engine, session_id, agent_id)


@step(when, "detection and processing are triggered", target_fixture="emitted_events")
def when_detection_and_processing_are_triggered(
    context: ContextOfTest,
    engine: CompassEngine,
    session_id: SessionId,
    agent_id: AgentId,
) -> list[EmittedEvent]:
    # Compass tracks previously-applied rules on its own (the ranker's
    # previously-applied analysis + the session's ever-matched ledger), so
    # unlike alpha there is no separate detection pass to run first.
    return _run_processing(context, engine, session_id, agent_id)


@step(when, "processing is triggered and cancelled in the middle", target_fixture="emitted_events")
def when_processing_is_triggered_and_cancelled_in_the_middle(
    context: ContextOfTest,
    engine: CompassEngine,
    agent_id: AgentId,
    session_id: SessionId,
    no_cache: None,
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

    context.sync_await(asyncio.sleep(0.5))

    processing_task.cancel()

    assert not context.sync_await(processing_task)

    return event_buffer.events


@step(when, "messages are emitted", target_fixture="emitted_events")
def when_messages_are_emitted(
    context: ContextOfTest,
    agent_id: AgentId,
    session_id: SessionId,
) -> list[EmittedEvent]:
    """Drive the responder directly with the scenario's declared matches,
    bypassing matching — the compass translation of alpha's direct message-
    composer invocation."""
    agent = context.sync_await(context.container[AgentStore].read_agent(agent_id))
    session_store = context.container[SessionStore]
    session = context.sync_await(session_store.read_session(session_id))
    customer = context.sync_await(
        context.container[CustomerStore].read_customer(session.customer_id)
    )

    session_event_emitter = EventBuffer(agent)

    matches = list(context.rule_matches.values())
    matched_rules = [m.rule for m in matches]

    engine_context = CoreEngineContext(
        info=Context(
            session_id=session.id,
            agent_id=agent.id,
        ),
        logger=context.container[Logger],
        tracer=context.container[Tracer],
        agent=agent,
        customer=customer,
        session=session,
        session_event_emitter=session_event_emitter,
        response_event_emitter=EventBuffer(agent),
        interaction=Interaction(events=context.events),
        state=ResponseState(
            agent_effort=agent.effort,
            usable_rules=[*context.rules.values()],
            discovered_rules=set(matched_rules),
            ordinary_rule_matches=matches,
        ),
        dependencies=context.container[CompassDependencies],
        session_metadata=SessionMetadataWorkspace(session, session_store),
    )

    responder = context.container[Responder]
    result = context.sync_await(responder.respond(engine_context))

    assert result.steps, "The responder produced no steps"

    return session_event_emitter.events


@step(when, "uttering is triggered", target_fixture="emitted_events")
def when_uttering_is_triggered(
    context: ContextOfTest,
    engine: CompassEngine,
    session_id: SessionId,
    agent_id: AgentId,
) -> list[EmittedEvent]:
    buffer = EventBuffer(
        context.sync_await(
            context.container[AgentStore].read_agent(agent_id),
        )
    )

    context.sync_await(
        engine.utter(
            Context(
                session_id=session_id,
                agent_id=agent_id,
            ),
            buffer,
            context.actions,
        )
    )

    return buffer.events
