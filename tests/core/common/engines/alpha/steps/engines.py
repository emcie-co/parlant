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

import asyncio
from collections.abc import Sequence
from typing import cast
from pytest_bdd import given, when, parsers
from unittest.mock import AsyncMock

from parlant.core.agents import AgentId, AgentStore, CompositionMode
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableStore,
    ContextVariableValue,
)
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer
from parlant.core.customers import CustomerId, CustomerStore
from parlant.core.engines.alpha.engine import AlphaEngine
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.generic.response_analysis_batch import (
    GenericResponseAnalysisBatch,
    GenericResponseAnalysisSchema,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    ResponseAnalysisContext,
)
from parlant.core.engines.alpha.engine_context import Interaction, EngineContext, ResponseState
from parlant.core.engines.alpha.message_generator import MessageGenerator
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.utils import context_variables_to_json
from parlant.core.engines.alpha.canned_response_generator import (
    CannedResponseGenerator,
)
from parlant.core.engines.alpha.message_event_composer import MessageEventComposer
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.types import Context, UtteranceRationale, UtteranceRequest
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.entity_cq import EntityCommands, EntityQueries
from parlant.core.glossary import Term
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import (
    AgentState,
    EventSource,
    SessionId,
    SessionStore,
    SessionUpdateParams,
)

from tests.core.common.engines.alpha.utils import step
from tests.core.common.utils import ContextOfTest


@step(given, "the alpha engine", target_fixture="engine")
def given_the_alpha_engine(
    context: ContextOfTest,
) -> AlphaEngine:
    return context.container[AlphaEngine]


@step(given, "a faulty message production mechanism")
def given_a_faulty_message_production_mechanism(
    context: ContextOfTest,
) -> None:
    generator = context.container[MessageGenerator]
    generator.generate_response = AsyncMock(side_effect=Exception())  # type: ignore


@step(
    given,
    parsers.parse('an utterance request "{action}", to {do_something}'),
)
def given_a_follow_up_utterance_request(
    context: ContextOfTest, action: str, do_something: str
) -> UtteranceRequest:
    canned_response_request = UtteranceRequest(
        action=action,
        rationale={
            "follow up with the customer": UtteranceRationale.FOLLOW_UP,
            "buy time": UtteranceRationale.BUY_TIME,
        }[do_something],
    )

    context.actions.append(canned_response_request)

    return canned_response_request


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


def _load_context_variables(
    context: ContextOfTest,
    customer_id: CustomerId,
    agent_id: AgentId,
) -> list[tuple[ContextVariable, ContextVariableValue]]:
    customer = context.sync_await(
        context.container[CustomerStore].read_customer(customer_id),
    )
    # TODO The function need to be replaced by AlphaEngine._load_context_variables once will be public
    variables_supported_by_agent = context.sync_await(
        context.container[EntityQueries].find_context_variables_for_context(
            agent_id=agent_id,
        )
    )

    result = []

    keys_to_check_in_order_of_importance = (
        [customer_id]  # Customer-specific value
        + [f"tag:{tag_id}" for tag_id in customer.tags]  # Tag-specific value
        + [ContextVariableStore.GLOBAL_KEY]  # Global value
    )

    for variable in variables_supported_by_agent:
        # Try keys in order of importance, stopping at and using
        # the first (and most important) set key for each variable.
        for key in keys_to_check_in_order_of_importance:
            if value := context.sync_await(
                context.container[EntityQueries].read_context_variable_value(
                    variable_id=variable.id,
                    key=key,
                )
            ):
                result.append((variable, value))
                break

    return result


def _load_glossary_terms(
    context: ContextOfTest,
    agent_id: AgentId,
    context_variables: list[tuple[ContextVariable, ContextVariableValue]],
) -> Sequence[Term]:
    # TODO The function need to be replaced by AlphaEngine._load_glossary_terms once will be public
    query = ""

    if context_variables:
        query += f"\n{context_variables_to_json(context_variables)}"

    if context.events:
        query += str([e.data for e in context.events])

    if context.guidelines:
        query += str(
            [
                f"When {g.content.condition}, then {g.content.action}"
                for g in context.guidelines.values()
            ]
        )
    if query:
        return context.sync_await(
            context.container[EntityQueries].find_glossary_terms_for_context(
                agent_id=agent_id,
                query=query,
            )
        )

    return []


@step(when, "detection and processing are triggered", target_fixture="emitted_events")
def when_detection_and_processing_are_triggered(
    context: ContextOfTest,
    engine: AlphaEngine,
    session_id: SessionId,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> list[EmittedEvent]:
    agent = context.sync_await(
        context.container[AgentStore].read_agent(agent_id),
    )
    customer = context.sync_await(
        context.container[CustomerStore].read_customer(customer_id),
    )

    buffer = EventBuffer(agent)
    session = context.sync_await(context.container[SessionStore].read_session(session_id))

    context_variables = _load_context_variables(
        context,
        customer_id,
        agent_id,
    )

    terms = _load_glossary_terms(context, agent_id, context_variables)

    matches_to_prepare = [
        g
        for g in context.guideline_matches.values()
        if (
            not session.agent_states
            or g.guideline.id not in session.agent_states[-1].applied_guideline_ids
        )
        and not g.guideline.metadata.get("continuous", False)
    ]

    interaction_history = (
        context.events[:-1] if context.events[-1].source == EventSource.CUSTOMER else context.events
    )

    response_analysis = GenericResponseAnalysisBatch(
        logger=context.container[Logger],
        meter=context.container[Meter],
        optimization_policy=context.container[OptimizationPolicy],
        schematic_generator=context.container[SchematicGenerator[GenericResponseAnalysisSchema]],
        context=ResponseAnalysisContext(
            agent=agent,
            session=session,
            customer=customer,
            context_variables=context_variables,
            interaction_history=interaction_history,
            terms=terms,
            staged_tool_events=[],
            staged_message_events=[],
        ),
        guideline_matches=matches_to_prepare,
    )

    applied_guideline_ids = [
        a.guideline.id
        for a in (context.sync_await(response_analysis.process())).analyzed_guidelines
        if a.is_previously_applied
    ]

    applied_guideline_ids.extend(
        session.agent_states[-1].applied_guideline_ids if session.agent_states else []
    )

    context.sync_await(
        context.container[EntityCommands].update_session(
            session_id=session.id,
            params=SessionUpdateParams(
                agent_states=list(session.agent_states)
                + [
                    AgentState(
                        trace_id="<main>",
                        applied_guideline_ids=applied_guideline_ids,
                        journey_paths={},
                    )
                ]
            ),
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
    agent = context.sync_await(context.container[AgentStore].read_agent(agent_id))
    session = context.sync_await(context.container[SessionStore].read_session(session_id))
    customer = context.sync_await(
        context.container[CustomerStore].read_customer(session.customer_id)
    )

    message_event_composer: MessageEventComposer

    match agent.composition_mode:
        case CompositionMode.FLUID:
            message_event_composer = context.container[MessageGenerator]
        case (
            CompositionMode.CANNED_STRICT
            | CompositionMode.CANNED_COMPOSITED
            | CompositionMode.CANNED_FLUID
        ):
            message_event_composer = context.container[CannedResponseGenerator]

    loaded_context = EngineContext(
        info=Context(
            session_id=session.id,
            agent_id=agent.id,
        ),
        logger=context.container[Logger],
        tracer=context.container[Tracer],
        agent=agent,
        customer=customer,
        session=session,
        session_event_emitter=EventBuffer(agent),
        response_event_emitter=EventBuffer(agent),
        interaction=Interaction(events=context.events),
        state=ResponseState(
            context_variables=[],
            glossary_terms=set(),
            capabilities=[],
            iterations=[],
            ordinary_guideline_matches=list(context.guideline_matches.values()),
            tool_enabled_guideline_matches={},
            journeys=[],
            journey_paths={k: list(v) for k, v in session.agent_states[-1].journey_paths.items()}
            if session.agent_states
            else {},
            tool_events=[],
            tool_insights=ToolInsights(),
            prepared_to_respond=False,
            message_events=[],
        ),
    )

    result = context.sync_await(message_event_composer.generate_response(loaded_context))

    assert len(result) > 0
    assert all(e is not None for e in result[0].events)

    return list(cast(list[EmittedEvent], result[0].events))


@step(when, "uttering is triggered", target_fixture="emitted_events")
def when_uttering_is_triggered(
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
