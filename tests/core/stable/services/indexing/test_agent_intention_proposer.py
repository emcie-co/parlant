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

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from typing import Sequence
from lagom import Container
from pytest import fixture

from parlant.core.agents import Agent
from parlant.core.capabilities import Capability, CapabilityId
from parlant.core.common import Criticality, JSONSerializable, generate_id
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer
from parlant.core.customers import Customer
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.alpha.guideline_matching.generic.response_analysis_batch import (
    GenericResponseAnalysisBatch,
    GenericResponseAnalysisSchema,
)
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatcher,
    ResponseAnalysisContext,
)
from parlant.core.engines.alpha.engine_context import Interaction, EngineContext, ResponseState
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.types import Context
from parlant.core.entity_cq import EntityCommands
from parlant.core.evaluations import GuidelinePayload, PayloadOperation
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.indexing.behavioral_change_evaluation import GuidelineEvaluator
from parlant.core.services.indexing.guideline_agent_intention_proposer import AgentIntentionProposer
from parlant.core.sessions import (
    AgentState,
    Event,
    EventSource,
    Session,
    SessionId,
    SessionStore,
    SessionUpdateParams,
)
from tests.core.common.utils import create_event_message
from tests.test_utilities import SyncAwaiter

GUIDELINES_DICT = {
    "medical_advice": {
        "condition": "You provide health-related information or advice",
        "action": "Include a disclaimer that this is not medical advice",
    },
    "recommend_product": {
        "condition": "You recommend on a product or a service",
        "action": "Ensure that the recommendation is unbiased and based on reliable information",
    },
    "international_transaction": {
        "condition": "You explain international transaction fees or card usage policies",
        "action": "Be clear about potential fees and offer tips to avoid them",
    },
    "reset_password_offer": {
        "condition": "You offer a password reset option",
        "action": "Ensure that the instruction email is sent in the customer's native language",
    },
    "multiple_capabilities": {
        "condition": "The agent discusses multiple capabilities in a single message",
        "action": "do not offer more than 3 capabilities in a single message",
    },
}


@dataclass
class ContextOfTest:
    container: Container
    sync_await: SyncAwaiter
    guidelines: list[Guideline]
    logger: Logger


@fixture
def context(
    sync_await: SyncAwaiter,
    container: Container,
) -> ContextOfTest:
    return ContextOfTest(
        container,
        sync_await,
        guidelines=list(),
        logger=container[Logger],
    )


def match_guidelines(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
    session_id: SessionId,
    interaction_history: Sequence[Event],
    capabilities: Sequence[Capability] = [],
) -> Sequence[GuidelineMatch]:
    session = context.sync_await(context.container[SessionStore].read_session(session_id))

    loaded_context = EngineContext(
        info=Context(
            session_id=session.id,
            agent_id=agent.id,
        ),
        logger=context.logger,
        tracer=context.container[Tracer],
        agent=agent,
        customer=customer,
        session=session,
        session_event_emitter=EventBuffer(agent),
        response_event_emitter=EventBuffer(agent),
        interaction=Interaction(events=interaction_history),
        state=ResponseState(
            context_variables=[],
            glossary_terms=set(),
            capabilities=[],
            iterations=[],
            ordinary_guideline_matches=[],
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

    guideline_matching_result = context.sync_await(
        context.container[GuidelineMatcher].match_guidelines(
            context=loaded_context,
            active_journeys=[],
            guidelines=context.guidelines,
        )
    )

    return list(chain.from_iterable(guideline_matching_result.batches))


def create_guideline(
    context: ContextOfTest,
    condition: str,
    action: str | None = None,
) -> Guideline:
    metadata: dict[str, JSONSerializable] = {}
    if action:
        guideline_evaluator = context.container[GuidelineEvaluator]
        guideline_evaluation_data = context.sync_await(
            guideline_evaluator.evaluate(
                payloads=[
                    GuidelinePayload(
                        content=GuidelineContent(
                            condition=condition,
                            action=action,
                        ),
                        tool_ids=[],
                        operation=PayloadOperation.ADD,
                        action_proposition=True,
                        properties_proposition=True,
                        journey_node_proposition=False,
                    )
                ],
            )
        )

        metadata = guideline_evaluation_data[0].properties_proposition or {}

    guideline = Guideline(
        id=GuidelineId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        content=GuidelineContent(
            condition=condition,
            action=action,
        ),
        criticality=Criticality.MEDIUM,
        enabled=True,
        tags=[],
        metadata=metadata,
    )

    context.guidelines.append(guideline)

    return guideline


def create_guideline_by_name(
    context: ContextOfTest,
    guideline_name: str,
) -> Guideline | None:
    if guideline_name in GUIDELINES_DICT:
        guideline = create_guideline(
            context=context,
            condition=GUIDELINES_DICT[guideline_name]["condition"],
            action=GUIDELINES_DICT[guideline_name]["action"],
        )
    else:
        guideline = None
    return guideline


def update_previously_applied_guidelines(
    context: ContextOfTest,
    session_id: SessionId,
    applied_guideline_ids: list[GuidelineId],
) -> None:
    session = context.sync_await(context.container[SessionStore].read_session(session_id))
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


def analyze_response_and_update_session(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
    session_id: SessionId,
    previously_matched_guidelines: list[Guideline],
    interaction_history: list[Event],
) -> None:
    session = context.sync_await(context.container[SessionStore].read_session(session_id))

    matches_to_analyze = [
        GuidelineMatch(
            guideline=g,
            rationale="",
            score=10,
        )
        for g in previously_matched_guidelines
        if (not session.agent_states or g.id not in session.agent_states[-1].applied_guideline_ids)
        and not g.metadata.get("continuous", False)
    ]

    interaction_history_for_analysis = (
        interaction_history[:-1] if len(interaction_history) > 1 else interaction_history
    )  # assume the last message is customer's

    generic_response_analysis_batch = GenericResponseAnalysisBatch(
        logger=context.container[Logger],
        meter=context.container[Meter],
        optimization_policy=context.container[OptimizationPolicy],
        schematic_generator=context.container[SchematicGenerator[GenericResponseAnalysisSchema]],
        context=ResponseAnalysisContext(
            agent=agent,
            session=session,
            customer=customer,
            interaction_history=interaction_history_for_analysis,
            context_variables=[],
            terms=[],
            staged_tool_events=[],
            staged_message_events=[],
        ),
        guideline_matches=matches_to_analyze,
    )

    applied_guideline_ids = [
        g.guideline.id
        for g in (context.sync_await(generic_response_analysis_batch.process())).analyzed_guidelines
        if g.is_previously_applied
    ]

    update_previously_applied_guidelines(context, session_id, applied_guideline_ids)


def base_test_that_correct_guidelines_are_matched(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
    session_id: SessionId,
    conversation_context: list[tuple[EventSource, str]],
    conversation_guideline_names: list[str],
    relevant_guideline_names: list[str],
    previously_applied_guidelines_names: list[str] = [],
    previously_matched_guidelines_names: list[str] = [],
    capabilities: list[Capability] = [],
) -> None:
    interaction_history = [
        create_event_message(
            offset=i,
            source=source,
            message=message,
        )
        for i, (source, message) in enumerate(conversation_context)
    ]

    conversation_guidelines = {
        name: create_guideline_by_name(context, name) for name in conversation_guideline_names
    }

    relevant_guidelines = [conversation_guidelines[name] for name in relevant_guideline_names]

    previously_matched_guidelines = [
        guideline
        for name in previously_matched_guidelines_names
        if (guideline := conversation_guidelines.get(name)) is not None
    ]
    previously_applied_guidelines = [
        guideline.id
        for name in previously_applied_guidelines_names
        if (guideline := conversation_guidelines.get(name)) is not None
    ]

    update_previously_applied_guidelines(
        context=context,
        session_id=session_id,
        applied_guideline_ids=previously_applied_guidelines,
    )

    analyze_response_and_update_session(
        context=context,
        agent=agent,
        session_id=session_id,
        customer=customer,
        previously_matched_guidelines=previously_matched_guidelines,
        interaction_history=interaction_history,
    )

    guideline_matches = match_guidelines(
        context=context,
        agent=agent,
        customer=customer,
        session_id=session_id,
        interaction_history=interaction_history,
        capabilities=capabilities,
    )

    matched_guidelines = [p.guideline for p in guideline_matches]

    assert set(matched_guidelines) == set(relevant_guidelines)


async def check_guideline(
    context: ContextOfTest, guideline: GuidelineContent, is_agent_intention: bool
) -> None:
    agent_intention_detector = context.container[AgentIntentionProposer]
    result = await agent_intention_detector.propose_agent_intention(
        guideline=guideline,
    )
    assert (
        is_agent_intention == result.is_agent_intention
    ), f"""Guideline incorrectly marked as {"not " if is_agent_intention else ""} agent's intention:
Condition: {guideline.condition}
Action: {guideline.action}"""


async def test_that_actions_which_are_agent_intention_are_classified_correctly(
    context: ContextOfTest,
) -> None:
    guidelines = [
        GuidelineContent(
            condition="You answer a question about pricing options",
            action="Include the most up-to-date pricing from the official source",
        ),
        GuidelineContent(
            condition="You are going to provide medical advice",
            action="Add a disclaimer that the information is not a substitute for professional medical care",
        ),
        GuidelineContent(
            condition="You make a recommendation about a product",
            action="Ensure the recommendation is based on factual information",
        ),
        GuidelineContent(
            condition="You likely to make a recommendation about a product",
            action="Ensure the recommendation is based on factual information",
        ),
    ]

    for g in guidelines:
        await check_guideline(context=context, guideline=g, is_agent_intention=True)


async def test_that_actions_which_are_not_agent_intention_are_classified_correctly(
    context: ContextOfTest,
) -> None:
    guidelines = [
        GuidelineContent(
            condition="The customer is going to confirm their shipping address",
            action="Acknowledge and proceed with order processing",
        ),
        GuidelineContent(
            condition="You have already apologized for the inconvenience",
            action="Do not repeat the apology",
        ),
        GuidelineContent(
            condition="The customer asked about return policies",
            action="Provide a link to the official return policy page",
        ),
        GuidelineContent(
            condition="Customer indicated your behavior is likely to cause them harm",
            action="Apologize and ask about what worries the customer",
        ),
    ]

    for g in guidelines:
        await check_guideline(context=context, guideline=g, is_agent_intention=False)


def test_that_guideline_with_agent_intention_is_rewritten_and_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I've been having headaches for the past few days. Could it be something serious?",
        ),
    ]
    conversation_guideline_names: list[str] = ["medical_advice"]
    relevant_guideline_names = conversation_guideline_names

    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


def test_that_guideline_with_agent_intention_is_rewritten_and_matched_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I'm looking for a budget-friendly smartphone under $300. What do you suggest?",
        ),
    ]
    conversation_guideline_names: list[str] = ["recommend_product"]
    relevant_guideline_names = conversation_guideline_names

    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


def test_that_guideline_with_agent_intention_is_rewritten_and_matched_3(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I'm traveling abroad next month and I want to make sure I won’t get charged unexpected fees on my credit card.",
        ),
    ]
    conversation_guideline_names: list[str] = ["international_transaction"]
    relevant_guideline_names = conversation_guideline_names

    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


def test_that_guideline_with_agent_intention_that_was_matched_is_rewritten_and_matched_again(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I’m shopping for laptops. I want something lightweight with good battery life.",
        ),
        (
            EventSource.AI_AGENT,
            "You might want to look at the MacBook Air or the Dell XPS 13. Both are known for being lightweight and having strong battery performance.",
        ),
        (
            EventSource.CUSTOMER,
            "What about something a bit cheaper?",
        ),
    ]
    conversation_guideline_names: list[str] = ["recommend_product"]
    relevant_guideline_names: list[str] = ["recommend_product"]
    previously_matched_guidelines_names: list[str] = ["recommend_product"]
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
        previously_applied_guidelines_names=[],
        previously_matched_guidelines_names=previously_matched_guidelines_names,
    )


def test_that_agent_intention_guideline_is_matched_based_on_capabilities_1(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    capabilities = [
        Capability(
            id=CapabilityId("cap_123"),
            creation_utc=datetime.now(timezone.utc),
            title="Reset Password",
            description="The ability to send the customer an email with a link to reset their password. The password can only be reset via this link",
            signals=["reset password", "password"],
            tags=[],
        )
    ]
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I can't remember the password to my account",
        ),
    ]
    conversation_guideline_names: list[str] = ["multiple_capabilities", "reset_password_offer"]
    relevant_guideline_names: list[str] = ["reset_password_offer"]
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
        capabilities=capabilities,
        previously_applied_guidelines_names=[],
    )
