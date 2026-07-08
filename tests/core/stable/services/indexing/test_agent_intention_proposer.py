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
from typing import Sequence
from lagom import Container
from pytest import fixture

from parlant.core.agents import Agent
from parlant.core.capabilities import Capability, CapabilityId
from parlant.core.common import Weight, JSONSerializable, generate_id
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer
from parlant.core.customers import Customer
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.alpha.rule_matching.generic.response_analysis_batch import (
    GenericResponseAnalysisBatch,
    GenericResponseAnalysisSchema,
)
from parlant.core.engines.rule_match import RuleMatch
from parlant.core.engines.alpha.rule_matching.rule_matcher import (
    RuleMatcher,
    ResponseAnalysisContext,
)
from parlant.core.engines.alpha.engine_context import Interaction, EngineContext, ResponseState
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.types import Context
from parlant.core.entity_cq import EntityCommands
from parlant.core.evaluations import RulePayload, PayloadOperation
from parlant.core.rules import Rule, RuleContent, RuleId
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.indexing.evaluation_service import RuleEvaluator
from parlant.core.services.indexing.rule_agent_intention_proposer import AgentIntentionProposer
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

RULES_DICT = {
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
    rules: list[Rule]
    logger: Logger


@fixture
def context(
    sync_await: SyncAwaiter,
    container: Container,
) -> ContextOfTest:
    return ContextOfTest(
        container,
        sync_await,
        rules=list(),
        logger=container[Logger],
    )


def match_rules(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
    session_id: SessionId,
    interaction_history: Sequence[Event],
    capabilities: Sequence[Capability] = [],
) -> Sequence[RuleMatch]:
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
            ordinary_rule_matches=[],
            tool_enabled_rule_matches={},
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

    rule_matching_result = context.sync_await(
        context.container[RuleMatcher].match_rules(
            context=loaded_context,
            active_journeys=[],
            rules=context.rules,
        )
    )

    return list(rule_matching_result.matched)


def create_rule(
    context: ContextOfTest,
    condition: str,
    action: str | None = None,
) -> Rule:
    metadata: dict[str, JSONSerializable] = {}
    if action:
        rule_evaluator = context.container[RuleEvaluator]
        rule_evaluation_data = context.sync_await(
            rule_evaluator.evaluate(
                payloads=[
                    RulePayload(
                        content=RuleContent(
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

        metadata = rule_evaluation_data[0].properties_proposition or {}

    rule = Rule(
        id=RuleId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        modified_utc=datetime.now(timezone.utc),
        content=RuleContent(
            condition=condition,
            action=action,
        ),
        weight=Weight.MEDIUM,
        enabled=True,
        groups=[],
        metadata=metadata,
    )

    context.rules.append(rule)

    return rule


def create_rule_by_name(
    context: ContextOfTest,
    rule_name: str,
) -> Rule | None:
    if rule_name in RULES_DICT:
        rule = create_rule(
            context=context,
            condition=RULES_DICT[rule_name]["condition"],
            action=RULES_DICT[rule_name]["action"],
        )
    else:
        rule = None
    return rule


def update_previously_applied_rules(
    context: ContextOfTest,
    session_id: SessionId,
    applied_rule_ids: list[RuleId],
) -> None:
    session = context.sync_await(context.container[SessionStore].read_session(session_id))
    applied_rule_ids.extend(
        session.agent_states[-1].applied_rule_ids if session.agent_states else []
    )

    context.sync_await(
        context.container[EntityCommands].update_session(
            session_id=session.id,
            params=SessionUpdateParams(
                agent_states=list(session.agent_states)
                + [
                    AgentState(
                        trace_id="<main>",
                        applied_rule_ids=applied_rule_ids,
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
    previously_matched_rules: list[Rule],
    interaction_history: list[Event],
) -> None:
    session = context.sync_await(context.container[SessionStore].read_session(session_id))

    matches_to_analyze = [
        RuleMatch(
            rule=g,
            rationale="",
        )
        for g in previously_matched_rules
        if (not session.agent_states or g.id not in session.agent_states[-1].applied_rule_ids)
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
        rule_matches=matches_to_analyze,
    )

    applied_rule_ids = [
        g.rule.id
        for g in (context.sync_await(generic_response_analysis_batch.process())).analyzed_rules
        if g.is_previously_applied
    ]

    update_previously_applied_rules(context, session_id, applied_rule_ids)


def base_test_that_correct_rules_are_matched(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
    session_id: SessionId,
    conversation_context: list[tuple[EventSource, str]],
    conversation_rule_names: list[str],
    relevant_rule_names: list[str],
    previously_applied_rules_names: list[str] = [],
    previously_matched_rules_names: list[str] = [],
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

    conversation_rules = {
        name: create_rule_by_name(context, name) for name in conversation_rule_names
    }

    relevant_rules = [conversation_rules[name] for name in relevant_rule_names]

    previously_matched_rules = [
        rule
        for name in previously_matched_rules_names
        if (rule := conversation_rules.get(name)) is not None
    ]
    previously_applied_rules = [
        rule.id
        for name in previously_applied_rules_names
        if (rule := conversation_rules.get(name)) is not None
    ]

    update_previously_applied_rules(
        context=context,
        session_id=session_id,
        applied_rule_ids=previously_applied_rules,
    )

    analyze_response_and_update_session(
        context=context,
        agent=agent,
        session_id=session_id,
        customer=customer,
        previously_matched_rules=previously_matched_rules,
        interaction_history=interaction_history,
    )

    rule_matches = match_rules(
        context=context,
        agent=agent,
        customer=customer,
        session_id=session_id,
        interaction_history=interaction_history,
        capabilities=capabilities,
    )

    matched_rules = [p.rule for p in rule_matches]

    assert set(matched_rules) == set(relevant_rules)


async def check_rule(context: ContextOfTest, rule: RuleContent, is_agent_intention: bool) -> None:
    agent_intention_detector = context.container[AgentIntentionProposer]
    result = await agent_intention_detector.propose_agent_intention(
        rule=rule,
    )
    assert (
        is_agent_intention == result.is_agent_intention
    ), f"""Rule incorrectly marked as {"not " if is_agent_intention else ""} agent's intention:
Condition: {rule.condition}
Action: {rule.action}"""


async def test_that_actions_which_are_agent_intention_are_classified_correctly(
    context: ContextOfTest,
) -> None:
    rules = [
        RuleContent(
            condition="You answer a question about pricing options",
            action="Include the most up-to-date pricing from the official source",
        ),
        RuleContent(
            condition="You are going to provide medical advice",
            action="Add a disclaimer that the information is not a substitute for professional medical care",
        ),
        RuleContent(
            condition="You make a recommendation about a product",
            action="Ensure the recommendation is based on factual information",
        ),
        RuleContent(
            condition="You likely to make a recommendation about a product",
            action="Ensure the recommendation is based on factual information",
        ),
    ]

    for g in rules:
        await check_rule(context=context, rule=g, is_agent_intention=True)


async def test_that_actions_which_are_not_agent_intention_are_classified_correctly(
    context: ContextOfTest,
) -> None:
    rules = [
        RuleContent(
            condition="The customer is going to confirm their shipping address",
            action="Acknowledge and proceed with order processing",
        ),
        RuleContent(
            condition="You have already apologized for the inconvenience",
            action="Do not repeat the apology",
        ),
        RuleContent(
            condition="The customer asked about return policies",
            action="Provide a link to the official return policy page",
        ),
        RuleContent(
            condition="Customer indicated your behavior is likely to cause them harm",
            action="Apologize and ask about what worries the customer",
        ),
        RuleContent(
            condition="The customer gives very short snappy responses like 'ok', 'sure', 'got it'",
            action="Keep the next point brief, one sentence maximum",
        ),
        RuleContent(
            condition="The customer has an inquiry that sounds high-level or basic, not drilling into specifics or details",
            action="Answer ONLY based on the information provided",
        ),
    ]

    for g in rules:
        await check_rule(context=context, rule=g, is_agent_intention=False)


async def test_that_actions_using_the_word_likely_arent_falsely_detected_as_agent_intention(
    context: ContextOfTest,
) -> None:
    rules = [
        RuleContent(
            condition="You are likely to encounter a very short and vague question from the customer, like 'credit cards' or 'dispute'",
            action="refer the customer to our manual",
        ),
    ]

    for g in rules:
        await check_rule(context=context, rule=g, is_agent_intention=False)


def test_that_rule_with_agent_intention_is_rewritten_and_matched(
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
    conversation_rule_names: list[str] = ["medical_advice"]
    relevant_rule_names = conversation_rule_names

    base_test_that_correct_rules_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_rule_names,
        relevant_rule_names,
    )


def test_that_rule_with_agent_intention_is_rewritten_and_matched_2(
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
    conversation_rule_names: list[str] = ["recommend_product"]
    relevant_rule_names = conversation_rule_names

    base_test_that_correct_rules_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_rule_names,
        relevant_rule_names,
    )


def test_that_rule_with_agent_intention_is_rewritten_and_matched_3(
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
    conversation_rule_names: list[str] = ["international_transaction"]
    relevant_rule_names = conversation_rule_names

    base_test_that_correct_rules_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_rule_names,
        relevant_rule_names,
    )


def test_that_rule_with_agent_intention_that_was_matched_is_rewritten_and_matched_again(
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
    conversation_rule_names: list[str] = ["recommend_product"]
    relevant_rule_names: list[str] = ["recommend_product"]
    previously_matched_rules_names: list[str] = ["recommend_product"]
    base_test_that_correct_rules_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_rule_names,
        relevant_rule_names,
        previously_applied_rules_names=[],
        previously_matched_rules_names=previously_matched_rules_names,
    )


def test_that_agent_intention_rule_is_matched_based_on_capabilities_1(
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
            groups=[],
        )
    ]
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I can't remember the password to my account",
        ),
    ]
    conversation_rule_names: list[str] = ["multiple_capabilities", "reset_password_offer"]
    relevant_rule_names: list[str] = ["reset_password_offer"]
    base_test_that_correct_rules_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_rule_names,
        relevant_rule_names,
        capabilities=capabilities,
        previously_applied_rules_names=[],
    )
