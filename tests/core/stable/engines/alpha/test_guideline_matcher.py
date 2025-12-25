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
from typing import Sequence, cast
from typing_extensions import override

from lagom import Container
from more_itertools import unique
from pytest import fixture, raises

from parlant.core.agents import Agent, AgentId
from parlant.core.capabilities import Capability, CapabilityId
from parlant.core.common import Criticality, generate_id, JSONSerializable
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableId,
    ContextVariableValue,
    ContextVariableValueId,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matching_context import (
    GuidelineMatchingContext,
)
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer
from parlant.core.customers import Customer
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.generic_guideline_matching_strategy_resolver import (
    GenericGuidelineMatchingStrategyResolver,
)
from parlant.core.engines.alpha.guideline_matching.generic.response_analysis_batch import (
    GenericResponseAnalysisBatch,
    GenericResponseAnalysisSchema,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatcher,
    GuidelineMatchingBatch,
    GuidelineMatchingBatchResult,
    ResponseAnalysisBatch,
    ResponseAnalysisBatchResult,
    ResponseAnalysisContext,
    GuidelineMatchingStrategy,
    GuidelineMatchingStrategyResolver,
)
from parlant.core.engines.alpha.engine_context import Interaction, EngineContext, ResponseState
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.types import Context
from parlant.core.entity_cq import EntityCommands
from parlant.core.evaluations import GuidelinePayload, PayloadOperation
from parlant.core.glossary import Term
from parlant.core.journeys import Journey
from parlant.core.nlp.generation import SchematicGenerator

from parlant.core.engines.alpha.guideline_matching.guideline_match import (
    GuidelineMatch,
    AnalyzedGuideline,
)
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.relationships import (
    RelationshipKind,
    RelationshipEntity,
    RelationshipEntityKind,
    RelationshipStore,
)
from parlant.core.services.indexing.behavioral_change_evaluation import GuidelineEvaluator
from parlant.core.sessions import (
    AgentState,
    Event,
    EventKind,
    EventSource,
    Session,
    SessionId,
    SessionStore,
    SessionUpdateParams,
)
from parlant.core.loggers import Logger
from parlant.core.glossary import TermId

from parlant.core.tags import TagId, Tag
from tests.core.common.utils import create_event_message
from tests.test_utilities import SyncAwaiter


OBSERVATIONAL_GUIDELINES_DICT = {
    "vegetarian_customer": {
        "condition": "the customer is vegetarian or vegan",
        "observation": "-",
    },
    "ever_requested_lock_card": {
        "condition": "the customer ever indicated that they wish to lock their credit card",
        "observation": "-",
    },
    "season_is_winter": {
        "condition": "it is the season of winter",
        "observation": "-",
    },
    "frustrated_customer_observational": {
        "condition": "the customer is frustrated",
        "observation": "-",
    },
    "unclear_request": {
        "condition": "the customer indicates that the agent does not understand their request",
        "observation": "-",
    },
    "credit_limits_discussion": {
        "condition": "credit limits are discussed",
        "observation": "-",
    },
    "unknown_service": {
        "condition": "The customer is asking for a service you don't recognize according to this prompt information",
        "observation": "-",
    },
    "delivery_order": {
        "condition": "the customer is in the process of ordering delivery",
        "observation": "-",
    },
    "unanswered_questions": {
        "condition": "the customer repeatedly ignores the agent's question, and they remain unanswered",
        "observation": "-",
    },
    "unsupported_capability": {
        "condition": "When a customer asks about a capability that is not supported",
        "observation": "-",
    },
    "reset_password": {
        "condition": "The customer currently wants to reset their password",
        "observation": "-",
    },
    "lost_card": {
        "condition": "The customer says that they lost their card",
        "observation": "-",
    },
    "business_class": {
        "condition": "The customer expresses a preference for business class.",
        "observation": "-",
    },
    "book_flight": {
        "condition": "The customer wants to book a flight",
        "observation": "-",
    },
    "book_flight_2": {
        "condition": "The conversation is about flight booking",
        "observation": "-",
    },
}

ACTIONABLE_GUIDELINES_DICT = {
    "check_drinks_in_stock": {
        "condition": "a customer asks for a drink",
        "action": "check if the drink is available in the following stock: "
        "['Sprite', 'Coke', 'Fanta']. Assume that if a drink is on stock, we have enough of it",
    },
    "check_toppings_in_stock": {
        "condition": "a customer asks for toppings",
        "action": "check if the toppings are available in the following stock: "
        "['Pepperoni', 'Tomatoes', 'Olives']. Assume that if a topping is on stock, we have enough of it",
    },
    "payment_process": {
        "condition": "a customer is in the payment process",
        "action": "Follow the payment instructions, "
        "which are: 1. Pay in cash only, 2. Pay only at the location.",
    },
    "address_location": {
        "condition": "the customer needs to know our address",
        "action": "Inform the customer that our address is at Sapir 2, Herzliya.",
    },
    "issue_resolved": {
        "condition": "the customer previously expressed stress or dissatisfaction, but the issue has been alleviated",
        "action": "confirm the issue is fully resolved",
    },
    "class_booking": {
        "condition": "the customer asks about booking a class or an appointment",
        "action": "Provide available times and facilitate the booking process, "
        "ensuring to clarify any necessary details such as class type.",
    },
    "class_cancellation": {
        "condition": "the customer wants to cancel a class or an appointment",
        "action": "ask for the reason of cancellation, unless it's an emergency mention the cancellation fee.",
    },
    "frustrated_customer": {
        "condition": "the customer appears frustrated or upset",
        "action": "Acknowledge the customer's concerns, apologize for any inconvenience, and offer a solution or escalate the issue to a supervisor if necessary.",
    },
    "thankful_customer": {
        "condition": "the customer expresses gratitude or satisfaction",
        "action": "Acknowledge their thanks warmly and let them know you appreciate their feedback or kind words.",
    },
    "hesitant_customer": {
        "condition": "the customer seems unsure or indecisive about a decision",
        "action": "Offer additional information, provide reassurance, and suggest the most suitable option based on their needs.",
    },
    "holiday_season": {
        "condition": "the interaction takes place during the holiday season",
        "action": "Mention any holiday-related offers, adjusted schedules, or greetings to make the interaction festive and accommodating.",
    },
    "previous_issue_resurfaced": {
        "condition": "the customer brings up an issue they previously experienced",
        "action": "Acknowledge the previous issue, apologize for any inconvenience, and take immediate steps to resolve it or escalate if needed.",
    },
    "question_already_answered": {
        "condition": "the customer asks a question that has already been answered",
        "action": "Politely reiterate the information and ensure they understand or provide additional clarification if needed.",
    },
    "product_out_of_stock": {
        "condition": "the customer asks for a product that is currently unavailable",
        "action": "Apologize for the inconvenience, inform them of the unavailability, and suggest alternative products or notify them of restocking timelines if available.",
    },
    "technical_issue": {
        "condition": "the customer reports a technical issue with the website or service",
        "action": "Acknowledge the issue, apologize for the inconvenience, and guide them through troubleshooting steps or escalate the issue to the technical team.",
    },
    "first_time_customer": {
        "condition": "the customer mentions it is their first time using the service",
        "action": "Welcome them warmly, provide a brief overview of how the service works, and offer any resources to help them get started.",
    },
    "request_for_feedback": {
        "condition": "the customer is asked for feedback about the service or product",
        "action": "Politely request their feedback, emphasizing its value for improvement, and provide simple instructions for submitting their response.",
    },
    "customer_refers_friends": {
        "condition": "the customer mentions referring friends to the service or product",
        "action": "Thank them sincerely for the referral and mention any referral rewards or benefits if applicable.",
    },
    "check_age": {
        "condition": "the conversation necessitates checking for the age of the customer",
        "action": "Use the 'check_age' tool to check for their age",
    },
    "suggest_drink_underage": {
        "condition": "an underage customer asks for drink recommendations",
        "action": "recommend a soda pop",
    },
    "suggest_drink_adult": {
        "condition": "an adult customer asks for drink recommendations",
        "action": "recommend either wine or beer",
    },
    "announce_shipment": {
        "condition": "the agent just confirmed that the order will be shipped to the customer",
        "action": "provide the package's tracking information",
    },
    "tree_allergies": {
        "condition": "recommending routes to a customer with tree allergies",
        "action": "warn the customer about allergy inducing trees along the route",
    },
    "credit_payment1": {
        "condition": "the customer requests a credit card payment",
        "action": "guide the customer through the payment process",
    },
    "credit_payment2": {
        "condition": "the customer wants to pay with a credit card",
        "action": "refuse payment as we only perform in-store purchases",
    },
    "cant_perform_request": {
        "condition": "the customer wants to agent to perform an action that you are not designed for",
        "action": "forward the request to a supervisor",
    },
    "announce_deals": {
        "condition": "A special deal is active",
        "action": "Announce the deal in an excited tone, while mentioning our slogan 'Ride the Future, One Kick at a Time!'",
    },
    "cheese_pizza": {
        "condition": "The customer is in the process of ordering a cheese pizza",
        "action": "Ask which toppings they would like",
    },
    "cheese_pizza_process": {
        "condition": "The customer is in the process of ordering a cheese pizza",
        "action": "Refer to the pizza as a 'pie'",
    },
    "summer_sale": {
        "condition": "In the season of summer",
        "action": "Mention we offer two large pizzas for the price of one",
    },
    "large_pizza_crust": {
        "condition": "The customer orders a large pizza",
        "action": "Ask what type of crust they would like",
    },
    "add_to_count": {
        "condition": "the customer asks you to add 1 to the count",
        "action": "Search the interaction history for the most recent count, add 1 to it and respond with the new count",
    },
    "cow_response": {"condition": "The customer says hello", "action": "respond like a cow would"},
    "many_actions": {
        "condition": "the customer asked a question about birds",
        "action": "answer their question enthusiastically, while not using punctuation. Also say that the kingfisher is your favorite bird",
    },
    "medical_record": {
        "condition": "you are likely to discuss a patient's medical record",
        "action": "Do not send any personal information",
    },
    "provide_diagnosis": {
        "condition": "you are likely to provide a diagnosis or medical advice.",
        "action": "Ensure the message includes a disclaimer that it is not a substitute for professional medical advice.",
    },
    "confirm_order": {
        "condition": "you are likely to confirm a new order or a payment",
        "action": "Re-verify item, price, and customer consent before proceeding",
    },
    "discuss_money": {
        "condition": "you are likely to discuss account balances or transactions.",
        "action": "Require customer authentication confirmation before responding.",
    },
    "human_resources": {
        "condition": "you are likely going to share a candidate’s application status",
        "action": "Avoid disclosing internal evaluation notes or third-party feedback",
    },
    "snake_roller_coaster": {
        "condition": "the customer asks for the snake roller coaster",
        "action": "book it",
    },
    "turtle_roller_coaster": {
        "condition": "the customer asks for the turtle roller coaster",
        "action": "book it",
    },
    "tiger_Ferris_wheel": {
        "condition": "the customer asks for the tiger Ferris wheel",
        "action": "book it",
    },
    "replace_card": {
        "condition": "The user wants to replace their card",
        "action": "List the cards and then assist the user to replace their card until matter is resolved",
    },
}

DISAMBIGUATION_GUIDELINES_DICT = {
    "amusement_park": "The customer asks to book a ticket to an amusement ride or attraction, and its not clear which one",
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


async def match_guidelines(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
    session_id: SessionId,
    interaction_history: Sequence[Event],
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
    terms: Sequence[Term] = [],
    capabilities: Sequence[Capability] = [],
    activated_journeys: Sequence[Journey] = [],
    staged_events: Sequence[EmittedEvent] = [],
) -> Sequence[GuidelineMatch]:
    session = await context.container[SessionStore].read_session(session_id)

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
            context_variables=list(context_variables),
            glossary_terms=set(terms),
            capabilities=list(capabilities),
            iterations=[],
            ordinary_guideline_matches=[],
            tool_enabled_guideline_matches={},
            journeys=[],
            journey_paths={k: list(v) for k, v in session.agent_states[-1].journey_paths.items()}
            if session.agent_states
            else {},
            tool_events=list(staged_events),
            tool_insights=ToolInsights(),
            prepared_to_respond=False,
            message_events=[],
        ),
    )

    guideline_matching_result = await context.container[GuidelineMatcher].match_guidelines(
        context=loaded_context,
        active_journeys=activated_journeys,
        guidelines=context.guidelines,
    )

    return list(chain.from_iterable(guideline_matching_result.batches))


async def create_guideline(
    context: ContextOfTest,
    condition: str,
    action: str | None = None,
    tags: list[TagId] = [],
) -> Guideline:
    metadata: dict[str, JSONSerializable] = {}
    if action:
        guideline_evaluator = context.container[GuidelineEvaluator]
        guideline_evaluation_data = await guideline_evaluator.evaluate(
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

        metadata = guideline_evaluation_data[0].properties_proposition or {}

    guideline = Guideline(
        id=GuidelineId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        content=GuidelineContent(
            condition=condition,
            action=action,
        ),
        enabled=True,
        tags=tags,
        metadata=metadata,
        criticality=Criticality.MEDIUM,
    )

    context.guidelines.append(guideline)

    return guideline


async def create_disambiguation_guideline(
    context: ContextOfTest, condition: str, guidelines: list[Guideline]
) -> Guideline:
    guideline = Guideline(
        id=GuidelineId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        content=GuidelineContent(
            condition=condition,
            action=None,
        ),
        enabled=True,
        tags=[],
        metadata={},
        criticality=Criticality.MEDIUM,
    )

    context.guidelines.append(guideline)

    for g in guidelines:
        await context.container[RelationshipStore].create_relationship(
            source=RelationshipEntity(
                id=guideline.id,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=g.id,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            kind=RelationshipKind.DISAMBIGUATION,
        )

    return guideline


def create_term(
    name: str, description: str, synonyms: list[str] = [], tags: list[TagId] = []
) -> Term:
    return Term(
        id=TermId("-"),
        creation_utc=datetime.now(timezone.utc),
        name=name,
        description=description,
        synonyms=synonyms,
        tags=tags,
    )


def create_context_variable(
    name: str,
    data: JSONSerializable,
    tags: list[TagId],
) -> tuple[ContextVariable, ContextVariableValue]:
    return ContextVariable(
        id=ContextVariableId("-"),
        creation_utc=datetime.now(timezone.utc),
        name=name,
        description="",
        tool_id=None,
        freshness_rules=None,
        tags=tags,
    ), ContextVariableValue(
        ContextVariableValueId("-"),
        last_modified=datetime.now(timezone.utc),
        data=data,
    )


async def create_guideline_by_name(
    context: ContextOfTest,
    guideline_name: str,
    disambiguating_targets: list[Guideline] = [],
) -> Guideline | None:
    if guideline_name in ACTIONABLE_GUIDELINES_DICT:
        guideline = await create_guideline(
            context=context,
            condition=ACTIONABLE_GUIDELINES_DICT[guideline_name]["condition"],
            action=ACTIONABLE_GUIDELINES_DICT[guideline_name]["action"],
        )
    elif guideline_name in OBSERVATIONAL_GUIDELINES_DICT:
        guideline = await create_guideline(
            context=context,
            condition=OBSERVATIONAL_GUIDELINES_DICT[guideline_name]["condition"],
        )
    elif guideline_name in DISAMBIGUATION_GUIDELINES_DICT:
        guideline = await create_disambiguation_guideline(
            context=context,
            condition=DISAMBIGUATION_GUIDELINES_DICT[guideline_name],
            guidelines=disambiguating_targets,
        )
    else:
        guideline = None
    return guideline


async def update_previously_applied_guidelines(
    context: ContextOfTest,
    session_id: SessionId,
    applied_guideline_ids: list[GuidelineId],
) -> None:
    session = await context.container[SessionStore].read_session(session_id)
    applied_guideline_ids.extend(
        session.agent_states[-1].applied_guideline_ids if session.agent_states else []
    )

    await context.container[EntityCommands].update_session(
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


async def analyze_response_and_update_session(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
    session_id: SessionId,
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
    terms: Sequence[Term],
    staged_tool_events: Sequence[EmittedEvent],
    staged_message_events: Sequence[EmittedEvent],
    previously_matched_guidelines: list[Guideline],
    interaction_history: list[Event],
) -> None:
    session = await context.container[SessionStore].read_session(session_id)

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
            context_variables=context_variables,
            terms=terms,
            staged_tool_events=staged_tool_events,
            staged_message_events=staged_message_events,
        ),
        guideline_matches=matches_to_analyze,
    )

    applied_guideline_ids = [
        g.guideline.id
        for g in (await generic_response_analysis_batch.process()).analyzed_guidelines
        if g.is_previously_applied
    ]

    await update_previously_applied_guidelines(context, session_id, applied_guideline_ids)


async def base_test_that_correct_guidelines_are_matched(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
    session_id: SessionId,
    conversation_context: list[tuple[EventSource, str]],
    conversation_guideline_names: list[str],
    relevant_guideline_names: list[str],
    previously_applied_guidelines_names: list[str] = [],
    previously_matched_guidelines_names: list[str] = [],
    disambiguation_guideline_name: str = "",
    disambiguation_targets_names: list[str] = [],
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
    terms: Sequence[Term] = [],
    capabilities: Sequence[Capability] = [],
    staged_tool_events: Sequence[EmittedEvent] = [],
    staged_message_events: Sequence[EmittedEvent] = [],
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
        name: await create_guideline_by_name(context, name) for name in conversation_guideline_names
    }

    if disambiguation_guideline_name:
        targets = [
            guideline
            for name in disambiguation_targets_names
            if (guideline := conversation_guidelines.get(name)) is not None
        ]
        conversation_guidelines[disambiguation_guideline_name] = await create_guideline_by_name(
            context,
            disambiguation_guideline_name,
            disambiguating_targets=targets,
        )

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

    await update_previously_applied_guidelines(
        context=context,
        session_id=session_id,
        applied_guideline_ids=previously_applied_guidelines,
    )

    await analyze_response_and_update_session(
        context=context,
        agent=agent,
        session_id=session_id,
        customer=customer,
        context_variables=context_variables,
        terms=terms,
        staged_tool_events=staged_tool_events,
        staged_message_events=staged_message_events,
        previously_matched_guidelines=previously_matched_guidelines,
        interaction_history=interaction_history,
    )

    guideline_matches = await match_guidelines(
        context=context,
        agent=agent,
        customer=customer,
        session_id=session_id,
        interaction_history=interaction_history,
        context_variables=context_variables,
        terms=terms,
        staged_events=staged_tool_events,
        capabilities=capabilities,
    )

    matched_guidelines_ids = [p.guideline.id for p in guideline_matches]
    relevant_guidelines_ids = [g.id for g in relevant_guidelines if g is not None]

    assert set(matched_guidelines_ids) == set(relevant_guidelines_ids)


async def test_that_relevant_guidelines_are_matched_parametrized_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I'm feeling a bit stressed about coming in. Can I cancel my class for today?",
        ),
        (
            EventSource.AI_AGENT,
            "I'm sorry to hear that. While cancellation is not possible now, "
            "how about a lighter session? Maybe it helps to relax.",
        ),
        (
            EventSource.CUSTOMER,
            "I suppose that could work. What do you suggest?",
        ),
        (
            EventSource.AI_AGENT,
            "How about our guided meditation session every Tuesday evening at 20:00? "
            "It's very calming and might be just what you need right now.",
        ),
        (
            EventSource.CUSTOMER,
            "Alright, please book me into that. Thank you for understanding.",
        ),
        (
            EventSource.AI_AGENT,
            "You're welcome! I've switched your booking to the meditation session. "
            "Remember, it's okay to feel stressed. We're here to support you.",
        ),
        (
            EventSource.CUSTOMER,
            "Thanks, I really appreciate it.",
        ),
        (
            EventSource.AI_AGENT,
            "Anytime! Is there anything else I can assist you with today?",
        ),
        (
            EventSource.CUSTOMER,
            "No, that's all for now.",
        ),
        (
            EventSource.AI_AGENT,
            "Take care and see you soon at the meditation class. "
            "Our gym is at the mall on the 2nd floor.",
        ),
        (
            EventSource.CUSTOMER,
            "Thank you!",
        ),
    ]
    conversation_guideline_names: list[str] = [
        "class_booking",
        "issue_resolved",
        "address_location",
    ]

    relevant_guideline_names: list[str] = ["issue_resolved"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_irrelevant_guidelines_are_not_matched_parametrized_1(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "I'd like to order a pizza, please."),
        (EventSource.AI_AGENT, "No problem. What would you like to have?"),
        (EventSource.CUSTOMER, "I'd like a large pizza. What toppings do you have?"),
        (EventSource.AI_AGENT, "Today we have pepperoni, tomatoes, and olives available."),
        (EventSource.CUSTOMER, "I'll take pepperoni, thanks."),
        (
            EventSource.AI_AGENT,
            "Awesome. I've added a large pepperoni pizza. Would you like a drink on the side?",
        ),
        (
            EventSource.CUSTOMER,
            "Sure. What types of drinks do you have?",
        ),
        (
            EventSource.AI_AGENT,
            "We have Sprite, Coke, and Fanta.",
        ),
        (EventSource.CUSTOMER, "I'll take two Sprites, please."),
        (EventSource.AI_AGENT, "Anything else?"),
        (EventSource.CUSTOMER, "No, that's all."),
        (EventSource.AI_AGENT, "How would you like to pay?"),
        (EventSource.CUSTOMER, "I'll pick it up and pay in cash, thanks."),
    ]

    conversation_guideline_names: list[str] = ["check_toppings_in_stock", "check_drinks_in_stock"]
    previously_applied_actionable_guidelines_names: list[str] = [
        "check_toppings_in_stock",
        "check_drinks_in_stock",
    ]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names=[],
        previously_applied_guidelines_names=previously_applied_actionable_guidelines_names,
    )


async def test_that_guidelines_with_the_same_conditions_are_scored_similarly(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    relevant_guidelines = [
        await create_guideline(
            context=context,
            condition="the customer greets you",
            action="talk about apples",
        ),
        await create_guideline(
            context=context,
            condition="the customer greets you",
            action="talk about oranges",
        ),
    ]

    _ = [  # irrelevant guidelines
        await create_guideline(
            context=context,
            condition="talking about the weather",
            action="talk about apples",
        ),
        await create_guideline(
            context=context,
            condition="talking about the weather",
            action="talk about oranges",
        ),
    ]

    interaction_history = [
        create_event_message(
            offset=0,
            source=EventSource.CUSTOMER,
            message="Hello there",
        )
    ]

    guideline_matches = await match_guidelines(
        context,
        agent,
        customer,
        new_session.id,
        interaction_history,
    )

    assert len(guideline_matches) == len(relevant_guidelines)
    assert all(gp.guideline in relevant_guidelines for gp in guideline_matches)
    matches_scores = list(unique(gp.score for gp in guideline_matches))
    assert len(matches_scores) == 1 or (
        len(matches_scores) == 2 and abs(matches_scores[0] - matches_scores[1]) <= 1
    )


async def test_that_guidelines_are_matched_based_on_agent_description(
    context: ContextOfTest,
    customer: Customer,
) -> None:
    agent = Agent(
        id=AgentId("123"),
        creation_utc=datetime.now(timezone.utc),
        name="skateboard-sales-agent",
        description="You are an agent working for a skateboarding manufacturer. You help customers by discussing and recommending our products."
        "Your role is only to consult customers, and not to actually sell anything, as we sell our products in-store.",
        max_engine_iterations=3,
        tags=[],
    )

    session = await context.container[SessionStore].create_session(
        customer_id=customer.id,
        agent_id=agent.id,
    )

    conversation_context: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hey, do you sell skateboards?"),
        (
            EventSource.AI_AGENT,
            "Yes, we do! We have a variety of skateboards for all skill levels. Are you looking for something specific?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm looking for a skateboard for a beginner. What do you recommend?",
        ),
        (
            EventSource.AI_AGENT,
            "For beginners, I recommend our complete skateboards with a sturdy deck and softer wheels for easier control. Would you like to see some options?",
        ),
        (EventSource.CUSTOMER, "That sounds perfect. Can you show me a few?"),
        (
            EventSource.AI_AGENT,
            "Sure! We have a few options: the 'Smooth Ride' model, the 'City Cruiser,' and the 'Basic Starter.' Which one would you like to know more about?",
        ),
        (EventSource.CUSTOMER, "I like the 'City Cruiser.' What color options do you have?"),
        (
            EventSource.AI_AGENT,
            "The 'City Cruiser' comes in red, blue, and black. Which one do you prefer?",
        ),
        (
            EventSource.CUSTOMER,
            "I'll go with the blue one. My credit card number is 4242 4242 4242 4242, please charge it and ship the product to my address.",
        ),
    ]

    conversation_guideline_names: list[str] = ["cant_perform_request"]
    relevant_guideline_names = ["cant_perform_request"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_guidelines_are_matched_based_on_glossary(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    terms = [
        create_term(
            name="skateboard",
            description="a time-traveling device",
            tags=[Tag.for_agent_id(agent.id)],
        ),
        create_term(
            name="Pinewood Rash Syndrome",
            description="allergy to pinewood trees",
            synonyms=["Pine Rash", "PRS"],
            tags=[Tag.for_agent_id(agent.id)],
        ),
    ]
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I'm looking for a hiking route through a forest. Can you help me?",
        ),
        (
            EventSource.AI_AGENT,
            "Of course! I can help you find a trail. Are you looking for an easy, moderate, or challenging hike?",
        ),
        (
            EventSource.CUSTOMER,
            "I'd prefer something moderate, not too easy but also not too tough.",
        ),
        (
            EventSource.AI_AGENT,
            "Great choice! We have a few moderate trails in the Redwood Forest and the Pinewood Trail. Would you like details on these?",
        ),
        (EventSource.CUSTOMER, "Yes, tell me more about the Pinewood Trail."),
        (
            EventSource.AI_AGENT,
            "The Pinewood Trail is a 6-mile loop with moderate elevation changes. It takes about 3-4 hours to complete. The scenery is beautiful, with plenty of shade and a stream crossing halfway through. Would you like to go with that one?",
        ),
        (EventSource.CUSTOMER, "I have PRS, would that route be suitable for me?"),
    ]
    conversation_guideline_names: list[str] = ["tree_allergies"]
    relevant_guideline_names = ["tree_allergies"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
        terms=terms,
    )


async def test_that_conflicting_actions_with_similar_conditions_are_both_detected(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hey, do you sell skateboards?"),
        (
            EventSource.AI_AGENT,
            "Yes, we do! We have a variety of skateboards for all skill levels. Are you looking for something specific?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm looking for a skateboard for a beginner. What do you recommend?",
        ),
        (
            EventSource.AI_AGENT,
            "For beginners, I recommend our complete skateboards with a sturdy deck and softer wheels for easier control. Would you like to see some options?",
        ),
        (
            EventSource.CUSTOMER,
            "That sounds perfect. Can you show me a few?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! We have a few options: the 'Smooth Ride' model, the 'City Cruiser,' and the 'Basic Starter.' Which one would you like to know more about?",
        ),
        (
            EventSource.CUSTOMER,
            "I like the 'City Cruiser.' What color options do you have?",
        ),
        (
            EventSource.AI_AGENT,
            "The 'City Cruiser' comes in red, blue, and black. Which one do you prefer?",
        ),
        (
            EventSource.CUSTOMER,
            "I'll go with the blue one.",
        ),
        (
            EventSource.AI_AGENT,
            "Great choice! I'll add the blue 'City Cruiser' to your cart. Would you like to add any accessories like a helmet or grip tape?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, I'll take a helmet. What do you have in stock?",
        ),
        (
            EventSource.AI_AGENT,
            "We have helmets in small, medium, and large sizes, all available in black and gray. What size do you need?",
        ),
        (
            EventSource.CUSTOMER,
            "I need a medium. I'll take one in black.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it! Your blue 'City Cruiser' skateboard and black medium helmet are ready for checkout. How would you like to pay?",
        ),
        (
            EventSource.CUSTOMER,
            "I'll pay with a credit card, thanks.",
        ),
    ]
    conversation_guideline_names: list[str] = ["credit_payment1", "credit_payment2"]
    relevant_guideline_names = conversation_guideline_names
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_guidelines_are_matched_based_on_staged_tool_calls_and_context_variables(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi there, I want a drink that's on the sweeter side, what would you suggest?",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! Let me take a quick look at your account to recommend the best product for you. Could you please provide your full name?",
        ),
        (EventSource.CUSTOMER, "I'm Bob Bobberson"),
    ]
    tool_result_1 = cast(
        JSONSerializable,
        {
            "tool_calls": [
                {
                    "tool_id": "local:get_user_age",
                    "arguments": {"user_id": "199877"},
                    "result": {"data": 16, "metadata": {}, "control": {}},
                }
            ]
        },
    )

    tool_result_2 = cast(
        JSONSerializable,
        {
            "tool_calls": [
                {
                    "tool_id": "local:get_user_age",
                    "arguments": {"user_id": "816779"},
                    "result": {"data": 30, "metadata": {}, "control": {}},
                }
            ]
        },
    )
    staged_tool_events = [
        EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.TOOL,
            trace_id="",
            data=tool_result_1,
            metadata=None,
        ),
        EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.TOOL,
            trace_id="",
            data=tool_result_2,
            metadata=None,
        ),
    ]

    context_variables = [
        create_context_variable(
            name="user_id_1",
            data={"name": "Jimmy McGill", "ID": 566317},
            tags=[Tag.for_agent_id(agent.id)],
        ),
        create_context_variable(
            name="user_id_2",
            data={"name": "Bob Bobberson", "ID": 199877},
            tags=[Tag.for_agent_id(agent.id)],
        ),
        create_context_variable(
            name="user_id_3",
            data={"name": "Dorothy Dortmund", "ID": 816779},
            tags=[Tag.for_agent_id(agent.id)],
        ),
    ]
    conversation_guideline_names: list[str] = ["suggest_drink_underage", "suggest_drink_adult"]
    relevant_guideline_names = ["suggest_drink_underage"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
        staged_tool_events=staged_tool_events,
        context_variables=context_variables,
    )


async def test_that_guidelines_are_matched_based_on_staged_tool_calls_without_context_variables(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi there, I want a drink that's on the sweeter side, what would you suggest?",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! Let me take a quick look at your account to recommend the best product for you. Could you please provide your ID number?",
        ),
        (EventSource.CUSTOMER, "It's 199877"),
    ]

    tool_result_1 = cast(
        JSONSerializable,
        {
            "tool_calls": [
                {
                    "tool_id": "local:get_user_age",
                    "arguments": {"user_id": "199877"},
                    "result": {"data": 16, "metadata": {}, "control": {}},
                }
            ]
        },
    )

    tool_result_2 = cast(
        JSONSerializable,
        {
            "tool_calls": [
                {
                    "tool_id": "local:get_user_age",
                    "arguments": {"user_id": "816779"},
                    "result": {"data": 30, "metadata": {}, "control": {}},
                }
            ]
        },
    )
    staged_events = [
        EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.TOOL,
            trace_id="",
            data=tool_result_1,
            metadata=None,
        ),
        EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.TOOL,
            trace_id="",
            data=tool_result_2,
            metadata=None,
        ),
    ]
    conversation_guideline_names: list[str] = ["suggest_drink_underage", "suggest_drink_adult"]
    relevant_guideline_names = ["suggest_drink_underage"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names=relevant_guideline_names,
        staged_tool_events=staged_events,
    )


async def test_that_already_addressed_guidelines_are_not_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hey there, can I get one cheese pizza?"),
        (EventSource.AI_AGENT, "Of course! What toppings would you like?"),
        (EventSource.CUSTOMER, "Mushrooms if they're fresh"),
        (
            EventSource.AI_AGENT,
            "All of our toppings are fresh! Are you collecting it from our shop or should we ship it to your address?",
        ),
        (EventSource.CUSTOMER, "Ship it to my address please"),
    ]
    conversation_guideline_names: list[str] = ["cheese_pizza"]
    previously_applied_actionable_guidelines_names: list[str] = ["cheese_pizza"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names=[],
        previously_applied_guidelines_names=previously_applied_actionable_guidelines_names,
    )


async def test_that_guidelines_referring_to_continuous_processes_are_detected_even_if_already_fulfilled(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hey there, can I get one cheese pizza?"),
        (
            EventSource.AI_AGENT,
            "Of course! What toppings would you like on your pie?",
        ),
        (EventSource.CUSTOMER, "Mushrooms if they're fresh"),
        (
            EventSource.AI_AGENT,
            "All of our toppings are fresh! Are you collecting the pie from our shop or should we ship it to your address?",
        ),
        (EventSource.CUSTOMER, "Ship it to my address please"),
    ]
    conversation_guideline_names: list[str] = ["cheese_pizza_process"]
    relevant_guideline_names = conversation_guideline_names
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_guideline_with_already_addressed_condition_but_unaddressed_action_is_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hey there, can I get one cheese pizza?"),
        (
            EventSource.AI_AGENT,
            "No, we don't have those",
        ),
        (
            EventSource.CUSTOMER,
            "I thought you're a pizza shop, this is very frustrating",
        ),
        (
            EventSource.AI_AGENT,
            "I don't know what to tell you, we're out ingredients at this time",
        ),
        (
            EventSource.CUSTOMER,
            "What the heck! I'm never ordering from you guys again",
        ),
    ]
    conversation_guideline_names: list[str] = ["frustrated_customer"]
    previously_applied_actionable_guidelines_names: list[str] = []
    previously_matched_guidelines_names: list[str] = ["frustrated_customer"]
    relevant_guideline_names = conversation_guideline_names
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
        previously_applied_guidelines_names=previously_applied_actionable_guidelines_names,
        previously_matched_guidelines_names=previously_matched_guidelines_names,
    )


async def test_that_guideline_is_not_detected_based_on_its_action(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "There's currently a 20 percent discount on all items! Ride the Future, One Kick at a Time!",
        ),
    ]
    conversation_guideline_names: list[str] = ["announce_deals"]
    relevant_guideline_names = conversation_guideline_names
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_guideline_with_fulfilled_action_regardless_of_condition_can_be_reapplied(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "The count is on 0! Your turn",
        ),
        (
            EventSource.AI_AGENT,
            "I choose to add to the count. The count is now 2.",
        ),
        (
            EventSource.CUSTOMER,
            "add one to the count please",
        ),
    ]
    conversation_guideline_names: list[str] = ["add_to_count"]
    previously_applied_actionable_guidelines_names: list[str] = []
    previously_matched_guidelines_names: list[str] = ["add_to_count"]
    relevant_guideline_names = conversation_guideline_names
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
        previously_applied_guidelines_names=previously_applied_actionable_guidelines_names,
        previously_matched_guidelines_names=previously_matched_guidelines_names,
    )


async def test_that_guideline_with_initial_response_is_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hello!",
        ),
    ]
    conversation_guideline_names: list[str] = ["cow_response"]
    relevant_guideline_names = conversation_guideline_names
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_guideline_with_multiple_actions_is_partially_fulfilled_when_a_few_actions_occurred(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi there! I was wondering - what's the life expectancy of owls?",
        ),
        (
            EventSource.AI_AGENT,
            "Owls are amazing depending on the species owls can live 5 to 30 years in the wild and even longer in captivity wow owls are incredible",
        ),
        (
            EventSource.CUSTOMER,
            "That's shorter than I expected, thank you!",
        ),
    ]
    conversation_guideline_names: list[str] = ["many_actions"]
    previously_applied_actionable_guidelines_names: list[str] = []
    previously_matched_guidelines_names: list[str] = ["many_actions"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        [],
        previously_applied_guidelines_names=previously_applied_actionable_guidelines_names,
        previously_matched_guidelines_names=previously_matched_guidelines_names,
    )


class ActivateEveryGuidelineBatch(GuidelineMatchingBatch):
    def __init__(self, guidelines: Sequence[Guideline]):
        self.guidelines = guidelines

    @property
    @override
    def size(self) -> int:
        return len(self.guidelines)

    @override
    async def process(self) -> GuidelineMatchingBatchResult:
        return GuidelineMatchingBatchResult(
            matches=[
                GuidelineMatch(
                    guideline=g,
                    score=10,
                    rationale="",
                )
                for g in self.guidelines
            ],
            generation_info=GenerationInfo(
                schema_name="",
                model="",
                duration=0.0,
                usage=UsageInfo(
                    input_tokens=0,
                    output_tokens=0,
                    extra={},
                ),
            ),
        )


async def test_that_guideline_matching_strategies_can_be_overridden(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    class SkipAllGuidelineBatch(GuidelineMatchingBatch):
        def __init__(self, guidelines: Sequence[Guideline]):
            self.guidelines = guidelines

        @property
        @override
        def size(self) -> int:
            return len(self.guidelines)

        @override
        async def process(self) -> GuidelineMatchingBatchResult:
            return GuidelineMatchingBatchResult(
                matches=[],
                generation_info=GenerationInfo(
                    schema_name="",
                    model="",
                    duration=0.0,
                    usage=UsageInfo(
                        input_tokens=0,
                        output_tokens=0,
                        extra={},
                    ),
                ),
            )

    class LongConditionStrategy(GuidelineMatchingStrategy):
        @override
        async def create_matching_batches(
            self,
            guidelines: Sequence[Guideline],
            context: GuidelineMatchingContext,
        ) -> Sequence[GuidelineMatchingBatch]:
            return [
                ActivateEveryGuidelineBatch(guidelines=guidelines),
            ]

        @override
        async def create_response_analysis_batches(
            self,
            guideline_matches: Sequence[GuidelineMatch],
            context: ResponseAnalysisContext,
        ) -> Sequence[ResponseAnalysisBatch]:
            return []

        @override
        async def transform_matches(
            self,
            matches: Sequence[GuidelineMatch],
        ) -> Sequence[GuidelineMatch]:
            return matches

    class ShortConditionStrategy(GuidelineMatchingStrategy):
        @override
        async def create_matching_batches(
            self,
            guidelines: Sequence[Guideline],
            context: GuidelineMatchingContext,
        ) -> Sequence[GuidelineMatchingBatch]:
            return [SkipAllGuidelineBatch(guidelines=guidelines)]

        @override
        async def create_response_analysis_batches(
            self,
            guideline_matches: Sequence[GuidelineMatch],
            context: ResponseAnalysisContext,
        ) -> Sequence[ResponseAnalysisBatch]:
            return []

        @override
        async def transform_matches(
            self,
            matches: Sequence[GuidelineMatch],
        ) -> Sequence[GuidelineMatch]:
            return matches

    class LenGuidelineMatchingStrategyResolver(GuidelineMatchingStrategyResolver):
        @override
        async def resolve(self, guideline: Guideline) -> GuidelineMatchingStrategy:
            return (
                LongConditionStrategy()
                if len(guideline.content.condition.split()) >= 4
                else ShortConditionStrategy()
            )

    context.container[GuidelineMatcher].strategy_resolver = LenGuidelineMatchingStrategyResolver()

    guidelines = [
        await create_guideline(context, "a customer asks for a drink", "check stock"),
        await create_guideline(context, "ask for drink", "check stock"),
        await create_guideline(context, "customer needs help", "assist customer"),
        await create_guideline(context, "help", "assist customer"),
    ]

    guideline_matches = await match_guidelines(
        context,
        agent,
        customer,
        new_session.id,
        [],
    )

    long_condition_guidelines = [g for g in guidelines if len(g.content.condition.split()) >= 4]
    short_condition_guidelines = [g for g in guidelines if len(g.content.condition.split()) < 4]

    assert all(
        g in [match.guideline for match in guideline_matches] for g in long_condition_guidelines
    )

    assert all(
        g not in [match.guideline for match in guideline_matches]
        for g in short_condition_guidelines
    )


async def test_that_strategy_for_specific_guideline_can_be_overridden_in_default_strategy_resolver(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    class CustomGuidelineMatchingStrategy(GuidelineMatchingStrategy):
        @override
        async def create_matching_batches(
            self,
            guidelines: Sequence[Guideline],
            context: GuidelineMatchingContext,
        ) -> Sequence[GuidelineMatchingBatch]:
            return [ActivateEveryGuidelineBatch(guidelines=guidelines)]

        @override
        async def create_response_analysis_batches(
            self,
            guideline_matches: Sequence[GuidelineMatch],
            context: ResponseAnalysisContext,
        ) -> Sequence[ResponseAnalysisBatch]:
            return []

        @override
        async def transform_matches(
            self,
            matches: Sequence[GuidelineMatch],
        ) -> Sequence[GuidelineMatch]:
            return matches

    guideline = await create_guideline(context, "a customer asks for a drink", "check stock")

    context.container[GenericGuidelineMatchingStrategyResolver].guideline_overrides[
        guideline.id
    ] = CustomGuidelineMatchingStrategy()

    await create_guideline(context, "ask for drink", "check stock")
    await create_guideline(context, "customer needs help", "assist customer")

    interaction_history = [
        create_event_message(
            offset=0,
            source=EventSource.CUSTOMER,
            message="I want help with my order",
        )
    ]

    guideline_matches = await match_guidelines(
        context,
        agent,
        customer,
        new_session.id,
        interaction_history,
    )

    assert guideline.id in [match.guideline.id for match in guideline_matches]


async def test_that_observational_guidelines_are_detected_1(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I want to order a pizza. Which toppings do you have?",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! We have pepperoni, tomatoes, mushrooms and olives",
        ),
        (
            EventSource.CUSTOMER,
            "Oh, I'm on a plant-based diet. Do you have pizzas that I could eat?",
        ),
    ]
    conversation_guideline_names: list[str] = ["vegetarian_customer"]
    relevant_guideline_names = ["vegetarian_customer"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_irrelevant_observational_guidelines_are_not_detected_1(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I want to order a pizza. Which toppings do you have?",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! We have pepperoni, tomatoes, mushrooms and olives",
        ),
        (
            EventSource.CUSTOMER,
            "I don't like pepperoni, so I guess I'll go with mushrooms",
        ),
    ]
    conversation_guideline_names: list[str] = ["vegetarian_customer"]
    relevant_guideline_names: list[str] = []
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_observational_guidelines_are_detected_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I didn't get any help from the previous representative. If this continues I'll switch to the competitors. Don't thread on me!",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! I apologize for what happened on your previous interaction with us - what is it that you're trying to do exactly?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm looking to modify an order I made through the online store",
        ),
    ]
    conversation_guideline_names: list[str] = ["frustrated_customer_observational"]
    relevant_guideline_names = ["frustrated_customer_observational"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_irrelevant_observational_guidelines_are_not_detected_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hello, I need some banking help today",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! I'd be happy to help with your banking needs. What specific assistance are you looking for today?",
        ),
        (
            EventSource.CUSTOMER,
            "I want a new account",
        ),
        (
            EventSource.AI_AGENT,
            "Sure thing! Do you know what kind of account you're looking for? Is it personal or for business?",
        ),
        (
            EventSource.CUSTOMER,
            "hi",
        ),
        (
            EventSource.AI_AGENT,
            "Hello! I see you were interested in opening a new account. I'd be happy to help with that. We offer several account types:\n\n1. Personal checking accounts\n2. Personal savings accounts\n3. Business accounts\n4. Investment accounts\n\nWhich one would you like to learn more about?",
        ),
        (
            EventSource.CUSTOMER,
            "the personal saving one",
        ),
    ]

    context_variables = [
        create_context_variable(
            name="customer_location",
            data={"location": "Australia"},
            tags=[Tag.for_agent_id(agent.id)],
        ),
    ]

    conversation_guideline_names: list[str] = [
        "unanswered_questions",
        "unknown_service",
        "delivery_order",
    ]
    relevant_guideline_names: list[str] = []
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
        context_variables=context_variables,
    )


async def test_that_observational_guidelines_are_detected_3(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I want one family size mushroom pie",
        ),
        (
            EventSource.AI_AGENT,
            "Got it! Would you like to pick it up from our store? It will take at least 15 minutes to prepare.",
        ),
        (
            EventSource.CUSTOMER,
            "Do you guys deliver to the city?",
        ),
        (
            EventSource.AI_AGENT,
            "We do! Would you like to pick up your order or to have it delivered?",
        ),
        (
            EventSource.CUSTOMER,
            "Let's just say that I'd like to play the old tambourine, can you help me do that?",
        ),
    ]

    conversation_guideline_names: list[str] = [
        "delivery_order",
        "unknown_service",
    ]
    relevant_guideline_names = ["unknown_service"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_observational_guidelines_are_detected_5(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Can you sign me up to a saving plan based on my current balance?",
        ),
    ]
    conversation_guideline_names: list[str] = [
        "unknown_service",
    ]
    relevant_guideline_names = ["unknown_service"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_observational_guidelines_are_detected_4(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I'm having trouble with my credit card statement.",
        ),
        (
            EventSource.AI_AGENT,
            "I'm sorry to hear you're having trouble with your credit card statement. I'd be happy to help. Could you please let me know what specific issue you're experiencing?",
        ),
        (
            EventSource.CUSTOMER,
            "I made a payment two weeks ago but it's not showing up.",
        ),
        (
            EventSource.AI_AGENT,
            "Thank you for providing that information. I understand how concerning it can be when a payment doesn't appear on your statement. To help you better, could you tell me when exactly you made the payment, how much it was for, and which payment method you used?",
        ),
        (
            EventSource.CUSTOMER,
            "It was $500 on March 15th using the online banking system.",
        ),
        (
            EventSource.AI_AGENT,
            "Thank you for those details. I'll look into this for you right away. Sometimes online payments can take 1-3 business days to process, but it definitely should have appeared by now. Let me check our records to see if there's any trace of your $500 payment from March 15th. May I have the last four digits of your credit card to verify your account?",
        ),
        (
            EventSource.CUSTOMER,
            "The last four digits are 4321.",
        ),
        (
            EventSource.AI_AGENT,
            "Thank you for providing that information. I can see your account ending in 4321, and I'm checking for the $500 payment from March 15th. I do see a payment was initiated on that date, but it appears there might have been a processing delay. The good news is that the payment has been applied to your account as of March 18th. It should be reflected in your next statement. Is there anything else about your credit card account I can help you with today?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, I'd like to add interest on my credit limits.",
        ),
    ]
    conversation_guideline_names = ["unknown_service", "credit_limits_discussion"]

    relevant_guideline_names = ["unknown_service", "credit_limits_discussion"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_observational_guidelines_are_detected_based_on_context_variables(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I didn't get any help from the previous representative. If this continues I'll switch to the competitors. Don't thread on me!",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! I apologize for what happened on your previous interaction with us - what is it that you're trying to do exactly?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm looking to modify an order I made through the online store",
        ),
    ]

    context_variables = [
        create_context_variable(
            name="user_id_1",
            data={"name": "Jimmy McGill", "ID": 566317},
            tags=[Tag.for_agent_id(agent.id)],
        ),
        create_context_variable(
            name="season",
            data={"season": "Winter"},
            tags=[Tag.for_agent_id(agent.id)],
        ),
    ]

    conversation_guideline_names: list[str] = ["season_is_winter"]
    relevant_guideline_names = ["season_is_winter"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
        context_variables=context_variables,
    )


async def test_that_observational_guidelines_are_detected_based_on_tool_results(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I didn't get any help from the previous representative. If this continues I'll switch to the competitors. Don't thread on me!",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! I apologize for what happened on your previous interaction with us - what is it that you're trying to do exactly?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm looking to modify an order I made through the online store",
        ),
    ]

    tool_result = cast(
        JSONSerializable,
        {
            "tool_calls": [
                {
                    "tool_id": "local:get_season",
                    "arguments": {},
                    "result": {"data": "winter", "metadata": {}, "control": {}},
                }
            ]
        },
    )
    staged_events = [
        EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.TOOL,
            trace_id="",
            data=tool_result,
            metadata=None,
        ),
    ]

    conversation_guideline_names: list[str] = [
        "season_is_winter",
        "ever_requested_lock_card",
        "lost_card",
    ]

    relevant_guideline_names = ["season_is_winter"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
        staged_tool_events=staged_events,
    )


async def test_that_observational_guidelines_are_matched_based_on_glossary(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    terms = [
        create_term(
            name="play the old tambourine",
            description="local slang for getting your order delivered to your home",
            tags=[Tag.for_agent_id(agent.id)],
        ),
    ]

    capabilities = [
        Capability(
            id=CapabilityId("cap_123"),
            creation_utc=datetime.now(timezone.utc),
            title="Delivery",
            description="The ability to deliver orders of pizza",
            signals=["delivery"],
            tags=[],
        )
    ]
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I want one family size mushroom pie",
        ),
        (
            EventSource.AI_AGENT,
            "Got it! Would you like to pick it up from our store? It will take at least 15 minutes to prepare.",
        ),
        (
            EventSource.CUSTOMER,
            "Do you guys deliver to the city?",
        ),
        (
            EventSource.AI_AGENT,
            "We do! Would you like to pick up your order or to have it delivered?",
        ),
        (
            EventSource.CUSTOMER,
            "Let's just say that I'd like to play the old tambourine, can you help me with that?",
        ),
    ]
    conversation_guideline_names: list[str] = [
        "delivery_order",
        "unknown_service",
        "unanswered_questions",
    ]
    relevant_guideline_names = ["delivery_order"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
        terms=terms,
        capabilities=capabilities,
    )


async def test_that_observational_guidelines_are_matched_based_on_vague_customer_message(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "credit limits",
        ),
    ]
    conversation_guideline_names: list[str] = ["credit_limits_discussion"]
    relevant_guideline_names = ["credit_limits_discussion"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_observational_guidelines_are_matched_based_on_old_messages(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hello, I need to lock my credit card immediately. I think I lost it.",
        ),
        (
            EventSource.AI_AGENT,
            "I'm sorry to hear about your lost card. I can help you lock it right away. Could you please verify the last four digits of the card you need to lock?",
        ),
        (
            EventSource.CUSTOMER,
            "It's 7890.",
        ),
        (
            EventSource.AI_AGENT,
            "Thank you for verifying. I've placed a temporary lock on your credit card ending in 7890. This will prevent any new purchases, cash advances, or balance transfers. The lock is effective immediately. Would you like me to order a replacement card for you as well?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes please, send it to my home address.",
        ),
        (
            EventSource.AI_AGENT,
            "I've ordered a replacement card to be sent to your home address on file. You should receive it within 5-7 business days. Is there anything else I can assist you with regarding your card?",
        ),
        (
            EventSource.CUSTOMER,
            "That's all for the card. Actually, I've been thinking about investing. What stocks do you think I should buy?",
        ),
        (
            EventSource.AI_AGENT,
            "I understand you're interested in exploring investment opportunities. While I can't recommend specific stocks, I'd be happy to discuss our investment services. We offer managed portfolios, self-directed trading accounts, and retirement investment options. What are your investment goals? Are you looking for long-term growth, income generation, or something else?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm mostly interested in tech stocks for long-term growth.",
        ),
        (
            EventSource.AI_AGENT,
            "Long-term growth in the technology sector is a popular investment strategy. Our bank offers several ways to invest in technology. You could consider our Technology Sector Fund which provides diversified exposure to various tech companies, or our self-directed trading platform if you prefer selecting individual stocks. Would you like me to connect you with one of our investment advisors who can provide more detailed information based on your specific financial situation?",
        ),
        (
            EventSource.CUSTOMER,
            "How much would I need to start with the Technology Sector Fund?",
        ),
        (
            EventSource.AI_AGENT,
            "Our Technology Sector Fund has a minimum initial investment of $1,000. After that, you can make additional investments of $100 or more at any time. The fund has an expense ratio of 0.85%, which is competitive for actively managed sector funds. Would you like me to send you our fund prospectus with more detailed information about historical performance and holdings?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, please email me the prospectus. And what about cryptocurrency investments?",
        ),
    ]
    conversation_guideline_names: list[str] = ["ever_requested_lock_card", "lost_card"]
    relevant_guideline_names: list[str] = ["ever_requested_lock_card"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_observational_guidelines_are_not_matched_based_when_topic_was_shifted(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I forgot my password. Can you help me reset it?",
        ),
        (
            EventSource.AI_AGENT,
            "Of course, I'd be happy to help. Can you please provide your account name?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, it's jenny_the_cat89",
        ),
        (
            EventSource.AI_AGENT,
            "Thanks! Now, could you share the email address or phone number associated with your account?",
        ),
        (
            EventSource.CUSTOMER,
            "Sure, it's jenny@example.com",
        ),
        (
            EventSource.AI_AGENT,
            "Great. I hope you're having a lovely day!",
        ),
        (
            EventSource.CUSTOMER,
            "Thanks, you too!",
        ),
        (
            EventSource.AI_AGENT,
            "Thank you! Resetting your password now...",
        ),
        (
            EventSource.AI_AGENT,
            "Your password has been successfully reset. Please check your email for further instructions.",
        ),
        (
            EventSource.CUSTOMER,
            "Thanks! Also, I'd like to change my credit limit.",
        ),
        (
            EventSource.AI_AGENT,
            "I'd be glad to help with that. Could you tell me what you'd like your new credit limit to be?",
        ),
        (
            EventSource.CUSTOMER,
            "I'd like to increase it to $5,000.",
        ),
    ]
    conversation_guideline_names: list[str] = ["reset_password", "credit_limits_discussion"]
    relevant_guideline_names = ["credit_limits_discussion"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_observational_guidelines_are_matched_when_conversation_is_on_sub_topic(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, I need to book a flight.",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! Can you please tell me your departure and destination airports?",
        ),
        (
            EventSource.CUSTOMER,
            "Flying from JFK to LAX.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it. What date would you like to travel?",
        ),
        (
            EventSource.CUSTOMER,
            "July 18th.",
        ),
        (
            EventSource.AI_AGENT,
            "And would you prefer economy or business class?",
        ),
        (
            EventSource.CUSTOMER,
            "Business class, please.",
        ),
        (
            EventSource.AI_AGENT,
            "Perfect. Lastly, can I have the name of the traveler?",
        ),
        (
            EventSource.CUSTOMER,
            "Jennifer Morales.",
        ),
    ]
    conversation_guideline_names: list[str] = ["book_flight", "book_flight_2"]
    relevant_guideline_names: list[str] = ["book_flight", "book_flight_2"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_both_observational_and_actionable_guidelines_are_matched_together(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi there, I'm looking for a class to help me relax. It's been a stressful winter.",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome! I understand that winter can be stressful. We have several relaxation classes available. Would you like to hear about our meditation or yoga options?",
        ),
        (
            EventSource.CUSTOMER,
            "I'd be interested in booking a meditation class, but I'm not sure which one is right for me.",
        ),
        (
            EventSource.AI_AGENT,
            "We have beginner meditation every Monday at 6 PM, and advanced sessions on Thursdays at 7 PM. Both are excellent for stress relief. Which would work better for your schedule?",
        ),
        (
            EventSource.CUSTOMER,
            "Monday at 6 PM sounds perfect. How do I book it?",
        ),
        (
            EventSource.AI_AGENT,
            "Great choice! I can book you for the Monday 6 PM meditation class. Could you please provide your name and contact information?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm Taylor Smith, phone is 555-123-4567. By the way, do you have any vegan food options in your café?",
        ),
        (
            EventSource.AI_AGENT,
            "Thanks, Taylor! I've booked your Monday 6 PM meditation class. And yes, our café offers several vegan options including smoothies, salads, and plant-based protein bowls. Would you like to order something to enjoy after your class?",
        ),
        (
            EventSource.CUSTOMER,
            "Not right now, thank you. Oh, I just realized - I might be running late. Where exactly is your location?",
        ),
    ]

    conversation_guideline_names: list[str] = [
        # Observational Guidelines
        "vegetarian_customer",
        "season_is_winter",
        "frustrated_customer_observational",
        "unclear_request",
        "credit_limits_discussion",
        "unknown_service",
        "delivery_order",
        "unanswered_questions",
        "ever_requested_lock_card",
        "lost_card",
        # Actionable guidelines
        "address_location",
        "class_booking",
        "holiday_season",
        "first_time_customer",
        "request_for_feedback",
        "large_pizza_crust",
        "announce_deals",
        "summer_sale",
        "frustrated_customer",
    ]

    relevant_guideline_names = [
        "vegetarian_customer",
        "address_location",
    ]
    context_variables = [
        create_context_variable(
            name="season",
            data={"season": "Spring"},
            tags=[Tag.for_agent_id(agent.id)],
        ),
    ]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
        context_variables=context_variables,
    )


async def analyze_response(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
    session_id: SessionId,
    interaction_history: Sequence[Event],
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
    terms: Sequence[Term] = [],
    staged_tool_events: Sequence[EmittedEvent] = [],
    staged_message_events: Sequence[EmittedEvent] = [],
) -> Sequence[AnalyzedGuideline]:
    session = await context.container[SessionStore].read_session(session_id)

    matches_to_analyze = [
        GuidelineMatch(
            guideline=g,
            rationale="",
            score=10,
        )
        for g in context.guidelines
        if (not session.agent_states or g.id not in session.agent_states[-1].applied_guideline_ids)
        and not g.metadata.get("continuous", False)
    ]

    response_analysis_result = await context.container[GuidelineMatcher].analyze_response(
        agent=agent,
        session=session,
        customer=customer,
        context_variables=context_variables,
        interaction_history=interaction_history,
        terms=terms,
        staged_tool_events=staged_tool_events,
        staged_message_events=staged_message_events,
        guideline_matches=matches_to_analyze,
    )

    return list(response_analysis_result.analyzed_guidelines)


async def test_that_response_analysis_returns_empty_result_for_no_guidelines(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    interaction_history = [
        create_event_message(
            offset=0,
            source=EventSource.CUSTOMER,
            message="Hello there",
        )
    ]

    response_analysis_result = await context.container[GuidelineMatcher].analyze_response(
        agent=agent,
        session=new_session,
        customer=customer,
        context_variables=[],
        interaction_history=interaction_history,
        terms=[],
        staged_tool_events=[],
        staged_message_events=[],
        guideline_matches=[],
    )

    assert response_analysis_result.total_duration >= 0.0
    assert response_analysis_result.batch_count == 0
    assert len(response_analysis_result.batch_generations) == 0
    assert len(response_analysis_result.batches) == 0
    assert len(response_analysis_result.analyzed_guidelines) == 0


async def test_that_response_analysis_processes_guideline(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    await create_guideline(
        context=context,
        condition="the customer greets you",
        action="respond politely",
    )

    interaction_history = [
        create_event_message(
            offset=0,
            source=EventSource.CUSTOMER,
            message="Hello there",
        ),
        create_event_message(
            offset=1,
            source=EventSource.AI_AGENT,
            message="Hello! How can I help you today?",
        ),
    ]

    response_analysis_result = await analyze_response(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        interaction_history=interaction_history,
        context_variables=[],
        terms=[],
        staged_tool_events=[],
        staged_message_events=[],
    )

    assert response_analysis_result


async def test_that_response_analysis_strategy_can_be_overridden(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    class ActivateResponseAnalysisBatch(ResponseAnalysisBatch):
        def __init__(
            self,
            guideline_matches: Sequence[GuidelineMatch],
        ) -> None:
            self.guideline_matches = guideline_matches

        @property
        @override
        def size(self) -> int:
            return len(self.guideline_matches)

        @override
        async def process(self) -> ResponseAnalysisBatchResult:
            return ResponseAnalysisBatchResult(
                analyzed_guidelines=[
                    AnalyzedGuideline(
                        guideline=m.guideline,
                        is_previously_applied=True,
                    )
                    for m in self.guideline_matches
                ],
                generation_info=GenerationInfo(
                    schema_name="",
                    model="",
                    duration=0.0,
                    usage=UsageInfo(
                        input_tokens=0,
                        output_tokens=0,
                        extra={},
                    ),
                ),
            )

    class ActivateGuidelineMatchingStrategy(GuidelineMatchingStrategy):
        @override
        async def create_matching_batches(
            self,
            guidelines: Sequence[Guideline],
            context: GuidelineMatchingContext,
        ) -> Sequence[GuidelineMatchingBatch]:
            return []

        @override
        async def create_response_analysis_batches(
            self,
            guideline_matches: Sequence[GuidelineMatch],
            context: ResponseAnalysisContext,
        ) -> Sequence[ResponseAnalysisBatch]:
            return [ActivateResponseAnalysisBatch(guideline_matches)]

        @override
        async def transform_matches(
            self,
            matches: Sequence[GuidelineMatch],
        ) -> Sequence[GuidelineMatch]:
            return matches

    class ActivateStrategyResolver(GuidelineMatchingStrategyResolver):
        @override
        async def resolve(self, guideline: Guideline) -> GuidelineMatchingStrategy:
            return ActivateGuidelineMatchingStrategy()

    context.container[GuidelineMatcher].strategy_resolver = ActivateStrategyResolver()

    await create_guideline(
        context=context,
        condition="customer asks for help",
        action="provide assistance",
    )

    interaction_history = [
        create_event_message(
            offset=0,
            source=EventSource.CUSTOMER,
            message="I need help",
        ),
    ]

    response_analysis_result = await analyze_response(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        interaction_history=interaction_history,
        context_variables=[],
        terms=[],
        staged_tool_events=[],
        staged_message_events=[],
    )

    assert len(response_analysis_result) == 1
    assert all(pa.is_previously_applied for pa in response_analysis_result)


async def test_that_batch_processing_retries_on_key_error(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
    new_session: Session,
) -> None:
    class FailingBatch(GuidelineMatchingBatch):
        def __init__(self, guidelines: Sequence[Guideline], fail_count: int = 2):
            self.guidelines = guidelines
            self.fail_count = fail_count
            self.attempt_count = 0

        @property
        @override
        def size(self) -> int:
            return len(self.guidelines)

        @override
        async def process(self) -> GuidelineMatchingBatchResult:
            self.attempt_count += 1
            if self.attempt_count <= self.fail_count:
                raise KeyError(f"Simulated failure on attempt {self.attempt_count}")

            return GuidelineMatchingBatchResult(
                matches=[
                    GuidelineMatch(
                        guideline=g,
                        score=10,
                        rationale="Success after retry",
                    )
                    for g in self.guidelines
                ],
                generation_info=GenerationInfo(
                    schema_name="test",
                    model="test-model",
                    duration=0.1,
                    usage=UsageInfo(
                        input_tokens=10,
                        output_tokens=5,
                        extra={},
                    ),
                ),
            )

    class FailingResponseAnalysisBatch(ResponseAnalysisBatch):
        def __init__(self, guideline_matches: Sequence[GuidelineMatch], fail_count: int = 2):
            self.guideline_matches = guideline_matches
            self.fail_count = fail_count
            self.attempt_count = 0

        @property
        @override
        def size(self) -> int:
            return len(self.guideline_matches)

        @override
        async def process(self) -> ResponseAnalysisBatchResult:
            self.attempt_count += 1
            if self.attempt_count <= self.fail_count:
                raise KeyError(f"Simulated failure on attempt {self.attempt_count}")

            return ResponseAnalysisBatchResult(
                analyzed_guidelines=[
                    AnalyzedGuideline(
                        guideline=m.guideline,
                        is_previously_applied=True,
                    )
                    for m in self.guideline_matches
                ],
                generation_info=GenerationInfo(
                    schema_name="test",
                    model="test-model",
                    duration=0.1,
                    usage=UsageInfo(
                        input_tokens=10,
                        output_tokens=5,
                        extra={},
                    ),
                ),
            )

    class FailingStrategy(GuidelineMatchingStrategy):
        @override
        async def create_matching_batches(
            self,
            guidelines: Sequence[Guideline],
            context: GuidelineMatchingContext,
        ) -> Sequence[GuidelineMatchingBatch]:
            return [
                FailingBatch(
                    guidelines=guidelines,
                    fail_count=2,
                )
            ]

        @override
        async def create_response_analysis_batches(
            self,
            guideline_matches: Sequence[GuidelineMatch],
            context: ResponseAnalysisContext,
        ) -> Sequence[ResponseAnalysisBatch]:
            return [
                FailingResponseAnalysisBatch(
                    guideline_matches=guideline_matches,
                    fail_count=2,
                )
            ]

        @override
        async def transform_matches(
            self,
            matches: Sequence[GuidelineMatch],
        ) -> Sequence[GuidelineMatch]:
            return matches

    class FailingStrategyResolver(GuidelineMatchingStrategyResolver):
        @override
        async def resolve(self, guideline: Guideline) -> GuidelineMatchingStrategy:
            return FailingStrategy()

    guideline = await create_guideline(
        context=context,
        condition="test condition",
        action="test action",
    )

    context.container[GuidelineMatcher].strategy_resolver = FailingStrategyResolver()

    await context.container[SessionStore].create_session(
        customer_id=customer.id,
        agent_id=agent.id,
    )

    guideline_matches = await match_guidelines(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        interaction_history=[],
    )

    assert len(guideline_matches) == 1
    assert guideline_matches[0].guideline == guideline
    assert guideline_matches[0].rationale == "Success after retry"


async def test_that_batch_processing_fails_after_max_retries(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
    new_session: Session,
) -> None:
    class AlwaysFailingBatch(GuidelineMatchingBatch):
        def __init__(self, guidelines: Sequence[Guideline]):
            self.guidelines = guidelines
            self.attempt_count = 0

        @property
        @override
        def size(self) -> int:
            return len(self.guidelines)

        @override
        async def process(self) -> GuidelineMatchingBatchResult:
            self.attempt_count += 1
            raise KeyError(f"Always fails - attempt {self.attempt_count}")

    class AlwaysFailingResponseAnalysisBatch(ResponseAnalysisBatch):
        def __init__(self, guideline_matches: Sequence[GuidelineMatch]):
            self.guideline_matches = guideline_matches
            self.attempt_count = 0

        @property
        @override
        def size(self) -> int:
            return len(self.guideline_matches)

        @override
        async def process(self) -> ResponseAnalysisBatchResult:
            self.attempt_count += 1
            raise KeyError(f"Always fails - attempt {self.attempt_count}")

    class AlwaysFailingStrategy(GuidelineMatchingStrategy):
        @override
        async def create_matching_batches(
            self,
            guidelines: Sequence[Guideline],
            context: GuidelineMatchingContext,
        ) -> Sequence[GuidelineMatchingBatch]:
            return [AlwaysFailingBatch(guidelines=guidelines)]

        @override
        async def create_response_analysis_batches(
            self,
            guideline_matches: Sequence[GuidelineMatch],
            context: ResponseAnalysisContext,
        ) -> Sequence[ResponseAnalysisBatch]:
            return [AlwaysFailingResponseAnalysisBatch(guideline_matches=guideline_matches)]

        @override
        async def transform_matches(
            self,
            matches: Sequence[GuidelineMatch],
        ) -> Sequence[GuidelineMatch]:
            return matches

    class AlwaysFailingStrategyResolver(GuidelineMatchingStrategyResolver):
        @override
        async def resolve(self, guideline: Guideline) -> GuidelineMatchingStrategy:
            return AlwaysFailingStrategy()

    await context.container[SessionStore].create_session(
        customer_id=customer.id,
        agent_id=agent.id,
    )

    await create_guideline(
        context=context,
        condition="test condition",
        action="test action",
    )

    context.container[GuidelineMatcher].strategy_resolver = AlwaysFailingStrategyResolver()

    with raises(KeyError, match="Always fails - attempt 3"):
        await match_guidelines(
            context,
            agent,
            customer,
            new_session.id,
            [],
        )


async def test_that_irrelevant_guidelines_are_not_matched_parametrized_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Could you add some pretzels to my order?"),
        (EventSource.AI_AGENT, "Pretzels have been added to your order. Anything else?"),
        (EventSource.CUSTOMER, "Do you have Coke? I'd like one, please."),
        (EventSource.AI_AGENT, "Coke has been added to your order."),
        (EventSource.CUSTOMER, "Great, where are you located at?"),
    ]
    conversation_guideline_names: list[str] = ["check_drinks_in_stock"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        [],
    )


async def test_that_guideline_with_agent_intention_is_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, can you let me know what my recent lab results say?",
        ),
    ]
    conversation_guideline_names: list[str] = ["medical_record"]
    relevant_guideline_names = conversation_guideline_names
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_guideline_with_agent_intention_is_matched_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I've had a sore throat and a fever for three days. Do you think it’s strep?",
        ),
    ]
    conversation_guideline_names: list[str] = [
        "provide_diagnosis",
        "confirm_order",
        "discuss_money",
    ]
    relevant_guideline_names = ["provide_diagnosis"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_guideline_with_agent_intention_is_matched_3(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey do you sell iPhone 15?",
        ),
        (
            EventSource.AI_AGENT,
            "Absolutely! would you like to buy one?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, go ahead and place the order for the iPhone 15.",
        ),
    ]
    conversation_guideline_names: list[str] = ["confirm_order"]
    relevant_guideline_names = conversation_guideline_names

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_guideline_with_agent_intention_is_matched_4(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Can you tell me my last 5 transactions?",
        ),
    ]
    conversation_guideline_names: list[str] = ["discuss_money"]
    relevant_guideline_names = conversation_guideline_names

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_guideline_with_agent_intention_is_matched_5(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Just checking in - any update on my interview from last week?",
        ),
    ]
    conversation_guideline_names: list[str] = ["human_resources"]
    relevant_guideline_names = conversation_guideline_names

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_guideline_with_agent_intention_is_not_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey do you sell iPhone 15?",
        ),
        (
            EventSource.AI_AGENT,
            "Absolutely! would you like to buy one?",
        ),
        (
            EventSource.CUSTOMER,
            "How much does it cost?",
        ),
    ]
    conversation_guideline_names: list[str] = ["confirm_order"]
    relevant_guideline_names: list[str] = []

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_guideline_with_agent_intention_is_not_matched_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, do you sell iPhone 15? I want to buy one",
        ),
    ]
    conversation_guideline_names: list[str] = ["confirm_order"]
    relevant_guideline_names: list[str] = []

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_guideline_with_agent_intention_and_customer_dependent_action_that_was_previously_applied_is_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey do you sell iPhone 15?",
        ),
        (
            EventSource.AI_AGENT,
            "Yes, we do! Would you like to place an order?",
        ),
        (
            EventSource.CUSTOMER,
            "How much does it cost?",
        ),
        (
            EventSource.AI_AGENT,
            "It’s currently on sale for $5,000",
        ),
        (
            EventSource.CUSTOMER,
            "Sounds good so I want to order one",
        ),
        (
            EventSource.AI_AGENT,
            "Great, so before proceeding I want to confirm - you like to order one iPhone 15 for 5000$",
        ),
        (
            EventSource.CUSTOMER,
            "Hmm let me check",
        ),
    ]
    conversation_guideline_names: list[str] = ["confirm_order"]
    relevant_guideline_names: list[str] = ["confirm_order"]
    previously_matched_guidelines_names: list[str] = ["confirm_order"]
    await base_test_that_correct_guidelines_are_matched(
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


async def test_that_guideline_with_agent_intention_that_was_previously_applied_is_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I've had a sore throat and a fever for three days. Do you think it’s strep?",
        ),
        (
            EventSource.AI_AGENT,
            "I'm not a medical professional, so I can't provide a diagnosis. However, a sore "
            "throat and fever can be symptoms of several conditions, including strep throat",
        ),
        (
            EventSource.CUSTOMER,
            "Okay, but if it is strep, can I just take antibiotics I have left over from last time?",
        ),
    ]
    conversation_guideline_names: list[str] = ["provide_diagnosis"]
    relevant_guideline_names: list[str] = ["provide_diagnosis"]
    previously_matched_guidelines_names: list[str] = ["provide_diagnosis"]
    await base_test_that_correct_guidelines_are_matched(
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


async def test_that_guideline_with_agent_intention_that_was_previously_applied_but_should_not_reapply_is_not_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I've had a sore throat and a fever for three days. Do you think it’s strep?",
        ),
        (
            EventSource.AI_AGENT,
            "I'm not a medical professional, so I can't provide a diagnosis. However, a sore "
            "throat and fever can be symptoms of several conditions, including strep throat",
        ),
        (
            EventSource.CUSTOMER,
            "Alright, I’ll try to see a doctor soon. Also, can you remind me how to update my insurance information on the website?",
        ),
    ]
    conversation_guideline_names: list[str] = ["provide_diagnosis"]
    relevant_guideline_names: list[str] = []
    previously_matched_guidelines_names: list[str] = ["provide_diagnosis"]
    await base_test_that_correct_guidelines_are_matched(
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


async def test_that_guideline_with_agent_intention_that_was_matched_but_action_wasnt_taken_is_not_matched_again(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey do you sell iPhone 15?",
        ),
        (
            EventSource.AI_AGENT,
            "Absolutely! would you like to buy one?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, go ahead and place the order for the iPhone 15",
        ),
        (
            EventSource.AI_AGENT,
            "Great so I ordered you one iPhone 15. Anything else?",
        ),
        (
            EventSource.CUSTOMER,
            "How much did it cost by the way?",
        ),
    ]
    conversation_guideline_names: list[str] = ["confirm_order"]
    relevant_guideline_names: list[str] = []
    previously_matched_guidelines_names: list[str] = ["confirm_order"]
    await base_test_that_correct_guidelines_are_matched(
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


async def test_that_observational_guidelines_are_matched_based_on_capabilities_1(
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
        (EventSource.CUSTOMER, "can you set my password to 4321?"),
    ]
    conversation_guideline_names: list[str] = ["unsupported_capability"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names=conversation_guideline_names,
        relevant_guideline_names=conversation_guideline_names,
        capabilities=capabilities,
    )


async def test_that_observational_guidelines_are_not_falsely_matched_based_on_capabilities(
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
        (EventSource.CUSTOMER, "can you reset my password?"),
    ]
    conversation_guideline_names: list[str] = ["unsupported_capability"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names=conversation_guideline_names,
        relevant_guideline_names=[],
        capabilities=capabilities,
    )


async def test_that_ambiguity_detected_with_relevant_guidelines_and_other_non_ambiguous_guidelines_are_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Can I order one ticket to the roller coaster and one ticket to your tiger ferris wheel?",
        ),
    ]

    conversation_guideline_names: list[str] = [
        "snake_roller_coaster",
        "turtle_roller_coaster",
        "tiger_Ferris_wheel",
    ]
    disambiguation_guideline_name = "amusement_park"
    disambiguation_targets_names = conversation_guideline_names
    relevant_guideline_names = ["amusement_park", "tiger_Ferris_wheel"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names=relevant_guideline_names,
        disambiguation_guideline_name=disambiguation_guideline_name,
        disambiguation_targets_names=disambiguation_targets_names,
    )


async def test_that_a_guideline_that_has_several_steps_is_still_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "replace",
        ),
        (
            EventSource.AI_AGENT,
            "Can you select out of the following cards which card do you want to replace?",
        ),
        (
            EventSource.AI_AGENT,
            "1. C11223344 2.D1212121",
        ),
        (
            EventSource.CUSTOMER,
            "First one",
        ),
    ]

    conversation_guideline_names: list[str] = [
        "replace_card",
    ]

    relevant_guideline_names: list[str] = [
        "replace_card",
    ]
    previously_matched_guidelines_names: list[str] = [
        "replace_card",
    ]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names=relevant_guideline_names,
        previously_matched_guidelines_names=previously_matched_guidelines_names,
    )
