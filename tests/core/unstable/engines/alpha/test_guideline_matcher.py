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

from lagom import Container
from pytest import fixture

from parlant.core.agents import Agent
from parlant.core.capabilities import Capability
from parlant.core.common import Criticality, generate_id, JSONSerializable
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableId,
    ContextVariableValue,
    ContextVariableValueId,
)
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer
from parlant.core.customers import Customer
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.generic.response_analysis_batch import (
    GenericResponseAnalysisBatch,
    GenericResponseAnalysisSchema,
)
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
from parlant.core.glossary import Term
from parlant.core.journeys import Journey
from parlant.core.nlp.generation import SchematicGenerator

from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
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
    "lock_card_request_1": {
        "condition": "the customer indicated that they wish to lock their credit card",
        "observation": "-",
    },
    "lock_card_request_2": {
        "condition": "the customer lost their credit card",
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
        "condition": "The customer is asking for a service you have no information about within this prompt",
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
}

ACTIONABLE_GUIDELINES_DICT = {
    "check_drinks_in_stock": {
        "condition": "a customer asks for a drink",
        "action": "check if the drink is available in the following stock: "
        "['Sprite', 'Coke', 'Fanta']",
    },
    "check_toppings_in_stock": {
        "condition": "a customer asks for toppings",
        "action": "check if the toppings are available in the following stock: "
        "['Pepperoni', 'Tomatoes', 'Olives']",
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
        "action": "Provide comforting responses and suggest alternatives "
        "or support to alleviate the customer's mood.",
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
    journeys: Sequence[Journey] = [],
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
        active_journeys=journeys,
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
        criticality=Criticality.MEDIUM,
        enabled=True,
        tags=tags,
        metadata=metadata,
    )

    context.guidelines.append(guideline)

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
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
    terms: Sequence[Term] = [],
    staged_events: Sequence[EmittedEvent] = [],
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
        staged_tool_events=[e for e in staged_events if e.kind == EventKind.TOOL],
        staged_message_events=[e for e in staged_events if e.kind == EventKind.MESSAGE],
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
        staged_events=staged_events,
    )

    matched_guidelines = [p.guideline for p in guideline_matches]

    assert set(matched_guidelines) == set(relevant_guidelines)


async def test_that_many_guidelines_are_classified_correctly(  # a stress test
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
        (EventSource.CUSTOMER, "I'll go with the blue one."),
        (
            EventSource.AI_AGENT,
            "Great choice! I'll add the blue 'City Cruiser' to your cart. Would you like to add any accessories like a helmet or grip tape?",
        ),
        (EventSource.CUSTOMER, "Yes, I'll take a helmet. What do you have in stock?"),
        (
            EventSource.AI_AGENT,
            "We have helmets in small, medium, and large sizes, all available in black and gray. What size do you need?",
        ),
        (EventSource.CUSTOMER, "I need a medium. I'll take one in black."),
        (
            EventSource.AI_AGENT,
            "Got it! Your blue 'City Cruiser' skateboard and black medium helmet are ready for checkout. How would you like to pay?",
        ),
        (EventSource.CUSTOMER, "I'll pay with a credit card, thank you very much!"),
        (
            EventSource.AI_AGENT,
            "Thank you for your order! Your skateboard and helmet will be shipped shortly. Enjoy your ride!",
        ),
        (EventSource.CUSTOMER, "That's great! Thanks!"),
    ]

    exceptions = [
        "credit_payment1",
        "credit_payment2",
        "cow_response",
        "thankful_customer",
        "payment_process",
    ]

    conversation_guideline_names: list[str] = [
        guideline_name
        for guideline_name in ACTIONABLE_GUIDELINES_DICT.keys()
        if guideline_name not in exceptions
    ]
    relevant_guideline_names = ["announce_shipment", "second_thanks"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_relevant_guidelines_are_matched_parametrized_1(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "I'd like to order a pizza, please."),
        (EventSource.AI_AGENT, "No problem. What would you like to have?"),
        (EventSource.CUSTOMER, "I'd like a large pizza. What toppings do you have?"),
        (EventSource.AI_AGENT, "Today, we have pepperoni, tomatoes, and olives available."),
        (EventSource.CUSTOMER, "I'll take pepperoni, thanks."),
        (
            EventSource.AI_AGENT,
            "Awesome. I've added a large pepperoni pizza. Would you like a drink on the side?",
        ),
        (EventSource.CUSTOMER, "Sure. What types of drinks do you have?"),
        (EventSource.AI_AGENT, "We have Sprite, Coke, and Fanta."),
        (EventSource.CUSTOMER, "I'll take two Sprites, please."),
        (EventSource.AI_AGENT, "Anything else?"),
        (EventSource.CUSTOMER, "No, that's all. I want to pay."),
        (EventSource.AI_AGENT, "No problem! We accept only cash."),
        (EventSource.CUSTOMER, "Sure, I'll pay the delivery guy."),
        (EventSource.AI_AGENT, "Unfortunately, we accept payments only at our location."),
        (EventSource.CUSTOMER, "So what should I do now?"),
    ]
    conversation_guideline_names: list[str] = [
        "check_toppings_in_stock",
        "check_drinks_in_stock",
        "payment_process",
        "address_location",
    ]
    relevant_guideline_names: list[str] = [
        "address_location",
    ]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


async def test_that_guideline_that_needs_to_be_reapplied_is_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.AI_AGENT,
            "Hi! Welcome to PizzaBot, your virtual pizza assistant. How can I help you today?",
        ),
        (EventSource.CUSTOMER, "Hi, I’d like to order a pizza."),
        (EventSource.AI_AGENT, "Great choice! What size would you like—small, medium, or large?"),
        (EventSource.CUSTOMER, "I’ll take a large pizza."),
        (
            EventSource.AI_AGENT,
            "Got it! And what type of crust would you like? We have classic, thin, stuffed, or gluten-free.",
        ),
        (EventSource.CUSTOMER, "Let’s go with stuffed crust."),
        (
            EventSource.AI_AGENT,
            "Yum! Now for the toppings. Would you like a classic like pepperoni or something custom?",
        ),
        (EventSource.CUSTOMER, "Can I do half pepperoni and half veggie?"),
        (
            EventSource.AI_AGENT,
            "Absolutely! Your pizza will have half pepperoni and half veggie. For the veggie side, would you like a mix of mushrooms, onions, bell peppers, and olives?",
        ),
        (EventSource.CUSTOMER, "Yes, that sounds perfect."),
        (EventSource.AI_AGENT, "Great! Would you like any extra cheese or dipping sauces?"),
        (EventSource.CUSTOMER, "Extra cheese for sure, and ranch dipping sauce, please."),
        (
            EventSource.AI_AGENT,
            "Got it. A large, stuffed crust pizza with half pepperoni, half veggie, extra cheese, and ranch dipping sauce. Anything else?",
        ),
        (EventSource.CUSTOMER, "Nope, that’s all. How long will it take?"),
        (
            EventSource.AI_AGENT,
            "Your pizza will be ready in about 25 minutes. Would you like delivery or pickup?",
        ),
        (EventSource.CUSTOMER, "Delivery please?"),
        (
            EventSource.AI_AGENT,
            "Great, the total would be 10$, would you like to pay by credit or cash?",
        ),
        (EventSource.CUSTOMER, "Actually hold up, could you add another large pizza to the order?"),
    ]

    conversation_guideline_names: list[str] = ["large_pizza_crust"]
    relevant_guideline_names = conversation_guideline_names
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
        context_variables=[],
    )


async def test_that_guidelines_based_on_context_variables_arent_matched_repetitively(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.AI_AGENT,
            "Hi! Welcome to PizzaBot, your virtual pizza assistant. We have a special summer deal - two large pizzas for the price of one! How can I help you today?",
        ),
        (EventSource.CUSTOMER, "Hi, I’d like to order a pizza."),
        (EventSource.AI_AGENT, "Great choice! What size would you like—small, medium, or large?"),
        (EventSource.CUSTOMER, "I’ll take a large pizza."),
        (
            EventSource.AI_AGENT,
            "Got it! And what type of crust would you like? We have classic, thin, stuffed, or gluten-free.",
        ),
        (EventSource.CUSTOMER, "Let’s go with stuffed crust."),
        (
            EventSource.AI_AGENT,
            "Yum! Now for the toppings. Would you like a classic like pepperoni or something custom?",
        ),
        (EventSource.CUSTOMER, "Can I do half pepperoni and half veggie?"),
        (
            EventSource.AI_AGENT,
            "Absolutely! Your pizza will have half pepperoni and half veggie. For the veggie side, would you like a mix of mushrooms, onions, bell peppers, and olives?",
        ),
        (EventSource.CUSTOMER, "Yes, that sounds perfect."),
        (EventSource.AI_AGENT, "Great! Would you like any extra cheese or dipping sauces?"),
        (EventSource.CUSTOMER, "Extra cheese for sure, and ranch dipping sauce, please."),
        (
            EventSource.AI_AGENT,
            "Got it. A large, stuffed crust pizza with half pepperoni, half veggie, extra cheese, and ranch dipping sauce. Anything else?",
        ),
        (EventSource.CUSTOMER, "Nope, that’s all. How long will it take?"),
        (
            EventSource.AI_AGENT,
            "Your pizza will be ready in about 25 minutes. Would you like delivery or pickup?",
        ),
        (EventSource.CUSTOMER, "Delivery please?"),
        (
            EventSource.AI_AGENT,
            "Great, the total would be 10$, would you like to pay by credit or cash?",
        ),
        (EventSource.CUSTOMER, "Actually hold up, could you add another large pizza to the order?"),
    ]
    context_variables = [
        create_context_variable(
            name="season",
            data={"season": "Summer"},
            tags=[Tag.for_agent_id(agent.id)],
        )
    ]

    conversation_guideline_names: list[str] = ["summer_sale"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        [],
        context_variables=context_variables,
    )


async def test_that_guidelines_are_not_considered_done_when_they_strictly_arent(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (EventSource.AI_AGENT, "Hey there, how can I help you?"),
        (EventSource.CUSTOMER, "I'd like to pay my credit card bill"),
        (
            EventSource.AI_AGENT,
            "Sure thing. For which card, and how much would you like to pay right now?",
        ),
        (EventSource.CUSTOMER, "For my amex please"),
    ]

    conversation_guideline_names: list[str] = ["pay_cc_bill"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        ["pay_cc_bill"],
    )


async def test_that_observational_guidelines_arent_wrongly_implied(
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
            name="Date",
            data={"Year": "2025", "Month": "January", "Day": 24},
            tags=[Tag.for_agent_id(agent.id)],
        ),
    ]

    tool_result = cast(
        JSONSerializable,
        {
            "tool_calls": [
                {
                    "tool_id": "local:get_weather",
                    "arguments": {},
                    "result": {"data": "The weather is rainy", "metadata": {}, "control": {}},
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

    conversation_guideline_names: list[str] = ["season_is_winter"]
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
        staged_events=staged_events,
    )


async def test_that_observational_guidelines_are_detected_correctly_when_lots_of_data_is_available(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    terms = [
        create_term(
            name="blorgnet",
            description="a figure of speech, meaning being annoyed by whoever you're interacting with",
            tags=[Tag.for_agent_id(agent.id)],
        ),
    ]
    context_variables = [
        create_context_variable(
            name="customer_location",
            data={"location": "Australia"},
            tags=[Tag.for_agent_id(agent.id)],
        ),
        create_context_variable(
            name="date",
            data={"date": "August 15th, 2024"},
            tags=[Tag.for_agent_id(agent.id)],
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
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I think I lost my credit card somewhere yesterday.",
        ),
        (
            EventSource.AI_AGENT,
            "I'm sorry to hear about your lost card. I can help you with that right away. For security purposes, could you please verify the last four digits of your card?",
        ),
        (
            EventSource.CUSTOMER,
            "It's the Platinum Rewards one.",
        ),
        (
            EventSource.AI_AGENT,
            "Thank you for that information. To proceed with locking your Platinum Rewards card, I'll need to verify the last four digits of the card number. This helps ensure we're securing the correct card on your account. Could you please provide those digits?",
        ),
        (
            EventSource.CUSTOMER,
            "blorgnet, I just need it locked right now. I have multiple cards and I don't memorize all the numbers.",
        ),
        (
            EventSource.AI_AGENT,
            "I understand your concern and the urgency of the situation. Let me try another approach. I can see your account has two Platinum Rewards cards. Would it be helpful if I locked both cards temporarily until we identify which one is missing? Or would you prefer to check your online banking app where you can see the last four digits of your cards?",
        ),
        (
            EventSource.CUSTOMER,
            "Just lock all of them. I need to go to a meeting in 5 minutes.",
        ),
        (
            EventSource.AI_AGENT,
            "I've placed temporary locks on all your Platinum Rewards cards. This will prevent any unauthorized transactions while keeping your account active. The locks are effective immediately. Would you like me to order replacement cards for you now, or would you prefer to wait in case you find the missing card?",
        ),
        (
            EventSource.CUSTOMER,
            "When will your branch on Main Street be open?",
        ),
        (
            EventSource.AI_AGENT,
            "Our Main Street branch is open Monday through Friday from 9:00 AM to 5:00 PM, and Saturday from 9:00 AM to 1:00 PM. It's closed on Sundays. Regarding your cards, would you like me to order replacements for any of them at this time?",
        ),
        (
            EventSource.CUSTOMER,
            "And do they handle international wire transfers there?",
        ),
        (
            EventSource.AI_AGENT,
            "Yes, our Main Street branch does handle international wire transfers. You would need to bring a valid ID and the recipient's banking details. The branch can also help with your card situation if you prefer to handle it in person. In the meantime, your cards remain locked for your protection. Is there anything else you need assistance with today?",
        ),
        (
            EventSource.CUSTOMER,
            "What's the maximum amount for international transfers?",
        ),
    ]
    conversation_guideline_names: list[str] = list(OBSERVATIONAL_GUIDELINES_DICT.keys())
    relevant_guideline_names = [
        "lock_card_request_1",
        "lock_card_request_2",
        "season_is_winter",
        "frustrated_customer_observational",
        "unanswered_questions",
    ]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        new_session.id,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
        staged_events=staged_events,
        context_variables=context_variables,
        terms=terms,
    )
