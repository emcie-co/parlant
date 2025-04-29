# Copyright 2025 Emcie Co Ltd.
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
from pytest import fixture

from parlant.core.agents import Agent, AgentId
from parlant.core.common import generate_id, JSONSerializable
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableId,
    ContextVariableValue,
    ContextVariableValueId,
)
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent
from parlant.core.glossary import Term
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.engines.alpha.guideline_matcher import (
    DefaultGuidelineMatchingStrategyResolver,
    GuidelineMatcher,
    GenericGuidelineMatchesSchema,
    GuidelineMatchingBatch,
    GuidelineMatchingBatchResult,
    GuidelineMatchingStrategy,
    GuidelineMatchingContext,
    GuidelineMatchingStrategyResolver,
)
from parlant.core.engines.alpha.guideline_match import (
    GuidelineMatch,
    PreviouslyAppliedType,
)
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.sessions import EventKind, EventSource
from parlant.core.loggers import Logger
from parlant.core.glossary import TermId

from parlant.core.tags import TagId, Tag
from tests.core.common.utils import create_event_message
from tests.test_utilities import SyncAwaiter


GUIDELINES_DICT = {
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
        "condition": "the customer wants to agent to perform an action that the agent is not designed for",
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
    schematic_generator: SchematicGenerator[GenericGuidelineMatchesSchema]
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
        schematic_generator=container[SchematicGenerator[GenericGuidelineMatchesSchema]],
    )


def match_guidelines(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
    conversation_context: list[tuple[EventSource, str]],
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
    terms: Sequence[Term] = [],
    staged_events: Sequence[EmittedEvent] = [],
) -> Sequence[GuidelineMatch]:
    interaction_history = [
        create_event_message(
            offset=i,
            source=source,
            message=message,
        )
        for i, (source, message) in enumerate(conversation_context)
    ]

    guideline_matching_result = context.sync_await(
        context.container[GuidelineMatcher].match_guidelines(
            agent=agent,
            customer=customer,
            context_variables=context_variables,
            interaction_history=interaction_history,
            terms=terms,
            staged_events=staged_events,
            guidelines=context.guidelines,
        )
    )

    return list(chain.from_iterable(guideline_matching_result.batches))


def create_guideline(
    context: ContextOfTest,
    condition: str,
    action: str,
    tags: list[TagId] = [],
) -> Guideline:
    guideline = Guideline(
        id=GuidelineId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        content=GuidelineContent(
            condition=condition,
            action=action,
        ),
        enabled=True,
        tags=tags,
        metadata={},
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
        name=name,
        description="",
        tool_id=None,
        freshness_rules=None,
        tags=tags,
    ), ContextVariableValue(
        ContextVariableValueId("-"), last_modified=datetime.now(timezone.utc), data=data
    )


def create_guideline_by_name(
    context: ContextOfTest,
    guideline_name: str,
) -> Guideline:
    guideline = create_guideline(
        context=context,
        condition=GUIDELINES_DICT[guideline_name]["condition"],
        action=GUIDELINES_DICT[guideline_name]["action"],
    )
    return guideline


def base_test_that_correct_guidelines_are_matched(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
    conversation_context: list[tuple[EventSource, str]],
    conversation_guideline_names: list[str],
    relevant_guideline_names: list[str],
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
    terms: Sequence[Term] = [],
    staged_events: Sequence[EmittedEvent] = [],
) -> None:
    conversation_guidelines = {
        name: create_guideline_by_name(context, name) for name in conversation_guideline_names
    }
    relevant_guidelines = [
        conversation_guidelines[name]
        for name in conversation_guidelines
        if name in relevant_guideline_names
    ]

    guideline_matches = match_guidelines(
        context,
        agent,
        customer,
        conversation_context,
        context_variables=context_variables,
        terms=terms,
        staged_events=staged_events,
    )
    matched_guidelines = [p.guideline for p in guideline_matches]

    assert set(matched_guidelines) == set(relevant_guidelines)


def test_that_relevant_guidelines_are_matched_parametrized_2(
    context: ContextOfTest,
    agent: Agent,
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
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


def test_that_irrelevant_guidelines_are_not_matched_parametrized_1(
    context: ContextOfTest,
    agent: Agent,
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
            "Awesome. I've added a large pepperoni pizza. " "Would you like a drink on the side?",
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
    base_test_that_correct_guidelines_are_matched(
        context, agent, customer, conversation_context, conversation_guideline_names, []
    )


def test_that_guidelines_with_the_same_conditions_are_scored_similarly(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
) -> None:
    relevant_guidelines = [
        create_guideline(
            context=context,
            condition="the customer greets you",
            action="talk about apples",
        ),
        create_guideline(
            context=context,
            condition="the customer greets you",
            action="talk about oranges",
        ),
    ]

    _ = [  # irrelevant guidelines
        create_guideline(
            context=context,
            condition="talking about the weather",
            action="talk about apples",
        ),
        create_guideline(
            context=context,
            condition="talking about the weather",
            action="talk about oranges",
        ),
    ]

    guideline_matches = match_guidelines(
        context,
        agent,
        customer,
        [(EventSource.CUSTOMER, "Hello there")],
    )

    assert len(guideline_matches) == len(relevant_guidelines)
    assert all(gp.guideline in relevant_guidelines for gp in guideline_matches)
    matches_scores = list(unique(gp.score for gp in guideline_matches))
    assert len(matches_scores) == 1 or (
        len(matches_scores) == 2 and abs(matches_scores[0] - matches_scores[1]) <= 1
    )


def test_that_guidelines_are_matched_based_on_agent_description(
    context: ContextOfTest,
    customer: Customer,
) -> None:
    agent = Agent(
        id=AgentId("123"),
        creation_utc=datetime.now(timezone.utc),
        name="skaetboard-sales-agent",
        description="You are an agent working for a skateboarding manufacturer. You help customers by discussing and recommending our products."
        "Your role is only to consult customers, and not to actually sell anything, as we sell our products in-store.",
        max_engine_iterations=3,
        tags=[],
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
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


def test_that_guidelines_are_matched_based_on_glossary(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
) -> None:
    terms = [
        create_term(
            name="skateboard",
            description="a time-travelling device",
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
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
        terms=terms,
    )


def test_that_conflicting_actions_with_similar_conditions_are_both_detected(
    context: ContextOfTest,
    agent: Agent,
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
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


def test_that_guidelines_are_matched_based_on_staged_tool_calls_and_context_variables(
    context: ContextOfTest,
    agent: Agent,
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
    staged_events = [
        EmittedEvent(
            source=EventSource.AI_AGENT, kind=EventKind.TOOL, correlation_id="", data=tool_result_1
        ),
        EmittedEvent(
            source=EventSource.AI_AGENT, kind=EventKind.TOOL, correlation_id="", data=tool_result_2
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
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
        staged_events=staged_events,
        context_variables=context_variables,
    )


def test_that_guidelines_are_matched_based_on_staged_tool_calls_without_context_variables(
    context: ContextOfTest,
    agent: Agent,
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
            correlation_id="",
            data=tool_result_1,
        ),
        EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.TOOL,
            correlation_id="",
            data=tool_result_2,
        ),
    ]
    conversation_guideline_names: list[str] = ["suggest_drink_underage", "suggest_drink_adult"]
    relevant_guideline_names = ["suggest_drink_underage"]
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names=relevant_guideline_names,
        staged_events=staged_events,
    )


def test_that_already_addressed_guidelines_arent_matched(
    context: ContextOfTest,
    agent: Agent,
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
    base_test_that_correct_guidelines_are_matched(
        context, agent, customer, conversation_context, conversation_guideline_names, []
    )


def test_that_guidelines_referring_to_continuous_processes_are_detected_even_if_already_fulfilled(
    context: ContextOfTest,
    agent: Agent,
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
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


def test_that_guideline_with_already_addressed_condition_but_unaddressed_action_is_matched(
    context: ContextOfTest,
    agent: Agent,
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
    relevant_guideline_names = conversation_guideline_names
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


def test_that_guideline_isnt_detected_based_on_its_action(
    context: ContextOfTest,
    agent: Agent,
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
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


def test_that_guideline_with_fulfilled_action_regardless_of_condition_can_be_reapplied(
    context: ContextOfTest,
    agent: Agent,
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
    relevant_guideline_names = conversation_guideline_names
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


def test_that_guideline_with_initial_response_is_matched(
    context: ContextOfTest,
    agent: Agent,
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
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


def test_that_guideline_with_multiple_actions_is_partially_fulfilled_when_a_few_actions_occured(
    context: ContextOfTest,
    agent: Agent,
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
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        [],
    )


class ActivateEveryGuidelineBatch(GuidelineMatchingBatch):
    def __init__(self, guidelines: Sequence[Guideline]):
        self.guidelines = guidelines

    @override
    async def process(self) -> GuidelineMatchingBatchResult:
        return GuidelineMatchingBatchResult(
            matches=[
                GuidelineMatch(
                    guideline=g,
                    score=10,
                    rationale="",
                    guideline_previously_applied=PreviouslyAppliedType.NO,
                    guideline_is_continuous=False,
                    should_reapply=False,
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


def test_that_guideline_matching_strategies_can_be_overridden(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
) -> None:
    class SkipAllGuidelineBatch(GuidelineMatchingBatch):
        def __init__(self, guidelines: Sequence[Guideline]):
            self.guidelines = guidelines

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
        async def create_batches(
            self,
            guidelines: Sequence[Guideline],
            context: GuidelineMatchingContext,
        ) -> Sequence[GuidelineMatchingBatch]:
            return [
                ActivateEveryGuidelineBatch(guidelines=guidelines),
            ]

    class ShortConditionStrategy(GuidelineMatchingStrategy):
        @override
        async def create_batches(
            self,
            guidelines: Sequence[Guideline],
            context: GuidelineMatchingContext,
        ) -> Sequence[GuidelineMatchingBatch]:
            return [SkipAllGuidelineBatch(guidelines=guidelines)]

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
        create_guideline(context, "a customer asks for a drink", "check stock"),
        create_guideline(context, "ask for drink", "check stock"),
        create_guideline(context, "customer needs help", "assist customer"),
        create_guideline(context, "help", "assist customer"),
    ]

    guideline_matches = match_guidelines(context, agent, customer, [])

    long_condition_guidelines = [g for g in guidelines if len(g.content.condition.split()) >= 4]
    short_condition_guidelines = [g for g in guidelines if len(g.content.condition.split()) < 4]

    assert all(
        g in [match.guideline for match in guideline_matches] for g in long_condition_guidelines
    )

    assert all(
        g not in [match.guideline for match in guideline_matches]
        for g in short_condition_guidelines
    )


def test_that_strategy_for_specific_guideline_can_be_overridden_in_default_strategy_resolver(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
) -> None:
    class CustomGuidelineMatchingStrategy(GuidelineMatchingStrategy):
        @override
        async def create_batches(
            self,
            guidelines: Sequence[Guideline],
            context: GuidelineMatchingContext,
        ) -> Sequence[GuidelineMatchingBatch]:
            return [ActivateEveryGuidelineBatch(guidelines=guidelines)]

    guideline = create_guideline(context, "a customer asks for a drink", "check stock")

    context.container[DefaultGuidelineMatchingStrategyResolver].guideline_overrides[
        guideline.id
    ] = CustomGuidelineMatchingStrategy()

    create_guideline(context, "ask for drink", "check stock")
    create_guideline(context, "customer needs help", "assist customer")

    conversation_context: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "I want help with my order"),
    ]

    guideline_matches = match_guidelines(context, agent, customer, conversation_context)

    assert guideline.id in [match.guideline.id for match in guideline_matches]
