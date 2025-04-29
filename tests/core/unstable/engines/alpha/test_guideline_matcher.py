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

from datetime import datetime, timezone
from itertools import chain
from typing import Sequence

from parlant.core.agents import Agent
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
from parlant.core.engines.alpha.guideline_matcher import (
    GuidelineMatcher,
    GuidelineMatchingStrategyResolver,
)
from parlant.core.engines.alpha.guideline_match import (
    GuidelineMatch,
)
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.sessions import EventSource
from parlant.core.glossary import TermId

from parlant.core.tags import Tag, TagId
from tests.core.common.utils import ContextOfTest, create_event_message


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
        "ensuring to clarify any necessary details such as class type, date, and requirements.",
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
    "second_thanks": {
        "condition": "the customer is thanking you for the second time in the interaction",
        "action": "compliment the customer for their manners",
    },
    "pay_cc_bill": {
        "condition": "the customer wants to pay their credit card bill",
        "action": "determine which card and how much they want to pay",
    },
}


def match_guidelines(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
    conversation_context: list[tuple[EventSource, str]],
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
    terms: Sequence[Term] = [],
    staged_events: Sequence[EmittedEvent] = [],
) -> Sequence[GuidelineMatch]:
    guideline_matcher = GuidelineMatcher(
        context.container[Logger],
        context.container[GuidelineMatchingStrategyResolver],
    )

    interaction_history = [
        create_event_message(
            offset=i,
            source=source,
            message=message,
        )
        for i, (source, message) in enumerate(conversation_context)
    ]

    guideline_matching_result = context.sync_await(
        guideline_matcher.match_guidelines(
            agent=agent,
            customer=customer,
            context_variables=context_variables,
            interaction_history=interaction_history,
            terms=terms,
            staged_events=staged_events,
            guidelines=list(context.guidelines.values()),
        )
    )

    return list(chain.from_iterable(guideline_matching_result.batches))


def create_guideline(
    context: ContextOfTest,
    guideline_name: str,
    condition: str,
    action: str,
    tags: list[TagId],
) -> Guideline:
    guideline = Guideline(
        id=GuidelineId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        enabled=True,
        content=GuidelineContent(
            condition=condition,
            action=action,
        ),
        tags=tags,
        metadata={},
    )

    context.guidelines[guideline_name] = guideline

    return guideline


def create_term(
    name: str,
    description: str,
    synonyms: list[str] = [],
    tags: list[TagId] = [],
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
    tags: list[TagId],
) -> Guideline:
    guideline = create_guideline(
        context=context,
        guideline_name=guideline_name,
        condition=GUIDELINES_DICT[guideline_name]["condition"],
        action=GUIDELINES_DICT[guideline_name]["action"],
        tags=tags,
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
        name: create_guideline_by_name(context, name, tags=[Tag.for_agent_id(agent.id)])
        for name in conversation_guideline_names
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


def test_that_guideline_with_multiple_actions_is_partially_fulfilled_when_one_action_occured(
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
            "Great Question! Owls can live 5 to 30 years in the wild, and even longer in captivity.",
        ),
        (EventSource.CUSTOMER, "That's shorter than I expected, thank you!"),
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


def test_that_irrelevant_guidelines_are_not_matched_parametrized_2(
    context: ContextOfTest,
    agent: Agent,
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
    base_test_that_correct_guidelines_are_matched(
        context, agent, customer, conversation_context, conversation_guideline_names, []
    )


def test_that_many_guidelines_are_classified_correctly(  # a stress test
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
        for guideline_name in GUIDELINES_DICT.keys()
        if guideline_name not in exceptions
    ]
    relevant_guideline_names = ["announce_shipment", "second_thanks"]
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


def test_that_relevant_guidelines_are_matched_parametrized_1(
    context: ContextOfTest,
    agent: Agent,
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
            "Awesome. I've added a large pepperoni pizza. " "Would you like a drink on the side?",
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
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
    )


def test_that_guideline_that_needs_to_be_reapplied_is_matched(
    context: ContextOfTest,
    agent: Agent,
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
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        relevant_guideline_names,
        context_variables=[],
    )


def test_that_guidelines_based_on_context_variables_arent_matched_repetitively(
    context: ContextOfTest,
    agent: Agent,
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
    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        [],
        context_variables=context_variables,
    )


def test_that_guidelines_are_not_considered_done_when_they_strictly_arent(
    context: ContextOfTest,
    agent: Agent,
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

    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        customer,
        conversation_context,
        conversation_guideline_names,
        ["pay_cc_bill"],
    )
