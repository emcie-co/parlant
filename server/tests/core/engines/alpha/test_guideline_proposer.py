from dataclasses import dataclass
from typing import Sequence, cast
from lagom import Container
from pytest import fixture, mark
from datetime import datetime, timezone

from emcie.server.core.agents import Agent, AgentId
from emcie.server.core.common import generate_id
from emcie.server.core.nlp.generation import SchematicGenerator
from emcie.server.core.engines.alpha.guideline_proposer import (
    GuidelineProposer,
    GuidelinePropositionsSchema,
)
from emcie.server.core.engines.alpha.guideline_proposition import (
    GuidelineProposition,
)
from emcie.server.core.logging import Logger
from emcie.server.core.guidelines import Guideline, GuidelineContent, GuidelineId
from emcie.server.core.sessions import Event, EventId, EventSource

from tests.test_utilities import SyncAwaiter


@dataclass
class ContextOfTest:
    sync_await: SyncAwaiter
    guidelines: list[Guideline]
    schematic_generator: SchematicGenerator[GuidelinePropositionsSchema]
    logger: Logger


@fixture
def context(
    sync_await: SyncAwaiter,
    container: Container,
) -> ContextOfTest:
    return ContextOfTest(
        sync_await,
        guidelines=list(),
        logger=container[Logger],
        schematic_generator=container[SchematicGenerator[GuidelinePropositionsSchema]],
    )


def propose_guidelines(
    context: ContextOfTest,
    conversation_context: list[tuple[str, str]],
) -> Sequence[GuidelineProposition]:
    guideline_filter = GuidelineProposer(context.logger, context.schematic_generator)
    agents = [
        Agent(
            id=AgentId("123"),
            creation_utc=datetime.now(timezone.utc),
            name="Test Agent",
            description="You are an agent that works for Emcie",
        )
    ]

    interaction_history = [
        create_event_message(
            offset=i,
            source=cast(EventSource, source),
            message=message,
        )
        for i, (source, message) in enumerate(conversation_context)
    ]

    guideline_propositions = context.sync_await(
        guideline_filter.propose_guidelines(
            agents=agents,
            guidelines=context.guidelines,
            context_variables=[],
            interaction_history=interaction_history,
            terms=[],
            staged_events=[],
        )
    )

    return guideline_propositions


def create_event_message(
    offset: int,
    source: EventSource,
    message: str,
) -> Event:
    event = Event(
        id=EventId(generate_id()),
        source=source,
        kind="message",
        offset=offset,
        correlation_id="test_correlation_id",
        data={"message": message},
        creation_utc=datetime.now(timezone.utc),
    )

    return event


def create_guideline_by_name(
    context: ContextOfTest,
    guideline_name: str,
) -> Guideline:
    guidelines = {
        "check_drinks_in_stock": {
            "predicate": "a client asks for a drink",
            "action": "check if the drink is available in the following stock: "
            "['Sprite', 'Coke', 'Fanta']",
        },
        "check_toppings_in_stock": {
            "predicate": "a client asks for toppings",
            "action": "check if the toppings are available in the following stock: "
            "['Pepperoni', 'Tomatoes', 'Olives']",
        },
        "payment_process": {
            "predicate": "a client is in the payment process",
            "action": "Follow the payment instructions, "
            "which are: 1. Pay in cash only, 2. Pay only at the location.",
        },
        "address_location": {
            "predicate": "the client needs to know our address",
            "action": "Inform the client that our address is at Sapir 2, Herzliya.",
        },
        "mood_support": {
            "predicate": "the client is experiencing stress or dissatisfaction",
            "action": "Provide comforting responses and suggest alternatives "
            "or support to alleviate the client's mood.",
        },
        "class_booking": {
            "predicate": "the client asks about booking a class or an appointment",
            "action": "Provide available times and facilitate the booking process, "
            "ensuring to clarify any necessary details such as class type, date, and requirements.",
        },
    }

    guideline = Guideline(
        id=GuidelineId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        content=GuidelineContent(
            predicate=guidelines[guideline_name]["predicate"],
            action=guidelines[guideline_name]["action"],
        ),
    )
    context.guidelines.append(guideline)
    return guideline


@mark.parametrize(
    "conversation_context, conversation_guideline_names, relevant_guideline_names",
    [
        (
            [
                ("client", "I'd like to order a pizza, please."),
                ("server", "No problem. What would you like to have?"),
                ("client", "I'd like a large pizza. What toppings do you have?"),
                ("server", "Today, we have pepperoni, tomatoes, and olives available."),
                ("client", "I'll take pepperoni, thanks."),
                (
                    "server",
                    "Awesome. I've added a large pepperoni pizza. "
                    "Would you like a drink on the side?",
                ),
                ("client", "Sure. What types of drinks do you have?"),
                ("server", "We have Sprite, Coke, and Fanta."),
                ("client", "I'll take two Sprites, please."),
                ("server", "Anything else?"),
                ("client", "No, that's all. I want to pay."),
                ("server", "No problem! We accept only cash."),
                ("client", "Sure, I'll pay the delivery guy."),
                ("server", "Unfortunately, we accept payments only at our location."),
                ("client", "So what should I do now?"),
            ],
            [
                "check_toppings_in_stock",
                "check_drinks_in_stock",
                "payment_process",
                "address_location",
            ],
            [
                "payment_process",
            ],
        ),
        (
            [
                (
                    "client",
                    "I'm feeling a bit stressed about coming in. Can I cancel my class for today?",
                ),
                (
                    "server",
                    "I'm sorry to hear that. While cancellation is not possible now, "
                    "how about a lighter session? Maybe it helps to relax.",
                ),
                ("client", "I suppose that could work. What do you suggest?"),
                (
                    "server",
                    "How about our guided meditation session? "
                    "It’s very calming and might be just what you need right now.",
                ),
                ("client", "Alright, please book me into that. Thank you for understanding."),
                (
                    "server",
                    "You're welcome! I've switched your booking to the meditation session. "
                    "Remember, it's okay to feel stressed. We're here to support you.",
                ),
                ("client", "Thanks, I really appreciate it."),
                ("server", "Anytime! Is there anything else I can assist you with today?"),
                ("client", "No, that's all for now."),
                (
                    "server",
                    "Take care and see you soon at the meditation class. "
                    "Our gym is at Sapir 2, Herzliya, in case you need directions.",
                ),
                ("client", "Thank you!"),
            ],
            [
                "class_booking",
                "mood_support",
                "address_location",
            ],
            [
                "mood_support",
            ],
        ),
    ],
)
def test_that_relevant_guidelines_are_proposed(
    context: ContextOfTest,
    conversation_context: list[tuple[str, str]],
    conversation_guideline_names: list[str],
    relevant_guideline_names: list[str],
) -> None:
    conversation_guidelines = {
        name: create_guideline_by_name(context, name) for name in conversation_guideline_names
    }
    relevant_guidelines = [
        conversation_guidelines[name]
        for name in conversation_guidelines
        if name in relevant_guideline_names
    ]

    guideline_propositions = propose_guidelines(context, conversation_context)
    guidelines = [p.guideline for p in guideline_propositions]

    for guideline in relevant_guidelines:
        assert guideline in guidelines


@mark.parametrize(
    "conversation_context, conversation_guideline_names, irrelevant_guideline_names",
    [
        (
            [
                ("client", "I'd like to order a pizza, please."),
                ("server", "No problem. What would you like to have?"),
                ("client", "I'd like a large pizza. What toppings do you have?"),
                ("server", "Today we have pepperoni, tomatoes, and olives available."),
                ("client", "I'll take pepperoni, thanks."),
                (
                    "server",
                    "Awesome. I've added a large pepperoni pizza. "
                    "Would you like a drink on the side?",
                ),
                ("client", "Sure. What types of drinks do you have?"),
                ("server", "We have Sprite, Coke, and Fanta."),
                ("client", "I'll take two Sprites, please."),
                ("server", "Anything else?"),
                ("client", "No, that's all."),
                ("server", "How would you like to pay?"),
                ("client", "I'll pick it up and pay in cash, thanks."),
            ],
            ["check_toppings_in_stock", "check_drinks_in_stock"],
            ["check_toppings_in_stock", "check_drinks_in_stock"],
        ),
        (
            [
                ("client", "Could you add some pretzels to my order?"),
                ("server", "Pretzels have been added to your order. Anything else?"),
                ("client", "Do you have Coke? I'd like one, please."),
                ("server", "Coke has been added to your order."),
                ("client", "Great, where are you located at?"),
            ],
            ["check_drinks_in_stock"],
            ["check_drinks_in_stock"],
        ),
    ],
)
def test_that_irrelevant_guidelines_are_not_proposed(
    context: ContextOfTest,
    conversation_context: list[tuple[str, str]],
    conversation_guideline_names: list[str],
    irrelevant_guideline_names: list[str],
) -> None:
    conversation_guidelines = {
        name: create_guideline_by_name(context, name) for name in conversation_guideline_names
    }

    irrelevant_guidelines = [
        conversation_guidelines[name]
        for name in conversation_guidelines
        if name in irrelevant_guideline_names
    ]

    guideline_propositions = propose_guidelines(context, conversation_context)
    guidelines = [p.guideline for p in guideline_propositions]

    for guideline in guidelines:
        assert guideline not in irrelevant_guidelines