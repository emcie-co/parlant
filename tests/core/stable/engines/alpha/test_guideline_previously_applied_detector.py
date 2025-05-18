from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Optional, Sequence, cast

from lagom import Container
from pytest import fixture
from parlant.core.agents import Agent
from parlant.core.common import JSONSerializable, generate_id
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.generic_guideline_previously_applied_detector import (
    GenericGuidelinePreviouslyAppliedDetector,
    GenericGuidelinePreviouslyAppliedDetectorSchema,
)
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import EventKind, EventSource
from parlant.core.tags import TagId
from parlant.core.tools import ToolId
from tests.core.common.utils import create_event_message
from tests.test_utilities import SyncAwaiter


GUIDELINES_DICT = {
    "offer_two_pizza_for_one": {
        "condition": "When customer wants to order 2 pizzas",
        "action": "tell them that we offer two large pizzas for the price of one",
    },
    "sorry_and_discount": {
        "condition": "When customer complains that they didn't get the order on time",
        "action": "tell them you are sorry and offer a discount",
    },
    "discount_and_check_status": {
        "condition": "When customer complains that they didn't get the order on time",
        "action": "offer a discount and check the order status",
    },
    "late_so_discount": {
        "condition": "When customer complains that they didn't get the order on time",
        "action": "offer a discount",
    },
    "cold_so_discount": {
        "condition": "When a customer complains that their food was delivered cold",
        "action": "offer a discount",
    },
    "check_stock": {
        "condition": "When a customer wants to order something",
        "action": "check we have it on stock",
    },
    "register": {
        "condition": "When a customer wants to register to our service",
        "action": "get their full name",
    },
    "express_solidarity_and_discount": {
        "condition": "When customer complains that they didn't get the order on time",
        "action": "express solidarity and offer a discount",
    },
    "link_when_asks_where_order": {
        "condition": "When customer asks where their order currently",
        "action": "provide the tracking link - https://trackinglink.com/abc123",
    },
}


@dataclass
class ContextOfTest:
    container: Container
    sync_await: SyncAwaiter
    guidelines: list[Guideline]
    guidelines_to_tools: Mapping[Guideline, Sequence[ToolId]]
    schematic_generator: SchematicGenerator[GenericGuidelinePreviouslyAppliedDetectorSchema]
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
        guidelines_to_tools=dict(),
        schematic_generator=container[
            SchematicGenerator[GenericGuidelinePreviouslyAppliedDetectorSchema]
        ],
        logger=container[Logger],
    )


def create_guideline_by_name(
    context: ContextOfTest,
    guideline_name: str,
    tool_ids: Sequence[ToolId] = [],
) -> Guideline:
    if tool_ids:
        guideline = create_guideline_with_tools(
            context=context,
            condition=GUIDELINES_DICT[guideline_name]["condition"],
            action=GUIDELINES_DICT[guideline_name]["action"],
            tool_ids=tool_ids,
        )
    else:
        guideline = create_guideline(
            context=context,
            condition=GUIDELINES_DICT[guideline_name]["condition"],
            action=GUIDELINES_DICT[guideline_name]["action"],
        )
    return guideline


def create_guideline(
    context: ContextOfTest,
    condition: str,
    action: str | None = None,
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


def create_guideline_with_tools(
    context: ContextOfTest,
    condition: str,
    action: str | None = None,
    tool_ids: Sequence[ToolId] = [],
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

    context.guidelines_to_tools = {guideline: tool_ids}

    return guideline


def base_test_that_correct_guidelines_detect_as_previously_applied(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
    conversation_context: list[tuple[EventSource, str]],
    guidelines_target_names: list[str] = [],
    ordinary_guidelines_names: list[str] = [],
    guidelines_with_tools: Optional[dict[str, Sequence[ToolId]]] = None,
    staged_events: Sequence[EmittedEvent] = [],
) -> None:
    conversation_guidelines: dict[str, Guideline] = defaultdict()
    if ordinary_guidelines_names:
        for name in ordinary_guidelines_names:
            conversation_guidelines[name] = create_guideline_by_name(context, name)

    if guidelines_with_tools:
        for name, tool_ids in guidelines_with_tools.items():
            conversation_guidelines[name] = create_guideline_by_name(context, name, tool_ids)

    previously_applied_target_guidelines = [
        conversation_guidelines[name] for name in guidelines_target_names
    ]

    interaction_history = [
        create_event_message(
            offset=i,
            source=source,
            message=message,
        )
        for i, (source, message) in enumerate(conversation_context)
    ]

    guideline_previously_applied_detector = GenericGuidelinePreviouslyAppliedDetector(
        logger=context.container[Logger],
        schematic_generator=context.schematic_generator,
    )

    ordinary_guideline_matches = [
        GuidelineMatch(
            guideline=guideline,
            score=10,
            rationale="",
        )
        for guideline in context.guidelines
    ]

    tool_enabled_guideline_matches = {
        GuidelineMatch(
            guideline=g,
            score=10,
            rationale="",
        ): t_ids
        for g, t_ids in context.guidelines_to_tools.items()
    }

    result = context.sync_await(
        guideline_previously_applied_detector.process(
            agent=agent,
            customer=customer,
            context_variables=[],
            interaction_history=interaction_history,
            terms=[],
            staged_events=staged_events,
            ordinary_guideline_matches=ordinary_guideline_matches,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
        )
    )

    assert set(result.previously_applied_guidelines) == set(previously_applied_target_guidelines)


def test_that_correct_guidelines_detect_as_previously_applied(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I want to order 2 pizzas please",
        ),
        (
            EventSource.AI_AGENT,
            "Hi! Great news — we’re currently offering two large pizzas for the price of one! Go ahead "
            "and let me know which two pizzas you’d like to order, and I’ll get that ready for you.",
        ),
    ]
    guidelines: list[str] = ["offer_two_pizza_for_one"]

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context,
        agent,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        ordinary_guidelines_names=guidelines,
    )


def test_that_correct_guidelines_detect_as_previously_applied_when_guideline_action_also_depends_on_the_user_response(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I want to register please",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! give me your full name and I will do that for you.",
        ),
    ]
    guidelines: list[str] = ["register"]

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context,
        agent,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        ordinary_guidelines_names=guidelines,
    )


def test_that_correct_guidelines_detect_as_previously_applied_when_guideline_has_partially_applied_but_behavioral(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, what’s happening with my order? It’s been over an hour and I still haven’t received it!",
        ),
        (
            EventSource.AI_AGENT,
            "I see your order is running late. I’m going to look into it right now and make sure it gets sorted. I’ll also apply a discount to your order for the delay.",
        ),
        (
            EventSource.CUSTOMER,
            "Ok let's make it quick. Also can you help me make another order for tomorrow?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure I'm here to help. What would you like to order?",
        ),
    ]
    guidelines: list[str] = ["express_solidarity_and_discount"]

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context,
        agent,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        ordinary_guidelines_names=guidelines,
    )


def test_that_correct_guideline_does_not_detect_as_previously_applied_when_guideline_has_partially_applied_and_functional(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, what’s happening with my order? It’s been over an hour and I still haven’t received it!",
        ),
        (
            EventSource.AI_AGENT,
            "I see your order is an hour late — I'll check the status right away and make sure it's on the way.",
        ),
    ]
    guidelines: list[str] = ["discount_and_check_status"]

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context,
        agent,
        customer,
        conversation_context,
        guidelines_target_names=[],
        ordinary_guidelines_names=guidelines,
    )


def test_that_correct_guidelines_detect_as_previously_applied_when_guideline_action_has_several_parts_that_applied_in_different_interaction_messages(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, what’s happening with my order? It’s been over an hour and I still haven’t received it!",
        ),
        (
            EventSource.AI_AGENT,
            "I "
            "see your order is an hour late — I'll check the status right away and make sure it's on the way.",
        ),
        (
            EventSource.CUSTOMER,
            "Okay, but this is really frustrating. I was expecting it a long time ago.",
        ),
        (
            EventSource.AI_AGENT,
            "I totally understand. To make up for the delay I’ve applied a discount to your order. Thanks for your patience",
        ),
    ]
    guidelines: list[str] = ["discount_and_check_status"]

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context,
        agent,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        ordinary_guidelines_names=guidelines,
    )


def test_that_correct_guidelines_detect_as_previously_applied_when_guideline_action_applied_but_from_different_condition_1(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, what’s happening with my order? It’s been over an hour and I still haven’t received it!",
        ),
        (
            EventSource.AI_AGENT,
            " I see your order is running late. I’m going to look into it right now and make sure it gets sorted. I’ll also apply a discount to your order for the delay.",
        ),
    ]
    guidelines: list[str] = ["late_so_discount", "cold_so_discount"]

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context,
        agent,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        ordinary_guidelines_names=guidelines,
    )


def test_that_correct_guidelines_detect_as_previously_applied_when_guideline_action_applied_but_from_different_condition_2(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, when is my package supposed to arrive?",
        ),
        (
            EventSource.AI_AGENT,
            "It’s on the way! You can track it here: https://trackinglink.com/abc123",
        ),
    ]

    guidelines: list[str] = ["link_when_asks_where_order"]

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context,
        agent,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        ordinary_guidelines_names=guidelines,
    )


def test_that_correct_guidelines_detect_as_previously_applied_when_guideline_action_fully_understood_when_considering_the_tools(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
) -> None:
    tool_names = ["get_available_drinks", "get_available_toppings"]
    tool_ids = [ToolId(service_name="local", tool_name=tool_name) for tool_name in tool_names]

    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, can I order pepperoni pizza and 2 soda?",
        ),
        (
            EventSource.AI_AGENT,
            "I see that we have pepperoni in stock.",
        ),
    ]

    tool_result_1 = cast(
        JSONSerializable,
        {
            "tool_calls": [
                {
                    "tool_id": "local:get_available_toppings",
                    "arguments": {},
                    "result": {"data": ["Pepperoni"], "metadata": {}, "control": {}},
                }
            ]
        },
    )
    tool_result_2 = cast(
        JSONSerializable,
        {
            "tool_calls": [
                {
                    "tool_id": "local:get_available_drinks",
                    "arguments": {},
                    "result": {"data": ["error"], "metadata": {}, "control": {}},
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

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context,
        agent,
        customer,
        conversation_context,
        guidelines_target_names=[],
        ordinary_guidelines_names=[],
        guidelines_with_tools={"check_stock": tool_ids},
        staged_events=staged_events,
    )
