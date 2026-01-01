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

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

from lagom import Container
from pytest import fixture
from parlant.core.agents import Agent
from parlant.core.common import Criticality, generate_id
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.generic.response_analysis_batch import (
    GenericResponseAnalysisSchema,
    GenericResponseAnalysisBatch,
)
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    ResponseAnalysisContext,
)
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import EventSource, Session, SessionId, SessionStore
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
    guidelines_to_tools: Mapping[Guideline, list[ToolId]]
    schematic_generator: SchematicGenerator[GenericResponseAnalysisSchema]
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
        schematic_generator=container[SchematicGenerator[GenericResponseAnalysisSchema]],
        logger=container[Logger],
    )


def create_guideline_by_name(
    context: ContextOfTest,
    guideline_name: str,
    tool_ids: list[ToolId] = [],
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
        criticality=Criticality.MEDIUM,
    )

    context.guidelines.append(guideline)

    return guideline


def create_guideline_with_tools(
    context: ContextOfTest,
    condition: str,
    action: str | None = None,
    tool_ids: list[ToolId] = [],
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
        criticality=Criticality.MEDIUM,
    )

    context.guidelines_to_tools = {guideline: tool_ids}

    return guideline


async def base_test_that_correct_guidelines_are_detected_as_previously_applied(
    context: ContextOfTest,
    agent: Agent,
    session_id: SessionId,
    customer: Customer,
    conversation_context: list[tuple[EventSource, str]],
    guidelines_target_names: list[str] = [],
    guidelines_names: list[str] = [],
    staged_events: Sequence[EmittedEvent] = [],
) -> None:
    conversation_guidelines: dict[str, Guideline] = defaultdict()
    if guidelines_names:
        for name in guidelines_names:
            conversation_guidelines[name] = create_guideline_by_name(context, name)

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

    for e in interaction_history:
        await context.container[SessionStore].create_event(
            session_id=session_id,
            source=e.source,
            kind=e.kind,
            trace_id=e.trace_id,
            data=e.data,
        )

    session = await context.container[SessionStore].read_session(session_id)

    guideline_matches = [
        GuidelineMatch(
            guideline=guideline,
            score=10,
            rationale="",
        )
        for guideline in context.guidelines
    ]

    response_analysis = GenericResponseAnalysisBatch(
        logger=context.container[Logger],
        meter=context.container[Meter],
        optimization_policy=context.container[OptimizationPolicy],
        schematic_generator=context.container[SchematicGenerator[GenericResponseAnalysisSchema]],
        context=ResponseAnalysisContext(
            agent=agent,
            session=session,
            customer=customer,
            context_variables=[],
            interaction_history=interaction_history,
            terms=[],
            staged_tool_events=staged_events,
            staged_message_events=[],
        ),
        guideline_matches=guideline_matches,
    )

    session = await context.container[SessionStore].read_session(session_id)

    result = await response_analysis.process()

    assert set([p.guideline for p in result.analyzed_guidelines if p.is_previously_applied]) == set(
        previously_applied_target_guidelines
    )


async def test_that_correct_guidelines_detect_as_previously_applied(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
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

    await base_test_that_correct_guidelines_are_detected_as_previously_applied(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_not_performed_guideline_is_not_detected_as_previously_applied(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I want to order 2 pizzas please",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! Which toppings would you like on your pizzas?",
        ),
    ]
    guidelines: list[str] = ["offer_two_pizza_for_one"]
    await base_test_that_correct_guidelines_are_detected_as_previously_applied(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


async def test_that_correct_guidelines_detect_as_previously_applied_when_guideline_action_also_depends_on_the_user_response(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
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

    await base_test_that_correct_guidelines_are_detected_as_previously_applied(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_correct_guidelines_detect_as_previously_applied_when_guideline_has_partially_applied_but_behavioral(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, what’s happening with my order? It’s been over an hour and I still haven’t received it!",
        ),
        (
            EventSource.AI_AGENT,
            "I’ll apply a discount to your order for the delay.",
        ),
    ]
    guidelines: list[str] = ["express_solidarity_and_discount"]

    await base_test_that_correct_guidelines_are_detected_as_previously_applied(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_correct_guideline_does_not_detect_as_previously_applied_when_guideline_has_partially_applied_and_functional(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
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

    await base_test_that_correct_guidelines_are_detected_as_previously_applied(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


async def test_that_correct_guidelines_detect_as_previously_applied_when_guideline_action_has_several_parts_that_applied_in_different_interaction_messages(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
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

    await base_test_that_correct_guidelines_are_detected_as_previously_applied(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_correct_guidelines_detect_as_previously_applied_when_guideline_action_applied_but_from_different_condition_1(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
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

    await base_test_that_correct_guidelines_are_detected_as_previously_applied(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_correct_guidelines_detect_as_previously_applied_when_guideline_action_applied_but_from_different_condition_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
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

    await base_test_that_correct_guidelines_are_detected_as_previously_applied(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_multiple_guidelines_detect_as_previously_applied_in_single_response(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, my order is late and when it finally arrived the food was cold!",
        ),
        (
            EventSource.AI_AGENT,
            "I'm so sorry to hear that your order arrived late and cold. "
            "I've applied a discount to your order to make up for this experience.",
        ),
    ]
    guidelines: list[str] = ["late_so_discount", "cold_so_discount"]

    await base_test_that_correct_guidelines_are_detected_as_previously_applied(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )
