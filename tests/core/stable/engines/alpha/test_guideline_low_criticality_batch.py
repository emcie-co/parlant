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

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from lagom import Container
from pytest import fixture
from parlant.core.agents import Agent
from parlant.core.capabilities import Capability
from parlant.core.common import Criticality, generate_id
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.generic.guideline_low_criticality_batch import (
    GenericLowCriticalityGuidelineMatchesSchema,
    GenericLowCriticalityGuidelineMatchingBatch,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matching_context import (
    GuidelineMatchingContext,
)
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import EventSource, Session, SessionId, SessionStore
from parlant.core.tags import TagId
from tests.core.common.utils import create_event_message
from tests.test_utilities import SyncAwaiter


GUIDELINES_DICT = {
    "transfer_to_manager": {
        "condition": "When customer ask to talk with a manager",
        "action": "Hand them over to a manager immediately.",
    },
    "problem_so_restart": {
        "condition": "The customer has a problem with the app and hasn't tried troubleshooting yet",
        "action": "Suggest to do restart",
    },
    "frustrated_so_discount": {
        "condition": "The customer expresses frustration, impatience, or dissatisfaction",
        "action": "apologize and offer a discount",
    },
    "don't_transfer_to_manager": {
        "condition": "When customer ask to talk with a manager",
        "action": "Explain that it's not possible to talk with a manager and that you are here to help",
    },
    "first_order_and_order_more_than_2": {
        "condition": "When this is the customer first time ordering in the restaurant and the order they made includes more than 2 pizzas",
        "action": "offer 2 for 1 sale",
    },
    "first_order_and_order_exactly_2": {
        "condition": "When this is the customer first time ordering in the restaurant and the order they made includes exactly 2 pizzas",
        "action": "offer 2 for 1 sale",
    },
    "identify_problem": {
        "condition": "When customer say that they got an error or that something is not working",
        "action": "help them identify the source of the problem",
    },
    "frustrated_customer": {
        "condition": "the customer appears frustrated or upset",
        "action": "Acknowledge the customer's concerns, apologize for any inconvenience, and offer a solution or escalate the issue to a supervisor if necessary.",
    },
    "do_payment": {
        "condition": "the customer wants to pay for a product",
        "action": "Use the do_payment tool to process their payment.",
    },
    "problem_with_order": {
        "condition": "The customer is reporting a problem with their order.",
        "action": "Apologize and ask for more details about the issue.",
    },
    "delivery_time_inquiry": {
        "condition": "When the customer asks about the estimated delivery time for their order.",
        "action": "Always use Imperial units",
    },
    "cancel_subscription": {
        "condition": "When the user asks for help canceling a subscription.",
        "action": "Help them cancel it",
    },
    "ordering_sandwich": {
        "condition": "the customer wants to order a sandwich",
        "action": "only discuss options which are in stock",
    },
    "unsupported_capability": {
        "condition": "When a customer asks about a capability that is not supported",
        "action": "Tell them that you can not help them with this matter",
    },
    "multiple_capabilities": {
        "condition": "When there are multiple capabilities that are relevant for the customer's request",
        "action": "ask the customer which of the capabilities they want to use",
    },
    "rebook_reservation": {
        "condition": "The customer requests to change or rebook an existing reservation or flight",
        "action": "process the rebooking, confirm the new details, and check if anything else should be added before finalizing",
    },
    "be_polite": {
        "condition": "The customer is interacting with the agent",
        "action": "Be polite and helpful",
    },
    "unknown_issue_selling_pizza": {
        "condition": "The customer asks for help with an issue that is not directly related to selling a pizza",
        "action": "Tell them that you can not help with unknown issues",
    },
    "greeting": {
        "condition": "greeting a customer",
        "action": "Refer to them by name and welcome them warmly",
    },
    "ask_question": {
        "condition": "The customer asks a question",
        "action": "Ask them more clarifying questions to better understand what they need",
    },
    "combo_deal": {
        "condition": "The customer wants to make a pizza order",
        "action": "Suggest our combo deals to save money",
    },
    "extra_combo": {
        "condition": "The customer wants to make a pizza order",
        "action": "Offer an extra combo deal for drinks and sides",
    },
    "first_time_customer": {
        "condition": "The customer is ordering for the first time",
        "action": "Consider offering a first-time customer discount",
    },
}


@dataclass
class ContextOfTest:
    container: Container
    sync_await: SyncAwaiter
    guidelines: list[Guideline]
    schematic_generator: SchematicGenerator[GenericLowCriticalityGuidelineMatchesSchema]
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
        schematic_generator=container[
            SchematicGenerator[GenericLowCriticalityGuidelineMatchesSchema]
        ],
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
        criticality=Criticality.LOW,
        enabled=True,
        tags=tags,
        metadata={},
    )

    context.guidelines.append(guideline)

    return guideline


async def base_test_that_correct_guidelines_are_matched(
    context: ContextOfTest,
    agent: Agent,
    session_id: SessionId,
    customer: Customer,
    conversation_context: list[tuple[EventSource, str]],
    guidelines_target_names: list[str],
    guidelines_names: list[str],
    staged_events: Sequence[EmittedEvent] = [],
    capabilities: list[Capability] = [],
) -> None:
    conversation_guidelines = {
        name: create_guideline_by_name(context, name) for name in guidelines_names
    }

    target_guidelines = [conversation_guidelines[name] for name in guidelines_target_names]

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

    guideline_matching_context = GuidelineMatchingContext(
        agent=agent,
        session=session,
        customer=customer,
        context_variables=[],
        interaction_history=interaction_history,
        terms=[],
        capabilities=capabilities,
        staged_events=staged_events,
        active_journeys=[],
        journey_paths={k: list(v) for k, v in session.agent_states[-1].journey_paths.items()}
        if session.agent_states
        else {},
    )

    guideline_actionable_matcher = GenericLowCriticalityGuidelineMatchingBatch(
        logger=context.container[Logger],
        meter=context.container[Meter],
        optimization_policy=context.container[OptimizationPolicy],
        schematic_generator=context.schematic_generator,
        guidelines=context.guidelines,
        journeys=[],
        context=guideline_matching_context,
    )

    result = await guideline_actionable_matcher.process()

    matched_guidelines = [p.guideline for p in result.matches]

    assert set(matched_guidelines) == set(target_guidelines)


async def test_relevant_guideline_with_low_criticality_are_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, it's my first time here!",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome to our pizza store! what would you like?",
        ),
        (
            EventSource.CUSTOMER,
            "I want 2 pizzas please",
        ),
    ]

    guidelines: list[str] = [
        "first_time_customer",
        "first_order_and_order_more_than_2",
        "first_order_and_order_exactly_2",
        "transfer_to_manager",
        "identify_problem",
        "frustrated_so_discount",
        "problem_with_order",
        "delivery_time_inquiry",
        "ordering_sandwich",
        "rebook_reservation",
        "problem_so_restart",
        "don't_transfer_to_manager",
        "do_payment",
        "be_polite",
        "combo_deal",
        "extra_combo",
    ]
    guidelines_target_names: list[str] = [
        "first_order_and_order_exactly_2",
        "be_polite",
        "combo_deal",
        "extra_combo",
        "first_time_customer",
    ]
    await base_test_that_correct_guidelines_are_matched(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        guidelines_target_names=guidelines_target_names,
        guidelines_names=guidelines,
    )


async def test_guidelines_with_low_criticality_are_not_matched_when_no_longer_relevant(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Ugh, why is this taking so long? I placed my order 40 minutes ago.",
        ),
        (
            EventSource.AI_AGENT,
            "I'm really sorry for the delay, and I completely understand how frustrating that must be. I'll look into it right away, and I can also offer you a discount for the inconvenience.",
        ),
        (
            EventSource.CUSTOMER,
            "OK, thanks. I will be waiting",
        ),
        (
            EventSource.AI_AGENT,
            "Of course. I'm here to help, and I'll keep you updated as soon as I know more",
        ),
        (
            EventSource.CUSTOMER,
            "I got the delivery now and it's totally broken! Are you serious, you guys? This is ridiculous.",
        ),
        (
            EventSource.AI_AGENT,
            "I'm so sorry—that should absolutely not have happened. I'll report this right away, and I can offer you a discount for the trouble.",
        ),
        (
            EventSource.CUSTOMER,
            "Thank you that's nice of you.",
        ),
    ]

    guidelines: list[str] = [
        "first_order_and_order_more_than_2",
        "first_order_and_order_exactly_2",
        "transfer_to_manager",
        "identify_problem",
        "frustrated_so_discount",
        "problem_with_order",
        "delivery_time_inquiry",
        "ordering_sandwich",
        "rebook_reservation",
        "problem_so_restart",
        "don't_transfer_to_manager",
        "do_payment",
    ]
    guidelines_target_names: list[str] = []
    await base_test_that_correct_guidelines_are_matched(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        guidelines_target_names=guidelines_target_names,
        guidelines_names=guidelines,
    )


async def test_relevant_guideline_with_low_criticality_are_matched_when_still_relevant(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, it's my first time here!",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome to our pizza store! What would you like?",
        ),
        (
            EventSource.CUSTOMER,
            "Can you send me the recipe of your pizza?",
        ),
    ]

    guidelines: list[str] = [
        "unknown_issue_selling_pizza",
        "first_order_and_order_more_than_2",
        "first_order_and_order_exactly_2",
        "transfer_to_manager",
        "identify_problem",
        "frustrated_so_discount",
        "problem_with_order",
        "delivery_time_inquiry",
        "ordering_sandwich",
        "rebook_reservation",
        "problem_so_restart",
        "don't_transfer_to_manager",
        "do_payment",
        "be_polite",
        "ask_question",
    ]
    guidelines_target_names: list[str] = [
        "be_polite",
        "unknown_issue_selling_pizza",
        "ask_question",
    ]
    await base_test_that_correct_guidelines_are_matched(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        guidelines_target_names=guidelines_target_names,
        guidelines_names=guidelines,
    )


async def test_relevant_guideline_with_low_criticality_are_matched_when_still_relevant_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Your app keeps crashing when I try to open it.",
        ),
        (
            EventSource.AI_AGENT,
            "I'm sorry to hear that! Could you tell me the exact error message you're seeing?",
        ),
        (
            EventSource.CUSTOMER,
            "Anyway, I was also wondering if you have any discounts available right now?",
        ),
    ]

    guidelines: list[str] = [
        "do_payment",
        "be_polite",
        "greeting",
        "ask_question",
        "frustrated_so_discount",
        "problem_with_order",
        "identify_problem",
        "problem_so_restart",
    ]
    guidelines_target_names: list[str] = [
        "be_polite",
        "ask_question",
    ]
    await base_test_that_correct_guidelines_are_matched(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        guidelines_target_names=guidelines_target_names,
        guidelines_names=guidelines,
    )
