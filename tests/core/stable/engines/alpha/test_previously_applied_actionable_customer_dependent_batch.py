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
from parlant.core.capabilities import Capability, CapabilityId
from parlant.core.common import Criticality, generate_id
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.generic.guideline_previously_applied_actionable_customer_dependent_batch import (
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema,
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingBatch,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matching_context import (
    GuidelineMatchingContext,
)
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.journeys import Journey
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import EventSource, Session, SessionId, SessionStore
from parlant.core.tags import TagId
from tests.core.common.utils import create_event_message
from tests.test_utilities import SyncAwaiter


GUIDELINES_DICT = {
    "reservation_location": {
        "condition": "customer wants to make a reservation",
        "action": "check if they prefer inside or outside",
    },
    "issue_reporting": {
        "condition": "The customer is reporting a technical issue",
        "action": "Ask for the exact error message or steps to reproduce the issue",
    },
    "order_lookup": {
        "condition": "The customer wants to check their order status",
        "action": "Ask for their order number",
    },
    "order_alcohol": {
        "condition": "The customer wants to order alcohol",
        "action": "Check their age",
    },
    "unsupported_capability": {
        "condition": "When a customer asks about a capability that is not supported",
        "action": "ask the customer for their age before proceeding",
    },
    "multiple_capabilities": {
        "condition": "When there are multiple capabilities that are relevant for the customer's request",
        "action": "ask the customer which of the capabilities they want to use",
    },
}


@dataclass
class ContextOfTest:
    container: Container
    sync_await: SyncAwaiter
    guidelines: list[Guideline]
    schematic_generator: SchematicGenerator[
        GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema
    ]
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
            SchematicGenerator[
                GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema
            ]
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
        enabled=True,
        tags=tags,
        metadata={},
        criticality=Criticality.MEDIUM,
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
    capabilities: Sequence[Capability] = [],
    relevant_journeys: Sequence[Journey] = [],
) -> None:
    conversation_guidelines = {
        name: create_guideline_by_name(context, name) for name in guidelines_names
    }

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
        active_journeys=relevant_journeys,
        journey_paths={k: list(v) for k, v in session.agent_states[-1].journey_paths.items()}
        if session.agent_states
        else {},
    )

    guideline_previously_applied_matcher = (
        GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingBatch(
            logger=context.container[Logger],
            meter=context.container[Meter],
            optimization_policy=context.container[OptimizationPolicy],
            schematic_generator=context.schematic_generator,
            guidelines=context.guidelines,
            journeys=[],
            context=guideline_matching_context,
        )
    )

    result = await guideline_previously_applied_matcher.process()

    matched_guidelines = [p.guideline for p in result.matches]

    assert set(matched_guidelines) == set(previously_applied_target_guidelines)


async def test_that_customer_dependent_guideline_is_matched_when_customer_hasnt_completed_their_side(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I’d like to book a table for tomorrow night.",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! Would you prefer to sit inside or outside?",
        ),
        (
            EventSource.CUSTOMER,
            "7 PM would be great.",
        ),
    ]

    guidelines: list[str] = ["reservation_location"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_customer_dependent_guideline_is_not_matched_when_customer_has_completed_their_side(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I’d like to book a table for tomorrow night.",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! Would you prefer to sit inside or outside?",
        ),
        (
            EventSource.CUSTOMER,
            "I prefer it outside, thanks",
        ),
    ]

    guidelines: list[str] = ["reservation_location"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


async def test_that_customer_dependent_guideline_is_matched_when_customer_hasnt_completed_their_side_over_several_messages(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I’d like to book a table for tomorrow night.",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! Would you prefer to sit inside or outside?",
        ),
        (
            EventSource.CUSTOMER,
            "Tomorrow at 7 PM would be great.",
        ),
        (
            EventSource.AI_AGENT,
            "Great, I’ve noted 7 PM. Do you have a seating preference?",
        ),
        (
            EventSource.CUSTOMER,
            "And can it be a quiet table if possible?",
        ),
    ]

    guidelines: list[str] = ["reservation_location"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_customer_dependent_guideline_is_not_matched_when_customer_hasnt_completed_their_side_but_change_subject(
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
            "I’m sorry to hear that! Could you tell me the exact error message you’re seeing?",
        ),
        (
            EventSource.CUSTOMER,
            "Anyway, I was also wondering if you have any discounts available right now?",
        ),
    ]

    guidelines: list[str] = ["issue_reporting"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


async def test_that_customer_dependent_guideline_is_matched_when_customer_hasnt_completed_their_side_on_the_second_time(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Can you check the status of my phone order?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! Could you share the order number?",
        ),
        (
            EventSource.CUSTOMER,
            "It’s 12345. Thanks.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it. It's on the way and should arrive by Thursday.",
        ),
        (
            EventSource.CUSTOMER,
            "Great. What about the headphones I ordered last week?",
        ),
        (
            EventSource.AI_AGENT,
            "I'll check right now. Whats the order number for them?",
        ),
        (
            EventSource.CUSTOMER,
            "I need to check just a second",
        ),
    ]

    guidelines: list[str] = ["order_lookup"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_customer_dependent_guideline_is_matched_when_condition_arises_for_the_second_time(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Can you check the status of my phone order?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! Could you share the order number?",
        ),
        (
            EventSource.CUSTOMER,
            "It’s 12345. Thanks.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it. It's on the way and should arrive by Thursday.",
        ),
        (
            EventSource.CUSTOMER,
            "Great. What about the headphones I ordered last week?",
        ),
    ]

    guidelines: list[str] = ["order_lookup"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_customer_dependent_guideline_is_not_matched_when_condition_arises_for_the_second_time_but_completed(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Can you check the status of my phone order?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! Could you share the order number?",
        ),
        (
            EventSource.CUSTOMER,
            "It’s 12345. Thanks.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it. It's on the way and should arrive by Thursday.",
        ),
        (
            EventSource.CUSTOMER,
            "Great. What about the headphones I ordered last week?",
        ),
        (
            EventSource.AI_AGENT,
            "I'll check right now. Whats the order number for them?",
        ),
        (
            EventSource.CUSTOMER,
            "It’s 11122.",
        ),
    ]

    guidelines: list[str] = ["order_lookup"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


async def test_that_customer_dependent_guideline_is_not_matched_when_condition_arises_for_the_second_time_but_dont_need_to_take_the_action_again(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi can I get 2 beers?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure, but first, may I ask your age?",
        ),
        (
            EventSource.CUSTOMER,
            "I'm 25 thank God!",
        ),
        (
            EventSource.AI_AGENT,
            "Perfect — I’ve added 2 beers to your order. Would you like anything else?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, I'd also like some wine, please.",
        ),
    ]

    guidelines: list[str] = ["order_alcohol"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


async def test_that_customer_dependent_guideline_is_matched_based_on_capabilities_1(
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
            "Teach me how to tame dinosaurs",
        ),
        (
            EventSource.AI_AGENT,
            "Before proceeding, may I ask for your age?",
        ),
        (
            EventSource.CUSTOMER,
            "Sure! But can you help me get ice cream first?",
        ),
    ]
    conversation_guideline_names: list[str] = ["unsupported_capability", "multiple_capabilities"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=["unsupported_capability"],
        guidelines_names=conversation_guideline_names,
        capabilities=capabilities,
    )


async def test_that_customer_dependent_guideline_is_matched_based_on_capabilities_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    capabilities = [
        Capability(
            id=CapabilityId("cap_123"),
            creation_utc=datetime.now(timezone.utc),
            title="Increase Credit Limit",
            description="The ability to increase the customer's credit limit",
            signals=["increase credit limit", "credit limit"],
            tags=[],
        ),
        Capability(
            id=CapabilityId("cap_123"),
            creation_utc=datetime.now(timezone.utc),
            title="Decrease Credit Limit",
            description="The ability to decrease the customer's credit limit",
            signals=["decrease credit limit", "credit limit"],
            tags=[],
        ),
    ]
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Can you help me change my credit limits",
        ),
        (
            EventSource.AI_AGENT,
            "I can help you either increase or decrease your credit limit. Which option are you interested in?",
        ),
        (
            EventSource.CUSTOMER,
            "I just want to change them...",
        ),
    ]
    conversation_guideline_names: list[str] = ["unsupported_capability", "multiple_capabilities"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=["multiple_capabilities"],
        guidelines_names=conversation_guideline_names,
        capabilities=capabilities,
    )
