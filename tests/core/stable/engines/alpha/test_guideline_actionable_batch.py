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
from parlant.core.engines.alpha.guideline_matching.generic.guideline_actionable_batch import (
    GenericActionableGuidelineMatchesSchema,
    GenericActionableGuidelineMatchingBatch,
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
        "action": "ask the customer for their age before proceeding",
    },
    "multiple_capabilities": {
        "condition": "When there are multiple capabilities that are relevant for the customer's request",
        "action": "ask the customer which of the capabilities they want to use",
    },
    "rebook_reservation": {
        "condition": "The customer requests to change or rebook an existing reservation or flight",
        "action": "process the rebooking, confirm the new details, and check if anything else should be added before finalizing",
    },
}


@dataclass
class ContextOfTest:
    container: Container
    sync_await: SyncAwaiter
    guidelines: list[Guideline]
    schematic_generator: SchematicGenerator[GenericActionableGuidelineMatchesSchema]
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
        schematic_generator=container[SchematicGenerator[GenericActionableGuidelineMatchesSchema]],
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
        criticality=Criticality.MEDIUM,
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

    guideline_actionable_matcher = GenericActionableGuidelineMatchingBatch(
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


async def test_that_a_guideline_whose_condition_is_partially_satisfied_not_matched(
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

    guidelines: list[str] = ["first_order_and_order_more_than_2"]

    await base_test_that_correct_guidelines_are_matched(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


async def test_that_guideline_whose_condition_was_partially_fulfilled_now_matches(
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
        (
            EventSource.AI_AGENT,
            "Cool so I will process your order right away. Anything else?",
        ),
        (
            EventSource.CUSTOMER,
            "Actually I want another pizza please.",
        ),
    ]

    guidelines: list[str] = ["first_order_and_order_more_than_2"]

    await base_test_that_correct_guidelines_are_matched(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_guideline_whose_condition_was_initially_not_fulfilled_now_matches(
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
            "I want 3 pizzas please",
        ),
        (
            EventSource.AI_AGENT,
            "Cool so I will process your order right away. Anything else?",
        ),
        (
            EventSource.CUSTOMER,
            "Actually I want 2 pizzas please.",
        ),
    ]

    guidelines: list[str] = ["first_order_and_order_exactly_2"]

    await base_test_that_correct_guidelines_are_matched(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_guideline_whose_condition_was_initially_not_fulfilled_now_matches_with_subtopic(
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
            "I want 3 pizzas please",
        ),
        (
            EventSource.AI_AGENT,
            "Cool so I will process your order right away. Anything else?",
        ),
        (
            EventSource.CUSTOMER,
            "I went to this other pizza place and they had some great pizza/",
        ),
        (
            EventSource.AI_AGENT,
            "Happy to hear that! We also have some great pizzas here. Would you like anything else?",
        ),
        (
            EventSource.CUSTOMER,
            "Actually I want 2 pizzas please.",
        ),
    ]

    guidelines: list[str] = ["first_order_and_order_exactly_2"]

    await base_test_that_correct_guidelines_are_matched(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_guideline_whose_condition_was_initially_not_fulfilled_now_matches_after_long_conversation(
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
            "Can you tell me about your menu?",
        ),
        (
            EventSource.AI_AGENT,
            "Our menu includes a variety of pizzas, sandwiches, and drinks. What are you in the mood for?",
        ),
        (
            EventSource.CUSTOMER,
            "When was this place opened?",
        ),
        (
            EventSource.AI_AGENT,
            "We opened in 2020. Would you like to order something?",
        ),
        (EventSource.CUSTOMER, "Are you guys open on weekends?"),
        (EventSource.AI_AGENT, "Yes, we are open on weekends. What would you like to order?"),
        (
            EventSource.CUSTOMER,
            "I want 2 pizzas please",
        ),
        (
            EventSource.AI_AGENT,
            "Cool so I will process your order right away. Anything else?",
        ),
        (
            EventSource.CUSTOMER,
            "Actually I want another pizza please.",
        ),
    ]

    guidelines: list[str] = ["first_order_and_order_more_than_2"]

    await base_test_that_correct_guidelines_are_matched(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_conflicting_actions_with_similar_conditions_are_both_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Look it's been over an hour and my problem was not solved. You are not helping and "
            "I want to talk with a manager immediately!",
        ),
    ]

    guidelines: list[str] = ["transfer_to_manager", "don't_transfer_to_manager"]

    await base_test_that_correct_guidelines_are_matched(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_guideline_with_already_applied_condition_but_unaddressed_action_is_not_matched_when_conversation_was_drifted(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            " Hi, can you help me cancel my subscription?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure, I can walk you through the process. Are you using the mobile app or the website?",
        ),
        (
            EventSource.CUSTOMER,
            "Actually, before that — how do I change my billing address?",
        ),
    ]

    guidelines: list[str] = ["cancel_subscription"]

    await base_test_that_correct_guidelines_are_matched(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


async def test_that_guideline_with_already_applied_condition_but_unaddressed_action_is_not_matched_when_conversation_was_drifted_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, the app keeps crashing on my phone.",
        ),
        (
            EventSource.AI_AGENT,
            "Sorry to hear that! Can you tell me a bit more about what you were doing when it crashed?",
        ),
        (
            EventSource.CUSTOMER,
            "Sure, but can you help me back up my data first?",
        ),
    ]

    guidelines: list[str] = ["identify_problem"]

    await base_test_that_correct_guidelines_are_matched(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


async def test_that_guideline_with_already_matched_condition_but_unaddressed_action_is_matched(
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
    guidelines: list[str] = ["frustrated_customer"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_guideline_is_still_matched_when_conversation_still_on_the_same_topic_that_made_condition_hold(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hey can I order 2 cheese pizzas please?"),
        (
            EventSource.AI_AGENT,
            "Sure! would you like a drink with that?",
        ),
        (
            EventSource.CUSTOMER,
            "No, thanks. How can I pay?",
        ),
        (
            EventSource.AI_AGENT,
            "It will cost $20.9. Could you please provide your credit card number?",
        ),
        (
            EventSource.CUSTOMER,
            "Sure, it's 1111 2222 3333 4444.",
        ),
    ]
    guidelines: list[str] = ["do_payment"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_guideline_is_still_matched_when_conversation_still_on_sub_topic_that_made_condition_hold(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hi, I just received my order, and the pizza is cold."),
        (
            EventSource.AI_AGENT,
            "I'm so sorry to hear that. Could you tell me more about the issue?",
        ),
        (EventSource.CUSTOMER, "Yeah, it's not just cold — the box was crushed too."),
        (EventSource.AI_AGENT, "That's really unacceptable. Let me make this right."),
        (EventSource.CUSTOMER, "And this isn’t the first time, honestly."),
    ]
    guidelines: list[str] = ["problem_with_order"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_guideline_is_still_matched_when_conversation_still_on_sub_topic_that_made_condition_hold_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I wanted to order a sandwich",
        ),
        (
            EventSource.AI_AGENT,
            "Hello there! We currently have either PB&J or cream cheese, which one would you like",
        ),
        (EventSource.CUSTOMER, "What's lower on calories, PB&J or cream cheese?"),
    ]
    guidelines: list[str] = ["ordering_sandwich"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_that_previously_applied_guidelines_are_matched_based_on_capabilities(
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
            "Set my password to 1234",
        ),
    ]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=["unsupported_capability"],
        guidelines_names=["unsupported_capability"],
        capabilities=capabilities,
    )


async def test_that_previously_applied_guidelines_are_not_matched_based_on_irrelevant_capabilities(
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
            "I want to reset my password",
        ),
    ]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=[],
        guidelines_names=["unsupported_capability", "multiple_capabilities"],
        capabilities=capabilities,
    )
