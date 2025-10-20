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

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from lagom import Container
from pytest import fixture
from parlant.core.agents import Agent
from parlant.core.capabilities import Capability, CapabilityId
from parlant.core.common import generate_id
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingContext,
)
from parlant.core.engines.alpha.guideline_matching.generic.guideline_actionable_batch import (
    GenericActionableGuidelineMatchesSchema,
    GenericActionableGuidelineMatchingBatch,
)
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import Logger
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
        "condition": "When this is the customer first order and they order more than 2 pizzas",
        "action": "offer 2 for 1 sale",
    },
    "first_order_and_order_exactly_2": {
        "condition": "When this is the customer first order and they order exactly 2 pizzas",
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
            correlation_id=e.correlation_id,
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


async def test_that_guideline_are_not_matched_when_there_is_no_reason(
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
            "Sorry to hear that! Let’s try restarting the app and clearing the cache.",
        ),
        (
            EventSource.CUSTOMER,
            "It worked. That was so annoying!",
        ),
    ]

    guidelines: list[str] = ["frustrated_so_discount"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
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


# ---------------------------
# New 2-step tests for unaddressed flows
# ---------------------------


async def test_info_then_info_subscription(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    convo: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "When does my free trial end?"),
        (EventSource.AI_AGENT, "Your trial ends on Sept 30."),
        (EventSource.CUSTOMER, "Yes, what’s the monthly cost?"),
    ]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context=convo,
        guidelines_target_names=[],
        guidelines_names=[],
    )


async def test_info_then_complaint_food_delivery(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    convo: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "How long does delivery usually take?"),
        (EventSource.AI_AGENT, "Typically 30–40 minutes."),
        (EventSource.CUSTOMER, "It's been over an hour; I want a refund."),
    ]
    guidelines: list[str] = ["frustrated_so_discount"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        convo,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_info_then_inexplicit_flight(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    convo: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "What’s the baggage allowance for economy?"),
        (EventSource.AI_AGENT, "One carry-on and one checked bag up to 23 kg."),
        (EventSource.CUSTOMER, "My ticket shows something different..."),
    ]
    guidelines: list[str] = ["identify_problem"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        convo,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_action_then_inexplicit_banking(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    convo: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Transfer $500 to my savings."),
        (EventSource.AI_AGENT, "Initiated the transfer."),
        (EventSource.CUSTOMER, "Hmm, something doesn’t look right."),
    ]
    guidelines: list[str] = ["identify_problem"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        convo,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_complaint_then_info_retail(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    convo: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "The headphones I bought arrived broken."),
        (EventSource.AI_AGENT, "I'm sorry to hear that — I can help."),
        (EventSource.CUSTOMER, "Do you also sell replacement cables?"),
    ]
    guidelines: list[str] = ["problem_with_order"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        convo,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


async def test_complaint_then_complaint_hotel(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    convo: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "My hotel room wasn’t cleaned today."),
        (EventSource.AI_AGENT, "I’ll arrange housekeeping right away."),
        (EventSource.CUSTOMER, "Also the AC is broken — I expect a discount."),
    ]
    guidelines: list[str] = ["problem_with_order", "frustrated_customer"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        convo,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_complaint_then_action_flight(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    convo: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "My flight was delayed 5 hours; I want compensation."),
        (EventSource.AI_AGENT, "I'm sorry about the delay, I can start the process."),
        (EventSource.CUSTOMER, "Please rebook me for tomorrow morning."),
    ]
    guidelines: list[str] = ["frustrated_so_discount", "rebook_reservation"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        convo,
        guidelines_target_names=["rebook_reservation"],
        guidelines_names=guidelines,
    )


async def test_inexplicit_then_info_tech(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    convo: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "I can’t seem to log in."),
        (EventSource.AI_AGENT, "Are you seeing an error message?"),
        (EventSource.CUSTOMER, "Yes, it says ‘invalid credentials’. What does that mean?"),
    ]
    guidelines: list[str] = ["problem_so_restart"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        convo,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_inexplicit_then_complaint_food(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    convo: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "My order doesn’t look right…"),
        (EventSource.AI_AGENT, "Could you tell me what’s missing?"),
        (EventSource.CUSTOMER, "This is the third time — I want a refund."),
    ]
    guidelines: list[str] = ["problem_with_order", "frustrated_so_discount"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        convo,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_inexplicit_then_action_subscription(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    convo: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "The app keeps crashing."),
        (EventSource.AI_AGENT, "When does it usually happen?"),
        (EventSource.CUSTOMER, "Just cancel my subscription — I’m done with this."),
    ]
    guidelines: list[str] = ["cancel_subscription"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        convo,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


async def test_inexplicit_then_inexplicit_banking(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    convo: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Something’s off with my account balance."),
        (EventSource.AI_AGENT, "Which transactions seem incorrect?"),
        (EventSource.CUSTOMER, "One of the ATM withdrawals didn’t register properly."),
    ]
    guidelines: list[str] = ["identify_problem"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        convo,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


# 2-002  info → complaint (frustration)  → match "frustrated_so_discount"
async def test_info_then_complaint_triggers_frustration_discount(
    context: ContextOfTest, agent: Agent, new_session: Session, customer: Customer
) -> None:
    conversation_context = [
        (EventSource.CUSTOMER, "Hi, what are your pizza sizes?"),
        (EventSource.AI_AGENT, "Small, medium, large. Anything else I can help with?"),
        (EventSource.CUSTOMER, "This took forever to get an answer. I'm really annoyed."),
    ]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=["frustrated_so_discount"],
        guidelines_names=["frustrated_so_discount"],
    )


# 2-004  info → inexplicit issue probing  → match "identify_problem"
async def test_info_then_inexplicit_issue_triggers_identify_problem(
    context: ContextOfTest, agent: Agent, new_session: Session, customer: Customer
) -> None:
    conversation_context = [
        (EventSource.CUSTOMER, "Which plans do you offer?"),
        (EventSource.AI_AGENT, "Basic and Premium. Which one interests you?"),
        (EventSource.CUSTOMER, "The app is acting weird lately..."),
    ]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=["identify_problem"],
        guidelines_names=["identify_problem"],
    )


# 2-005  complaint about an order  → match "problem_with_order"
async def test_complaint_about_order_triggers_problem_with_order(
    context: ContextOfTest, agent: Agent, new_session: Session, customer: Customer
) -> None:
    conversation_context = [
        (EventSource.CUSTOMER, "My pizza arrived cold and the box is crushed."),
    ]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=["problem_with_order"],
        guidelines_names=["problem_with_order"],
    )


# (ARQ: hallucination prevention) Keyword trap:
# mention of "manager" without asking to speak → match NONE of manager guidelines
async def test_no_manager_activation_on_keyword_without_request(
    context: ContextOfTest, agent: Agent, new_session: Session, customer: Customer
) -> None:
    conversation_context = [
        (EventSource.CUSTOMER, "My manager friend loved your service.")  # no escalation request
    ]
    guidelines = ["transfer_to_manager", "don't_transfer_to_manager"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


# (ARQ: intent precision) Delivery ETA vs. business hours boundary
async def test_delivery_time_inquiry_matches_only_true_eta_questions(
    context: ContextOfTest, agent: Agent, new_session: Session, customer: Customer
) -> None:
    # True ETA → matches
    convo_eta = [(EventSource.CUSTOMER, "What's the estimated delivery time for my order?")]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        convo_eta,
        guidelines_target_names=["delivery_time_inquiry"],
        guidelines_names=["delivery_time_inquiry"],
    )
    # Near-miss (hours) → does NOT match
    convo_hours = [(EventSource.CUSTOMER, "Are you open at noon?")]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        convo_hours,
        guidelines_target_names=[],
        guidelines_names=["delivery_time_inquiry"],
    )


# (ARQ: precise matching; sibling conditions)
# First order with exactly 2 vs. more than 2 present → match only the exact-2 guideline
async def test_exactly_2_wins_over_more_than_2_when_both_present(
    context: ContextOfTest, agent: Agent, new_session: Session, customer: Customer
) -> None:
    conversation_context = [
        (EventSource.CUSTOMER, "It's my first time here."),
        (EventSource.AI_AGENT, "Welcome! What would you like?"),
        (EventSource.CUSTOMER, "I want exactly 2 pizzas, no more."),
    ]
    guidelines = ["first_order_and_order_more_than_2", "first_order_and_order_exactly_2"]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=["first_order_and_order_exactly_2"],
        guidelines_names=guidelines,
    )


# 2-012  action request → inexplicit issue probing (banking) → match "identify_problem"
async def test_action_then_inexplicit_triggers_identify_problem(
    context: ContextOfTest, agent: Agent, new_session: Session, customer: Customer
) -> None:
    conversation_context = [
        (EventSource.CUSTOMER, "Refund my last payment."),
        (EventSource.AI_AGENT, "I can help. Could you share the transaction ID?"),
        (EventSource.CUSTOMER, "It just fails with some error."),
    ]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=["identify_problem"],
        guidelines_names=["identify_problem"],
    )


# (ARQ: multi-capability guardrail; do not guess) → match "multiple_capabilities"
async def test_multiple_capabilities_triggers_clarification_when_user_requests_two_actions(
    context: ContextOfTest, agent: Agent, new_session: Session, customer: Customer
) -> None:
    capabilities = [
        Capability(
            id=CapabilityId("cap_pay"),
            creation_utc=datetime.now(timezone.utc),
            title="Pay Bill",
            description="Pay a bill",
            signals=["pay bill", "payment"],
            tags=[],
        ),
        Capability(
            id=CapabilityId("cap_balance"),
            creation_utc=datetime.now(timezone.utc),
            title="Check Balance",
            description="Check account balance",
            signals=["check balance", "balance"],
            tags=[],
        ),
    ]
    conversation_context = [
        (EventSource.CUSTOMER, "I want to pay my bill and also see my balance.")
    ]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=["multiple_capabilities"],
        guidelines_names=["multiple_capabilities"],
        capabilities=capabilities,
    )


# 2-013  action → complaint combo (order issue arises following an action)
async def test_action_then_complaint_about_order_triggers_problem_with_order(
    context: ContextOfTest, agent: Agent, new_session: Session, customer: Customer
) -> None:
    conversation_context = [
        (EventSource.CUSTOMER, "Place an order for a large veggie pizza."),
        (EventSource.AI_AGENT, "Done. Your order is on its way."),
        (EventSource.CUSTOMER, "It arrived cold and half the toppings are missing."),
    ]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=["problem_with_order", "frustrated_so_discount"],
        guidelines_names=["problem_with_order", "frustrated_so_discount"],
    )


# 2-016  inexplicit → inexplicit (issue persists / still unclear) → match "identify_problem"
async def test_inexplicit_then_inexplicit_still_triggers_identify_problem(
    context: ContextOfTest, agent: Agent, new_session: Session, customer: Customer
) -> None:
    conversation_context = [
        (EventSource.CUSTOMER, "The app seems off lately."),
        (EventSource.AI_AGENT, "I can help—what exactly is wrong?"),
        (EventSource.CUSTOMER, "It just does weird stuff sometimes."),
    ]
    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=["identify_problem"],
        guidelines_names=["identify_problem"],
    )
