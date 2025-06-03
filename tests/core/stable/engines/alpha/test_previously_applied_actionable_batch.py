from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from lagom import Container
from pytest import fixture
from parlant.core.agents import Agent
from parlant.core.common import generate_id
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import GuidelineMatchingContext
from parlant.core.engines.alpha.guideline_matching.generic.guideline_previously_applied_actionable_batch import (
    GenericPreviouslyAppliedActionableGuidelineMatchesSchema,
    GenericPreviouslyAppliedActionableGuidelineMatchingBatch,
)
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import EventSource, Session, SessionId, SessionStore
from parlant.core.tags import TagId
from tests.core.common.utils import create_event_message
from tests.test_utilities import SyncAwaiter


GUIDELINES_DICT = {
    "problem_so_restart": {
        "condition": "The customer has a problem with the app and hasn't tried anything yet",
        "action": "Suggest to do restart",
    },
    "reset_password": {
        "condition": "When a customer wants to reset their password",
        "action": "ask for their email address to send them a password",
    },
    "calm_and_reset_password": {
        "condition": "When a customer wants to reset their password",
        "action": "tell them that it's ok and it happens to everyone and ask for their email address to send them a password",
    },
    "frustrated_so_discount": {
        "condition": "The customer expresses frustration, impatience, or dissatisfaction",
        "action": "apologize and offer a discount",
    },
    "confirm_reservation": {
        "condition": "The customer has placed a reservation, submitted an order, or added items to an order.",
        "action": "ask whether the customer would like to add anything else before finalizing the reservation or order",
    },
    "order_status": {
        "condition": "The customer is asking about a status of an order.",
        "action": "retrieve it's status and inform the customer",
    },
    "return_conditions": {
        "condition": "The customer is asking about return terms.",
        "action": "refer them to the company's website",
    },
}


@dataclass
class ContextOfTest:
    container: Container
    sync_await: SyncAwaiter
    guidelines: list[Guideline]
    schematic_generator: SchematicGenerator[
        GenericPreviouslyAppliedActionableGuidelineMatchesSchema
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
            SchematicGenerator[GenericPreviouslyAppliedActionableGuidelineMatchesSchema]
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
    )

    context.guidelines.append(guideline)

    return guideline


def base_test_that_correct_guidelines_are_matched(
    context: ContextOfTest,
    agent: Agent,
    session_id: SessionId,
    customer: Customer,
    conversation_context: list[tuple[EventSource, str]],
    guidelines_target_names: list[str],
    guidelines_names: list[str],
    staged_events: Sequence[EmittedEvent] = [],
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

    session = context.sync_await(context.container[SessionStore].read_session(session_id))

    guideline_matching_context = GuidelineMatchingContext(
        agent=agent,
        session=session,
        customer=customer,
        context_variables=[],
        interaction_history=interaction_history,
        terms=[],
        staged_events=staged_events,
    )

    guideline_previously_applied_matcher = GenericPreviouslyAppliedActionableGuidelineMatchingBatch(
        logger=context.container[Logger],
        schematic_generator=context.schematic_generator,
        guidelines=context.guidelines,
        context=guideline_matching_context,
    )

    result = context.sync_await(guideline_previously_applied_matcher.process())

    matched_guidelines = [p.guideline for p in result.matches]

    assert set(matched_guidelines) == set(target_guidelines)


def test_that_previously_matched_guideline_are_not_matched_when_there_is_no_new_reason(
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
            "I did that but it's crashing!",
        ),
    ]

    guidelines: list[str] = ["problem_so_restart"]

    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


def test_that_partially_fulfilled_action_with_missing_behavioral_part_is_not_matched_again(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, can you reset my password?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure, for that I will need your email please so I will send you the password. What's your email address?",
        ),
        (
            EventSource.CUSTOMER,
            "123@emcie.co",
        ),
    ]

    guidelines: list[str] = ["calm_and_reset_password"]

    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


def test_that_guideline_that_was_reapplied_earlier_and_should_not_reapply_based_on_the_most_recent_interaction_is_not_matched_1(
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
            "I'm really sorry for the delay, and I completely understand how frustrating that must be. I’ll look into it right away, and I can also offer you a discount for the inconvenience.",
        ),
        (
            EventSource.CUSTOMER,
            "OK, thanks. I will be waiting",
        ),
        (
            EventSource.AI_AGENT,
            "Of course. I'm here to help, and I’ll keep you updated as soon as I know more",
        ),
        (
            EventSource.CUSTOMER,
            "I got the delivery now and it's totally broken! Are you serious, you guys? This is ridiculous.",
        ),
        (
            EventSource.AI_AGENT,
            "I'm so sorry—that should absolutely not have happened. I’ll report this right away, and I can offer you a discount for the trouble.",
        ),
        (
            EventSource.CUSTOMER,
            "Thank you that's nice of you.",
        ),
    ]

    guidelines: list[str] = ["frustrated_so_discount"]

    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


def test_that_guideline_that_was_reapplied_earlier_and_should_not_reapply_based_on_the_most_recent_interaction_is_not_matched_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey I haven’t receive my order, I placed it 2 weeks ago.",
        ),
        (
            EventSource.AI_AGENT,
            "Let me check on that for you. Can you provide the order number?",
        ),
        (
            EventSource.CUSTOMER,
            "12233",
        ),
        (
            EventSource.AI_AGENT,
            "Thanks! I see it’s on the way and should arrive this weekend.",
        ),
        (
            EventSource.CUSTOMER,
            "Okay, thanks. I also have another order from a different store—what’s the status of that one?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure, let me take a look. Could you share the order number for that one too?",
        ),
        (
            EventSource.CUSTOMER,
            "I think 111222.",
        ),
        (
            EventSource.AI_AGENT,
            "Hmm, that number doesn’t seem right. Could you double-check it?",
        ),
        (
            EventSource.CUSTOMER,
            "How can I change the address of an order?",
        ),
    ]

    guidelines: list[str] = ["order_status"]

    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


def test_that_guideline_that_was_reapplied_earlier_and_should_reapply_again_based_on_the_most_recent_interaction_is_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I’d like to book a table for 2 at 7 PM tonight.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it — a table for 2 at 7 PM. Would you like to add anything else before I confirm the reservation?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, actually — it’s for a birthday. Can we get a small cake?",
        ),
        (
            EventSource.AI_AGENT,
            "Absolutely! I’ve added a birthday cake to your reservation. Would you like anything else before I send it through?",
        ),
        (
            EventSource.CUSTOMER,
            "Oh, and can we have a table near the window if possible?",
        ),
    ]

    guidelines: list[str] = ["confirm_reservation"]

    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


def test_that_guideline_that_should_reapply_is_matched_when_condition_holds_in_the_last_several_messages(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I’d like to book a table for 2 at 7 PM tonight.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it — a table for 2 at 7 PM. Would you like to add anything else before I confirm the reservation?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, actually — it’s for a birthday. Can we get a small cake? Do you have chocolate cakes?",
        ),
        (
            EventSource.AI_AGENT,
            "Yes we have chocolate and cheese cakes. What would you want?",
        ),
        (
            EventSource.CUSTOMER,
            "Great so add one chocolate cake please.",
        ),
    ]

    guidelines: list[str] = ["confirm_reservation"]

    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


def test_that_reapplied_guideline_is_still_applied_when_handling_conditions_subissue(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I’d like to book a table for 2 at 7 PM tonight.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it — a table for 2 at 7 PM. Would you like to add anything else before I confirm the reservation?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, actually — it’s for a birthday. Can we get a small cake? Do you have chocolate cakes?",
        ),
        (
            EventSource.AI_AGENT,
            "Yes we have chocolate and cheese cakes. What would you want?",
        ),
        (
            EventSource.CUSTOMER,
            "Great so add one chocolate cake please.",
        ),
    ]

    guidelines: list[str] = ["confirm_reservation"]

    base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )
