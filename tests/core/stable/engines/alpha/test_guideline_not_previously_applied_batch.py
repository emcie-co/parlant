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
from parlant.core.engines.alpha.guideline_matching.generic_guideline_not_previously_applied_batch import (
    GenericNotPreviouslyAppliedGuidelineMatchesSchema,
    GenericNotPreviouslyAppliedGuidelineMatchingBatch,
)
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import EventSource, Session, SessionId, SessionStore
from parlant.core.tags import TagId
from tests.core.common.utils import create_event_message
from tests.test_utilities import SyncAwaiter


GUIDELINES_DICT = {
    "first_order_and_order_more_than_2": {
        "condition": "When this is the customer first order and they order more than 2 pizzas",
        "action": "offer 2 for 1 sale",
    },
    "transfer_to_manager": {
        "condition": "When customer ask to talk with a manager",
        "action": "Hand them over to a manager immediately.",
    },
    "don't_transfer_to_manager": {
        "condition": "When customer ask to talk with a manager",
        "action": "Explain that it's not possible to talk with a manager and that you are here to help",
    },
    "identify_problem": {
        "condition": "When customer say that they got an error in the app",
        "action": "help them identify the source of the problem",
    },
    "frustrated_customer": {
        "condition": "the customer appears frustrated or upset",
        "action": "Acknowledge the customer's concerns, apologize for any inconvenience, and offer a solution or escalate the issue to a supervisor if necessary.",
    },
}


@dataclass
class ContextOfTest:
    container: Container
    sync_await: SyncAwaiter
    guidelines: list[Guideline]
    schematic_generator: SchematicGenerator[GenericNotPreviouslyAppliedGuidelineMatchesSchema]
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
            SchematicGenerator[GenericNotPreviouslyAppliedGuidelineMatchesSchema]
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


def base_test_that_correct_guidelines_detect_as_previously_applied(
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
        context.sync_await(
            context.container[SessionStore].create_event(
                session_id=session_id,
                source=e.source,
                kind=e.kind,
                correlation_id=e.correlation_id,
                data=e.data,
            )
        )

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

    guideline_previously_applied_detector = GenericNotPreviouslyAppliedGuidelineMatchingBatch(
        logger=context.container[Logger],
        schematic_generator=context.schematic_generator,
        guidelines=context.guidelines,
        context=guideline_matching_context,
    )

    result = context.sync_await(guideline_previously_applied_detector.process())

    matched_guidelines = [p.guideline for p in result.matches]

    assert set(matched_guidelines) == set(previously_applied_target_guidelines)


def test_that_guideline_that_its_condition_partially_satisfied_not_matched(
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

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


def test_that_guideline_that_its_condition_was_partially_fulfilled_now_match(
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

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


def test_that_conflicting_actions_with_similar_conditions_are_both_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Look it's been over an hour and my problem was not solve. You are not helping and "
            "I want to talk with a manager immediately!",
        ),
    ]

    guidelines: list[str] = ["transfer_to_manager", "don't_transfer_to_manager"]

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


def test_that_guideline_with_already_applied_condition_but_unaddressed_action_is_not_matched_when_conversation_was_drifted(
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

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


def test_that_guideline_with_already_applied_condition_but_unaddressed_action_is_matched(
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

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


def test_that_guideline_is_still_matched_when_conversation_still_on_the_same_topic_that_made_condition_hold(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    return


def test_that_guideline_is_still_matched_when_conversation_still_on_sub_topic_that_made_condition_hold(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    return
