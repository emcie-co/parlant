from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast
from lagom import Container
from pytest import fixture
from parlant.core.agents import Agent
from parlant.core.common import JSONSerializable, generate_id
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import GuidelineMatchingContext
from parlant.core.engines.alpha.guideline_matching.generic_guideline_previously_applied_batch import (
    GenericPreviouslyAppliedGuidelineMatchesSchema,
    GenericPreviouslyAppliedGuidelineMatchingBatch,
)
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import EventKind, EventSource
from parlant.core.tags import TagId
from tests.core.common.utils import create_event_message
from tests.test_utilities import SyncAwaiter


GUIDELINES_DICT = {
    "problem_so_restart": {
        "condition": "The customer has a problem with the app anc haven't tried anything yet",
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
}


@dataclass
class ContextOfTest:
    container: Container
    sync_await: SyncAwaiter
    guidelines: list[Guideline]
    schematic_generator: SchematicGenerator[GenericPreviouslyAppliedGuidelineMatchesSchema]
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
            SchematicGenerator[GenericPreviouslyAppliedGuidelineMatchesSchema]
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

    guideline_matching_context = GuidelineMatchingContext(
        agent=agent,
        customer=customer,
        context_variables=[],
        interaction_history=interaction_history,
        terms=[],
        staged_events=staged_events,
    )

    guideline_previously_applied_detector = GenericPreviouslyAppliedGuidelineMatchingBatch(
        logger=context.container[Logger],
        schematic_generator=context.schematic_generator,
        guidelines=context.guidelines,
        context=guideline_matching_context,
    )

    result = context.sync_await(guideline_previously_applied_detector.process())

    matched_guidelines = [p.guideline for p in result.matches]

    assert set(matched_guidelines) == set(previously_applied_target_guidelines)


def test_that_previously_matched_guideline_not_matched_when_there_is_no_new_reason(
    context: ContextOfTest,
    agent: Agent,
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
            "That worked! By the way, how can I upgrade to the premium plan?",
        ),
    ]

    guidelines: list[str] = ["problem_so_restart"]

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context,
        agent,
        customer,
        conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )


def test_that_guideline_action_that_was_performed_but_result_in_an_error_is_matched_again(
    context: ContextOfTest,
    agent: Agent,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, can you reset my password?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure, for that I will need your email please so I will send you the password",
        ),
        (
            EventSource.CUSTOMER,
            "111@emcie.co",
        ),
    ]
    tool_result = cast(
        JSONSerializable,
        {
            "tool_calls": [
                {
                    "tool_id": "local:reset_password",
                    "arguments": {"email": "111@emcie.co"},
                    "result": {
                        "data": ["error with reset - unknown mail"],
                        "metadata": {},
                        "control": {},
                    },
                }
            ]
        },
    )

    staged_events = [
        EmittedEvent(
            source=EventSource.AI_AGENT, kind=EventKind.TOOL, correlation_id="", data=tool_result
        ),
    ]

    guidelines: list[str] = ["reset_password"]

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context,
        agent,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
        staged_events=staged_events,
    )


def test_that_guideline_action_that_was_performed_but_result_in_undesired_user_response_is_matched_again(
    context: ContextOfTest,
    agent: Agent,
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
            "I think I have an email address but let me check what it is.",
        ),
    ]

    guidelines: list[str] = ["reset_password"]

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context,
        agent,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
    )


def test_that_partially_fulfilled_action_but_cosmetic_is_not_match_again(
    context: ContextOfTest,
    agent: Agent,
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

    base_test_that_correct_guidelines_detect_as_previously_applied(
        context,
        agent,
        customer,
        conversation_context,
        guidelines_target_names=[],
        guidelines_names=guidelines,
    )
