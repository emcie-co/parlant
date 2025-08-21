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
from parlant.core.agents import Agent, AgentStore, CompositionMode
from parlant.core.canned_responses import CannedResponseId
from parlant.core.common import generate_id
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent, EventEmitterFactory
from parlant.core.engines.alpha.canned_response_generator import (
    _CannedResponseSelectionResult,
    CannedResponseContext,
    CannedResponseGenerator,
    SupplementalCannedResponseSelectionSchema,
)
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import EventSource, Session, SessionId, SessionStore
from parlant.core.tags import TagId
from tests.core.common.utils import create_event_message
from tests.test_utilities import SyncAwaiter


@dataclass
class ContextOfTest:
    container: Container
    sync_await: SyncAwaiter
    schematic_generator: SchematicGenerator[SupplementalCannedResponseSelectionSchema]
    logger: Logger


@fixture
def context(
    sync_await: SyncAwaiter,
    container: Container,
) -> ContextOfTest:
    return ContextOfTest(
        sync_await=sync_await,
        container=container,
        schematic_generator=container[
            SchematicGenerator[SupplementalCannedResponseSelectionSchema]
        ],
        logger=container[Logger],
    )


@fixture
def agent(
    context: ContextOfTest,
) -> Agent:
    store = context.container[AgentStore]
    agent = context.sync_await(
        store.create_agent(
            name="test-agent",
            max_engine_iterations=2,
            composition_mode=CompositionMode.CANNED_STRICT,
        )
    )
    return agent


def create_guideline(
    content: GuidelineContent,
    tags: list[TagId] = [],
) -> Guideline:
    guideline = Guideline(
        id=GuidelineId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        content=GuidelineContent(
            condition=content.condition,
            action=content.action,
        ),
        enabled=True,
        tags=tags,
        metadata={},
    )

    return guideline


async def base_test_that_correct_canrep_is_selected(
    context: ContextOfTest,
    agent: Agent,
    session_id: SessionId,
    customer: Customer,
    canned_responses_text: Sequence[str],
    last_generation_draft: str,
    target_canned_response: str | None,
    conversation_context: list[tuple[EventSource, str]],
    guidelines: list[GuidelineContent] = [],
    staged_events: Sequence[EmittedEvent] = [],
) -> None:
    # Create response context
    canned_responses = [
        (CannedResponseId(str(i)), canrep_text)
        for i, canrep_text in enumerate(canned_responses_text, start=1)
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
            correlation_id=e.correlation_id,
            data=e.data,
        )
    last_agent_message = next(
        message
        for source, message in reversed(conversation_context)
        if source == EventSource.AI_AGENT
    )

    event_emitter_factory = context.container[EventEmitterFactory]
    event_emitter = await event_emitter_factory.create_event_emitter(
        emitting_agent_id=agent.id,
        session_id=session_id,
    )

    guideline_matches = [
        GuidelineMatch(
            guideline=create_guideline(gc),
            score=10,
            rationale="This guideline was activated based on the current state of the interaction",
            metadata={},
        )
        for gc in guidelines
    ]

    supplemental_canrep_context = CannedResponseContext(
        event_emitter=event_emitter,
        agent=agent,
        customer=customer,
        context_variables=[],
        interaction_history=interaction_history,
        terms=[],  # TODO maybe add glossary tests
        capabilities=[],
        ordinary_guideline_matches=guideline_matches,
        tool_enabled_guideline_matches={},
        journeys=[],
        tool_insights=ToolInsights(evaluations=[], missing_data=[], invalid_data=[]),
        staged_tool_events=[],
        staged_message_events=[],
    )  # TODO ask Dor if it's ok to call this private class in the test

    last_response_generation = _CannedResponseSelectionResult(
        message=last_agent_message,
        draft=last_generation_draft,
        canned_responses=canned_responses,
    )

    canrep_generator: CannedResponseGenerator = context.container[CannedResponseGenerator]

    _, response = await canrep_generator.generate_supplemental_response(
        context=supplemental_canrep_context,
        last_response_generation=last_response_generation,
    )
    if target_canned_response:
        assert response and response.message == target_canned_response
    else:
        assert response is None


async def test_that_simple_correct_supplemental_canned_response_is_chosen_1(
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
            "What toppings would you like?",
        ),
    ]

    last_generation_draft = "Great! What kind of crust and toppings would you like?"
    canned_responses: list[str] = [
        "What kind of crust would you like?",
        "What is your delivery address?",
        "What kind of topping would you like for each of your pizzas?",
        "All of pizzas are peanut-free",
        "Sorry, I didn't catch that. Could you try phrasing that another way?",
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=canned_responses[0],
        conversation_context=conversation_context,
    )


async def test_that_simple_correct_supplemental_canned_response_is_chosen_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I'm on your app trying to report a problem with a transaction but I can't",
        ),
        (
            EventSource.AI_AGENT,
            "I see. Let me check that for you.",
        ),
        (
            EventSource.AI_AGENT,
            "What is the issue with the transaction?",
        ),
    ]

    last_generation_draft = "Sorry to hear that. What is the problem you're trying to report? There's a 'report problem' button on your transactions dashboard that lets you quickly flag issues with specific transactions. Can you find it?"
    canned_responses: list[str] = [
        "I'm sorry but I didn't catch that.",
        "What is the problem you're experiencing?",
        "You can report a problem with a transaction by clicking on the 'report problem' button on your transactions dashboard.",
        "Please report your problem using the 'report problem' button",
        "The capital of France is Paris",
        "Disputing a transaction can be done through the transactions dashboard on our app or through this chat",
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=canned_responses[2],
        conversation_context=conversation_context,
    )


async def test_that_simple_correct_supplemental_canned_response_is_chosen_3(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Are you the bot that helps you remember song lyrics?",
        ),
        (
            EventSource.AI_AGENT,
            "Yes, that's me!",
        ),
        (
            EventSource.CUSTOMER,
            "What's that song that goes daboodeedaboodee",
        ),
        (
            EventSource.AI_AGENT,
            "I'm sorry but that could apply to multiple songs.",
        ),
        (
            EventSource.AI_AGENT,
            "Do you remember anything else about the song? Such as lyrics or genre",
        ),
        (
            EventSource.CUSTOMER,
            "It's this electronic song with chipmunk-like vocals",
        ),
        (
            EventSource.AI_AGENT,
            "I'm not sure. Could it possibly be the Ketchup song?",
        ),
        (
            EventSource.CUSTOMER,
            "I listened to it but that's not it. I think it starts with 'Im boo...' and then the daboodee part ",
        ),
        (
            EventSource.AI_AGENT,
            "I'm not sure. Could it possibly be the Ketchup song?",
        ),
    ]

    last_generation_draft = "Sorry to hear that. What is the problem you're trying to report? There's a 'report problem' button on your transactions dashboard that lets you quickly flag issues with specific transactions. Can you find it?"
    canned_responses: list[str] = [
        "I'm sorry but I didn't catch that.",
        "What is the problem you're experiencing?",
        "You can report a problem with a transaction by clicking on the 'report problem' button on your transactions dashboard.",
        "Please report your problem using the 'report problem' button",
        "The capital of France is Paris",
        "Disputing a transaction can be done through the transactions dashboard on our app or through this chat",
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=canned_responses[2],
        conversation_context=conversation_context,
    )


async def test_that_most_crucial_supplemental_canned_response_is_selected_when_multiple_canned_responses_apply(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I'm having trouble with my online banking login",
        ),
        (
            EventSource.AI_AGENT,
            "I can help you with that. What specific issue are you experiencing?",
        ),
        (
            EventSource.CUSTOMER,
            "I keep getting locked out after entering my password wrong",
        ),
        (
            EventSource.AI_AGENT,
            "That's frustrating. Let me help you resolve this login issue.",
        ),
        (
            EventSource.CUSTOMER,
            "Yes please, I need to check my account urgently",
        ),
        (
            EventSource.AI_AGENT,
            "I can unlock your account right now through this chat system.",
        ),
    ]

    last_generation_draft = "I can unlock your account right now through this chat system. After I unlock it, please wait 15 minutes before trying to log in again to ensure the system updates properly. To verify your identity, can you provide me with your account recovery number?"
    canned_responses: list[str] = [
        "Please allow up to 15 minutes for account changes to take effect before attempting to log in.",
        "Can you please provide me with your account recovery number?",
        "Your account has been successfully unlocked and is ready for use.",
        "For security reasons, we recommend changing your password after regaining access.",
        "You can also visit any branch location for in-person assistance with account issues.",
        "Sorry, I need more information to help you with that request.",
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=canned_responses[1],
        conversation_context=conversation_context,
    )
