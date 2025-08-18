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
from parlant.core.canned_responses import CannedResponseId
from parlant.core.capabilities import Capability, CapabilityId
from parlant.core.common import generate_id
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent, EventEmitter, EventEmitterFactory
from parlant.core.engines.alpha.canned_response_generator import (
    _CannedResponseSelectionResult,
    CannedResponseContext,
    CannedResponseGenerator,
    SupplementalCannedResponseSelectionSchema,
)
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingContext,
)

from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
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
}


@dataclass
class ContextOfTest:
    container: Container
    sync_await: SyncAwaiter
    canned_responses: list[str]
    guidelines: list[Guideline]
    schematic_generator: SchematicGenerator[SupplementalCannedResponseSelectionSchema]
    logger: Logger


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

    _, response = await canrep_generator.generate_supplemental_canned_response(
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
