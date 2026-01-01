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

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from lagom import Container
from pytest import fixture

from parlant.core.agents import Agent
from parlant.core.capabilities import Capability
from parlant.core.common import Criticality, JSONSerializable, generate_id
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableId,
    ContextVariableValue,
    ContextVariableValueId,
)
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.generic.disambiguation_batch import (
    DisambiguationGuidelineMatchesSchema,
    GenericDisambiguationGuidelineMatchingBatch,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matching_context import (
    GuidelineMatchingContext,
)
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.evaluations import GuidelinePayload, PayloadOperation
from parlant.core.journeys import JourneyStore
from parlant.core.glossary import Term, TermId
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.indexing.behavioral_change_evaluation import GuidelineEvaluator
from parlant.core.sessions import EventSource, Session
from parlant.core.tags import TagId
from tests.core.common.utils import create_event_message
from tests.test_utilities import SyncAwaiter, nlp_test


@dataclass
class ContextOfTest:
    container: Container
    sync_await: SyncAwaiter
    schematic_generator: SchematicGenerator[DisambiguationGuidelineMatchesSchema]
    logger: Logger


@fixture
def context(
    sync_await: SyncAwaiter,
    container: Container,
) -> ContextOfTest:
    return ContextOfTest(
        container,
        sync_await,
        logger=container[Logger],
        schematic_generator=container[SchematicGenerator[DisambiguationGuidelineMatchesSchema]],
    )


GUIDELINES_DICT = {
    "snake_roller_coaster": {
        "condition": "the customer asks for the snake roller coaster",
        "action": "book it",
    },
    "turtle_roller_coaster": {
        "condition": "the customer asks for the turtle roller coaster",
        "action": "book it",
    },
    "tiger_Ferris_wheel": {
        "condition": "the customer asks for the tiger Ferris wheel",
        "action": "book it",
    },
    "adult_colliding_cars": {
        "condition": "the customer asks for adult colliding cars",
        "action": "book it",
    },
    "children_colliding_cars": {
        "condition": "the customer asks for children colliding cars",
        "action": "book it",
    },
    "report_lost": {
        "condition": "the customer wants to report a card lost",
        "action": "report card lost",
    },
    "lock_card": {
        "condition": "the customer wants to lock their card",
        "action": "do locking",
    },
    "replacement_card": {
        "condition": "the customer requests a replacement card",
        "action": "order them a new card",
    },
    "freeze_card": {
        "condition": "the customer wants to freeze their card (temporary lock)",
        "action": "freeze their card",
    },
    "report_stealing": {
        "condition": "the customer wants to report a stolen card",
        "action": "report a stolen card",
    },
    "report_to_police": {
        "condition": "the customer wants to file a police report",
        "action": "file a police report",
    },
    "dispute_charge": {
        "condition": "the customer wants to dispute an unknown charge",
        "action": "dispute the unknown charge",
    },
    "vip_refund": {
        "condition": "the customer is VIP and they ask for a refund on a flight to original payment method or to travel credit",
        "action": "Do a full refund to original payment method or travel credit",
    },
    "vip_reschedule": {
        "condition": "the customer is VIP and they ask for rescheduling the flight",
        "action": "Do free rescheduling",
    },
    "vip_cancel": {
        "condition": "the customer is VIP and they ask to fully cancel the flight",
        "action": "Do free cancelling",
    },
    "regular_refund_travel_credit": {
        "condition": "the customer is regular and ask for a refund on a flight to travel credit",
        "action": "Refund as travel credit with a fee",
    },
    "regular_reschedule": {
        "condition": "the customer is regular and they ask for rescheduling the flight",
        "action": "do rescheduling with a fee",
    },
    "regular_cancel": {
        "condition": "the customer is regular and they ask to cancel the flight",
        "action": "do cancelling with a fee",
    },
    "CoreTrace": {
        "condition": "The customer asks to submit a CoreTrace",
        "action": "submit a CoreTrace",
    },
    "QuickPatch": {
        "condition": "The customer asks to activate QuickPatch",
        "action": "activate QuickPatch",
    },
    "FixFlow": {
        "condition": "The customer asks to start a FixFlow session",
        "action": "start a FixFlow session",
    },
}

CONDITION_HEAD_DICT = {
    "amusement_park": "The customer asks to book a ticket to an amusement ride or attraction, and its not clear which one",
    "lost_card": "The customer lost their card and didn't specify what they want to do",
    "stolen_card": "The customer indicates that their card was stolen and didn't specify what they want to do",
    "cancel_flight": "The customer if asks to make a change in booked flight but doesn’t specify whether they want to reschedule, request a refund, or fully cancel the booking",
    "fix_bug": "The customer has a technical problem, and they didn't specify what kind of help they want to have",
}


def create_term(
    name: str, description: str, synonyms: list[str] = [], tags: list[TagId] = []
) -> Term:
    return Term(
        id=TermId("-"),
        creation_utc=datetime.now(timezone.utc),
        name=name,
        description=description,
        synonyms=synonyms,
        tags=tags,
    )


def create_context_variable(
    name: str,
    data: JSONSerializable,
    tags: list[TagId],
) -> tuple[ContextVariable, ContextVariableValue]:
    return ContextVariable(
        id=ContextVariableId("-"),
        creation_utc=datetime.now(timezone.utc),
        name=name,
        description="",
        tool_id=None,
        freshness_rules=None,
        tags=tags,
    ), ContextVariableValue(
        id=ContextVariableValueId("-"),
        last_modified=datetime.now(timezone.utc),
        data=data,
    )


async def create_guideline(
    context: ContextOfTest,
    condition: str,
    action: str | None = None,
    tags: list[TagId] = [],
) -> Guideline:
    metadata: dict[str, JSONSerializable] = {}
    if action:
        guideline_evaluator = context.container[GuidelineEvaluator]
        guideline_evaluation_data = await guideline_evaluator.evaluate(
            payloads=[
                GuidelinePayload(
                    content=GuidelineContent(
                        condition=condition,
                        action=action,
                    ),
                    tool_ids=[],
                    operation=PayloadOperation.ADD,
                    action_proposition=True,
                    properties_proposition=True,
                    journey_node_proposition=False,
                )
            ],
        )

        metadata = guideline_evaluation_data[0].properties_proposition or {}

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
        metadata=metadata,
    )

    return guideline


async def create_guideline_by_name(
    context: ContextOfTest,
    guideline_name: str,
) -> Guideline | None:
    if guideline_name in GUIDELINES_DICT:
        guideline = await create_guideline(
            context=context,
            condition=GUIDELINES_DICT[guideline_name]["condition"],
            action=GUIDELINES_DICT[guideline_name]["action"],
        )
    else:
        guideline = None
    return guideline


async def base_test_that_ambiguity_detected_with_relevant_guidelines(
    context: ContextOfTest,
    agent: Agent,
    session: Session,
    customer: Customer,
    conversation_context: list[tuple[EventSource, str]],
    head_condition: str,
    is_ambiguous: bool,
    to_disambiguate_guidelines_names: list[str],
    disambiguating_guideline_names: list[str],
    clarification_must_contain: str = "",
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
    terms: Sequence[Term] = [],
    capabilities: Sequence[Capability] = [],
    staged_events: Sequence[EmittedEvent] = [],
) -> None:
    interaction_history = [
        create_event_message(
            offset=i,
            source=source,
            message=message,
        )
        for i, (source, message) in enumerate(conversation_context)
    ]

    to_disambiguate_guidelines = {
        name: await create_guideline_by_name(context, name)
        for name in to_disambiguate_guidelines_names
    }
    to_ids = {g.id: g for g in to_disambiguate_guidelines.values() if g is not None}

    guideline_head = await create_guideline(
        context=context,
        condition=head_condition,
    )

    guideline_targets = [g for g in to_disambiguate_guidelines.values() if g is not None]

    disambiguating_guideline = [
        guideline
        for name in disambiguating_guideline_names
        if (guideline := to_disambiguate_guidelines.get(name)) is not None
    ]

    guideline_matching_context = GuidelineMatchingContext(
        agent,
        session,
        customer,
        context_variables,
        interaction_history,
        terms,
        capabilities,
        staged_events,
        active_journeys=[],
        journey_paths={k: list(v) for k, v in session.agent_states[-1].journey_paths.items()}
        if session.agent_states
        else {},
    )

    disambiguation_resolver = GenericDisambiguationGuidelineMatchingBatch(
        logger=context.logger,
        meter=context.container[Meter],
        journey_store=context.container[JourneyStore],
        optimization_policy=context.container[OptimizationPolicy],
        schematic_generator=context.schematic_generator,
        disambiguation_guideline=guideline_head,
        disambiguation_targets=guideline_targets,
        context=guideline_matching_context,
    )
    result = await disambiguation_resolver.process()

    assert (result.matches[0].score == 10) == is_ambiguous

    data = result.matches[0].metadata
    if data and isinstance(data, dict):
        if is_ambiguous:
            disambiguation = data.get("disambiguation")
            assert disambiguation, "Disambiguation key missing or falsy"

            if isinstance(disambiguation, dict):
                targets = disambiguation.get("targets")
                if targets:
                    guideline_targets = [to_ids[id] for id in targets]
                    assert set(disambiguating_guideline) == set(guideline_targets)

                clarification = disambiguation.get("enriched_action")
                if clarification:
                    assert await nlp_test(
                        context=f"Here's a clarification message in the form of ask the customer something: {clarification}",
                        condition=f"The message contains {clarification_must_contain}",
                    ), (
                        f"clarification message: '{clarification}', expected to contain: '{clarification_must_contain}'"
                    )


# TODO : allow skipping guidelines


async def test_that_ambiguity_is_not_detected_when_not_needed_based_on_earlier_part_of_the_conversation(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi which roller coasters are currently running?",
        ),
        (
            EventSource.AI_AGENT,
            "Right now, only the Snake roller coaster is active. We also have other rides, like the Tiger ferris wheel.",
        ),
        (
            EventSource.CUSTOMER,
            "Ok so book me to the first one please",
        ),
    ]
    to_disambiguate_guidelines = [
        "snake_roller_coaster",
        "turtle_roller_coaster",
        "tiger_Ferris_wheel",
    ]
    disambiguating_guidelines: list[str] = []
    head_condition = CONDITION_HEAD_DICT["amusement_park"]
    await base_test_that_ambiguity_detected_with_relevant_guidelines(
        context,
        agent,
        new_session,
        customer,
        conversation_context,
        head_condition,
        is_ambiguous=False,
        to_disambiguate_guidelines_names=to_disambiguate_guidelines,
        disambiguating_guideline_names=disambiguating_guidelines,
    )


# TODO: problematic test, decide how to change
async def test_that_ambiguity_detects_with_relevant_guidelines_based_on_glossary(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, my screen just goes black after I open the app. I don’t really know what happened — I didn’t touch anything in the settings. It worked yesterday."
            "I have no technical knowledge",
        ),
    ]
    terms = [
        create_term(
            name="FixFlow",
            description="A live, guided troubleshooting session with a technical agent.",
        ),
        create_term(
            name="CoreTrace",
            description="Relevant when the customer is an engineering- A deeper diagnostic log meant for engineering-level review.",
        ),
        create_term(
            name="QuickPatch",
            description="A remote patching tool that attempts to fix common bugs or corrupted settings.",
        ),
    ]
    to_disambiguate_guidelines = [
        "FixFlow",
        "CoreTrace",
        "QuickPatch",
    ]
    disambiguating_guidelines: list[str] = ["FixFlow", "QuickPatch"]
    head_condition = CONDITION_HEAD_DICT["fix_bug"]
    clarification_must_contain = "FixFlow or QuickPatch as ways to help to solve the problem"
    await base_test_that_ambiguity_detected_with_relevant_guidelines(
        context,
        agent,
        new_session,
        customer,
        conversation_context,
        head_condition,
        is_ambiguous=True,
        to_disambiguate_guidelines_names=to_disambiguate_guidelines,
        disambiguating_guideline_names=disambiguating_guidelines,
        clarification_must_contain=clarification_must_contain,
        terms=terms,
    )


# TODO: test is ok, need to rewrite the nlp test
async def test_that_ambiguity_is_detected_when_previously_applied_and_should_reapply(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Please book me for the roller coaster",
        ),
        (
            EventSource.AI_AGENT,
            "We have a snake roller coaster and turtle roller coaster. Which one would you like?",
        ),
        (
            EventSource.CUSTOMER,
            "Turtle roller coaster sound boring. Book me to the snake one",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! anything else?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes do you have colliding cars right? so that one too",
        ),
    ]
    to_disambiguate_guidelines = [
        "snake_roller_coaster",
        "turtle_roller_coaster",
        "tiger_Ferris_wheel",
        "adult_colliding_cars",
        "children_colliding_cars",
    ]
    disambiguating_guidelines: list[str] = ["children_colliding_cars", "adult_colliding_cars"]
    head_condition = CONDITION_HEAD_DICT["amusement_park"]
    clarification_must_contain = "options to adult colliding cars or children colliding cars"
    await base_test_that_ambiguity_detected_with_relevant_guidelines(
        context,
        agent,
        new_session,
        customer,
        conversation_context,
        head_condition,
        is_ambiguous=True,
        to_disambiguate_guidelines_names=to_disambiguate_guidelines,
        disambiguating_guideline_names=disambiguating_guidelines,
        clarification_must_contain=clarification_must_contain,
    )
