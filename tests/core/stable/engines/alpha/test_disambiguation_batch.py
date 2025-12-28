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
from parlant.core.glossary import Term, TermId
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.journeys import JourneyStore
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.indexing.behavioral_change_evaluation import GuidelineEvaluator
from parlant.core.sessions import EventSource, Session
from parlant.core.tags import Tag, TagId
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
    "scheduling_journey": {
        "condition": "The patient wants to schedule an appointment",
    },
    "lab_results_journey": {"condition": "The patient wants to see their lab results"},
}

CONDITION_HEAD_DICT = {
    "amusement_park": "The customer asks to book a ticket to an amusement ride or attraction, and its not clear which one",
    "lost_card": "The customer lost their card and didn't specify what they want to do",
    "stolen_card": "The customer indicates that their card was stolen and didn't specify what they want to do",
    "cancel_flight": "The customer if asks to make a change in booked flight but doesn’t specify whether they want to reschedule, request a refund, or fully cancel the booking",
    "fix_bug": "The customer has a technical problem, and they didn't specify what kind of help they want to have",
    "suspicious_transaction": "The user suspects fraud but it's not clear whether they want to dispute a transaction or lock a card.",
    "healthcare_inquiry": "The patient asks to follow up on their visit, but it's not clear in which way",
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
        ContextVariableValueId("-"),
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
            action=GUIDELINES_DICT[guideline_name].get("action", None),
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
        journey_paths={},
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


async def test_that_ambiguity_detected_with_relevant_guidelines(
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
    ]

    to_disambiguate_guidelines = [
        "snake_roller_coaster",
        "turtle_roller_coaster",
        "tiger_Ferris_wheel",
    ]
    disambiguating_guidelines = [
        "snake_roller_coaster",
        "turtle_roller_coaster",
    ]
    head_condition = CONDITION_HEAD_DICT["amusement_park"]
    clarification_must_contain = "snake roller coaster and turtle roller coaster as options"
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


async def test_that_ambiguity_detected_with_relevant_guidelines_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I can’t find my card. I think I lost it in my house",
        ),
    ]

    to_disambiguate_guidelines = [
        "report_lost",
        "lock_card",
        # "report_stealing",
        "replacement_card",
        "freeze_card",
        "report_to_police",
        "dispute_charge",
    ]
    disambiguating_guidelines = [
        "report_lost",
        "lock_card",
        "replacement_card",
        "freeze_card",
    ]
    head_condition = CONDITION_HEAD_DICT["lost_card"]
    clarification_must_contain = (
        "option to report lost card, to lock or freeze it or replace it with a new one"
    )
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


async def test_that_ambiguity_detected_with_relevant_guidelines_3(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I saw a charge I didn’t make. I'm pretty sure it was stolen",
        ),
    ]

    to_disambiguate_guidelines = [
        "lock_card",
        "report_stealing",
        "replacement_card",
        "freeze_card",
        "dispute_charge",
    ]
    # report lost card is not really likely, but better offer not very relevant options then omit ones.
    #  It sometimes add it and sometimes not so for now i dont include it in the test
    disambiguating_guidelines = [
        "lock_card",
        "replacement_card",
        "freeze_card",
        "report_stealing",
        "dispute_charge",
    ]
    head_condition = CONDITION_HEAD_DICT["stolen_card"]
    clarification_must_contain = (
        "option to report to lock or freeze the card, report stealing, or dispute a charge"
    )
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


async def test_that_ambiguity_detected_based_on_context_variable(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I see I can't make it to the flight. Can you help me?",
        ),
    ]
    context_variables = [
        create_context_variable(
            name="Customer tier",
            data={"tier": "VIP"},
            tags=[Tag.for_agent_id(agent.id)],
        ),
    ]
    to_disambiguate_guidelines = [
        "vip_refund",
        "vip_reschedule",
        "vip_cancel",
        "regular_refund_travel_credit",
        "regular_reschedule",
        "regular_cancel",
    ]

    disambiguating_guidelines = [
        "vip_refund",
        "vip_reschedule",
        "vip_cancel",
    ]
    head_condition = CONDITION_HEAD_DICT["cancel_flight"]
    clarification_must_contain = "options to cancel the flight, totally cancel or get a refund to payment method or to travel credit"
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
        context_variables=context_variables,
    )


async def test_that_ambiguity_detected_based_on_context_variable_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I see I can't make it to the flight. Can you help me?",
        ),
    ]
    context_variables = [
        create_context_variable(
            name="Customer tier",
            data={"tier": "Basic"},
            tags=[Tag.for_agent_id(agent.id)],
        ),
    ]
    to_disambiguate_guidelines = [
        "vip_refund",
        "vip_reschedule",
        "vip_cancel",
        "regular_refund_travel_credit",
        "regular_reschedule",
        "regular_cancel",
    ]

    disambiguating_guidelines = [
        "regular_refund_travel_credit",
        "regular_reschedule",
        "regular_cancel",
    ]
    head_condition = CONDITION_HEAD_DICT["cancel_flight"]
    clarification_must_contain = (
        "options to reschedule the flight, totally cancel or get a refund to travel credit"
    )
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
        context_variables=context_variables,
    )


async def test_that_ambiguity_is_not_detected_when_there_is_no_ambiguity(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Please book me for the snake roller coaster",
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


async def test_that_when_agent_already_asked_for_clarification_new_clarification_guideline_does_created(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi book me to the roller coaster please",
        ),
        (
            EventSource.AI_AGENT,
            "Sure, We have snake roller coaster and turtle roller coaster. Which one would you like?.",
        ),
        (
            EventSource.CUSTOMER,
            "Hmm Let me see",
        ),
    ]
    to_disambiguate_guidelines = [
        "snake_roller_coaster",
        "turtle_roller_coaster",
        "tiger_Ferris_wheel",
    ]
    clarification_must_contain = "A snake roller coaster a turtle roller coaster"
    disambiguating_guidelines: list[str] = ["snake_roller_coaster", "turtle_roller_coaster"]
    head_condition = CONDITION_HEAD_DICT["amusement_park"]
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


async def test_that_when_agent_already_asked_for_clarification_new_clarification_guideline_does_created_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I can’t find my card. I think I lost it in my house",
        ),
        (
            EventSource.AI_AGENT,
            "To help you with your card, could you please confirm what you'd like to do? Would you like to report it as lost, lock it, freeze it temporarily, or request a replacement?",
        ),
        (
            EventSource.CUSTOMER,
            "I know that it's lost",
        ),
    ]

    to_disambiguate_guidelines = [
        "report_lost",
        "lock_card",
        # "report_stealing",
        "replacement_card",
        "freeze_card",
        "report_to_police",
        "dispute_charge",
    ]
    disambiguating_guidelines = [
        "report_lost",
        "lock_card",
        "replacement_card",
        "freeze_card",
    ]
    head_condition = CONDITION_HEAD_DICT["lost_card"]
    clarification_must_contain = (
        "option to report lost card, to lock or freeze it or replace it with a new one"
    )
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


async def test_that_ambiguity_is_not_detected_when_agent_asked_for_clarification_but_customer_changed_its_mind(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I can’t find my card. I think I lost it in my house",
        ),
        (
            EventSource.AI_AGENT,
            "To help you with your card, could you please confirm what you'd like to do? Would you like to report it as lost, lock it, freeze it temporarily, or request a replacement?",
        ),
        (
            EventSource.CUSTOMER,
            "Actually I found it. All good",
        ),
    ]

    to_disambiguate_guidelines = [
        "report_lost",
        "lock_card",
        "report_stealing",
        "replacement_card",
        "freeze_card",
        "report_to_police",
        "dispute_charge",
    ]
    disambiguating_guidelines: list[str] = []
    head_condition = CONDITION_HEAD_DICT["lost_card"]
    clarification_must_contain = (
        "option to report lost card, to lock or freeze it or replace it with a new one"
    )
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
        clarification_must_contain=clarification_must_contain,
    )


async def test_that_ambiguity_is_not_detected_when_agent_asked_for_clarification_but_customer_changed_subject(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I can’t find my card. I think I lost it in my house",
        ),
        (
            EventSource.AI_AGENT,
            "To help you with your card, could you please confirm what you'd like to do? Would you like to report it as lost, lock it, freeze it temporarily, or request a replacement?",
        ),
        (
            EventSource.CUSTOMER,
            "Actually I will handle this later. Can you check my balance?",
        ),
    ]

    to_disambiguate_guidelines = [
        "report_lost",
        "lock_card",
        "report_stealing",
        "replacement_card",
        "freeze_card",
        "report_to_police",
        "dispute_charge",
    ]
    disambiguating_guidelines: list[str] = []
    head_condition = CONDITION_HEAD_DICT["lost_card"]
    clarification_must_contain = (
        "option to report lost card, to lock or freeze it or replace it with a new one"
    )
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
        clarification_must_contain=clarification_must_contain,
    )


async def test_that_ambiguity_is_not_detected_when_clarification_was_asked_and_customer_responded(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I need to lock my card",
        ),
        (
            EventSource.AI_AGENT,
            "I can help you with locking your card. You have the following cards available to lock: Direct or Visa. Please let me know which card you'd like to lock. Please let me know how to proceed.",
        ),
        (
            EventSource.CUSTOMER,
            "Direct",
        ),
        (
            EventSource.AI_AGENT,
            "Could you please provide the reason for locking the card?",
        ),
        (
            EventSource.CUSTOMER,
            "Actually, before that, let me discuss a weird tx",
        ),
        (
            EventSource.AI_AGENT,
            "I can help you with discussing any concerns you have with the weird transaction or proceeding with locking the card. Please let me know how to proceed.",
        ),
        (
            EventSource.CUSTOMER,
            "Dispute",
        ),
        (
            EventSource.AI_AGENT,
            "Would you like to dispute a transaction or lock your card?",
        ),
        (
            EventSource.CUSTOMER,
            "Dispute",
        ),
    ]

    to_disambiguate_guidelines = [
        "lock_card",
        "dispute_charge",
    ]
    disambiguating_guidelines: list[str] = []
    head_condition = CONDITION_HEAD_DICT["suspicious_transaction"]
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


async def test_that_ambiguity_is_not_detected_on_clear_request_1(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hello there.",
        ),
        (
            EventSource.AI_AGENT,
            "Hello.",
        ),
        (
            EventSource.AI_AGENT,
            "How can I assist you today?",
        ),
        (
            EventSource.CUSTOMER,
            "I need to book an appointment with my doctor,",
        ),
    ]

    to_disambiguate_guidelines = ["scheduling_journey", "lab_results_journey"]
    disambiguating_guidelines: list[str] = []
    head_condition = CONDITION_HEAD_DICT["healthcare_inquiry"]
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


async def test_that_ambiguity_is_not_detected_on_clear_request_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I need an appointment with my dr",
        ),
    ]

    to_disambiguate_guidelines = ["scheduling_journey", "lab_results_journey"]
    disambiguating_guidelines: list[str] = []
    head_condition = CONDITION_HEAD_DICT["healthcare_inquiry"]
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
