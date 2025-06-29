from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence, cast

from lagom import Container
from pytest import fixture

from parlant.core.agents import Agent
from parlant.core.capabilities import Capability
from parlant.core.common import JSONSerializable
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableId,
    ContextVariableValue,
    ContextVariableValueId,
)
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent

from parlant.core.engines.alpha.guideline_matching.generic.journey_step_selection_batch import (
    GenericJourneyStepSelectionBatch,
    JourneyStepSelectionSchema,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingContext,
)
from parlant.core.glossary import Term, TermId
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.journeys import Journey, JourneyId, JourneyStepId
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
    schematic_generator: SchematicGenerator[JourneyStepSelectionSchema]
    logger: Logger


@dataclass
class _StepData:
    id: str
    condition: str | None
    action: str | None
    customer_dependent_action: bool = False
    requires_tool_calls: bool = False
    follow_up_ids: list[str] = field(default_factory=list)


@dataclass
class _JourneyData:
    title: str
    steps: list[_StepData]


@fixture
def context(
    sync_await: SyncAwaiter,
    container: Container,
) -> ContextOfTest:
    return ContextOfTest(
        container,
        sync_await,
        logger=container[Logger],
        schematic_generator=container[SchematicGenerator[JourneyStepSelectionSchema]],
    )


JOURNEYS_DICT: dict[str, _JourneyData] = {
    "compliment_customer_journey": _JourneyData(
        title="Compliment Customer Journey",
        steps=[
            _StepData(
                id="1",
                condition=None,
                action="ask the customer for their name",
                follow_up_ids=["2"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="2",
                condition=None,
                action="tell them their name is pretty",
                follow_up_ids=["3"],
            ),
            _StepData(
                id="3",
                condition=None,
                action="ask them their surname",
                follow_up_ids=["4"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="4",
                condition=None,
                action="ask for their phone number",
                follow_up_ids=["5"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="5",
                condition=None,
                action="send the customer a link to our terms of service page",
                follow_up_ids=["6"],
            ),
            _StepData(
                id="6",
                condition=None,
                action="ask the customer for their favorite color",
                follow_up_ids=[],
                customer_dependent_action=True,
            ),
        ],
    ),
    "forgot_keys_journey": _JourneyData(
        title="Help Customer Find Their Keys",
        steps=[
            _StepData(
                id="1",
                condition=None,
                action="Ask the customer what type of keys they lost",
                follow_up_ids=["2"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="2",
                condition=None,
                action="Ask them when's the last time they used their keys",
                follow_up_ids=["3"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="3",
                condition=None,
                action="Tell them to check if it's near where they last used their keys",
                follow_up_ids=["4", "5"],
                customer_dependent_action=False,
            ),
            _StepData(
                id="4",
                condition="The customer hasn't found their keys",
                action="Tell them that they better get a new house",
                follow_up_ids=[],
                customer_dependent_action=True,
            ),
            _StepData(
                id="5",
                condition=None,
                action=None,
                follow_up_ids=[],
                customer_dependent_action=False,
            ),
        ],
    ),
    "reset_password_journey": _JourneyData(
        title="Reset Password Journey",
        steps=[
            _StepData(
                id="1",
                condition="The customer has not provided their account number",
                action="Ask for their account number",
                follow_up_ids=["2", "1"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="2",
                condition="The customer provided their account number",
                action="Ask for their email address or phone number",
                follow_up_ids=["3"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="3",
                condition=None,
                action="Wish them a good day",
                follow_up_ids=["4", "5"],
            ),
            _StepData(
                id="4",
                condition=None,
                action=None,
                follow_up_ids=[],
            ),
            _StepData(
                id="5",
                condition="The customer wished you a good day in return",
                action="Use the reset_password tool with the provided information",
                follow_up_ids=["6", "7"],
                requires_tool_calls=True,
            ),
            _StepData(
                id="6",
                condition="reset_password tool returned that the password was successfully reset",
                action="Report the result to the customer",
                follow_up_ids=[],
            ),
            _StepData(
                id="7",
                condition="reset_password tool returned that the password was not successfully reset, or otherwise failed",
                action="Apologize to the customer and report that the password cannot be reset at this time",
                follow_up_ids=[],
            ),
        ],
    ),
    "calzone_journey": _JourneyData(
        title="Deliver Calzone Journey",
        steps=[
            _StepData(
                id="1",
                condition=None,
                action="Welcome the customer to the Low Cal Calzone Zone",
                follow_up_ids=["2"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="2",
                condition="Always",
                action="Ask them how many calzones they want",
                follow_up_ids=["3", "7"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="3",
                condition="more than 5",
                action="Warn the customer that delivery is likely to take more than an hour",
                follow_up_ids=["4"],
            ),
            _StepData(
                id="4",
                condition="Always",
                action="Ask if they are able to call a human representative",
                follow_up_ids=["5", "6"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="5",
                condition="They can",
                action="Tell them to order by phone to ensure correct delivery",
                follow_up_ids=[],
            ),
            _StepData(
                id="6",
                condition=None,
                action="Apologize and say you support orders of up to 5 calzones",
                follow_up_ids=[],
            ),
            _StepData(
                id="7",
                condition="5 or less",
                action="Ask what type of calzones they want out of the options - Classic Italian Calzone, Spinach and Ricotta Calzone, Chicken and Broccoli Calzone",
                follow_up_ids=["8"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="8",
                condition="The customer chose their calzone type",
                action="Ask which size of calzone they want between small, medium, and large",
                follow_up_ids=["9"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="9",
                condition="The customer chose their calzone size",
                action="Ask if they want any drinks with their order",
                follow_up_ids=["10"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="10",
                condition="The customer chose if they want drinks, and which ones",
                action="Check if all ordered items are available in stock",
                follow_up_ids=["11", "12"],
                requires_tool_calls=True,
            ),
            _StepData(
                id="11",
                condition="All items are available",
                action="Confirm the order details with the customer",
                follow_up_ids=["13"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="12",
                condition="Some items are not available",
                action="Apologize for the inconvenience and ask them to remove missing items from their order",
                follow_up_ids=["10"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="13",
                condition="The customer confirmed their order",
                action="Ask for the delivery address",
                follow_up_ids=["14"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="14",
                condition="The customer provided their delivery address",
                action="Place the order and thank them for choosing the Low Cal Calzone Zone",
                follow_up_ids=[],
            ),
        ],
    ),
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
        name=name,
        description="",
        tool_id=None,
        freshness_rules=None,
        tags=tags,
    ), ContextVariableValue(
        ContextVariableValueId("-"), last_modified=datetime.now(timezone.utc), data=data
    )


async def create_journey(
    title: str,
    steps: list[_StepData],
) -> tuple[Journey, Sequence[Guideline]]:
    journey_step_guidelines: Sequence[Guideline] = [
        Guideline(
            id=GuidelineId(step.id),
            creation_utc=datetime.now(timezone.utc),
            content=GuidelineContent(
                condition=step.condition or "",
                action=step.action,
            ),
            enabled=False,
            tags=[],
            metadata={
                "journey_step": step.id,
                "customer_dependent_action_data": {
                    "is_customer_dependent": step.customer_dependent_action
                },
                "tool_running_only": step.requires_tool_calls,
                "sub_steps": [GuidelineId(follow_up_id) for follow_up_id in step.follow_up_ids],
            },
        )
        for step in steps
    ]
    journey = Journey(
        id=JourneyId("-"),
        creation_utc=datetime.now(timezone.utc),
        conditions=[],
        steps=[cast(JourneyStepId, g.id) for g in journey_step_guidelines],
        title=title,
        description="",
        tags=[],
    )
    return journey, journey_step_guidelines


async def base_test_that_correct_step_is_selected(
    context: ContextOfTest,
    agent: Agent,
    session_id: SessionId,
    customer: Customer,
    conversation_context: list[tuple[EventSource, str]],
    journey_name: str,
    expected_next_step_id: str | None,
    expected_path: list[str] | None = None,
    journey_previous_path: Sequence[str | None] = [],
    capabilities: Sequence[Capability] = [],
    staged_events: Sequence[EmittedEvent] = [],
) -> None:
    session = await context.container[SessionStore].read_session(session_id)

    interaction_history = [
        create_event_message(
            offset=i,
            source=source,
            message=message,
        )
        for i, (source, message) in enumerate(conversation_context)
    ]

    journey, journey_step_guidelines = await create_journey(
        title=JOURNEYS_DICT[journey_name].title,
        steps=JOURNEYS_DICT[journey_name].steps,
    )

    journey_step_selector = GenericJourneyStepSelectionBatch(
        logger=context.logger,
        schematic_generator=context.schematic_generator,
        examined_journey=journey,
        step_guidelines=journey_step_guidelines,
        journey_path=journey_previous_path,
        context=GuidelineMatchingContext(
            agent=agent,
            session=session,
            customer=customer,
            context_variables=[],
            interaction_history=interaction_history,
            terms=[],
            capabilities=capabilities,
            staged_events=staged_events,
            relevant_journeys=[],
        ),
    )
    result = await journey_step_selector.process()
    if len(result.matches) == 0:
        assert expected_next_step_id is None
    else:
        result_path: Sequence[str] = cast(list[str], result.matches[0].metadata["journey_path"])
        if expected_path:
            assert len(result_path) == len(expected_path)
            for result_step, expected_step in zip(result_path, expected_path):
                assert result_step == expected_step
        elif expected_next_step_id:  # Only test that the next step is correct
            assert result_path[-1] == expected_next_step_id


async def test_that_journey_selector_correctly_advances_by_multiple_steps(  # Occasionally fast-forwards by too little, to step 7 instead of 9
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome to the Low Cal Calzone Zone!",
        ),
        (
            EventSource.CUSTOMER,
            "Thanks! Can I order 3 medium classical Italian calzones please?",
        ),
    ]

    await base_test_that_correct_step_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="calzone_journey",
        journey_previous_path=["1"],
        expected_path=["1", "2", "7", "8", "9"],
        expected_next_step_id="9",
    )


async def test_that_multistep_advancement_is_stopped_at_step_that_requires_saying_something(  # Final decision is good, subpath it takes isn't
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi",
        ),
        (
            EventSource.AI_AGENT,
            "Hello! What's your name?",
        ),
        (
            EventSource.CUSTOMER,
            "My name is Jez",
        ),
        (
            EventSource.AI_AGENT,
            "What a beautiful name!",
        ),
        (
            EventSource.CUSTOMER,
            "Thank you! Since you show so much interest in me, you should also know that my surname is Osborne, my phone number is 555-123-4567, and my favorite color is orange.",
        ),
    ]

    await base_test_that_correct_step_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="compliment_customer_journey",
        journey_previous_path=["1", "2"],
        expected_path=["2", "3", "4", "5"],
        expected_next_step_id="5",
    )
