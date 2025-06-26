from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence, Tuple, cast

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
from parlant.core.journeys import Journey, JourneyId
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
                condition="the customer provided their name",
                action="tell them their name is pretty",
                follow_up_ids=[],
            ),
            _StepData(
                id="3",
                condition="the agent told the customer their name is pretty",
                action=None,
                follow_up_ids=[],
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
                condition="The customer provided either their email address or phone number",
                action="Wish them a good day",
                follow_up_ids=["4", "5"],
            ),
            _StepData(
                id="4",
                condition="The customer did not wish you a good day in return",
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
) -> Tuple[Journey, Sequence[Guideline]]:
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
        steps=[g.id for g in journey_step_guidelines],
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
    expected_next_step_id: str,
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
        guidelines=journey_step_guidelines,
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


async def test_that_journey_selector_correctly_advances_to_follow_up_step_1(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi there, I need to reset my password",
        ),
        (
            EventSource.AI_AGENT,
            "I'm here to help you with that. What is your name?",
        ),
        (
            EventSource.CUSTOMER,
            "My name is Bartholomew",
        ),
    ]
    await base_test_that_correct_step_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="compliment_customer_journey",
        expected_next_step_id="2",
    )


async def test_that_journey_selector_correctly_advances_to_follow_up_step_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi there, I need to reset my password",
        ),
        (
            EventSource.AI_AGENT,
            "I'm here to help you with that. What is your account number?",
        ),
        (
            EventSource.CUSTOMER,
            "318475",
        ),
    ]
    await base_test_that_correct_step_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="reset_password_journey",
        journey_previous_path=["1"],
        expected_next_step_id="2",
    )
