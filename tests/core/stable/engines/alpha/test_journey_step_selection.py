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
from parlant.core.sessions import EventKind, EventSource, Session, SessionId, SessionStore
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
    expected_next_step_id: str | Sequence[str] | None,
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
            if isinstance(expected_next_step_id, list):
                assert result_path[-1] in expected_next_step_id
            else:
                assert result_path[-1] == expected_next_step_id


async def test_that_journey_selector_repeats_step_if_incomplete_1(
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
            "How is that relevant?",
        ),
    ]
    await base_test_that_correct_step_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="compliment_customer_journey",
        expected_next_step_id="1",
    )


async def test_that_journey_selector_repeats_step_if_incomplete_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I'd like to order some calzones",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome to the Low Cal Calzone Zone! How many calzones would you like?",
        ),
        (
            EventSource.CUSTOMER,
            "I'll take 3 calzones",
        ),
        (
            EventSource.AI_AGENT,
            "Great! What type of calzones would you like? We have Classic Italian Calzone, Spinach and Ricotta Calzone, and Chicken and Broccoli Calzone.",
        ),
        (
            EventSource.CUSTOMER,
            "I'll go with Classic Italian",
        ),
        (
            EventSource.AI_AGENT,
            "Perfect! What size would you like - small, medium, or large?",
        ),
        (
            EventSource.CUSTOMER,
            "Let me check for a sec",
        ),
    ]

    await base_test_that_correct_step_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="calzone_journey",
        journey_previous_path=["1", "2", "7", "8"],
        expected_next_step_id="8",
    )


# 1 step advancement tests


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


async def test_that_journey_selector_correctly_exits_journey_1(
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
        (
            EventSource.AI_AGENT,
            "Thank you. Now I need your email address or phone number.",
        ),
        (
            EventSource.CUSTOMER,
            "john.doe@email.com",
        ),
        (
            EventSource.AI_AGENT,
            "Great! Have a good day!",
        ),
        (
            EventSource.CUSTOMER,
            "Okay, thanks.",
        ),
    ]
    await base_test_that_correct_step_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="reset_password_journey",
        journey_previous_path=["1", "2", "3"],
        expected_next_step_id=None,
    )


async def test_that_journey_selector_correctly_advances_to_follow_up_step_3(
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
        (
            EventSource.AI_AGENT,
            "Thank you. Now I need your email address or phone number.",
        ),
        (
            EventSource.CUSTOMER,
            "john.doe@email.com",
        ),
        (
            EventSource.AI_AGENT,
            "Great! Have a good day!",
        ),
        (
            EventSource.CUSTOMER,
            "Thank you, have a good day too!",
        ),
    ]
    await base_test_that_correct_step_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="reset_password_journey",
        journey_previous_path=["1", "2", "3"],
        expected_next_step_id="5",
    )


async def test_that_journey_selector_correctly_advances_based_on_tool_result(
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
        (
            EventSource.AI_AGENT,
            "Thank you. Now I need your email address or phone number.",
        ),
        (
            EventSource.CUSTOMER,
            "john.doe@email.com",
        ),
        (
            EventSource.AI_AGENT,
            "Great! Have a good day!",
        ),
        (
            EventSource.CUSTOMER,
            "Thank you, have a good day too!",
        ),
    ]

    tool_result = cast(
        JSONSerializable,
        {
            "tool_calls": [
                {
                    "tool_id": "local:reset_password",
                    "arguments": {"account_number": "199877", "email": "john.doe@email.com"},
                    "result": {
                        "data": "Password reset successfully",
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

    await base_test_that_correct_step_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="reset_password_journey",
        journey_previous_path=["1", "2", "3", "5"],
        expected_next_step_id="6",
        staged_events=staged_events,
    )


async def test_that_journey_selector_correctly_exits_journey_that_no_longer_applies(
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
        (
            EventSource.AI_AGENT,
            "Thank you. Now I need your email address or phone number.",
        ),
        (
            EventSource.CUSTOMER,
            "Oh actually never mind, can you help me with an existing order first?",
        ),
    ]

    await base_test_that_correct_step_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="reset_password_journey",
        journey_previous_path=["1", "2"],
        expected_next_step_id=None,
    )


# Multistep advancement tests


async def test_that_multistep_advancement_is_stopped_at_tool_requiring_steps(
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
            "I'd like 3 Classic Italian calzones, medium size, no drinks. My address is 1234 Main Street, NYC, USA",
        ),
    ]

    await base_test_that_correct_step_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="calzone_journey",
        journey_previous_path=["1", "2"],
        expected_path=["2", "7", "8", "9", "10"],
        expected_next_step_id="10",
    )


async def test_that_multistep_advancement_completes_and_exits_journey(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I lost my keys.",
        ),
        (
            EventSource.AI_AGENT,
            "I'm sorry to hear that! What type of keys did you lose?",
        ),
        (
            EventSource.CUSTOMER,
            "Car keys, last used them at the office, and I just found them, thanks!",
        ),
    ]

    await base_test_that_correct_step_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="forgot_keys_journey",
        journey_previous_path=["1"],
        expected_next_step_id=None,
    )


# backtracking tests


async def test_that_journey_selector_backtracks_when_customer_changes_earlier_choice_1(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I'd like to order some calzones",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome to the Low Cal Calzone Zone! How many calzones would you like?",
        ),
        (
            EventSource.CUSTOMER,
            "I'll take 3 calzones",
        ),
        (
            EventSource.AI_AGENT,
            "Great! What type of calzones would you like? We have Classic Italian Calzone, Spinach and Ricotta Calzone, and Chicken and Broccoli Calzone.",
        ),
        (
            EventSource.CUSTOMER,
            "I'll go with Classic Italian",
        ),
        (
            EventSource.AI_AGENT,
            "Perfect! What size would you like - small, medium, or large?",
        ),
        (
            EventSource.CUSTOMER,
            "Actually, I changed my mind. I want 2 calzones instead of 3",
        ),
    ]

    await base_test_that_correct_step_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="calzone_journey",
        journey_previous_path=["1", "2", "7", "8"],
        expected_next_step_id="7",  # Should return to asking about calzone type
    )


async def test_that_journey_selector_backtracks_when_customer_changes_earlier_choice_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    """Test backtracking when customer changes quantity from 3 to 10, triggering the 'over 5' path"""
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I want to order calzones please",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome to the Low Cal Calzone Zone! How many calzones would you like?",
        ),
        (
            EventSource.CUSTOMER,
            "Just 3 calzones",
        ),
        (
            EventSource.AI_AGENT,
            "What type of calzones would you like? We have Classic Italian Calzone, Spinach and Ricotta Calzone, and Chicken and Broccoli Calzone.",
        ),
        (
            EventSource.CUSTOMER,
            "Spinach and Ricotta please",
        ),
        (
            EventSource.AI_AGENT,
            "Excellent choice! What size would you like - small, medium, or large?",
        ),
        (
            EventSource.CUSTOMER,
            "Medium please",
        ),
        (
            EventSource.AI_AGENT,
            "Would you like any drinks with your order?",
        ),
        (
            EventSource.CUSTOMER,
            "Actually, I need to change my order. I want 10 calzones instead of 3",
        ),
    ]

    await base_test_that_correct_step_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="calzone_journey",
        journey_previous_path=["1", "2", "7", "8", "9"],
        expected_next_step_id="3",  # Should go to step 3 (warn about delivery time for over 5 calzones)
    )


async def test_that_journey_selector_backtracks_when_customer_changes_earlier_choice_3(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    """Test backtracking when customer changes size after items were checked for availability"""
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I'd like to place an order",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome to the Low Cal Calzone Zone! How many calzones would you like?",
        ),
        (
            EventSource.CUSTOMER,
            "4 calzones please",
        ),
        (
            EventSource.AI_AGENT,
            "What type of calzones would you like? We have Classic Italian Calzone, Spinach and Ricotta Calzone, and Chicken and Broccoli Calzone.",
        ),
        (
            EventSource.CUSTOMER,
            "Classic Italian",
        ),
        (
            EventSource.AI_AGENT,
            "What size would you like - small, medium, or large?",
        ),
        (
            EventSource.CUSTOMER,
            "Large please",
        ),
        (
            EventSource.AI_AGENT,
            "Would you like any drinks with your order?",
        ),
        (
            EventSource.CUSTOMER,
            "No drinks, thanks",
        ),
        (
            EventSource.AI_AGENT,
            "Let me check if all items are available... Great! All items are in stock. Let me confirm your order: 4 large Classic Italian Calzones, no drinks.",
        ),
        (
            EventSource.CUSTOMER,
            "Actually, can I change those to medium size instead of large?",
        ),
    ]

    stock_check_result = cast(
        JSONSerializable,
        {
            "tool_calls": [
                {
                    "tool_id": "local:check_stock",
                    "arguments": {"items": ["4 large Classic Italian Calzones"]},
                    "result": {
                        "data": {
                            "all_available": True,
                            "available_items": ["4 large Classic Italian Calzones"],
                        },
                        "metadata": {},
                        "control": {},
                    },
                }
            ]
        },
    )

    staged_events = [
        EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.TOOL,
            correlation_id="",
            data=stock_check_result,
        ),
    ]

    await base_test_that_correct_step_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="calzone_journey",
        journey_previous_path=["1", "2", "7", "8", "9", "10", "11"],
        expected_next_step_id="9",  # Should return to asking about drinks since size affects the entire order
        staged_events=staged_events,
    )


# Next test fails occasionally, but it's because it backtracks and fast forwards at the same time. This is not officially supported, but it does it correctly occasionally
async def test_that_journey_selector_backtracks_when_customer_changes_much_earlier_choice(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    """Test maximum backtracking when customer realizes they gave wrong account info after tool failure"""
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
        (
            EventSource.AI_AGENT,
            "Thank you. Now I need your email address or phone number.",
        ),
        (
            EventSource.CUSTOMER,
            "john.doe@email.com",
        ),
        (
            EventSource.AI_AGENT,
            "Great! Have a good day!",
        ),
        (
            EventSource.CUSTOMER,
            "Thank you, have a good day too!",
        ),
        (
            EventSource.AI_AGENT,
            "I apologize, but the password could not be reset at this time since your account was not found.",
        ),
        (
            EventSource.CUSTOMER,
            "Oh wait, I think I gave you the wrong account number. It should be 987654, not 318475. Can we try again?",
        ),
    ]

    # Mock tool result showing password reset failed
    failed_tool_result = cast(
        JSONSerializable,
        {
            "tool_calls": [
                {
                    "tool_id": "local:reset_password",
                    "arguments": {"account_number": "318475", "email": "john.doe@email.com"},
                    "result": {
                        "data": "Password reset failed - account not found",
                        "metadata": {"error": "ACCOUNT_NOT_FOUND"},
                        "control": {},
                    },
                }
            ]
        },
    )

    staged_events = [
        EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.TOOL,
            correlation_id="",
            data=failed_tool_result,
        ),
    ]

    await base_test_that_correct_step_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="reset_password_journey",
        journey_previous_path=["1", "2", "3", "5", "7"],
        expected_next_step_id=["2", "5"],
        staged_events=staged_events,
    )
