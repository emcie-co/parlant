from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Sequence
from lagom import Container
from pytest import fixture

from parlant.core.common import Criticality
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.journeys import Journey, JourneyId, JourneyNodeId
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.indexing.relative_action_proposer import (
    RelativeActionProposer,
    RelativeActionSchema,
)
from tests.test_utilities import SyncAwaiter, nlp_test


@dataclass
class ContextOfTest:
    container: Container
    sync_await: SyncAwaiter
    schematic_generator: SchematicGenerator[RelativeActionSchema]
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
    conditions: Sequence[str] = field(default_factory=list)


@fixture
def context(
    sync_await: SyncAwaiter,
    container: Container,
) -> ContextOfTest:
    return ContextOfTest(
        container,
        sync_await,
        logger=container[Logger],
        schematic_generator=container[SchematicGenerator[RelativeActionSchema]],
    )


def create_journey(
    title: str,
    steps: list[_StepData],
    conditions: Sequence[str],
) -> tuple[Journey, Sequence[Guideline], Sequence[Guideline]]:
    # 1. Create conditions, get IDs
    # 2. Create guidelines from step data
    # 3. Return journey, guidelines, conditions
    journey_id = JourneyId("j1")

    condition_guidelines: Sequence[Guideline] = [
        Guideline(
            id=GuidelineId(f"c-{i}"),
            creation_utc=datetime.now(timezone.utc),
            content=GuidelineContent(condition=condition, action=None),
            criticality=Criticality.MEDIUM,
            enabled=False,
            tags=[],
            metadata={},
        )
        for i, condition in enumerate(conditions)
    ]

    root_guideline = Guideline(
        id=GuidelineId("root"),
        creation_utc=datetime.now(timezone.utc),
        content=GuidelineContent(condition="", action=None),
        criticality=Criticality.MEDIUM,
        enabled=True,
        tags=[],
        metadata={
            "journey_node": {
                "follow_ups": ["1"],
                "index": "0",
                "journey_id": journey_id,
            }
        },
    )

    step_guidelines: Sequence[Guideline] = [
        Guideline(
            id=GuidelineId(step.id),
            creation_utc=datetime.now(timezone.utc),
            content=GuidelineContent(
                condition=step.condition or "",
                action=step.action,
            ),
            criticality=Criticality.MEDIUM,
            enabled=False,
            tags=[],
            metadata={
                "journey_node": {
                    "follow_ups": [
                        GuidelineId(follow_up_id) for follow_up_id in step.follow_up_ids
                    ],
                    "index": step.id,
                    "journey_id": journey_id,
                },
                "customer_dependent_action_data": {
                    "is_customer_dependent": step.customer_dependent_action,
                    "customer_action": "",
                    "agent_action": "",
                },
                "tool_running_only": step.requires_tool_calls,
            },
        )
        for step in steps
    ]

    journey = Journey(
        id=journey_id,
        root_id=JourneyNodeId(root_guideline.id),
        creation_utc=datetime.now(timezone.utc),
        description="",
        conditions=[g.id for g in condition_guidelines],
        title=title,
        tags=[],
    )

    return journey, [root_guideline] + list(step_guidelines), condition_guidelines


async def base_test_that_related_action_step_proposed(
    context: ContextOfTest,
    journey: _JourneyData,
    to_propose_actions: Mapping[str, str],
) -> None:
    relative_action_proposer = context.container[RelativeActionProposer]

    examined_journey, step_guidelines, condition_guidelines = create_journey(
        title=journey.title,
        steps=journey.steps,
        conditions=journey.conditions,
    )
    result = await relative_action_proposer.propose_relative_action(
        examined_journey,
        step_guidelines,
        condition_guidelines,
    )
    proposed_actions = {a.index: a.rewritten_actions for a in result.actions}

    assert set(proposed_actions.keys()) == set(to_propose_actions.keys())

    for a in to_propose_actions.keys():
        assert await nlp_test(
            context=f"Here's an action description: {proposed_actions[a]}",
            condition=f"The description contains {to_propose_actions[a]}",
        ), (
            f"proposed action: '{proposed_actions[a]}', expected to contain: '{to_propose_actions[a]}'"
        )


async def test_action_is_proposed_when_needed(
    context: ContextOfTest,
) -> None:
    journey = _JourneyData(
        conditions=["the customer wants to apply for a personal loan"],
        title="Personal Loan Application",
        steps=[
            _StepData(
                id="1",
                condition=None,
                action="Ask how much they want to borrow",
                follow_up_ids=["2"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="2",
                condition="",
                action="Ask what they need that for",
                follow_up_ids=["3"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="3",
                condition="Customer provided loan purpose",
                action="Ask for their employment details",
                follow_up_ids=["4"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="4",
                condition="Employment details provided",
                action="Run the initial eligibility check for the loan application",
                follow_up_ids=["5", "6"],
                requires_tool_calls=True,
            ),
            _StepData(
                id="5",
                condition="Initial check passed",
                action="Tell them it looks good so far",
                follow_up_ids=["7"],
            ),
            _StepData(
                id="6",
                condition="Initial check failed",
                action="Explain why they don't qualify",
                follow_up_ids=[],
            ),
            _StepData(
                id="7",
                condition="Customer wants to continue",
                action="Ask for the required loan documents",
                follow_up_ids=["8"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="8",
                condition="Documents provided",
                action="Submit it for review",
                follow_up_ids=["9"],
                requires_tool_calls=True,
            ),
            _StepData(
                id="9",
                condition="Application submitted",
                action="Give them the reference number and timeline",
                follow_up_ids=[],
            ),
        ],
    )
    to_propose_action = {
        "2": "Ask what they need the loan for",
        "5": "The loan application process looks good or the initial eligibility check looks good",
        "8": "Submit the loan application for review",
    }
    await base_test_that_related_action_step_proposed(
        context,
        journey,
        to_propose_action,
    )


async def test_action_is_not_proposed_when_not_needed(
    context: ContextOfTest,
) -> None:
    journey = _JourneyData(
        conditions=["the customer wants to order a calzone"],
        title="Deliver Calzone Journey",
        steps=[
            _StepData(
                id="1",
                condition=None,
                action="Welcome the customer to the Low Cal Calzone Zone",
                follow_up_ids=["2"],
            ),
            _StepData(
                id="2",
                condition="Always",
                action="Ask them how many calzones they want",
                follow_up_ids=["3", "7"],
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
            ),
            _StepData(
                id="8",
                condition="The customer chose their calzone type",
                action="Ask which size of calzone they want between small, medium, and large",
                follow_up_ids=["9"],
            ),
            _StepData(
                id="9",
                condition="The customer chose their calzone size",
                action="Ask if they want any drinks with their order",
                follow_up_ids=["10"],
            ),
            _StepData(
                id="10",
                condition="The customer chose if they want drinks, and which ones",
                action="Check if all ordered items are available in stock",
                follow_up_ids=["11", "12"],
            ),
            _StepData(
                id="11",
                condition="All items are available",
                action="Confirm the order details with the customer",
                follow_up_ids=["13"],
            ),
            _StepData(
                id="12",
                condition="Some items are not available",
                action="Apologize for the inconvenience and ask them to remove missing items from their order",
                follow_up_ids=["10"],
            ),
            _StepData(
                id="13",
                condition="The customer confirmed their order",
                action="Ask for the delivery address",
                follow_up_ids=["14"],
            ),
            _StepData(
                id="14",
                condition="The customer provided their delivery address",
                action="Place the order and thank them for choosing the Low Cal Calzone Zone",
                follow_up_ids=[],
            ),
        ],
    )
    to_propose_actions: Mapping[str, str] = {}
    await base_test_that_related_action_step_proposed(
        context,
        journey,
        to_propose_actions,
    )


async def test_action_is_proposed_when_needed_2(
    context: ContextOfTest,
) -> None:
    journey = _JourneyData(
        conditions=["the customer wants to order a calzone"],
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
                action="Ask them how many they want",
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
                action="Ask the customer if they are able to call a human representative",
                follow_up_ids=["5", "6"],
                customer_dependent_action=True,
            ),
            _StepData(
                id="5",
                condition="They can",
                action="Tell them to do it that way instead",
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
                action="Check availability",
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
                action="Place it and thank them for choosing the Low Cal Calzone Zone",
                follow_up_ids=[],
            ),
        ],
    )
    to_propose_actions = {
        "2": "ask how many calzones",
        "5": "to tell them to call human representative",
        "10": "to check availability of calzones and drinks",
        "14": "to place the order",
    }
    await base_test_that_related_action_step_proposed(
        context,
        journey,
        to_propose_actions,
    )
