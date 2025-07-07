from lagom import Container
from pytest import fixture
from parlant.core.agents import Agent
from parlant.core.customers import Customer
from parlant.core.engines.alpha.guideline_matching.generic.journey_step_selection_batch import (
    JourneyStepSelectionSchema,
)
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import EventSource, Session

from tests.core.stable.engines.alpha.test_journey_step_selection import (
    ContextOfTest,
    base_test_that_correct_step_is_selected,
)
from tests.test_utilities import SyncAwaiter


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
