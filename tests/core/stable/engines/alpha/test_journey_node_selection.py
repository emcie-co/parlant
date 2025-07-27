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

from parlant.core.engines.alpha.guideline_matching.generic.journey_node_selection_batch import (
    GenericJourneyNodeSelectionBatch,
    JourneyNodeSelectionSchema,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingContext,
)
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.glossary import Term, TermId
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId, GuidelineStore
from parlant.core.journeys import Journey, JourneyId, JourneyNodeId
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import EventKind, EventSource, Session, SessionId, SessionStore
from parlant.core.tags import Tag, TagId
from tests.core.common.utils import create_event_message
from tests.test_utilities import SyncAwaiter


@dataclass
class ContextOfTest:
    container: Container
    sync_await: SyncAwaiter
    schematic_generator: SchematicGenerator[JourneyNodeSelectionSchema]
    logger: Logger


@dataclass
class _NodeData:
    id: str
    condition: str | None
    action: str | None
    customer_dependent_action: bool = False
    customer_action: str | None = None
    requires_tool_calls: bool = False
    follow_up_ids: list[str] = field(default_factory=list)


@dataclass
class _JourneyData:
    title: str
    nodes: list[_NodeData]
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
        schematic_generator=container[SchematicGenerator[JourneyNodeSelectionSchema]],
    )


JOURNEYS_DICT: dict[str, _JourneyData] = {
    "compliment_customer_journey": _JourneyData(
        conditions=["the customer wishes to reset their password"],
        title="Compliment Customer Journey",
        nodes=[
            _NodeData(
                id="1",
                condition=None,
                action="ask the customer for their name",
                follow_up_ids=["2"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="2",
                condition=None,
                action="tell them their name is pretty",
                follow_up_ids=["3"],
            ),
            _NodeData(
                id="3",
                condition=None,
                action="ask them their surname",
                follow_up_ids=["4"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="4",
                condition=None,
                action="ask for their phone number",
                follow_up_ids=["5"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="5",
                condition=None,
                action="send the customer a link to our terms of service page",
                follow_up_ids=["6"],
            ),
            _NodeData(
                id="6",
                condition=None,
                action="ask the customer for their favorite color",
                follow_up_ids=[],
                customer_dependent_action=True,
            ),
        ],
    ),
    "forgot_keys_journey": _JourneyData(
        conditions=["the customer doesn't know where their keys are"],
        title="Help Customer Find Their Keys",
        nodes=[
            _NodeData(
                id="1",
                condition=None,
                action="Ask the customer what type of keys they lost",
                follow_up_ids=["2"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="2",
                condition=None,
                action="Ask them when's the last time they used their keys",
                follow_up_ids=["3"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="3",
                condition=None,
                action="Tell them to check if it's near where they last used their keys",
                follow_up_ids=["4", "5"],
                customer_dependent_action=False,
            ),
            _NodeData(
                id="4",
                condition="The customer hasn't found their keys",
                action="Tell them that they better get a new house",
                follow_up_ids=[],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="5",
                condition=None,
                action=None,
                follow_up_ids=[],
                customer_dependent_action=False,
            ),
        ],
    ),
    "reset_password_journey": _JourneyData(
        conditions=[
            "the customer wants to reset their password",
            "the customer can't remember their password",
        ],
        title="Reset Password Journey",
        nodes=[
            _NodeData(
                id="1",
                condition="The customer has not provided their account number",
                action="Ask for their account number",
                follow_up_ids=["2", "1"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="2",
                condition="The customer provided their account number",
                action="Ask for their email address or phone number",
                follow_up_ids=["3"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="3",
                condition=None,
                action="Wish them a good day",
                follow_up_ids=["4", "5"],
            ),
            _NodeData(
                id="4",
                condition="The customer did not immediately wish you a good day in return",
                action=None,
                follow_up_ids=[],
            ),
            _NodeData(
                id="5",
                condition="The customer wished you a good day in return",
                action="Use the reset_password tool with the provided information",
                follow_up_ids=["6", "7"],
                requires_tool_calls=True,
            ),
            _NodeData(
                id="6",
                condition="reset_password tool returned that the password was successfully reset",
                action="Report the result to the customer",
                follow_up_ids=[],
            ),
            _NodeData(
                id="7",
                condition="reset_password tool returned that the password was not successfully reset, or otherwise failed",
                action="Apologize to the customer and report that the password cannot be reset at this time",
                follow_up_ids=[],
            ),
        ],
    ),
    "calzone_journey": _JourneyData(
        conditions=["the customer wants to order a calzone"],
        title="Deliver Calzone Journey",
        nodes=[
            _NodeData(
                id="1",
                condition=None,
                action="Welcome the customer to the Low Cal Calzone Zone",
                follow_up_ids=["2"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="2",
                condition="Always",
                action="Ask them how many calzones they want",
                follow_up_ids=["3", "7"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="3",
                condition="more than 5",
                action="Warn the customer that delivery is likely to take more than an hour",
                follow_up_ids=["4"],
            ),
            _NodeData(
                id="4",
                condition="Always",
                action="Ask if they are able to call a human representative",
                follow_up_ids=["5", "6"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="5",
                condition="They can",
                action="Tell them to order by phone to ensure correct delivery",
                follow_up_ids=[],
            ),
            _NodeData(
                id="6",
                condition=None,
                action="Apologize and say you support orders of up to 5 calzones",
                follow_up_ids=[],
            ),
            _NodeData(
                id="7",
                condition="5 or less",
                action="Ask what type of calzones they want out of the options - Classic Italian Calzone, Spinach and Ricotta Calzone, Chicken and Broccoli Calzone",
                follow_up_ids=["8"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="8",
                condition="The customer chose a calzone type for every calzone they ordered",
                action="Ask which size of calzone they want between small, medium, and large",
                follow_up_ids=["9"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="9",
                condition="The customer chose their calzone size for every calzone they'd like to order",
                action="Ask if they want any drinks with their order",
                follow_up_ids=["10"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="10",
                condition="The customer chose if they want drinks, and which ones",
                action="Check if all ordered items are available in stock",
                follow_up_ids=["11", "12"],
                requires_tool_calls=True,
            ),
            _NodeData(
                id="11",
                condition="All items are available",
                action="Confirm the order details with the customer",
                follow_up_ids=["13"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="12",
                condition="Some items are not available",
                action="Apologize for the inconvenience and ask them to remove missing items from their order",
                follow_up_ids=["10"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="13",
                condition="The customer confirmed their order",
                action="Ask for the delivery address",
                follow_up_ids=["14"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="14",
                condition="The customer provided their delivery address",
                action="Place the order and thank them for choosing the Low Cal Calzone Zone",
                follow_up_ids=[],
            ),
        ],
    ),
    "tech_experience_journey": _JourneyData(
        conditions=["the customer needs technical help"],
        title="Technical Experience Journey",
        nodes=[
            _NodeData(
                id="1",
                condition=None,
                action="Ask the customer how much technical experience they have",
                customer_action="the customer provided enough information to determine if they have plenty of technical experience",
                follow_up_ids=["2"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="2",
                condition=None,
                action="Ask if the issue they have is internet or password related",
                customer_action="the customer provided enough information to identify if the issue is related to internet connectivity, password issues or something entirely different",
                follow_up_ids=["3", "4"],
                customer_dependent_action=True,
            ),
            _NodeData(
                id="3",
                condition="The issue is internet related",
                action="No action necessary - always advance to the next step based on the relevant transition",
                follow_up_ids=["5", "6"],
                customer_dependent_action=False,
            ),
            _NodeData(
                id="4",
                condition="The issue is password related, or something entirely different",
                action="No action necessary - always advance to the next step based on the relevant transition",
                follow_up_ids=["7", "8"],
                customer_dependent_action=False,
            ),
            _NodeData(
                id="5",
                condition="They have much technical experience",
                action="Provide advanced internet troubleshooting steps",
                follow_up_ids=[],
                customer_dependent_action=False,
            ),
            _NodeData(
                id="6",
                condition="They do not have much technical experience",
                action="Provide basic internet troubleshooting steps",
                follow_up_ids=[],
                customer_dependent_action=False,
            ),
            _NodeData(
                id="7",
                condition="They have much technical experience",
                action="Provide advanced non-internet troubleshooting steps",
                follow_up_ids=[],
                customer_dependent_action=False,
            ),
            _NodeData(
                id="8",
                condition="They do not have much technical experience",
                action="Provide basic non-internet troubleshooting steps",
                follow_up_ids=[],
                customer_dependent_action=False,
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
    context: ContextOfTest, title: str, nodes: Sequence[_NodeData], conditions: Sequence[str]
) -> tuple[Journey, Sequence[Guideline]]:
    journey_id = JourneyId("j1")
    guideline_store = context.container[GuidelineStore]
    condition_ids: list[GuidelineId] = []

    for c in conditions:
        g = await guideline_store.create_guideline(condition=c, action=None)
        await guideline_store.upsert_tag(
            guideline_id=g.id,
            tag_id=Tag.for_journey_id(journey_id=journey_id),
        )
        condition_ids.append(g.id)

    root_guideline = Guideline(
        id=GuidelineId("root"),
        creation_utc=datetime.now(timezone.utc),
        content=GuidelineContent(condition="", action=None),
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

    node_guidelines: Sequence[Guideline] = [
        Guideline(
            id=GuidelineId(node.id),
            creation_utc=datetime.now(timezone.utc),
            content=GuidelineContent(
                condition=node.condition or "",
                action=node.action,
            ),
            enabled=False,
            tags=[],
            metadata={
                "journey_node": {
                    "follow_ups": [
                        GuidelineId(follow_up_id) for follow_up_id in node.follow_up_ids
                    ],
                    "index": node.id,
                    "journey_id": journey_id,
                },
                "customer_dependent_action_data": {
                    "is_customer_dependent": node.customer_dependent_action,
                    "customer_action": node.customer_action or "",
                    "agent_action": "",
                },
                "tool_running_only": node.requires_tool_calls,
            },
        )
        for node in nodes
    ]

    journey = Journey(
        id=journey_id,
        root_id=JourneyNodeId(root_guideline.id),
        creation_utc=datetime.now(timezone.utc),
        description="",
        conditions=condition_ids,
        title=title,
        tags=[],
    )

    return journey, [root_guideline] + list(node_guidelines)


async def base_test_that_correct_node_is_selected(
    context: ContextOfTest,
    agent: Agent,
    session_id: SessionId,
    customer: Customer,
    conversation_context: list[tuple[EventSource, str]],
    journey_name: str,
    expected_next_node_index: str | Sequence[str] | None,
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

    journey, journey_node_guidelines = await create_journey(
        context=context,
        title=JOURNEYS_DICT[journey_name].title,
        nodes=JOURNEYS_DICT[journey_name].nodes,
        conditions=JOURNEYS_DICT[journey_name].conditions,
    )

    journey_node_selector = GenericJourneyNodeSelectionBatch(
        logger=context.logger,
        guideline_store=context.container[GuidelineStore],
        schematic_generator=context.schematic_generator,
        examined_journey=journey,
        node_guidelines=journey_node_guidelines,
        journey_path=journey_previous_path,
        optimization_policy=context.container[OptimizationPolicy],
        context=GuidelineMatchingContext(
            agent=agent,
            session=session,
            customer=customer,
            context_variables=[],
            interaction_history=interaction_history,
            terms=[],
            capabilities=capabilities,
            staged_events=staged_events,
            active_journeys=[],
            journey_paths=session.agent_states[-1]["journey_paths"] if session.agent_states else {},
        ),
    )
    result = await journey_node_selector.process()
    if len(result.matches) == 0:
        assert expected_next_node_index is None
    else:
        result_path: Sequence[str] = cast(list[str], result.matches[0].metadata["journey_path"])
        if expected_path:
            assert len(result_path) == len(expected_path)
            for result_node, expected_node in zip(result_path, expected_path):
                assert result_node == expected_node
        elif expected_next_node_index:  # Only test that the next node is correct
            if isinstance(expected_next_node_index, list):
                assert result_path[-1] in expected_next_node_index
            else:
                assert result_path[-1] == expected_next_node_index


async def test_that_journey_selector_repeats_node_if_incomplete_1(
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
    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="compliment_customer_journey",
        journey_previous_path=["1"],
        expected_next_node_index="1",
    )


async def test_that_journey_selector_repeats_node_if_incomplete_2(
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

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="calzone_journey",
        journey_previous_path=["1", "2", "7", "8"],
        expected_next_node_index="8",
    )


# 1 node advancement tests


async def test_that_journey_selector_correctly_advances_to_follow_up_node_1(
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
    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_previous_path=["1"],
        journey_name="compliment_customer_journey",
        expected_next_node_index="2",
    )


async def test_that_journey_selector_correctly_advances_to_follow_up_node_2(
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
    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="reset_password_journey",
        journey_previous_path=["1"],
        expected_next_node_index="2",
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
    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="reset_password_journey",
        journey_previous_path=["1", "2", "3"],
        expected_next_node_index=None,
    )


async def test_that_journey_selector_correctly_advances_to_follow_up_node_3(
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
            "Thank you, have a good day too! Now what's up with my password?",
        ),
    ]
    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="reset_password_journey",
        journey_previous_path=["1", "2", "3"],
        expected_next_node_index="5",
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

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="reset_password_journey",
        journey_previous_path=["1", "2", "3", "5"],
        expected_next_node_index="6",
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

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="reset_password_journey",
        journey_previous_path=["1", "2"],
        expected_next_node_index=None,
    )


# Multinode advancement tests


async def test_that_multinode_advancement_is_stopped_at_tool_requiring_nodes(
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

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="calzone_journey",
        journey_previous_path=["1"],
        expected_path=["1", "2", "7", "8", "9", "10"],
        expected_next_node_index="10",
    )


async def test_that_multinode_advancement_completes_and_exits_journey(
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

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="forgot_keys_journey",
        journey_previous_path=["1"],
        expected_next_node_index=None,
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

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="calzone_journey",
        journey_previous_path=["1", "2", "7", "8"],
        expected_next_node_index="8",  # Should return to asking about calzone type
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

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="calzone_journey",
        journey_previous_path=["1", "2", "7", "8", "9"],
        expected_next_node_index="3",  # Should go to node 3 (warn about delivery time for over 5 calzones)
    )


async def test_that_journey_selector_backtracks_and_fast_forwards_when_customer_changes_earlier_choice_1(
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
            "Large for all of them, please",
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

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="calzone_journey",
        journey_previous_path=["1", "2", "7", "8", "9", "10", "11"],
        expected_path=["8", "9", "10"],
        expected_next_node_index="10",  # Should check stock again
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

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="reset_password_journey",
        journey_previous_path=["1", "2", "3", "5", "7"],
        expected_next_node_index=["1", "2", "5"],
        staged_events=staged_events,
    )


async def test_that_multinode_advancement_is_stopped_at_node_that_requires_saying_something(  # Final decision is good, subpath it takes isn't
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

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="compliment_customer_journey",
        journey_previous_path=["1", "2"],
        expected_path=["2", "3", "4", "5"],
        expected_next_node_index="5",
    )


# TODO always stops too early - right at backtracking node
async def test_that_journey_selector_backtracks_and_fast_forwards_when_customer_changes_earlier_choice_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    """Test backtracking when customer changes calzone type mid-order, then fast forwards through size/drinks to stock check"""
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I'd like to order calzones",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome to the Low Cal Calzone Zone! How many calzones would you like?",
        ),
        (
            EventSource.CUSTOMER,
            "3 calzones please",
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
            "What size would you like - small, medium, or large?",
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
            "Yes, I'll take 2 sodas",
        ),
        (
            EventSource.AI_AGENT,
            "Great! Can you please confirm your order details? We have 3 medium spinach and ricotta calzones and 2 sodas.",
        ),
        (
            EventSource.CUSTOMER,
            "Actually, I want to change the calzone type for one of the orders to Chicken and Broccoli instead.",
        ),
    ]

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="calzone_journey",
        journey_previous_path=["1", "2", "7", "8", "9", "10", "11"],
        expected_path=[
            "7",
            "8",
            "9",
            "10",
        ],  # Backtrack to type selection, then fast forward through size/drinks to stock check
        expected_next_node_index="10",
    )


async def test_that_journey_selector_backtracks_and_fast_forwards_when_customer_changes_earlier_choice_3(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    """Test backtracking when customer changes account number after email was provided, then fast forwards through email collection"""
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
            "I just realized I gave you the wrong account number. It should be 987654, not 318475. My email is still john.doe@email.com though.",
        ),
    ]

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="reset_password_journey",
        journey_previous_path=["1", "2", "3"],
        expected_path=[
            "1",
            "2",
            "3",
        ],  # Backtrack to account collection, then fast forward through email to good day
        expected_next_node_index="3",
    )  # This test is slightly ambiguous, advancing to either node 3 or 5 (its followup) is considered valid, but we only test for node 3


async def test_that_journey_selector_backtracks_and_fast_forwards_when_customer_changes_earlier_choice_4(  # Sometimes skips a node in the returned path,  but outputs the correct decision
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
        (
            EventSource.AI_AGENT,
            "I apologize, but the password could not be reset at this time due to a system error.",
        ),
        (
            EventSource.CUSTOMER,
            "Oh wait, I think I gave you the wrong account number. It should be 987654, not 318475",
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

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="reset_password_journey",
        journey_previous_path=["1", "2", "3", "5", "7"],
        staged_events=staged_events,
        expected_path=[
            "1",
            "2",
            "3",
            "5",
        ],  # Backtrack to account collection, then fast forward through email to good day
        expected_next_node_index="5",
    )


# TODO sometimes passes, sometimes fails by fast forwards over the calzone type choice
async def test_that_journey_selector_does_not_fast_forward_when_earlier_customer_decision_no_longer_applies(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    """Test backtracking when customer changes calzone type mid-order, then fast forwards through size/drinks to stock check"""
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I'd like to order calzones",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome to the Low Cal Calzone Zone! How many calzones would you like?",
        ),
        (
            EventSource.CUSTOMER,
            "3 calzones please",
        ),
        (
            EventSource.AI_AGENT,
            "What type of calzones would you like? We have Classic Italian Calzone, Spinach and Ricotta Calzone, and Chicken and Broccoli Calzone.",
        ),
        (
            EventSource.CUSTOMER,
            "2 Spinach and Ricotta and 1 Italian please",
        ),
        (
            EventSource.AI_AGENT,
            "What size would you like - small, medium, or large?",
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
            "Yes, I'll take 2 sodas",
        ),
        (
            EventSource.AI_AGENT,
            "Great! Can you please confirm your order details? We have 2 medium spinach and ricotta calzones, one medium classic Italian and 2 sodas.",
        ),
        (
            EventSource.CUSTOMER,
            "Wait I got confused. I want 4 calzones please.",
        ),
    ]

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="calzone_journey",
        journey_previous_path=["1", "2", "7", "8", "9", "10", "11"],
        expected_path=[
            "2",
            "7",
        ],  # Backtrack to type selection, then fast forwards through number of calzones. Should stop at calzone type since
        expected_next_node_index="7",
    )


async def test_that_journey_selector_backtracks_back_does_not_fast_forward_upon_new_customer_request(
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
        (
            EventSource.AI_AGENT,
            "I'll now reset your password for account 318475.",
        ),
        (
            EventSource.CUSTOMER,
            "Wait! Actually, I want to reset my husband's password first - the info I'm looking for is under his account. I think his account number is 123655.",
        ),
    ]

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="reset_password_journey",
        journey_previous_path=["1", "2", "3", "5"],
        expected_path=[
            "1",
            "2",
        ],  # From tool execution node, back to account collection, then fast forward through email/good day to tool execution
        expected_next_node_index="2",
    )


async def test_that_journey_selector_correctly_advances_by_multiple_nodes(  # Occasionally fast-forwards by too little, to node 7 instead of 9
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

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="calzone_journey",
        journey_previous_path=["1"],
        expected_path=["1", "2", "7", "8", "9"],
        expected_next_node_index="9",
    )


async def test_that_fork_steps_are_correctly_traversed(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
):
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "google is not loading up",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! I'm sorry to hear that. Before we begin troubleshooting - how technically experienced are you?",
        ),
        (
            EventSource.CUSTOMER,
            "Not much, I just browse the internet on my iphone",
        ),
        (
            EventSource.AI_AGENT,
            "I see, that's not a problem. Can you describe the exact issue you're experiencing?",
        ),
        (
            EventSource.CUSTOMER,
            "I type in google.com, but it doesn't load up",
        ),
    ]

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="tech_experience_journey",
        journey_previous_path=["1", "2"],
        expected_path=["2", "3", "6"],
        expected_next_node_index="6",
    )


async def test_that_fork_steps_are_correctly_fast_forwarded_through(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
):
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I can't remember the password for my PC and I have no technological experience pls help me",
        ),
    ]

    await base_test_that_correct_node_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        conversation_context=conversation_context,
        journey_name="tech_experience_journey",
        journey_previous_path=[],
        expected_path=["1", "2", "4", "8"],
        expected_next_node_index="8",
    )
