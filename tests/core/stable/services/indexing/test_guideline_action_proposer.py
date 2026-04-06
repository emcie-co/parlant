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

from itertools import chain
from typing import Any
from lagom import Container
from pytest import fixture

from parlant.core.agents import Agent
from parlant.core.tracer import Tracer
from parlant.core.customers import Customer
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import GuidelineMatcher
from parlant.core.engines.alpha.engine_context import Interaction, EngineContext, ResponseState
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.types import Context
from parlant.core.guidelines import GuidelineContent
from parlant.core.loggers import Logger
from parlant.core.services.indexing.guideline_action_proposer import GuidelineActionProposer
from parlant.core.sessions import EventSource, Session, SessionId, SessionStore
from parlant.core.tools import LocalToolService, Tool, ToolId
from tests.core.common.engines.alpha.steps.tools import TOOLS
from tests.core.common.utils import create_event_message
from tests.core.stable.engines.alpha.test_guideline_matcher import (
    ContextOfTest,
    create_guideline,
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
        guidelines=list(),
        logger=container[Logger],
    )


async def test_that_no_action_is_proposed_when_guideline_already_contains_action_or_no_tools(
    container: Container,
) -> None:
    action_proposer = container[GuidelineActionProposer]

    guideline = GuidelineContent(
        condition="the customer greets the agent",
        action="reply with a greeting",
    )

    result = await action_proposer.propose_action(
        guideline=guideline,
        tool_ids=[],
    )

    assert result
    assert result.content == guideline
    assert result.rationale == "No action proposed"


async def test_that_action_is_proposed_when_guideline_lacks_action_and_tools_are_supplied(
    container: Container,
) -> None:
    local_tool_service = container[LocalToolService]

    dummy_tool = await local_tool_service.create_tool(
        name="dummy_tool",
        module_path="dummy.module",
        description="A dummy testing tool",
        parameters={},
        required=[],
    )

    guideline_without_action = GuidelineContent(
        condition="customer asks for something",
        action=None,
    )

    tool_id = ToolId(service_name="local", tool_name=dummy_tool.name)

    action_proposer = container[GuidelineActionProposer]

    result = await action_proposer.propose_action(
        guideline=guideline_without_action,
        tool_ids=[tool_id],
    )

    # Assertions: an action was proposed and it references the tool name
    assert result
    assert result.content.action is not None
    assert result.content.condition == guideline_without_action.condition


async def test_that_guideline_with_proposed_action_and_two_tools_is_matched_1(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    local_tool_service = context.container[LocalToolService]

    tool_names = ["get_available_drinks", "get_available_toppings"]
    condition = "the customer specifies toppings or drinks"
    conversation = [(EventSource.CUSTOMER, "Hey, can I order a large pepperoni pizza with Sprite?")]
    tools = [await local_tool_service.create_tool(**TOOLS[tool_name]) for tool_name in tool_names]
    await base_test_action_proposition(
        context, agent, new_session.id, customer, tools, conversation, condition
    )


async def test_that_guideline_with_proposed_action_and_two_tools_is_matched_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    local_tool_service = context.container[LocalToolService]

    tool_names = ["add", "multiply"]
    condition = "customers ask arithmetic questions"
    conversation = [
        (EventSource.CUSTOMER, "What is 8+2 and 4*6?"),
    ]
    tools = [await local_tool_service.create_tool(**TOOLS[tool_name]) for tool_name in tool_names]
    await base_test_action_proposition(
        context, agent, new_session.id, customer, tools, conversation, condition
    )


async def test_that_guideline_with_proposed_action_and_two_tools_is_matched_3(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    local_tool_service = context.container[LocalToolService]

    tool_names = ["consult_policy", "other_inquiries"]
    condition = "the user asks policy-related matters"
    conversation = [
        (EventSource.CUSTOMER, "I'd like to return a product please?"),
    ]
    tools = [await local_tool_service.create_tool(**TOOLS[tool_name]) for tool_name in tool_names]
    await base_test_action_proposition(
        context, agent, new_session.id, customer, tools, conversation, condition
    )


async def test_that_guideline_with_proposed_action_and_one_tool_is_matched_1(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    local_tool_service = context.container[LocalToolService]

    tool_names = ["get_account_balance"]
    condition = "customers inquire about account-related information"
    conversation = [
        (EventSource.CUSTOMER, "What's my account balance?"),
    ]
    tools = [await local_tool_service.create_tool(**TOOLS[tool_name]) for tool_name in tool_names]
    await base_test_action_proposition(
        context, agent, new_session.id, customer, tools, conversation, condition
    )


async def test_that_guideline_with_proposed_action_and_one_tool_is_matched_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    local_tool_service = context.container[LocalToolService]

    tool_names = ["get_available_drinks"]
    condition = "the customer specifies drinks"
    conversation = [
        (EventSource.CUSTOMER, "Hey, can I order a large pepperoni pizza with Sprite?"),
    ]
    tools = [await local_tool_service.create_tool(**TOOLS[tool_name]) for tool_name in tool_names]
    await base_test_action_proposition(
        context, agent, new_session.id, customer, tools, conversation, condition
    )


async def test_that_guideline_with_proposed_action_and_one_tool_is_matched_32(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    local_tool_service = context.container[LocalToolService]

    tool_names = ["pay_cc_bill"]
    condition = "they want to pay their credit card bill"
    conversation = [
        (EventSource.CUSTOMER, "Let's please pay my credit card bill"),
    ]
    tools = [await local_tool_service.create_tool(**TOOLS[tool_name]) for tool_name in tool_names]
    await base_test_action_proposition(
        context, agent, new_session.id, customer, tools, conversation, condition
    )


async def test_that_guideline_with_proposed_action_and_tool_name_not_informative_but_description_is(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    local_tool_service = context.container[LocalToolService]

    tool_names = ["other_inquiries"]
    condition = "the user asks policy-related matters like return of a product"
    conversation = [
        (EventSource.CUSTOMER, "I'd like to return a product please?"),
    ]
    tools = [await local_tool_service.create_tool(**TOOLS[tool_name]) for tool_name in tool_names]
    await base_test_action_proposition(
        context, agent, new_session.id, customer, tools, conversation, condition
    )


async def test_that_guideline_with_proposed_action_and_tool_with_no_description_is_matched(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    local_tool_service = context.container[LocalToolService]

    tool: dict[str, Any] = {
        "name": "update_status",
        "description": "",
        "module_path": "tests.tool_utilities",
        "parameters": {
            "ticket_id": {
                "type": "string",
                "description": "The ID of the support or issue ticket",
            },
            "new_status": {
                "type": "string",
                "description": "The new status to apply (e.g., 'resolved', 'in_progress', 'closed')",
            },
        },
        "required": ["ticket_id", "new_status"],
    }

    condition = "the customer wants to update status"
    conversation = [
        (
            EventSource.CUSTOMER,
            "Hey, I've finished with the task you gave me so yo can mark it as closed",
        ),
    ]
    tools = [await local_tool_service.create_tool(**tool)]
    await base_test_action_proposition(
        context, agent, new_session.id, customer, tools, conversation, condition
    )


async def base_test_action_proposition(
    context: ContextOfTest,
    agent: Agent,
    session_id: SessionId,
    customer: Customer,
    tools: list[Tool],
    conversation: list[tuple[EventSource, str]],
    condition: str,
) -> None:
    await base_test_that_guideline_with_proposed_action_matched(
        context, agent, session_id, customer, tools, conversation, condition
    )


async def base_test_that_guideline_with_proposed_action_matched(
    context: ContextOfTest,
    agent: Agent,
    session_id: SessionId,
    customer: Customer,
    tools: list[Tool],
    conversation_context: list[tuple[EventSource, str]],
    condition: str,
) -> None:
    action_proposer = context.container[GuidelineActionProposer]

    guideline_without_action = GuidelineContent(
        condition=condition,
        action=None,
    )

    result = await action_proposer.propose_action(
        guideline=guideline_without_action,
        tool_ids=[ToolId(service_name="local", tool_name=tool.name) for tool in tools],
    )

    assert result
    guideline_with_action = await create_guideline(
        context=context,
        condition=guideline_without_action.condition,
        action=result.content.action,
    )

    interaction_history = [
        create_event_message(
            offset=i,
            source=source,
            message=message,
        )
        for i, (source, message) in enumerate(conversation_context)
    ]

    session = await context.container[SessionStore].read_session(session_id)

    loaded_context = EngineContext(
        info=Context(
            session_id=session.id,
            agent_id=agent.id,
        ),
        logger=context.logger,
        tracer=context.container[Tracer],
        agent=agent,
        customer=customer,
        session=session,
        session_event_emitter=EventBuffer(agent),
        response_event_emitter=EventBuffer(agent),
        interaction=Interaction(events=interaction_history),
        state=ResponseState(
            context_variables=[],
            glossary_terms=set(),
            capabilities=[],
            iterations=[],
            ordinary_guideline_matches=[],
            tool_enabled_guideline_matches={},
            journeys=[],
            journey_paths={k: list(v) for k, v in session.agent_states[-1].journey_paths.items()}
            if session.agent_states
            else {},
            tool_events=[],
            tool_insights=ToolInsights(),
            prepared_to_respond=False,
            message_events=[],
        ),
    )

    guideline_matching_result = await context.container[GuidelineMatcher].match_guidelines(
        context=loaded_context,
        active_journeys=[],
        guidelines=context.guidelines,
    )

    guideline_matches = list(chain.from_iterable(guideline_matching_result.batches))

    matched_guidelines = [p.guideline for p in guideline_matches]
    assert set(matched_guidelines) == set([guideline_with_action])
