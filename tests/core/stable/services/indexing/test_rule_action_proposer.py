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

from typing import Any
import httpx
from lagom import Container
from pytest import fixture, raises

from parlant.core.agents import Agent
from parlant.core.tracer import Tracer
from parlant.core.customers import Customer
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.alpha.rule_matching.rule_matcher import RuleMatcher
from parlant.core.engines.alpha.engine_context import Interaction, EngineContext, ResponseState
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.types import Context
from parlant.core.rules import RuleContent
from parlant.core.loggers import Logger
from parlant.core.services.indexing.rule_action_proposer import RuleActionProposer
from parlant.core.services.indexing.common import EvaluationError
from parlant.core.sessions import EventSource, Session, SessionId, SessionStore
from parlant.core.tools import LocalToolService, Tool, ToolId
from tests.core.common.engines.alpha.steps.tools import TOOLS
from tests.core.common.utils import create_event_message
from tests.core.stable.engines.alpha.test_rule_matcher import (
    ContextOfTest,
    create_rule,
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
        rules=list(),
        logger=container[Logger],
    )


async def test_that_no_action_is_proposed_when_rule_already_contains_action_or_no_tools(
    container: Container,
) -> None:
    action_proposer = container[RuleActionProposer]

    rule = RuleContent(
        condition="the customer greets the agent",
        action="reply with a greeting",
    )

    result = await action_proposer.propose_action(
        rule=rule,
        tool_ids=[],
    )

    assert result is None


async def test_that_tool_connection_failures_have_clean_evaluation_error() -> None:
    class FailingToolService:
        url = "http://localhost:12345"

        async def read_tool(self, name: str) -> Tool:
            raise httpx.ConnectError("All connection attempts failed")

    class FakeServiceRegistry:
        async def read_tool_service(self, service_name: str) -> FailingToolService:
            return FailingToolService()

    class FakeStoreProvider:
        def get_store(self, *args: Any, **kwargs: Any) -> FakeServiceRegistry:
            return FakeServiceRegistry()

    action_proposer = object.__new__(RuleActionProposer)
    action_proposer._store_provider = FakeStoreProvider()

    with raises(EvaluationError) as exc_info:
        await action_proposer._load_tools(
            [ToolId(service_name="tau2_airline", tool_name="search_direct_flight")]
        )

    assert str(exc_info.value) == (
        "Could not read tool 'search_direct_flight' from tool service "
        "'tau2_airline' at http://localhost:12345 while evaluating a rule. "
        "Make sure the tool service is running and reachable before SDK startup evaluations run."
    )


async def test_that_action_is_proposed_when_rule_lacks_action_and_tools_are_supplied(
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

    rule_without_action = RuleContent(
        condition="customer asks for something",
        action=None,
    )

    tool_id = ToolId(service_name="local", tool_name=dummy_tool.name)

    action_proposer = container[RuleActionProposer]

    result = await action_proposer.propose_action(
        rule=rule_without_action,
        tool_ids=[tool_id],
    )

    # Assertions: an action was proposed and it references the tool name
    assert result
    assert result.content.action is not None
    assert result.content.condition == rule_without_action.condition


async def test_that_rule_with_proposed_action_and_two_tools_is_matched_1(
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


async def test_that_rule_with_proposed_action_and_two_tools_is_matched_2(
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


async def test_that_rule_with_proposed_action_and_two_tools_is_matched_3(
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


async def test_that_rule_with_proposed_action_and_one_tool_is_matched_1(
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


async def test_that_rule_with_proposed_action_and_one_tool_is_matched_2(
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


async def test_that_rule_with_proposed_action_and_one_tool_is_matched_32(
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


async def test_that_rule_with_proposed_action_and_tool_name_not_informative_but_description_is(
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


async def test_that_rule_with_proposed_action_and_tool_with_no_description_is_matched(
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
    await base_test_that_rule_with_proposed_action_matched(
        context, agent, session_id, customer, tools, conversation, condition
    )


async def base_test_that_rule_with_proposed_action_matched(
    context: ContextOfTest,
    agent: Agent,
    session_id: SessionId,
    customer: Customer,
    tools: list[Tool],
    conversation_context: list[tuple[EventSource, str]],
    condition: str,
) -> None:
    action_proposer = context.container[RuleActionProposer]

    rule_without_action = RuleContent(
        condition=condition,
        action=None,
    )

    result = await action_proposer.propose_action(
        rule=rule_without_action,
        tool_ids=[ToolId(service_name="local", tool_name=tool.name) for tool in tools],
    )

    assert result
    rule_with_action = await create_rule(
        context=context,
        condition=rule_without_action.condition,
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
            ordinary_rule_matches=[],
            tool_enabled_rule_matches={},
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

    rule_matching_result = await context.container[RuleMatcher].match_rules(
        context=loaded_context,
        active_journeys=[],
        rules=context.rules,
    )

    rule_matches = list(rule_matching_result.matched)

    matched_rules = [p.rule for p in rule_matches]
    assert set(matched_rules) == set([rule_with_action])
