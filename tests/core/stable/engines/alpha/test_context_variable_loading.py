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

from datetime import datetime, timedelta, timezone
from croniter import croniter
from lagom import Container
from pytest import mark

from parlant.core.agents import AgentId
from parlant.core.sessions import Session
from parlant.core.context_variables import ContextVariableStore
from parlant.core.engines.alpha.engine import load_fresh_context_variable_value
from parlant.core.tags import Tag
from parlant.core.tools import LocalToolService, ToolId
from parlant.core.entity_cq import EntityQueries, EntityCommands

from tests.core.common.utils import ContextOfTest


async def create_fetch_account_balance_tool(container: Container) -> None:
    service = container[LocalToolService]

    await service.create_tool(
        name="fetch_account_balance",
        description="Fetch Account Balance",
        module_path="tests.tool_utilities",
        parameters={},
        required=[],
    )


@mark.parametrize(
    "freshness_rules, current_time",
    [
        (
            "0,15,30,45 * * * *",
            datetime.now(timezone.utc).replace(minute=14),
        ),
        (
            "0 6,12,18 * * *",
            datetime.now(timezone.utc).replace(hour=11, minute=30),
        ),
        (
            f"0 9 * * {datetime.now(timezone.utc).strftime('%a')}",
            datetime.now(timezone.utc).replace(hour=8),
        ),
        (
            f"0 0 {datetime.now(timezone.utc).day},{datetime.now(timezone.utc).day + 1} * *",
            datetime.now(timezone.utc).replace(day=datetime.now(timezone.utc).day, hour=23),
        ),
    ],
)
async def test_that_value_is_not_refreshed_when_freshness_rules_are_not_met(
    freshness_rules: str,
    current_time: datetime,
    context: ContextOfTest,
    agent_id: AgentId,
    new_session: Session,
) -> None:
    variable_name = "AccountBalance"
    test_key = "test-key"
    current_data = {"balance": 500.00}
    tool_id = ToolId(service_name="local", tool_name="fetch_account_balance")

    await create_fetch_account_balance_tool(context.container)

    context_variable_store = context.container[ContextVariableStore]
    entity_queries = context.container[EntityQueries]
    entity_commands = context.container[EntityCommands]

    context_variable = await context_variable_store.create_variable(
        name=variable_name,
        description="Customer's account balance",
        tool_id=tool_id,
        freshness_rules=freshness_rules,
    )

    await context_variable_store.add_variable_tag(
        variable_id=context_variable.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    await context_variable_store.update_value(
        variable_id=context_variable.id,
        key=test_key,
        data=current_data,
    )

    await load_fresh_context_variable_value(
        entity_queries=entity_queries,
        entity_commands=entity_commands,
        agent_id=agent_id,
        session=new_session,
        variable=context_variable,
        key=test_key,
        current_time=current_time,
    )

    value = await context_variable_store.read_value(
        variable_id=context_variable.id,
        key=test_key,
    )
    assert value
    assert value.data == {"balance": 500.00}


@mark.parametrize(
    "freshness_rules, current_time",
    [
        (
            "0,15,30,45 * * * *",
            croniter(
                "0,15,30,45 * * * *", datetime.now(timezone.utc) + timedelta(minutes=1)
            ).get_next(datetime),
        ),
        (
            "0 0,6,12,18 * * *",
            croniter(
                "0 0,6,12,18 * * *", datetime.now(timezone.utc) + timedelta(minutes=1)
            ).get_next(datetime),
        ),
    ],
)
async def test_that_value_refreshes_when_freshness_rules_are_met(
    freshness_rules: str,
    current_time: datetime,
    agent_id: AgentId,
    new_session: Session,
    context: ContextOfTest,
) -> None:
    variable_name = "AccountBalance"
    test_key = "test-key"
    current_data = {"balance": 500.0}
    tool_id = ToolId(service_name="local", tool_name="fetch_account_balance")

    await create_fetch_account_balance_tool(context.container)

    context_variable_store = context.container[ContextVariableStore]
    entity_queries = context.container[EntityQueries]
    entity_commands = context.container[EntityCommands]

    context_variable = await context_variable_store.create_variable(
        name=variable_name,
        description="Customer's account balance",
        tool_id=tool_id,
        freshness_rules=freshness_rules,
    )

    await context_variable_store.add_variable_tag(
        variable_id=context_variable.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    await context_variable_store.update_value(
        variable_id=context_variable.id,
        key=test_key,
        data=current_data,
    )

    value = await load_fresh_context_variable_value(
        entity_queries=entity_queries,
        entity_commands=entity_commands,
        agent_id=agent_id,
        session=new_session,
        variable=context_variable,
        key=test_key,
        current_time=current_time,
    )

    assert value
    assert value.data == {"balance": 1000.0}


async def test_that_value_is_created_when_need_to_be_freshed(
    context: ContextOfTest,
    agent_id: AgentId,
    new_session: Session,
) -> None:
    variable_name = "AccountBalance"
    test_key = "test-key"
    tool_id = ToolId(service_name="local", tool_name="fetch_account_balance")
    current_time = datetime.now(timezone.utc)

    await create_fetch_account_balance_tool(context.container)

    context_variable_store = context.container[ContextVariableStore]
    entity_queries = context.container[EntityQueries]
    entity_commands = context.container[EntityCommands]

    context_variable = await context_variable_store.create_variable(
        name=variable_name,
        description="Customer's account balance",
        tool_id=tool_id,
    )

    await context_variable_store.add_variable_tag(
        variable_id=context_variable.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    created_value = await load_fresh_context_variable_value(
        entity_queries=entity_queries,
        entity_commands=entity_commands,
        agent_id=agent_id,
        session=new_session,
        variable=context_variable,
        key=test_key,
        current_time=current_time,
    )

    stored_value = await context_variable_store.read_value(
        variable_id=context_variable.id,
        key=test_key,
    )
    assert stored_value == created_value
