# Copyright 2024 Emcie Co Ltd.
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

from datetime import datetime, timezone
from lagom import Container
from pytest import mark

from parlant.core.agents import AgentId
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import Session
from parlant.core.context_variables import ContextVariableStore
from parlant.core.engines.alpha.engine import fresh_value
from parlant.core.tools import LocalToolService, ToolId
from tests.core.engines.alpha.utils import ContextOfTest

now = datetime.now(timezone.utc)
base_time = datetime(now.year, now.month, now.day, 10, 0, 0, tzinfo=timezone.utc)


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
            base_time.replace(minute=14),
        ),
        (
            "0 6,12,18 * * *",
            base_time.replace(hour=11, minute=30),
        ),
        (
            f"0 9 * * {now.strftime('%a')}",
            base_time.replace(hour=8),
        ),
        (
            f"0 0 {base_time.day},{base_time.day + 1} * *",
            base_time.replace(day=base_time.day, hour=23),
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
    variable_set = "test_variable_set"

    test_key = "test-key"
    current_data = {"balance": 500.00}

    tool_id = ToolId(service_name="local", tool_name="fetch_account_balance")

    await create_fetch_account_balance_tool(context.container)

    context_variable_store = context.container[ContextVariableStore]
    service_registry = context.container[ServiceRegistry]

    context_variable = await context_variable_store.create_variable(
        variable_set=variable_set,
        name=variable_name,
        description="Customer's account balance",
        tool_id=tool_id,
        freshness_rules=freshness_rules,
    )

    await context_variable_store.update_value(
        variable_set=variable_set,
        variable_id=context_variable.id,
        key=test_key,
        data=current_data,
    )

    await fresh_value(
        context_variable_store=context_variable_store,
        service_registery=service_registry,
        agent_id=agent_id,
        session=new_session,
        variable_set=variable_set,
        variable_id=context_variable.id,
        key=test_key,
        current_time=current_time,
    )

    value = await context_variable_store.read_value(
        variable_set=variable_set,
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
            base_time.replace(minute=15),
        ),
        (
            "0 6,12,18 * * *",
            base_time.replace(hour=12, minute=0),
        ),
        (
            f"0 9 * * {now.strftime('%a')}",
            base_time.replace(hour=9, minute=0),
        ),
        (
            f"0 0 {base_time.day} * *",
            base_time.replace(day=base_time.day, hour=0, minute=0),
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
    variable_set = "test_variable_set"
    test_key = "test-key"
    tool_id = ToolId(service_name="local", tool_name="fetch_account_balance")

    await create_fetch_account_balance_tool(context.container)

    context_variable_store = context.container[ContextVariableStore]
    service_registry = context.container[ServiceRegistry]

    context_variable = await context_variable_store.create_variable(
        variable_set=variable_set,
        name=variable_name,
        description="Customer's account balance",
        tool_id=tool_id,
        freshness_rules=freshness_rules,
    )

    await fresh_value(
        context_variable_store=context_variable_store,
        service_registery=service_registry,
        agent_id=agent_id,
        session=new_session,
        variable_set=variable_set,
        variable_id=context_variable.id,
        key=test_key,
        current_time=current_time,
    )

    value = await context_variable_store.read_value(
        variable_set=variable_set,
        variable_id=context_variable.id,
        key=test_key,
    )
    assert value
    assert value.data == {"balance": 1000.00}


async def test_that_value_is_created_when_refresh_is_required(
    context: ContextOfTest,
    agent_id: AgentId,
    new_session: Session,
) -> None:
    variable_name = "AccountBalance"
    variable_set = "test_variable_set"
    test_key = "test-key"
    tool_id = ToolId(service_name="local", tool_name="fetch_account_balance")
    current_time = datetime.now(timezone.utc)

    await create_fetch_account_balance_tool(context.container)

    context_variable_store = context.container[ContextVariableStore]
    service_registry = context.container[ServiceRegistry]

    context_variable = await context_variable_store.create_variable(
        variable_set=variable_set,
        name=variable_name,
        description="Customer's account balance",
        tool_id=tool_id,
        freshness_rules=f"* * * {current_time.month} *",
    )

    await fresh_value(
        context_variable_store=context_variable_store,
        service_registery=service_registry,
        agent_id=agent_id,
        session=new_session,
        variable_set=variable_set,
        variable_id=context_variable.id,
        key=test_key,
        current_time=current_time,
    )

    value = await context_variable_store.read_value(
        variable_set=variable_set,
        variable_id=context_variable.id,
        key=test_key,
    )
    assert value
    assert value.data == {"balance": 1000.00}
