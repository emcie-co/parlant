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
from croniter import croniter
from lagom import Container
from pytest import mark

from parlant.core.agents import AgentId
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import Session
from parlant.core.context_variables import ContextVariableStore
from parlant.core.engines.alpha.engine import fresh_value
from parlant.core.tools import LocalToolService, ToolId
from tests.core.engines.alpha.utils import ContextOfTest


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
    service_registry = context.container[ServiceRegistry]

    context_variable = await context_variable_store.create_variable(
        variable_set=agent_id,
        name=variable_name,
        description="Customer's account balance",
        tool_id=tool_id,
        freshness_rules=freshness_rules,
    )

    await context_variable_store.update_value(
        variable_set=agent_id,
        variable_id=context_variable.id,
        key=test_key,
        data=current_data,
    )

    await fresh_value(
        context_variable_store=context_variable_store,
        service_registery=service_registry,
        agent_id=agent_id,
        session=new_session,
        variable_id=context_variable.id,
        key=test_key,
        current_time=current_time,
    )

    value = await context_variable_store.read_value(
        variable_set=agent_id,
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
            croniter("0,15,30,45 * * * *", datetime.now(timezone.utc)).get_next(datetime),
        ),
        (
            "0 0,6,12,18 * * *",
            croniter("0 0,6,12,18 * * *", datetime.now(timezone.utc)).get_next(datetime),
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
    current_data = {"balance": 500.00}
    tool_id = ToolId(service_name="local", tool_name="fetch_account_balance")

    await create_fetch_account_balance_tool(context.container)

    context_variable_store = context.container[ContextVariableStore]
    service_registry = context.container[ServiceRegistry]

    context_variable = await context_variable_store.create_variable(
        variable_set=agent_id,
        name=variable_name,
        description="Customer's account balance",
        tool_id=tool_id,
        freshness_rules=freshness_rules,
    )

    await context_variable_store.update_value(
        variable_set=agent_id,
        variable_id=context_variable.id,
        key=test_key,
        data=current_data,
    )

    value = await fresh_value(
        context_variable_store=context_variable_store,
        service_registery=service_registry,
        agent_id=agent_id,
        session=new_session,
        variable_id=context_variable.id,
        key=test_key,
        current_time=current_time,
    )

    assert value
    assert value.data == {"balance": 1000.00}


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
    service_registry = context.container[ServiceRegistry]

    context_variable = await context_variable_store.create_variable(
        variable_set=agent_id,
        name=variable_name,
        description="Customer's account balance",
        tool_id=tool_id,
    )

    created_value = await fresh_value(
        context_variable_store=context_variable_store,
        service_registery=service_registry,
        agent_id=agent_id,
        session=new_session,
        variable_id=context_variable.id,
        key=test_key,
        current_time=current_time,
    )

    stored_value = await context_variable_store.read_value(
        variable_set=agent_id,
        variable_id=context_variable.id,
        key=test_key,
    )
    assert stored_value == created_value
