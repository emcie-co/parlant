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

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from pytest import mark

from parlant.core.agents import AgentId
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import Session
from parlant.core.context_variables import (
    ContextVariableValueId,
    ContextVariableStore,
    ContextVariableValue,
    FreshnessRules,
)
from parlant.core.engines.alpha.engine import fresh_value
from parlant.core.tools import ToolId
from tests.core.engines.alpha.utils import ContextOfTest


@mark.parametrize(
    "freshness_rules, last_modified_delta",
    [
        (
            FreshnessRules(
                months=None,
                days_of_month=None,
                days_of_week=None,
                hours=None,
                minutes=[15, 30, 45],
                seconds=None,
            ),
            timedelta(minutes=10),  # Fresh: Within 15 minute interval
        ),
        (
            FreshnessRules(
                months=None,
                days_of_month=None,
                days_of_week=None,
                hours=[6, 12, 18],
                minutes=[0, 30],
                seconds=None,
            ),
            timedelta(minutes=25),  # Fresh: Within 30 minute interval
        ),
        (
            FreshnessRules(
                months=None,
                days_of_month=None,
                days_of_week=["Monday", "Thursday"],
                hours=[9],
                minutes=[0],
                seconds=None,
            ),
            timedelta(hours=2),  # Fresh: Same day, before next check
        ),
        (
            FreshnessRules(
                months=None,
                days_of_month=[1, 15],
                days_of_week=None,
                hours=[0],
                minutes=[0],
                seconds=None,
            ),
            timedelta(days=7),  # Fresh: Within first half of month
        ),
        (
            FreshnessRules(
                months=[1, 7],
                days_of_month=[1],
                days_of_week=["Monday"],
                hours=[9, 17],
                minutes=[30],
                seconds=None,
            ),
            timedelta(hours=5),  # Fresh: Same day, between checks
        ),
    ],
)
async def test_that_context_variable_not_refreshed_when_fresh(
    freshness_rules: FreshnessRules,
    last_modified_delta: timedelta,
    context: ContextOfTest,
    agent_id: AgentId,
    new_session: Session,
) -> None:
    variable_name = "AccountBalance"
    variable_set = "TestVariableSet"
    test_key = "test-key"
    tool_name = "GetAccountBalanceTool"
    tool_id = ToolId(service_name="local", tool_name=tool_name)
    last_modified = datetime.now(timezone.utc) - last_modified_delta
    initial_value = {"balance": 500.00}

    mock_tool_result = {"balance": 1000.00}
    mock_tool = AsyncMock(return_value=mock_tool_result)
    context.tools = {tool_name: mock_tool}

    mock_value = ContextVariableValue(
        id=ContextVariableValueId("test_value_id"),
        last_modified=last_modified,
        data=initial_value,
    )

    with (
        patch.object(
            ContextVariableStore, "call_tool", new=AsyncMock(return_value=mock_tool_result)
        ) as mock_call_tool,
        patch.object(ContextVariableStore, "read_value", new=AsyncMock(return_value=mock_value)),
    ):
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
            session_id=new_session.id,
            variable_set=variable_set,
            variable_id=context_variable.id,
            key=test_key,
        )

        mock_call_tool.assert_not_awaited()

        value = await context_variable_store.read_value(
            variable_set=variable_set,
            variable_id=context_variable.id,
            key=test_key,
        )

        assert value
        assert value.data == initial_value


@mark.parametrize(
    "freshness_rules, last_modified_delta",
    [
        (
            FreshnessRules(
                months=None,
                days_of_month=None,
                days_of_week=None,
                hours=None,
                minutes=[15, 30, 45],
                seconds=None,
            ),
            timedelta(minutes=20),  # Stale: Past 15 minute mark
        ),
        (
            FreshnessRules(
                months=None,
                days_of_month=None,
                days_of_week=None,
                hours=[6, 12, 18],
                minutes=[0],
                seconds=None,
            ),
            timedelta(hours=8),  # Stale: Past 6-hour check
        ),
        (
            FreshnessRules(
                months=None,
                days_of_month=None,
                days_of_week=["Monday", "Wednesday", "Friday"],
                hours=[9],
                minutes=[0],
                seconds=None,
            ),
            timedelta(days=3),  # Stale: Past the next scheduled day
        ),
        (
            FreshnessRules(
                months=None,
                days_of_month=[1, 15, 28],
                days_of_week=None,
                hours=[0],
                minutes=[0],
                seconds=None,
            ),
            timedelta(days=17),  # Stale: Past mid-month check
        ),
        (
            FreshnessRules(
                months=[3, 6, 9, 12],
                days_of_month=[1],
                days_of_week=None,
                hours=[12],
                minutes=[0],
                seconds=None,
            ),
            timedelta(days=92),  # Stale: Past quarterly check
        ),
    ],
)
async def test_that_context_variable_refreshes_when_stale(
    freshness_rules: FreshnessRules,
    last_modified_delta: timedelta,
    agent_id: AgentId,
    new_session: Session,
    context: ContextOfTest,
) -> None:
    variable_name = "AccountBalance"
    variable_set = "TestVariableSet"
    test_key = "test-key"
    tool_name = "GetAccountBalanceTool"
    tool_id = ToolId(service_name="local", tool_name=tool_name)
    last_modified = datetime.now(timezone.utc) - last_modified_delta
    initial_value = {"balance": 500.00}

    mock_tool_result = {"balance": 1000.00}
    mock_tool = AsyncMock(return_value=mock_tool_result)
    context.tools = {tool_name: mock_tool}

    mock_value = ContextVariableValue(
        id=ContextVariableValueId("test_value_id"),
        last_modified=last_modified,
        data=initial_value,
    )

    with (
        patch.object(
            ContextVariableStore, "call_tool", new=AsyncMock(return_value=mock_tool_result)
        ) as mock_call_tool,
        patch.object(ContextVariableStore, "read_value", new=AsyncMock(return_value=mock_value)),
    ):
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
            session_id=new_session.id,
            variable_set=variable_set,
            variable_id=context_variable.id,
            key=test_key,
        )

        mock_call_tool.assert_awaited_once_with(tool_id)

        value = await context_variable_store.read_value(
            variable_set=variable_set,
            variable_id=context_variable.id,
            key=test_key,
        )

        assert value
        assert value.data == mock_tool_result
