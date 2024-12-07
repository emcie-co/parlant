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

from fastapi import status
from fastapi.testclient import TestClient
from lagom import Container
from pytest import fixture

from parlant.core.agents import AgentId
from parlant.core.context_variables import ContextVariableStore
from parlant.core.tools import LocalToolService, ToolId


@fixture
async def tool_id(container: Container) -> ToolId:
    service = container[LocalToolService]
    _ = await service.create_tool(
        name="test_tool",
        description="Test Description",
        module_path="test.module.path",
        parameters={"test_parameter": {"type": "string"}},
        required=["test_parameter"],
    )

    return ToolId("local", "test_tool")


async def test_that_context_variable_can_be_created(
    client: TestClient,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    freshness_rules = {
        "months": [5],
        "days_of_month": [14],
        "days_of_week": ["Thursday"],
        "hours": [18],
        "minutes": None,
        "seconds": None,
    }

    response = client.post(
        f"/agents/{agent_id}/context-variables",
        json={
            "name": "test_variable",
            "description": "test of context variable",
            "tool_id": {
                "service_name": tool_id.service_name,
                "tool_name": tool_id.tool_name,
            },
            "freshness_rules": freshness_rules,
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    context_variable = response.json()
    assert context_variable["name"] == "test_variable"
    assert context_variable["description"] == "test of context variable"
    assert context_variable["freshness_rules"] == freshness_rules


async def test_that_context_variable_can_be_updated(
    container: Container,
    client: TestClient,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    context_variable = await context_variable_store.create_variable(
        variable_set=agent_id,
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
    )

    new_name = "updated_test_variable"
    new_description = "updated test of variable"
    freshness_rules = {
        "months": [5],
        "days_of_month": [14],
        "days_of_week": ["Thursday"],
        "hours": [18],
        "minutes": None,
        "seconds": None,
    }

    context_variable_dto = (
        client.patch(
            f"/agents/{agent_id}/context-variables/{context_variable.id}",
            json={
                "name": new_name,
                "description": new_description,
                "freshness_rules": freshness_rules,
            },
        )
        .raise_for_status()
        .json()
    )

    assert context_variable_dto["name"] == new_name
    assert context_variable_dto["description"] == new_description
    assert context_variable_dto["freshness_rules"] == freshness_rules


async def test_that_invalid_freshness_rules_raise_error_when_creating_context_variable(
    client: TestClient,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    invalid_freshness_rules = "invalid cron expression"

    response = client.post(
        f"/agents/{agent_id}/context-variables",
        json={
            "name": "test_variable_invalid_cron",
            "description": "Test variable with invalid cron expression",
            "tool_id": {
                "service_name": tool_id.service_name,
                "tool_name": tool_id.tool_name,
            },
            "freshness_rules": invalid_freshness_rules,
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    error_response = response.json()
    assert "detail" in error_response
    assert (
        "Value error, the provided freshness_rules. contain an invalid cron expression."
        in error_response["detail"][0]["msg"]
    )


async def test_that_all_context_variables_can_be_deleted(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    _ = await context_variable_store.create_variable(
        variable_set=agent_id,
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
    )

    _ = await context_variable_store.create_variable(
        variable_set=agent_id,
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
    )

    vars = await context_variable_store.list_variables(variable_set=agent_id)
    assert len(vars) == 2

    response = client.delete(f"/agents/{agent_id}/context-variables")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    vars = await context_variable_store.list_variables(variable_set=agent_id)
    assert len(vars) == 0


async def test_that_context_variable_can_be_deleted(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    variable_to_delete = await context_variable_store.create_variable(
        variable_set=agent_id,
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
    )

    client.delete(
        f"/agents/{agent_id}/context-variables/{variable_to_delete.id}"
    ).raise_for_status()

    response = client.get(f"/agents/{agent_id}/context-variables/{variable_to_delete.id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_context_variables_can_be_listed(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    first_variable = await context_variable_store.create_variable(
        variable_set=agent_id,
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
        freshness_rules="0,15,30,45 * * * *",
    )

    second_variable = await context_variable_store.create_variable(
        variable_set=agent_id,
        name="second_test_variable",
        description=None,
        tool_id=tool_id,
        freshness_rules=None,
    )

    variables = client.get(f"/agents/{agent_id}/context-variables/").raise_for_status().json()
    assert len(variables) == 2

    assert first_variable.id == variables[0]["id"]
    assert second_variable.id == variables[1]["id"]

    assert first_variable.name == variables[0]["name"]
    assert second_variable.name == variables[1]["name"]

    assert first_variable.description == variables[0]["description"]
    assert second_variable.description == variables[1]["description"]

    assert first_variable.freshness_rules is not None
    assert second_variable.freshness_rules is None


async def test_that_context_variable_value_can_be_retrieved(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    variable = await context_variable_store.create_variable(
        variable_set=agent_id,
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
    )

    key = "test_key"
    data = {"value": 42}

    _ = await context_variable_store.update_value(
        variable_set=agent_id,
        variable_id=variable.id,
        key=key,
        data=data,
    )

    value = (
        client.get(f"/agents/{agent_id}/context-variables/{variable.id}/{key}")
        .raise_for_status()
        .json()
    )

    assert value["data"] == data


async def test_that_context_variable_value_can_be_set(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    variable = await context_variable_store.create_variable(
        variable_set=agent_id,
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
        freshness_rules=None,
    )

    key = "yam_choock"
    data = {"zen_level": 5000}

    value = (
        client.put(
            f"/agents/{agent_id}/context-variables/{variable.id}/{key}",
            json={"data": data},
        )
        .raise_for_status()
        .json()
    )

    assert value["data"] == data

    data = {"zen_level": 9000}
    value = (
        client.put(
            f"/agents/{agent_id}/context-variables/{variable.id}/{key}",
            json={"data": data},
        )
        .raise_for_status()
        .json()
    )

    assert value["data"] == data


async def test_that_context_variable_values_can_be_listed(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    variable = await context_variable_store.create_variable(
        variable_set=agent_id,
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
    )

    keys_and_data = {
        "key1": {"value": 1},
        "key2": {"value": 2},
        "key3": {"value": 3},
    }

    for key, data in keys_and_data.items():
        _ = await context_variable_store.update_value(
            variable_set=agent_id,
            variable_id=variable.id,
            key=key,
            data=data,
        )

    response = client.get(f"/agents/{agent_id}/context-variables/{variable.id}")
    assert response.status_code == status.HTTP_200_OK

    retrieved_variable = response.json()["context_variable"]
    assert retrieved_variable["id"] == variable.id
    assert retrieved_variable["name"] == "test_variable"
    assert retrieved_variable["description"] == "test variable"

    retrieved_values = response.json()["key_value_pairs"]

    assert len(retrieved_values) == len(keys_and_data)
    for key in keys_and_data:
        assert key in retrieved_values
        assert retrieved_values[key]["data"] == keys_and_data[key]


async def test_that_context_variable_value_can_be_deleted(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    variable = await context_variable_store.create_variable(
        variable_set=agent_id,
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
    )

    key = "yam_choock"
    data = {"zen_level": 9000}

    response = client.put(
        f"/agents/{agent_id}/context-variables/{variable.id}/{key}",
        json={"data": data},
    )

    variable_value = response.json()
    assert variable_value["data"] == data
    assert "last_modified" in variable_value

    client.delete(f"/agents/{agent_id}/context-variables/{variable.id}/{key}")

    response = client.get(f"/agents/{agent_id}/context-variables/{variable.id}/{key}")
    assert response.status_code == status.HTTP_404_NOT_FOUND
