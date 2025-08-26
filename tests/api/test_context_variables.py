# Copyright 2025 Emcie Co Ltd.
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
import httpx
from lagom import Container
from pytest import fixture

from parlant.core.agents import AgentId
from parlant.core.context_variables import ContextVariableStore
from parlant.core.tags import Tag, TagId, TagStore
from parlant.core.tools import LocalToolService, ToolId, ToolOverlap


@fixture
async def tool_id(container: Container) -> ToolId:
    service = container[LocalToolService]
    _ = await service.create_tool(
        name="test_tool",
        description="Test Description",
        module_path="test.module.path",
        parameters={"test_parameter": {"type": "string"}},
        required=["test_parameter"],
        overlap=ToolOverlap.NONE,
    )

    return ToolId("local", "test_tool")


async def test_legacy_that_context_variable_can_be_created(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    freshness_rules = "0 18 14 5 4"

    response = await async_client.post(
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


async def test_legacy_that_context_variable_can_be_updated(
    container: Container,
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    context_variable = await context_variable_store.create_variable(
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
    )

    await context_variable_store.add_variable_tag(
        variable_id=context_variable.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    new_name = "updated_test_variable"
    new_description = "updated test of variable"

    context_variable_dto = (
        (
            await async_client.patch(
                f"/agents/{agent_id}/context-variables/{context_variable.id}",
                json={
                    "name": new_name,
                    "description": new_description,
                },
            )
        )
        .raise_for_status()
        .json()
    )

    assert context_variable_dto["name"] == new_name
    assert context_variable_dto["description"] == new_description


async def test_legacy_that_context_variable_can_be_updated_with_a_valid_freshness_rules(
    container: Container,
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    context_variable = await context_variable_store.create_variable(
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
    )

    await context_variable_store.add_variable_tag(
        variable_id=context_variable.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    new_name = "updated_test_variable"
    new_description = "updated test of variable"
    freshness_rules = "0 18 14 5 4"

    context_variable_dto = (
        (
            await async_client.patch(
                f"/agents/{agent_id}/context-variables/{context_variable.id}",
                json={
                    "name": new_name,
                    "description": new_description,
                    "freshness_rules": freshness_rules,
                },
            )
        )
        .raise_for_status()
        .json()
    )

    assert context_variable_dto["name"] == new_name
    assert context_variable_dto["description"] == new_description
    assert context_variable_dto["freshness_rules"] == freshness_rules


async def test_legacy_that_invalid_freshness_rules_raise_error_when_updating_context_variable(
    container: Container,
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    context_variable = await context_variable_store.create_variable(
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
    )

    await context_variable_store.add_variable_tag(
        variable_id=context_variable.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    new_name = "updated_test_variable"
    new_description = "updated test of variable"
    freshness_rules = "Invalid freshness"

    response = await async_client.patch(
        f"/agents/{agent_id}/context-variables/{context_variable.id}",
        json={
            "name": new_name,
            "description": new_description,
            "freshness_rules": freshness_rules,
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    error_response = response.json()
    assert "detail" in error_response
    assert (
        error_response["detail"]
        == "the provided freshness_rules. contain an invalid cron expression."
    )


async def test_legacy_that_invalid_freshness_rules_raise_error_when_creating_context_variable(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    invalid_freshness_rules = "invalid cron expression"

    response = await async_client.post(
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
        error_response["detail"]
        == "the provided freshness_rules. contain an invalid cron expression."
    )


async def test_legacy_that_all_context_variables_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    first_variable = await context_variable_store.create_variable(
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
    )

    await context_variable_store.add_variable_tag(
        variable_id=first_variable.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    second_variable = await context_variable_store.create_variable(
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
    )

    await context_variable_store.add_variable_tag(
        variable_id=second_variable.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    vars = await context_variable_store.list_variables(tags=[Tag.for_agent_id(agent_id)])
    assert len(vars) == 2

    response = await async_client.delete(f"/agents/{agent_id}/context-variables")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    vars = await context_variable_store.list_variables(tags=[Tag.for_agent_id(agent_id)])
    assert len(vars) == 0


async def test_legacy_that_context_variable_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    variable_to_delete = await context_variable_store.create_variable(
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
    )

    await context_variable_store.add_variable_tag(
        variable_id=variable_to_delete.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    (
        await async_client.delete(f"/agents/{agent_id}/context-variables/{variable_to_delete.id}")
    ).raise_for_status()

    response = await async_client.get(
        f"/agents/{agent_id}/context-variables/{variable_to_delete.id}"
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_legacy_that_context_variables_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    first_variable = await context_variable_store.create_variable(
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
        freshness_rules="0,15,30,45 * * * *",
    )

    second_variable = await context_variable_store.create_variable(
        name="second_test_variable",
        description=None,
        tool_id=tool_id,
        freshness_rules=None,
    )

    await context_variable_store.add_variable_tag(
        variable_id=first_variable.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    await context_variable_store.add_variable_tag(
        variable_id=second_variable.id,
        tag_id=Tag.for_agent_id(agent_id),
    )
    variables = (
        (await async_client.get(f"/agents/{agent_id}/context-variables")).raise_for_status().json()
    )
    assert len(variables) == 2

    first_json_variable = next(
        variable for variable in variables if variable["id"] == first_variable.id
    )
    second_json_variable = next(
        variable for variable in variables if variable["id"] == second_variable.id
    )

    assert first_variable.id == first_json_variable["id"]
    assert second_variable.id == second_json_variable["id"]

    assert first_variable.name == first_json_variable["name"]
    assert second_variable.name == second_json_variable["name"]

    assert first_variable.description == first_json_variable["description"]
    assert second_variable.description == second_json_variable["description"]

    assert first_variable.freshness_rules is not None
    assert second_variable.freshness_rules is None


async def test_legacy_that_context_variable_value_can_be_retrieved(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    variable = await context_variable_store.create_variable(
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
    )

    await context_variable_store.add_variable_tag(
        variable_id=variable.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    key = "test_key"
    data = {"value": 42}

    _ = await context_variable_store.update_value(
        variable_id=variable.id,
        key=key,
        data=data,
    )

    value = (
        (await async_client.get(f"/agents/{agent_id}/context-variables/{variable.id}/{key}"))
        .raise_for_status()
        .json()
    )

    assert value["data"] == data


async def test_legacy_that_context_variable_value_can_be_set(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    variable = await context_variable_store.create_variable(
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
        freshness_rules=None,
    )

    await context_variable_store.add_variable_tag(
        variable_id=variable.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    key = "yam_choock"
    data = {"zen_level": 5000}

    value = (
        (
            await async_client.put(
                f"/agents/{agent_id}/context-variables/{variable.id}/{key}",
                json={"data": data},
            )
        )
        .raise_for_status()
        .json()
    )

    assert value["data"] == data

    data = {"zen_level": 9000}
    value = (
        (
            await async_client.put(
                f"/agents/{agent_id}/context-variables/{variable.id}/{key}",
                json={"data": data},
            )
        )
        .raise_for_status()
        .json()
    )

    assert value["data"] == data


async def test_legacy_that_context_variable_values_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    variable = await context_variable_store.create_variable(
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
    )

    await context_variable_store.add_variable_tag(
        variable_id=variable.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    keys_and_data = {
        "key1": {"value": 1},
        "key2": {"value": 2},
        "key3": {"value": 3},
    }

    for key, data in keys_and_data.items():
        _ = await context_variable_store.update_value(
            variable_id=variable.id,
            key=key,
            data=data,
        )

    response = await async_client.get(f"/agents/{agent_id}/context-variables/{variable.id}")
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


async def test_legacy_that_context_variable_value_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    variable = await context_variable_store.create_variable(
        name="test_variable",
        description="test variable",
        tool_id=tool_id,
    )

    await context_variable_store.add_variable_tag(
        variable_id=variable.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    key = "yam_choock"
    data = {"zen_level": 9000}

    response = await async_client.put(
        f"/agents/{agent_id}/context-variables/{variable.id}/{key}",
        json={"data": data},
    )

    variable_value = response.json()
    assert variable_value["data"] == data
    assert "last_modified" in variable_value

    (await async_client.delete(f"/agents/{agent_id}/context-variables/{variable.id}/{key}"))

    response = await async_client.get(f"/agents/{agent_id}/context-variables/{variable.id}/{key}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_legacy_that_reading_context_variable_with_wrong_agent_id_returns_404(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    variable = await context_variable_store.create_variable(
        name="test_variable",
        description="test variable",
        tool_id=ToolId("local", "test_tool"),
    )

    await context_variable_store.add_variable_tag(
        variable_id=variable.id,
        tag_id=Tag.for_agent_id(AgentId("wrong_agent_id")),
    )

    response = await async_client.get(f"/agents/{agent_id}/context-variables/{variable.id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_legacy_that_updating_context_variable_with_wrong_agent_id_returns_404(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    variable = await context_variable_store.create_variable(
        name="test_variable",
        description="test variable",
        tool_id=ToolId("local", "test_tool"),
    )

    await context_variable_store.add_variable_tag(
        variable_id=variable.id,
        tag_id=Tag.for_agent_id("wrong_agent_id"),
    )

    response = await async_client.patch(
        f"/agents/{agent_id}/context-variables/{variable.id}",
        json={"description": "updated description"},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_legacy_that_deleting_context_variable_with_wrong_agent_id_returns_404(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    variable = await context_variable_store.create_variable(
        name="test_variable",
        description="test variable",
        tool_id=ToolId("local", "test_tool"),
    )

    await context_variable_store.add_variable_tag(
        variable_id=variable.id,
        tag_id=Tag.for_agent_id(AgentId("wrong_agent_id")),
    )

    response = await async_client.delete(f"/agents/{agent_id}/context-variables/{variable.id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_a_context_variable_can_be_created(
    async_client: httpx.AsyncClient,
    tool_id: ToolId,
) -> None:
    freshness_rules = "0 18 14 5 4"

    response = await async_client.post(
        "/context-variables",
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
    assert context_variable["tags"] == []


async def test_that_a_context_variable_can_be_created_with_tags(
    async_client: httpx.AsyncClient,
    container: Container,
    tool_id: ToolId,
) -> None:
    tag_store = container[TagStore]
    tag1 = await tag_store.create_tag("tag1")
    tag2 = await tag_store.create_tag("tag2")

    response = await async_client.post(
        "/context-variables",
        json={
            "name": "test_variable",
            "description": "test of context variable",
            "tool_id": {
                "service_name": tool_id.service_name,
                "tool_name": tool_id.tool_name,
            },
            "tags": [tag1.id, tag1.id, tag2.id],
        },
    )
    assert response.status_code == status.HTTP_201_CREATED

    context_variable_dto = (
        (await async_client.get(f"/context-variables/{response.json()['id']}"))
        .raise_for_status()
        .json()
    )

    assert len(context_variable_dto["context_variable"]["tags"]) == 2
    assert set(context_variable_dto["context_variable"]["tags"]) == {tag1.id, tag2.id}


async def test_that_a_context_variable_can_be_read(
    async_client: httpx.AsyncClient,
    container: Container,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    name = "test_variable"
    description = "test of context variable"
    freshness_rules = "0 18 14 5 4"

    variable = await context_variable_store.create_variable(
        name=name,
        description=description,
        tool_id=tool_id,
        freshness_rules=freshness_rules,
    )

    read_response = await async_client.get(f"/context-variables/{variable.id}")
    assert read_response.status_code == status.HTTP_200_OK

    data = read_response.json()
    context_variable_dto = data["context_variable"]
    assert context_variable_dto["id"] == variable.id
    assert context_variable_dto["name"] == name
    assert context_variable_dto["description"] == description
    assert context_variable_dto["freshness_rules"] == freshness_rules
    assert context_variable_dto["tags"] == []


async def test_that_context_variables_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    first_variable = await context_variable_store.create_variable(
        name="variable1",
        description="description 1",
        tool_id=tool_id,
        freshness_rules="0 18 14 5 4",
    )

    second_variable = await context_variable_store.create_variable(
        name="variable2",
        description="description 2",
        tool_id=tool_id,
    )

    returned_variables = (await async_client.get("/context-variables")).raise_for_status().json()

    assert len(returned_variables) >= 2
    first_variable_dto = next(v for v in returned_variables if v["id"] == first_variable.id)
    second_variable_dto = next(v for v in returned_variables if v["id"] == second_variable.id)

    assert first_variable_dto["name"] == first_variable.name
    assert second_variable_dto["name"] == second_variable.name

    assert first_variable_dto["description"] == first_variable.description
    assert second_variable_dto["description"] == second_variable.description

    assert first_variable_dto["freshness_rules"] == first_variable.freshness_rules
    assert second_variable_dto["freshness_rules"] == second_variable.freshness_rules


async def test_that_a_context_variable_can_be_updated_with_new_values(
    async_client: httpx.AsyncClient,
    container: Container,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]
    tag_store = container[TagStore]

    tag1 = await tag_store.create_tag("tag1")
    tag2 = await tag_store.create_tag("tag2")

    name = "test_variable"
    description = "test of context variable"

    variable = await context_variable_store.create_variable(
        name=name,
        description=description,
        tool_id=tool_id,
    )

    updated_name = "updated_test_variable"
    updated_description = "updated test of variable"
    freshness_rules = "0 18 14 5 4"
    tags_to_add = [tag1.id, tag2.id]

    update_response = await async_client.patch(
        f"/context-variables/{variable.id}",
        json={
            "name": updated_name,
            "description": updated_description,
            "freshness_rules": freshness_rules,
            "tags": {
                "add": tags_to_add,
            },
        },
    )

    assert update_response.status_code == status.HTTP_200_OK

    data = update_response.json()
    assert data["name"] == updated_name
    assert data["description"] == updated_description
    assert data["freshness_rules"] == freshness_rules
    assert set(data["tags"]) == set(tags_to_add)


async def test_that_tags_can_be_removed_from_a_context_variable(
    async_client: httpx.AsyncClient,
    container: Container,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    name = "test_variable"
    description = "test of context variable"

    variable = await context_variable_store.create_variable(
        name=name,
        description=description,
        tool_id=tool_id,
    )

    await context_variable_store.add_variable_tag(
        variable_id=variable.id,
        tag_id=TagId("tag1"),
    )

    await context_variable_store.add_variable_tag(
        variable_id=variable.id,
        tag_id=TagId("tag2"),
    )

    update_response = await async_client.patch(
        f"/context-variables/{variable.id}",
        json={
            "tags": {
                "remove": ["tag1"],
            },
        },
    )

    assert update_response.status_code == status.HTTP_200_OK
    data = update_response.json()
    assert set(data["tags"]) == {"tag2"}


async def test_that_a_context_variable_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    name = "test_variable"
    description = "test of context variable"

    variable = await context_variable_store.create_variable(
        name=name,
        description=description,
        tool_id=tool_id,
    )

    (await async_client.delete(f"/context-variables/{variable.id}")).raise_for_status()

    read_response = await async_client.get(f"/context-variables/{variable.id}")
    assert read_response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_context_variable_value_can_be_set_and_retrieved(
    async_client: httpx.AsyncClient,
    container: Container,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    name = "test_variable"
    description = "test of context variable"

    variable = await context_variable_store.create_variable(
        name=name,
        description=description,
        tool_id=tool_id,
    )

    key = "test_key"
    data = {"value": 42}

    (
        await async_client.put(
            f"/context-variables/{variable.id}/{key}",
            json={"data": data},
        )
    ).raise_for_status()

    retrieved_value = (
        (await async_client.get(f"/context-variables/{variable.id}/{key}"))
        .raise_for_status()
        .json()
    )

    assert retrieved_value["data"] == data

    retrieved_value = (
        (await async_client.get(f"/context-variables/{variable.id}/{key}"))
        .raise_for_status()
        .json()
    )

    assert retrieved_value["data"] == data


async def test_that_context_variable_values_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    name = "test_variable"
    description = "test of context variable"

    variable = await context_variable_store.create_variable(
        name=name,
        description=description,
        tool_id=tool_id,
    )

    keys_and_data = {
        "key1": {"value": 1},
        "key2": {"value": 2},
        "key3": {"value": 3},
    }

    for key, data in keys_and_data.items():
        await async_client.put(
            f"/context-variables/{variable.id}/{key}",
            json={"data": data},
        )

    response = await async_client.get(f"/context-variables/{variable.id}")
    assert response.status_code == status.HTTP_200_OK

    retrieved_variable = response.json()["context_variable"]
    assert retrieved_variable["id"] == variable.id
    assert retrieved_variable["name"] == name
    assert retrieved_variable["description"] == description
    assert set(retrieved_variable["tags"]) == set()

    retrieved_values = response.json()["key_value_pairs"]

    assert len(retrieved_values) == len(keys_and_data)
    for key in keys_and_data:
        assert key in retrieved_values
        assert retrieved_values[key]["data"] == keys_and_data[key]


async def test_that_context_variable_value_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
    tool_id: ToolId,
) -> None:
    context_variable_store = container[ContextVariableStore]

    name = "test_variable"
    description = "test of context variable"

    variable = await context_variable_store.create_variable(
        name=name,
        description=description,
        tool_id=tool_id,
    )

    key = "test_key"
    data = {"value": 42}

    # Create value
    create_response = await async_client.put(
        f"/context-variables/{variable.id}/{key}",
        json={"data": data},
    )
    assert create_response.status_code == status.HTTP_200_OK

    # Delete value
    delete_response = await async_client.delete(f"/context-variables/{variable.id}/{key}")
    assert delete_response.status_code == status.HTTP_204_NO_CONTENT

    # Verify value is deleted
    read_response = await async_client.get(f"/context-variables/{variable.id}")
    assert read_response.status_code == status.HTTP_200_OK
    assert key not in read_response.json()["key_value_pairs"]


async def test_that_adding_nonexistent_agent_tag_to_context_variable_returns_404(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    context_variable_store = container[ContextVariableStore]

    variable = await context_variable_store.create_variable(
        name="test_variable",
        description="test of context variable",
        tool_id=ToolId("local", "test_tool"),
    )

    response = await async_client.patch(
        f"/context-variables/{variable.id}",
        json={"tags": {"add": ["agent-id:nonexistent_agent"]}},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_adding_nonexistent_tag_to_guideline_returns_404(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    context_variable_store = container[ContextVariableStore]

    variable = await context_variable_store.create_variable(
        name="test_variable",
        description="test of context variable",
        tool_id=ToolId("local", "test_tool"),
    )

    response = await async_client.patch(
        f"/context-variables/{variable.id}",
        json={"tags": {"add": ["nonexistent_tag"]}},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND
