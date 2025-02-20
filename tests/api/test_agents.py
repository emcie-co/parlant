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

from typing import Any
from fastapi import status
import httpx
from lagom import Container
from pytest import mark, raises

from parlant.core.agents import AgentStore
from parlant.core.common import ItemNotFoundError


async def test_that_an_agent_can_be_created_without_description(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/agents",
        json={"name": "test-agent"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    agent = response.json()

    assert agent["name"] == "test-agent"
    assert agent["description"] is None


async def test_that_an_agent_can_be_created_with_description(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/agents",
        json={"name": "test-agent", "description": "You are a test agent"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    agent = response.json()

    assert agent["name"] == "test-agent"
    assert agent["description"] == "You are a test agent"


async def test_that_an_agent_can_be_created_without_max_engine_iterations(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/agents",
        json={"name": "test-agent"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    agent = response.json()

    assert agent["name"] == "test-agent"
    assert agent["max_engine_iterations"] == 1


async def test_that_an_agent_can_be_created_with_max_engine_iterations(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/agents",
        json={"name": "test-agent", "max_engine_iterations": 1},
    )

    assert response.status_code == status.HTTP_201_CREATED

    agent = response.json()

    assert agent["name"] == "test-agent"
    assert agent["max_engine_iterations"] == 1


async def test_that_an_agent_can_be_listed(
    async_client: httpx.AsyncClient,
) -> None:
    _ = (
        (
            await async_client.post(
                "/agents",
                json={"name": "test-agent"},
            )
        )
        .raise_for_status()
        .json()
    )

    agents = (
        (
            await async_client.get(
                "/agents",
            )
        )
        .raise_for_status()
        .json()
    )

    assert len(agents) == 1
    assert agents[0]["name"] == "test-agent"
    assert agents[0]["description"] is None


async def test_that_an_agent_can_be_read(
    async_client: httpx.AsyncClient,
) -> None:
    agent = (
        (
            await async_client.post(
                "/agents",
                json={"name": "test-agent"},
            )
        )
        .raise_for_status()
        .json()
    )

    agent_dto = (
        (
            await async_client.get(
                f"/agents/{agent['id']}",
            )
        )
        .raise_for_status()
        .json()
    )

    assert agent_dto["name"] == "test-agent"
    assert agent_dto["description"] is None
    assert agent_dto["composition_mode"] == "fluid"


@mark.parametrize(
    "update_payload, expected_name, expected_description, expected_iterations, expected_composition",
    [
        ({"name": "New Name"}, "New Name", None, 1, "fluid"),
        ({"description": None}, "test-agent", None, 1, "fluid"),
        ({"description": "You are a test agent"}, "test-agent", "You are a test agent", 1, "fluid"),
        (
            {"description": "Changed desc", "max_engine_iterations": 2},
            "test-agent",
            "Changed desc",
            2,
            "fluid",
        ),
        ({"max_engine_iterations": 5}, "test-agent", None, 5, "fluid"),
        ({"composition_mode": "strict_assembly"}, "test-agent", None, 1, "strict_assembly"),
    ],
)
async def test_that_an_agent_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
    update_payload: dict[str, Any],
    expected_name: str,
    expected_description: str | None,
    expected_iterations: int,
    expected_composition: str,
) -> None:
    agent_store = container[AgentStore]
    agent = await agent_store.create_agent("test-agent")

    response = await async_client.patch(f"/agents/{agent.id}", json=update_payload)
    response.raise_for_status()
    updated_agent = response.json()

    assert updated_agent["name"] == update_payload.get("name", "test-agent")
    assert updated_agent["name"] == expected_name
    assert updated_agent["description"] == expected_description
    assert updated_agent["max_engine_iterations"] == expected_iterations
    assert updated_agent["composition_mode"] == expected_composition


async def test_that_an_agent_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent_store = container[AgentStore]
    agent = await agent_store.create_agent("test-agent")

    delete_response = await async_client.delete(
        f"/agents/{agent.id}",
    )
    assert delete_response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await agent_store.read_agent(agent.id)
