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
        "/agents/",
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
        "/agents/",
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
        "/agents/",
        json={"name": "test-agent"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    agent = response.json()

    assert agent["name"] == "test-agent"
    assert agent["max_engine_iterations"] == 3


async def test_that_an_agent_can_be_created_with_max_engine_iterations(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/agents/",
        json={"name": "test-agent", "max_engine_iterations": 1},
    )

    assert response.status_code == status.HTTP_201_CREATED

    agent = response.json()

    assert agent["name"] == "test-agent"
    assert agent["max_engine_iterations"] == 1


@mark.parametrize(
    "patch_request",
    (
        {"name": "New Name"},
        {"description": None},
        {"description": "You are a test agent"},
        {"description": "You are a test agent", "max_engine_iterations": 1},
        {"max_engine_iterations": 1},
    ),
)
async def test_that_agent_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
    patch_request: dict[str, Any],
) -> None:
    agent_store = container[AgentStore]
    agent = await agent_store.create_agent("test-agent")

    agent_dto = (
        (
            await async_client.patch(
                f"/agents/{agent.id}",
                json=patch_request,
            )
        )
        .raise_for_status()
        .json()
    )

    assert agent_dto["name"] == patch_request.get("name", "test-agent")
    assert agent_dto["description"] == patch_request.get("description")
    assert agent_dto["max_engine_iterations"] == patch_request.get("max_engine_iterations", 3)


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
