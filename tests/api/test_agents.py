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

from fastapi import status
import httpx

from parlant.core.agents import Agent, AgentStore
from parlant.core.container import Container


async def make_agent(container: Container, name: str = "test-agent") -> Agent:
    return await container[AgentStore].create_agent(
        name=name,
        instructions="You are a helpful billing assistant.",
    )


async def test_that_listing_agents_returns_only_id_and_name(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent = await make_agent(container)

    agents = (await async_client.get("/agents")).raise_for_status().json()

    assert agents == [{"id": agent.id, "name": "test-agent"}]


async def test_that_reading_an_agent_returns_only_id_and_name(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent = await make_agent(container)

    agent_dto = (await async_client.get(f"/agents/{agent.id}")).raise_for_status().json()

    assert agent_dto == {"id": agent.id, "name": "test-agent"}


async def test_that_reading_a_missing_agent_returns_404(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.get("/agents/nonexistent-agent")

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_agent_write_endpoints_do_not_exist(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent = await make_agent(container)

    assert (
        await async_client.post("/agents", json={"name": "new-agent"})
    ).status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    assert (
        await async_client.patch(f"/agents/{agent.id}", json={"name": "renamed"})
    ).status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    assert (
        await async_client.delete(f"/agents/{agent.id}")
    ).status_code == status.HTTP_405_METHOD_NOT_ALLOWED
