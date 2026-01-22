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

"""API tests for agent-playbook relationship."""

from fastapi import status
import httpx


async def test_that_agent_can_be_created_without_playbook_via_api(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/agents",
        json={"name": "Test Agent"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    agent = response.json()

    assert agent["name"] == "Test Agent"
    assert agent["playbook_id"] is None


async def test_that_agent_can_be_created_with_playbook_via_api(
    async_client: httpx.AsyncClient,
) -> None:
    # Create a playbook first
    playbook_response = await async_client.post(
        "/playbooks",
        json={"name": "Test Playbook"},
    )
    playbook_id = playbook_response.json()["id"]

    # Create agent with playbook
    response = await async_client.post(
        "/agents",
        json={"name": "Test Agent", "playbook_id": playbook_id},
    )

    assert response.status_code == status.HTTP_201_CREATED

    agent = response.json()

    assert agent["name"] == "Test Agent"
    assert agent["playbook_id"] == playbook_id


async def test_that_agent_playbook_can_be_updated_via_api(
    async_client: httpx.AsyncClient,
) -> None:
    # Create a playbook
    playbook_response = await async_client.post(
        "/playbooks",
        json={"name": "Test Playbook"},
    )
    playbook_id = playbook_response.json()["id"]

    # Create agent without playbook
    agent_response = await async_client.post(
        "/agents",
        json={"name": "Test Agent"},
    )
    agent_id = agent_response.json()["id"]

    # Update agent with playbook
    response = await async_client.patch(
        f"/agents/{agent_id}",
        json={"playbook_id": playbook_id},
    )

    assert response.status_code == status.HTTP_200_OK

    agent = response.json()

    assert agent["playbook_id"] == playbook_id


async def test_that_agent_playbook_can_be_cleared_via_api(
    async_client: httpx.AsyncClient,
) -> None:
    # Create a playbook
    playbook_response = await async_client.post(
        "/playbooks",
        json={"name": "Test Playbook"},
    )
    playbook_id = playbook_response.json()["id"]

    # Create agent with playbook
    agent_response = await async_client.post(
        "/agents",
        json={"name": "Test Agent", "playbook_id": playbook_id},
    )
    agent_id = agent_response.json()["id"]
    assert agent_response.json()["playbook_id"] == playbook_id

    # Clear the playbook
    response = await async_client.patch(
        f"/agents/{agent_id}",
        json={"playbook_id": None},
    )

    assert response.status_code == status.HTTP_200_OK

    agent = response.json()

    assert agent["playbook_id"] is None


async def test_that_agent_playbook_persists_after_read_via_api(
    async_client: httpx.AsyncClient,
) -> None:
    # Create a playbook
    playbook_response = await async_client.post(
        "/playbooks",
        json={"name": "Test Playbook"},
    )
    playbook_id = playbook_response.json()["id"]

    # Create agent with playbook
    agent_response = await async_client.post(
        "/agents",
        json={"name": "Test Agent", "playbook_id": playbook_id},
    )
    agent_id = agent_response.json()["id"]

    # Read the agent
    response = await async_client.get(f"/agents/{agent_id}")

    assert response.status_code == status.HTTP_200_OK

    agent = response.json()

    assert agent["playbook_id"] == playbook_id


async def test_that_agent_playbook_appears_in_list_via_api(
    async_client: httpx.AsyncClient,
) -> None:
    # Create a playbook
    playbook_response = await async_client.post(
        "/playbooks",
        json={"name": "Test Playbook"},
    )
    playbook_id = playbook_response.json()["id"]

    # Create agents
    await async_client.post(
        "/agents",
        json={"name": "Agent With Playbook", "playbook_id": playbook_id},
    )
    await async_client.post(
        "/agents",
        json={"name": "Agent Without Playbook"},
    )

    # List agents
    response = await async_client.get("/agents")

    assert response.status_code == status.HTTP_200_OK

    agents = response.json()
    agent_with_playbook = next(a for a in agents if a["name"] == "Agent With Playbook")
    agent_without_playbook = next(a for a in agents if a["name"] == "Agent Without Playbook")

    assert agent_with_playbook["playbook_id"] == playbook_id
    assert agent_without_playbook["playbook_id"] is None


async def test_that_creating_agent_with_nonexistent_playbook_fails(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/agents",
        json={"name": "Test Agent", "playbook_id": "nonexistent-playbook"},
    )

    # Should fail with 404 (playbook not found)
    assert response.status_code == status.HTTP_404_NOT_FOUND
