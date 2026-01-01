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

from typing import Any
from fastapi import status
import httpx
from lagom import Container
from pytest import mark, raises

from parlant.core.agents import AgentId, AgentStore
from parlant.core.common import ItemNotFoundError
from parlant.core.tags import TagId, TagStore


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
    assert agent["max_engine_iterations"] == 3  # Default value


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


async def test_that_an_agent_can_be_created_with_default_composition_mode(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/agents",
        json={"name": "test-agent"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    agent = response.json()

    assert agent["name"] == "test-agent"
    assert agent["composition_mode"] == "fluid"  # Default mode


async def test_that_an_agent_can_be_created_with_specific_composition_mode(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/agents",
        json={"name": "test-agent", "composition_mode": "strict_canned"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    agent = response.json()

    assert agent["name"] == "test-agent"
    assert agent["composition_mode"] == "strict_canned"


async def test_that_an_agent_can_be_created_with_tags(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]

    tag1 = await tag_store.create_tag("tag1")
    tag2 = await tag_store.create_tag("tag2")

    response = await async_client.post(
        "/agents",
        json={"name": "test-agent", "tags": [tag1.id, tag1.id, tag2.id]},
    )

    assert response.status_code == status.HTTP_201_CREATED

    agent_dto = (
        (await async_client.get(f"/agents/{response.json()['id']}")).raise_for_status().json()
    )

    assert agent_dto["name"] == "test-agent"

    assert len(agent_dto["tags"]) == 2
    assert set(agent_dto["tags"]) == {tag1.id, tag2.id}


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
        ({"name": "New Name"}, "New Name", None, 3, "fluid"),
        ({"description": None}, "test-agent", None, 3, "fluid"),
        ({"description": "You are a test agent"}, "test-agent", "You are a test agent", 3, "fluid"),
        (
            {"description": "Changed desc", "max_engine_iterations": 2},
            "test-agent",
            "Changed desc",
            2,
            "fluid",
        ),
        ({"max_engine_iterations": 5}, "test-agent", None, 5, "fluid"),
        (
            {"composition_mode": "strict_canned"},
            "test-agent",
            None,
            3,
            "strict_canned",
        ),
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


async def test_that_tags_can_be_added_to_an_agent(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent_store = container[AgentStore]
    tag_store = container[TagStore]

    tag1 = await tag_store.create_tag("tag1")
    tag2 = await tag_store.create_tag("tag2")

    agent = await agent_store.create_agent("test-agent")

    update_payload = {"tags": {"add": [tag1.id, tag2.id]}}
    response = await async_client.patch(f"/agents/{agent.id}", json=update_payload)
    response.raise_for_status()
    updated_agent = response.json()

    assert tag1.id in updated_agent["tags"]
    assert tag2.id in updated_agent["tags"]

    agent_dto = (await async_client.get(f"/agents/{agent.id}")).raise_for_status().json()
    assert tag1.id in agent_dto["tags"]
    assert tag2.id in agent_dto["tags"]


async def test_that_tags_can_be_removed_from_an_agent(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent_store = container[AgentStore]
    agent = await agent_store.create_agent("test-agent")

    await agent_store.upsert_tag(agent.id, TagId("tag1"))
    await agent_store.upsert_tag(agent.id, TagId("tag2"))
    await agent_store.upsert_tag(agent.id, TagId("tag3"))

    update_payload = {"tags": {"remove": ["tag1", "tag3"]}}
    response = await async_client.patch(f"/agents/{agent.id}", json=update_payload)
    response.raise_for_status()
    updated_agent = response.json()

    assert "tag1" not in updated_agent["tags"]
    assert "tag2" in updated_agent["tags"]
    assert "tag3" not in updated_agent["tags"]


async def test_that_tags_can_be_added_and_removed_in_same_request(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent_store = container[AgentStore]
    tag_store = container[TagStore]

    tag1 = await tag_store.create_tag("tag1")
    tag2 = await tag_store.create_tag("tag2")
    tag3 = await tag_store.create_tag("tag3")
    tag4 = await tag_store.create_tag("tag4")

    agent = await agent_store.create_agent("test-agent")

    await agent_store.upsert_tag(agent.id, tag1.id)
    await agent_store.upsert_tag(agent.id, tag2.id)

    update_payload = {"tags": {"add": [tag3.id, tag4.id], "remove": [tag1.id]}}
    response = await async_client.patch(f"/agents/{agent.id}", json=update_payload)
    response.raise_for_status()
    updated_agent = response.json()

    assert tag1.id not in updated_agent["tags"]
    assert tag2.id in updated_agent["tags"]
    assert tag3.id in updated_agent["tags"]
    assert tag4.id in updated_agent["tags"]


async def test_that_an_agent_cannot_be_created_with_a_nonexistent_tag(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent_store = container[AgentStore]

    agent = await agent_store.create_agent("test-agent")

    response = await async_client.patch(
        f"/agents/{agent.id}",
        json={"tags": {"add": ["nonexistent-tag"]}},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_an_agent_can_be_created_with_custom_id(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    custom_id = "my_custom_agent_id"
    name = "Custom ID Agent"
    description = "An agent with a custom ID"

    response = await async_client.post(
        "/agents",
        json={
            "name": name,
            "description": description,
            "id": custom_id,
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    agent = response.json()
    assert agent["id"] == custom_id
    assert agent["name"] == name
    assert agent["description"] == description


async def test_that_multiple_agents_can_be_created_with_different_custom_ids(
    async_client: httpx.AsyncClient,
) -> None:
    # Create first agent with custom ID
    response1 = await async_client.post(
        "/agents",
        json={
            "name": "First Agent",
            "description": "First agent",
            "id": "agent_1",
        },
    )
    assert response1.status_code == status.HTTP_201_CREATED
    assert response1.json()["id"] == "agent_1"

    # Create second agent with different custom ID
    response2 = await async_client.post(
        "/agents",
        json={
            "name": "Second Agent",
            "description": "Second agent",
            "id": "agent_2",
        },
    )
    assert response2.status_code == status.HTTP_201_CREATED
    assert response2.json()["id"] == "agent_2"


async def test_that_creating_agent_with_duplicate_custom_id_fails(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    custom_id = AgentId("duplicate_agent_id")

    # Create first agent with custom ID
    response1 = await async_client.post(
        "/agents",
        json={
            "name": "First Agent",
            "description": "First agent",
            "id": custom_id,
        },
    )
    assert response1.status_code == status.HTTP_201_CREATED
    assert response1.json()["id"] == custom_id

    # Try to create second agent with same ID at the store level - should fail
    agent_store = container[AgentStore]
    with raises(ValueError, match="already exists"):
        await agent_store.create_agent(
            name="Second Agent",
            description="Second agent",
            id=custom_id,
        )


async def test_that_agent_composition_mode_can_be_set_and_updated(
    async_client: httpx.AsyncClient,
) -> None:
    # Create agent with CANNED_COMPOSITED mode
    response = await async_client.post(
        "/agents",
        json={
            "name": "test-agent",
            "composition_mode": "composited_canned",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    agent = response.json()
    agent_id = agent["id"]

    # Check that the composition mode is set correctly after creation
    assert agent["composition_mode"] == "composited_canned"

    # Retrieve agent and verify composition mode
    response = await async_client.get(f"/agents/{agent_id}")
    assert response.status_code == status.HTTP_200_OK
    agent = response.json()
    assert agent["composition_mode"] == "composited_canned"

    # Update agent to CANNED_STRICT mode
    response = await async_client.patch(
        f"/agents/{agent_id}",
        json={
            "composition_mode": "strict_canned",
        },
    )

    assert response.status_code == status.HTTP_200_OK
    agent = response.json()

    # Check that the composition mode is updated correctly
    assert agent["composition_mode"] == "strict_canned"

    # Retrieve agent again and verify composition mode
    response = await async_client.get(f"/agents/{agent_id}")
    assert response.status_code == status.HTTP_200_OK
    agent = response.json()
    assert agent["composition_mode"] == "strict_canned"
