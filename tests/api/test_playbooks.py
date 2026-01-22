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
from lagom import Container

from parlant.core.playbooks import PlaybookId, PlaybookStore
from parlant.core.tags import TagStore


async def test_that_a_playbook_can_be_created_with_minimal_params(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/playbooks",
        json={"name": "Test Playbook"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    playbook = response.json()

    assert playbook["name"] == "Test Playbook"
    assert playbook["description"] is None
    assert playbook["parent_id"] is None
    assert playbook["disabled_rules"] == []
    assert playbook["tags"] == []


async def test_that_a_playbook_can_be_created_with_description(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/playbooks",
        json={"name": "Test Playbook", "description": "A test playbook"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    playbook = response.json()

    assert playbook["name"] == "Test Playbook"
    assert playbook["description"] == "A test playbook"


async def test_that_a_playbook_can_be_created_with_parent(
    async_client: httpx.AsyncClient,
) -> None:
    # Create parent playbook first
    parent_response = await async_client.post(
        "/playbooks",
        json={"name": "Parent Playbook"},
    )
    assert parent_response.status_code == status.HTTP_201_CREATED
    parent_id = parent_response.json()["id"]

    # Create child playbook
    response = await async_client.post(
        "/playbooks",
        json={"name": "Child Playbook", "parent_id": parent_id},
    )

    assert response.status_code == status.HTTP_201_CREATED

    playbook = response.json()

    assert playbook["name"] == "Child Playbook"
    assert playbook["parent_id"] == parent_id


async def test_that_a_playbook_can_be_created_with_custom_id(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/playbooks",
        json={"name": "Custom ID Playbook", "id": "my-custom-playbook-id"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    playbook = response.json()

    assert playbook["id"] == "my-custom-playbook-id"
    assert playbook["name"] == "Custom ID Playbook"


async def test_that_playbooks_can_be_listed(
    async_client: httpx.AsyncClient,
) -> None:
    # Create some playbooks
    await async_client.post("/playbooks", json={"name": "Playbook 1"})
    await async_client.post("/playbooks", json={"name": "Playbook 2"})

    response = await async_client.get("/playbooks")

    assert response.status_code == status.HTTP_200_OK

    playbooks = response.json()
    names = {p["name"] for p in playbooks}

    assert "Playbook 1" in names
    assert "Playbook 2" in names


async def test_that_a_playbook_can_be_read_by_id(
    async_client: httpx.AsyncClient,
) -> None:
    # Create a playbook
    create_response = await async_client.post(
        "/playbooks",
        json={"name": "Readable Playbook", "description": "Can be read"},
    )
    playbook_id = create_response.json()["id"]

    # Read it back
    response = await async_client.get(f"/playbooks/{playbook_id}")

    assert response.status_code == status.HTTP_200_OK

    playbook = response.json()

    assert playbook["id"] == playbook_id
    assert playbook["name"] == "Readable Playbook"
    assert playbook["description"] == "Can be read"


async def test_that_reading_nonexistent_playbook_returns_404(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.get("/playbooks/nonexistent-id")

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_a_playbook_can_be_updated(
    async_client: httpx.AsyncClient,
) -> None:
    # Create a playbook
    create_response = await async_client.post(
        "/playbooks",
        json={"name": "Original Name", "description": "Original description"},
    )
    playbook_id = create_response.json()["id"]

    # Update it
    response = await async_client.patch(
        f"/playbooks/{playbook_id}",
        json={"name": "Updated Name", "description": "Updated description"},
    )

    assert response.status_code == status.HTTP_200_OK

    playbook = response.json()

    assert playbook["name"] == "Updated Name"
    assert playbook["description"] == "Updated description"


async def test_that_a_playbook_can_be_deleted(
    async_client: httpx.AsyncClient,
) -> None:
    # Create a playbook
    create_response = await async_client.post(
        "/playbooks",
        json={"name": "To Delete"},
    )
    playbook_id = create_response.json()["id"]

    # Delete it
    response = await async_client.delete(f"/playbooks/{playbook_id}")

    assert response.status_code == status.HTTP_204_NO_CONTENT

    # Verify it's gone
    get_response = await async_client.get(f"/playbooks/{playbook_id}")
    assert get_response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_deleting_nonexistent_playbook_returns_404(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.delete("/playbooks/nonexistent-id")

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_disabled_rules_can_be_added_to_playbook(
    async_client: httpx.AsyncClient,
) -> None:
    # Create a playbook
    create_response = await async_client.post(
        "/playbooks",
        json={"name": "Playbook with disabled rules"},
    )
    playbook_id = create_response.json()["id"]

    # Add disabled rules
    response = await async_client.patch(
        f"/playbooks/{playbook_id}/disabled-rules",
        json={"add": ["guideline:abc123", "term:xyz789"]},
    )

    assert response.status_code == status.HTTP_200_OK

    playbook = response.json()

    assert "guideline:abc123" in playbook["disabled_rules"]
    assert "term:xyz789" in playbook["disabled_rules"]


async def test_that_disabled_rules_can_be_removed_from_playbook(
    async_client: httpx.AsyncClient,
) -> None:
    # Create a playbook
    create_response = await async_client.post(
        "/playbooks",
        json={"name": "Playbook with disabled rules"},
    )
    playbook_id = create_response.json()["id"]

    # Add disabled rules
    await async_client.patch(
        f"/playbooks/{playbook_id}/disabled-rules",
        json={"add": ["guideline:abc123", "term:xyz789"]},
    )

    # Remove one
    response = await async_client.patch(
        f"/playbooks/{playbook_id}/disabled-rules",
        json={"remove": ["guideline:abc123"]},
    )

    assert response.status_code == status.HTTP_200_OK

    playbook = response.json()

    assert "guideline:abc123" not in playbook["disabled_rules"]
    assert "term:xyz789" in playbook["disabled_rules"]


async def test_that_tags_can_be_added_to_playbook(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]
    tag = await tag_store.create_tag("test-tag")

    # Create a playbook
    create_response = await async_client.post(
        "/playbooks",
        json={"name": "Tagged Playbook"},
    )
    playbook_id = create_response.json()["id"]

    # Add tag
    response = await async_client.patch(
        f"/playbooks/{playbook_id}/tags",
        json={"add": [tag.id]},
    )

    assert response.status_code == status.HTTP_200_OK

    playbook = response.json()

    assert tag.id in playbook["tags"]


async def test_that_tags_can_be_removed_from_playbook(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]
    tag1 = await tag_store.create_tag("tag1")
    tag2 = await tag_store.create_tag("tag2")

    # Create a playbook with tags
    create_response = await async_client.post(
        "/playbooks",
        json={"name": "Tagged Playbook", "tags": [tag1.id, tag2.id]},
    )
    playbook_id = create_response.json()["id"]

    # Remove one tag
    response = await async_client.patch(
        f"/playbooks/{playbook_id}/tags",
        json={"remove": [tag1.id]},
    )

    assert response.status_code == status.HTTP_200_OK

    playbook = response.json()

    assert tag1.id not in playbook["tags"]
    assert tag2.id in playbook["tags"]


async def test_that_playbook_inheritance_chain_can_be_retrieved(
    async_client: httpx.AsyncClient,
) -> None:
    # Create grandparent
    grandparent_response = await async_client.post(
        "/playbooks",
        json={"name": "Grandparent"},
    )
    grandparent_id = grandparent_response.json()["id"]

    # Create parent
    parent_response = await async_client.post(
        "/playbooks",
        json={"name": "Parent", "parent_id": grandparent_id},
    )
    parent_id = parent_response.json()["id"]

    # Create child
    child_response = await async_client.post(
        "/playbooks",
        json={"name": "Child", "parent_id": parent_id},
    )
    child_id = child_response.json()["id"]

    # Get inheritance chain
    response = await async_client.get(f"/playbooks/{child_id}/inheritance")

    assert response.status_code == status.HTTP_200_OK

    chain = response.json()

    # Chain should be from root to child
    assert len(chain) == 3
    assert chain[0]["name"] == "Grandparent"
    assert chain[1]["name"] == "Parent"
    assert chain[2]["name"] == "Child"


async def test_that_circular_inheritance_is_prevented(
    async_client: httpx.AsyncClient,
) -> None:
    # Create playbook A
    a_response = await async_client.post(
        "/playbooks",
        json={"name": "Playbook A"},
    )
    a_id = a_response.json()["id"]

    # Create playbook B with parent A
    b_response = await async_client.post(
        "/playbooks",
        json={"name": "Playbook B", "parent_id": a_id},
    )
    b_id = b_response.json()["id"]

    # Try to update A to have parent B (creates cycle)
    response = await async_client.patch(
        f"/playbooks/{a_id}",
        json={"parent_id": b_id},
    )

    # Should fail with 400 or 422
    assert response.status_code in [
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
    ]


async def test_that_playbook_with_children_cannot_be_deleted(
    async_client: httpx.AsyncClient,
) -> None:
    # Create parent
    parent_response = await async_client.post(
        "/playbooks",
        json={"name": "Parent"},
    )
    parent_id = parent_response.json()["id"]

    # Create child
    await async_client.post(
        "/playbooks",
        json={"name": "Child", "parent_id": parent_id},
    )

    # Try to delete parent
    response = await async_client.delete(f"/playbooks/{parent_id}")

    # Should fail with 400 or 422
    assert response.status_code in [
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
    ]
