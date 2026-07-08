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

from parlant.core.glossary import GlossaryStore
from parlant.core.groups import GroupId, GroupStore


async def test_that_a_term_can_be_created(
    async_client: httpx.AsyncClient,
) -> None:
    name = "rule"
    description = "when and then statements"
    synonyms = ["rule", "principle"]

    response = await async_client.post(
        "/terms",
        json={
            "name": name,
            "description": description,
            "synonyms": synonyms,
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    data = response.json()

    assert data["name"] == name
    assert data["description"] == description
    assert data["synonyms"] == synonyms
    assert data["groups"] == []
    assert "modified_utc" in data


async def test_that_a_term_can_be_created_with_tags(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    group_store = container[GroupStore]

    group1 = await group_store.create_group(name="group1")
    group2 = await group_store.create_group(name="group2")

    response = await async_client.post(
        "/terms",
        json={
            "name": "rule",
            "description": "when and then statements",
            "groups": [group1.id, group1.id, group2.id],
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    term_dto = (await async_client.get(f"/terms/{response.json()['id']}")).raise_for_status().json()

    assert term_dto["name"] == "rule"
    assert term_dto["description"] == "when and then statements"

    assert len(term_dto["groups"]) == 2
    assert set(term_dto["groups"]) == {group1.id, group2.id}


async def test_that_a_term_can_be_read(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    name = "rule"
    description = "when and then statements"
    synonyms = ["rule", "principle"]

    create_response = await async_client.post(
        "/terms",
        json={
            "name": name,
            "description": description,
            "synonyms": synonyms,
        },
    )
    assert create_response.status_code == status.HTTP_201_CREATED
    term = create_response.json()

    read_response = await async_client.get(f"/terms/{term['id']}")
    assert read_response.status_code == status.HTTP_200_OK

    data = read_response.json()
    assert data["name"] == name
    assert data["description"] == description
    assert data["synonyms"] == synonyms
    assert data["groups"] == []


async def test_that_terms_can_be_listed(
    async_client: httpx.AsyncClient,
) -> None:
    terms = [
        {"name": "rule1", "description": "description 1", "synonyms": ["synonym1"]},
        {"name": "rule2", "description": "description 2", "synonyms": ["synonym2"]},
    ]

    for term in terms:
        response = await async_client.post(
            "/terms",
            json={
                "name": term["name"],
                "description": term["description"],
                "synonyms": term["synonyms"],
            },
        )
        assert response.status_code == status.HTTP_201_CREATED

    returned_terms = (await async_client.get("/terms")).raise_for_status().json()

    assert len(returned_terms) >= 2

    created_terms = []
    for term in returned_terms:
        term_data = {
            "name": term["name"],
            "description": term["description"],
            "synonyms": term["synonyms"],
        }
        if term_data in terms:
            created_terms.append(term_data)

    assert len(created_terms) == 2


async def test_that_terms_can_be_listed_with_a_tag(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    glossary_store = container[GlossaryStore]

    first_term = await glossary_store.create_term(
        name="rule1",
        description="description 1",
        synonyms=["synonym1"],
    )
    await glossary_store.upsert_group(
        term_id=first_term.id,
        group_id=GroupId("group1"),
    )

    second_term = await glossary_store.create_term(
        name="rule2",
        description="description 2",
        synonyms=["synonym2"],
    )
    await glossary_store.upsert_group(
        term_id=second_term.id,
        group_id=GroupId("group2"),
    )

    third_term = await glossary_store.create_term(
        name="rule3",
        description="description 3",
        synonyms=["synonym3"],
    )
    await glossary_store.upsert_group(
        term_id=third_term.id,
        group_id=GroupId("group1"),
    )

    response = await async_client.get("/terms?group_id=group1")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 2

    first_term_dto = next(term for term in data if term["id"] == first_term.id)
    third_term_dto = next(term for term in data if term["id"] == third_term.id)

    assert first_term_dto["name"] == first_term.name
    assert first_term_dto["description"] == first_term.description
    assert first_term_dto["synonyms"] == first_term.synonyms

    assert third_term_dto["name"] == third_term.name
    assert third_term_dto["description"] == third_term.description
    assert third_term_dto["synonyms"] == third_term.synonyms


async def test_that_a_term_can_be_updated_with_new_values(
    async_client: httpx.AsyncClient,
) -> None:
    group1 = (await async_client.post("/groups", json={"name": "group1"})).raise_for_status().json()
    group2 = (await async_client.post("/groups", json={"name": "group2"})).raise_for_status().json()

    name = "rule"
    description = "when and then statements"
    synonyms = ["rule", "principle"]

    term = (
        (
            await async_client.post(
                "/terms",
                json={
                    "name": name,
                    "description": description,
                    "synonyms": synonyms,
                },
            )
        )
        .raise_for_status()
        .json()
    )

    updated_name = "updated rule"
    updated_description = "Updated rule description"
    updated_synonyms = ["instruction"]
    tags_to_add = [group1["id"], group2["id"]]

    update_response = await async_client.patch(
        f"/terms/{term['id']}",
        json={
            "name": updated_name,
            "description": updated_description,
            "synonyms": updated_synonyms,
            "groups": {
                "add": tags_to_add,
            },
        },
    )

    assert update_response.status_code == status.HTTP_200_OK

    data = update_response.json()
    assert data["name"] == updated_name
    assert data["description"] == updated_description
    assert data["synonyms"] == updated_synonyms
    assert set(data["groups"]) == set(tags_to_add)
    assert data["modified_utc"] != term["modified_utc"]


async def test_that_tags_can_be_removed_from_a_term(
    async_client: httpx.AsyncClient,
) -> None:
    group1 = (await async_client.post("/groups", json={"name": "group1"})).raise_for_status().json()
    group2 = (await async_client.post("/groups", json={"name": "group2"})).raise_for_status().json()

    name = "rule"
    description = "when and then statements"
    synonyms = ["rule", "principle"]

    term = (
        (
            await async_client.post(
                "/terms",
                json={
                    "name": name,
                    "description": description,
                    "synonyms": synonyms,
                },
            )
        )
        .raise_for_status()
        .json()
    )

    await async_client.patch(
        f"/terms/{term['id']}",
        json={
            "groups": {
                "add": [group1["id"], group2["id"]],
            },
        },
    )

    update_response = await async_client.patch(
        f"/terms/{term['id']}",
        json={
            "groups": {
                "remove": [group1["id"]],
            },
        },
    )

    assert update_response.status_code == status.HTTP_200_OK
    data = update_response.json()
    assert set(data["groups"]) == {group2["id"]}


async def test_that_a_term_can_be_deleted(
    async_client: httpx.AsyncClient,
) -> None:
    name = "rule"
    description = "when and then statements"
    synonyms = ["rule", "principle"]

    term = (
        (
            await async_client.post(
                "/terms",
                json={
                    "name": name,
                    "description": description,
                    "synonyms": synonyms,
                },
            )
        )
        .raise_for_status()
        .json()
    )

    (await async_client.delete(f"/terms/{term['id']}")).raise_for_status()

    read_response = await async_client.get(f"/terms/{term['id']}")
    assert read_response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_adding_nonexistent_agent_group_to_term_returns_404(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    glossary_store = container[GlossaryStore]

    term = await glossary_store.create_term(
        name="rule",
        description="when and then statements",
        synonyms=["rule", "principle"],
    )

    response = await async_client.patch(
        f"/terms/{term.id}",
        json={"groups": {"add": ["agent-id:nonexistent_agent"]}},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_adding_nonexistent_group_to_term_returns_404(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    glossary_store = container[GlossaryStore]

    term = await glossary_store.create_term(
        name="rule",
        description="when and then statements",
        synonyms=["rule", "principle"],
    )

    response = await async_client.patch(
        f"/terms/{term.id}",
        json={"groups": {"add": ["nonexistent_group"]}},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_a_term_can_be_created_with_custom_id(
    async_client: httpx.AsyncClient,
) -> None:
    name = "Custom Term"
    description = "A term with a custom ID"
    synonyms = ["custom", "test"]
    custom_id = "custom-term-123"

    response = await async_client.post(
        "/terms",
        json={
            "name": name,
            "description": description,
            "synonyms": synonyms,
            "id": custom_id,
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    data = response.json()

    assert data["id"] == custom_id
    assert data["name"] == name
    assert data["description"] == description
    assert data["synonyms"] == synonyms
    assert data["groups"] == []


async def test_that_creating_term_with_duplicate_id_returns_422(
    async_client: httpx.AsyncClient,
) -> None:
    custom_id = "duplicate-term-id"

    # Create first term with custom ID
    response1 = await async_client.post(
        "/terms",
        json={
            "name": "First Term",
            "description": "First term",
            "id": custom_id,
        },
    )
    assert response1.status_code == status.HTTP_201_CREATED

    # Try to create second term with same ID
    response2 = await async_client.post(
        "/terms",
        json={
            "name": "Second Term",
            "description": "Second term",
            "id": custom_id,
        },
    )
    assert response2.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "already exists" in response2.json()["detail"]
