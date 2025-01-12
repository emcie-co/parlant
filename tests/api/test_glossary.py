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
import httpx

from parlant.core.common import AgentId


async def test_that_a_term_can_be_created(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    name = "guideline"
    description = "when and then statements"
    synonyms = ["rule", "principle"]

    response = await async_client.post(
        f"/agents/{agent_id}/terms",
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


async def test_that_a_term_can_be_created_without_synonyms(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    name = "guideline"
    description = "when and then statements"

    response = await async_client.post(
        f"/agents/{agent_id}/terms",
        json={
            "name": name,
            "description": description,
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    data = response.json()

    assert data["name"] == name
    assert data["description"] == description
    assert data["synonyms"] == []


async def test_that_a_term_can_be_read(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    name = "guideline"
    description = "when and then statements"
    synonyms = ["rule", "principle"]

    create_response = await async_client.post(
        f"/agents/{agent_id}/terms",
        json={
            "name": name,
            "description": description,
            "synonyms": synonyms,
        },
    )
    assert create_response.status_code == status.HTTP_201_CREATED
    term = create_response.json()

    read_response = await async_client.get(f"agents/{agent_id}/terms/{term['id']}")
    assert read_response.status_code == status.HTTP_200_OK

    data = read_response.json()
    assert data["name"] == name
    assert data["description"] == description
    assert data["synonyms"] == synonyms


async def test_that_a_term_can_be_read_without_synonyms(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    name = "guideline"
    description = "when and then statements"

    create_response = await async_client.post(
        f"/agents/{agent_id}/terms",
        json={
            "name": name,
            "description": description,
        },
    )
    assert create_response.status_code == status.HTTP_201_CREATED
    term = create_response.json()

    read_response = await async_client.get(f"/agents/{agent_id}/terms/{term['id']}")
    assert read_response.status_code == status.HTTP_200_OK

    data = read_response.json()
    assert data["name"] == name
    assert data["description"] == description
    assert data["synonyms"] == []


async def test_that_terms_can_be_listed_for_an_agent(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    terms = [
        {"name": "guideline1", "description": "description 1", "synonyms": ["synonym1"]},
        {"name": "guideline2", "description": "description 2", "synonyms": ["synonym2"]},
    ]

    for term in terms:
        response = await async_client.post(
            f"/agents/{agent_id}/terms",
            json={
                "name": term["name"],
                "description": term["description"],
                "synonyms": term["synonyms"],
            },
        )
        assert response.status_code == status.HTTP_201_CREATED

    returned_terms = (await async_client.get(f"/agents/{agent_id}/terms")).raise_for_status().json()

    assert len(returned_terms) == 2
    assert {
        "name": returned_terms[1]["name"],
        "description": returned_terms[1]["description"],
        "synonyms": returned_terms[1]["synonyms"],
    } in terms

    assert {
        "name": returned_terms[0]["name"],
        "description": returned_terms[0]["description"],
        "synonyms": returned_terms[0]["synonyms"],
    } in terms


async def test_that_a_term_can_be_updated(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    name = "guideline"
    description = "when and then statements"
    synonyms = ["rule", "principle"]

    term = (
        (
            await async_client.post(
                f"/agents/{agent_id}/terms",
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

    updated_name = "updated guideline"
    updated_description = "Updated guideline description"
    updated_synonyms = ["instruction"]

    update_response = await async_client.patch(
        f"/agents/{agent_id}/terms/{term['id']}",
        json={
            "name": updated_name,
            "description": updated_description,
            "synonyms": updated_synonyms,
        },
    )

    assert update_response.status_code == status.HTTP_200_OK

    data = update_response.json()
    assert data["name"] == updated_name
    assert data["description"] == updated_description
    assert data["synonyms"] == updated_synonyms


async def test_that_a_term_can_be_deleted(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    name = "guideline"
    description = "when and then statements"
    synonyms = ["rule", "principle"]

    term = (
        (
            await async_client.post(
                f"/agents/{agent_id}/terms",
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

    (await async_client.delete(f"/agents/{agent_id}/terms/{term['id']}")).raise_for_status()

    read_response = await async_client.get(f"/agents/{agent_id}/terms/{name}")
    assert read_response.status_code == status.HTTP_404_NOT_FOUND
