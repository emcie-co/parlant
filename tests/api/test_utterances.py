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

import dateutil.parser
from fastapi import status
import httpx
from lagom import Container
from pytest import raises

from parlant.core.common import ItemNotFoundError
from parlant.core.utterances import UtteranceStore, UtteranceField
from parlant.core.tags import TagStore


async def test_that_an_utterance_can_be_created(
    async_client: httpx.AsyncClient,
) -> None:
    payload = {
        "value": "Your account balance is {{balance}}",
        "fields": [
            {
                "name": "balance",
                "description": "Account's balance",
                "examples": ["9000"],
            }
        ],
    }

    response = await async_client.post("/utterances", json=payload)
    assert response.status_code == status.HTTP_201_CREATED

    utterance = response.json()

    assert utterance["value"] == payload["value"]
    assert utterance["fields"] == payload["fields"]

    assert "id" in utterance
    assert "creation_utc" in utterance


async def test_that_an_utterance_can_be_created_with_tags(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]

    tag_1 = await tag_store.create_tag(name="VIP")
    tag_2 = await tag_store.create_tag(name="Finance")

    payload = {
        "value": "Your account balance is {{balance}}",
        "fields": [
            {
                "name": "balance",
                "description": "Account's balance",
                "examples": ["9000"],
            }
        ],
        "tags": [tag_1.id, tag_2.id],
    }

    response = await async_client.post("/utterances", json=payload)
    assert response.status_code == status.HTTP_201_CREATED

    utterance_dto = (
        (await async_client.get(f"/utterances/{response.json()['id']}")).raise_for_status().json()
    )

    assert len(utterance_dto["tags"]) == 2
    assert set(utterance_dto["tags"]) == {tag_1.id, tag_2.id}


async def test_that_an_utterance_can_be_created_with_queries(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    payload = {
        "value": "Your account balance is {{balance}}",
        "fields": [
            {
                "name": "balance",
                "description": "Account's balance",
                "examples": ["9000"],
            }
        ],
        "queries": ["One", "Two", "Three"],
    }

    response = await async_client.post("/utterances", json=payload)
    assert response.status_code == status.HTTP_201_CREATED

    utterance_dto = (
        (await async_client.get(f"/utterances/{response.json()['id']}")).raise_for_status().json()
    )

    assert len(utterance_dto["queries"]) == 3
    assert set(utterance_dto["queries"]) == {"One", "Two", "Three"}


async def test_that_an_utterance_can_be_read(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    utterance_store = container[UtteranceStore]

    value = "Your account balance is {{balance}}"
    fields = [UtteranceField(name="balance", description="Account's balance", examples=["9000"])]

    utterance = await utterance_store.create_utterance(value=value, fields=fields)

    response = await async_client.get(f"/utterances/{utterance.id}")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["id"] == utterance.id
    assert data["value"] == value

    assert len(data["fields"]) == 1
    utterance_field = data["fields"][0]
    assert utterance_field["name"] == fields[0].name
    assert utterance_field["description"] == fields[0].description
    assert utterance_field["examples"] == fields[0].examples

    assert dateutil.parser.parse(data["creation_utc"]) == utterance.creation_utc


async def test_that_all_utterances_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    utterance_store = container[UtteranceStore]

    first_value = "Your account balance is {{balance}}"
    first_fields = [
        UtteranceField(name="balance", description="Account's balance", examples=["9000"])
    ]

    second_value = "It will take {{day_count}} days to deliver to {{address}}"
    second_fields = [
        UtteranceField(
            name="day_count", description="Time required for delivery in days", examples=["8"]
        ),
        UtteranceField(name="address", description="Customer's address", examples=["Some Address"]),
    ]

    await utterance_store.create_utterance(value=first_value, fields=first_fields)
    await utterance_store.create_utterance(value=second_value, fields=second_fields)

    response = await async_client.get("/utterances")
    assert response.status_code == status.HTTP_200_OK
    utterances = response.json()

    assert len(utterances) >= 2
    assert any(f["value"] == first_value for f in utterances)
    assert any(f["value"] == second_value for f in utterances)


async def test_that_relevant_utterances_can_be_retrieved_based_on_closest_queries(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    utterance_store = container[UtteranceStore]

    utterances = [
        await utterance_store.create_utterance(value="Red", queries=[]),
        await utterance_store.create_utterance(value="Green", queries=[]),
        await utterance_store.create_utterance(value="Blue", queries=[]),
        await utterance_store.create_utterance(value="Paneer Cheese", queries=["Colors"]),
    ]

    closest_utterance = next(
        iter(
            await utterance_store.find_relevant_utterances(
                query="Colors",
                available_utterances=utterances,
                max_count=1,
            )
        )
    )

    assert closest_utterance.value == "Paneer Cheese"


async def test_that_an_utterance_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    utterance_store = container[UtteranceStore]

    value = "Your account balance is {{balance}}"
    fields = [UtteranceField(name="balance", description="Account's balance", examples=["9000"])]

    utterance = await utterance_store.create_utterance(value=value, fields=fields)

    update_payload = {
        "value": "Updated balance: {{balance}}",
        "fields": [
            {
                "name": "balance",
                "description": "Updated account balance",
                "examples": ["10000"],
            }
        ],
    }

    response = await async_client.patch(f"/utterances/{utterance.id}", json=update_payload)
    assert response.status_code == status.HTTP_200_OK

    updated_utterance = response.json()
    assert updated_utterance["value"] == update_payload["value"]
    assert updated_utterance["fields"] == update_payload["fields"]


async def test_that_an_utterance_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    utterance_store = container[UtteranceStore]

    value = "Your account balance is {{balance}}"
    fields = [UtteranceField(name="balance", description="Account's balance", examples=["9000"])]

    utterance = await utterance_store.create_utterance(value=value, fields=fields)

    delete_response = await async_client.delete(f"/utterances/{utterance.id}")
    assert delete_response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await utterance_store.read_utterance(utterance.id)


async def test_that_a_tag_can_be_added_to_an_utterance(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    utterance_store = container[UtteranceStore]
    tag_store = container[TagStore]

    tag = await tag_store.create_tag(name="VIP")

    value = "Your account balance is {{balance}}"
    fields = [UtteranceField(name="balance", description="Account's balance", examples=["9000"])]

    utterance = await utterance_store.create_utterance(value=value, fields=fields)

    response = await async_client.patch(
        f"/utterances/{utterance.id}", json={"tags": {"add": [tag.id]}}
    )
    assert response.status_code == status.HTTP_200_OK

    updated_utterance = await utterance_store.read_utterance(utterance.id)
    assert tag.id in updated_utterance.tags


async def test_that_a_tag_can_be_removed_from_an_utterance(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    utterance_store = container[UtteranceStore]
    tag_store = container[TagStore]

    tag = await tag_store.create_tag(name="VIP")

    value = "Your account balance is {{balance}}"
    fields = [UtteranceField(name="balance", description="Account's balance", examples=["9000"])]

    utterance = await utterance_store.create_utterance(value=value, fields=fields)

    await utterance_store.upsert_tag(utterance_id=utterance.id, tag_id=tag.id)
    response = await async_client.patch(
        f"/utterances/{utterance.id}", json={"tags": {"remove": [tag.id]}}
    )
    assert response.status_code == status.HTTP_200_OK

    updated_utterance = await utterance_store.read_utterance(utterance.id)
    assert tag.id not in updated_utterance.tags


async def test_that_utterances_can_be_filtered_by_tags(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    utterance_store = container[UtteranceStore]
    tag_store = container[TagStore]

    tag_vip = await tag_store.create_tag(name="VIP")
    tag_finance = await tag_store.create_tag(name="Finance")
    tag_greeting = await tag_store.create_tag(name="Greeting")

    first_utterance = await utterance_store.create_utterance(
        value="Welcome {{username}}!",
        fields=[
            UtteranceField(name="username", description="User's name", examples=["Alice", "Bob"])
        ],
    )
    await utterance_store.upsert_tag(first_utterance.id, tag_greeting.id)

    second_utterance = await utterance_store.create_utterance(
        value="Your balance is {{balance}}",
        fields=[
            UtteranceField(
                name="balance", description="Account balance", examples=["5000", "10000"]
            )
        ],
    )
    await utterance_store.upsert_tag(second_utterance.id, tag_finance.id)

    third_utterance = await utterance_store.create_utterance(
        value="Exclusive VIP offer for {{username}}",
        fields=[UtteranceField(name="username", description="VIP customer", examples=["Charlie"])],
    )
    await utterance_store.upsert_tag(third_utterance.id, tag_vip.id)

    response = await async_client.get(f"/utterances?tags={tag_greeting.id}")
    assert response.status_code == status.HTTP_200_OK
    utterances = response.json()
    assert len(utterances) == 1
    assert utterances[0]["value"] == "Welcome {{username}}!"

    response = await async_client.get(f"/utterances?tags={tag_finance.id}&tags={tag_vip.id}")
    assert response.status_code == status.HTTP_200_OK
    utterances = response.json()
    assert len(utterances) == 2
    values = {f["value"] for f in utterances}
    assert "Your balance is {{balance}}" in values
    assert "Exclusive VIP offer for {{username}}" in values

    response = await async_client.get("/utterances?tags=non_existent_tag")
    assert response.status_code == status.HTTP_200_OK
    utterances = response.json()
    assert len(utterances) == 0
