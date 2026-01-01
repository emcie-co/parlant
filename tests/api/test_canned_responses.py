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

import dateutil.parser
from fastapi import status
import httpx
from lagom import Container
from pytest import raises

from parlant.core.agents import AgentStore
from parlant.core.common import ItemNotFoundError
from parlant.core.canned_responses import CannedResponseStore, CannedResponseField
from parlant.core.journeys import JourneyStore
from parlant.core.tags import Tag, TagStore


async def test_that_a_canned_response_can_be_created(
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

    response = await async_client.post("/canned_responses", json=payload)
    assert response.status_code == status.HTTP_201_CREATED

    canned_response = response.json()

    assert canned_response["value"] == payload["value"]
    assert canned_response["fields"] == payload["fields"]

    assert "id" in canned_response
    assert "creation_utc" in canned_response


async def test_that_a_canned_response_can_be_created_with_tags(
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

    response = await async_client.post("/canned_responses", json=payload)
    assert response.status_code == status.HTTP_201_CREATED

    canned_response_dto = (
        (await async_client.get(f"/canned_responses/{response.json()['id']}"))
        .raise_for_status()
        .json()
    )

    assert len(canned_response_dto["tags"]) == 2
    assert set(canned_response_dto["tags"]) == {tag_1.id, tag_2.id}


async def test_that_a_canned_response_can_be_created_with_signals(
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
        "signals": ["One", "Two", "Three"],
    }

    response = await async_client.post("/canned_responses", json=payload)
    assert response.status_code == status.HTTP_201_CREATED

    canned_response_dto = (
        (await async_client.get(f"/canned_responses/{response.json()['id']}"))
        .raise_for_status()
        .json()
    )

    assert len(canned_response_dto["signals"]) == 3
    assert set(canned_response_dto["signals"]) == {"One", "Two", "Three"}


async def test_that_a_canned_response_can_be_read(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    canned_response_store = container[CannedResponseStore]

    value = "Your account balance is {{balance}}"
    fields = [
        CannedResponseField(name="balance", description="Account's balance", examples=["9000"])
    ]

    canned_response = await canned_response_store.create_canned_response(value=value, fields=fields)

    response = await async_client.get(f"/canned_responses/{canned_response.id}")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["id"] == canned_response.id
    assert data["value"] == value

    assert len(data["fields"]) == 1
    canned_response_field = data["fields"][0]
    assert canned_response_field["name"] == fields[0].name
    assert canned_response_field["description"] == fields[0].description
    assert canned_response_field["examples"] == fields[0].examples

    assert dateutil.parser.parse(data["creation_utc"]) == canned_response.creation_utc


async def test_that_all_canned_responses_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    canned_response_store = container[CannedResponseStore]

    first_value = "Your account balance is {{balance}}"
    first_fields = [
        CannedResponseField(name="balance", description="Account's balance", examples=["9000"])
    ]

    second_value = "It will take {{day_count}} days to deliver to {{address}}"
    second_fields = [
        CannedResponseField(
            name="day_count", description="Time required for delivery in days", examples=["8"]
        ),
        CannedResponseField(
            name="address", description="Customer's address", examples=["Some Address"]
        ),
    ]

    await canned_response_store.create_canned_response(value=first_value, fields=first_fields)
    await canned_response_store.create_canned_response(value=second_value, fields=second_fields)

    response = await async_client.get("/canned_responses")
    assert response.status_code == status.HTTP_200_OK
    canned_responses = response.json()

    assert len(canned_responses) >= 2
    assert any(f["value"] == first_value for f in canned_responses)
    assert any(f["value"] == second_value for f in canned_responses)


async def test_that_relevant_canned_responses_can_be_retrieved_based_on_closest_signals(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    canned_response_store = container[CannedResponseStore]

    canned_responses = [
        await canned_response_store.create_canned_response(value="Red", signals=[]),
        await canned_response_store.create_canned_response(value="Green", signals=[]),
        await canned_response_store.create_canned_response(value="Blue", signals=[]),
        await canned_response_store.create_canned_response(
            value="Paneer Cheese", signals=["Colors"]
        ),
    ]

    closest_canned_response = next(
        iter(
            await canned_response_store.filter_relevant_canned_responses(
                query="Colors",
                available_canned_responses=canned_responses,
                max_count=1,
            )
        )
    )

    assert closest_canned_response.canned_response.value == "Paneer Cheese"


async def test_that_a_canned_response_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    canned_response_store = container[CannedResponseStore]

    value = "Your account balance is {{balance}}"
    fields = [
        CannedResponseField(name="balance", description="Account's balance", examples=["9000"])
    ]

    canned_response = await canned_response_store.create_canned_response(value=value, fields=fields)

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

    response = await async_client.patch(
        f"/canned_responses/{canned_response.id}", json=update_payload
    )
    assert response.status_code == status.HTTP_200_OK

    updated_canned_response = response.json()
    assert updated_canned_response["value"] == update_payload["value"]
    assert updated_canned_response["fields"] == update_payload["fields"]


async def test_that_a_canned_response_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    canned_response_store = container[CannedResponseStore]

    value = "Your account balance is {{balance}}"
    fields = [
        CannedResponseField(name="balance", description="Account's balance", examples=["9000"])
    ]

    canned_response = await canned_response_store.create_canned_response(value=value, fields=fields)

    delete_response = await async_client.delete(f"/canned_responses/{canned_response.id}")
    assert delete_response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await canned_response_store.read_canned_response(canned_response.id)


async def test_that_a_tag_can_be_added_to_a_canned_response(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    canned_response_store = container[CannedResponseStore]
    tag_store = container[TagStore]

    tag = await tag_store.create_tag(name="VIP")

    value = "Your account balance is {{balance}}"
    fields = [
        CannedResponseField(name="balance", description="Account's balance", examples=["9000"])
    ]

    canned_response = await canned_response_store.create_canned_response(value=value, fields=fields)

    response = await async_client.patch(
        f"/canned_responses/{canned_response.id}", json={"tags": {"add": [tag.id]}}
    )
    assert response.status_code == status.HTTP_200_OK

    updated_canned_response = await canned_response_store.read_canned_response(canned_response.id)
    assert tag.id in updated_canned_response.tags


async def test_that_a_tag_can_be_removed_from_a_canned_response(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    canned_response_store = container[CannedResponseStore]
    tag_store = container[TagStore]

    tag = await tag_store.create_tag(name="VIP")

    value = "Your account balance is {{balance}}"
    fields = [
        CannedResponseField(name="balance", description="Account's balance", examples=["9000"])
    ]

    canned_response = await canned_response_store.create_canned_response(value=value, fields=fields)

    await canned_response_store.upsert_tag(canned_response_id=canned_response.id, tag_id=tag.id)
    response = await async_client.patch(
        f"/canned_responses/{canned_response.id}", json={"tags": {"remove": [tag.id]}}
    )
    assert response.status_code == status.HTTP_200_OK

    updated_canned_response = await canned_response_store.read_canned_response(canned_response.id)
    assert tag.id not in updated_canned_response.tags


async def test_that_canned_responses_can_be_filtered_by_tags(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    canned_response_store = container[CannedResponseStore]
    tag_store = container[TagStore]

    tag_vip = await tag_store.create_tag(name="VIP")
    tag_finance = await tag_store.create_tag(name="Finance")
    tag_greeting = await tag_store.create_tag(name="Greeting")

    first_canned_response = await canned_response_store.create_canned_response(
        value="Welcome {{username}}!",
        fields=[
            CannedResponseField(
                name="username", description="User's name", examples=["Alice", "Bob"]
            )
        ],
    )
    await canned_response_store.upsert_tag(first_canned_response.id, tag_greeting.id)

    second_canned_response = await canned_response_store.create_canned_response(
        value="Your balance is {{balance}}",
        fields=[
            CannedResponseField(
                name="balance", description="Account balance", examples=["5000", "10000"]
            )
        ],
    )
    await canned_response_store.upsert_tag(second_canned_response.id, tag_finance.id)

    third_canned_response = await canned_response_store.create_canned_response(
        value="Exclusive VIP offer for {{username}}",
        fields=[
            CannedResponseField(name="username", description="VIP customer", examples=["Charlie"])
        ],
    )
    await canned_response_store.upsert_tag(third_canned_response.id, tag_vip.id)

    response = await async_client.get(f"/canned_responses?tags={tag_greeting.id}")
    assert response.status_code == status.HTTP_200_OK
    canned_responses = response.json()
    assert len(canned_responses) == 1
    assert canned_responses[0]["value"] == "Welcome {{username}}!"

    response = await async_client.get(f"/canned_responses?tags={tag_finance.id}&tags={tag_vip.id}")
    assert response.status_code == status.HTTP_200_OK
    canned_responses = response.json()
    assert len(canned_responses) == 2
    values = {f["value"] for f in canned_responses}
    assert "Your balance is {{balance}}" in values
    assert "Exclusive VIP offer for {{username}}" in values

    response = await async_client.get("/canned_responses?tags=non_existent_tag")
    assert response.status_code == status.HTTP_200_OK
    canned_responses = response.json()
    assert len(canned_responses) == 0


async def test_that_agent_tag_can_be_added_to_a_canned_response(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    canned_response_store = container[CannedResponseStore]
    agent = await container[AgentStore].create_agent("Test Agent")

    canrep = await canned_response_store.create_canned_response(
        value="Welcome {{username}}!",
        fields=[
            CannedResponseField(
                name="username", description="User's name", examples=["Alice", "Bob"]
            )
        ],
    )
    agent_tag = Tag.for_agent_id(agent.id)

    update_payload = {"tags": {"add": [agent_tag]}}
    response = await async_client.patch(f"/canned_responses/{canrep.id}", json=update_payload)
    response.raise_for_status()
    updated_canned_response = response.json()

    assert updated_canned_response["tags"] == [agent_tag]


async def test_that_agent_tag_can_be_removed_from_a_canned_response(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]
    canned_response_store = container[CannedResponseStore]

    agent = await container[AgentStore].create_agent("Test Agent")

    tag1 = await tag_store.create_tag("tag1")

    agent_tag = Tag.for_agent_id(agent.id)

    canrep = await canned_response_store.create_canned_response(
        value="Welcome {{username}}!",
        fields=[
            CannedResponseField(
                name="username", description="User's name", examples=["Alice", "Bob"]
            )
        ],
        tags=[tag1.id, agent_tag],
    )

    update_payload = {"tags": {"remove": [agent_tag]}}
    _ = (
        await async_client.patch(f"/canned_responses/{canrep.id}", json=update_payload)
    ).raise_for_status()

    canrep_after_update = (
        (await async_client.get(f"/canned_responses/{canrep.id}")).raise_for_status().json()
    )

    assert agent_tag not in canrep_after_update["tags"]
    assert tag1.id in canrep_after_update["tags"]


async def test_that_journey_tags_can_be_added_to_a_canned_response(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    canned_response_store = container[CannedResponseStore]

    journey = await journey_store.create_journey(
        title="Customer Support Journey",
        description="A journey for customer support interactions.",
        conditions=[],
    )
    journey_tag = Tag.for_journey_id(journey.id)

    tag1 = await tag_store.create_tag("tag1")

    canrep = await canned_response_store.create_canned_response(
        value="Welcome {{username}}!",
        fields=[
            CannedResponseField(
                name="username", description="User's name", examples=["Alice", "Bob"]
            )
        ],
    )

    update_payload = {"tags": {"add": [tag1.id, journey_tag]}}
    response = await async_client.patch(f"/canned_responses/{canrep.id}", json=update_payload)
    response.raise_for_status()
    updated_canrep = response.json()

    assert tag1.id in updated_canrep["tags"]
    assert journey_tag in updated_canrep["tags"]


async def test_that_journey_tags_can_be_removed_from_a_canned_response(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]
    canned_response_store = container[CannedResponseStore]
    journey_store = container[JourneyStore]

    journey = await journey_store.create_journey(
        title="Customer Support Journey",
        description="A journey for customer support interactions.",
        conditions=[],
    )
    journey_tag = Tag.for_journey_id(journey.id)

    tag1 = await tag_store.create_tag("tag1")

    canrep = await canned_response_store.create_canned_response(
        value="Welcome {{username}}!",
        fields=[
            CannedResponseField(
                name="username", description="User's name", examples=["Alice", "Bob"]
            )
        ],
        tags=[tag1.id, journey_tag],
    )

    update_payload = {"tags": {"remove": [journey_tag]}}
    _ = (
        await async_client.patch(f"/canned_responses/{canrep.id}", json=update_payload)
    ).raise_for_status()

    canrep_after_update = (
        (await async_client.get(f"/canned_responses/{canrep.id}")).raise_for_status().json()
    )

    assert journey_tag not in canrep_after_update["tags"]
    assert tag1.id in canrep_after_update["tags"]


async def test_that_a_canned_response_can_be_created_with_metadata(
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
        "metadata": {"priority": "high", "category": "finance"},
    }

    response = await async_client.post("/canned_responses", json=payload)
    assert response.status_code == status.HTTP_201_CREATED

    canned_response = response.json()

    assert canned_response["value"] == payload["value"]
    assert canned_response["fields"] == payload["fields"]
    assert canned_response["metadata"] == {"priority": "high", "category": "finance"}

    assert "id" in canned_response
    assert "creation_utc" in canned_response


async def test_that_canned_response_metadata_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    canned_response_store = container[CannedResponseStore]

    canned_response = await canned_response_store.create_canned_response(
        value="Your balance is {{balance}}",
        fields=[
            CannedResponseField(name="balance", description="Account balance", examples=["5000"])
        ],
        metadata={"old_key": "old_value", "category": "finance"},
    )

    response = await async_client.patch(
        f"/canned_responses/{canned_response.id}",
        json={
            "metadata": {
                "set": {
                    "priority": "high",
                    "category": "support",
                },
                "unset": ["old_key"],
            }
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_canned_response = response.json()

    assert updated_canned_response["id"] == canned_response.id
    assert updated_canned_response["metadata"] == {"priority": "high", "category": "support"}


async def test_that_canned_response_metadata_can_be_set_without_unset(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    canned_response_store = container[CannedResponseStore]

    canned_response = await canned_response_store.create_canned_response(
        value="Your balance is {{balance}}",
        fields=[
            CannedResponseField(name="balance", description="Account balance", examples=["5000"])
        ],
        metadata={"existing_key": "existing_value"},
    )

    response = await async_client.patch(
        f"/canned_responses/{canned_response.id}",
        json={
            "metadata": {
                "set": {
                    "priority": "high",
                    "category": "support",
                }
            }
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_canned_response = response.json()

    assert updated_canned_response["id"] == canned_response.id
    assert updated_canned_response["metadata"] == {
        "existing_key": "existing_value",
        "priority": "high",
        "category": "support",
    }


async def test_that_canned_response_metadata_can_be_unset_without_set(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    canned_response_store = container[CannedResponseStore]

    canned_response = await canned_response_store.create_canned_response(
        value="Your balance is {{balance}}",
        fields=[
            CannedResponseField(name="balance", description="Account balance", examples=["5000"])
        ],
        metadata={"key1": "value1", "key2": "value2", "key3": "value3"},
    )

    response = await async_client.patch(
        f"/canned_responses/{canned_response.id}",
        json={"metadata": {"unset": ["key2", "key3"]}},
    )

    assert response.status_code == status.HTTP_200_OK
    updated_canned_response = response.json()

    assert updated_canned_response["id"] == canned_response.id
    assert updated_canned_response["metadata"] == {"key1": "value1"}


async def test_that_canned_response_metadata_can_handle_empty_operations(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    canned_response_store = container[CannedResponseStore]

    canned_response = await canned_response_store.create_canned_response(
        value="Your balance is {{balance}}",
        fields=[
            CannedResponseField(name="balance", description="Account balance", examples=["5000"])
        ],
        metadata={"existing": "value"},
    )

    # Test with empty set and unset
    response = await async_client.patch(
        f"/canned_responses/{canned_response.id}",
        json={"metadata": {"set": {}, "unset": []}},
    )

    assert response.status_code == status.HTTP_200_OK
    updated_canned_response = response.json()

    assert updated_canned_response["id"] == canned_response.id
    assert updated_canned_response["metadata"] == {"existing": "value"}


async def test_that_canned_response_metadata_unset_nonexistent_key_is_ignored(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    canned_response_store = container[CannedResponseStore]

    canned_response = await canned_response_store.create_canned_response(
        value="Your balance is {{balance}}",
        fields=[
            CannedResponseField(name="balance", description="Account balance", examples=["5000"])
        ],
        metadata={"key1": "value1"},
    )

    response = await async_client.patch(
        f"/canned_responses/{canned_response.id}",
        json={"metadata": {"set": {"key2": "value2"}, "unset": ["nonexistent_key"]}},
    )

    assert response.status_code == status.HTTP_200_OK
    updated_canned_response = response.json()

    assert updated_canned_response["id"] == canned_response.id
    assert updated_canned_response["metadata"] == {"key1": "value1", "key2": "value2"}
