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

import httpx
from fastapi import status, HTTPException
from lagom import Container
from pytest import mark, raises

from parlant.core.journeys import JourneyStore
from parlant.core.guidelines import GuidelineStore
from parlant.core.tags import Tag, TagStore
from parlant.core.common import ItemNotFoundError


async def test_that_a_journey_can_be_created(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    payload = {
        "title": "Customer Onboarding",
        "description": "Guide new customers through onboarding steps",
        "conditions": ["Customer asks for onboarding help"],
    }
    response = await async_client.post("/journeys", json=payload)

    assert response.status_code == status.HTTP_201_CREATED

    journey = response.json()

    assert journey["title"] == payload["title"]
    assert journey["description"] == payload["description"]
    assert journey["tags"] == []

    assert len(journey["conditions"]) == 1
    guideline = await guideline_store.read_guideline(guideline_id=journey["conditions"][0])
    assert guideline.id == journey["conditions"][0]

    guideline_after_update = await guideline_store.read_guideline(guideline.id)
    assert guideline_after_update.tags == [Tag.for_journey_id(journey["id"])]


async def test_that_a_journey_can_be_created_with_multiple_conditions(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    payload = {
        "title": "Customer Onboarding",
        "description": "Guide new customers through onboarding steps",
        "conditions": ["Customer asks for onboarding help", "Customer wants to signup"],
    }
    response = await async_client.post("/journeys", json=payload)

    assert response.status_code == status.HTTP_201_CREATED

    journey = response.json()

    assert journey["title"] == payload["title"]
    assert journey["description"] == payload["description"]
    assert journey["tags"] == []

    assert len(journey["conditions"]) == 2
    first_guideline = await guideline_store.read_guideline(guideline_id=journey["conditions"][0])
    second_guideline = await guideline_store.read_guideline(guideline_id=journey["conditions"][1])
    assert first_guideline.id == journey["conditions"][0]
    assert second_guideline.id == journey["conditions"][1]


async def test_that_a_journey_can_be_created_with_tags(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]

    tag1 = await tag_store.create_tag("tag1")
    tag2 = await tag_store.create_tag("tag2")

    response = await async_client.post(
        "/journeys",
        json={
            "title": "Product Support",
            "description": "Assist customers with product issues",
            "conditions": ["Customer reports an issue"],
            "tags": [tag1.id, tag2.id],
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    journey_dto = (
        (await async_client.get(f"/journeys/{response.json()['id']}")).raise_for_status().json()
    )

    assert journey_dto["title"] == "Product Support"
    assert set(journey_dto["tags"]) == {tag1.id, tag2.id}


async def test_that_a_journey_can_be_created_with_custom_id(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    """Test that a journey can be created with a custom ID."""
    guideline_store = container[GuidelineStore]
    custom_id = "custom-journey-id-123"

    payload = {
        "id": custom_id,
        "title": "Custom ID Journey",
        "description": "Journey with a custom identifier",
        "conditions": ["Custom ID condition"],
    }
    response = await async_client.post("/journeys", json=payload)

    assert response.status_code == status.HTTP_201_CREATED

    journey = response.json()

    # Verify that the custom ID was used
    assert journey["id"] == custom_id
    assert journey["title"] == payload["title"]
    assert journey["description"] == payload["description"]
    assert journey["tags"] == []

    assert len(journey["conditions"]) == 1
    guideline = await guideline_store.read_guideline(guideline_id=journey["conditions"][0])
    assert guideline.id == journey["conditions"][0]


async def test_that_creating_journey_with_duplicate_id_fails(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    """Test that creating a journey with a duplicate ID fails appropriately."""
    payload = {
        "id": "duplicate-id-test",
        "title": "First Journey",
        "description": "First journey with this ID",
        "conditions": ["First condition"],
    }

    # Create first journey
    response1 = await async_client.post("/journeys", json=payload)
    assert response1.status_code == status.HTTP_201_CREATED

    # Try to create second journey with same ID
    payload["title"] = "Second Journey"
    payload["description"] = "This should fail due to duplicate ID"

    with raises(HTTPException) as exc_info:
        _ = await async_client.post("/journeys", json=payload)

    # Should fail due to duplicate ID
    assert exc_info.value.detail == "Journey with id 'duplicate-id-test' already exists"


async def test_that_journeys_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    _ = (
        (
            await async_client.post(
                "/journeys",
                json={
                    "title": "Customer Onboarding",
                    "description": "Guide new customers",
                    "conditions": ["Customer asks for onboarding help"],
                },
            )
        )
        .raise_for_status()
        .json()
    )

    journeys = (await async_client.get("/journeys")).raise_for_status().json()

    assert len(journeys) == 1
    first_journey = journeys[0]
    assert first_journey["title"] == "Customer Onboarding"

    assert len(first_journey["conditions"]) == 1
    guideline = await guideline_store.read_guideline(guideline_id=first_journey["conditions"][0])
    assert guideline.id == first_journey["conditions"][0]


async def test_that_a_journey_can_be_read(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    journey = (
        (
            await async_client.post(
                "/journeys",
                json={
                    "title": "Customer Onboarding",
                    "description": "Guide new customers",
                    "conditions": ["Customer asks for onboarding help"],
                },
            )
        )
        .raise_for_status()
        .json()
    )

    journey_dto = (await async_client.get(f"/journeys/{journey['id']}")).raise_for_status().json()

    assert journey_dto["title"] == "Customer Onboarding"
    assert journey_dto["description"] == "Guide new customers"

    assert len(journey_dto["conditions"]) == 1
    guideline = await guideline_store.read_guideline(guideline_id=journey_dto["conditions"][0])
    assert guideline.id == journey_dto["conditions"][0]


@mark.parametrize(
    "update_payload, expected_title, expected_description, expected_condition",
    [
        (
            {"title": "New Title"},
            "New Title",
            "Guide new customers",
            "Customer asks for onboarding help",
        ),
        (
            {"description": "Updated description"},
            "Customer Onboarding",
            "Updated description",
            "Customer asks for onboarding help",
        ),
    ],
)
async def test_that_a_journey_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
    update_payload: dict[str, Any],
    expected_title: str,
    expected_description: str,
    expected_condition: str,
) -> None:
    journey = (
        (
            await async_client.post(
                "/journeys",
                json={
                    "title": "Customer Onboarding",
                    "description": "Guide new customers",
                    "conditions": ["Customer asks for onboarding help"],
                },
            )
        )
        .raise_for_status()
        .json()
    )

    response = await async_client.patch(f"/journeys/{journey['id']}", json=update_payload)
    response.raise_for_status()
    updated_journey = response.json()

    assert updated_journey["title"] == expected_title
    assert updated_journey["description"] == expected_description


async def test_that_tags_can_be_added_to_a_journey(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]

    tag1 = await tag_store.create_tag("tag1")
    tag2 = await tag_store.create_tag("tag2")
    tag3 = await tag_store.create_tag("tag3")

    journey = (
        (
            await async_client.post(
                "/journeys",
                json={
                    "title": "Customer Onboarding",
                    "description": "Guide new customers",
                    "conditions": ["Customer asks for onboarding help"],
                    "tags": [tag1.id],
                },
            )
        )
        .raise_for_status()
        .json()
    )

    update_payload = {"tags": {"add": [tag2.id, tag3.id]}}
    response = await async_client.patch(f"/journeys/{journey['id']}", json=update_payload)
    response.raise_for_status()
    updated_journey = response.json()

    assert tag1.id in updated_journey["tags"]
    assert tag2.id in updated_journey["tags"]
    assert tag3.id in updated_journey["tags"]


async def test_that_tags_can_be_removed_from_a_journey(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]

    tag2 = await tag_store.create_tag("tag2")
    tag3 = await tag_store.create_tag("tag3")

    journey = await journey_store.create_journey(
        title="Customer Onboarding",
        description="Guide new customers",
        conditions=[],
        tags=[tag2.id, tag3.id],
    )

    update_payload = {"tags": {"remove": [tag2.id]}}
    _ = (
        await async_client.patch(f"/journeys/{journey.id}", json=update_payload)
    ).raise_for_status()
    journey_after_second_update = (
        (await async_client.get(f"/journeys/{journey.id}")).raise_for_status().json()
    )
    assert tag2.id not in journey_after_second_update["tags"]
    assert tag3.id in journey_after_second_update["tags"]


async def test_that_a_journey_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    journey_store = container[JourneyStore]
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="Customer asks for onboarding help",
        action=None,
    )

    journey = await journey_store.create_journey(
        title="Customer Onboarding",
        description="Guide new customers",
        conditions=[guideline.id],
    )

    delete_response = await async_client.delete(f"/journeys/{journey.id}")
    assert delete_response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await journey_store.read_journey(journey.id)


async def test_that_a_guideline_is_deleted_when_it_is_removed_from_all_journeys(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    journey_store = container[JourneyStore]
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="Customer asks for onboarding help",
        action=None,
    )

    journey = await journey_store.create_journey(
        title="Customer Onboarding",
        description="Guide new customers",
        conditions=[guideline.id],
    )

    delete_response = await async_client.delete(f"/journeys/{journey.id}")
    assert delete_response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await guideline_store.read_guideline(guideline.id)


async def test_that_a_guideline_is_not_deleted_when_it_is_used_in_multiple_journeys(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]

    guideline = await guideline_store.create_guideline(
        condition="Customer asks for onboarding help",
        action=None,
    )

    journey_to_delete = await journey_store.create_journey(
        title="Customer Onboarding",
        description="Guide new customers",
        conditions=[guideline.id],
    )

    journey_to_keep = await journey_store.create_journey(
        title="Customer Signup",
        description="Guide new customers to signup",
        conditions=[guideline.id],
    )

    await guideline_store.upsert_tag(
        guideline_id=guideline.id, tag_id=Tag.for_journey_id(journey_to_delete.id)
    )

    await guideline_store.upsert_tag(
        guideline_id=guideline.id, tag_id=Tag.for_journey_id(journey_to_keep.id)
    )

    delete_response = await async_client.delete(f"/journeys/{journey_to_delete.id}")
    assert delete_response.status_code == status.HTTP_204_NO_CONTENT

    guideline_after_update = await guideline_store.read_guideline(guideline.id)
    assert guideline_after_update.tags == [Tag.for_journey_id(journey_to_keep.id)]


async def test_that_a_tag_can_be_added_to_a_journey(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]

    tag = await tag_store.create_tag("new_tag")

    journey = await journey_store.create_journey(
        title="Customer Onboarding",
        description="Guide new customers",
        conditions=[],
    )

    response = await async_client.patch(
        f"/journeys/{journey.id}",
        json={"tags": {"add": [tag.id]}},
    )
    response.raise_for_status()
    updated_journey = response.json()

    assert tag.id in updated_journey["tags"]


async def test_that_a_tag_can_be_removed_from_a_journey(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]

    tag = await tag_store.create_tag("removable_tag")
    journey = await journey_store.create_journey(
        title="Customer Onboarding",
        description="Guide new customers",
        conditions=[],
        tags=[tag.id],
    )

    response = await async_client.patch(
        f"/journeys/{journey.id}",
        json={"tags": {"remove": [tag.id]}},
    )
    response.raise_for_status()
    updated_journey = response.json()

    assert tag.id not in updated_journey["tags"]


async def test_that_conditions_can_be_added_to_a_journey(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]

    guideline = await guideline_store.create_guideline(
        condition="New Condition",
        action=None,
    )
    journey = await journey_store.create_journey(
        title="Customer Onboarding",
        description="Guide new customers",
        conditions=[],
    )

    response = await async_client.patch(
        f"/journeys/{journey.id}",
        json={"conditions": {"add": [guideline.id]}},
    )
    response.raise_for_status()
    updated_journey = response.json()

    assert guideline.id in updated_journey["conditions"]

    guideline_after_update = await guideline_store.read_guideline(guideline.id)
    assert guideline_after_update.tags == [Tag.for_journey_id(journey.id)]


async def test_that_conditions_can_be_removed_from_a_journey(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]

    guideline = await guideline_store.create_guideline(
        condition="Removable Condition",
        action=None,
    )

    journey_to_delete = await journey_store.create_journey(
        title="Customer Onboarding",
        description="Guide new customers",
        conditions=[guideline.id],
    )

    journey_to_keep = await journey_store.create_journey(
        title="Customer Signup",
        description="Guide new customers to signup",
        conditions=[guideline.id],
    )

    await guideline_store.upsert_tag(
        guideline_id=guideline.id, tag_id=Tag.for_journey_id(journey_to_keep.id)
    )

    await guideline_store.upsert_tag(
        guideline_id=guideline.id, tag_id=Tag.for_journey_id(journey_to_delete.id)
    )

    response = await async_client.patch(
        f"/journeys/{journey_to_delete.id}",
        json={"conditions": {"remove": [guideline.id]}},
    )
    response.raise_for_status()
    updated_journey = response.json()

    assert guideline.id not in updated_journey["conditions"]

    guideline_after_update = await guideline_store.read_guideline(guideline.id)
    assert guideline_after_update.tags == [Tag.for_journey_id(journey_to_keep.id)]


async def test_that_a_guideline_is_deleted_when_conditions_are_removed_from_all_journeys(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]

    guideline = await guideline_store.create_guideline(
        condition="Removable Condition",
        action=None,
    )

    journey = await journey_store.create_journey(
        title="Customer Onboarding",
        description="Guide new customers",
        conditions=[guideline.id],
    )

    await journey_store.create_journey(
        title="Customer Signup",
        description="Guide new customers to signup",
        conditions=[guideline.id],
    )

    await guideline_store.upsert_tag(
        guideline_id=guideline.id, tag_id=Tag.for_journey_id(journey.id)
    )

    response = await async_client.patch(
        f"/journeys/{journey.id}",
        json={"conditions": {"remove": [guideline.id]}},
    )
    response.raise_for_status()

    with raises(ItemNotFoundError):
        await guideline_store.read_guideline(guideline.id)


async def test_that_journeys_can_be_filtered_by_tag(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]

    tag = await tag_store.create_tag("tag1")
    journey = await journey_store.create_journey(
        title="Customer Onboarding",
        description="Guide new customers",
        conditions=[],
        tags=[tag.id],
    )

    _ = await journey_store.create_journey(
        title="Customer Onboarding",
        description="Guide new customers",
        conditions=[],
    )

    response = await async_client.get(f"/journeys?tag_id={tag.id}")
    response.raise_for_status()
    journeys = response.json()

    assert len(journeys) == 1
    assert journeys[0]["id"] == journey.id


async def test_that_journey_composition_mode_can_be_set_and_updated(
    async_client: httpx.AsyncClient,
) -> None:
    # Create journey with CANNED_COMPOSITED mode
    response = await async_client.post(
        "/journeys",
        json={
            "title": "Customer Onboarding",
            "description": "Guide new customers through onboarding",
            "conditions": ["User asks about onboarding"],
            "composition_mode": "composited_canned",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    journey = response.json()
    journey_id = journey["id"]

    # Check that the composition mode is set correctly after creation
    assert journey["composition_mode"] == "composited_canned"

    # Retrieve journey and verify composition mode
    response = await async_client.get(f"/journeys/{journey_id}")
    assert response.status_code == status.HTTP_200_OK
    journey = response.json()
    assert journey["composition_mode"] == "composited_canned"

    # Update journey to CANNED_STRICT mode
    response = await async_client.patch(
        f"/journeys/{journey_id}",
        json={
            "composition_mode": "strict_canned",
        },
    )

    assert response.status_code == status.HTTP_200_OK
    journey = response.json()

    # Check that the composition mode is updated correctly
    assert journey["composition_mode"] == "strict_canned"

    # Retrieve journey again and verify composition mode
    response = await async_client.get(f"/journeys/{journey_id}")
    assert response.status_code == status.HTTP_200_OK
    journey = response.json()
    assert journey["composition_mode"] == "strict_canned"
