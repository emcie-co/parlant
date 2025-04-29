from typing import Any

import httpx
from fastapi import status
from lagom import Container
from pytest import mark, raises

from parlant.core.journeys import JourneyStore
from parlant.core.guidelines import GuidelineStore
from parlant.core.tags import TagStore
from parlant.core.common import ItemNotFoundError


async def test_that_a_journey_can_be_created(
    async_client: httpx.AsyncClient,
) -> None:
    payload = {
        "title": "Customer Onboarding",
        "description": "Guide new customers through onboarding steps",
        "condition": "Customer asks for onboarding help",
    }
    response = await async_client.post("/journeys", json=payload)

    assert response.status_code == status.HTTP_201_CREATED

    journey = response.json()

    assert journey["title"] == payload["title"]
    assert journey["description"] == payload["description"]
    assert journey["condition"] == payload["condition"]
    assert journey["tags"] == []


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
            "condition": "Customer reports an issue",
            "tags": [tag1.id, tag2.id],
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    journey_dto = (
        (await async_client.get(f"/journeys/{response.json()['id']}")).raise_for_status().json()
    )

    assert journey_dto["title"] == "Product Support"
    assert set(journey_dto["tags"]) == {tag1.id, tag2.id}


async def test_that_journeys_can_be_listed(
    async_client: httpx.AsyncClient,
) -> None:
    _ = (
        (
            await async_client.post(
                "/journeys",
                json={
                    "title": "Customer Onboarding",
                    "description": "Guide new customers",
                    "condition": "Customer asks for onboarding help",
                },
            )
        )
        .raise_for_status()
        .json()
    )

    journeys = (await async_client.get("/journeys")).raise_for_status().json()

    assert len(journeys) == 1
    assert journeys[0]["title"] == "Customer Onboarding"


async def test_that_a_journey_can_be_read(
    async_client: httpx.AsyncClient,
) -> None:
    journey = (
        (
            await async_client.post(
                "/journeys",
                json={
                    "title": "Customer Onboarding",
                    "description": "Guide new customers",
                    "condition": "Customer asks for onboarding help",
                },
            )
        )
        .raise_for_status()
        .json()
    )

    journey_dto = (await async_client.get(f"/journeys/{journey['id']}")).raise_for_status().json()

    assert journey_dto["title"] == "Customer Onboarding"
    assert journey_dto["description"] == "Guide new customers"
    assert journey_dto["condition"] == "Customer asks for onboarding help"


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
        (
            {"condition": "Customer requests onboarding"},
            "Customer Onboarding",
            "Guide new customers",
            "Customer requests onboarding",
        ),
    ],
)
async def test_that_a_journey_can_be_updated(
    async_client: httpx.AsyncClient,
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
                    "condition": "Customer asks for onboarding help",
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
    assert updated_journey["condition"] == expected_condition


async def test_that_tags_can_be_added_and_removed_from_a_journey(
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
                    "condition": "Customer asks for onboarding help",
                    "tags": [tag1.id],
                },
            )
        )
        .raise_for_status()
        .json()
    )

    update_payload = {"tags": [tag2.id, tag3.id]}
    response = await async_client.patch(f"/journeys/{journey['id']}", json=update_payload)
    response.raise_for_status()
    updated_journey = response.json()

    assert tag1.id not in updated_journey["tags"]
    assert tag2.id in updated_journey["tags"]
    assert tag3.id in updated_journey["tags"]

    update_payload = {"tags": [tag3.id]}
    _ = (
        await async_client.patch(f"/journeys/{journey['id']}", json=update_payload)
    ).raise_for_status()
    journey_after_second_update = (
        (await async_client.get(f"/journeys/{journey['id']}")).raise_for_status().json()
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
        condition=guideline.id,
    )

    delete_response = await async_client.delete(f"/journeys/{journey.id}")
    assert delete_response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await journey_store.read_journey(journey.id)
