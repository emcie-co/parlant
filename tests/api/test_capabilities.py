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

import httpx
from fastapi import status
from lagom import Container
from pytest import mark, raises

from parlant.core.agents import AgentStore
from parlant.core.capabilities import CapabilityStore
from parlant.core.journeys import JourneyStore
from parlant.core.tags import Tag, TagStore
from parlant.core.common import ItemNotFoundError


async def test_that_a_capability_can_be_created(
    async_client: httpx.AsyncClient,
) -> None:
    payload = {
        "title": "Provide Replacement Phone",
        "description": "Provide a replacement phone when a customer needs repair for their phone.",
        "signals": ["My phone is broken", "I need a replacement while my phone is being repaired"],
    }

    response = await async_client.post("/capabilities", json=payload)
    assert response.status_code == status.HTTP_201_CREATED

    capability = response.json()
    assert capability["title"] == payload["title"]
    assert capability["description"] == payload["description"]
    assert capability["signals"] == payload["signals"]
    assert capability["tags"] == []


async def test_that_a_capability_can_be_created_with_tags(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]
    agent_store = container[AgentStore]
    journey_store = container[JourneyStore]

    agent = await agent_store.create_agent("Test Agent")
    journey = await journey_store.create_journey(
        title="Customer Support Journey",
        description="A journey for customer support interactions.",
        conditions=[],
    )

    tag1 = await tag_store.create_tag("tag1")
    tag2 = await tag_store.create_tag("tag2")
    agent_tag = Tag.for_agent_id(agent.id)
    journey_tag = Tag.for_journey_id(journey.id)

    payload = {
        "title": "Summarization",
        "description": "Summarizes long documents.",
        "signals": ["Summarize this article", "Give me a summary"],
        "tags": [tag1.id, tag2.id, agent_tag, journey_tag],
    }

    response = await async_client.post("/capabilities", json=payload)
    assert response.status_code == status.HTTP_201_CREATED

    capability = response.json()
    assert capability["title"] == payload["title"]
    assert set(capability["tags"]) == {tag1.id, tag2.id, agent_tag, journey_tag}


async def test_that_capabilities_can_be_listed(
    async_client: httpx.AsyncClient,
) -> None:
    _ = (
        (
            await async_client.post(
                "/capabilities",
                json={
                    "title": "Provide Replacement Phone",
                    "description": "Provide a replacement phone when a customer needs repair for their phone.",
                    "signals": [
                        "My phone is broken",
                        "I need a replacement while my phone is being repaired",
                    ],
                },
            )
        )
        .raise_for_status()
        .json()
    )

    capabilities = (await async_client.get("/capabilities")).raise_for_status().json()

    assert len(capabilities) == 1
    assert capabilities[0]["title"] == "Provide Replacement Phone"


async def test_that_a_capability_can_be_read(
    async_client: httpx.AsyncClient,
) -> None:
    capability = (
        (
            await async_client.post(
                "/capabilities",
                json={
                    "title": "Q&A",
                    "description": "Answers questions.",
                    "signals": ["What is Parlant?"],
                },
            )
        )
        .raise_for_status()
        .json()
    )

    capability_dto = (
        (await async_client.get(f"/capabilities/{capability['id']}")).raise_for_status().json()
    )

    assert capability_dto["title"] == "Q&A"
    assert capability_dto["description"] == "Answers questions."
    assert capability_dto["signals"] == ["What is Parlant?"]


@mark.parametrize(
    "update_payload, expected_title, expected_description, expected_signals",
    [
        (
            {"title": "New Title"},
            "New Title",
            "Answers questions.",
            ["What is Parlant?"],
        ),
        (
            {"description": "Updated description"},
            "Q&A",
            "Updated description",
            ["What is Parlant?"],
        ),
        (
            {"signals": ["How does it work?"]},
            "Q&A",
            "Answers questions.",
            ["How does it work?"],
        ),
    ],
)
async def test_that_a_capability_can_be_updated(
    async_client: httpx.AsyncClient,
    update_payload: dict[str, str],
    expected_title: str,
    expected_description: str,
    expected_signals: list[str],
) -> None:
    capability = (
        (
            await async_client.post(
                "/capabilities",
                json={
                    "title": "Q&A",
                    "description": "Answers questions.",
                    "signals": ["What is Parlant?"],
                },
            )
        )
        .raise_for_status()
        .json()
    )

    response = await async_client.patch(f"/capabilities/{capability['id']}", json=update_payload)
    response.raise_for_status()
    updated_capability = response.json()

    assert updated_capability["title"] == expected_title
    assert updated_capability["description"] == expected_description
    assert updated_capability["signals"] == expected_signals


async def test_that_tags_can_be_added_to_a_capability(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]

    tag1 = await tag_store.create_tag("tag1")
    tag2 = await tag_store.create_tag("tag2")

    capability = (
        (
            await async_client.post(
                "/capabilities",
                json={
                    "title": "Provide Replacement Phone",
                    "description": "Provide a replacement phone when a customer needs repair for their phone.",
                    "signals": [
                        "My phone is broken",
                        "I need a replacement while my phone is being repaired",
                    ],
                },
            )
        )
        .raise_for_status()
        .json()
    )

    update_payload = {"tags": {"add": [tag1.id, tag2.id]}}
    response = await async_client.patch(f"/capabilities/{capability['id']}", json=update_payload)
    response.raise_for_status()
    updated_capability = response.json()

    assert tag1.id in updated_capability["tags"]
    assert tag2.id in updated_capability["tags"]


async def test_that_tags_can_be_removed_from_a_capability(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]
    capability_store = container[CapabilityStore]

    tag1 = await tag_store.create_tag("tag1")
    tag2 = await tag_store.create_tag("tag2")

    capability = await capability_store.create_capability(
        title="Translation",
        description="Translates text.",
        signals=["Translate this sentence"],
        tags=[tag1.id, tag2.id],
    )

    update_payload = {"tags": {"remove": [tag1.id]}}
    _ = (
        await async_client.patch(f"/capabilities/{capability.id}", json=update_payload)
    ).raise_for_status()

    capability_after_update = (
        (await async_client.get(f"/capabilities/{capability.id}")).raise_for_status().json()
    )

    assert tag1.id not in capability_after_update["tags"]
    assert tag2.id in capability_after_update["tags"]


async def test_that_agent_tag_can_be_added_to_a_capability(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent = await container[AgentStore].create_agent("Test Agent")

    capability = (
        (
            await async_client.post(
                "/capabilities",
                json={
                    "title": "Provide Replacement Phone",
                    "description": "Provide a replacement phone when a customer needs repair for their phone.",
                    "signals": [
                        "My phone is broken",
                        "I need a replacement while my phone is being repaired",
                    ],
                },
            )
        )
        .raise_for_status()
        .json()
    )
    agent_tag = Tag.for_agent_id(agent.id)

    update_payload = {"tags": {"add": [agent_tag]}}
    response = await async_client.patch(f"/capabilities/{capability['id']}", json=update_payload)
    response.raise_for_status()
    updated_capability = response.json()

    assert updated_capability["tags"] == [agent_tag]


async def test_that_agent_tag_can_be_removed_from_a_capability(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]
    capability_store = container[CapabilityStore]

    agent = await container[AgentStore].create_agent("Test Agent")

    tag1 = await tag_store.create_tag("tag1")

    agent_tag = Tag.for_agent_id(agent.id)

    capability = await capability_store.create_capability(
        title="Translation",
        description="Translates text.",
        signals=["Translate this sentence"],
        tags=[agent_tag, tag1.id],
    )

    update_payload = {"tags": {"remove": [agent_tag]}}
    _ = (
        await async_client.patch(f"/capabilities/{capability.id}", json=update_payload)
    ).raise_for_status()

    capability_after_update = (
        (await async_client.get(f"/capabilities/{capability.id}")).raise_for_status().json()
    )

    assert agent_tag not in capability_after_update["tags"]
    assert tag1.id in capability_after_update["tags"]


async def test_that_journey_tags_can_be_added_to_a_capability(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]

    journey = await journey_store.create_journey(
        title="Customer Support Journey",
        description="A journey for customer support interactions.",
        conditions=[],
    )
    journey_tag = Tag.for_journey_id(journey.id)

    tag1 = await tag_store.create_tag("tag1")

    capability = (
        (
            await async_client.post(
                "/capabilities",
                json={
                    "title": "Provide Replacement Phone",
                    "description": "Provide a replacement phone when a customer needs repair for their phone.",
                    "signals": [
                        "My phone is broken",
                        "I need a replacement while my phone is being repaired",
                    ],
                },
            )
        )
        .raise_for_status()
        .json()
    )

    update_payload = {"tags": {"add": [tag1.id, journey_tag]}}
    response = await async_client.patch(f"/capabilities/{capability['id']}", json=update_payload)
    response.raise_for_status()
    updated_capability = response.json()

    assert tag1.id in updated_capability["tags"]
    assert journey_tag in updated_capability["tags"]


async def test_that_journey_tags_can_be_removed_from_a_capability(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]
    capability_store = container[CapabilityStore]
    journey_store = container[JourneyStore]

    journey = await journey_store.create_journey(
        title="Customer Support Journey",
        description="A journey for customer support interactions.",
        conditions=[],
    )
    journey_tag = Tag.for_journey_id(journey.id)

    tag1 = await tag_store.create_tag("tag1")

    capability = await capability_store.create_capability(
        title="Translation",
        description="Translates text.",
        signals=["Translate this sentence"],
        tags=[tag1.id, journey_tag],
    )

    update_payload = {"tags": {"remove": [journey_tag]}}
    _ = (
        await async_client.patch(f"/capabilities/{capability.id}", json=update_payload)
    ).raise_for_status()

    capability_after_update = (
        (await async_client.get(f"/capabilities/{capability.id}")).raise_for_status().json()
    )

    assert journey_tag not in capability_after_update["tags"]
    assert tag1.id in capability_after_update["tags"]


async def test_that_a_capability_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    capability_store = container[CapabilityStore]

    capability = await capability_store.create_capability(
        title="Provide Replacement Phone",
        description="Provide a replacement phone when a customer needs repair for their phone.",
        signals=["My phone is broken", "I need a replacement while my phone is being repaired"],
    )

    delete_response = await async_client.delete(f"/capabilities/{capability.id}")
    assert delete_response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await capability_store.read_capability(capability.id)


async def test_that_capabilities_can_be_filtered_by_tag(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]
    capability_store = container[CapabilityStore]

    tag = await tag_store.create_tag("tag1")

    _ = await capability_store.create_capability(
        title="Provide Replacement Phone",
        description="Provide a replacement phone when a customer needs repair for their phone.",
        signals=["My phone is broken", "I need a replacement while my phone is being repaired"],
        tags=[tag.id],
    )

    _ = await capability_store.create_capability(
        title="Reset Password",
        description="Helping customer reset their account password",
        signals=["My password isn't what I thought"],
    )

    response = await async_client.get(f"/capabilities?tag_id={tag.id}")
    response.raise_for_status()
    capabilities = response.json()

    assert len(capabilities) == 1
