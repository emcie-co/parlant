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

from parlant.core.agents import AgentId, AgentStore
from parlant.core.journeys import JourneyStore
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipEntity,
    RelationshipStore,
)
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineStore
from parlant.core.tags import Tag, TagId, TagStore
from parlant.core.tools import LocalToolService, ToolOverlap


async def create_guidelines_and_create_relationships_between_them(
    container: Container,
    agent_id: AgentId,
    guideline_contents: list[GuidelineContent],
) -> list[Guideline]:
    guidelines = [
        await container[GuidelineStore].create_guideline(
            condition=gc.condition,
            action=gc.action,
        )
        for gc in guideline_contents
    ]

    for guideline in guidelines:
        _ = await container[GuidelineStore].upsert_tag(
            guideline_id=guideline.id,
            tag_id=Tag.for_agent_id(agent_id),
        )

    for source, target in zip(guidelines, guidelines[1:]):
        await container[RelationshipStore].create_relationship(
            source=RelationshipEntity(
                id=source.id,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=target.id,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            kind=RelationshipKind.ENTAILMENT,
        )

    return guidelines


async def test_that_a_guideline_can_be_created(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/guidelines",
        json={
            "condition": "the customer asks about pricing",
            "action": "provide current pricing information",
            "enabled": True,
            "metadata": {"key1": "value1", "key2": "value2"},
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    guideline = response.json()
    assert guideline["condition"] == "the customer asks about pricing"
    assert guideline["action"] == "provide current pricing information"
    assert guideline["enabled"] is True
    assert guideline["tags"] == []
    assert guideline["metadata"] == {"key1": "value1", "key2": "value2"}


async def test_that_a_guideline_can_be_created_without_an_action(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/guidelines",
        json={"condition": "the customer asks about pricing"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    guideline = response.json()
    assert guideline["condition"] == "the customer asks about pricing"
    assert guideline["action"] is None


async def test_that_a_guideline_can_be_created_with_custom_id(
    async_client: httpx.AsyncClient,
) -> None:
    """Test that a guideline can be created with a custom ID."""
    custom_id = "custom-guideline-id-456"

    response = await async_client.post(
        "/guidelines",
        json={
            "id": custom_id,
            "condition": "the customer mentions a custom requirement",
            "action": "provide personalized assistance",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    guideline = response.json()

    # Verify that the custom ID was used
    assert guideline["id"] == custom_id
    assert guideline["condition"] == "the customer mentions a custom requirement"
    assert guideline["action"] == "provide personalized assistance"
    assert guideline["enabled"] is True
    assert guideline["tags"] == []
    assert guideline["metadata"] == {}


async def test_that_creating_guideline_with_duplicate_id_fails(
    async_client: httpx.AsyncClient,
) -> None:
    """Test that creating a guideline with a duplicate ID fails appropriately."""
    custom_id = "duplicate-guideline-id"

    # Create first guideline
    response1 = await async_client.post(
        "/guidelines",
        json={
            "id": custom_id,
            "condition": "first condition",
            "action": "first action",
        },
    )
    assert response1.status_code == status.HTTP_201_CREATED

    # Try to create second guideline with same ID
    response2 = await async_client.post(
        "/guidelines",
        json={
            "id": custom_id,
            "condition": "second condition",
            "action": "second action",
        },
    )

    # Should fail due to duplicate ID
    assert response2.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert "already exists" in response2.text


async def test_that_a_guideline_can_be_created_with_tags(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]
    agent_store = container[AgentStore]
    journey_store = container[JourneyStore]

    agent = await agent_store.create_agent("Test Agent")
    agent_tag = Tag.for_agent_id(agent.id)

    journey = await journey_store.create_journey(
        title="Customer Support Journey",
        description="A journey for customer support interactions.",
        conditions=[],
    )
    journey_tag = Tag.for_journey_id(journey.id)

    tag_1 = await tag_store.create_tag(name="pricing")
    tag_2 = await tag_store.create_tag(name="sales")

    response = await async_client.post(
        "/guidelines",
        json={
            "condition": "the customer asks about pricing",
            "action": "provide current pricing information",
            "tags": [
                tag_1.id,
                tag_1.id,
                tag_2.id,
                agent_tag,
                journey_tag,
            ],
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    guideline_dto = (
        (await async_client.get(f"/guidelines/{response.json()['id']}")).raise_for_status().json()
    )

    assert guideline_dto["guideline"]["condition"] == "the customer asks about pricing"
    assert guideline_dto["guideline"]["action"] == "provide current pricing information"

    assert len(guideline_dto["guideline"]["tags"]) == 4
    assert set(guideline_dto["guideline"]["tags"]) == {tag_1.id, tag_2.id, agent_tag, journey_tag}


async def test_that_guidelines_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    first_guideline = [
        await guideline_store.create_guideline(
            condition=f"condition {i}",
            action=f"action {i}",
        )
        for i in range(2)
    ]
    second_guideline = await guideline_store.create_guideline(
        condition="condition 2",
        action="action 2",
    )

    response_guidelines = (await async_client.get("/guidelines")).raise_for_status().json()

    assert len(response_guidelines) >= 2
    assert any(first_guideline[0].id == g["id"] for g in response_guidelines)
    assert any(first_guideline[1].id == g["id"] for g in response_guidelines)
    assert any(second_guideline.id == g["id"] for g in response_guidelines)


async def test_that_guidelines_can_be_listed_by_tag(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    first_guideline = await guideline_store.create_guideline(
        condition="condition 1",
        action="action 1",
    )

    second_guideline = await guideline_store.create_guideline(
        condition="condition 2",
        action="action 2",
    )

    await guideline_store.upsert_tag(
        guideline_id=first_guideline.id,
        tag_id=TagId("tag_1"),
    )

    await guideline_store.upsert_tag(
        guideline_id=second_guideline.id,
        tag_id=TagId("tag_2"),
    )

    response_guidelines = (
        (await async_client.get("/guidelines?tag_id=tag_1")).raise_for_status().json()
    )

    assert len(response_guidelines) == 1
    assert response_guidelines[0]["id"] == first_guideline.id

    response_guidelines = (
        (await async_client.get("/guidelines?tag_id=tag_2")).raise_for_status().json()
    )


async def test_that_a_guideline_can_be_read(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer asks about the weather",
        action="provide the current weather update",
        metadata={"key1": "value1", "key2": "value2"},
    )

    item = (await async_client.get(f"/guidelines/{guideline.id}")).raise_for_status().json()

    assert item["guideline"]["id"] == guideline.id
    assert item["guideline"]["condition"] == "the customer asks about the weather"
    assert item["guideline"]["action"] == "provide the current weather update"
    assert item["guideline"]["metadata"] == {"key1": "value1", "key2": "value2"}
    assert len(item["relationships"]) == 0
    assert len(item["tool_associations"]) == 0


async def test_that_a_guideline_condition_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    response = await async_client.patch(
        f"/guidelines/{guideline.id}",
        json={
            "condition": "the customer inquires about weather",
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_guideline = response.json()["guideline"]

    assert updated_guideline["id"] == guideline.id
    assert updated_guideline["condition"] == "the customer inquires about weather"
    assert updated_guideline["action"] == guideline.content.action


async def test_that_a_guideline_action_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    response = await async_client.patch(
        f"/guidelines/{guideline.id}",
        json={
            "action": "give current weather information",
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_guideline = response.json()["guideline"]

    assert updated_guideline["id"] == guideline.id
    assert updated_guideline["condition"] == guideline.content.condition
    assert updated_guideline["action"] == "give current weather information"


async def test_that_a_guideline_can_be_disabled(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    response = await async_client.patch(
        f"/guidelines/{guideline.id}",
        json={
            "enabled": False,
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_guideline = response.json()["guideline"]

    assert updated_guideline["id"] == guideline.id
    assert updated_guideline["enabled"] is False


async def test_that_a_tag_can_be_added_to_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]

    tag = await tag_store.create_tag("test_tag")

    guideline = await guideline_store.create_guideline(
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    response = await async_client.patch(
        f"/guidelines/{guideline.id}",
        json={
            "tags": {
                "add": [tag.id],
            },
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_guideline = response.json()["guideline"]

    assert updated_guideline["id"] == guideline.id
    assert tag.id in updated_guideline["tags"]


async def test_that_a_tag_can_be_removed_from_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    # First add a tag
    await guideline_store.upsert_tag(
        guideline_id=guideline.id,
        tag_id=TagId("test_tag"),
    )

    response = await async_client.patch(
        f"/guidelines/{guideline.id}",
        json={
            "tags": {
                "remove": ["test_tag"],
            },
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_guideline = response.json()["guideline"]

    assert updated_guideline["id"] == guideline.id
    assert "test_tag" not in updated_guideline["tags"]


async def test_that_an_agent_tag_can_be_added_to_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    agent_store = container[AgentStore]

    agent = await agent_store.create_agent("test_agent")
    agent_tag = Tag.for_agent_id(agent.id)

    guideline = await guideline_store.create_guideline(
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    response = await async_client.patch(
        f"/guidelines/{guideline.id}",
        json={
            "tags": {
                "add": [agent_tag],
            },
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_guideline = response.json()["guideline"]

    assert updated_guideline["id"] == guideline.id
    assert agent_tag in updated_guideline["tags"]


async def test_that_a_journey_tag_can_be_added_to_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]

    journey = await journey_store.create_journey(
        title="test_journey",
        description="test_description",
        conditions=[],
    )
    journey_tag = Tag.for_journey_id(journey.id)

    guideline = await guideline_store.create_guideline(
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    response = await async_client.patch(
        f"/guidelines/{guideline.id}",
        json={
            "tags": {
                "add": [journey_tag],
            },
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_guideline = response.json()["guideline"]

    assert updated_guideline["id"] == guideline.id
    assert journey_tag in updated_guideline["tags"]


async def test_that_a_guideline_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer wants to unsubscribe",
        action="ask for confirmation",
    )

    (await async_client.delete(f"/guidelines/{guideline.id}")).raise_for_status()

    response = await async_client.get(f"/guidelines/{guideline.id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_a_tool_association_can_be_added_to_a_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    local_tool_service = container[LocalToolService]

    await local_tool_service.create_tool(
        name="fetch_event_data",
        module_path="some.module",
        description="",
        parameters={},
        required=[],
        overlap=ToolOverlap.NONE,
    )

    guideline = await guideline_store.create_guideline(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
    )

    service_name = "local"
    tool_name = "fetch_event_data"

    request_data = {
        "tool_associations": {
            "add": [
                {
                    "service_name": service_name,
                    "tool_name": tool_name,
                }
            ]
        }
    }

    response = await async_client.patch(
        f"/guidelines/{guideline.id}",
        json=request_data,
    )

    assert response.status_code == status.HTTP_200_OK

    tool_associations = response.json()["tool_associations"]

    assert any(
        a["guideline_id"] == guideline.id
        and a["tool_id"]["service_name"] == service_name
        and a["tool_id"]["tool_name"] == tool_name
        for a in tool_associations
    )


async def test_that_a_tag_can_be_added_to_a_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]

    tag = await tag_store.create_tag("test_tag")
    guideline = await guideline_store.create_guideline(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
    )

    response = await async_client.patch(
        f"/guidelines/{guideline.id}",
        json={"tags": {"add": [tag.id]}},
    )

    assert response.status_code == status.HTTP_200_OK
    updated_guideline = response.json()["guideline"]

    assert updated_guideline["id"] == guideline.id
    assert tag.id in updated_guideline["tags"]


async def test_that_a_tag_can_be_removed_from_a_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]

    tag = await tag_store.create_tag("test_tag")

    guideline = await guideline_store.create_guideline(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
    )

    await guideline_store.upsert_tag(
        guideline_id=guideline.id,
        tag_id=tag.id,
    )

    response = await async_client.patch(
        f"/guidelines/{guideline.id}",
        json={"tags": {"remove": [tag.id]}},
    )

    assert response.status_code == status.HTTP_200_OK
    updated_guideline = response.json()["guideline"]

    assert updated_guideline["id"] == guideline.id
    assert updated_guideline["tags"] == []


async def test_that_adding_nonexistent_agent_tag_to_guideline_returns_404(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
    )

    response = await async_client.patch(
        f"/guidelines/{guideline.id}",
        json={"tags": {"add": ["agent-id:nonexistent_agent"]}},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_adding_nonexistent_tag_to_guideline_returns_404(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
    )

    response = await async_client.patch(
        f"/guidelines/{guideline.id}",
        json={"tags": {"add": ["nonexistent_tag"]}},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_metadata_can_be_updated_for_a_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
        metadata={"key3": "value2"},
    )

    response = await async_client.patch(
        f"/guidelines/{guideline.id}",
        json={
            "metadata": {
                "set": {
                    "key1": "value1",
                    "key2": "value2",
                },
                "unset": ["key3"],
            }
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_guideline = response.json()["guideline"]

    assert updated_guideline["id"] == guideline.id
    assert updated_guideline["metadata"] == {"key1": "value1", "key2": "value2"}


async def test_that_condition_association_is_deleted_when_a_guideline_is_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer wants to get meeting details",
    )

    journey = await journey_store.create_journey(
        title="test_journey",
        description="test_description",
        conditions=[guideline.id],
    )

    response = await async_client.delete(f"/guidelines/{guideline.id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    updated_journey = await journey_store.read_journey(journey.id)
    assert updated_journey.conditions == []


async def test_that_guideline_relationships_can_be_read(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    relationship_store = container[RelationshipStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
    )

    connected_guideline = await guideline_store.create_guideline(
        condition="reply with 'Hello'",
        action="finish with a smile",
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=connected_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )

    response = await async_client.get(f"/guidelines/{guideline.id}")

    assert response.status_code == status.HTTP_200_OK
    relationships = response.json()["relationships"]

    assert len(relationships) == 1
    assert relationships[0]["source_guideline"]["id"] == guideline.id
    assert relationships[0]["target_guideline"]["id"] == connected_guideline.id
    assert relationships[0]["kind"] == "entailment"


async def test_that_guideline_with_relationships_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    relationship_store = container[RelationshipStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
    )

    connected_guideline = await guideline_store.create_guideline(
        condition="reply with 'Hello'",
        action="finish with a smile",
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=connected_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )

    (await async_client.delete(f"/guidelines/{guideline.id}")).raise_for_status()

    response = await async_client.get(f"/guidelines/{guideline.id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_a_guideline_can_be_created_with_description(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/guidelines",
        json={
            "condition": "the customer asks about premium features",
            "action": "explain the premium features available",
            "description": "Premium features are only available to customers with active subscriptions",
            "enabled": True,
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    guideline = response.json()
    assert guideline["condition"] == "the customer asks about premium features"
    assert guideline["action"] == "explain the premium features available"
    assert (
        guideline["description"]
        == "Premium features are only available to customers with active subscriptions"
    )
    assert guideline["enabled"] is True

    guideline_id = guideline["id"]
    item = (await async_client.get(f"/guidelines/{guideline_id}")).raise_for_status().json()

    assert item["guideline"]["id"] == guideline_id
    assert (
        item["guideline"]["description"]
        == "Premium features are only available to customers with active subscriptions"
    )


async def test_that_a_guideline_description_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer asks about refunds",
        action="explain the refund policy",
        metadata={},
    )

    response = await async_client.patch(
        f"/guidelines/{guideline.id}",
        json={
            "description": "Refunds are only available within 30 days of purchase",
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_guideline = response.json()["guideline"]

    assert updated_guideline["id"] == guideline.id
    assert (
        updated_guideline["description"] == "Refunds are only available within 30 days of purchase"
    )


async def test_that_a_guideline_description_can_be_updated_to_none(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer asks about shipping",
        action="explain shipping options",
        metadata={},
    )

    response = await async_client.patch(
        f"/guidelines/{guideline.id}",
        json={
            "description": None,
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_guideline = response.json()["guideline"]

    assert updated_guideline["id"] == guideline.id
    assert updated_guideline["description"] is None


async def test_that_guideline_can_be_created_with_criticality_via_api(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/guidelines",
        json={
            "condition": "Customer reports a critical security issue",
            "action": "Escalate to security team immediately",
            "criticality": "high",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    guideline = response.json()
    assert guideline["condition"] == "Customer reports a critical security issue"
    assert guideline["action"] == "Escalate to security team immediately"
    assert guideline["criticality"] == "high"


async def test_that_guideline_defaults_to_medium_criticality_via_api(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/guidelines",
        json={
            "condition": "Customer asks about product features",
            "action": "Provide detailed feature information",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    guideline = response.json()
    assert guideline["condition"] == "Customer asks about product features"
    assert guideline["action"] == "Provide detailed feature information"
    assert guideline["criticality"] == "medium"


async def test_that_guideline_criticality_can_be_updated_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    # Create a guideline with LOW criticality
    create_response = await async_client.post(
        "/guidelines",
        json={
            "condition": "Customer has a minor question",
            "action": "Provide basic information",
            "criticality": "low",
        },
    )

    assert create_response.status_code == status.HTTP_201_CREATED
    guideline = create_response.json()
    guideline_id = guideline["id"]

    # Update criticality to HIGH
    update_response = await async_client.patch(
        f"/guidelines/{guideline_id}",
        json={
            "criticality": "high",
        },
    )

    assert update_response.status_code == status.HTTP_200_OK
    updated_guideline = update_response.json()["guideline"]

    assert updated_guideline["id"] == guideline_id
    assert updated_guideline["criticality"] == "high"


async def test_that_guideline_composition_mode_can_be_set_and_updated(
    async_client: httpx.AsyncClient,
) -> None:
    # Create guideline with CANNED_COMPOSITED mode
    response = await async_client.post(
        "/guidelines",
        json={
            "condition": "User asks about pricing",
            "action": "Provide pricing information",
            "composition_mode": "composited_canned",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    guideline = response.json()
    guideline_id = guideline["id"]

    # Check that the composition mode is set correctly after creation
    assert guideline["composition_mode"] == "composited_canned"

    # Retrieve guideline and verify composition mode
    response = await async_client.get(f"/guidelines/{guideline_id}")
    assert response.status_code == status.HTTP_200_OK
    guideline = response.json()["guideline"]
    assert guideline["composition_mode"] == "composited_canned"

    # Update guideline to CANNED_STRICT mode
    response = await async_client.patch(
        f"/guidelines/{guideline_id}",
        json={
            "composition_mode": "strict_canned",
        },
    )

    assert response.status_code == status.HTTP_200_OK
    guideline = response.json()["guideline"]

    # Check that the composition mode is updated correctly
    assert guideline["composition_mode"] == "strict_canned"

    # Retrieve guideline again and verify composition mode
    response = await async_client.get(f"/guidelines/{guideline_id}")
    assert response.status_code == status.HTTP_200_OK
    guideline = response.json()["guideline"]
    assert guideline["composition_mode"] == "strict_canned"
