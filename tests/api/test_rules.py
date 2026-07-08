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
from parlant.core.rules import Rule, RuleContent, RuleStore
from parlant.core.groups import GroupIds, GroupId, GroupStore
from parlant.core.tools import LocalToolService, ToolOverlap


async def create_rules_and_create_relationships_between_them(
    container: Container,
    agent_id: AgentId,
    rule_contents: list[RuleContent],
) -> list[Rule]:
    rules = [
        await container[RuleStore].create_rule(
            condition=gc.condition,
            action=gc.action,
        )
        for gc in rule_contents
    ]

    for rule in rules:
        _ = await container[RuleStore].upsert_group(
            rule_id=rule.id,
            group_id=GroupIds.for_agent_id(agent_id),
        )

    for source, target in zip(rules, rules[1:]):
        await container[RelationshipStore].create_relationship(
            source=RelationshipEntity(
                id=source.id,
                kind=RelationshipEntityKind.RULE,
            ),
            target=RelationshipEntity(
                id=target.id,
                kind=RelationshipEntityKind.RULE,
            ),
            kind=RelationshipKind.ENTAILMENT,
        )

    return rules


async def test_that_a_rule_can_be_created(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/rules",
        json={
            "condition": "the customer asks about pricing",
            "action": "provide current pricing information",
            "enabled": True,
            "metadata": {"key1": "value1", "key2": "value2"},
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    rule = response.json()
    assert rule["condition"] == "the customer asks about pricing"
    assert rule["action"] == "provide current pricing information"
    assert rule["enabled"] is True
    assert rule["groups"] == []
    assert rule["metadata"] == {"key1": "value1", "key2": "value2"}
    assert "modified_utc" in rule


async def test_that_a_rule_can_be_created_with_a_title(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/rules",
        json={
            "condition": "the customer asks about pricing",
            "action": "provide current pricing information",
            "title": "Pricing inquiries",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    rule = response.json()
    assert rule["title"] == "Pricing inquiries"


async def test_that_a_rule_title_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    rule = await rule_store.create_rule(
        condition="the customer asks about the weather",
        action="provide the current weather update",
        title="Old title",
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={"title": "Weather inquiries"},
    )

    assert response.status_code == status.HTTP_200_OK
    updated_rule = response.json()["rule"]

    assert updated_rule["id"] == rule.id
    assert updated_rule["title"] == "Weather inquiries"


async def test_that_a_rule_can_be_created_without_an_action(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/rules",
        json={"condition": "the customer asks about pricing"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    rule = response.json()
    assert rule["condition"] == "the customer asks about pricing"
    assert rule["action"] is None


async def test_that_a_rule_can_be_created_with_custom_id(
    async_client: httpx.AsyncClient,
) -> None:
    """Test that a rule can be created with a custom ID."""
    custom_id = "custom-rule-id-456"

    response = await async_client.post(
        "/rules",
        json={
            "id": custom_id,
            "condition": "the customer mentions a custom requirement",
            "action": "provide personalized assistance",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    rule = response.json()

    # Verify that the custom ID was used
    assert rule["id"] == custom_id
    assert rule["condition"] == "the customer mentions a custom requirement"
    assert rule["action"] == "provide personalized assistance"
    assert rule["enabled"] is True
    assert rule["groups"] == []
    assert rule["metadata"] == {}


async def test_that_creating_rule_with_duplicate_id_fails(
    async_client: httpx.AsyncClient,
) -> None:
    """Test that creating a rule with a duplicate ID fails appropriately."""
    custom_id = "duplicate-rule-id"

    # Create first rule
    response1 = await async_client.post(
        "/rules",
        json={
            "id": custom_id,
            "condition": "first condition",
            "action": "first action",
        },
    )
    assert response1.status_code == status.HTTP_201_CREATED

    # Try to create second rule with same ID
    response2 = await async_client.post(
        "/rules",
        json={
            "id": custom_id,
            "condition": "second condition",
            "action": "second action",
        },
    )

    # Should fail due to duplicate ID
    assert response2.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert "already exists" in response2.text


async def test_that_a_rule_can_be_created_with_tags(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    group_store = container[GroupStore]
    agent_store = container[AgentStore]
    journey_store = container[JourneyStore]

    agent = await agent_store.create_agent("Test Agent")
    agent_group = GroupIds.for_agent_id(agent.id)

    journey = await journey_store.create_journey(
        title="Customer Support Journey",
        description="A journey for customer support interactions.",
        triggers=[],
    )
    journey_group = GroupIds.for_journey_id(journey.id)

    tag_1 = await group_store.create_group(name="pricing")
    tag_2 = await group_store.create_group(name="sales")

    response = await async_client.post(
        "/rules",
        json={
            "condition": "the customer asks about pricing",
            "action": "provide current pricing information",
            "groups": [
                tag_1.id,
                tag_1.id,
                tag_2.id,
                agent_group,
                journey_group,
            ],
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    rule_dto = (await async_client.get(f"/rules/{response.json()['id']}")).raise_for_status().json()

    assert rule_dto["rule"]["condition"] == "the customer asks about pricing"
    assert rule_dto["rule"]["action"] == "provide current pricing information"

    assert len(rule_dto["rule"]["groups"]) == 4
    assert set(rule_dto["rule"]["groups"]) == {tag_1.id, tag_2.id, agent_group, journey_group}


async def test_that_rules_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    first_rule = [
        await rule_store.create_rule(
            condition=f"condition {i}",
            action=f"action {i}",
        )
        for i in range(2)
    ]
    second_rule = await rule_store.create_rule(
        condition="condition 2",
        action="action 2",
    )

    response_rules = (await async_client.get("/rules")).raise_for_status().json()

    assert len(response_rules) >= 2
    assert any(first_rule[0].id == g["id"] for g in response_rules)
    assert any(first_rule[1].id == g["id"] for g in response_rules)
    assert any(second_rule.id == g["id"] for g in response_rules)


async def test_that_rules_can_be_listed_by_tag(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    first_rule = await rule_store.create_rule(
        condition="condition 1",
        action="action 1",
    )

    second_rule = await rule_store.create_rule(
        condition="condition 2",
        action="action 2",
    )

    await rule_store.upsert_group(
        rule_id=first_rule.id,
        group_id=GroupId("tag_1"),
    )

    await rule_store.upsert_group(
        rule_id=second_rule.id,
        group_id=GroupId("tag_2"),
    )

    response_rules = (await async_client.get("/rules?group_id=tag_1")).raise_for_status().json()

    assert len(response_rules) == 1
    assert response_rules[0]["id"] == first_rule.id

    response_rules = (await async_client.get("/rules?group_id=tag_2")).raise_for_status().json()


async def test_that_a_rule_can_be_read(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    rule = await rule_store.create_rule(
        condition="the customer asks about the weather",
        action="provide the current weather update",
        metadata={"key1": "value1", "key2": "value2"},
    )

    item = (await async_client.get(f"/rules/{rule.id}")).raise_for_status().json()

    assert item["rule"]["id"] == rule.id
    assert item["rule"]["condition"] == "the customer asks about the weather"
    assert item["rule"]["action"] == "provide the current weather update"
    assert item["rule"]["metadata"] == {"key1": "value1", "key2": "value2"}
    assert len(item["relationships"]) == 0
    assert len(item["tool_associations"]) == 0


async def test_that_a_rule_condition_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    rule = await rule_store.create_rule(
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={
            "condition": "the customer inquires about weather",
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_rule = response.json()["rule"]

    assert updated_rule["id"] == rule.id
    assert updated_rule["condition"] == "the customer inquires about weather"
    assert updated_rule["action"] == rule.content.action
    assert updated_rule["modified_utc"] != rule.creation_utc.isoformat()


async def test_that_a_rule_action_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    rule = await rule_store.create_rule(
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={
            "action": "give current weather information",
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_rule = response.json()["rule"]

    assert updated_rule["id"] == rule.id
    assert updated_rule["condition"] == rule.content.condition
    assert updated_rule["action"] == "give current weather information"


async def test_that_a_rule_can_be_disabled(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    rule = await rule_store.create_rule(
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={
            "enabled": False,
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_rule = response.json()["rule"]

    assert updated_rule["id"] == rule.id
    assert updated_rule["enabled"] is False


async def test_that_a_tag_can_be_added_to_rule(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    group_store = container[GroupStore]

    group = await group_store.create_group("test_group")

    rule = await rule_store.create_rule(
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={
            "groups": {
                "add": [group.id],
            },
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_rule = response.json()["rule"]

    assert updated_rule["id"] == rule.id
    assert group.id in updated_rule["groups"]


async def test_that_a_tag_can_be_removed_from_rule(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    rule = await rule_store.create_rule(
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    # First add a group
    await rule_store.upsert_group(
        rule_id=rule.id,
        group_id=GroupId("test_group"),
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={
            "groups": {
                "remove": ["test_group"],
            },
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_rule = response.json()["rule"]

    assert updated_rule["id"] == rule.id
    assert "test_group" not in updated_rule["groups"]


async def test_that_an_agent_group_can_be_added_to_rule(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    agent_store = container[AgentStore]

    agent = await agent_store.create_agent("test_agent")
    agent_group = GroupIds.for_agent_id(agent.id)

    rule = await rule_store.create_rule(
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={
            "groups": {
                "add": [agent_group],
            },
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_rule = response.json()["rule"]

    assert updated_rule["id"] == rule.id
    assert agent_group in updated_rule["groups"]


async def test_that_a_journey_group_can_be_added_to_rule(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    journey_store = container[JourneyStore]

    journey = await journey_store.create_journey(
        title="test_journey",
        description="test_description",
        triggers=[],
    )
    journey_group = GroupIds.for_journey_id(journey.id)

    rule = await rule_store.create_rule(
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={
            "groups": {
                "add": [journey_group],
            },
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_rule = response.json()["rule"]

    assert updated_rule["id"] == rule.id
    assert journey_group in updated_rule["groups"]


async def test_that_a_rule_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    rule = await rule_store.create_rule(
        condition="the customer wants to unsubscribe",
        action="ask for confirmation",
    )

    (await async_client.delete(f"/rules/{rule.id}")).raise_for_status()

    response = await async_client.get(f"/rules/{rule.id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_a_tool_association_can_be_added_to_a_rule(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    local_tool_service = container[LocalToolService]

    await local_tool_service.create_tool(
        name="fetch_event_data",
        module_path="some.module",
        description="",
        parameters={},
        required=[],
        overlap=ToolOverlap.NONE,
    )

    rule = await rule_store.create_rule(
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
        f"/rules/{rule.id}",
        json=request_data,
    )

    assert response.status_code == status.HTTP_200_OK

    tool_associations = response.json()["tool_associations"]

    assert any(
        a["rule_id"] == rule.id
        and a["tool_id"]["service_name"] == service_name
        and a["tool_id"]["tool_name"] == tool_name
        for a in tool_associations
    )


async def test_that_a_tag_can_be_added_to_a_rule(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    group_store = container[GroupStore]

    group = await group_store.create_group("test_group")
    rule = await rule_store.create_rule(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={"groups": {"add": [group.id]}},
    )

    assert response.status_code == status.HTTP_200_OK
    updated_rule = response.json()["rule"]

    assert updated_rule["id"] == rule.id
    assert group.id in updated_rule["groups"]


async def test_that_a_tag_can_be_removed_from_a_rule(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    group_store = container[GroupStore]

    group = await group_store.create_group("test_group")

    rule = await rule_store.create_rule(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
    )

    await rule_store.upsert_group(
        rule_id=rule.id,
        group_id=group.id,
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={"groups": {"remove": [group.id]}},
    )

    assert response.status_code == status.HTTP_200_OK
    updated_rule = response.json()["rule"]

    assert updated_rule["id"] == rule.id
    assert updated_rule["groups"] == []


async def test_that_adding_nonexistent_agent_group_to_rule_returns_404(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    rule = await rule_store.create_rule(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={"groups": {"add": ["agent-id:nonexistent_agent"]}},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_adding_nonexistent_group_to_rule_returns_404(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    rule = await rule_store.create_rule(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={"groups": {"add": ["nonexistent_group"]}},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_metadata_can_be_updated_for_a_rule(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    rule = await rule_store.create_rule(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
        metadata={"key3": "value2"},
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
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
    updated_rule = response.json()["rule"]

    assert updated_rule["id"] == rule.id
    assert updated_rule["metadata"] == {"key1": "value1", "key2": "value2"}


async def test_that_condition_association_is_deleted_when_a_rule_is_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    journey_store = container[JourneyStore]

    rule = await rule_store.create_rule(
        condition="the customer wants to get meeting details",
    )

    journey = await journey_store.create_journey(
        title="test_journey",
        description="test_description",
        triggers=[rule.id],
    )

    response = await async_client.delete(f"/rules/{rule.id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    updated_journey = await journey_store.read_journey(journey.id)
    assert updated_journey.triggers == []


async def test_that_rule_relationships_can_be_read(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    relationship_store = container[RelationshipStore]

    rule = await rule_store.create_rule(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
    )

    connected_rule = await rule_store.create_rule(
        condition="reply with 'Hello'",
        action="finish with a smile",
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=connected_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )

    response = await async_client.get(f"/rules/{rule.id}")

    assert response.status_code == status.HTTP_200_OK
    relationships = response.json()["relationships"]

    assert len(relationships) == 1
    assert relationships[0]["source_rule"]["id"] == rule.id
    assert relationships[0]["target_rule"]["id"] == connected_rule.id
    assert relationships[0]["kind"] == "entailment"


async def test_that_rule_dependency_any_relationships_can_be_read(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    relationship_store = container[RelationshipStore]

    rule = await rule_store.create_rule(
        condition="source condition",
        action="source action",
    )

    target_rule = await rule_store.create_rule(
        condition="target condition",
        action="target action",
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=target_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id="group-abc",
    )

    response = await async_client.get(f"/rules/{rule.id}")

    assert response.status_code == status.HTTP_200_OK
    relationships = response.json()["relationships"]

    assert len(relationships) == 1
    assert relationships[0]["source_rule"]["id"] == rule.id
    assert relationships[0]["target_rule"]["id"] == target_rule.id
    assert relationships[0]["kind"] == "dependency_any"
    assert relationships[0]["group_id"] == "group-abc"


async def test_that_rule_with_relationships_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    relationship_store = container[RelationshipStore]

    rule = await rule_store.create_rule(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
    )

    connected_rule = await rule_store.create_rule(
        condition="reply with 'Hello'",
        action="finish with a smile",
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=connected_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )

    (await async_client.delete(f"/rules/{rule.id}")).raise_for_status()

    response = await async_client.get(f"/rules/{rule.id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_a_rule_can_be_created_with_description(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/rules",
        json={
            "condition": "the customer asks about premium features",
            "action": "explain the premium features available",
            "description": "Premium features are only available to customers with active subscriptions",
            "enabled": False,
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    rule = response.json()
    assert rule["condition"] == "the customer asks about premium features"
    assert rule["action"] == "explain the premium features available"
    assert (
        rule["description"]
        == "Premium features are only available to customers with active subscriptions"
    )
    assert rule["enabled"] is False

    rule_id = rule["id"]
    item = (await async_client.get(f"/rules/{rule_id}")).raise_for_status().json()

    assert item["rule"]["id"] == rule_id
    assert (
        item["rule"]["description"]
        == "Premium features are only available to customers with active subscriptions"
    )


async def test_that_a_rule_description_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    rule = await rule_store.create_rule(
        condition="the customer asks about refunds",
        action="explain the refund policy",
        metadata={},
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={
            "description": "Refunds are only available within 30 days of purchase",
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_rule = response.json()["rule"]

    assert updated_rule["id"] == rule.id
    assert updated_rule["description"] == "Refunds are only available within 30 days of purchase"


async def test_that_a_rule_description_can_be_updated_to_none(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    rule = await rule_store.create_rule(
        condition="the customer asks about shipping",
        action="explain shipping options",
        metadata={},
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={
            "description": None,
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_rule = response.json()["rule"]

    assert updated_rule["id"] == rule.id
    assert updated_rule["description"] is None


async def test_that_rule_can_be_created_with_criticality_via_api(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/rules",
        json={
            "condition": "Customer reports a critical security issue",
            "action": "Escalate to security team immediately",
            "criticality": "high",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    rule = response.json()
    assert rule["condition"] == "Customer reports a critical security issue"
    assert rule["action"] == "Escalate to security team immediately"
    assert rule["criticality"] == "high"


async def test_that_rule_defaults_to_medium_criticality_via_api(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/rules",
        json={
            "condition": "Customer asks about product features",
            "action": "Provide detailed feature information",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    rule = response.json()
    assert rule["condition"] == "Customer asks about product features"
    assert rule["action"] == "Provide detailed feature information"
    assert rule["criticality"] == "medium"


async def test_that_rule_criticality_can_be_updated_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    # Create a rule with LOW criticality
    create_response = await async_client.post(
        "/rules",
        json={
            "condition": "Customer has a minor question",
            "action": "Provide basic information",
            "criticality": "low",
        },
    )

    assert create_response.status_code == status.HTTP_201_CREATED
    rule = create_response.json()
    rule_id = rule["id"]

    # Update criticality to HIGH
    update_response = await async_client.patch(
        f"/rules/{rule_id}",
        json={
            "criticality": "high",
        },
    )

    assert update_response.status_code == status.HTTP_200_OK
    updated_rule = update_response.json()["rule"]

    assert updated_rule["id"] == rule_id
    assert updated_rule["criticality"] == "high"


async def test_that_rule_composition_mode_can_be_set_and_updated(
    async_client: httpx.AsyncClient,
) -> None:
    # Create rule with CANNED_COMPOSITED mode
    response = await async_client.post(
        "/rules",
        json={
            "condition": "User asks about pricing",
            "action": "Provide pricing information",
            "composition_mode": "composited_canned",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    rule = response.json()
    rule_id = rule["id"]

    # Check that the composition mode is set correctly after creation
    assert rule["composition_mode"] == "composited_canned"

    # Retrieve rule and verify composition mode
    response = await async_client.get(f"/rules/{rule_id}")
    assert response.status_code == status.HTTP_200_OK
    rule = response.json()["rule"]
    assert rule["composition_mode"] == "composited_canned"

    # Update rule to CANNED_STRICT mode
    response = await async_client.patch(
        f"/rules/{rule_id}",
        json={
            "composition_mode": "strict_canned",
        },
    )

    assert response.status_code == status.HTTP_200_OK
    rule = response.json()["rule"]

    # Check that the composition mode is updated correctly
    assert rule["composition_mode"] == "strict_canned"

    # Retrieve rule again and verify composition mode
    response = await async_client.get(f"/rules/{rule_id}")
    assert response.status_code == status.HTTP_200_OK
    rule = response.json()["rule"]
    assert rule["composition_mode"] == "strict_canned"


async def test_that_rule_effort_can_be_set_and_updated(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/rules",
        json={
            "condition": "User asks about regulated investing",
            "action": "Apply extra care before responding",
            "effort": "high",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    rule = response.json()
    rule_id = rule["id"]
    assert rule["effort"] == "high"

    response = await async_client.get(f"/rules/{rule_id}")
    assert response.status_code == status.HTTP_200_OK
    rule = response.json()["rule"]
    assert rule["effort"] == "high"

    response = await async_client.patch(
        f"/rules/{rule_id}",
        json={
            "effort": "max",
        },
    )

    assert response.status_code == status.HTTP_200_OK
    rule = response.json()["rule"]
    assert rule["effort"] == "max"

    response = await async_client.get(f"/rules/{rule_id}")
    assert response.status_code == status.HTTP_200_OK
    rule = response.json()["rule"]
    assert rule["effort"] == "max"


###############################################################################
## Labels Tests
###############################################################################


async def test_that_a_rule_can_be_created_with_labels(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/rules",
        json={
            "condition": "the customer asks about pricing",
            "action": "provide current pricing information",
            "labels": ["premium", "sales"],
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    rule = response.json()
    assert rule["condition"] == "the customer asks about pricing"
    assert rule["action"] == "provide current pricing information"
    assert set(rule["labels"]) == {"premium", "sales"}


async def test_that_a_rule_is_created_with_empty_labels_by_default(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/rules",
        json={
            "condition": "the customer asks about something",
            "action": "help them out",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    rule = response.json()
    assert rule["labels"] == []


async def test_that_labels_can_be_added_to_a_rule(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    rule = await rule_store.create_rule(
        condition="the customer wants help",
        action="help them",
        labels={"initial"},
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={"labels": {"upsert": ["new_label", "another_label"]}},
    )

    assert response.status_code == status.HTTP_200_OK
    updated_rule = response.json()["rule"]

    assert set(updated_rule["labels"]) == {"initial", "new_label", "another_label"}


async def test_that_labels_can_be_removed_from_a_rule(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    rule = await rule_store.create_rule(
        condition="the customer wants help",
        action="help them",
        labels={"label1", "label2", "label3"},
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={"labels": {"remove": ["label2"]}},
    )

    assert response.status_code == status.HTTP_200_OK
    updated_rule = response.json()["rule"]

    assert set(updated_rule["labels"]) == {"label1", "label3"}


async def test_that_labels_can_be_upserted_and_removed_in_same_operation(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    rule = await rule_store.create_rule(
        condition="test condition",
        action="test action",
        labels={"keep", "remove_me"},
    )

    response = await async_client.patch(
        f"/rules/{rule.id}",
        json={
            "labels": {
                "upsert": ["new_label"],
                "remove": ["remove_me"],
            }
        },
    )

    assert response.status_code == status.HTTP_200_OK
    updated_rule = response.json()["rule"]

    assert set(updated_rule["labels"]) == {"keep", "new_label"}
