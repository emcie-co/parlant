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

# Import necessary modules and classes
from fastapi import status
import httpx
from lagom import Container
from pytest import raises

from parlant.core.agents import AgentStore
from parlant.core.journeys import JourneyStore
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipEntity,
    RelationshipStore,
)
from parlant.core.rules import RuleStore
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.groups import GroupIds, GroupStore
from parlant.core.common import ItemNotFoundError
from parlant.core.tools import ToolId, ToolContext, ToolResult
from parlant.core.services.tools.plugins import tool

from tests.test_utilities import run_service_server


async def test_that_relationship_can_be_created_between_two_rules(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    source_rule = await rule_store.create_rule(
        condition="source condition",
        action="source action",
    )

    target_rule = await rule_store.create_rule(
        condition="target condition",
        action="target action",
    )

    response = await async_client.post(
        "/relationships",
        json={
            "source_rule": source_rule.id,
            "target_rule": target_rule.id,
            "kind": "entailment",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_rule"]["id"] == source_rule.id
    assert relationship["source_rule"]["condition"] == "source condition"
    assert relationship["source_rule"]["action"] == "source action"

    assert relationship["source_group"] is None

    assert relationship["target_rule"]["id"] == target_rule.id
    assert relationship["target_rule"]["condition"] == "target condition"
    assert relationship["target_rule"]["action"] == "target action"

    assert relationship["target_group"] is None


async def test_that_relationship_can_be_created_between_two_tags(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    group_store = container[GroupStore]

    source_group = await group_store.create_group(
        name="source group",
    )

    target_group = await group_store.create_group(
        name="target group",
    )

    response = await async_client.post(
        "/relationships",
        json={
            "source_group": source_group.id,
            "target_group": target_group.id,
            "kind": "entailment",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_group"]["id"] == source_group.id
    assert relationship["source_group"]["name"] == "source group"

    assert relationship["source_rule"] is None

    assert relationship["target_group"]["id"] == target_group.id
    assert relationship["target_group"]["name"] == "target group"

    assert relationship["target_rule"] is None


async def test_that_relationship_can_be_created_between_a_rule_and_a_tag(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    group_store = container[GroupStore]

    source_rule = await rule_store.create_rule(
        condition="source condition",
        action="source action",
    )

    target_group = await group_store.create_group(
        name="target group",
    )

    response = await async_client.post(
        "/relationships",
        json={
            "source_rule": source_rule.id,
            "target_group": target_group.id,
            "kind": "entailment",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_rule"]["id"] == source_rule.id
    assert relationship["source_rule"]["condition"] == "source condition"
    assert relationship["source_rule"]["action"] == "source action"

    assert relationship["source_group"] is None

    assert relationship["target_group"]["id"] == target_group.id
    assert relationship["target_group"]["name"] == "target group"

    assert relationship["target_rule"] is None


async def test_that_relationships_can_be_listed_by_rule_id(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    relationship_store = container[RelationshipStore]

    rule = await rule_store.create_rule(
        condition="condition",
        action="action",
    )

    group = await group_store.create_group(
        name="group",
    )

    relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=group.id,
            kind=RelationshipEntityKind.GROUP_ALL,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    response = await async_client.get(f"/relationships?rule_id={rule.id}&kind=priority")
    assert response.status_code == status.HTTP_200_OK
    relationships = response.json()
    assert len(relationships) == 1
    assert relationships[0]["id"] == relationship.id
    assert relationships[0]["source_rule"]["id"] == rule.id
    assert relationships[0]["target_group"]["id"] == group.id
    assert relationships[0]["kind"] == "priority"


async def test_that_relationships_can_be_listed_by_group_id(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    relationship_store = container[RelationshipStore]

    rule = await rule_store.create_rule(
        condition="condition",
        action="action",
    )

    group = await group_store.create_group(
        name="group",
    )

    relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=group.id,
            kind=RelationshipEntityKind.GROUP_ALL,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    response = await async_client.get(f"/relationships?group_id={group.id}&kind=priority")
    assert response.status_code == status.HTTP_200_OK
    relationships = response.json()
    assert len(relationships) == 1
    assert relationships[0]["id"] == relationship.id
    assert relationships[0]["source_rule"]["id"] == rule.id
    assert relationships[0]["target_group"]["id"] == group.id


async def test_that_relationship_can_be_read(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    relationship_store = container[RelationshipStore]

    rule = await rule_store.create_rule(
        condition="condition",
        action="action",
    )

    group = await group_store.create_group(
        name="group",
    )

    relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=group.id,
            kind=RelationshipEntityKind.GROUP_ALL,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )

    response = await async_client.get(f"/relationships/{relationship.id}")

    assert response.status_code == status.HTTP_200_OK

    relationship_data = response.json()
    assert relationship_data["id"] == relationship.id
    assert relationship_data["source_rule"]["id"] == rule.id
    assert relationship_data["target_group"]["id"] == group.id
    assert relationship_data["kind"] == "entailment"


async def test_that_entailment_relationship_can_be_created(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    source_rule = await rule_store.create_rule(
        condition="source condition",
        action="source action",
    )

    target_rule = await rule_store.create_rule(
        condition="target condition",
        action="target action",
    )

    response = await async_client.post(
        "/relationships",
        json={
            "source_rule": source_rule.id,
            "target_rule": target_rule.id,
            "kind": "entailment",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_rule"]["id"] == source_rule.id
    assert relationship["target_rule"]["id"] == target_rule.id
    assert relationship["kind"] == "entailment"


async def test_that_entailment_relationship_can_be_deleted(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    rule_store = container[RuleStore]
    relationship_store = container[RelationshipStore]

    source_rule = await rule_store.create_rule(
        condition="source condition",
        action="source action",
    )

    target_rule = await rule_store.create_rule(
        condition="target condition",
        action="target action",
    )

    relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=source_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=target_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )

    response = await async_client.delete(f"/relationships/{relationship.id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await relationship_store.read_relationship(relationship_id=relationship.id)


async def test_that_dependency_relationship_can_be_created(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    source_rule = await rule_store.create_rule(
        condition="source condition",
        action="source action",
    )

    target_rule = await rule_store.create_rule(
        condition="target condition",
        action="target action",
    )

    response = await async_client.post(
        "/relationships",
        json={
            "source_rule": source_rule.id,
            "target_rule": target_rule.id,
            "kind": "dependency",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_rule"]["id"] == source_rule.id
    assert relationship["target_rule"]["id"] == target_rule.id
    assert relationship["kind"] == "dependency"


async def test_that_dependency_relationship_can_be_deleted(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    rule_store = container[RuleStore]
    relationship_store = container[RelationshipStore]

    source_rule = await rule_store.create_rule(
        condition="condition",
        action="action",
    )

    relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=source_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=source_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    response = await async_client.delete(f"/relationships/{relationship.id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await relationship_store.read_relationship(relationship_id=relationship.id)


async def test_that_dependency_any_relationship_can_be_created_with_group_id(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    source_rule = await rule_store.create_rule(
        condition="source condition",
        action="source action",
    )

    target_rule = await rule_store.create_rule(
        condition="target condition",
        action="target action",
    )

    response = await async_client.post(
        "/relationships",
        json={
            "source_rule": source_rule.id,
            "target_rule": target_rule.id,
            "kind": "dependency_any",
            "group_id": "group-1",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_rule"]["id"] == source_rule.id
    assert relationship["target_rule"]["id"] == target_rule.id
    assert relationship["kind"] == "dependency_any"
    assert relationship["group_id"] == "group-1"


async def test_that_dependency_any_relationships_can_be_listed_by_kind(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    relationship_store = container[RelationshipStore]

    g1 = await rule_store.create_rule(condition="A", action="B")
    g2 = await rule_store.create_rule(condition="C", action="D")

    relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id="group-xyz",
    )

    response = await async_client.get("/relationships?kind=dependency_any")
    assert response.status_code == status.HTTP_200_OK

    relationships = response.json()
    assert len(relationships) == 1
    assert relationships[0]["id"] == relationship.id
    assert relationships[0]["kind"] == "dependency_any"
    assert relationships[0]["group_id"] == "group-xyz"


async def test_that_priority_relationship_can_be_created(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    source_rule = await rule_store.create_rule(
        condition="source condition",
        action="source action",
    )

    target_rule = await rule_store.create_rule(
        condition="target condition",
        action="target action",
    )

    response = await async_client.post(
        "/relationships",
        json={
            "source_rule": source_rule.id,
            "target_rule": target_rule.id,
            "kind": "priority",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_rule"]["id"] == source_rule.id
    assert relationship["target_rule"]["id"] == target_rule.id
    assert relationship["kind"] == "priority"


async def test_that_priority_relationship_can_be_deleted(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    rule_store = container[RuleStore]
    relationship_store = container[RelationshipStore]

    source_rule = await rule_store.create_rule(
        condition="source condition",
        action="source action",
    )

    target_rule = await rule_store.create_rule(
        condition="target condition",
        action="target action",
    )

    relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=source_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=target_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    response = await async_client.delete(f"/relationships/{relationship.id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await relationship_store.read_relationship(relationship_id=relationship.id)


async def test_that_disambiguation_relationship_can_be_created(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    source_rule = await rule_store.create_rule(
        condition="source condition",
        action="source action",
    )

    target_rule = await rule_store.create_rule(
        condition="target condition",
        action="target action",
    )

    response = await async_client.post(
        "/relationships",
        json={
            "source_rule": source_rule.id,
            "target_rule": target_rule.id,
            "kind": "disambiguation",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_rule"]["id"] == source_rule.id
    assert relationship["target_rule"]["id"] == target_rule.id
    assert relationship["kind"] == "disambiguation"


async def test_that_disambiguation_relationship_can_be_deleted(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    rule_store = container[RuleStore]
    relationship_store = container[RelationshipStore]

    source_rule = await rule_store.create_rule(
        condition="source condition",
        action="source action",
    )

    target_rule = await rule_store.create_rule(
        condition="target condition",
        action="target action",
    )

    relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=source_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=target_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.DISAMBIGUATION,
    )

    response = await async_client.delete(f"/relationships/{relationship.id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await relationship_store.read_relationship(relationship_id=relationship.id)


async def test_that_reevaluation_relationship_can_be_created(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]

    source_rule = await rule_store.create_rule(
        condition="source condition",
        action="source action",
    )

    target_rule = await rule_store.create_rule(
        condition="target condition",
        action="target action",
    )

    response = await async_client.post(
        "/relationships",
        json={
            "source_rule": source_rule.id,
            "target_rule": target_rule.id,
            "kind": "reevaluation",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_rule"]["id"] == source_rule.id
    assert relationship["target_rule"]["id"] == target_rule.id
    assert relationship["kind"] == "reevaluation"


async def test_that_reevaluation_relationship_can_be_deleted(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    rule_store = container[RuleStore]
    relationship_store = container[RelationshipStore]

    source_rule = await rule_store.create_rule(
        condition="source condition",
        action="source action",
    )

    target_rule = await rule_store.create_rule(
        condition="target condition",
        action="target action",
    )

    relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=source_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=target_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.REEVALUATION,
    )

    response = await async_client.delete(f"/relationships/{relationship.id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await relationship_store.read_relationship(relationship_id=relationship.id)


async def test_that_overlap_relationship_can_be_created(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    service_registry = container[ServiceRegistry]

    @tool
    def first_tool(context: ToolContext, arg_1: int, arg_2: int) -> ToolResult:
        return ToolResult(arg_1 + arg_2)

    @tool
    def second_tool(context: ToolContext, message: str) -> ToolResult:
        return ToolResult(f"Echo: {message}")

    async with run_service_server([first_tool, second_tool]) as server:
        await service_registry.update_tool_service(
            name="test_service",
            kind="sdk",
            url=server.url,
        )

        first_tool_id = ToolId(service_name="test_service", tool_name="first_tool")
        second_tool_id = ToolId(service_name="test_service", tool_name="second_tool")

        response = await async_client.post(
            "/relationships",
            json={
                "source_tool": {
                    "service_name": first_tool_id.service_name,
                    "tool_name": first_tool_id.tool_name,
                },
                "target_tool": {
                    "service_name": second_tool_id.service_name,
                    "tool_name": second_tool_id.tool_name,
                },
                "kind": "overlap",
            },
        )

        assert response.status_code == status.HTTP_201_CREATED

        relationship = response.json()
        assert relationship["source_tool"]["name"] == "first_tool"
        assert relationship["target_tool"]["name"] == "second_tool"
        assert relationship["kind"] == "overlap"


async def test_that_overlap_relationship_can_be_deleted(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    relationship_store = container[RelationshipStore]

    first_tool_id = ToolId(service_name="test_service", tool_name="first_tool")
    second_tool_id = ToolId(service_name="test_service", tool_name="second_tool")

    relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=first_tool_id,
            kind=RelationshipEntityKind.TOOL,
        ),
        target=RelationshipEntity(
            id=second_tool_id,
            kind=RelationshipEntityKind.TOOL,
        ),
        kind=RelationshipKind.OVERLAP,
    )

    response = await async_client.delete(f"/relationships/{relationship.id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await relationship_store.read_relationship(relationship_id=relationship.id)


async def test_that_all_relationships_can_be_listed(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    rule_store = container[RuleStore]
    relationship_store = container[RelationshipStore]

    g1 = await rule_store.create_rule(condition="A", action="B")
    g2 = await rule_store.create_rule(condition="C", action="D")
    g3 = await rule_store.create_rule(condition="E", action="F")

    r1 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.PRIORITY,
    )

    r2 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    r3 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DISAMBIGUATION,
    )

    r4 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.REEVALUATION,
    )

    response = await async_client.get("/relationships")
    assert response.status_code == status.HTTP_200_OK

    relationships = response.json()

    returned_ids = {rel["id"] for rel in relationships}

    assert r1.id in returned_ids
    assert r2.id in returned_ids
    assert r3.id in returned_ids
    assert r4.id in returned_ids


async def test_that_relationships_can_be_listed_by_kind_only(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    rule_store = container[RuleStore]
    relationship_store = container[RelationshipStore]

    g1 = await rule_store.create_rule(condition="AA", action="BB")
    g2 = await rule_store.create_rule(condition="CC", action="DD")

    priority_relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.PRIORITY,
    )

    _ = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.ENTAILMENT,
    )

    response = await async_client.get("/relationships?kind=priority")
    assert response.status_code == status.HTTP_200_OK

    relationships = response.json()

    assert len(relationships) == 1
    assert relationships[0]["id"] == priority_relationship.id
    assert relationships[0]["kind"] == "priority"


async def test_that_relationships_can_be_listed_by_rule_id_without_kind_filter_via_api(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    rule_store = container[RuleStore]
    relationship_store = container[RelationshipStore]

    g1 = await rule_store.create_rule(condition="X", action="Y")
    g2 = await rule_store.create_rule(condition="Y", action="Z")
    g3 = await rule_store.create_rule(condition="Z", action="W")

    rel1 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.ENTAILMENT,
    )

    rel2 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    response = await async_client.get(f"/relationships?rule_id={g1.id}")
    assert response.status_code == status.HTTP_200_OK

    relationships = response.json()

    returned_ids = {rel["id"] for rel in relationships}

    assert rel1.id in returned_ids
    assert rel2.id in returned_ids


async def test_that_relationships_can_be_listed_by_tool_id(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    service_registry = container[ServiceRegistry]
    relationship_store = container[RelationshipStore]

    @tool
    def first_tool(context: ToolContext, arg_1: int, arg_2: int) -> ToolResult:
        return ToolResult(arg_1 + arg_2)

    @tool
    def second_tool(context: ToolContext, message: str) -> ToolResult:
        return ToolResult(f"Echo: {message}")

    @tool
    def third_tool(context: ToolContext, message: str) -> ToolResult:
        return ToolResult(f"Echo: {message}")

    async with run_service_server([first_tool, second_tool, third_tool]) as server:
        await service_registry.update_tool_service(
            name="test_service",
            kind="sdk",
            url=server.url,
        )

        first_tool_id = ToolId(service_name="test_service", tool_name="first_tool")
        second_tool_id = ToolId(service_name="test_service", tool_name="second_tool")
        third_tool_id = ToolId(service_name="test_service", tool_name="third_tool")

        rel1 = await relationship_store.create_relationship(
            source=RelationshipEntity(id=first_tool_id, kind=RelationshipEntityKind.TOOL),
            target=RelationshipEntity(id=second_tool_id, kind=RelationshipEntityKind.TOOL),
            kind=RelationshipKind.OVERLAP,
        )

        rel2 = await relationship_store.create_relationship(
            source=RelationshipEntity(id=first_tool_id, kind=RelationshipEntityKind.TOOL),
            target=RelationshipEntity(id=third_tool_id, kind=RelationshipEntityKind.TOOL),
            kind=RelationshipKind.OVERLAP,
        )

        response = await async_client.get(
            f"/relationships?tool_id={first_tool_id.service_name}:{first_tool_id.tool_name}"
        )
        assert response.status_code == status.HTTP_200_OK

        relationships = response.json()

        returned_ids = {rel["id"] for rel in relationships}

        assert rel1.id in returned_ids
        assert rel2.id in returned_ids


async def test_that_relationships_of_rule_and_a_journey_can_be_listed(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    rule_store = container[RuleStore]
    journey_store = container[JourneyStore]
    relationship_store = container[RelationshipStore]

    g1 = await rule_store.create_rule(condition="A", action="B")

    j1 = await journey_store.create_journey(
        title="Journey 1",
        description="Description of Journey 1",
        triggers=[],
    )

    r1 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(
            id=GroupIds.for_journey_id(j1.id), kind=RelationshipEntityKind.GROUP_ALL
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    response = await async_client.get(f"/relationships?rule_id={g1.id}")
    assert response.status_code == status.HTTP_200_OK

    relationships = response.json()

    returned_ids = {rel["id"] for rel in relationships}

    assert r1.id in returned_ids


async def test_that_relationships_of_a_journey_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    journey_store = container[JourneyStore]
    relationship_store = container[RelationshipStore]

    g1 = await rule_store.create_rule(condition="A", action="B")

    j1 = await journey_store.create_journey(
        title="Journey 1",
        description="Description of Journey 1",
        triggers=[],
    )

    r1 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(
            id=GroupIds.for_journey_id(j1.id), kind=RelationshipEntityKind.GROUP_ALL
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    response = await async_client.get(f"/relationships?group_id=journey:{j1.id}")
    assert response.status_code == status.HTTP_200_OK

    relationships = response.json()

    returned_ids = {rel["id"] for rel in relationships}

    assert r1.id in returned_ids


async def test_that_relationships_of_rule_and_an_agent_can_be_listed(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    rule_store = container[RuleStore]
    agent_store = container[AgentStore]
    relationship_store = container[RelationshipStore]

    g1 = await rule_store.create_rule(condition="A", action="B")

    a1 = await agent_store.create_agent(name="Agent 1", description="Description of Agent 1")

    r1 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(
            id=GroupIds.for_agent_id(a1.id), kind=RelationshipEntityKind.GROUP_ALL
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    response = await async_client.get(f"/relationships?rule_id={g1.id}")
    assert response.status_code == status.HTTP_200_OK

    relationships = response.json()

    returned_ids = {rel["id"] for rel in relationships}

    assert r1.id in returned_ids


async def test_that_relationships_of_an_agent_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    agent_store = container[AgentStore]
    relationship_store = container[RelationshipStore]

    g1 = await rule_store.create_rule(condition="A", action="B")

    a1 = await agent_store.create_agent(name="Agent 1", description="Description of Agent 1")

    r1 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(
            id=GroupIds.for_agent_id(a1.id), kind=RelationshipEntityKind.GROUP_ALL
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    response = await async_client.get(f"/relationships?group_id=agent:{a1.id}")
    assert response.status_code == status.HTTP_200_OK

    relationships = response.json()

    returned_ids = {rel["id"] for rel in relationships}

    assert r1.id in returned_ids
