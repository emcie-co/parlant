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

# Import necessary modules and classes
from fastapi import status
import httpx
from lagom import Container
from pytest import raises

from parlant.core.relationships import (
    EntityType,
    GuidelineRelationshipKind,
    ToolRelationshipKind,
    RelationshipEntity,
    RelationshipStore,
)
from parlant.core.guidelines import GuidelineStore
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tags import TagStore
from parlant.core.common import ItemNotFoundError
from parlant.core.tools import ToolId, ToolContext, ToolResult
from parlant.core.services.tools.plugins import tool

from tests.test_utilities import run_service_server


async def test_that_relationship_can_be_created_between_two_guidelines(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    source_guideline = await guideline_store.create_guideline(
        condition="source condition",
        action="source action",
    )

    target_guideline = await guideline_store.create_guideline(
        condition="target condition",
        action="target action",
    )

    response = await async_client.post(
        "/relationships",
        json={
            "source_guideline": source_guideline.id,
            "target_guideline": target_guideline.id,
            "kind": "entailment",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_guideline"]["id"] == source_guideline.id
    assert relationship["source_guideline"]["condition"] == "source condition"
    assert relationship["source_guideline"]["action"] == "source action"

    assert relationship["source_tag"] is None

    assert relationship["target_guideline"]["id"] == target_guideline.id
    assert relationship["target_guideline"]["condition"] == "target condition"
    assert relationship["target_guideline"]["action"] == "target action"

    assert relationship["target_tag"] is None


async def test_that_relationship_can_be_created_between_two_tags(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    tag_store = container[TagStore]

    source_tag = await tag_store.create_tag(
        name="source tag",
    )

    target_tag = await tag_store.create_tag(
        name="target tag",
    )

    response = await async_client.post(
        "/relationships",
        json={
            "source_tag": source_tag.id,
            "target_tag": target_tag.id,
            "kind": "entailment",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_tag"]["id"] == source_tag.id
    assert relationship["source_tag"]["name"] == "source tag"

    assert relationship["source_guideline"] is None

    assert relationship["target_tag"]["id"] == target_tag.id
    assert relationship["target_tag"]["name"] == "target tag"

    assert relationship["target_guideline"] is None


async def test_that_relationship_can_be_created_between_a_guideline_and_a_tag(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]

    source_guideline = await guideline_store.create_guideline(
        condition="source condition",
        action="source action",
    )

    target_tag = await tag_store.create_tag(
        name="target tag",
    )

    response = await async_client.post(
        "/relationships",
        json={
            "source_guideline": source_guideline.id,
            "target_tag": target_tag.id,
            "kind": "entailment",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_guideline"]["id"] == source_guideline.id
    assert relationship["source_guideline"]["condition"] == "source condition"
    assert relationship["source_guideline"]["action"] == "source action"

    assert relationship["source_tag"] is None

    assert relationship["target_tag"]["id"] == target_tag.id
    assert relationship["target_tag"]["name"] == "target tag"

    assert relationship["target_guideline"] is None


async def test_that_relationships_can_be_listed_by_guideline_id(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    relationship_store = container[RelationshipStore]

    guideline = await guideline_store.create_guideline(
        condition="condition",
        action="action",
    )

    tag = await tag_store.create_tag(
        name="tag",
    )

    relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=guideline.id,
            type=EntityType.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=tag.id,
            type=EntityType.TAG,
        ),
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    response = await async_client.get(f"/relationships?guideline_id={guideline.id}&kind=priority")
    assert response.status_code == status.HTTP_200_OK
    relationships = response.json()
    assert len(relationships) == 1
    assert relationships[0]["id"] == relationship.id
    assert relationships[0]["source_guideline"]["id"] == guideline.id
    assert relationships[0]["target_tag"]["id"] == tag.id
    assert relationships[0]["kind"] == "priority"


async def test_that_relationships_can_be_listed_by_tag_id(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    relationship_store = container[RelationshipStore]

    guideline = await guideline_store.create_guideline(
        condition="condition",
        action="action",
    )

    tag = await tag_store.create_tag(
        name="tag",
    )

    relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=guideline.id,
            type=EntityType.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=tag.id,
            type=EntityType.TAG,
        ),
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    response = await async_client.get(f"/relationships?tag_id={tag.id}&kind=priority")
    assert response.status_code == status.HTTP_200_OK
    relationships = response.json()
    assert len(relationships) == 1
    assert relationships[0]["id"] == relationship.id
    assert relationships[0]["source_guideline"]["id"] == guideline.id
    assert relationships[0]["target_tag"]["id"] == tag.id


async def test_that_relationship_cannot_be_listed_without_guideline_id_or_tag_id(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.get("/relationships?kind=priority")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


async def test_that_relationship_cannot_be_listed_with_both_guideline_id_and_tag_id(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.get("/relationships?guideline_id=1&tag_id=2")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


async def test_that_relationship_cannot_be_listed_without_kind(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="condition",
        action="action",
    )

    response = await async_client.get(f"/relationships?guideline_id={guideline.id}")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


async def test_that_relationship_can_be_read(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    relationship_store = container[RelationshipStore]

    guideline = await guideline_store.create_guideline(
        condition="condition",
        action="action",
    )

    tag = await tag_store.create_tag(
        name="tag",
    )

    relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=guideline.id,
            type=EntityType.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=tag.id,
            type=EntityType.TAG,
        ),
        kind=GuidelineRelationshipKind.ENTAILMENT,
    )

    response = await async_client.get(f"/relationships/{relationship.id}")

    assert response.status_code == status.HTTP_200_OK

    relationship_data = response.json()
    assert relationship_data["id"] == relationship.id
    assert relationship_data["source_guideline"]["id"] == guideline.id
    assert relationship_data["target_tag"]["id"] == tag.id
    assert relationship_data["kind"] == "entailment"


async def test_that_entailment_relationship_can_be_created(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    source_guideline = await guideline_store.create_guideline(
        condition="source condition",
        action="source action",
    )

    target_guideline = await guideline_store.create_guideline(
        condition="target condition",
        action="target action",
    )

    response = await async_client.post(
        "/relationships",
        json={
            "source_guideline": source_guideline.id,
            "target_guideline": target_guideline.id,
            "kind": "entailment",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_guideline"]["id"] == source_guideline.id
    assert relationship["target_guideline"]["id"] == target_guideline.id
    assert relationship["kind"] == "entailment"


async def test_that_entailment_relationship_can_be_deleted(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    guideline_store = container[GuidelineStore]
    relationship_store = container[RelationshipStore]

    source_guideline = await guideline_store.create_guideline(
        condition="source condition",
        action="source action",
    )

    target_guideline = await guideline_store.create_guideline(
        condition="target condition",
        action="target action",
    )

    relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=source_guideline.id,
            type=EntityType.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=target_guideline.id,
            type=EntityType.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.ENTAILMENT,
    )

    response = await async_client.delete(f"/relationships/{relationship.id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await relationship_store.read_relationship(id=relationship.id)


async def test_that_dependency_relationship_can_be_created(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    source_guideline = await guideline_store.create_guideline(
        condition="source condition",
        action="source action",
    )

    target_guideline = await guideline_store.create_guideline(
        condition="target condition",
        action="target action",
    )

    response = await async_client.post(
        "/relationships",
        json={
            "source_guideline": source_guideline.id,
            "target_guideline": target_guideline.id,
            "kind": "dependency",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_guideline"]["id"] == source_guideline.id
    assert relationship["target_guideline"]["id"] == target_guideline.id
    assert relationship["kind"] == "dependency"


async def test_that_dependency_relationship_can_be_deleted(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    guideline_store = container[GuidelineStore]
    relationship_store = container[RelationshipStore]

    source_guideline = await guideline_store.create_guideline(
        condition="condition",
        action="action",
    )

    relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=source_guideline.id,
            type=EntityType.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=source_guideline.id,
            type=EntityType.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.DEPENDENCY,
    )

    response = await async_client.delete(f"/relationships/{relationship.id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await relationship_store.read_relationship(id=relationship.id)


async def test_that_priority_relationship_can_be_created(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]

    source_guideline = await guideline_store.create_guideline(
        condition="source condition",
        action="source action",
    )

    target_guideline = await guideline_store.create_guideline(
        condition="target condition",
        action="target action",
    )

    response = await async_client.post(
        "/relationships",
        json={
            "source_guideline": source_guideline.id,
            "target_guideline": target_guideline.id,
            "kind": "priority",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_guideline"]["id"] == source_guideline.id
    assert relationship["target_guideline"]["id"] == target_guideline.id
    assert relationship["kind"] == "priority"


async def test_that_priority_relationship_can_be_deleted(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    guideline_store = container[GuidelineStore]
    relationship_store = container[RelationshipStore]

    source_guideline = await guideline_store.create_guideline(
        condition="source condition",
        action="source action",
    )

    target_guideline = await guideline_store.create_guideline(
        condition="target condition",
        action="target action",
    )

    relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=source_guideline.id,
            type=EntityType.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=target_guideline.id,
            type=EntityType.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    response = await async_client.delete(f"/relationships/{relationship.id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await relationship_store.read_relationship(id=relationship.id)


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
            type=EntityType.TOOL,
        ),
        target=RelationshipEntity(
            id=second_tool_id,
            type=EntityType.TOOL,
        ),
        kind=ToolRelationshipKind.OVERLAP,
    )

    response = await async_client.delete(f"/relationships/{relationship.id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await relationship_store.read_relationship(id=relationship.id)
