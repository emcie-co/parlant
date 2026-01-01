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
from parlant.core.guidelines import GuidelineStore
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tags import Tag, TagStore
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
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=tag.id,
            kind=RelationshipEntityKind.TAG,
        ),
        kind=RelationshipKind.PRIORITY,
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
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=tag.id,
            kind=RelationshipEntityKind.TAG,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    response = await async_client.get(f"/relationships?tag_id={tag.id}&kind=priority")
    assert response.status_code == status.HTTP_200_OK
    relationships = response.json()
    assert len(relationships) == 1
    assert relationships[0]["id"] == relationship.id
    assert relationships[0]["source_guideline"]["id"] == guideline.id
    assert relationships[0]["target_tag"]["id"] == tag.id


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
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=tag.id,
            kind=RelationshipEntityKind.TAG,
        ),
        kind=RelationshipKind.ENTAILMENT,
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
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=target_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
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
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=source_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    response = await async_client.delete(f"/relationships/{relationship.id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await relationship_store.read_relationship(relationship_id=relationship.id)


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
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=target_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
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
            "kind": "disambiguation",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_guideline"]["id"] == source_guideline.id
    assert relationship["target_guideline"]["id"] == target_guideline.id
    assert relationship["kind"] == "disambiguation"


async def test_that_disambiguation_relationship_can_be_deleted(
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
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=target_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
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
            "kind": "reevaluation",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    relationship = response.json()
    assert relationship["source_guideline"]["id"] == source_guideline.id
    assert relationship["target_guideline"]["id"] == target_guideline.id
    assert relationship["kind"] == "reevaluation"


async def test_that_reevaluation_relationship_can_be_deleted(
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
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=target_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
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
    guideline_store = container[GuidelineStore]
    relationship_store = container[RelationshipStore]

    g1 = await guideline_store.create_guideline(condition="A", action="B")
    g2 = await guideline_store.create_guideline(condition="C", action="D")
    g3 = await guideline_store.create_guideline(condition="E", action="F")

    r1 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.PRIORITY,
    )

    r2 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    r3 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DISAMBIGUATION,
    )

    r4 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
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
    guideline_store = container[GuidelineStore]
    relationship_store = container[RelationshipStore]

    g1 = await guideline_store.create_guideline(condition="AA", action="BB")
    g2 = await guideline_store.create_guideline(condition="CC", action="DD")

    priority_relationship = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.PRIORITY,
    )

    _ = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.ENTAILMENT,
    )

    response = await async_client.get("/relationships?kind=priority")
    assert response.status_code == status.HTTP_200_OK

    relationships = response.json()

    assert len(relationships) == 1
    assert relationships[0]["id"] == priority_relationship.id
    assert relationships[0]["kind"] == "priority"


async def test_that_relationships_can_be_listed_by_guideline_id_without_kind_filter_via_api(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    guideline_store = container[GuidelineStore]
    relationship_store = container[RelationshipStore]

    g1 = await guideline_store.create_guideline(condition="X", action="Y")
    g2 = await guideline_store.create_guideline(condition="Y", action="Z")
    g3 = await guideline_store.create_guideline(condition="Z", action="W")

    rel1 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.ENTAILMENT,
    )

    rel2 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    response = await async_client.get(f"/relationships?guideline_id={g1.id}")
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


async def test_that_relationships_of_guideline_and_a_journey_can_be_listed(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    relationship_store = container[RelationshipStore]

    g1 = await guideline_store.create_guideline(condition="A", action="B")

    j1 = await journey_store.create_journey(
        title="Journey 1",
        description="Description of Journey 1",
        conditions=[],
    )

    r1 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=Tag.for_journey_id(j1.id), kind=RelationshipEntityKind.TAG),
        kind=RelationshipKind.DEPENDENCY,
    )

    response = await async_client.get(f"/relationships?guideline_id={g1.id}")
    assert response.status_code == status.HTTP_200_OK

    relationships = response.json()

    returned_ids = {rel["id"] for rel in relationships}

    assert r1.id in returned_ids


async def test_that_relationships_of_a_journey_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    relationship_store = container[RelationshipStore]

    g1 = await guideline_store.create_guideline(condition="A", action="B")

    j1 = await journey_store.create_journey(
        title="Journey 1",
        description="Description of Journey 1",
        conditions=[],
    )

    r1 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=Tag.for_journey_id(j1.id), kind=RelationshipEntityKind.TAG),
        kind=RelationshipKind.DEPENDENCY,
    )

    response = await async_client.get(f"/relationships?tag_id=journey:{j1.id}")
    assert response.status_code == status.HTTP_200_OK

    relationships = response.json()

    returned_ids = {rel["id"] for rel in relationships}

    assert r1.id in returned_ids


async def test_that_relationships_of_guideline_and_an_agent_can_be_listed(
    async_client: httpx.AsyncClient, container: Container
) -> None:
    guideline_store = container[GuidelineStore]
    agent_store = container[AgentStore]
    relationship_store = container[RelationshipStore]

    g1 = await guideline_store.create_guideline(condition="A", action="B")

    a1 = await agent_store.create_agent(name="Agent 1", description="Description of Agent 1")

    r1 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=Tag.for_agent_id(a1.id), kind=RelationshipEntityKind.TAG),
        kind=RelationshipKind.DEPENDENCY,
    )

    response = await async_client.get(f"/relationships?guideline_id={g1.id}")
    assert response.status_code == status.HTTP_200_OK

    relationships = response.json()

    returned_ids = {rel["id"] for rel in relationships}

    assert r1.id in returned_ids


async def test_that_relationships_of_an_agent_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    guideline_store = container[GuidelineStore]
    agent_store = container[AgentStore]
    relationship_store = container[RelationshipStore]

    g1 = await guideline_store.create_guideline(condition="A", action="B")

    a1 = await agent_store.create_agent(name="Agent 1", description="Description of Agent 1")

    r1 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=Tag.for_agent_id(a1.id), kind=RelationshipEntityKind.TAG),
        kind=RelationshipKind.DEPENDENCY,
    )

    response = await async_client.get(f"/relationships?tag_id=agent:{a1.id}")
    assert response.status_code == status.HTTP_200_OK

    relationships = response.json()

    returned_ids = {rel["id"] for rel in relationships}

    assert r1.id in returned_ids
