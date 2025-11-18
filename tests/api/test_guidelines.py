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

from fastapi import status
import httpx
from lagom import Container
from pytest import raises

from parlant.core.agents import AgentId, AgentStore
from parlant.core.common import ItemNotFoundError
from parlant.core.journeys import JourneyStore
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipEntity,
    RelationshipStore,
)
from parlant.core.guideline_tool_associations import GuidelineToolAssociationStore
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineStore
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tags import Tag, TagId, TagStore
from parlant.core.tools import LocalToolService, ToolId, ToolOverlap

from tests.test_utilities import (
    run_openapi_server,
    run_service_server,
)


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


async def test_legacy_that_a_guideline_can_be_created(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    request_data = {
        "invoices": [
            {
                "payload": {
                    "kind": "guideline",
                    "guideline": {
                        "content": {
                            "condition": "the customer greets you",
                            "action": "greet them back with 'Hello'",
                        },
                        "operation": "add",
                        "coherence_check": True,
                        "connection_proposition": True,
                    },
                },
                "checksum": "checksum_value",
                "approved": True,
                "data": {
                    "guideline": {
                        "coherence_checks": [],
                        "connection_propositions": None,
                    }
                },
                "error": None,
            }
        ],
    }

    response = await async_client.post(f"/agents/{agent_id}/guidelines", json=request_data)
    assert response.status_code == status.HTTP_201_CREATED
    items = response.json()["items"]

    assert len(items) == 1
    assert items[0]["guideline"]["condition"] == "the customer greets you"
    assert items[0]["guideline"]["action"] == "greet them back with 'Hello'"


async def test_legacy_that_a_guideline_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline_to_delete = await guideline_store.create_guideline(
        condition="the customer wants to unsubscribe",
        action="ask for confirmation",
    )

    await guideline_store.upsert_tag(
        guideline_id=guideline_to_delete.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    (
        await async_client.delete(f"/agents/{agent_id}/guidelines/{guideline_to_delete.id}")
    ).raise_for_status()

    with raises(ItemNotFoundError):
        await guideline_store.read_guideline(guideline_id=guideline_to_delete.id)


async def test_legacy_that_an_unapproved_invoice_is_rejected(
    async_client: httpx.AsyncClient,
) -> None:
    request_data = {
        "invoices": [
            {
                "payload": {
                    "kind": "guideline",
                    "guideline": {
                        "content": {
                            "condition": "the customer says goodbye",
                            "action": "say 'Goodbye' back",
                        },
                        "operation": "add",
                        "coherence_check": True,
                        "connection_proposition": True,
                    },
                },
                "checksum": "checksum_value",
                "approved": False,
                "data": {
                    "guideline": {
                        "coherence_checks": [],
                        "connection_propositions": [],
                    },
                },
                "error": None,
            }
        ],
    }

    response = await async_client.post("/agents/{agent_id}/guidelines", json=request_data)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    response_data = response.json()
    assert "detail" in response_data
    assert response_data["detail"] == "Unapproved invoice"


async def test_legacy_that_a_connection_between_two_introduced_guidelines_is_created(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    invoices = [
        {
            "payload": {
                "kind": "guideline",
                "guideline": {
                    "content": {
                        "condition": "the customer asks about nearby restaurants",
                        "action": "provide a list of restaurants",
                    },
                    "operation": "add",
                    "coherence_check": True,
                    "connection_proposition": True,
                },
            },
            "checksum": "checksum1",
            "approved": True,
            "data": {
                "guideline": {
                    "coherence_checks": [],
                    "connection_propositions": [
                        {
                            "check_kind": "connection_with_another_evaluated_guideline",
                            "source": {
                                "condition": "the customer asks about nearby restaurants",
                                "action": "provide a list of restaurants",
                            },
                            "target": {
                                "condition": "highlight the best-reviewed restaurant",
                                "action": "recommend the top choice",
                            },
                        }
                    ],
                }
            },
            "error": None,
        },
        {
            "payload": {
                "kind": "guideline",
                "guideline": {
                    "content": {
                        "condition": "highlight the best-reviewed restaurant",
                        "action": "recommend the top choice",
                    },
                    "operation": "add",
                    "coherence_check": True,
                    "connection_proposition": True,
                },
            },
            "checksum": "checksum2",
            "approved": True,
            "data": {
                "guideline": {
                    "coherence_checks": [],
                    "connection_propositions": [
                        {
                            "check_kind": "connection_with_another_evaluated_guideline",
                            "source": {
                                "condition": "the customer asks about nearby restaurants",
                                "action": "provide a list of restaurants",
                            },
                            "target": {
                                "condition": "highlight the best-reviewed restaurant",
                                "action": "recommend the top choice",
                            },
                        }
                    ],
                }
            },
            "error": None,
        },
    ]

    items = (
        (
            await async_client.post(
                f"/agents/{agent_id}/guidelines",
                json={
                    "invoices": invoices,
                },
            )
        )
        .raise_for_status()
        .json()["items"]
    )

    source_guideline_item = next(
        (
            i
            for i in items
            if i["guideline"]["condition"] == "the customer asks about nearby restaurants"
        ),
        None,
    )
    assert source_guideline_item

    relationships = await container[RelationshipStore].list_relationships(
        kind=RelationshipKind.ENTAILMENT,
        indirect=False,
        source_id=source_guideline_item["guideline"]["id"],
    )

    assert len(relationships) == 1


async def test_legacy_that_a_connection_to_an_existing_guideline_is_created(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    existing_guideline = await guideline_store.create_guideline(
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    invoice = {
        "payload": {
            "kind": "guideline",
            "guideline": {
                "content": {
                    "condition": "provide the current weather update",
                    "action": "include temperature and humidity",
                },
                "operation": "add",
                "coherence_check": True,
                "connection_proposition": True,
            },
        },
        "checksum": "checksum_new",
        "approved": True,
        "data": {
            "guideline": {
                "coherence_checks": [],
                "connection_propositions": [
                    {
                        "check_kind": "connection_with_existing_guideline",
                        "source": {
                            "condition": "the customer asks about the weather",
                            "action": "provide the current weather update",
                        },
                        "target": {
                            "condition": "provide the current weather update",
                            "action": "include temperature and humidity",
                        },
                    }
                ],
            }
        },
        "error": None,
    }

    introduced_guideline = (
        (
            await async_client.post(
                f"/agents/{agent_id}/guidelines",
                json={
                    "invoices": [invoice],
                },
            )
        )
        .raise_for_status()
        .json()["items"][0]["guideline"]
    )

    relationships = await container[RelationshipStore].list_relationships(
        kind=RelationshipKind.ENTAILMENT,
        indirect=False,
        source_id=existing_guideline.id,
    )

    assert len(relationships) == 1
    assert relationships[0].source.id == existing_guideline.id
    assert relationships[0].target.id == introduced_guideline["id"]


async def test_legacy_that_a_guideline_can_be_read_by_id(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    _ = await guideline_store.upsert_tag(
        guideline_id=guideline.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    item = (
        (await async_client.get(f"/agents/{agent_id}/guidelines/{guideline.id}"))
        .raise_for_status()
        .json()
    )

    assert item["guideline"]["id"] == guideline.id
    assert item["guideline"]["condition"] == "the customer asks about the weather"
    assert item["guideline"]["action"] == "provide the current weather update"
    assert len(item["connections"]) == 0


async def test_legacy_that_guidelines_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline = await create_guidelines_and_create_relationships_between_them(
        container,
        agent_id,
        [
            GuidelineContent("A", "B"),
            GuidelineContent("B", "C"),
        ],
    )

    response_guidelines = (
        (await async_client.get(f"/agents/{agent_id}/guidelines")).raise_for_status().json()
    )

    assert len(response_guidelines) == 2
    assert any(guideline[0].id == g["id"] for g in response_guidelines)
    assert any(guideline[1].id == g["id"] for g in response_guidelines)


async def test_legacy_that_a_connection_can_be_added_to_a_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guidelines = await create_guidelines_and_create_relationships_between_them(
        container,
        agent_id,
        [
            GuidelineContent("A", "B"),
            GuidelineContent("B", "C"),
        ],
    )

    response_connections = (
        (
            await async_client.patch(
                f"/agents/{agent_id}/guidelines/{guidelines[0].id}",
                json={
                    "connections": {
                        "add": [
                            {
                                "source": guidelines[0].id,
                                "target": guidelines[1].id,
                            }
                        ],
                    },
                },
            )
        )
        .raise_for_status()
        .json()["connections"]
    )

    stored_relationships = list(
        await container[RelationshipStore].list_relationships(
            kind=RelationshipKind.ENTAILMENT,
            indirect=False,
            source_id=guidelines[0].id,
        )
    )

    assert len(stored_relationships) == 1
    assert stored_relationships[0].source.id == guidelines[0].id
    assert stored_relationships[0].target.id == guidelines[1].id

    assert len(response_connections) == 1
    assert response_connections[0]["source"]["id"] == guidelines[0].id
    assert response_connections[0]["target"]["id"] == guidelines[1].id


async def test_legacy_that_a_direct_target_connection_can_be_removed_from_a_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guidelines = await create_guidelines_and_create_relationships_between_them(
        container,
        agent_id,
        [
            GuidelineContent("A", "B"),
            GuidelineContent("B", "C"),
        ],
    )

    response_collections = (
        (
            await async_client.patch(
                f"/agents/{agent_id}/guidelines/{guidelines[0].id}",
                json={
                    "connections": {
                        "remove": [guidelines[1].id],
                    },
                },
            )
        )
        .raise_for_status()
        .json()["connections"]
    )

    assert len(response_collections) == 0

    stored_relationships = await container[RelationshipStore].list_relationships(
        kind=RelationshipKind.ENTAILMENT,
        indirect=True,
        source_id=guidelines[0].id,
    )

    assert len(stored_relationships) == 0


async def test_legacy_that_an_indirect_connection_cannot_be_removed_from_a_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guidelines = await create_guidelines_and_create_relationships_between_them(
        container,
        agent_id,
        [
            GuidelineContent("A", "B"),
            GuidelineContent("B", "C"),
            GuidelineContent("C", "D"),
        ],
    )

    response = await async_client.patch(
        f"/agents/{agent_id}/guidelines/{guidelines[0].id}",
        json={
            "connections": {
                "remove": [guidelines[2].id],
            },
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    stored_relationships = await container[RelationshipStore].list_relationships(
        kind=RelationshipKind.ENTAILMENT,
        indirect=True,
        source_id=guidelines[0].id,
    )

    assert len(stored_relationships) == 2


async def test_legacy_that_deleting_a_guideline_also_deletes_all_of_its_direct_connections(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guidelines = await create_guidelines_and_create_relationships_between_them(
        container,
        agent_id,
        [
            GuidelineContent("A", "B"),
            GuidelineContent("B", "C"),
        ],
    )

    (
        await async_client.delete(f"/agents/{agent_id}/guidelines/{guidelines[0].id}")
    ).raise_for_status()

    stored_relationships = await container[RelationshipStore].list_relationships(
        kind=RelationshipKind.ENTAILMENT,
        indirect=False,
        source_id=guidelines[0].id,
    )

    assert not stored_relationships


async def test_legacy_that_reading_a_guideline_lists_both_direct_and_indirect_connections(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    guidelines = [
        await guideline_store.create_guideline(
            condition=condition,
            action=action,
        )
        for condition, action in [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("D", "E"),
            ("E", "F"),
        ]
    ]

    for guideline in guidelines:
        _ = await guideline_store.upsert_tag(
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

    third_item = (
        (await async_client.get(f"/agents/{agent_id}/guidelines/{guidelines[2].id}"))
        .raise_for_status()
        .json()
    )

    assert 2 == len([c for c in third_item["connections"] if c["indirect"]])
    assert 2 == len([c for c in third_item["connections"] if not c["indirect"]])

    relationships = sorted(third_item["connections"], key=lambda c: c["source"]["condition"])

    for i, c in enumerate(relationships):
        guideline_a = guidelines[i]
        guideline_b = guidelines[i + 1]

        assert c["source"] == {
            "id": guideline_a.id,
            "condition": guideline_a.content.condition,
            "action": guideline_a.content.action,
            "enabled": guideline_a.enabled,
        }

        assert c["target"] == {
            "id": guideline_b.id,
            "condition": guideline_b.content.condition,
            "action": guideline_b.content.action,
            "enabled": guideline_b.enabled,
        }

        is_direct = third_item["guideline"]["id"] in (c["source"]["id"], c["target"]["id"])
        assert c["indirect"] is not is_direct


async def test_legacy_that_a_tool_association_can_be_added(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
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

    _ = await guideline_store.upsert_tag(
        guideline_id=guideline.id,
        tag_id=Tag.for_agent_id(agent_id),
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
        f"/agents/{agent_id}/guidelines/{guideline.id}",
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

    association_store = container[GuidelineToolAssociationStore]
    associations = await association_store.list_associations()

    matching_associations = [
        assoc
        for assoc in associations
        if assoc.guideline_id == guideline.id and assoc.tool_id == (service_name, tool_name)
    ]

    assert len(matching_associations) == 1

    tool_associations = (
        (await async_client.get(f"/agents/{agent_id}/guidelines/{guideline.id}"))
        .raise_for_status()
        .json()["tool_associations"]
    )

    assert any(
        a["guideline_id"] == guideline.id
        and a["tool_id"]["service_name"] == service_name
        and a["tool_id"]["tool_name"] == tool_name
        for a in tool_associations
    )


async def test_legacy_that_a_tool_association_can_be_removed(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]
    local_tool_service = container[LocalToolService]
    association_store = container[GuidelineToolAssociationStore]

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

    _ = await guideline_store.upsert_tag(
        guideline_id=guideline.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    service_name = "local"
    tool_name = "fetch_event_data"

    await association_store.create_association(
        guideline_id=guideline.id,
        tool_id=ToolId(service_name=service_name, tool_name=tool_name),
    )

    request_data = {
        "tool_associations": {
            "remove": [
                {
                    "service_name": service_name,
                    "tool_name": tool_name,
                }
            ]
        }
    }

    response = await async_client.patch(
        f"/agents/{agent_id}/guidelines/{guideline.id}",
        json=request_data,
    )

    assert response.status_code == status.HTTP_200_OK

    response_data = response.json()

    assert "tool_associations" in response_data
    assert response_data["tool_associations"] == []

    associations_after = await association_store.list_associations()
    assert not any(
        assoc.guideline_id == guideline.id and assoc.tool_id == (service_name, tool_name)
        for assoc in associations_after
    )

    tool_associations = (
        (await async_client.get(f"/agents/{agent_id}/guidelines/{guideline.id}"))
        .raise_for_status()
        .json()["tool_associations"]
    )

    assert tool_associations == []


async def test_legacy_that_guideline_deletion_removes_tool_associations(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]
    local_tool_service = container[LocalToolService]
    association_store = container[GuidelineToolAssociationStore]

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

    _ = await guideline_store.upsert_tag(
        guideline_id=guideline.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    service_name = "local"
    tool_name = "fetch_event_data"

    await association_store.create_association(
        guideline_id=guideline.id,
        tool_id=ToolId(service_name=service_name, tool_name=tool_name),
    )

    await async_client.delete(f"/agents/{agent_id}/guidelines/{guideline.id}")

    associations_after = await association_store.list_associations()
    assert not any(assoc.guideline_id == guideline.id for assoc in associations_after)


async def test_legacy_that_an_http_404_is_thrown_when_associating_with_a_nonexistent_local_tool_to_a_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
    )

    _ = await guideline_store.upsert_tag(
        guideline_id=guideline.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    service_name = "local"
    tool_name = "nonexistent_tool"

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
        f"/agents/{agent_id}/guidelines/{guideline.id}",
        json=request_data,
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_legacy_that_an_http_404_is_thrown_when_associating_with_a_nonexistent_openapi_tool_to_a_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]
    service_registry = container[ServiceRegistry]

    guideline = await guideline_store.create_guideline(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
    )

    _ = await guideline_store.upsert_tag(
        guideline_id=guideline.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    tool_name = "nonexistent_tool"

    async with run_openapi_server() as server_info:
        url = f"{server_info.url}:{server_info.port}"
        source = f"{url}/openapi.json"
        await service_registry.update_tool_service(
            name="my_openapi_service",
            kind="openapi",
            url=url,
            source=source,
        )

        request_data = {
            "tool_associations": {
                "add": [
                    {
                        "service_name": "my_openapi_service",
                        "tool_name": tool_name,
                    }
                ]
            }
        }

        response = await async_client.patch(
            f"/agents/{agent_id}/guidelines/{guideline.id}",
            json=request_data,
        )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_legacy_that_an_http_404_is_thrown_when_associating_with_a_nonexistent_sdk_tool_to_a_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]
    service_registry = container[ServiceRegistry]

    guideline = await guideline_store.create_guideline(
        condition="the customer wants to get meeting details",
        action="get meeting event information",
    )

    _ = await guideline_store.upsert_tag(
        guideline_id=guideline.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    tool_name = "nonexistent_tool"

    async with run_service_server([]) as server:
        await service_registry.update_tool_service(
            name="my_sdk_service",
            kind="sdk",
            url=server.url,
        )

        request_data = {
            "tool_associations": {
                "add": [
                    {
                        "service_name": "my_sdk_service",
                        "tool_name": tool_name,
                    }
                ]
            }
        }

        response = await async_client.patch(
            f"/agents/{agent_id}/guidelines/{guideline.id}",
            json=request_data,
        )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_legacy_that_an_existing_guideline_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]
    relationship_store = container[RelationshipStore]

    existing_guideline = await guideline_store.create_guideline(
        condition="the customer greets you",
        action="reply with 'Hello'",
    )

    _ = await guideline_store.upsert_tag(
        guideline_id=existing_guideline.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    connected_guideline = await guideline_store.create_guideline(
        condition="reply with 'Hello'",
        action="finish with a smile",
    )

    _ = await guideline_store.upsert_tag(
        guideline_id=connected_guideline.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    connected_guideline_post_update = await guideline_store.create_guideline(
        condition="reply with 'Howdy!'",
        action="finish with a smile",
    )

    _ = await guideline_store.upsert_tag(
        guideline_id=connected_guideline_post_update.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=existing_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=connected_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )

    new_action = "reply with 'Howdy!'"

    request_data = {
        "invoices": [
            {
                "payload": {
                    "kind": "guideline",
                    "guideline": {
                        "content": {
                            "condition": "the customer greets you",
                            "action": new_action,
                        },
                        "operation": "update",
                        "coherence_check": True,
                        "connection_proposition": True,
                        "updated_id": existing_guideline.id,
                    },
                },
                "checksum": "checksum_new",
                "approved": True,
                "data": {
                    "guideline": {
                        "coherence_checks": [],
                        "connection_propositions": [
                            {
                                "check_kind": "connection_with_existing_guideline",
                                "source": {
                                    "condition": "the customer greets you",
                                    "action": new_action,
                                },
                                "target": {
                                    "condition": connected_guideline_post_update.content.condition,
                                    "action": connected_guideline_post_update.content.action,
                                },
                            }
                        ],
                    },
                },
                "error": None,
            }
        ]
    }

    items = (
        (await async_client.post(f"/agents/{agent_id}/guidelines", json=request_data))
        .raise_for_status()
        .json()["items"]
    )

    assert len(items) == 1
    updated_guideline = items[0]["guideline"]
    assert updated_guideline["id"] == existing_guideline.id
    assert updated_guideline["condition"] == "the customer greets you"
    assert updated_guideline["action"] == new_action

    updated_relationships = await relationship_store.list_relationships(
        kind=RelationshipKind.ENTAILMENT,
        indirect=False,
        source_id=existing_guideline.id,
    )
    assert len(updated_relationships) == 1
    assert updated_relationships[0].source.id == existing_guideline.id
    assert updated_relationships[0].target.id == connected_guideline_post_update.id


async def test_legacy_that_an_updated_guideline_can_entail_an_added_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]
    relationship_store = container[RelationshipStore]

    existing_guideline = await guideline_store.create_guideline(
        condition="the customer greets you",
        action="reply with 'Hello'",
    )

    _ = await guideline_store.upsert_tag(
        guideline_id=existing_guideline.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    new_action = "reply with 'Howdy!'"

    request_data = {
        "invoices": [
            {
                "payload": {
                    "kind": "guideline",
                    "guideline": {
                        "content": {
                            "condition": "the customer greets you",
                            "action": new_action,
                        },
                        "operation": "update",
                        "coherence_check": True,
                        "connection_proposition": True,
                        "updated_id": existing_guideline.id,
                    },
                },
                "checksum": "checksum_update",
                "approved": True,
                "data": {
                    "guideline": {
                        "coherence_checks": [],
                        "connection_propositions": [
                            {
                                "check_kind": "connection_with_another_evaluated_guideline",
                                "source": {
                                    "condition": "the customer greets you",
                                    "action": new_action,
                                },
                                "target": {
                                    "condition": "replying to greeting message",
                                    "action": "ask how they are",
                                },
                            }
                        ],
                    }
                },
                "error": None,
            },
            {
                "payload": {
                    "kind": "guideline",
                    "guideline": {
                        "content": {
                            "condition": "replying to greeting message",
                            "action": "ask how they are",
                        },
                        "operation": "add",
                        "coherence_check": True,
                        "connection_proposition": True,
                    },
                },
                "checksum": "checksum_new_guideline",
                "approved": True,
                "data": {
                    "guideline": {
                        "coherence_checks": [],
                        "connection_propositions": [
                            {
                                "check_kind": "connection_with_another_evaluated_guideline",
                                "source": {
                                    "condition": "the customer greets you",
                                    "action": new_action,
                                },
                                "target": {
                                    "condition": "replying to greeting message",
                                    "action": "ask how they are",
                                },
                            }
                        ],
                    }
                },
                "error": None,
            },
        ]
    }

    items = (
        (await async_client.post(f"/agents/{agent_id}/guidelines", json=request_data))
        .raise_for_status()
        .json()["items"]
    )

    assert len(items) == 2

    updated_guideline = await guideline_store.read_guideline(existing_guideline.id)

    added_guideline_id = (
        items[1]["guideline"]["id"]
        if items[0]["guideline"]["id"] == existing_guideline.id
        else items[0]["guideline"]["id"]
    )

    added_guideline = await guideline_store.read_guideline(added_guideline_id)

    updated_relationships = await relationship_store.list_relationships(
        kind=RelationshipKind.ENTAILMENT,
        indirect=False,
        source_id=updated_guideline.id,
    )

    assert len(updated_relationships) == 1
    assert updated_relationships[0].source.id == updated_guideline.id
    assert updated_relationships[0].target.id == added_guideline.id


async def test_legacy_that_guideline_update_retains_existing_connections_with_disabled_connection_proposition(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]
    relationship_store = container[RelationshipStore]

    existing_guideline = await guideline_store.create_guideline(
        condition="the customer greets you",
        action="reply with 'Hello'",
    )

    _ = await guideline_store.upsert_tag(
        guideline_id=existing_guideline.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    connected_guideline = await guideline_store.create_guideline(
        condition="reply with 'Hello'",
        action="finish with a smile",
    )

    _ = await guideline_store.upsert_tag(
        guideline_id=connected_guideline.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=existing_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=connected_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )

    new_action = "reply with 'Howdy!'"

    request_data = {
        "invoices": [
            {
                "payload": {
                    "kind": "guideline",
                    "guideline": {
                        "content": {
                            "condition": "the customer greets you",
                            "action": new_action,
                        },
                        "operation": "update",
                        "coherence_check": True,
                        "connection_proposition": False,
                        "updated_id": existing_guideline.id,
                    },
                },
                "checksum": "checksum_new",
                "approved": True,
                "data": {
                    "guideline": {
                        "coherence_checks": [],
                        "connection_propositions": None,
                    }
                },
                "error": None,
            }
        ]
    }

    items = (
        (await async_client.post(f"/agents/{agent_id}/guidelines", json=request_data))
        .raise_for_status()
        .json()["items"]
    )

    assert len(items) == 1
    updated_guideline = items[0]["guideline"]
    assert updated_guideline["id"] == existing_guideline.id
    assert updated_guideline["condition"] == "the customer greets you"
    assert updated_guideline["action"] == new_action

    updated_relationships = await relationship_store.list_relationships(
        kind=RelationshipKind.ENTAILMENT,
        indirect=False,
        source_id=existing_guideline.id,
    )
    assert len(updated_relationships) == 1
    assert updated_relationships[0].source.id == existing_guideline.id
    assert updated_relationships[0].target.id == connected_guideline.id


async def test_legacy_that_a_guideline_can_be_disabled(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer greets you",
        action="reply with 'Hello'",
    )

    _ = await guideline_store.upsert_tag(
        guideline_id=guideline.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    response = (
        await async_client.patch(
            f"/agents/{agent_id}/guidelines/{guideline.id}",
            json={"enabled": False},
        )
    ).raise_for_status()

    assert response.status_code == status.HTTP_200_OK

    updated_guideline = response.json()["guideline"]
    assert not updated_guideline["enabled"]


async def test_legacy_that_retrieving_a_guideline_associated_with_a_wrong_agent_id_returns_a_404(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer greets you",
        action="reply with 'Hello'",
    )

    _ = await guideline_store.upsert_tag(
        guideline_id=guideline.id,
        tag_id=Tag.for_agent_id("wrong_agent"),
    )

    response = await async_client.get(f"/agents/{agent_id}/guidelines/{guideline.id}")

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_legacy_that_updating_a_guideline_with_a_wrong_agent_id_returns_a_404(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer greets you",
        action="reply with 'Hello'",
    )

    _ = await guideline_store.upsert_tag(
        guideline_id=guideline.id,
        tag_id=Tag.for_agent_id("wrong_agent"),
    )

    response = await async_client.patch(
        f"/agents/{agent_id}/guidelines/{guideline.id}",
        json={"enabled": False},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_legacy_that_deleting_a_guideline_with_a_wrong_agent_id_returns_a_404(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer greets you",
        action="reply with 'Hello'",
    )

    _ = await guideline_store.upsert_tag(
        guideline_id=guideline.id,
        tag_id=Tag.for_agent_id("wrong_agent"),
    )

    response = await async_client.delete(f"/agents/{agent_id}/guidelines/{guideline.id}")

    assert response.status_code == status.HTTP_404_NOT_FOUND


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
