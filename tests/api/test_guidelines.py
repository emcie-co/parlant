# Copyright 2024 Emcie Co Ltd.
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

from parlant.core.agents import AgentId
from parlant.core.common import ItemNotFoundError
from parlant.core.guideline_connections import ConnectionKind, GuidelineConnectionStore
from parlant.core.guideline_tool_associations import GuidelineToolAssociationStore
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineStore
from parlant.core.tools import LocalToolService, ToolId


async def create_and_connect(
    container: Container,
    agent_id: AgentId,
    guideline_contents: list[GuidelineContent],
) -> list[Guideline]:
    guidelines = [
        await container[GuidelineStore].create_guideline(
            guideline_set=agent_id,
            condition=gc.condition,
            action=gc.action,
        )
        for gc in guideline_contents
    ]

    for source, target in zip(guidelines, guidelines[1:]):
        await container[GuidelineConnectionStore].create_connection(
            source=source.id, target=target.id, kind=ConnectionKind.ENTAILS
        )

    return guidelines


async def test_that_a_guideline_can_be_created(
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


async def test_that_a_guideline_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline_to_delete = await guideline_store.create_guideline(
        guideline_set=agent_id,
        condition="the customer wants to unsubscribe",
        action="ask for confirmation",
    )

    (
        await async_client.delete(f"/agents/{agent_id}/guidelines/{guideline_to_delete.id}")
    ).raise_for_status()

    with raises(ItemNotFoundError):
        await guideline_store.read_guideline(
            guideline_set=agent_id, guideline_id=guideline_to_delete.id
        )


async def test_that_an_unapproved_invoice_is_rejected(
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
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    response_data = response.json()
    assert "detail" in response_data
    assert response_data["detail"] == "Unapproved invoice"


async def test_that_a_connection_between_two_introduced_guidelines_is_created(
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
                            "connection_kind": "entails",
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
                            "connection_kind": "entails",
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

    connections = await container[GuidelineConnectionStore].list_connections(
        indirect=False,
        source=source_guideline_item["guideline"]["id"],
    )

    assert len(connections) == 1


async def test_that_a_connection_to_an_existing_guideline_is_created(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    existing_guideline = await guideline_store.create_guideline(
        guideline_set=agent_id,
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
                        "connection_kind": "entails",
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

    connections = await container[GuidelineConnectionStore].list_connections(
        indirect=False,
        source=existing_guideline.id,
    )

    assert len(connections) == 1
    assert connections[0].source == existing_guideline.id
    assert connections[0].target == introduced_guideline["id"]


async def test_that_a_guideline_can_be_read_by_id(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    stored_guideline = await guideline_store.create_guideline(
        guideline_set=agent_id,
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    item = (
        (await async_client.get(f"/agents/{agent_id}/guidelines/{stored_guideline.id}"))
        .raise_for_status()
        .json()
    )

    assert item["guideline"]["id"] == stored_guideline.id
    assert item["guideline"]["condition"] == "the customer asks about the weather"
    assert item["guideline"]["action"] == "provide the current weather update"
    assert len(item["connections"]) == 0


async def test_that_guidelines_can_be_listed_for_an_agent(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    stored_guidelines = await create_and_connect(
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
    assert any(stored_guidelines[0].id == g["id"] for g in response_guidelines)
    assert any(stored_guidelines[1].id == g["id"] for g in response_guidelines)


async def test_that_a_connection_can_be_added_to_a_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guidelines = await create_and_connect(
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
                                "kind": "entails",
                            }
                        ],
                    },
                },
            )
        )
        .raise_for_status()
        .json()["connections"]
    )

    stored_connections = list(
        await container[GuidelineConnectionStore].list_connections(
            indirect=False,
            source=guidelines[0].id,
        )
    )

    assert len(stored_connections) == 1
    assert stored_connections[0].source == guidelines[0].id
    assert stored_connections[0].target == guidelines[1].id
    assert stored_connections[0].kind == ConnectionKind.ENTAILS

    assert len(response_connections) == 1
    assert response_connections[0]["source"]["id"] == guidelines[0].id
    assert response_connections[0]["target"]["id"] == guidelines[1].id
    assert response_connections[0]["kind"] == "entails"


async def test_that_a_direct_target_connection_can_be_removed_from_a_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guidelines = await create_and_connect(
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

    stored_connections = await container[GuidelineConnectionStore].list_connections(
        indirect=True,
        source=guidelines[0].id,
    )

    assert len(stored_connections) == 0


async def test_that_an_indirect_connection_cannot_be_removed_from_a_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guidelines = await create_and_connect(
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

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    stored_connections = await container[GuidelineConnectionStore].list_connections(
        indirect=True,
        source=guidelines[0].id,
    )

    assert len(stored_connections) == 2


async def test_that_deleting_a_guideline_also_deletes_all_of_its_direct_connections(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guidelines = await create_and_connect(
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

    stored_connections = await container[GuidelineConnectionStore].list_connections(
        indirect=False,
        source=guidelines[0].id,
    )

    assert not stored_connections


async def test_that_reading_a_guideline_lists_both_direct_and_indirect_connections(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guidelines = [
        await container[GuidelineStore].create_guideline(
            guideline_set=agent_id,
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

    for source, target in zip(guidelines, guidelines[1:]):
        await container[GuidelineConnectionStore].create_connection(
            source=source.id, target=target.id, kind=ConnectionKind.ENTAILS
        )

    third_item = (
        (await async_client.get(f"/agents/{agent_id}/guidelines/{guidelines[2].id}"))
        .raise_for_status()
        .json()
    )

    assert 2 == len([c for c in third_item["connections"] if c["indirect"]])
    assert 2 == len([c for c in third_item["connections"] if not c["indirect"]])

    connections = sorted(third_item["connections"], key=lambda c: c["source"]["condition"])

    for i, c in enumerate(connections):
        guideline_a = guidelines[i]
        guideline_b = guidelines[i + 1]

        assert c["source"] == {
            "id": guideline_a.id,
            "condition": guideline_a.content.condition,
            "action": guideline_a.content.action,
        }

        assert c["target"] == {
            "id": guideline_b.id,
            "condition": guideline_b.content.condition,
            "action": guideline_b.content.action,
        }

        is_direct = third_item["guideline"]["id"] in (c["source"]["id"], c["target"]["id"])
        assert c["indirect"] is not is_direct


async def test_that_a_tool_association_can_be_added(
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
    )

    guideline = await guideline_store.create_guideline(
        guideline_set=agent_id,
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


async def test_that_a_tool_association_can_be_removed(
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
    )

    guideline = await guideline_store.create_guideline(
        guideline_set=agent_id,
        condition="the customer wants to get meeting details",
        action="get meeting event information",
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


async def test_that_guideline_deletion_removes_tool_associations(
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
    )

    guideline = await guideline_store.create_guideline(
        guideline_set=agent_id,
        condition="the customer wants to get meeting details",
        action="get meeting event information",
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


async def test_that_an_existing_guideline_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]
    connection_store = container[GuidelineConnectionStore]

    existing_guideline = await guideline_store.create_guideline(
        guideline_set=agent_id,
        condition="the customer greets you",
        action="reply with 'Hello'",
    )
    connected_guideline = await guideline_store.create_guideline(
        guideline_set=agent_id,
        condition="reply with 'Hello'",
        action="finish with a smile",
    )

    connected_guideline_post_update = await guideline_store.create_guideline(
        guideline_set=agent_id,
        condition="reply with 'Howdy!'",
        action="finish with a smile",
    )

    await connection_store.create_connection(
        source=existing_guideline.id,
        target=connected_guideline.id,
        kind=ConnectionKind.ENTAILS,
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
                                "connection_kind": "entails",
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

    updated_connections = await connection_store.list_connections(
        indirect=False, source=existing_guideline.id
    )
    assert len(updated_connections) == 1
    assert updated_connections[0].source == existing_guideline.id
    assert updated_connections[0].target == connected_guideline_post_update.id
    assert updated_connections[0].kind == ConnectionKind.ENTAILS


async def test_that_an_updated_guideline_can_entail_an_added_guideline(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]
    connection_store = container[GuidelineConnectionStore]

    existing_guideline = await guideline_store.create_guideline(
        guideline_set=agent_id,
        condition="the customer greets you",
        action="reply with 'Hello'",
    )

    new_aciton = "reply with 'Howdy!'"

    request_data = {
        "invoices": [
            {
                "payload": {
                    "kind": "guideline",
                    "guideline": {
                        "content": {
                            "condition": "the customer greets you",
                            "action": new_aciton,
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
                                    "action": new_aciton,
                                },
                                "target": {
                                    "condition": "replying to greeting message",
                                    "action": "ask how they are",
                                },
                                "connection_kind": "entails",
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
                                    "action": new_aciton,
                                },
                                "target": {
                                    "condition": "replying to greeting message",
                                    "action": "ask how they are",
                                },
                                "connection_kind": "entails",
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

    updated_guideline = await guideline_store.read_guideline(agent_id, existing_guideline.id)

    added_guideline_id = (
        items[1]["guideline"]["id"]
        if items[0]["guideline"]["id"] == existing_guideline.id
        else items[0]["guideline"]["id"]
    )

    added_guideline = await guideline_store.read_guideline(agent_id, added_guideline_id)

    updated_connections = await connection_store.list_connections(
        indirect=False, source=updated_guideline.id
    )

    assert len(updated_connections) == 1
    assert updated_connections[0].source == updated_guideline.id
    assert updated_connections[0].target == added_guideline.id
    assert updated_connections[0].kind == ConnectionKind.ENTAILS


async def test_that_guideline_update_retains_existing_connections_with_disabled_connection_proposition(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]
    connection_store = container[GuidelineConnectionStore]

    existing_guideline = await guideline_store.create_guideline(
        guideline_set=agent_id,
        condition="the customer greets you",
        action="reply with 'Hello'",
    )
    connected_guideline = await guideline_store.create_guideline(
        guideline_set=agent_id,
        condition="reply with 'Hello'",
        action="finish with a smile",
    )

    await connection_store.create_connection(
        source=existing_guideline.id,
        target=connected_guideline.id,
        kind=ConnectionKind.ENTAILS,
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

    updated_connections = await connection_store.list_connections(
        indirect=False, source=existing_guideline.id
    )
    assert len(updated_connections) == 1
    assert updated_connections[0].source == existing_guideline.id
    assert updated_connections[0].target == connected_guideline.id
    assert updated_connections[0].kind == ConnectionKind.ENTAILS
