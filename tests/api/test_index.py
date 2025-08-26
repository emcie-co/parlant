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

import asyncio
from fastapi import status
import httpx
from lagom import Container

from parlant.core.agents import AgentId
from parlant.core.evaluations import EvaluationStore
from parlant.core.guidelines import GuidelineStore

from parlant.core.tags import Tag
from tests.conftest import NoCachedGenerations
from tests.core.stable.services.indexing.test_evaluator import (
    AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING,
)


async def test_that_an_evaluation_can_be_created_and_fetched_with_completed_status(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    evaluation_store = container[EvaluationStore]

    response = await async_client.post(
        "/index/evaluations",
        json={
            "agent_id": agent_id,
            "payloads": [
                {
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
                }
            ],
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    evaluation_id = response.json()["id"]

    evaluation = await evaluation_store.read_evaluation(evaluation_id=evaluation_id)
    assert evaluation.id == evaluation_id

    content = (
        (await async_client.get(f"/index/evaluations/{evaluation_id}")).raise_for_status().json()
    )

    assert content["status"] == "completed"
    assert len(content["invoices"]) == 1
    invoice = content["invoices"][0]

    assert invoice["approved"]
    assert len(invoice["data"]["guideline"]["coherence_checks"]) == 0


async def test_that_an_evaluation_can_be_fetched_with_running_status(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
    no_cache: NoCachedGenerations,
) -> None:
    evaluation_id = (
        (
            await async_client.post(
                "/index/evaluations",
                json={
                    "agent_id": agent_id,
                    "payloads": [
                        {
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
                        {
                            "kind": "guideline",
                            "guideline": {
                                "content": {
                                    "condition": "the customer greeting you",
                                    "action": "greet them back with 'Hola'",
                                },
                                "operation": "add",
                                "coherence_check": True,
                                "connection_proposition": True,
                            },
                        },
                    ],
                },
            )
        )
        .raise_for_status()
        .json()["id"]
    )

    await asyncio.sleep(AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING)

    content = (
        (
            await async_client.get(
                f"/index/evaluations/{evaluation_id}", params={"wait_for_completion": 0}
            )
        )
        .raise_for_status()
        .json()
    )

    assert content["status"] == "running"


async def test_that_an_evaluation_can_be_fetched_with_a_completed_status_containing_a_detailed_approved_invoice(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    evaluation_id = (
        (
            await async_client.post(
                "/index/evaluations",
                json={
                    "agent_id": agent_id,
                    "payloads": [
                        {
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
                        }
                    ],
                },
            )
        )
        .raise_for_status()
        .json()["id"]
    )

    content = (
        (await async_client.get(f"/index/evaluations/{evaluation_id}")).raise_for_status().json()
    )

    assert content["status"] == "completed"

    assert len(content["invoices"]) == 1
    invoice = content["invoices"][0]

    assert invoice["approved"]
    assert len(invoice["data"]["guideline"]["coherence_checks"]) == 0


async def test_that_an_evaluation_can_be_fetched_with_a_completed_status_containing_a_detailed_unapproved_invoice(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    evaluation_id = (
        (
            await async_client.post(
                "/index/evaluations",
                json={
                    "agent_id": agent_id,
                    "payloads": [
                        {
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
                        {
                            "kind": "guideline",
                            "guideline": {
                                "content": {
                                    "condition": "the customer greeting you",
                                    "action": "greet them back with 'Good bye'",
                                },
                                "operation": "add",
                                "coherence_check": True,
                                "connection_proposition": True,
                            },
                        },
                    ],
                },
            )
        )
        .raise_for_status()
        .json()["id"]
    )

    content = (
        (await async_client.get(f"/index/evaluations/{evaluation_id}")).raise_for_status().json()
    )

    assert content["status"] == "completed"
    assert len(content["invoices"]) == 2
    first_invoice = content["invoices"][0]

    assert not first_invoice["approved"]

    assert len(first_invoice["data"]["guideline"]["coherence_checks"]) >= 1


async def test_that_an_evaluation_can_be_fetched_with_a_detailed_approved_invoice_with_existing_guideline_connection_proposition(
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
        guideline.id,
        Tag.for_agent_id(agent_id),
    )

    evaluation_id = (
        (
            await async_client.post(
                "/index/evaluations",
                json={
                    "agent_id": agent_id,
                    "payloads": [
                        {
                            "kind": "guideline",
                            "guideline": {
                                "content": {
                                    "condition": "providing the weather update",
                                    "action": "mention the best time to go for a walk",
                                },
                                "operation": "add",
                                "coherence_check": True,
                                "connection_proposition": True,
                            },
                        },
                    ],
                },
            )
        )
        .raise_for_status()
        .json()["id"]
    )

    content = (
        (await async_client.get(f"/index/evaluations/{evaluation_id}")).raise_for_status().json()
    )

    assert content["status"] == "completed"

    assert len(content["invoices"]) == 1
    invoice = content["invoices"][0]

    assert len(invoice["data"]["guideline"]["connection_propositions"]) == 1
    assert (
        invoice["data"]["guideline"]["connection_propositions"][0]["check_kind"]
        == "connection_with_existing_guideline"
    )

    assert (
        invoice["data"]["guideline"]["connection_propositions"][0]["source"]["condition"]
        == "the customer asks about the weather"
    )
    assert (
        invoice["data"]["guideline"]["connection_propositions"][0]["target"]["condition"]
        == "providing the weather update"
    )


async def test_that_an_evaluation_can_be_fetched_with_a_detailed_approved_invoice_with_other_proposed_guideline_connection_proposition(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    evaluation_id = (
        (
            await async_client.post(
                "/index/evaluations",
                json={
                    "agent_id": agent_id,
                    "payloads": [
                        {
                            "kind": "guideline",
                            "guideline": {
                                "content": {
                                    "condition": "the customer asks about nearby restaurants",
                                    "action": "provide a list of popular restaurants",
                                },
                                "operation": "add",
                                "coherence_check": True,
                                "connection_proposition": True,
                            },
                        },
                        {
                            "kind": "guideline",
                            "guideline": {
                                "content": {
                                    "condition": "listing restaurants",
                                    "action": "highlight the one with the best reviews",
                                },
                                "operation": "add",
                                "coherence_check": True,
                                "connection_proposition": True,
                            },
                        },
                    ],
                },
            )
        )
        .raise_for_status()
        .json()["id"]
    )

    content = (
        (await async_client.get(f"/index/evaluations/{evaluation_id}")).raise_for_status().json()
    )

    assert content["status"] == "completed"

    assert len(content["invoices"]) == 2
    first_invoice = content["invoices"][0]

    assert first_invoice["data"]
    assert len(first_invoice["data"]["guideline"]["connection_propositions"]) == 1
    assert (
        first_invoice["data"]["guideline"]["connection_propositions"][0]["check_kind"]
        == "connection_with_another_evaluated_guideline"
    )

    assert (
        first_invoice["data"]["guideline"]["connection_propositions"][0]["source"]["condition"]
        == "the customer asks about nearby restaurants"
    )
    assert (
        first_invoice["data"]["guideline"]["connection_propositions"][0]["target"]["condition"]
        == "listing restaurants"
    )


async def test_that_an_evaluation_that_failed_due_to_duplicate_guidelines_payloads_contains_a_relevant_error_message(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    duplicate_payload = {
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
    }

    response = await async_client.post(
        "/index/evaluations",
        json={
            "agent_id": agent_id,
            "payloads": [
                duplicate_payload,
                duplicate_payload,
            ],
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()

    assert "detail" in data
    assert data["detail"] == "Duplicate guideline found among the provided guidelines."


async def test_that_an_evaluation_that_failed_due_to_guideline_duplication_with_existing_guidelines_contains_a_relevant_error_message(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer greets you",
        action="greet them back with 'Hello'",
    )

    _ = await guideline_store.upsert_tag(
        guideline.id,
        Tag.for_agent_id(agent_id),
    )

    duplicate_payload = {
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
    }

    response = await async_client.post(
        "/index/evaluations",
        json={
            "agent_id": agent_id,
            "payloads": [
                duplicate_payload,
            ],
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()

    assert (
        data["detail"]
        == f"Duplicate guideline found against existing guidelines: When the customer greets you, then greet them back with 'Hello' in {agent_id} guideline_set"
    )


async def test_that_an_error_is_returned_when_no_payloads_are_provided(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    response = await async_client.post(
        "/index/evaluations", json={"agent_id": agent_id, "payloads": []}
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()

    assert "detail" in data
    assert data["detail"] == "No payloads provided for the evaluation task."


async def test_that_an_evaluation_task_fails_if_another_task_is_already_running(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
    no_cache: NoCachedGenerations,
) -> None:
    payloads = [
        {
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
        {
            "kind": "guideline",
            "guideline": {
                "content": {
                    "condition": "the customer asks about the weather",
                    "action": "provide a weather update",
                },
                "operation": "add",
                "coherence_check": True,
                "connection_proposition": True,
            },
        },
    ]

    first_evaluation_id = (
        (
            await async_client.post(
                "/index/evaluations", json={"agent_id": agent_id, "payloads": payloads}
            )
        )
        .raise_for_status()
        .json()["id"]
    )

    await asyncio.sleep(AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING)

    second_evaluation_id = (
        (
            await async_client.post(
                "/index/evaluations", json={"agent_id": agent_id, "payloads": payloads}
            )
        )
        .raise_for_status()
        .json()["id"]
    )

    content = (
        (
            await async_client.get(
                f"/index/evaluations/{second_evaluation_id}", params={"wait_for_completion": 0}
            )
        )
        .raise_for_status()
        .json()
    )

    assert content["status"] == "failed"
    assert content["error"] == f"An evaluation task '{first_evaluation_id}' is already running."


async def test_that_evaluation_task_with_payload_containing_contradictions_is_approved_when_check_flag_is_false(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    evaluation_id = (
        (
            await async_client.post(
                "/index/evaluations",
                json={
                    "agent_id": agent_id,
                    "payloads": [
                        {
                            "kind": "guideline",
                            "guideline": {
                                "content": {
                                    "condition": "the customer greets you",
                                    "action": "ignore the customer",
                                },
                                "operation": "add",
                                "coherence_check": False,
                                "connection_proposition": True,
                            },
                        },
                        {
                            "kind": "guideline",
                            "guideline": {
                                "content": {
                                    "condition": "the customer greets you",
                                    "action": "greet them back with 'Hello'",
                                },
                                "operation": "add",
                                "coherence_check": False,
                                "connection_proposition": True,
                            },
                        },
                    ],
                },
            )
        )
        .raise_for_status()
        .json()["id"]
    )

    content = (
        (await async_client.get(f"/index/evaluations/{evaluation_id}")).raise_for_status().json()
    )

    assert content["status"] == "completed"

    assert len(content["invoices"]) == 2

    for invoice in content["invoices"]:
        assert invoice["approved"]


async def test_that_evaluation_task_skips_proposing_guideline_connections_when_index_flag_is_false(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    evaluation_id = (
        (
            await async_client.post(
                "/index/evaluations",
                json={
                    "agent_id": agent_id,
                    "payloads": [
                        {
                            "kind": "guideline",
                            "guideline": {
                                "content": {
                                    "condition": "the customer asks for help",
                                    "action": "provide assistance",
                                },
                                "operation": "add",
                                "coherence_check": True,
                                "connection_proposition": False,
                            },
                        },
                        {
                            "kind": "guideline",
                            "guideline": {
                                "content": {
                                    "condition": "provide assistance",
                                    "action": "offer support resources",
                                },
                                "operation": "add",
                                "coherence_check": True,
                                "connection_proposition": False,
                            },
                        },
                    ],
                },
            )
        )
        .raise_for_status()
        .json()["id"]
    )

    content = (
        (await async_client.get(f"/index/evaluations/{evaluation_id}")).raise_for_status().json()
    )

    assert content["status"] == "completed"

    assert len(content["invoices"]) == 2

    assert content["invoices"][0]["data"]
    assert content["invoices"][0]["data"]["guideline"]["connection_propositions"] is None

    assert content["invoices"][1]["data"]
    assert content["invoices"][1]["data"]["guideline"]["connection_propositions"] is None


async def test_that_evaluation_task_with_contradictions_is_approved_and_skips_indexing_when_check_and_index_flags_are_false(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    evaluation_id = (
        (
            await async_client.post(
                "/index/evaluations",
                json={
                    "agent_id": agent_id,
                    "payloads": [
                        {
                            "kind": "guideline",
                            "guideline": {
                                "content": {
                                    "condition": "the customer says 'goodbye'",
                                    "action": "say 'farewell'",
                                },
                                "operation": "add",
                                "coherence_check": False,
                                "connection_proposition": False,
                            },
                        },
                        {
                            "kind": "guideline",
                            "guideline": {
                                "content": {
                                    "condition": "the customer says 'goodbye'",
                                    "action": "ignore the customer",
                                },
                                "operation": "add",
                                "coherence_check": False,
                                "connection_proposition": False,
                            },
                        },
                        {
                            "kind": "guideline",
                            "guideline": {
                                "content": {
                                    "condition": "ignoring the customer",
                                    "action": "say that your favorite pizza topping is pineapple.",
                                },
                                "operation": "add",
                                "coherence_check": False,
                                "connection_proposition": False,
                            },
                        },
                    ],
                },
            )
        )
        .raise_for_status()
        .json()["id"]
    )

    content = (
        (await async_client.get(f"/index/evaluations/{evaluation_id}")).raise_for_status().json()
    )

    assert content["status"] == "completed"

    assert len(content["invoices"]) == 3

    assert content["invoices"][0]["approved"]
    assert content["invoices"][0]["data"]
    assert content["invoices"][0]["data"]["guideline"]["coherence_checks"] == []
    assert content["invoices"][0]["data"]["guideline"]["connection_propositions"] is None

    assert content["invoices"][1]["approved"]
    assert content["invoices"][1]["data"]
    assert content["invoices"][1]["data"]["guideline"]["coherence_checks"] == []
    assert content["invoices"][1]["data"]["guideline"]["connection_propositions"] is None

    assert content["invoices"][2]["approved"]
    assert content["invoices"][2]["data"]
    assert content["invoices"][2]["data"]["guideline"]["coherence_checks"] == []
    assert content["invoices"][2]["data"]["guideline"]["connection_propositions"] is None


async def test_that_evaluation_fails_when_updated_id_does_not_exist(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    non_existent_guideline_id = "non-existent-id"

    evaluation_id = (
        (
            await async_client.post(
                "/index/evaluations",
                json={
                    "agent_id": agent_id,
                    "payloads": [
                        {
                            "kind": "guideline",
                            "guideline": {
                                "content": {
                                    "condition": "the customer greets you",
                                    "action": "greet them back with 'Hello'",
                                },
                                "operation": "update",
                                "updated_id": non_existent_guideline_id,
                                "coherence_check": True,
                                "connection_proposition": True,
                            },
                        }
                    ],
                },
            )
        )
        .raise_for_status()
        .json()["id"]
    )

    content = (
        (await async_client.get(f"/index/evaluations/{evaluation_id}")).raise_for_status().json()
    )

    assert content["status"] == "failed"

    assert (
        content["error"]
        == f"Guideline ID(s): {', '.join([non_existent_guideline_id])} in '{agent_id}' agent do not exist."
    )


async def test_that_evaluation_task_with_update_of_existing_guideline_is_approved(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    existing_guideline = await guideline_store.create_guideline(
        condition="the customer asks for help",
        action="provide assistance",
    )

    _ = await guideline_store.upsert_tag(
        existing_guideline.id,
        Tag.for_agent_id(agent_id),
    )

    update_payload = {
        "kind": "guideline",
        "guideline": {
            "content": {
                "condition": "the customer asks for help",
                "action": "provide updated assistance with additional resources",
            },
            "operation": "update",
            "updated_id": existing_guideline.id,
            "coherence_check": True,
            "connection_proposition": True,
        },
    }

    response = await async_client.post(
        "/index/evaluations",
        json={"agent_id": agent_id, "payloads": [update_payload]},
    )

    assert response.status_code == status.HTTP_201_CREATED
    evaluation_id = response.json()["id"]

    content = (
        (await async_client.get(f"/index/evaluations/{evaluation_id}")).raise_for_status().json()
    )

    assert content["status"] == "completed"

    assert len(content["invoices"]) == 1
    invoice = content["invoices"][0]

    assert invoice["approved"]
    assert invoice["data"]["guideline"]["coherence_checks"] == []


async def test_that_evaluation_task_with_update_of_existing_guideline_is_unapproved(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    guideline = await guideline_store.create_guideline(
        condition="the customer greets you",
        action="respond with 'Hello'",
    )

    _ = await guideline_store.upsert_tag(
        guideline.id,
        Tag.for_agent_id(agent_id),
    )

    guideline_to_override = await guideline_store.create_guideline(
        condition="the customer greets you",
        action="respond with 'Goodbye'",
    )

    _ = await guideline_store.upsert_tag(
        guideline_to_override.id,
        Tag.for_agent_id(agent_id),
    )

    update_payload = {
        "kind": "guideline",
        "guideline": {
            "content": {
                "condition": "the customer greets you",
                "action": "ignore the customer",
            },
            "operation": "update",
            "updated_id": guideline_to_override.id,
            "coherence_check": True,
            "connection_proposition": True,
        },
    }

    response = await async_client.post(
        "/index/evaluations",
        json={"agent_id": agent_id, "payloads": [update_payload]},
    )

    assert response.status_code == status.HTTP_201_CREATED
    evaluation_id = response.json()["id"]

    content = (
        (await async_client.get(f"/index/evaluations/{evaluation_id}")).raise_for_status().json()
    )

    assert content["status"] == "completed"

    assert len(content["invoices"]) == 1
    invoice = content["invoices"][0]

    assert not invoice["approved"]
    assert len(invoice["data"]["guideline"]["coherence_checks"]) > 0


async def test_that_evaluation_task_with_conflicting_guidelines_approves_only_payload_with_the_coherence_check_disabled(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    payloads = [
        {
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
        {
            "kind": "guideline",
            "guideline": {
                "content": {
                    "condition": "the customer greeting you",
                    "action": "greet them back with 'Good bye'",
                },
                "operation": "add",
                "coherence_check": False,
                "connection_proposition": True,
            },
        },
    ]

    response = await async_client.post(
        "/index/evaluations",
        json={"agent_id": agent_id, "payloads": payloads},
    )

    assert response.status_code == status.HTTP_201_CREATED
    evaluation_id = response.json()["id"]

    content = (
        (await async_client.get(f"/index/evaluations/{evaluation_id}")).raise_for_status().json()
    )

    assert content["status"] == "completed"

    invoices = content["invoices"]
    assert len(invoices) == 2

    assert any(
        i["payload"]["guideline"]["content"]["condition"] == "the customer greets you"
        and i["payload"]["guideline"]["content"]["action"] == "greet them back with 'Hello'"
        and not i["approved"]
        for i in invoices
    )

    assert any(
        i["payload"]["guideline"]["content"]["condition"] == "the customer greeting you"
        and i["payload"]["guideline"]["content"]["action"] == "greet them back with 'Good bye'"
        and i["approved"]
        for i in invoices
    )


async def test_that_evaluation_task_with_connected_guidelines_only_includes_details_for_enabled_connection_proposition(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    payloads = [
        {
            "kind": "guideline",
            "guideline": {
                "content": {
                    "condition": "the customer asks about the weather",
                    "action": "provide the current weather update",
                },
                "operation": "add",
                "coherence_check": True,
                "connection_proposition": True,
            },
        },
        {
            "kind": "guideline",
            "guideline": {
                "content": {
                    "condition": "providing the weather update",
                    "action": "mention the best time to go for a walk",
                },
                "operation": "add",
                "coherence_check": True,
                "connection_proposition": False,
            },
        },
    ]

    evaluation_id = (
        (
            await async_client.post(
                "/index/evaluations",
                json={"agent_id": agent_id, "payloads": payloads},
            )
        )
        .raise_for_status()
        .json()["id"]
    )

    content = (
        (await async_client.get(f"/index/evaluations/{evaluation_id}")).raise_for_status().json()
    )

    assert content["status"] == "completed"

    invoices = content["invoices"]
    assert len(invoices) == 2

    assert any(
        i["payload"]["guideline"]["content"]["condition"] == "the customer asks about the weather"
        and i["payload"]["guideline"]["content"]["action"] == "provide the current weather update"
        and i["approved"]
        and len(i["data"]["guideline"]["connection_propositions"]) > 0
        for i in invoices
    )

    assert any(
        i["payload"]["guideline"]["content"]["condition"] == "providing the weather update"
        and i["payload"]["guideline"]["content"]["action"]
        == "mention the best time to go for a walk"
        and i["approved"]
        and i["data"]["guideline"]["connection_propositions"] is None
        for i in invoices
    )


async def test_that_evaluation_task_with_conflicting_updated_and_added_guidelines_is_unapproved(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    existing_guideline = await guideline_store.create_guideline(
        condition="the customer greets you",
        action="reply with 'Hello'",
    )

    _ = await guideline_store.upsert_tag(
        existing_guideline.id,
        Tag.for_agent_id(agent_id),
    )

    updated_guideline_content = {
        "condition": "the customer greets you",
        "action": "reply with 'Howdy!'",
    }
    added_guideline_content = {
        "condition": "the customer greets you",
        "action": "reply with 'Goodbye!'",
    }

    payloads = [
        {
            "kind": "guideline",
            "guideline": {
                "content": updated_guideline_content,
                "operation": "update",
                "updated_id": existing_guideline.id,
                "coherence_check": True,
                "connection_proposition": True,
            },
        },
        {
            "kind": "guideline",
            "guideline": {
                "content": added_guideline_content,
                "operation": "add",
                "coherence_check": True,
                "connection_proposition": True,
            },
        },
    ]

    evaluation_id = (
        (
            await async_client.post(
                "/index/evaluations",
                json={"agent_id": agent_id, "payloads": payloads},
            )
        )
        .raise_for_status()
        .json()["id"]
    )

    content = (
        (await async_client.get(f"/index/evaluations/{evaluation_id}")).raise_for_status().json()
    )

    assert content["status"] == "completed"

    invoices = content["invoices"]
    assert len(invoices) == 2

    updated_invoice = next(
        (i for i in invoices if i["payload"]["guideline"]["operation"] == "update"), None
    )
    new_invoice = next(
        (i for i in invoices if i["payload"]["guideline"]["operation"] == "add"), None
    )

    assert updated_invoice is not None
    assert new_invoice is not None

    assert len(updated_invoice["data"]["guideline"]["coherence_checks"]) == 1
    conflict = updated_invoice["data"]["guideline"]["coherence_checks"][0]

    assert conflict["kind"] == "contradiction_with_another_evaluated_guideline"

    assert (
        conflict["first"] == updated_invoice["payload"]["guideline"]["content"]
        or new_invoice["payload"]["guideline"]["content"]
    )
    assert (
        conflict["second"] == updated_invoice["payload"]["guideline"]["content"]
        or new_invoice["payload"]["guideline"]["content"]
    )

    assert conflict["first"] != conflict["second"]
