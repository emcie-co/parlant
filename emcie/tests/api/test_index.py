import asyncio
from fastapi.testclient import TestClient
from fastapi import status
from lagom import Container

from emcie.server.core.agents import AgentId
from emcie.server.core.evaluations import EvaluationStore
from emcie.server.core.guidelines import GuidelineStore

from tests.core.services.indexing.test_evaluator import (
    AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING,
    TIME_TO_WAIT_PER_PAYLOAD,
)


async def test_that_an_evaluation_can_be_created_and_fetched_with_completed_status(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    evaluation_store = container[EvaluationStore]

    response = client.post(
        f"/agents/{agent_id}/index/evaluations",
        json={
            "payloads": [
                {
                    "kind": "guideline",
                    "content": {
                        "predicate": "the user greets you",
                        "action": "greet them back with 'Hello'",
                    },
                    "operation": "add",
                    "coherence_check": True,
                    "connection_proposition": True,
                }
            ],
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    evaluation_id = response.json()["evaluation_id"]

    evaluation = await evaluation_store.read_evaluation(evaluation_id=evaluation_id)
    assert evaluation.id == evaluation_id

    await asyncio.sleep(TIME_TO_WAIT_PER_PAYLOAD)

    content = client.get(f"/agents/index/evaluations/{evaluation_id}").raise_for_status().json()

    assert content["status"] == "completed"
    assert len(content["invoices"]) == 1
    invoice = content["invoices"][0]

    assert invoice["approved"]
    assert len(invoice["data"]["coherence_checks"]) == 0


async def test_that_an_evaluation_can_be_fetched_with_running_status(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    evaluation_id = (
        client.post(
            f"/agents/{agent_id}/index/evaluations",
            json={
                "payloads": [
                    {
                        "kind": "guideline",
                        "content": {
                            "predicate": "the user greets you",
                            "action": "greet them back with 'Hello'",
                        },
                        "operation": "add",
                        "coherence_check": True,
                        "connection_proposition": True,
                    },
                    {
                        "kind": "guideline",
                        "content": {
                            "predicate": "the user greeting you",
                            "action": "greet them back with 'Hola'",
                        },
                        "operation": "add",
                        "coherence_check": True,
                        "connection_proposition": True,
                    },
                ]
            },
        )
        .raise_for_status()
        .json()["evaluation_id"]
    )

    await asyncio.sleep(AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING)

    content = client.get(f"/agents/index/evaluations/{evaluation_id}").raise_for_status().json()

    assert content["status"] == "running"


async def test_that_an_evaluation_can_be_fetched_with_a_completed_status_containing_a_detailed_approved_invoice(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    evaluation_id = (
        client.post(
            f"/agents/{agent_id}/index/evaluations",
            json={
                "payloads": [
                    {
                        "kind": "guideline",
                        "content": {
                            "predicate": "the user greets you",
                            "action": "greet them back with 'Hello'",
                        },
                        "operation": "add",
                        "coherence_check": True,
                        "connection_proposition": True,
                    }
                ],
            },
        )
        .raise_for_status()
        .json()["evaluation_id"]
    )

    await asyncio.sleep(AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING)

    content = client.get(f"/agents/index/evaluations/{evaluation_id}").raise_for_status().json()

    assert content["status"] == "completed"

    assert len(content["invoices"]) == 1
    invoice = content["invoices"][0]

    assert invoice["approved"]
    assert len(invoice["data"]["coherence_checks"]) == 0


async def test_that_an_evaluation_can_be_fetched_with_a_completed_status_containing_a_detailed_unapproved_invoice(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    evaluation_id = (
        client.post(
            f"/agents/{agent_id}/index/evaluations",
            json={
                "payloads": [
                    {
                        "kind": "guideline",
                        "content": {
                            "predicate": "the user greets you",
                            "action": "greet them back with 'Hello'",
                        },
                        "operation": "add",
                        "coherence_check": True,
                        "connection_proposition": True,
                    },
                    {
                        "kind": "guideline",
                        "content": {
                            "predicate": "the user greeting you",
                            "action": "greet them back with 'Good bye'",
                        },
                        "operation": "add",
                        "coherence_check": True,
                        "connection_proposition": True,
                    },
                ],
            },
        )
        .raise_for_status()
        .json()["evaluation_id"]
    )

    await asyncio.sleep(TIME_TO_WAIT_PER_PAYLOAD * 2)

    content = client.get(f"/agents/index/evaluations/{evaluation_id}").raise_for_status().json()

    assert content["status"] == "completed"

    assert len(content["invoices"]) == 2
    first_invoice = content["invoices"][0]

    assert not first_invoice["approved"]

    assert len(first_invoice["data"]["coherence_checks"]) >= 1


async def test_that_an_evaluation_can_be_fetched_with_a_detailed_approved_invoice_with_existing_guideline_connection_proposition(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    await guideline_store.create_guideline(
        guideline_set=agent_id,
        predicate="the user asks about the weather",
        action="provide the current weather update",
    )

    evaluation_id = (
        client.post(
            f"/agents/{agent_id}/index/evaluations",
            json={
                "payloads": [
                    {
                        "kind": "guideline",
                        "content": {
                            "predicate": "providing the weather update",
                            "action": "mention the best time to go for a walk",
                        },
                        "operation": "add",
                        "coherence_check": True,
                        "connection_proposition": True,
                    }
                ],
            },
        )
        .raise_for_status()
        .json()["evaluation_id"]
    )

    await asyncio.sleep(TIME_TO_WAIT_PER_PAYLOAD * 2)

    content = client.get(f"/agents/index/evaluations/{evaluation_id}").raise_for_status().json()

    assert len(content["invoices"]) == 1
    invoice = content["invoices"][0]

    assert len(invoice["data"]["connection_propositions"]) == 1
    assert (
        invoice["data"]["connection_propositions"][0]["check_kind"]
        == "connection_with_existing_guideline"
    )

    assert (
        invoice["data"]["connection_propositions"][0]["source"]["predicate"]
        == "the user asks about the weather"
    )
    assert (
        invoice["data"]["connection_propositions"][0]["target"]["predicate"]
        == "providing the weather update"
    )


async def test_that_an_evaluation_can_be_fetched_with_a_detailed_approved_invoice_with_other_proposed_guideline_connection_proposition(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    evaluation_id = (
        client.post(
            f"/agents/{agent_id}/index/evaluations",
            json={
                "payloads": [
                    {
                        "kind": "guideline",
                        "content": {
                            "predicate": "the user asks about nearby restaurants",
                            "action": "provide a list of popular restaurants",
                        },
                        "operation": "add",
                        "coherence_check": True,
                        "connection_proposition": True,
                    },
                    {
                        "kind": "guideline",
                        "content": {
                            "predicate": "listing restaurants",
                            "action": "highlight the one with the best reviews",
                        },
                        "operation": "add",
                        "coherence_check": True,
                        "connection_proposition": True,
                    },
                ],
            },
        )
        .raise_for_status()
        .json()["evaluation_id"]
    )

    await asyncio.sleep(TIME_TO_WAIT_PER_PAYLOAD * 2)

    content = client.get(f"/agents/index/evaluations/{evaluation_id}").raise_for_status().json()

    assert content["status"] == "completed"

    assert len(content["invoices"]) == 2
    first_invoice = content["invoices"][0]

    assert first_invoice["data"]
    assert len(first_invoice["data"]["connection_propositions"]) == 1
    assert (
        first_invoice["data"]["connection_propositions"][0]["check_kind"]
        == "connection_with_another_evaluated_guideline"
    )

    assert (
        first_invoice["data"]["connection_propositions"][0]["source"]["predicate"]
        == "the user asks about nearby restaurants"
    )
    assert (
        first_invoice["data"]["connection_propositions"][0]["target"]["predicate"]
        == "listing restaurants"
    )


async def test_that_an_evaluation_that_failed_due_to_duplicate_guidelines_payloads_contains_a_relevant_error_message(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    duplicate_payload = {
        "kind": "guideline",
        "content": {
            "predicate": "the user greets you",
            "action": "greet them back with 'Hello'",
        },
        "operation": "add",
        "coherence_check": True,
        "connection_proposition": True,
    }

    response = client.post(
        f"/agents/{agent_id}/index/evaluations",
        json={
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
    client: TestClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    await guideline_store.create_guideline(
        guideline_set=agent_id,
        predicate="the user greets you",
        action="greet them back with 'Hello'",
    )

    duplicate_payload = {
        "kind": "guideline",
        "content": {
            "predicate": "the user greets you",
            "action": "greet them back with 'Hello'",
        },
        "operation": "add",
        "coherence_check": True,
        "connection_proposition": True,
    }

    response = client.post(
        f"/agents/{agent_id}/index/evaluations",
        json={
            "payloads": [
                duplicate_payload,
            ],
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()

    assert (
        data["detail"]
        == f"Duplicate guideline found against existing guidelines: When the user greets you, then greet them back with 'Hello' in {agent_id} guideline_set"
    )


async def test_that_an_error_is_returned_when_no_payloads_are_provided(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    response = client.post(f"/agents/{agent_id}/index/evaluations", json={"payloads": []})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()

    assert "detail" in data
    assert data["detail"] == "No payloads provided for the evaluation task."


async def test_that_an_evaluation_task_fails_if_another_task_is_already_running(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    payloads = {
        "payloads": [
            {
                "kind": "guideline",
                "content": {
                    "predicate": "the user greets you",
                    "action": "greet them back with 'Hello'",
                },
                "operation": "add",
                "coherence_check": True,
                "connection_proposition": True,
            },
            {
                "kind": "guideline",
                "content": {
                    "predicate": "the user asks about the weather",
                    "action": "provide a weather update",
                },
                "operation": "add",
                "coherence_check": True,
                "connection_proposition": True,
            },
        ],
    }

    first_evaluation_id = (
        client.post(f"/agents/{agent_id}/index/evaluations", json=payloads)
        .raise_for_status()
        .json()["evaluation_id"]
    )

    await asyncio.sleep(AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING)

    second_evaluation_id = (
        client.post(f"/agents/{agent_id}/index/evaluations", json=payloads)
        .raise_for_status()
        .json()["evaluation_id"]
    )

    content = (
        client.get(f"/agents/index/evaluations/{second_evaluation_id}").raise_for_status().json()
    )

    assert content["status"] == "failed"
    assert content["error"] == f"An evaluation task '{first_evaluation_id}' is already running."


async def test_that_evaluation_task_with_payload_containing_contradictions_is_approved_when_check_flag_is_false(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    evaluation_id = (
        client.post(
            f"/agents/{agent_id}/index/evaluations",
            json={
                "payloads": [
                    {
                        "kind": "guideline",
                        "content": {
                            "predicate": "the user greets you",
                            "action": "ignore the user",
                        },
                        "operation": "add",
                        "coherence_check": False,
                        "connection_proposition": True,
                    },
                    {
                        "kind": "guideline",
                        "content": {
                            "predicate": "the user greets you",
                            "action": "greet them back with 'Hello'",
                        },
                        "operation": "add",
                        "coherence_check": False,
                        "connection_proposition": True,
                    },
                ],
            },
        )
        .raise_for_status()
        .json()["evaluation_id"]
    )

    await asyncio.sleep(TIME_TO_WAIT_PER_PAYLOAD * 2)

    content = client.get(f"/agents/index/evaluations/{evaluation_id}").raise_for_status().json()

    assert content["status"] == "completed"
    assert len(content["invoices"]) == 2

    for invoice in content["invoices"]:
        assert invoice["approved"]


async def test_that_evaluation_task_skips_proposing_guideline_connections_when_index_flag_is_false(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    evaluation_id = (
        client.post(
            f"/agents/{agent_id}/index/evaluations",
            json={
                "payloads": [
                    {
                        "kind": "guideline",
                        "content": {
                            "predicate": "the user asks for help",
                            "action": "provide assistance",
                        },
                        "operation": "add",
                        "coherence_check": True,
                        "connection_proposition": False,
                    },
                    {
                        "kind": "guideline",
                        "content": {
                            "predicate": "provide assistance",
                            "action": "offer support resources",
                        },
                        "operation": "add",
                        "coherence_check": True,
                        "connection_proposition": False,
                    },
                ],
            },
        )
        .raise_for_status()
        .json()["evaluation_id"]
    )

    await asyncio.sleep(TIME_TO_WAIT_PER_PAYLOAD * 2)

    content = client.get(f"/agents/index/evaluations/{evaluation_id}").raise_for_status().json()
    assert content["status"] == "completed"

    assert len(content["invoices"]) == 2

    assert content["invoices"][0]["data"]
    assert content["invoices"][0]["data"]["connection_propositions"] is None

    assert content["invoices"][1]["data"]
    assert content["invoices"][1]["data"]["connection_propositions"] is None


async def test_that_evaluation_task_with_contradictions_is_approved_and_skips_indexing_when_check_and_index_flags_are_false(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    evaluation_id = (
        client.post(
            f"/agents/{agent_id}/index/evaluations",
            json={
                "payloads": [
                    {
                        "kind": "guideline",
                        "content": {
                            "predicate": "the user says 'goodbye'",
                            "action": "say 'farewell'",
                        },
                        "operation": "add",
                        "coherence_check": False,
                        "connection_proposition": False,
                    },
                    {
                        "kind": "guideline",
                        "content": {
                            "predicate": "the user says 'goodbye'",
                            "action": "ignore the user",
                        },
                        "operation": "add",
                        "coherence_check": False,
                        "connection_proposition": False,
                    },
                    {
                        "kind": "guideline",
                        "content": {
                            "predicate": "ignoring the user",
                            "action": "say that your favorite pizza topping is pineapple.",
                        },
                        "operation": "add",
                        "coherence_check": False,
                        "connection_proposition": False,
                    },
                ],
            },
        )
        .raise_for_status()
        .json()["evaluation_id"]
    )

    await asyncio.sleep(TIME_TO_WAIT_PER_PAYLOAD * 2)

    content = client.get(f"/agents/index/evaluations/{evaluation_id}").raise_for_status().json()
    assert content["status"] == "completed"

    assert len(content["invoices"]) == 3

    assert content["invoices"][0]["approved"]
    assert content["invoices"][0]["data"]
    assert content["invoices"][0]["data"]["coherence_checks"] == []
    assert content["invoices"][0]["data"]["connection_propositions"] is None

    assert content["invoices"][1]["approved"]
    assert content["invoices"][1]["data"]
    assert content["invoices"][1]["data"]["coherence_checks"] == []
    assert content["invoices"][1]["data"]["connection_propositions"] is None

    assert content["invoices"][2]["approved"]
    assert content["invoices"][2]["data"]
    assert content["invoices"][2]["data"]["coherence_checks"] == []
    assert content["invoices"][2]["data"]["connection_propositions"] is None


async def test_that_evaluation_fails_when_updated_id_does_not_exist(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    non_existent_guideline_id = "non-existent-id"

    evaluation_id = (
        client.post(
            f"/agents/{agent_id}/index/evaluations",
            json={
                "payloads": [
                    {
                        "kind": "guideline",
                        "content": {
                            "predicate": "the user greets you",
                            "action": "greet them back with 'Hello'",
                        },
                        "operation": "update",
                        "updated_id": non_existent_guideline_id,
                        "coherence_check": True,
                        "connection_proposition": True,
                    }
                ],
            },
        )
        .raise_for_status()
        .json()["evaluation_id"]
    )

    await asyncio.sleep(TIME_TO_WAIT_PER_PAYLOAD)

    agent_name = content = client.get(f"/agents/{agent_id}").raise_for_status().json()["name"]
    content = client.get(f"/agents/index/evaluations/{evaluation_id}").raise_for_status().json()

    assert content["status"] == "failed"
    assert (
        content["error"]
        == f"Guideline ID(s): {', '.join([non_existent_guideline_id])} in {agent_name} agent do not exist."
    )


async def test_that_evaluation_task_with_update_of_existing_guideline_is_approved(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    existing_guideline = await guideline_store.create_guideline(
        guideline_set=agent_id,
        predicate="the user asks for help",
        action="provide assistance",
    )

    update_payload = {
        "kind": "guideline",
        "content": {
            "predicate": "the user asks for help",
            "action": "provide updated assistance with additional resources",
        },
        "operation": "update",
        "updated_id": existing_guideline.id,
        "coherence_check": True,
        "connection_proposition": True,
    }

    response = client.post(
        f"/agents/{agent_id}/index/evaluations",
        json={"payloads": [update_payload]},
    )

    assert response.status_code == status.HTTP_201_CREATED
    evaluation_id = response.json()["evaluation_id"]

    await asyncio.sleep(TIME_TO_WAIT_PER_PAYLOAD)

    content = client.get(f"/agents/index/evaluations/{evaluation_id}").raise_for_status().json()

    assert content["status"] == "completed"

    assert len(content["invoices"]) == 1
    invoice = content["invoices"][0]

    assert invoice["approved"]
    assert invoice["data"]["coherence_checks"] == []


async def test_that_evaluation_task_with_update_of_existing_guideline_is_unapproved(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    _ = await guideline_store.create_guideline(
        guideline_set=agent_id,
        predicate="the user greets you",
        action="respond with 'Hello'",
    )

    guideline_to_override = await guideline_store.create_guideline(
        guideline_set=agent_id,
        predicate="the user greets you",
        action="respond with 'Goodbye'",
    )

    update_payload = {
        "kind": "guideline",
        "content": {
            "predicate": "the user greets you",
            "action": "ignore the user",
        },
        "operation": "update",
        "updated_id": guideline_to_override.id,
        "coherence_check": True,
        "connection_proposition": True,
    }

    response = client.post(
        f"/agents/{agent_id}/index/evaluations",
        json={"payloads": [update_payload]},
    )

    assert response.status_code == status.HTTP_201_CREATED
    evaluation_id = response.json()["evaluation_id"]

    await asyncio.sleep(TIME_TO_WAIT_PER_PAYLOAD)

    content = client.get(f"/agents/index/evaluations/{evaluation_id}").raise_for_status().json()

    assert content["status"] == "completed"

    assert len(content["invoices"]) == 1
    invoice = content["invoices"][0]

    assert not invoice["approved"]
    assert len(invoice["data"]["coherence_checks"]) > 0


async def test_that_evaluation_task_with_conflicting_guidelines_approves_only_payload_with_the_coherence_check_disabled(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    payloads = [
        {
            "kind": "guideline",
            "content": {
                "predicate": "the user greets you",
                "action": "greet them back with 'Hello'",
            },
            "operation": "add",
            "coherence_check": True,
            "connection_proposition": True,
        },
        {
            "kind": "guideline",
            "content": {
                "predicate": "the user greeting you",
                "action": "greet them back with 'Good bye'",
            },
            "operation": "add",
            "coherence_check": False,
            "connection_proposition": True,
        },
    ]

    response = client.post(
        f"/agents/{agent_id}/index/evaluations",
        json={"payloads": payloads},
    )

    assert response.status_code == status.HTTP_201_CREATED
    evaluation_id = response.json()["evaluation_id"]

    await asyncio.sleep(TIME_TO_WAIT_PER_PAYLOAD * 2)

    content = client.get(f"/agents/index/evaluations/{evaluation_id}").raise_for_status().json()

    assert content["status"] == "completed"

    invoices = content["invoices"]
    assert len(invoices) == 2

    assert any(
        i["payload"]["content"]["predicate"] == "the user greets you"
        and i["payload"]["content"]["action"] == "greet them back with 'Hello'"
        and not i["approved"]
        for i in invoices
    )

    assert any(
        i["payload"]["content"]["predicate"] == "the user greeting you"
        and i["payload"]["content"]["action"] == "greet them back with 'Good bye'"
        and i["approved"]
        for i in invoices
    )


async def test_that_evaluation_task_with_connected_guidelines_only_includes_details_for_enabled_connection_proposition(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    payloads = [
        {
            "kind": "guideline",
            "content": {
                "predicate": "the user asks about the weather",
                "action": "provide the current weather update",
            },
            "operation": "add",
            "coherence_check": True,
            "connection_proposition": True,
        },
        {
            "kind": "guideline",
            "content": {
                "predicate": "providing the weather update",
                "action": "mention the best time to go for a walk",
            },
            "operation": "add",
            "coherence_check": True,
            "connection_proposition": False,
        },
    ]

    evaluation_id = (
        client.post(
            f"/agents/{agent_id}/index/evaluations",
            json={"payloads": payloads},
        )
        .raise_for_status()
        .json()["evaluation_id"]
    )

    await asyncio.sleep(TIME_TO_WAIT_PER_PAYLOAD * 2)

    content = client.get(f"/agents/index/evaluations/{evaluation_id}").raise_for_status().json()

    assert content["status"] == "completed"

    invoices = content["invoices"]
    assert len(invoices) == 2

    assert any(
        i["payload"]["content"]["predicate"] == "the user asks about the weather"
        and i["payload"]["content"]["action"] == "provide the current weather update"
        and i["approved"]
        and len(i["data"]["connection_propositions"]) > 0
        for i in invoices
    )

    assert any(
        i["payload"]["content"]["predicate"] == "providing the weather update"
        and i["payload"]["content"]["action"] == "mention the best time to go for a walk"
        and i["approved"]
        and i["data"]["connection_propositions"] is None
        for i in invoices
    )


async def test_that_evaluation_task_with_conflicting_updated_and_added_guidelines_is_unapproved(
    client: TestClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    guideline_store = container[GuidelineStore]

    existing_guideline = await guideline_store.create_guideline(
        guideline_set=agent_id,
        predicate="the user greets you",
        action="reply with 'Hello'",
    )

    updated_guideline_content = {
        "predicate": "the user greets you",
        "action": "reply with 'Howdy!'",
    }
    added_guideline_content = {
        "predicate": "the user greets you",
        "action": "reply with 'Goodbye!'",
    }

    payloads = [
        {
            "kind": "guideline",
            "content": updated_guideline_content,
            "operation": "update",
            "updated_id": existing_guideline.id,
            "coherence_check": True,
            "connection_proposition": True,
        },
        {
            "kind": "guideline",
            "content": added_guideline_content,
            "operation": "add",
            "coherence_check": True,
            "connection_proposition": True,
        },
    ]

    evaluation_id = (
        client.post(
            f"/agents/{agent_id}/index/evaluations",
            json={"payloads": payloads},
        )
        .raise_for_status()
        .json()["evaluation_id"]
    )

    await asyncio.sleep(TIME_TO_WAIT_PER_PAYLOAD * 2)

    content = client.get(f"/agents/index/evaluations/{evaluation_id}").raise_for_status().json()
    assert content["status"] == "completed"

    invoices = content["invoices"]
    assert len(invoices) == 2

    updated_invoice = next((i for i in invoices if i["payload"]["operation"] == "update"), None)
    new_invoice = next((i for i in invoices if i["payload"]["operation"] == "add"), None)

    assert updated_invoice is not None
    assert new_invoice is not None

    assert len(updated_invoice["data"]["coherence_checks"]) == 1
    conflict = updated_invoice["data"]["coherence_checks"][0]

    assert conflict["kind"] == "contradiction_with_another_evaluated_guideline"

    assert (
        conflict["first"] == updated_invoice["payload"]["content"]
        or new_invoice["payload"]["content"]
    )
    assert (
        conflict["second"] == updated_invoice["payload"]["content"]
        or new_invoice["payload"]["content"]
    )

    assert conflict["first"] != conflict["second"]
