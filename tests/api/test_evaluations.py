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

import asyncio
from lagom import Container
from fastapi import status
import httpx


from parlant.core.services.tools.plugins import tool
from parlant.core.tools import ToolResult, ToolContext
from parlant.core.services.tools.service_registry import ServiceRegistry

from tests.test_utilities import run_service_server

AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING = 2


async def test_that_an_evaluation_can_be_created_and_fetched_with_completed_status(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/evaluations",
        json={
            "payloads": [
                {
                    "kind": "guideline",
                    "guideline": {
                        "content": {
                            "condition": "the customer greets you",
                            "action": "greet them back with 'Hello'",
                        },
                        "tool_ids": [
                            {"service_name": "google_calendar", "tool_name": "get_events"}
                        ],
                        "operation": "add",
                        "action_proposition": True,
                        "properties_proposition": True,
                    },
                }
            ],
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    evaluation_id = response.raise_for_status().json()["id"]

    content = (await async_client.get(f"/evaluations/{evaluation_id}")).raise_for_status().json()

    assert content["status"] == "completed"
    assert len(content["invoices"]) == 1

    invoice = content["invoices"][0]
    assert invoice["approved"]

    assert invoice["data"]
    assert invoice["data"]["guideline"]["properties_proposition"]["internal_action"] is None


async def test_that_an_evaluation_can_be_fetched_with_running_status(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/evaluations",
        json={
            "payloads": [
                {
                    "kind": "guideline",
                    "guideline": {
                        "content": {
                            "condition": "the customer greets you",
                            "action": "greet them back with 'Hello'",
                        },
                        "operation": "add",
                        "action_proposition": True,
                        "properties_proposition": True,
                        "tool_ids": [
                            {"service_name": "google_calendar", "tool_name": "get_events"}
                        ],
                    },
                }
            ],
        },
    )

    evaluation_id = response.raise_for_status().json()["id"]

    await asyncio.sleep(AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING)

    content = (
        (await async_client.get(f"/evaluations/{evaluation_id}", params={"wait_for_completion": 0}))
        .raise_for_status()
        .json()
    )

    assert content["status"] in {"running", "completed"}


async def test_that_an_error_is_returned_when_no_payloads_are_provided(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post("/evaluations", json={"payloads": []})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    data = response.json()

    assert "detail" in data
    assert data["detail"] == "No payloads provided for the evaluation task."


async def test_that_properties_proposition_is_evaluated(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/evaluations",
        json={
            "payloads": [
                {
                    "kind": "guideline",
                    "guideline": {
                        "content": {
                            "condition": "the customer asks for a discount",
                            "action": "maintain a helpful tone and ask the customer what discount they would like",
                        },
                        "operation": "add",
                        "action_proposition": True,
                        "properties_proposition": True,
                        "tool_ids": [
                            {"service_name": "google_calendar", "tool_name": "get_events"}
                        ],
                    },
                }
            ],
        },
    )
    assert response.status_code == status.HTTP_201_CREATED

    evaluation_id = response.raise_for_status().json()["id"]

    content = (await async_client.get(f"/evaluations/{evaluation_id}")).raise_for_status().json()

    assert content["status"] == "completed"
    assert len(content["invoices"]) == 1

    invoice = content["invoices"][0]
    assert invoice["approved"]

    assert invoice["data"]
    assert invoice["data"]["guideline"]["properties_proposition"]["continuous"]
    assert invoice["data"]["guideline"]["properties_proposition"]["customer_dependent_action_data"][
        "is_customer_dependent"
    ]
    assert invoice["data"]["guideline"]["properties_proposition"]["customer_dependent_action_data"][
        "customer_action"
    ]
    assert invoice["data"]["guideline"]["properties_proposition"]["customer_dependent_action_data"][
        "agent_action"
    ]


async def test_that_action_proposition_is_evaluated(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: int) -> ToolResult:
        return ToolResult(arg_1 + arg_2)

    service_registry = container[ServiceRegistry]

    async with run_service_server([my_tool]) as server:
        await service_registry.update_tool_service(
            name="my_service",
            kind="sdk",
            url=server.url,
        )

        response = await async_client.post(
            "/evaluations",
            json={
                "payloads": [
                    {
                        "kind": "guideline",
                        "guideline": {
                            "content": {
                                "condition": "the customer asks for a discount",
                            },
                            "tool_ids": [{"service_name": "my_service", "tool_name": "my_tool"}],
                            "operation": "add",
                            "action_proposition": True,
                            "properties_proposition": False,
                        },
                    }
                ],
            },
        )

        assert response.status_code == status.HTTP_201_CREATED

        evaluation_id = response.raise_for_status().json()["id"]

        content = (
            (await async_client.get(f"/evaluations/{evaluation_id}")).raise_for_status().json()
        )

        assert content["status"] == "completed"
        assert len(content["invoices"]) == 1

        invoice = content["invoices"][0]
        assert invoice["approved"]

        assert invoice["data"]
        assert isinstance(invoice["data"]["guideline"]["properties_proposition"], dict)
        assert (
            invoice["data"]["guideline"]["properties_proposition"].get("internal_action")
            is not None
        )


async def test_that_error_is_returned_when_no_propositions_are_provided_in_a_payload(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/evaluations",
        json={
            "payloads": [
                {
                    "kind": "guideline",
                    "guideline": {
                        "content": {
                            "condition": "the customer greets you",
                            "action": "greet them back with 'Hello'",
                        },
                        "tool_ids": [
                            {"service_name": "google_calendar", "tool_name": "get_events"}
                        ],
                        "operation": "add",
                    },
                }
            ],
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    data = response.json()

    assert "detail" in data
    assert (
        data["detail"]
        == "At least one of action_proposition, properties_proposition or journey_node_proposition must be enabled"
    )


async def test_that_error_is_returned_when_all_propositions_are_disabled_in_a_payload(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/evaluations",
        json={
            "payloads": [
                {
                    "kind": "guideline",
                    "guideline": {
                        "content": {
                            "condition": "the customer greets you",
                            "action": "greet them back with 'Hello'",
                        },
                        "tool_ids": [
                            {"service_name": "google_calendar", "tool_name": "get_events"}
                        ],
                        "operation": "add",
                        "action_proposition": False,
                        "properties_proposition": False,
                    },
                }
            ],
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    data = response.json()

    assert "detail" in data
    assert (
        data["detail"]
        == "At least one of action_proposition, properties_proposition or journey_node_proposition must be enabled"
    )
