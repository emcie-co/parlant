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


from tests.core.stable.services.indexing.test_evaluator import (
    AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING,
)


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
                        "operation": "add",
                        "action_proposition": True,
                    },
                }
            ],
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    evaluation_id = response.json()["id"]

    content = (
        (await async_client.get(f"/evaluations/evaluations/{evaluation_id}"))
        .raise_for_status()
        .json()
    )

    assert content["status"] == "completed"
    assert len(content["invoices"]) == 1

    invoice = content["invoices"][0]
    assert invoice["approved"]

    assert invoice["data"]
    assert invoice["data"]["guideline"]["action_proposition"] == "greet them back with 'Hello'"


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
                    },
                }
            ],
        },
    )

    evaluation_id = response.json()["id"]

    await asyncio.sleep(AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING)

    content = (
        (
            await async_client.get(
                f"/evaluations/evaluations/{evaluation_id}", params={"wait_for_completion": 0}
            )
        )
        .raise_for_status()
        .json()
    )

    assert content["status"] in {"running", "completed"}


async def test_that_an_error_is_returned_when_no_payloads_are_provided(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post("/evaluations", json={"payloads": []})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()

    assert "detail" in data
    assert data["detail"] == "No payloads provided for the evaluation task."
