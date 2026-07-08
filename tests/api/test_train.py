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

import httpx
from fastapi import status


async def test_that_training_can_be_started_and_polled_to_completion(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post("/train")
    assert response.status_code == status.HTTP_201_CREATED

    job = response.json()
    assert job["id"]
    assert job["status"] in ("pending", "running", "completed")

    polled = job
    for _ in range(200):
        polled = (await async_client.get(f"/train/{job['id']}")).raise_for_status().json()
        if polled["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.02)

    assert polled["status"] == "completed"
    assert polled["percentage"] == 100.0


async def test_that_reading_an_unknown_training_job_returns_404(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.get("/train/does-not-exist")
    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_training_an_unknown_agent_returns_404(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post("/train", json={"agent_ids": ["nonexistent-agent"]})
    assert response.status_code == status.HTTP_404_NOT_FOUND
