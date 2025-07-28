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


import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import httpx
from parlant.core.agents import Agent, AgentId
from parlant.core.customers import Customer, CustomerId
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.tags import TagId
from pytest import fixture

from tests.e2e.test_utilities import API, ContextOfTest


@fixture
def context() -> Iterator[ContextOfTest]:
    with tempfile.TemporaryDirectory(prefix="parlant-server_cli_test_") as home_dir:
        home_dir_path = Path(home_dir)

        yield ContextOfTest(
            home_dir=home_dir_path,
            api=API(),
        )


async def create_test_agent(
    client: httpx.AsyncClient,
) -> Agent:
    response = await client.post(
        "/agents",
        json={
            "name": "test-agent",
        },
    )
    result_json: dict[str, Any] = response.json()
    result: Agent = Agent(
        creation_utc=datetime.now(timezone.utc),
        **result_json,
    )
    return result


async def create_test_customer(
    client: httpx.AsyncClient,
) -> Customer:
    response = await client.post(
        "/customers",
        json={
            "name": "test-customer",
        },
    )
    result_json: dict[str, Any] = response.json()
    result: Customer = Customer(**result_json)
    return result


async def create_test_session(
    client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    await client.post(
        "/sessions",
        json={
            "agent_id": agent_id,
            "customer_id": customer_id,
        },
    )


async def create_test_guideline(
    client: httpx.AsyncClient,
    action: str,
    condition: str,
    tags: list[TagId] = [],
) -> Guideline:
    response = await client.post(
        "/guidelines",
        json={
            "condition": condition,
            "action": action,
            "tags": tags,
            "enabled": True,
            "metadata": {},
        },
    )
    result_json: dict[str, Any] = response.json()
    return Guideline(
        id=GuidelineId(result_json["id"]),
        creation_utc=datetime.now(timezone.utc),
        content=GuidelineContent(
            condition=result_json["condition"],
            action=result_json["action"],
        ),
        enabled=result_json["enabled"],
        tags=result_json["tags"],
        metadata=result_json["metadata"],
    )
