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
from datetime import datetime, timezone
from typing import Any

from pytest import fixture

from parlant.core.agents import Agent, AgentId
from parlant.core.customers import Customer, CustomerId
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.sessions import EventSource, Session, SessionId
from parlant.core.tags import TagId
from tests.e2e.test_utilities import API

GUIDELINES_DICT: dict[str, dict[str, str]] = {
    "greeting_response": {
        "condition": "the customer greets the agent",
        "action": "respond with a warm greeting and ask how you can help",
    },
    "product_inquiry": {
        "condition": "the customer asks about product features",
        "action": "provide detailed information about product capabilities and benefits",
    },
    "pricing_query": {
        "condition": "the customer inquires about pricing",
        "action": "give current pricing information and mention any active promotions",
    },
}


@fixture
def test_agent() -> Agent:
    api: API = API("http://localhost:8888")

    async def _async_get_test_agent() -> Agent:
        async with api.make_client() as client:
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

    return asyncio.run(_async_get_test_agent())


@fixture
def test_customer() -> Customer:
    api: API = API("http://localhost:8888")

    async def _async_get_test_customer() -> Customer:
        async with api.make_client() as client:
            response = await client.post(
                "/customers",
                json={
                    "name": "test-customer",
                },
            )
            result_json: dict[str, Any] = response.json()
            result: Customer = Customer(**result_json)
            return result

    return asyncio.run(_async_get_test_customer())


def create_session(
    agent_id: AgentId,
    customer_id: CustomerId,
) -> Session:
    api: API = API("http://localhost:8888")

    async def _async_create_session() -> Session:
        async with api.make_client() as client:
            response = await client.post(
                "/sessions",
                json={
                    "agent_id": agent_id,
                    "customer_id": customer_id,
                },
            )
            result_json: dict[str, Any] = response.json()
            result: Session = Session(
                mode="auto",
                **result_json,
            )
            return result

    return asyncio.run(_async_create_session())


def create_guideline(
    condition: str,
    action: str,
    tags: list[TagId] = [],
) -> Guideline:
    api: API = API("http://localhost:8888")

    async def _async_create_guideline() -> Guideline:
        async with api.make_client() as client:
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

    return asyncio.run(_async_create_guideline())


def create_guideline_by_name(
    guideline_name: str,
) -> Guideline:
    guideline: Guideline = create_guideline(
        condition=GUIDELINES_DICT[guideline_name]["condition"],
        action=GUIDELINES_DICT[guideline_name]["action"],
    )
    return guideline


def get_matched_guidelines(
    agent_id: AgentId,
    customer_id: CustomerId,
    conversation_history: list[tuple[EventSource, str]],
    guidelines: list[Guideline],
) -> list[Guideline]:
    api: API = API("http://localhost:8888")

    async def _get_session_id() -> SessionId:
        async with api.make_client() as client:
            response = await client.get(f"/sessions?agent_id={agent_id}&customer_id={customer_id}")
            if response.status_code != 200:
                raise Exception(f"Error getting session: {response.status_code} - {response.text}")
            sessions: list[dict[str, Any]] = response.json()
            if not sessions:
                raise Exception(f"No session found for agent {agent_id} and customer {customer_id}")
            return SessionId(sessions[0]["id"])

    session_id: SessionId = asyncio.run(_get_session_id())

    async def _add_customer_messages_to_session() -> list[str]:
        ids: list[str] = []
        async with api.make_client() as client:
            for source, content in conversation_history:
                if source == EventSource.CUSTOMER:
                    response = await client.post(
                        f"/sessions/{session_id}/events",
                        json={
                            "source": source.value,
                            "kind": "message",
                            "message": content,
                        },
                    )
                    if response.status_code != 201:
                        raise Exception(
                            f"Error adding message: {response.status_code} - {response.text}"
                        )
                    ids.append(response.json()["id"])

            response = await client.get(f"/sessions/{session_id}/events")
            if response.status_code != 200:
                raise Exception(f"Error getting events: {response.status_code} - {response.text}")

            all_events: list[dict[str, Any]] = response.json()
            return [event["id"] for event in all_events]

    event_ids: list[str] = asyncio.run(_add_customer_messages_to_session())

    guideline_ids: list[GuidelineId] = [guideline.id for guideline in guidelines]

    async def _async_get_matched_guidelines() -> list[Guideline]:
        async with api.make_client() as client:
            response = await client.post(
                "/test/alpha/guideline-matching",
                json={
                    "agent_id": agent_id,
                    "customer_id": customer_id,
                    "events": event_ids,
                    "guidelines": guideline_ids,
                    "context_variables": [],
                    "terms": [],
                    "staged_events": [],
                },
            )

            if response.status_code not in [200, 202]:
                raise Exception(f"API error: {response.status_code} - {response.text}")

            result: dict[str, Any] = response.json()

            matched_guideline_ids: list[GuidelineId] = []
            for batch in result.get("batches", []):
                for match in batch:
                    matched_guideline_ids.append(match["guideline"])

            return [g for g in guidelines if g.id in matched_guideline_ids]

    return asyncio.run(_async_get_matched_guidelines())


def test_pricing_guideline_matching(
    test_agent: Agent,
    test_customer: Customer,
) -> None:
    """Test that the pricing guideline is matched when a customer asks about pricing."""

    _ = create_session(test_agent.id, test_customer.id)

    pricing_guideline: Guideline = create_guideline_by_name("pricing_query")
    greeting_guideline: Guideline = create_guideline_by_name("greeting_response")
    product_guideline: Guideline = create_guideline_by_name("product_inquiry")

    conversation_history: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hello, I'm interested in your services."),
        (EventSource.AI_AGENT, "Hi there! Welcome to our service. How can I help you today?"),
        (EventSource.CUSTOMER, "Can you tell me how much your software costs?"),
    ]

    matched_guidelines: list[Guideline] = get_matched_guidelines(
        test_agent.id,
        test_customer.id,
        conversation_history,
        [pricing_guideline, greeting_guideline, product_guideline],
    )

    assert len(matched_guidelines) == 2
    assert matched_guidelines[0].id == pricing_guideline.id
    assert matched_guidelines[0].content.condition == GUIDELINES_DICT["pricing_query"]["condition"]

    assert pricing_guideline in matched_guidelines
    assert greeting_guideline in matched_guidelines
    assert product_guideline not in matched_guidelines
