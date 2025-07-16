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

from typing import Any

import httpx
from parlant.core.agents import Agent, AgentId
from parlant.core.customers import Customer, CustomerId
from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.sessions import EventSource, SessionId

from tests.e2e.test_api.conftest import (
    create_test_guideline,
    create_test_agent,
    create_test_customer,
    create_test_session,
)
from tests.e2e.test_utilities import ContextOfTest, run_server

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


async def create_guideline_by_name(
    client: httpx.AsyncClient,
    guideline_name: str,
) -> Guideline:
    guideline: Guideline = await create_test_guideline(
        client,
        action=GUIDELINES_DICT[guideline_name]["action"],
        condition=GUIDELINES_DICT[guideline_name]["condition"],
    )

    return guideline


async def get_session_id(
    client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> SessionId:
    response = await client.get(f"/sessions?agent_id={agent_id}&customer_id={customer_id}")
    if response.status_code != 200:
        raise Exception(f"Error getting session: {response.status_code} - {response.text}")
    sessions: list[dict[str, Any]] = response.json()
    if not sessions:
        raise Exception(f"No session found for agent {agent_id} and customer {customer_id}")
    return SessionId(sessions[0]["id"])


async def add_customer_messages_to_session(
    client: httpx.AsyncClient,
    session_id: SessionId,
    conversation_history: list[tuple[EventSource, str]],
) -> list[str]:
    ids: list[str] = []

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
                raise Exception(f"Error adding message: {response.status_code} - {response.text}")
            ids.append(response.json()["id"])

    response = await client.get(f"/sessions/{session_id}/events")
    if response.status_code != 200:
        raise Exception(f"Error getting events: {response.status_code} - {response.text}")

    all_events: list[dict[str, Any]] = response.json()
    return [event["id"] for event in all_events]


async def get_matched_guidelines(
    client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_id: CustomerId,
    event_ids: list[str],
    guideline_ids: list[GuidelineId],
    guidelines: list[Guideline],
) -> list[Guideline]:
    response = await client.post(
        "/test/alpha/guideline-matching",
        json={
            "agent_id": agent_id,
            "customer_id": customer_id,
            "interaction_history": event_ids,
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


async def test_pricing_guideline_matching(
    context: ContextOfTest,
) -> None:
    with run_server(context):
        async with context.api.make_client() as client:
            test_agent: Agent = await create_test_agent(client)
            test_customer: Customer = await create_test_customer(client)

            await create_test_session(client, test_agent.id, test_customer.id)

            pricing_guideline = await create_guideline_by_name(client, "pricing_query")
            product_guideline = await create_guideline_by_name(client, "product_inquiry")

            conversation_history: list[tuple[EventSource, str]] = [
                (EventSource.CUSTOMER, "Hello, I'm interested in your services."),
                (
                    EventSource.AI_AGENT,
                    "Hi there! Welcome to our service. How can I help you today?",
                ),
                (EventSource.CUSTOMER, "Can you tell me how much your software costs?"),
            ]

            agent_id = test_agent.id
            customer_id = test_customer.id

            session_id: SessionId = await get_session_id(client, agent_id, customer_id)

            event_ids: list[str] = await add_customer_messages_to_session(
                client, session_id, conversation_history
            )

            guidelines = [pricing_guideline, product_guideline]
            guideline_ids: list[GuidelineId] = [guideline.id for guideline in guidelines]

            matched_guidelines: list[Guideline] = await get_matched_guidelines(
                client, agent_id, customer_id, event_ids, guideline_ids, guidelines
            )

            assert len(matched_guidelines) == 1
            assert matched_guidelines[0].id == pricing_guideline.id
            assert (
                matched_guidelines[0].content.condition
                == GUIDELINES_DICT["pricing_query"]["condition"]
            )

            assert pricing_guideline in matched_guidelines
            assert product_guideline not in matched_guidelines
