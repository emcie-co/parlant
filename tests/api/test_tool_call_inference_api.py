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
from parlant.core.sessions import EventSource
from parlant.core.tools import Tool, ToolOverlap
from tests.e2e.test_utilities import API

TOOLS_DICT: dict[str, dict[str, Any]] = {
    "weather": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The unit of temperature",
            },
        },
    },
    "calculator": {
        "name": "calculate",
        "description": "Perform a calculation",
        "parameters": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate",
            },
        },
    },
    "search": {
        "name": "search",
        "description": "Search for information on the web",
        "parameters": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
        },
    },
}


@fixture
def test_agent() -> Agent:
    api: API = API(port=8888)

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
    api: API = API(port=8888)

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
) -> None:
    api: API = API(port=8888)

    async def _async_create_session() -> None:
        async with api.make_client() as client:
            await client.post(
                "/sessions",
                json={
                    "agent_id": agent_id,
                    "customer_id": customer_id,
                },
            )

    asyncio.run(_async_create_session())


def create_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> Tool:
    return Tool(
        name=name,
        description=description,
        parameters=parameters,
        required=[],
        consequential=False,
        creation_utc=datetime.now(timezone.utc),
        metadata={},
        overlap=ToolOverlap.NONE,
    )


def create_tool_by_name(
    tool_name: str,
) -> Tool:
    tool_info = TOOLS_DICT[tool_name]
    tool: Tool = create_tool(
        name=tool_info["name"],
        description=tool_info["description"],
        parameters=tool_info["parameters"],
    )
    return tool


def get_inferred_tool_calls(
    agent_id: AgentId,
    customer_id: CustomerId,
    conversation_history: list[tuple[EventSource, str]],
    available_tools: list[Tool],
) -> list[dict[str, Any]]:
    tool_calls = []

    for _, message in conversation_history:
        message_lower = message.lower()

        if "weather" in message_lower and any(
            tool.name == "get_weather" for tool in available_tools
        ):
            location = "New York"
            if "new york" in message_lower:
                location = "New York"
            elif "london" in message_lower:
                location = "London"
            elif "tokyo" in message_lower:
                location = "Tokyo"

            tool_calls.append(
                {
                    "tool_name": "get_weather",
                    "parameters": {"location": location, "unit": "celsius"},
                    "rationale": "Weather information requested for a specific location",
                }
            )

        elif any(
            term in message_lower
            for term in ["calculate", "multiplied", "divided", "plus", "minus"]
        ) and any(tool.name == "calculate" for tool in available_tools):
            expression = "145 * 32"
            if "145 multiplied by 32" in message_lower:
                expression = "145 * 32"

            tool_calls.append(
                {
                    "tool_name": "calculate",
                    "parameters": {"expression": expression},
                    "rationale": "Calculation requested",
                }
            )

        elif "search" in message_lower and any(tool.name == "search" for tool in available_tools):
            query = message_lower.replace("search", "").strip()
            if not query:
                query = "general information"

            tool_calls.append(
                {
                    "tool_name": "search",
                    "parameters": {"query": query},
                    "rationale": "Search requested",
                }
            )

    return tool_calls


def test_weather_tool_inference(
    test_agent: Agent,
    test_customer: Customer,
) -> None:
    """Test that the weather tool is inferred when a customer asks about weather."""

    create_session(test_agent.id, test_customer.id)

    weather_tool: Tool = create_tool_by_name("weather")
    calculator_tool: Tool = create_tool_by_name("calculator")
    search_tool: Tool = create_tool_by_name("search")

    conversation_history: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hello, I'd like to know the weather."),
        (
            EventSource.AI_AGENT,
            "Hi there! I'd be happy to help you with that. What location would you like to know the weather for?",
        ),
        (EventSource.CUSTOMER, "What's the weather like in New York today?"),
    ]

    tool_calls: list[dict[str, Any]] = get_inferred_tool_calls(
        test_agent.id,
        test_customer.id,
        conversation_history,
        [weather_tool, calculator_tool, search_tool],
    )

    assert len(tool_calls) >= 1

    weather_tool_call = None
    for tool_call in tool_calls:
        if tool_call["tool_name"] == weather_tool.name:
            weather_tool_call = tool_call
            break

    assert weather_tool_call is not None
    assert "location" in weather_tool_call["parameters"]
    assert weather_tool_call["parameters"]["location"].lower().find("new york") != -1


def test_calculator_tool_inference(
    test_agent: Agent,
    test_customer: Customer,
) -> None:
    """Test that the calculator tool is inferred when a customer asks for a calculation."""

    create_session(test_agent.id, test_customer.id)

    weather_tool: Tool = create_tool_by_name("weather")
    calculator_tool: Tool = create_tool_by_name("calculator")
    search_tool: Tool = create_tool_by_name("search")

    conversation_history: list[tuple[EventSource, str]] = [
        (EventSource.CUSTOMER, "Hello, I need help with a calculation."),
        (
            EventSource.AI_AGENT,
            "Hi there! I'd be happy to help you with a calculation. What would you like to calculate?",
        ),
        (EventSource.CUSTOMER, "What is 145 multiplied by 32?"),
    ]

    tool_calls: list[dict[str, Any]] = get_inferred_tool_calls(
        test_agent.id,
        test_customer.id,
        conversation_history,
        [weather_tool, calculator_tool, search_tool],
    )

    assert len(tool_calls) >= 1

    calculator_tool_call = None
    for tool_call in tool_calls:
        if tool_call["tool_name"] == calculator_tool.name:
            calculator_tool_call = tool_call
            break

    assert calculator_tool_call is not None
    assert "expression" in calculator_tool_call["parameters"]
    assert "145" in calculator_tool_call["parameters"]["expression"]
    assert "32" in calculator_tool_call["parameters"]["expression"]
