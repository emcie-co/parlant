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

from datetime import datetime, timezone
import enum
from itertools import chain
from typing import Any, Optional, cast
from lagom import Container
from pytest import fixture

from parlant.core.agents import Agent
from parlant.core.common import generate_id
from parlant.core.customers import Customer, CustomerStore, CustomerId
from parlant.core.engines.alpha.guideline_proposition import GuidelineProposition
from parlant.core.engines.alpha.tool_caller import ToolCallInferenceSchema, ToolCaller
from parlant.core.guidelines import Guideline, GuidelineId, GuidelineContent
from parlant.core.logging import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.tools.plugins import tool
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import Event, EventSource
from parlant.core.tools import LocalToolService, Tool, ToolContext, ToolId, ToolResult
from tests.core.engines.alpha.utils import create_event_message
from tests.core.services.tools.test_plugin_client import run_service_server


@fixture
def local_tool_service(container: Container) -> LocalToolService:
    return container[LocalToolService]


@fixture
def tool_caller(container: Container) -> ToolCaller:
    return ToolCaller(
        container[Logger],
        container[ServiceRegistry],
        container[SchematicGenerator[ToolCallInferenceSchema]],
    )


@fixture
async def customer(container: Container, customer_id: CustomerId) -> Customer:
    return await container[CustomerStore].read_customer(customer_id)


def create_interaction_history(
    conversation_context: list[tuple[str, str]],
    customer: Optional[Customer] = None,
) -> list[Event]:
    return [
        create_event_message(
            offset=i,
            source=cast(EventSource, source),
            message=message,
            customer=customer,
        )
        for i, (source, message) in enumerate(conversation_context)
    ]


def create_guideline_proposition(
    condition: str,
    action: str,
    score: int,
    rationale: str,
) -> GuidelineProposition:
    guideline = Guideline(
        id=GuidelineId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        content=GuidelineContent(
            condition=condition,
            action=action,
        ),
    )

    return GuidelineProposition(guideline=guideline, score=score, rationale=rationale)


async def create_local_tool(
    local_tool_service: LocalToolService,
    name: str,
    description: str = "",
    module_path: str = "tests.tool_utilities",
    parameters: dict[str, Any] = {},
    required: list[str] = [],
) -> Tool:
    return await local_tool_service.create_tool(
        name=name,
        module_path=module_path,
        description=description,
        parameters=parameters,
        required=required,
    )


async def test_that_a_tool_from_local_service_is_getting_called_with_an_enum_parameter(
    customer: Customer,
    local_tool_service: LocalToolService,
    tool_caller: ToolCaller,
    agent: Agent,
) -> None:
    tool = await create_local_tool(
        local_tool_service,
        name="available_products_by_category",
        parameters={
            "category": {
                "type": "string",
                "enum": ["laptops", "peripherals"],
            },
        },
        required=["category"],
    )

    conversation_context = [
        ("customer", "Are you selling computers products?"),
        ("ai_agent", "Yes"),
        ("customer", "What available keyboards do you have?"),
    ]

    interaction_history = create_interaction_history(conversation_context)

    ordinary_guideline_propositions = [
        create_guideline_proposition(
            condition="customer asking a question",
            action="response in concise and breif answer",
            score=9,
            rationale="customer ask a question of what available keyboard do we have",
        )
    ]

    tool_enabled_guideline_propositions = {
        create_guideline_proposition(
            condition="get all products by a specific category",
            action="a customer asks for the availability of products from a certain category",
            score=9,
            rationale="customer asks for keyboards availability",
        ): [ToolId(service_name="local", tool_name=tool.name)]
    }

    inference_tool_calls_result = await tool_caller.infer_tool_calls(
        agents=[agent],
        context_variables=[],
        interaction_history=interaction_history,
        terms=[],
        ordinary_guideline_propositions=ordinary_guideline_propositions,
        tool_enabled_guideline_propositions=tool_enabled_guideline_propositions,
        staged_events=[],
    )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]

    assert "category" in tool_call.arguments
    assert tool_call.arguments["category"] == "peripherals"


async def test_that_a_tool_from_plugin_is_getting_called_with_an_enum_parameter(
    container: Container,
    customer: Customer,
    tool_caller: ToolCaller,
    agent: Agent,
) -> None:
    service_registry = container[ServiceRegistry]

    class ProductCategory(enum.Enum):
        LAPTOPS = "laptops"
        PERIPHERALS = "peripherals"

    @tool
    def available_products_by_category(
        context: ToolContext, category: ProductCategory
    ) -> ToolResult:
        products_by_category = {
            ProductCategory.LAPTOPS: ["Lenovo", "Dell"],
            ProductCategory.PERIPHERALS: ["Razer Keyboard", "Logitech Mouse"],
        }

        return ToolResult(products_by_category[category])

    conversation_context = [
        ("customer", "Are you selling computers products?"),
        ("ai_agent", "Yes"),
        ("customer", "What available keyboards do you have?"),
    ]

    interaction_history = create_interaction_history(conversation_context)

    ordinary_guideline_propositions = [
        create_guideline_proposition(
            condition="customer asking a question",
            action="response in concise and breif answer",
            score=9,
            rationale="customer ask a question of what available keyboard do we have",
        )
    ]

    tool_enabled_guideline_propositions = {
        create_guideline_proposition(
            condition="get all products by a specific category",
            action="a customer asks for the availability of products from a certain category",
            score=9,
            rationale="customer asks for keyboards availability",
        ): [ToolId(service_name="my_sdk_service", tool_name="available_products_by_category")]
    }

    async with run_service_server([available_products_by_category]) as server:
        await service_registry.update_tool_service(
            name="my_sdk_service",
            kind="sdk",
            url=server.url,
        )

        inference_tool_calls_result = await tool_caller.infer_tool_calls(
            agents=[agent],
            context_variables=[],
            interaction_history=interaction_history,
            terms=[],
            ordinary_guideline_propositions=ordinary_guideline_propositions,
            tool_enabled_guideline_propositions=tool_enabled_guideline_propositions,
            staged_events=[],
        )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]

    assert "category" in tool_call.arguments
    assert tool_call.arguments["category"] == "peripherals"


async def test_that_a_tool_can_be_called_multiple_times_with_deduplication(
    customer: Customer, tool_caller: ToolCaller, agent: Agent, local_tool_service: LocalToolService
) -> None:
    add_tool = await create_local_tool(
        local_tool_service,
        name="add",
        parameters={
            "first_number": {"type": "integer"},
            "second_number": {"type": "integer"},
        },
        required=["first_number", "second_number"],
    )

    multiply_tool = await create_local_tool(
        local_tool_service,
        name="multiply",
        parameters={
            "first_number": {"type": "integer"},
            "second_number": {"type": "integer"},
        },
        required=["first_number", "second_number"],
    )

    conversation_context = [
        (
            "customer",
            (
                "Can you tell me the results of 8+2, 4*6, 9+5, and 3*5? Also, I need 10+2, "
                "7+3, 1+1, and 2*3. By the way, could you repeat 8+2, 4*6, and 9+5 again? "
                "And maybe throw in 50+50 and 10*10, 6*6, as well as 100*200. Actually, "
                "I mentioned 3*5, 2*3, and 7+3 more than once, sorry about that. Can you "
                "handle all these calculations?"
            ),
        )
    ]

    interaction_history = create_interaction_history(conversation_context)

    tool_enabled_guideline_propositions = {
        create_guideline_proposition(
            condition="customer asks arithmetic questions",
            action="calculate addition or multiplication",
            score=9,
            rationale="customer asks for multiple arithmetic calculations",
        ): [
            ToolId(service_name="local", tool_name=add_tool.name),
            ToolId(service_name="local", tool_name=multiply_tool.name),
        ]
    }

    inference_tool_calls_result = await tool_caller.infer_tool_calls(
        agents=[agent],
        context_variables=[],
        interaction_history=interaction_history,
        terms=[],
        ordinary_guideline_propositions=[],
        tool_enabled_guideline_propositions=tool_enabled_guideline_propositions,
        staged_events=[],
    )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 6

    expected_add_args = {(8, 2), (9, 5), (10, 2), (7, 3), (1, 1), (50, 50)}

    add_calls = []

    for call in tool_calls:
        if call.tool_id.tool_name == "add":
            add_calls.append((call.arguments["first_number"], call.arguments["second_number"]))

    assert len(add_calls) == 6
    assert set(add_calls) == expected_add_args
