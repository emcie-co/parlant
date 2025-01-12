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
from lagom import Container
from pytest import fixture
from typing import Any, Optional, cast

from parlant.core.agents import Agent
from parlant.core.common import (
    CustomerId,
    EventSource,
    GuidelineContent,
    GuidelineId,
    generate_id,
)
from parlant.core.customers import Customer, CustomerStore
from parlant.core.engines.alpha.guideline_proposition import GuidelineProposition
from parlant.core.engines.alpha.tool_caller import ToolCaller, ToolCallInferenceSchema
from parlant.core.guidelines import Guideline
from parlant.core.logging import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.tools.plugins import tool
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import Event
from parlant.core.tools import LocalToolService, Tool, ToolContext, ToolId, ToolResult

from tests.core.common.utils import create_event_message
from tests.test_utilities import run_service_server


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


async def test_that_a_tool_from_a_local_service_gets_called_with_an_enum_parameter(
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


async def test_that_a_tool_from_a_plugin_gets_called_with_an_enum_parameter(
    container: Container,
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
