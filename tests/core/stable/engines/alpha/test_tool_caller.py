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

from datetime import datetime, timezone
import enum
from itertools import chain
from typing import Annotated, Any, Mapping, Optional, Sequence
from lagom import Container
from pytest import fixture
from typing_extensions import override

from parlant.core.agents import Agent
from parlant.core.common import generate_id
from parlant.core.customers import Customer, CustomerStore, CustomerId
from parlant.core.engines.alpha.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.tool_calling.tool_caller import (
    ToolCall,
    ToolCallBatch,
    ToolCallBatchResult,
    ToolCallBatcher,
    ToolCallContext,
    ToolCallId,
    ToolCaller,
    ToolInsights,
)
from parlant.core.guidelines import Guideline, GuidelineId, GuidelineContent
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.services.tools.plugins import tool
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import Event, EventSource, SessionStore
from parlant.core.tags import TagId, Tag
from parlant.core.tools import (
    LocalToolService,
    Tool,
    ToolContext,
    ToolId,
    ToolParameterOptions,
    ToolResult,
)

from tests.core.common.utils import create_event_message
from tests.test_utilities import run_service_server


@fixture
def local_tool_service(container: Container) -> LocalToolService:
    return container[LocalToolService]


@fixture
async def customer(container: Container, customer_id: CustomerId) -> Customer:
    return await container[CustomerStore].read_customer(customer_id)


async def tool_context(
    container: Container,
    agent: Agent,
    customer: Optional[Customer] = None,
) -> ToolContext:
    if customer is None:
        customer_id = CustomerStore.GUEST_ID
    else:
        customer_id = customer.id

    session = await container[SessionStore].create_session(customer_id, agent.id)

    return ToolContext(
        agent_id=agent.id,
        customer_id=customer_id,
        session_id=session.id,
    )


def create_interaction_history(
    conversation_context: list[tuple[EventSource, str]],
    customer: Optional[Customer] = None,
) -> list[Event]:
    return [
        create_event_message(
            offset=i,
            source=source,
            message=message,
            customer=customer,
        )
        for i, (source, message) in enumerate(conversation_context)
    ]


def create_guideline_match(
    condition: str,
    action: str,
    score: int,
    rationale: str,
    tags: list[TagId],
) -> GuidelineMatch:
    guideline = Guideline(
        id=GuidelineId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        content=GuidelineContent(
            condition=condition,
            action=action,
        ),
        enabled=True,
        tags=tags,
        metadata={},
    )

    return GuidelineMatch(guideline=guideline, score=score, rationale=rationale)


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
    container: Container,
    local_tool_service: LocalToolService,
    agent: Agent,
) -> None:
    tool_caller = container[ToolCaller]

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
        (EventSource.CUSTOMER, "Are you selling computers products?"),
        (EventSource.AI_AGENT, "Yes"),
        (EventSource.CUSTOMER, "What available keyboards do you have?"),
    ]

    interaction_history = create_interaction_history(conversation_context)

    ordinary_guideline_matches = [
        create_guideline_match(
            condition="customer asking a question",
            action="response in concise and breif answer",
            score=9,
            rationale="customer ask a question of what available keyboard do we have",
            tags=[Tag.for_agent_id(agent.id)],
        )
    ]

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="get all products by a specific category",
            action="a customer asks for the availability of products from a certain category",
            score=9,
            rationale="customer asks for keyboards availability",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ToolId(service_name="local", tool_name=tool.name)]
    }

    inference_tool_calls_result = await tool_caller.infer_tool_calls(
        agent=agent,
        context_variables=[],
        interaction_history=interaction_history,
        terms=[],
        ordinary_guideline_matches=ordinary_guideline_matches,
        tool_enabled_guideline_matches=tool_enabled_guideline_matches,
        staged_events=[],
        tool_context=await tool_context(container, agent),
    )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]

    assert "category" in tool_call.arguments
    assert tool_call.arguments["category"] == "peripherals"


async def test_that_a_tool_from_a_plugin_gets_called_with_an_enum_parameter(
    container: Container,
    agent: Agent,
) -> None:
    tool_caller = container[ToolCaller]
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
        (EventSource.CUSTOMER, "Are you selling computers products?"),
        (EventSource.AI_AGENT, "Yes"),
        (EventSource.CUSTOMER, "What available keyboards do you have?"),
    ]

    interaction_history = create_interaction_history(conversation_context)

    ordinary_guideline_matches = [
        create_guideline_match(
            condition="customer asking a question",
            action="response in concise and breif answer",
            score=9,
            rationale="customer ask a question of what available keyboard do we have",
            tags=[Tag.for_agent_id(agent.id)],
        )
    ]

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="get all products by a specific category",
            action="a customer asks for the availability of products from a certain category",
            score=9,
            rationale="customer asks for keyboards availability",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ToolId(service_name="my_sdk_service", tool_name="available_products_by_category")]
    }

    async with run_service_server([available_products_by_category]) as server:
        await service_registry.update_tool_service(
            name="my_sdk_service",
            kind="sdk",
            url=server.url,
        )

        inference_tool_calls_result = await tool_caller.infer_tool_calls(
            agent=agent,
            context_variables=[],
            interaction_history=interaction_history,
            terms=[],
            ordinary_guideline_matches=ordinary_guideline_matches,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
            staged_events=[],
            tool_context=await tool_context(container, agent),
        )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]

    assert "category" in tool_call.arguments
    assert tool_call.arguments["category"] == "peripherals"


async def test_that_a_plugin_tool_is_called_with_required_parameters_with_default_value(
    container: Container,
    agent: Agent,
) -> None:
    tool_caller = container[ToolCaller]
    service_registry = container[ServiceRegistry]

    class AppointmentType(enum.Enum):
        GENERAL = "general"
        CHECK_UP = "checkup"
        RESULTS = "result"

    class AppointmentRoom(enum.Enum):
        TINY = "phone booth"
        SMALL = "private room"
        BIG = "meeting room"

    @tool
    async def schedule_appointment(
        context: ToolContext,
        when: datetime,
        type: Optional[AppointmentType] = AppointmentType.GENERAL,
        room: AppointmentRoom = AppointmentRoom.TINY,
        number_of_invites: int = 3,
        required_participants: list[str] = ["Donald Trump", "Donald Duck", "Ronald McDonald"],
        meeting_owner: str = "Donald Trump",
    ) -> ToolResult:
        if type is None:
            type_display = "NONE"
        else:
            type_display = type.value

        return ToolResult(f"Scheduled {type_display} appointment in {room.value} at {when}")

    conversation_context = [
        (EventSource.CUSTOMER, "I want to set up an appointment tomorrow at 10am"),
    ]

    interaction_history = create_interaction_history(conversation_context)

    ordinary_guideline_matches = [
        create_guideline_match(
            condition="customer asking a question",
            action="response in concise and breif answer",
            score=9,
            rationale="customer asks a question about appointments",
            tags=[Tag.for_agent_id(agent.id)],
        )
    ]

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="customer asks to schedule an appointment",
            action="schedule an appointment for the customer",
            score=9,
            rationale="customer wants to schedule some kind of an appointment",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ToolId(service_name="my_appointment_service", tool_name="schedule_appointment")]
    }

    async with run_service_server([schedule_appointment]) as server:
        await service_registry.update_tool_service(
            name="my_appointment_service",
            kind="sdk",
            url=server.url,
        )

        inference_tool_calls_result = await tool_caller.infer_tool_calls(
            agent=agent,
            context_variables=[],
            interaction_history=interaction_history,
            terms=[],
            ordinary_guideline_matches=ordinary_guideline_matches,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
            staged_events=[],
            tool_context=await tool_context(container, agent),
        )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert "when" in tool_call.arguments


async def test_that_a_tool_from_a_plugin_gets_called_with_an_enum_list_parameter(
    container: Container,
    agent: Agent,
) -> None:
    tool_caller = container[ToolCaller]
    service_registry = container[ServiceRegistry]

    class ProductCategory(enum.Enum):
        LAPTOPS = "laptops"
        PERIPHERALS = "peripherals"

    @tool
    def available_products_by_category(
        context: ToolContext, categories: list[ProductCategory]
    ) -> ToolResult:
        products_by_category = {
            ProductCategory.LAPTOPS: ["Lenovo", "Dell"],
            ProductCategory.PERIPHERALS: ["Razer Keyboard", "Logitech Mouse"],
        }

        return ToolResult([products_by_category[category] for category in categories])

    conversation_context = [
        (EventSource.CUSTOMER, "Are you selling computers products?"),
        (EventSource.AI_AGENT, "Yes"),
        (EventSource.CUSTOMER, "What available keyboards and laptops do you have?"),
    ]

    interaction_history = create_interaction_history(conversation_context)

    ordinary_guideline_matches = [
        create_guideline_match(
            condition="customer asking a question",
            action="response in concise and breif answer",
            score=9,
            rationale="customer ask a question of what available keyboard do we have",
            tags=[Tag.for_agent_id(agent.id)],
        )
    ]

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="get all products by a specific category",
            action="a customer asks for the availability of products from a certain category",
            score=9,
            rationale="customer asks for keyboards availability",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ToolId(service_name="my_sdk_service", tool_name="available_products_by_category")]
    }

    async with run_service_server([available_products_by_category]) as server:
        await service_registry.update_tool_service(
            name="my_sdk_service",
            kind="sdk",
            url=server.url,
        )

        inference_tool_calls_result = await tool_caller.infer_tool_calls(
            agent=agent,
            context_variables=[],
            interaction_history=interaction_history,
            terms=[],
            ordinary_guideline_matches=ordinary_guideline_matches,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
            staged_events=[],
            tool_context=await tool_context(container, agent),
        )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]

    assert "categories" in tool_call.arguments
    assert isinstance(tool_call.arguments["categories"], str)
    assert ProductCategory.LAPTOPS.value in tool_call.arguments["categories"]
    assert ProductCategory.PERIPHERALS.value in tool_call.arguments["categories"]


async def test_that_a_tool_from_a_plugin_gets_called_with_a_parameter_attached_to_a_choice_provider(
    container: Container,
    agent: Agent,
) -> None:
    tool_caller = container[ToolCaller]
    service_registry = container[ServiceRegistry]
    plugin_data = {"choices": ["laptops", "peripherals"]}

    async def my_choice_provider(choices: list[str]) -> list[str]:
        return choices

    @tool
    def available_products_by_category(
        context: ToolContext,
        categories: Annotated[list[str], ToolParameterOptions(choice_provider=my_choice_provider)],
    ) -> ToolResult:
        products_by_category = {
            "laptops": ["Lenovo", "Dell"],
            "peripherals": ["Razer Keyboard", "Logitech Mouse"],
        }

        return ToolResult([products_by_category[category] for category in categories])

    conversation_context = [
        (EventSource.CUSTOMER, "Are you selling computers products?"),
        (EventSource.AI_AGENT, "Yes"),
        (EventSource.CUSTOMER, "What available keyboards and laptops do you have?"),
    ]

    interaction_history = create_interaction_history(conversation_context)

    ordinary_guideline_matches = [
        create_guideline_match(
            condition="customer asking a question",
            action="response in concise and breif answer",
            score=9,
            rationale="customer ask a question of what available keyboard do we have",
            tags=[Tag.for_agent_id(agent.id)],
        )
    ]

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="get all products by a specific category",
            action="a customer asks for the availability of products from a certain category",
            score=9,
            rationale="customer asks for keyboards availability",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ToolId(service_name="my_sdk_service", tool_name="available_products_by_category")]
    }

    async with run_service_server([available_products_by_category], plugin_data) as server:
        await service_registry.update_tool_service(
            name="my_sdk_service",
            kind="sdk",
            url=server.url,
        )

        inference_tool_calls_result = await tool_caller.infer_tool_calls(
            agent=agent,
            context_variables=[],
            interaction_history=interaction_history,
            terms=[],
            ordinary_guideline_matches=ordinary_guideline_matches,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
            staged_events=[],
            tool_context=await tool_context(container, agent),
        )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]

    assert "categories" in tool_call.arguments
    assert isinstance(tool_call.arguments["categories"], str)
    assert "laptops" in tool_call.arguments["categories"]
    assert "peripherals" in tool_call.arguments["categories"]


async def test_that_a_tool_with_a_parameter_attached_to_a_choice_provider_gets_the_tool_context(
    container: Container,
    agent: Agent,
) -> None:
    service_registry = container[ServiceRegistry]
    customer_store = container[CustomerStore]
    tool_caller = container[ToolCaller]

    # Fabricate two customers and sessions
    customer_larry = await customer_store.create_customer(
        "Larry David", extra={"email": "larry@david.com"}
    )
    customer_harry = await customer_store.create_customer(
        "Harry Davis", extra={"email": "harry@davis.com"}
    )

    tool_context_larry = await tool_context(container, agent, customer_larry)
    tool_context_harry = await tool_context(container, agent, customer_harry)

    async def my_choice_provider(context: ToolContext, dummy: str) -> list[str]:
        if context.customer_id == customer_larry.id:
            return ["laptops", "peripherals"]
        elif context.customer_id == customer_harry.id:
            return ["cakes", "cookies"]
        else:
            return []

    @tool
    def available_products_by_category(
        context: ToolContext,
        categories: Annotated[list[str], ToolParameterOptions(choice_provider=my_choice_provider)],
    ) -> ToolResult:
        products_by_category = {
            "laptops": ["Lenovo", "Dell"],
            "peripherals": ["Razer Keyboard", "Logitech Mouse"],
            "cakes": ["Chocolate", "Vanilla"],
            "cookies": ["Chocolate Chip", "Oatmeal"],
        }

        return ToolResult({"choices": [products_by_category[category] for category in categories]})

    conversation_context_laptops = [
        (
            EventSource.CUSTOMER,
            "Hi, what products are available in category of laptops and peripherals ?",
        ),
    ]
    conversation_context_cakes = [
        (
            EventSource.CUSTOMER,
            "Hi, what products are available in category of cakes and cookies ?",
        ),
    ]

    interaction_history_larry = create_interaction_history(conversation_context_laptops)
    interaction_history_harry = create_interaction_history(conversation_context_cakes)

    ordinary_guideline_matches = [
        create_guideline_match(
            condition="customer asking a question",
            action="response in concise and breif answer",
            score=9,
            rationale="customer ask a question of what available keyboard do we have",
            tags=[Tag.for_agent_id(agent.id)],
        )
    ]

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="get all products by a category or categories",
            action="a customer asks for the availability of products from a certain category or categories",
            score=9,
            rationale="customer wants to know what products are available",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ToolId(service_name="my_sdk_service", tool_name="available_products_by_category")]
    }

    plugin_data = {"dummy": ["lorem", "ipsum", "dolor"]}
    async with run_service_server([available_products_by_category], plugin_data) as server:
        await service_registry.update_tool_service(
            name="my_sdk_service",
            kind="sdk",
            url=server.url,
        )

        inference_tool_calls_result_larry = await tool_caller.infer_tool_calls(
            agent=agent,
            context_variables=[],
            interaction_history=interaction_history_larry,
            terms=[],
            ordinary_guideline_matches=ordinary_guideline_matches,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
            staged_events=[],
            tool_context=tool_context_larry,
        )

        inference_tool_calls_result_harry = await tool_caller.infer_tool_calls(
            agent=agent,
            context_variables=[],
            interaction_history=interaction_history_harry,
            terms=[],
            ordinary_guideline_matches=ordinary_guideline_matches,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
            staged_events=[],
            tool_context=tool_context_harry,
        )

        # Check that mixing of "larry" chat and "harry" context doesn't work well
        inference_tool_calls_result_mixed = await tool_caller.infer_tool_calls(
            agent=agent,
            context_variables=[],
            interaction_history=interaction_history_larry,
            terms=[],
            ordinary_guideline_matches=ordinary_guideline_matches,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
            staged_events=[],
            tool_context=tool_context_harry,
        )

    assert len(inference_tool_calls_result_larry.batches) == 1
    assert len(inference_tool_calls_result_harry.batches) == 1
    assert (
        len(inference_tool_calls_result_mixed.batches) == 0
        or inference_tool_calls_result_mixed.batches[0] == []
    )
    tc_larry = inference_tool_calls_result_larry.batches[0][0]
    assert "categories" in tc_larry.arguments
    assert isinstance(tc_larry.arguments["categories"], str)
    assert "laptops" in tc_larry.arguments["categories"]
    assert "peripherals" in tc_larry.arguments["categories"]
    tc_harry = inference_tool_calls_result_harry.batches[0][0]
    assert "categories" in tc_harry.arguments
    assert isinstance(tc_harry.arguments["categories"], str)
    assert "cakes" in tc_harry.arguments["categories"]
    assert "cookies" in tc_harry.arguments["categories"]


async def test_that_a_tool_from_a_plugin_with_missing_parameters_returns_the_missing_ones_by_precedence(
    container: Container,
    agent: Agent,
) -> None:
    tool_caller = container[ToolCaller]
    service_registry = container[ServiceRegistry]

    @tool
    def register_sweepstake(
        context: ToolContext,
        full_name: Annotated[str, ToolParameterOptions()],
        city: Annotated[str, ToolParameterOptions(precedence=1)],
        street: Annotated[str, ToolParameterOptions(precedence=1)],
        house_number: Annotated[str, ToolParameterOptions(precedence=1)],
        number_of_entries: Annotated[int, ToolParameterOptions(hidden=True, precedence=2)],
        donation_amount: Annotated[Optional[int], ToolParameterOptions(required=False)] = None,
    ) -> ToolResult:
        return ToolResult({"success": True})

    conversation_context = [
        (
            EventSource.CUSTOMER,
            "Hi, can you register me for the sweepstake? I will donate 100 dollars if I win",
        )
    ]

    interaction_history = create_interaction_history(conversation_context)

    ordinary_guideline_matches = [
        create_guideline_match(
            condition="customer wishes to be registered for a sweepstake",
            action="response in concise and breif answer",
            score=9,
            rationale="customer is interested in registering for the sweepstake",
            tags=[Tag.for_agent_id(agent.id)],
        )
    ]

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="customer explicitly asks to be registered for a sweepstake",
            action="register the customer for the sweepstake using all provided information",
            score=9,
            rationale="customer wants to register for the sweepstake and provides all the relevant information",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ToolId(service_name="my_scharlatan_service", tool_name="register_sweepstake")]
    }

    async with run_service_server([register_sweepstake]) as server:
        await service_registry.update_tool_service(
            name="my_scharlatan_service",
            kind="sdk",
            url=server.url,
        )

        inference_tool_calls_result = await tool_caller.infer_tool_calls(
            agent=agent,
            context_variables=[],
            interaction_history=interaction_history,
            terms=[],
            ordinary_guideline_matches=ordinary_guideline_matches,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
            staged_events=[],
            tool_context=await tool_context(container, agent),
        )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))

    assert len(tool_calls) == 0
    # Check missing parameters by name
    missing_parameters = set(
        map(lambda x: x.parameter, inference_tool_calls_result.insights.missing_data)
    )
    assert missing_parameters == {"full_name", "city", "street", "house_number"}


async def test_that_tool_calling_batchers_can_be_overridden(
    container: Container,
    agent: Agent,
) -> None:
    tool_caller = container[ToolCaller]

    class ActivateToolCallBatch(ToolCallBatch):
        def __init__(self, tools: Mapping[tuple[ToolId, Tool], Sequence[GuidelineMatch]]):
            self.tools = tools

        @override
        async def process(self) -> ToolCallBatchResult:
            return ToolCallBatchResult(
                tool_calls=[
                    ToolCall(
                        id=ToolCallId(generate_id()),
                        tool_id=k[0],
                        arguments={},
                    )
                    for k, _ in self.tools.items()
                ],
                generation_info=GenerationInfo(
                    schema_name="",
                    model="",
                    duration=0.0,
                    usage=UsageInfo(
                        input_tokens=0,
                        output_tokens=0,
                        extra={},
                    ),
                ),
                insights=ToolInsights(
                    missing_data=[],
                ),
            )

    class NeverActivateToolCallBatch(ToolCallBatch):
        def __init__(self, tools: Mapping[tuple[ToolId, Tool], Sequence[GuidelineMatch]]):
            self.tools = tools

        @override
        async def process(self) -> ToolCallBatchResult:
            return ToolCallBatchResult(
                tool_calls=[],
                generation_info=GenerationInfo(
                    schema_name="",
                    model="",
                    duration=0.0,
                    usage=UsageInfo(
                        input_tokens=0,
                        output_tokens=0,
                        extra={},
                    ),
                ),
                insights=ToolInsights(
                    missing_data=[],
                ),
            )

    class ActivateOnlyPingToolBatcher(ToolCallBatcher):
        @override
        async def create_batches(
            self,
            tools: Mapping[tuple[ToolId, Tool], Sequence[GuidelineMatch]],
            context: ToolCallContext,
        ) -> Sequence[ToolCallBatch]:
            batches: list[ToolCallBatch] = []
            for tool_id, _tool in tools:
                if tool_id.tool_name == "ping":
                    batches.append(ActivateToolCallBatch({(tool_id, _tool): []}))
                else:
                    batches.append(NeverActivateToolCallBatch({(tool_id, _tool): []}))

            return batches

    local_tool_service = container[LocalToolService]

    for tool_name in ("echo", "ping"):
        await local_tool_service.create_tool(
            name=tool_name,
            module_path="tests.tool_utilities",
            description="dummy",
            parameters={},
            required=[],
        )

    echo_tool_id = ToolId(service_name="local", tool_name="echo")
    ping_tool_id = ToolId(service_name="local", tool_name="ping")

    container[ToolCaller].batcher = ActivateOnlyPingToolBatcher()

    interaction_history = [
        create_event_message(
            offset=0,
            source=EventSource.CUSTOMER,
            message="hello",
        )
    ]

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="customer asks to echo",
            action="echo the customer's message",
            score=9,
            rationale="customer wants to echo their message",
            tags=[Tag.for_agent_id(agent.id)],
        ): [echo_tool_id],
        create_guideline_match(
            condition="customer asks to ping",
            action="ping the customer's message",
            score=9,
            rationale="customer wants to ping their message",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ping_tool_id],
    }

    res = await tool_caller.infer_tool_calls(
        agent=agent,
        context_variables=[],
        interaction_history=interaction_history,
        terms=[],
        ordinary_guideline_matches=[],
        tool_enabled_guideline_matches=tool_enabled_guideline_matches,
        staged_events=[],
        tool_context=await tool_context(container, agent),
    )

    all_tool_ids = {tc.tool_id.to_string() for tc in chain.from_iterable(res.batches)}
    assert ping_tool_id.to_string() in all_tool_ids
    assert echo_tool_id.to_string() not in all_tool_ids
