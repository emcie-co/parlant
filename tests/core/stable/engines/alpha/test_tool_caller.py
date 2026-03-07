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

from datetime import date, datetime, timezone, timedelta
import enum
from itertools import chain
from typing import Annotated, Any, Mapping, Optional, Sequence, List, cast
import uuid
from pathlib import Path
from lagom import Container
from pytest import fixture
from typing_extensions import override
from ast import literal_eval

from parlant.core.agents import Agent
from parlant.core.common import Criticality, generate_id
from parlant.core.customers import Customer, CustomerStore, CustomerId
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.tool_calling.tool_caller import (
    ToolCall,
    ToolCallBatch,
    ToolCallBatchResult,
    ToolCallBatcher,
    ToolCallContext,
    ToolCallId,
    ToolCallInferenceResult,
    ToolCaller,
    ToolInsights,
)
from parlant.core.guidelines import Guideline, GuidelineId, GuidelineContent
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipEntity,
    RelationshipStore,
    RelationshipKind,
)
from parlant.core.services.tools.plugins import tool
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.emissions import EmittedEvent
from parlant.core.sessions import Event, EventKind, EventSource, SessionId, SessionStore
from parlant.core.tags import TagId, Tag
from parlant.core.tools import (
    LocalToolService,
    Tool,
    ToolContext,
    ToolId,
    ToolOverlap,
    ToolParameterOptions,
    ToolResult,
)

from tests.core.common.utils import create_event_message
from tests.test_utilities import run_service_server, get_random_port
from parlant.core.services.tools.mcp_service import MCPToolServer


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
        criticality=Criticality.MEDIUM,
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


async def _inference_tool_calls_result(
    container: Container,
    agent: Agent,
    interaction_history: list[Event],
    tool_enabled_guideline_matches: Mapping[GuidelineMatch, Sequence[ToolId]],
    tool_context_obj: ToolContext | None = None,
    staged_events: Sequence[EmittedEvent] | None = None,
) -> ToolCallInferenceResult:
    tool_caller = container[ToolCaller]

    tool_context_obj = tool_context_obj or await tool_context(container, agent)

    tool_call_context = ToolCallContext(
        agent=agent,
        session_id=cast(SessionId, tool_context_obj.session_id),
        customer_id=cast(CustomerId, tool_context_obj.customer_id),
        context_variables=[],
        interaction_history=interaction_history,
        terms=[],
        ordinary_guideline_matches=[],
        tool_enabled_guideline_matches=tool_enabled_guideline_matches,
        journeys=[],
        staged_events=staged_events or [],
    )

    return await tool_caller.infer_tool_calls(tool_call_context)


async def test_that_a_tool_from_a_local_service_gets_called_with_an_enum_parameter(
    container: Container,
    local_tool_service: LocalToolService,
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
        (EventSource.CUSTOMER, "Are you selling computers products?"),
        (EventSource.AI_AGENT, "Yes"),
        (EventSource.CUSTOMER, "What available keyboards do you have?"),
    ]

    interaction_history = create_interaction_history(conversation_context)

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="get all products by a specific category",
            action="a customer asks for the availability of products from a certain category",
            score=9,
            rationale="customer asks for keyboards availability",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ToolId(service_name="local", tool_name=tool.name)]
    }

    inference_tool_calls_result = await _inference_tool_calls_result(
        container=container,
        agent=agent,
        interaction_history=interaction_history,
        tool_enabled_guideline_matches=tool_enabled_guideline_matches,
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

        inference_tool_calls_result = await _inference_tool_calls_result(
            container=container,
            agent=agent,
            interaction_history=interaction_history,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
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
        (EventSource.CUSTOMER, "I want to set up an appointment tomorrow (10.26.23) at 10am"),
    ]

    interaction_history = create_interaction_history(conversation_context)

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

        inference_tool_calls_result = await _inference_tool_calls_result(
            container=container,
            agent=agent,
            interaction_history=interaction_history,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
        )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert "when" in tool_call.arguments


async def test_that_a_tool_from_a_plugin_gets_called_with_an_enum_list_parameter(
    container: Container,
    agent: Agent,
) -> None:
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

        inference_tool_calls_result = await _inference_tool_calls_result(
            container=container,
            agent=agent,
            interaction_history=interaction_history,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
        )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]

    assert "categories" in tool_call.arguments
    assert isinstance(tool_call.arguments["categories"], str)
    assert set(literal_eval(tool_call.arguments["categories"])) == set(
        [
            ProductCategory.LAPTOPS.value,
            ProductCategory.PERIPHERALS.value,
        ]
    )
    assert ProductCategory.LAPTOPS.value in tool_call.arguments["categories"]
    assert ProductCategory.PERIPHERALS.value in tool_call.arguments["categories"]


async def test_that_a_tool_is_called_with_typing_lists(
    container: Container,
    agent: Agent,
) -> None:
    service_registry = container[ServiceRegistry]

    class ProductCategory(enum.Enum):
        LAPTOPS = "laptops"
        PERIPHERALS = "peripherals"

    @tool
    def available_products_by_category(
        context: ToolContext, categories: List[ProductCategory]
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

        inference_tool_calls_result = await _inference_tool_calls_result(
            container=container,
            agent=agent,
            interaction_history=interaction_history,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
        )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]

    assert "categories" in tool_call.arguments
    assert isinstance(tool_call.arguments["categories"], str)
    assert literal_eval(tool_call.arguments["categories"]) == [
        ProductCategory.LAPTOPS.value,
        ProductCategory.PERIPHERALS.value,
    ]
    assert ProductCategory.LAPTOPS.value in tool_call.arguments["categories"]
    assert ProductCategory.PERIPHERALS.value in tool_call.arguments["categories"]


async def test_that_a_tool_from_a_plugin_gets_called_with_a_parameter_attached_to_a_choice_provider(
    container: Container,
    agent: Agent,
) -> None:
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

        inference_tool_calls_result = await _inference_tool_calls_result(
            container=container,
            agent=agent,
            interaction_history=interaction_history,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
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

        inference_tool_calls_result_larry = await _inference_tool_calls_result(
            container,
            agent=agent,
            interaction_history=interaction_history_larry,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
            tool_context_obj=tool_context_larry,
        )

        inference_tool_calls_result_harry = await _inference_tool_calls_result(
            container,
            agent=agent,
            interaction_history=interaction_history_harry,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
            tool_context_obj=tool_context_harry,
        )

        # Check that mixing of "larry" chat and "harry" context doesn't work well
        inference_tool_calls_result_mixed = await _inference_tool_calls_result(
            container,
            agent=agent,
            interaction_history=interaction_history_larry,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
            tool_context_obj=tool_context_harry,
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
    service_registry = container[ServiceRegistry]

    @tool(consequential=True)
    def register_sweepstake(
        context: ToolContext,
        full_name: Annotated[str, ToolParameterOptions()],
        city: Annotated[str, ToolParameterOptions(precedence=1)],
        street: Annotated[str, ToolParameterOptions(precedence=1)],
        house_number: Annotated[str, ToolParameterOptions(precedence=1)],
        number_of_entries: Annotated[int, ToolParameterOptions(hidden=True, precedence=2)],
        donation_amount: Annotated[Optional[int], ToolParameterOptions()] = None,
    ) -> ToolResult:
        return ToolResult({"success": True})

    conversation_context = [
        (
            EventSource.CUSTOMER,
            "Hi, can you register me for the sweepstake? I will donate 100 dollars if I win",
        )
    ]

    interaction_history = create_interaction_history(conversation_context)

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="customer explicitly asks to be registered for a sweepstake",
            action="register the customer for the sweepstake using all provided information",
            score=9,
            rationale="customer wants to register for the sweepstake and provides all the relevant information",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ToolId(service_name="my_charlatan_service", tool_name="register_sweepstake")]
    }

    async with run_service_server([register_sweepstake]) as server:
        await service_registry.update_tool_service(
            name="my_charlatan_service",
            kind="sdk",
            url=server.url,
        )

        inference_tool_calls_result = await _inference_tool_calls_result(
            container=container,
            agent=agent,
            interaction_history=interaction_history,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
        )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))

    assert len(tool_calls) == 0
    # Check missing parameters by name
    missing_parameters = set(
        map(lambda x: x.parameter, inference_tool_calls_result.insights.missing_data)
    )
    assert missing_parameters == {"full_name", "city", "street", "house_number"}


async def test_that_a_tool_with_an_invalid_choice_provider_parameter_and_a_missing_parameter_interacts_correctly(
    container: Container,
    agent: Agent,
) -> None:
    service_registry = container[ServiceRegistry]

    async def destination_choices() -> list[str]:
        return ["London", "Tokyo", "Reykjavik"]

    @tool(consequential=True)
    def book_flight(
        context: ToolContext,
        destination: Annotated[str, ToolParameterOptions(choice_provider=destination_choices)],
        passenger_id: int,
    ) -> ToolResult:
        return ToolResult(
            {"message": f"Successfully booked flight to {destination} for passenger {passenger_id}"}
        )

    conversation_context = [
        (EventSource.CUSTOMER, "Hi, my nemesis would like to book a one-way flight to Hell"),
    ]

    interaction_history = create_interaction_history(conversation_context)

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="customer wants to book a flight",
            action="book a flight for the customer",
            score=9,
            rationale="customer wants to book a flight",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ToolId(service_name="my_sdk_service", tool_name="book_flight")]
    }

    async with run_service_server([book_flight]) as server:
        await service_registry.update_tool_service(
            name="my_sdk_service",
            kind="sdk",
            url=server.url,
        )

        inference_tool_calls_result = await _inference_tool_calls_result(
            container=container,
            agent=agent,
            interaction_history=interaction_history,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
        )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 0 or tool_calls[0] == []
    insights = inference_tool_calls_result.insights
    assert len(insights.missing_data) == 1 and insights.missing_data[0].parameter == "passenger_id"
    assert len(insights.invalid_data) == 1 and insights.invalid_data[0].parameter == "destination"


async def test_that_a_tool_with_an_invalid_enum_parameter_and_a_missing_parameter_interacts_correctly(
    container: Container,
    agent: Agent,
) -> None:
    service_registry = container[ServiceRegistry]

    class Destination(enum.Enum):
        LONDON = "London"
        TOKYO = "Tokyo"
        REYKJAVIK = "Reykjavik"

    @tool(consequential=True)
    def book_flight(
        context: ToolContext,
        destination: Destination,
        passenger_id: int,
    ) -> ToolResult:
        return ToolResult(
            {"message": f"Successfully booked flight to {destination} for passenger {passenger_id}"}
        )

    conversation_context = [
        (
            EventSource.CUSTOMER,
            "Hi, I would like to book a flight to Singapore",
        ),
    ]

    interaction_history = create_interaction_history(conversation_context)

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="customer wants to book a flight",
            action="book a flight for the customer",
            score=9,
            rationale="customer wants to book a flight",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ToolId(service_name="my_sdk_service", tool_name="book_flight")]
    }

    async with run_service_server([book_flight]) as server:
        await service_registry.update_tool_service(
            name="my_sdk_service",
            kind="sdk",
            url=server.url,
        )

        inference_tool_calls_result = await _inference_tool_calls_result(
            container=container,
            agent=agent,
            interaction_history=interaction_history,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
        )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    insights = inference_tool_calls_result.insights
    assert len(tool_calls) == 0 or tool_calls[0] == []
    assert len(insights.missing_data) == 1 and insights.missing_data[0].parameter == "passenger_id"
    assert len(insights.invalid_data) == 1 and insights.invalid_data[0].parameter == "destination"
    assert (
        insights.invalid_data[0].choices is not None and len(insights.invalid_data[0].choices) > 0
    )


async def test_that_mcp_tool_with_uuid_path_timedelta_and_datetime_parameters_interacts_correctly(
    container: Container,
    agent: Agent,
) -> None:
    tool_caller = container[ToolCaller]
    service_registry = container[ServiceRegistry]

    async def report_update_duration(
        reporter: uuid.UUID,
        path: Path,
        update_start: datetime,
        update_duration: timedelta,
    ) -> str:
        return f"Agent {reporter} reported a duration of {update_duration} for {path} started from {update_start}"

    conversation_context = [
        (
            EventSource.CUSTOMER,
            "Hi, I am agent id deadface-fade-cafe-9876-000decade000 reporting that updating the file /secret/path/to.file started at 1999-11-01 03:22:41 and took me 2 hours 3 minutes and 31 seconds to complete",
        ),
    ]

    interaction_history = create_interaction_history(conversation_context)

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="agent wants to report an update duration",
            action="report the update duration and relevant details",
            score=9,
            rationale="agent wants to report that a file update took a long time",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ToolId(service_name="my_mcp_service", tool_name="report_update_duration")]
    }

    async with MCPToolServer([report_update_duration], port=get_random_port()) as server:
        await service_registry.update_tool_service(
            name="my_mcp_service",
            kind="mcp",
            url=f"http://localhost:{server.get_port()}",
        )

        inference_tool_calls_result = await _inference_tool_calls_result(
            container,
            agent=agent,
            interaction_history=interaction_history,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
        )

        tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
        assert len(tool_calls) == 1
        assert len(tool_calls[0].arguments) == 4

        tc = tool_calls[0]
        assert tc.arguments["reporter"] == "deadface-fade-cafe-9876-000decade000"
        assert tc.arguments["path"] == "/secret/path/to.file"
        assert tc.arguments["update_start"] == str(datetime(1999, 11, 1, 3, 22, 41))
        assert tc.arguments["update_duration"] == str(timedelta(hours=2, minutes=3, seconds=31))

        context = await tool_context(container, agent)

        results = await tool_caller.execute_tool_calls(context, tool_calls)

    assert len(results) == 1
    assert (
        results[0].result["data"]
        == "Agent deadface-fade-cafe-9876-000decade000 reported a duration of 2:03:31 for /secret/path/to.file started from 1999-11-01 03:22:41"
    )


async def test_that_mcp_tool_with_optional_lists_of_enum_date_and_bool_can_run(
    container: Container,
    agent: Agent,
) -> None:
    service_registry = container[ServiceRegistry]
    tool_caller = container[ToolCaller]

    class BirdType(enum.Enum):
        Angry = "AngryBird"
        Chatty = "Parrot"
        Funny = "Kakadu"
        Extinct = "Dodo"
        Fried = "Schnitzel"

    async def prepare_bird_delivery(
        date: Optional[date],
        birds: Optional[list[BirdType]],
        alive: list[bool],
    ) -> str:
        if birds is None:
            return "No birds to deliver"
        return (
            "Delivering birds: "
            + ", ".join(str(bird) for bird in birds)
            + f" on {date}, alive: {alive}"
        )

    conversation_context = [
        (
            EventSource.CUSTOMER,
            "Hi, please prepare the following list of birds for delivery for 1/1/25: AngryBird, Parrot, Kakadu and Schnitzel. first 3 are alive, but the schnitzel is not alive",
        ),
    ]

    interaction_history = create_interaction_history(conversation_context)

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="customer wants to prepare birds for delivery",
            action="prepare the birds for delivery as customer requested",
            score=9,
            rationale="customer wants to deliver a list of birds",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ToolId(service_name="my_mcp_service", tool_name="prepare_bird_delivery")]
    }

    async with MCPToolServer([prepare_bird_delivery], port=get_random_port()) as server:
        await service_registry.update_tool_service(
            name="my_mcp_service",
            kind="mcp",
            url=f"http://localhost:{server.get_port()}",
        )

        context = await tool_context(container, agent)
        inference_tool_calls_result = await _inference_tool_calls_result(
            container,
            agent=agent,
            interaction_history=interaction_history,
            tool_enabled_guideline_matches=tool_enabled_guideline_matches,
            tool_context_obj=context,
        )

        tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
        assert len(tool_calls) == 1
        assert len(tool_calls[0].arguments) == 3

        tc = tool_calls[0]
        assert tc.arguments["date"] == str(date(2025, 1, 1))
        assert "birds" in tc.arguments
        assert str(tc.arguments["alive"]).lower() == str([True, True, True, False]).lower()

        results = await tool_caller.execute_tool_calls(context, tool_calls)

    assert len(results) == 1
    result_data = results[0].result["data"]
    assert isinstance(result_data, str)
    assert "Delivering birds: " in result_data


async def test_that_tool_calling_batchers_can_be_overridden(
    container: Container,
    agent: Agent,
) -> None:
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

    result = await _inference_tool_calls_result(
        container,
        agent=agent,
        interaction_history=interaction_history,
        tool_enabled_guideline_matches=tool_enabled_guideline_matches,
    )

    all_tool_ids = {tc.tool_id.to_string() for tc in chain.from_iterable(result.batches)}
    assert ping_tool_id.to_string() in all_tool_ids
    assert echo_tool_id.to_string() not in all_tool_ids


async def test_that_two_non_overlapping_tools_are_overlapping_with_a_third_tool_they_are_all_considered_in_the_same_evaluation_batch(
    container: Container,
    agent: Agent,
) -> None:
    tool_caller = container[ToolCaller]
    relationship_store = container[RelationshipStore]

    interaction_history = [
        create_event_message(
            offset=0,
            source=EventSource.CUSTOMER,
            message="hello",
        )
    ]
    _tool = Tool(
        name="test_tool",
        creation_utc=datetime.now(),
        description="",
        metadata={},
        parameters={},
        required=[],
        consequential=True,
        overlap=ToolOverlap.AUTO,
    )

    a_tool_id = ToolId(service_name="local", tool_name="aa")
    b_tool_id = ToolId(service_name="local", tool_name="bb")
    c_tool_id = ToolId(service_name="local", tool_name="cc")

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="customer asks to a",
            action="do a",
            score=9,
            rationale="customer wants to a",
            tags=[Tag.for_agent_id(agent.id)],
        ): [a_tool_id],
        create_guideline_match(
            condition="customer asks to b",
            action="do b",
            score=9,
            rationale="customer wants to b",
            tags=[Tag.for_agent_id(agent.id)],
        ): [b_tool_id],
        create_guideline_match(
            condition="customer asks to c",
            action="do c",
            score=9,
            rationale="customer wants to c",
            tags=[Tag.for_agent_id(agent.id)],
        ): [c_tool_id],
    }

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=a_tool_id,
            kind=RelationshipEntityKind.TOOL,
        ),
        target=RelationshipEntity(
            id=b_tool_id,
            kind=RelationshipEntityKind.TOOL,
        ),
        kind=RelationshipKind.OVERLAP,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=b_tool_id,
            kind=RelationshipEntityKind.TOOL,
        ),
        target=RelationshipEntity(
            id=c_tool_id,
            kind=RelationshipEntityKind.TOOL,
        ),
        kind=RelationshipKind.OVERLAP,
    )

    tools: Mapping[tuple[ToolId, Tool], Sequence[GuidelineMatch]] = {
        (a_tool_id, _tool): [],
        (b_tool_id, _tool): [],
        (c_tool_id, _tool): [],
    }

    tool_context_obj = await tool_context(container, agent)
    tool_call_context = ToolCallContext(
        agent=agent,
        session_id=cast(SessionId, tool_context_obj.session_id),
        customer_id=cast(CustomerId, tool_context_obj.customer_id),
        context_variables=[],
        interaction_history=interaction_history,
        terms=[],
        ordinary_guideline_matches=[],
        tool_enabled_guideline_matches=tool_enabled_guideline_matches,
        journeys=[],
        staged_events=[],
    )

    batches: Sequence[ToolCallBatch] = await tool_caller.batcher.create_batches(
        tools, context=tool_call_context
    )

    assert len(batches) == 1


async def test_that_a_tool_with_unmatched_guideline_is_not_included_in_the_evaluation_batch_when_its_overlapped_tools_are_with_a_matched_guideline_and_does_not_indirectly_cause_overlap_between_those_tools(
    container: Container,
    agent: Agent,
) -> None:
    tool_caller = container[ToolCaller]
    relationship_store = container[RelationshipStore]

    interaction_history = [
        create_event_message(
            offset=0,
            source=EventSource.CUSTOMER,
            message="hello",
        )
    ]
    _tool = Tool(
        name="test_tool",
        creation_utc=datetime.now(),
        description="",
        metadata={},
        parameters={},
        required=[],
        consequential=True,
        overlap=ToolOverlap.AUTO,
    )

    a_tool_id = ToolId(service_name="local", tool_name="aa")
    b_tool_id = ToolId(service_name="local", tool_name="bb")
    c_tool_id = ToolId(service_name="local", tool_name="cc")

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="customer asks to a",
            action="do a",
            score=9,
            rationale="customer wants to a",
            tags=[Tag.for_agent_id(agent.id)],
        ): [a_tool_id],
        create_guideline_match(
            condition="customer asks to c",
            action="do c",
            score=9,
            rationale="customer wants to c",
            tags=[Tag.for_agent_id(agent.id)],
        ): [c_tool_id],
    }

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=a_tool_id,
            kind=RelationshipEntityKind.TOOL,
        ),
        target=RelationshipEntity(
            id=b_tool_id,
            kind=RelationshipEntityKind.TOOL,
        ),
        kind=RelationshipKind.OVERLAP,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=b_tool_id,
            kind=RelationshipEntityKind.TOOL,
        ),
        target=RelationshipEntity(
            id=c_tool_id,
            kind=RelationshipEntityKind.TOOL,
        ),
        kind=RelationshipKind.OVERLAP,
    )

    tools: Mapping[tuple[ToolId, Tool], Sequence[GuidelineMatch]] = {
        (a_tool_id, _tool): [],
        (c_tool_id, _tool): [],
    }

    tool_context_obj = await tool_context(container, agent)
    tool_call_context = ToolCallContext(
        agent=agent,
        session_id=cast(SessionId, tool_context_obj.session_id),
        customer_id=cast(CustomerId, tool_context_obj.customer_id),
        context_variables=[],
        interaction_history=interaction_history,
        terms=[],
        ordinary_guideline_matches=[],
        tool_enabled_guideline_matches=tool_enabled_guideline_matches,
        journeys=[],
        staged_events=[],
    )

    batches: Sequence[ToolCallBatch] = await tool_caller.batcher.create_batches(
        tools, context=tool_call_context
    )

    assert len(batches) == 2


async def test_that_non_consequential_tool_with_no_parameters_is_auto_approved_without_llm_inference(
    container: Container,
    local_tool_service: LocalToolService,
    agent: Agent,
) -> None:
    """
    A non-consequential tool with no parameters should be auto-approved
    without calling the LLM.
    """
    # Create a tool with no parameters and consequential=False (default)
    tool = await create_local_tool(
        local_tool_service,
        name="ping",
        description="A simple ping tool with no parameters",
        parameters={},
        required=[],
    )

    conversation_context = [
        (EventSource.CUSTOMER, "Hello, can you ping for me?"),
    ]

    interaction_history = create_interaction_history(conversation_context)

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="customer asks to ping",
            action="ping for the customer",
            score=9,
            rationale="customer wants to ping",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ToolId(service_name="local", tool_name=tool.name)]
    }

    inference_tool_calls_result = await _inference_tool_calls_result(
        container=container,
        agent=agent,
        interaction_history=interaction_history,
        tool_enabled_guideline_matches=tool_enabled_guideline_matches,
    )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]

    # Verify the tool call has no arguments
    assert tool_call.arguments == {}
    # Verify the tool_id is correct
    assert tool_call.tool_id == ToolId(service_name="local", tool_name="ping")
    # Verify no LLM was called (model should be "auto-approved")
    assert len(inference_tool_calls_result.batch_generations) == 1
    assert inference_tool_calls_result.batch_generations[0].model == "auto-approved"


async def test_that_staged_non_consequential_tool_with_no_parameters_is_not_auto_approved_again(
    container: Container,
    local_tool_service: LocalToolService,
    agent: Agent,
) -> None:
    """
    A non-consequential tool with no parameters that is already staged
    should NOT be auto-approved again.
    """
    # Create a tool with no parameters and consequential=False (default)
    await create_local_tool(
        local_tool_service,
        name="ping_staged",
        description="A simple ping tool with no parameters",
        parameters={},
        required=[],
    )

    tool_id = ToolId(service_name="local", tool_name="ping_staged")

    # Create a staged event representing this tool already being staged
    staged_event = EmittedEvent(
        source=EventSource.AI_AGENT,
        kind=EventKind.TOOL,
        trace_id="test-trace-id",
        data={
            "tool_calls": [
                {
                    "tool_id": tool_id.to_string(),
                    "arguments": {},
                    "result": {"data": "pong", "metadata": {}},
                }
            ]
        },
        metadata=None,
    )

    conversation_context = [
        (EventSource.CUSTOMER, "Hello, can you ping for me?"),
    ]

    interaction_history = create_interaction_history(conversation_context)

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="customer asks to ping",
            action="ping for the customer",
            score=9,
            rationale="customer wants to ping",
            tags=[Tag.for_agent_id(agent.id)],
        ): [tool_id]
    }

    inference_tool_calls_result = await _inference_tool_calls_result(
        container=container,
        agent=agent,
        interaction_history=interaction_history,
        tool_enabled_guideline_matches=tool_enabled_guideline_matches,
        staged_events=[staged_event],
    )

    # The tool should NOT be called again since it's already staged
    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 0


async def test_that_non_consequential_tool_with_parameters_uses_simplified_mode(
    container: Container,
    local_tool_service: LocalToolService,
    agent: Agent,
) -> None:
    """
    A non-consequential tool with parameters should use simplified evaluation mode
    and successfully extract parameter values from context.
    """
    tool = await local_tool_service.create_tool(
        name="get_weather",
        module_path="tests.tool_utilities",
        description="Get weather for a city",
        parameters={
            "city": {"type": "string", "description": "City name"},
        },
        required=["city"],
        consequential=False,  # Non-consequential
    )

    conversation_context = [
        (EventSource.CUSTOMER, "What's the weather in Paris?"),
    ]

    interaction_history = create_interaction_history(conversation_context)

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="customer asks about weather",
            action="get the weather for the requested city",
            score=9,
            rationale="customer wants weather info",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ToolId(service_name="local", tool_name=tool.name)]
    }

    inference_tool_calls_result = await _inference_tool_calls_result(
        container=container,
        agent=agent,
        interaction_history=interaction_history,
        tool_enabled_guideline_matches=tool_enabled_guideline_matches,
    )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]

    # Verify the parameter was extracted correctly
    assert "city" in tool_call.arguments
    city_value = str(tool_call.arguments["city"])
    assert city_value.lower() == "paris"

    # Verify simplified mode was used (schema name should be NonConsequentialToolBatchSchema)
    assert len(inference_tool_calls_result.batch_generations) == 1
    assert "NonConsequential" in inference_tool_calls_result.batch_generations[0].schema_name


async def test_that_consequential_tool_with_parameters_uses_full_mode(
    container: Container,
    local_tool_service: LocalToolService,
    agent: Agent,
) -> None:
    """
    A consequential tool should still use full evaluation mode, not simplified mode.
    """
    tool = await local_tool_service.create_tool(
        name="transfer_money",
        module_path="tests.tool_utilities",
        description="Transfer money to a recipient",
        parameters={
            "amount": {"type": "number", "description": "Amount to transfer"},
            "recipient": {"type": "string", "description": "Recipient name"},
        },
        required=["amount", "recipient"],
        consequential=True,  # Consequential - should use full mode
    )

    conversation_context = [
        (EventSource.CUSTOMER, "Transfer $100 to John please"),
    ]

    interaction_history = create_interaction_history(conversation_context)

    tool_enabled_guideline_matches = {
        create_guideline_match(
            condition="customer asks to transfer money",
            action="transfer money to the specified recipient",
            score=9,
            rationale="customer wants to transfer money",
            tags=[Tag.for_agent_id(agent.id)],
        ): [ToolId(service_name="local", tool_name=tool.name)]
    }

    inference_tool_calls_result = await _inference_tool_calls_result(
        container=container,
        agent=agent,
        interaction_history=interaction_history,
        tool_enabled_guideline_matches=tool_enabled_guideline_matches,
    )

    # Verify full mode was used (schema name should be SingleToolBatchSchema, not Simple)
    assert len(inference_tool_calls_result.batch_generations) == 1
    assert "Simple" not in inference_tool_calls_result.batch_generations[0].schema_name
    assert "SingleToolBatchSchema" in inference_tool_calls_result.batch_generations[0].schema_name
