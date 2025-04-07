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
from typing import Annotated, Any, Optional, cast
from lagom import Container
from pytest import fixture

from parlant.core.agents import Agent
from parlant.core.common import generate_id
from parlant.core.customers import Customer, CustomerStore, CustomerId
from parlant.core.engines.alpha.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.tool_caller import (
    ToolCallInferenceSchema,
    ToolCaller,
    ToolCallEvaluation,
    ArgumentEvaluation,
)
from parlant.core.guidelines import Guideline, GuidelineId, GuidelineContent
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.tools.plugins import tool
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import Event, EventSource
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
        )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]

    assert "category" in tool_call.arguments
    assert tool_call.arguments["category"] == "peripherals"


async def test_that_a_plugin_tool_is_called_with_required_parameters_with_default_value(
    container: Container,
    tool_caller: ToolCaller,
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
        ("customer", "I want to set up an appointment tomorrow at 10am"),
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
        )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert "when" in tool_call.arguments


async def test_that_a_tool_is_properly_called_with_parameters_that_are_inferred_as_none(
    container: Container,
    tool_caller: ToolCaller,
    agent: Agent,
) -> None:
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
        meeting_owner: Optional[str] = "Donald Trump",
    ) -> ToolResult:
        return ToolResult("Successfully scheduled appointment")

    tool_id = ToolId(service_name="my_appointment_service", tool_name="schedule_appointment")
    tool_ = Tool(
        name="schedule_appointment",
        creation_utc=datetime.now(timezone.utc),
        description="",
        metadata={},
        parameters={
            "when": ({"type": "string"}, ToolParameterOptions()),
            "type": (
                {"type": "string", "enum": ["general", "checkup", "result"], "has_default": True},
                ToolParameterOptions(),
            ),
            "room": (
                {
                    "type": "string",
                    "enum": ["phone booth", "private room", "meeting room"],
                    "has_default": True,
                },
                ToolParameterOptions(),
            ),
            "number_of_invites": (
                {"type": "integer", "has_default": True},
                ToolParameterOptions(),
            ),
            "required_participants": (
                {"type": "array", "item_type": "string", "has_default": True},
                ToolParameterOptions(),
            ),
            "meeting_owner": (
                {"type": "string", "has_default": True},
                ToolParameterOptions(),
            ),
        },
        required=["when", "room", "number_of_invites", "required_participants"],
        consequential=False,
    )

    candidate_descriptor: tuple[ToolId, Tool, list[GuidelineMatch]] = (
        tool_id,
        tool_,
        [],
    )

    # Create a mock inference output (based on a real one with slight modifications)
    inference_output = [
        ToolCallEvaluation(
            applicability_rationale="The customer explicitly requested to schedule an appointment for tomorrow at 10am, which aligns with the purpose of this tool.",
            applicability_score=10,
            argument_evaluations=[
                ArgumentEvaluation(
                    parameter_name="when",
                    acceptable_source_for_this_argument_according_to_its_tool_definition="This argument can be extracted in the best way you think",
                    evaluate_is_it_provided_by_an_acceptable_source="Yes, the customer explicitly mentioned 'tomorrow at 10am'.",
                    evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer has already provided this information clearly.",
                    evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be problematic to guess, but the customer has provided the necessary details.",
                    is_optional=False,
                    is_missing=False,
                    value_as_string="tomorrow at 10am",
                ),
                ArgumentEvaluation(
                    parameter_name="type",
                    acceptable_source_for_this_argument_according_to_its_tool_definition="This argument can be extracted in the best way you think",
                    evaluate_is_it_provided_by_an_acceptable_source="Yes",
                    evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer has already provided this information clearly.",
                    evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be problematic to guess, but the customer has provided the necessary details.",
                    is_optional=True,
                    is_missing=False,
                    value_as_string=None,
                ),
                ArgumentEvaluation(
                    parameter_name="room",
                    acceptable_source_for_this_argument_according_to_its_tool_definition="This argument can be extracted in the best way you think",
                    evaluate_is_it_provided_by_an_acceptable_source="Yes",
                    evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer has already provided this information clearly.",
                    evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be problematic to guess, but the customer has provided the necessary details.",
                    is_optional=False,
                    is_missing=False,
                    value_as_string=None,
                ),
                ArgumentEvaluation(
                    parameter_name="number_of_invites",
                    acceptable_source_for_this_argument_according_to_its_tool_definition="This argument can be extracted in the best way you think",
                    evaluate_is_it_provided_by_an_acceptable_source="Yes",
                    evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer has already provided this information clearly.",
                    evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be problematic to guess, but the customer has provided the necessary details.",
                    is_optional=False,
                    is_missing=False,
                    value_as_string=None,
                ),
                ArgumentEvaluation(
                    parameter_name="required_participants",
                    acceptable_source_for_this_argument_according_to_its_tool_definition="This argument can be extracted in the best way you think",
                    evaluate_is_it_provided_by_an_acceptable_source="Yes",
                    evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer has already provided this information clearly.",
                    evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be problematic to guess, but the customer has provided the necessary details.",
                    is_optional=False,
                    is_missing=False,
                    value_as_string=None,
                ),
                ArgumentEvaluation(
                    parameter_name="meeting_owner",
                    acceptable_source_for_this_argument_according_to_its_tool_definition="This argument can be extracted in the best way you think",
                    evaluate_is_it_provided_by_an_acceptable_source="Yes",
                    evaluate_was_it_already_provided_and_should_it_be_provided_again="The customer has already provided this information clearly.",
                    evaluate_is_it_potentially_problematic_to_guess_what_the_value_is_if_it_isnt_provided="It would be problematic to guess, but the customer has provided the necessary details.",
                    is_optional=True,
                    is_missing=False,
                    value_as_string=None,
                ),
            ],
            same_call_is_already_staged=False,
            comparison_with_rejected_tools_including_references_to_subtleties="There are no rejected tools that are more suitable for this task.",
            relevant_subtleties="The customer has provided the required 'when' parameter but no optional details.",
            a_rejected_tool_would_have_been_a_better_fit_if_it_werent_already_rejected=False,
            are_optional_arguments_missing=True,
            are_non_optional_arguments_missing=False,
            allowed_to_run_without_optional_arguments_even_if_they_are_missing=True,
            should_run=True,
        )
    ]

    # Evaluate the tool calls
    tool_calls, missing_data = await tool_caller._evaluate_tool_calls_parameters(
        inference_output, candidate_descriptor
    )

    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    # Expecting that only the parameters that were originally defined as Optional to be passed as None (type and meeting_owner)
    assert "when" in tool_call.arguments and tool_call.arguments["when"] is not None
    assert "type" in tool_call.arguments and tool_call.arguments["type"] is None
    assert "room" not in tool_call.arguments
    assert "number_of_invites" not in tool_call.arguments
    assert "required_participants" not in tool_call.arguments
    assert "meeting_owner" in tool_call.arguments and tool_call.arguments["meeting_owner"] is None


async def test_that_a_tool_from_a_plugin_gets_called_with_an_enum_list_parameter(
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
        context: ToolContext, categories: list[ProductCategory]
    ) -> ToolResult:
        products_by_category = {
            ProductCategory.LAPTOPS: ["Lenovo", "Dell"],
            ProductCategory.PERIPHERALS: ["Razer Keyboard", "Logitech Mouse"],
        }

        return ToolResult([products_by_category[category] for category in categories])

    conversation_context = [
        ("customer", "Are you selling computers products?"),
        ("ai_agent", "Yes"),
        ("customer", "What available keyboards and laptops do you have?"),
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
    tool_caller: ToolCaller,
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
        ("customer", "Are you selling computers products?"),
        ("ai_agent", "Yes"),
        ("customer", "What available keyboards and laptops do you have?"),
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
        )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]

    assert "categories" in tool_call.arguments
    assert isinstance(tool_call.arguments["categories"], str)
    assert "laptops" in tool_call.arguments["categories"]
    assert "peripherals" in tool_call.arguments["categories"]


async def test_that_a_tool_from_a_plugin_with_missing_parameters_returns_the_missing_ones_by_precedence(
    container: Container,
    tool_caller: ToolCaller,
    agent: Agent,
) -> None:
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
            "customer",
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
        )

    tool_calls = list(chain.from_iterable(inference_tool_calls_result.batches))

    assert len(tool_calls) == 0
    # Check missing parameters by name
    missing_parameters = set(
        map(lambda x: x.parameter, inference_tool_calls_result.insights.missing_data)
    )
    assert missing_parameters == {"full_name", "city", "street", "house_number"}
