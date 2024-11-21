from datetime import datetime, timezone
import enum
from typing import Any, cast
from lagom import Container
from pytest import fixture

from parlant.core.agents import Agent
from parlant.core.common import generate_id
from parlant.core.end_users import EndUser, EndUserStore, EndUserId
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
def local_service_tool(container: Container) -> LocalToolService:
    return container[LocalToolService]


@fixture
def tool_caller(container: Container) -> ToolCaller:
    return ToolCaller(
        container[Logger],
        container[ServiceRegistry],
        container[SchematicGenerator[ToolCallInferenceSchema]],
    )


@fixture
async def end_user(container: Container, end_user_id: EndUserId) -> EndUser:
    return await container[EndUserStore].read_end_user(end_user_id)


def create_interaction_history(conversation_context: list[tuple[str, str]]) -> list[Event]:
    return [
        create_event_message(
            offset=i,
            source=cast(EventSource, source),
            message=message,
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
    end_user: EndUser,
    local_service_tool: LocalToolService,
    tool_caller: ToolCaller,
    agent: Agent,
) -> None:
    tool = await create_local_tool(
        local_service_tool,
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
        ("end_user", "Are you selling computers products?"),
        ("ai_agent", "Yes"),
        ("end_user", "What available keyboards do you have?"),
    ]

    interaction_history = create_interaction_history(conversation_context)

    ordinary_guideline_propositions = [
        create_guideline_proposition(
            condition="user asking a question",
            action="response in concise and breif answer",
            score=9,
            rationale="user ask a question of what available keyboard do we have",
        )
    ]

    tool_enabled_guideline_propositions = {
        create_guideline_proposition(
            condition="get all products by a specific category",
            action="a user asks for the availability of products from a certain category",
            score=9,
            rationale="user asks for keyboards availability",
        ): [ToolId(service_name="local", tool_name=tool.name)]
    }

    _, tool_calls = await tool_caller.infer_tool_calls(
        agents=[agent],
        user_tags_pair=(end_user, []),
        context_variables=[],
        interaction_history=interaction_history,
        terms=[],
        ordinary_guideline_propositions=ordinary_guideline_propositions,
        tool_enabled_guideline_propositions=tool_enabled_guideline_propositions,
        staged_events=[],
    )

    assert len(tool_calls) == 1
    tool_call = tool_calls[0]

    assert "category" in tool_call.arguments
    assert tool_call.arguments["category"] == "peripherals"


async def test_that_a_tool_from_plugin_is_getting_called_with_an_enum_parameter(
    container: Container,
    end_user: EndUser,
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
        ("end_user", "Are you selling computers products?"),
        ("ai_agent", "Yes"),
        ("end_user", "What available keyboards do you have?"),
    ]

    interaction_history = create_interaction_history(conversation_context)

    ordinary_guideline_propositions = [
        create_guideline_proposition(
            condition="user asking a question",
            action="response in concise and breif answer",
            score=9,
            rationale="user ask a question of what available keyboard do we have",
        )
    ]

    tool_enabled_guideline_propositions = {
        create_guideline_proposition(
            condition="get all products by a specific category",
            action="a user asks for the availability of products from a certain category",
            score=9,
            rationale="user asks for keyboards availability",
        ): [ToolId(service_name="my_sdk_service", tool_name="available_products_by_category")]
    }

    async with run_service_server([available_products_by_category]) as server:
        await service_registry.update_tool_service(
            name="my_sdk_service",
            kind="sdk",
            url=server.url,
        )

        _, tool_calls = await tool_caller.infer_tool_calls(
            agents=[agent],
            user_tags_pair=(end_user, []),
            context_variables=[],
            interaction_history=interaction_history,
            terms=[],
            ordinary_guideline_propositions=ordinary_guideline_propositions,
            tool_enabled_guideline_propositions=tool_enabled_guideline_propositions,
            staged_events=[],
        )

    assert len(tool_calls) == 1
    tool_call = tool_calls[0]

    assert "category" in tool_call.arguments
    assert tool_call.arguments["category"] == "peripherals"
