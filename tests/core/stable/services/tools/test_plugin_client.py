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

import asyncio
from datetime import datetime
import enum
import json
from typing import Annotated, Any, Mapping, Optional, cast
from lagom import Container
from pydantic import BaseModel
from pytest import fixture, raises
import pytest

from parlant.core.loggers import StdoutLogger
from parlant.core.tools import (
    ToolContext,
    ToolError,
    ToolParameterOptions,
    ToolResult,
    ToolResultError,
    ToolOverlap,
)
from parlant.core.services.tools.plugins import PluginServer, tool
from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.tracer import LocalTracer
from parlant.core.emission.event_buffer import EventBuffer, EventBufferFactory
from parlant.core.emissions import EventEmitter, EventEmitterFactory
from parlant.core.services.tools.plugins import PluginClient
from parlant.core.sessions import SessionId, EventKind
from parlant.core.tools import ToolExecutionError
from tests.test_utilities import run_service_server


class SessionBuffers(EventEmitterFactory):
    def __init__(self, agent_store: AgentStore) -> None:
        self.agent_store = agent_store
        self.for_session: dict[SessionId, EventBuffer] = {}

    async def create_event_emitter(
        self,
        emitting_agent_id: AgentId,
        session_id: SessionId,
    ) -> EventEmitter:
        agent = await self.agent_store.read_agent(emitting_agent_id)
        buffer = EventBuffer(emitting_agent=agent)
        self.for_session[session_id] = buffer
        return buffer


@fixture
async def agent(container: Container) -> Agent:
    return await container[AgentStore].create_agent(
        name="Test Agent",
        max_engine_iterations=2,
    )


@fixture
async def tool_context(agent: Agent) -> ToolContext:
    return ToolContext(
        agent_id=agent.id,
        session_id="test_session",
        customer_id="test_customer",
    )


def create_client(
    server: PluginServer,
    event_emitter_factory: EventEmitterFactory,
) -> PluginClient:
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    return PluginClient(
        url=server.url,
        event_emitter_factory=event_emitter_factory,
        logger=logger,
        tracer=tracer,
    )


async def test_that_optional_tool_parameters_are_marked_as_optional() -> None:
    @tool
    def my_tool(
        context: ToolContext,
        arg_1: int,
        arg_2: Optional[int] = None,
        arg_3: int | None = None,
    ) -> ToolResult:
        return ToolResult({})

    assert len(my_tool.tool.required) == 1
    assert "arg_1" in my_tool.tool.required


async def test_that_a_plugin_with_no_configured_tools_returns_no_tools(
    container: Container,
) -> None:
    async with run_service_server([]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            tools = await client.list_tools()
            assert not tools


async def test_that_a_decorated_tool_can_be_called_directly(tool_context: ToolContext) -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: Optional[int]) -> ToolResult:
        """My tool's description"""
        return ToolResult(arg_1 * (arg_2 or 0))

    assert my_tool(tool_context, 2, None).data == 0
    assert my_tool(tool_context, 2, 1).data == 2
    assert my_tool(tool_context, 2, 2).data == 4
    assert my_tool(tool_context, arg_1=2, arg_2=3).data == 6


async def test_that_a_plugin_with_one_configured_tool_returns_that_tool(
    container: Container,
) -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: Optional[int]) -> ToolResult:
        """My tool's description"""
        return ToolResult(arg_1 * (arg_2 or 0))

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            listed_tools = await client.list_tools()
            assert len(listed_tools) == 1
            assert my_tool.tool == listed_tools[0]


async def test_that_a_plugin_reads_a_tool(container: Container) -> None:
    @tool(metadata={"test-metadata": {"one": 1}})
    def my_tool(context: ToolContext, arg_1: int, arg_2: Optional[int]) -> ToolResult:
        """My tool's description"""
        return ToolResult(arg_1 * (arg_2 or 0))

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            returned_tool = await client.read_tool(my_tool.tool.name)
            assert my_tool.tool.name == returned_tool.name
            assert my_tool.tool.description == returned_tool.description
            assert my_tool.tool.metadata == returned_tool.metadata
            assert my_tool.tool.required == returned_tool.required

            for param_name, (param_descriptor, param_options) in my_tool.tool.parameters.items():
                (returned_param_descriptor, returned_param_options) = returned_tool.parameters[
                    param_name
                ]
                assert param_descriptor == returned_param_descriptor

                for option_name, option_field in ToolParameterOptions.model_fields.items():
                    if not option_field.exclude:
                        assert (
                            param_options.model_dump()[option_name]
                            == returned_param_options.model_dump()[option_name]
                        )


async def test_that_a_plugin_calls_a_tool(tool_context: ToolContext, container: Container) -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: int) -> ToolResult:
        return ToolResult(arg_1 * arg_2)

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            result = await client.call_tool(
                my_tool.tool.name,
                tool_context,
                arguments={"arg_1": 2, "arg_2": 4},
            )
            assert result.data == 8


async def test_that_a_plugin_raises_an_informative_exception_if_tool_call_failed_on_server_side(
    tool_context: ToolContext,
    container: Container,
) -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: int) -> ToolResult:
        raise Exception("Bananas are tasty")

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            try:
                await client.call_tool(
                    my_tool.tool.name,
                    tool_context,
                    arguments={"arg_1": 2, "arg_2": 4},
                )
            except Exception as exc:
                assert "Bananas are tasty" in str(exc)
                return
            assert False, "Expected exception was not raised"


async def test_that_a_plugin_calls_an_async_tool(
    tool_context: ToolContext,
    container: Container,
) -> None:
    @tool
    async def my_tool(context: ToolContext, arg_1: int, arg_2: int) -> ToolResult:
        return ToolResult(arg_1 * arg_2)

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            result = await client.call_tool(
                my_tool.tool.name,
                tool_context,
                arguments={"arg_1": 2, "arg_2": 4},
            )
            assert result.data == 8


async def test_that_a_plugin_tool_has_access_to_the_current_session_agent_and_customer(
    tool_context: ToolContext,
    container: Container,
) -> None:
    @tool
    async def my_tool(context: ToolContext) -> ToolResult:
        return ToolResult(
            {
                "session_id": context.session_id,
                "agent_id": context.agent_id,
                "customer_id": context.customer_id,
            }
        )

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            result = await client.call_tool(
                my_tool.tool.name,
                tool_context,
                arguments={},
            )

            data = cast(Mapping[str, str], result.data)

            assert data["session_id"] == tool_context.session_id
            assert data["agent_id"] == tool_context.agent_id
            assert data["customer_id"] == tool_context.customer_id


async def test_that_a_plugin_tool_can_emit_events(
    tool_context: ToolContext,
    container: Container,
    agent: Agent,
) -> None:
    @tool
    async def my_tool(context: ToolContext) -> ToolResult:
        await context.emit_status("typing", {"tool": "my_tool"})
        await context.emit_message("Hello, cherry-pie!")
        await context.emit_message("How are you?")
        await context.emit_custom({"Custom": "Event"})
        return ToolResult({"number": 123})

    buffers = SessionBuffers(container[AgentStore])

    async with run_service_server([my_tool]) as server:
        async with create_client(
            server,
            event_emitter_factory=buffers,
        ) as client:
            result = await client.call_tool(
                my_tool.tool.name,
                tool_context,
                arguments={},
            )

            emitted_events = buffers.for_session[SessionId(tool_context.session_id)].events

            assert len(emitted_events) == 4

            assert emitted_events[0].kind == EventKind.STATUS
            assert emitted_events[0].data == {"status": "typing", "data": {"tool": "my_tool"}}

            assert emitted_events[1].kind == EventKind.MESSAGE
            assert emitted_events[1].data == {
                "message": "Hello, cherry-pie!",
                "participant": {"id": agent.id, "display_name": agent.name},
            }

            assert emitted_events[2].kind == EventKind.MESSAGE
            assert emitted_events[2].data == {
                "message": "How are you?",
                "participant": {"id": agent.id, "display_name": agent.name},
            }

            assert emitted_events[3].kind == EventKind.CUSTOM
            assert emitted_events[3].data == {"Custom": "Event"}

            assert result.data == {"number": 123}


async def test_that_a_plugin_tool_can_emit_events_and_ultimately_fail_with_an_error(
    tool_context: ToolContext,
    container: Container,
    agent: Agent,
) -> None:
    @tool
    async def my_tool(context: ToolContext) -> ToolResult:
        await context.emit_message("Hello, cherry-pie!")
        await context.emit_message("How are you?")
        await asyncio.sleep(1)
        raise Exception("Tool failed")

    buffers = SessionBuffers(container[AgentStore])

    async with run_service_server([my_tool]) as server:
        async with create_client(
            server,
            event_emitter_factory=buffers,
        ) as client:
            with pytest.raises(ToolExecutionError):
                await client.call_tool(
                    my_tool.tool.name,
                    tool_context,
                    arguments={},
                )

            emitted_events = buffers.for_session[SessionId(tool_context.session_id)].events

            assert len(emitted_events) == 2

            assert emitted_events[0].kind == EventKind.MESSAGE
            assert emitted_events[0].data == {
                "message": "Hello, cherry-pie!",
                "participant": {"id": agent.id, "display_name": agent.name},
            }

            assert emitted_events[1].kind == EventKind.MESSAGE
            assert emitted_events[1].data == {
                "message": "How are you?",
                "participant": {"id": agent.id, "display_name": agent.name},
            }


async def test_that_a_plugin_tool_with_enum_parameter_can_be_called(
    tool_context: ToolContext,
    container: Container,
) -> None:
    class ProductCategory(enum.Enum):
        CATEGORY_A = "category_a"
        CATEGORY_B = "category_b"

    @tool
    async def my_enum_tool(context: ToolContext, category: ProductCategory) -> ToolResult:
        return ToolResult(category.value)

    async with run_service_server([my_enum_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            tools = await client.list_tools()

            assert tools
            result = await client.call_tool(
                my_enum_tool.tool.name,
                tool_context,
                arguments={"category": "category_a"},
            )

            assert result.data == "category_a"


async def test_that_a_plugin_tool_with_optional_enum_parameter_can_be_called(
    tool_context: ToolContext,
    container: Container,
) -> None:
    class ProductCategory(enum.Enum):
        CATEGORY_A = "category_a"
        CATEGORY_B = "category_b"

    @tool
    async def my_enum_tool(context: ToolContext, category: Optional[ProductCategory]) -> ToolResult:
        return ToolResult({})

    async with run_service_server([my_enum_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            tools = await client.list_tools()

            assert tools
            result = await client.call_tool(
                my_enum_tool.tool.name,
                tool_context,
                arguments={"category": None},
            )

            assert result.data == {}


async def test_that_a_plugin_tool_with_enum_list_parameter_can_be_called(
    tool_context: ToolContext,
    container: Container,
) -> None:
    class ProductCategory(enum.Enum):
        CATEGORY_A = "category_a"
        CATEGORY_B = "category_b"

    @tool
    async def my_enum_tool(context: ToolContext, categories: list[ProductCategory]) -> ToolResult:
        return ToolResult(",".join(c.value for c in categories))

    async with run_service_server([my_enum_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            tools = await client.list_tools()

            assert tools
            result = await client.call_tool(
                my_enum_tool.tool.name,
                tool_context,
                arguments={"categories": ["category_a", "category_b"]},
            )

            assert result.data == "category_a,category_b"


async def test_that_a_plugin_tool_with_datetime_parameter_can_be_called(
    tool_context: ToolContext,
    container: Container,
) -> None:
    @tool
    async def my_tool(context: ToolContext, date: datetime) -> ToolResult:
        return ToolResult(date.day)

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            tools = await client.list_tools()

            assert tools
            result = await client.call_tool(
                my_tool.tool.name,
                tool_context,
                arguments={"date": "2025-01-01"},
            )

            assert result.data == 1


async def test_that_a_plugin_tool_with_base_model_parameter_can_be_called(
    tool_context: ToolContext,
    container: Container,
) -> None:
    class Person(BaseModel):
        name: str
        age: int

    @tool
    async def my_tool(context: ToolContext, person: Person) -> ToolResult:
        return ToolResult(f"{person.name} {person.age}")

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            tools = await client.list_tools()

            assert tools
            result = await client.call_tool(
                my_tool.tool.name,
                tool_context,
                arguments={"person": json.dumps({"name": "Dor", "age": 32})},
            )

            assert result.data == "Dor 32"


async def test_that_a_plugin_calls_a_tool_with_an_optional_param(
    tool_context: ToolContext,
    container: Container,
) -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: Optional[int] = None) -> ToolResult:
        assert arg_2
        return ToolResult(arg_1 * arg_2)

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            result = await client.call_tool(
                my_tool.tool.name,
                tool_context,
                arguments={"arg_1": 2, "arg_2": 4},
            )
            assert result.data == 8


async def test_that_a_plugin_calls_a_tool_with_an_optional_param_and_a_None_arg(
    tool_context: ToolContext,
    container: Container,
) -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: Optional[int] = None) -> ToolResult:
        if not arg_2:
            arg_2 = 1
        return ToolResult(arg_1 * arg_2)

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            result = await client.call_tool(
                my_tool.tool.name,
                tool_context,
                arguments={"arg_1": 2, "arg_2": None},
            )
            assert result.data == 2


async def test_that_a_plugin_tool_with_an_optional_base_model_parameter_can_be_called(
    tool_context: ToolContext,
    container: Container,
) -> None:
    class Person(BaseModel):
        name: str
        age: int

    @tool
    async def my_tool(context: ToolContext, person: Optional[Person] = None) -> ToolResult:
        assert person
        return ToolResult(f"{person.name} {person.age}")

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            tools = await client.list_tools()

            assert tools
            result = await client.call_tool(
                my_tool.tool.name,
                tool_context,
                arguments={"person": json.dumps({"name": "Dor", "age": 32})},
            )

            assert result.data == "Dor 32"


async def test_that_a_plugin_tool_with_an_optional_base_model_parameter_and_a_None_value_can_be_called(
    tool_context: ToolContext,
    container: Container,
) -> None:
    class Person(BaseModel):
        name: str
        age: int

    @tool
    async def my_tool(context: ToolContext, person: Optional[Person] = None) -> ToolResult:
        return ToolResult(person is None)

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            tools = await client.list_tools()

            assert tools
            result = await client.call_tool(
                my_tool.tool.name,
                tool_context,
                arguments={"person": None},
            )

            assert result.data


async def test_that_a_plugin_calls_a_tool_with_a_union_param(
    tool_context: ToolContext,
    container: Container,
) -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: int | None = None) -> ToolResult:
        assert arg_2
        return ToolResult(arg_1 * arg_2)

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            result = await client.call_tool(
                my_tool.tool.name,
                tool_context,
                arguments={"arg_1": 2, "arg_2": 4},
            )
            assert result.data == 8


async def test_that_a_plugin_tool_with_an_annotated_enum_parameter_can_be_called(
    tool_context: ToolContext,
    container: Container,
) -> None:
    class ProductCategory(enum.Enum):
        CATEGORY_A = "category_a"
        CATEGORY_B = "category_b"

    @tool
    async def my_enum_tool(
        context: ToolContext,
        category: Annotated[ProductCategory, ToolParameterOptions()],
    ) -> ToolResult:
        return ToolResult(category.value)

    async with run_service_server([my_enum_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            tools = await client.list_tools()

            assert tools
            result = await client.call_tool(
                my_enum_tool.tool.name,
                tool_context,
                arguments={"category": "category_a"},
            )

            assert result.data == "category_a"


async def test_that_a_plugin_calls_a_tool_with_an_annotated_optional_param(
    tool_context: ToolContext,
    container: Container,
) -> None:
    @tool
    def my_tool(
        context: ToolContext,
        arg_1: int,
        arg_2: Annotated[Optional[int], ToolParameterOptions()] = None,
    ) -> ToolResult:
        assert arg_2
        return ToolResult(arg_1 * arg_2)

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            result = await client.call_tool(
                my_tool.tool.name,
                tool_context,
                arguments={"arg_1": 2, "arg_2": 4},
            )
            assert result.data == 8


async def test_that_a_plugin_calls_a_tool_with_an_annotated_optional_param_and_a_None_arg(
    tool_context: ToolContext,
    container: Container,
) -> None:
    @tool
    def my_tool(
        context: ToolContext,
        arg_1: int,
        arg_2: Annotated[Optional[int], ToolParameterOptions()] = None,
    ) -> ToolResult:
        if not arg_2:
            arg_2 = 1
        return ToolResult(arg_1 * arg_2)

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            result = await client.call_tool(
                my_tool.tool.name,
                tool_context,
                arguments={"arg_1": 2, "arg_2": None},
            )
            assert result.data == 2


async def test_that_a_plugin_calls_a_tool_with_an_annotated_union_param(
    tool_context: ToolContext,
    container: Container,
) -> None:
    @tool
    def my_tool(
        context: ToolContext,
        arg_1: int,
        arg_2: Annotated[int | None, ToolParameterOptions()] = None,
    ) -> ToolResult:
        assert arg_2
        return ToolResult(arg_1 * arg_2)

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            result = await client.call_tool(
                my_tool.tool.name,
                tool_context,
                arguments={"arg_1": 2, "arg_2": 4},
            )
            assert result.data == 8


async def test_that_a_plugin_tool_that_returns_a_huge_payload_raises_an_error(
    tool_context: ToolContext,
    container: Container,
) -> None:
    @tool
    def huge_payload_tool(context: ToolContext) -> ToolResult:
        huge_payload = {f"key_{i}": "value" for i in range(10000)}
        return ToolResult({"size": len(huge_payload), "payload": huge_payload})

    async with run_service_server([huge_payload_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            with raises(ToolResultError) as exc:
                await client.call_tool(huge_payload_tool.tool.name, tool_context, arguments={})

            assert "Response exceeds 16KB limit" in str(exc.value)


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"paramA": 123, "paramX": 999},
    ],
)
async def test_that_a_plugin_raises_tool_error_for_argument_mismatch(
    tool_context: ToolContext,
    container: Container,
    arguments: dict[str, Any],
) -> None:
    @tool
    def mismatch_tool(context: ToolContext, paramA: int) -> ToolResult:
        return ToolResult(paramA)

    async with run_service_server([mismatch_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool(
                    mismatch_tool.tool.name,
                    tool_context,
                    arguments=arguments,
                )

            error_msg = str(exc_info.value)
            assert "Expected parameters" in error_msg


@pytest.mark.parametrize(
    "arguments",
    [
        {"paramA": "True"},
        {"paramA": "true"},
        {"paramA": "not_an_int"},
    ],
)
async def test_that_a_plugin_raises_tool_error_for_type_mismatch(
    tool_context: ToolContext,
    container: Container,
    arguments: dict[str, Any],
) -> None:
    @tool
    def typed_tool(context: ToolContext, paramA: int) -> ToolResult:
        return ToolResult(paramA)

    async with run_service_server([typed_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool(
                    typed_tool.tool.name,
                    tool_context,
                    arguments=arguments,
                )

            error_msg = str(exc_info.value)
            assert "paramA" in error_msg
            assert (
                "Expected" in error_msg
                or "must be" in error_msg
                or "Failed to convert" in error_msg
            )


@pytest.mark.asyncio
async def test_that_a_plugin_tool_can_return_canned_responses(
    tool_context: ToolContext,
    container: Container,
) -> None:
    canned_responses = [
        "This is a test canned response with {field_name}",
        "Another canned response for testing",
    ]

    @tool
    async def canned_response_tool(context: ToolContext) -> ToolResult:
        return ToolResult({"message": "Executed successfully"}, canned_responses=canned_responses)

    async with run_service_server([canned_response_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            result = await client.call_tool(
                canned_response_tool.tool.name, tool_context, arguments={}
            )

            assert result.canned_responses
            assert len(result.canned_responses) == 2

            assert canned_responses[0] in result.canned_responses
            assert canned_responses[1] in result.canned_responses


async def test_that_tool_decorator_has_default_overlap_auto() -> None:
    @tool
    def my_tool(context: ToolContext) -> ToolResult:
        return ToolResult({})

    assert my_tool.tool.overlap == ToolOverlap.AUTO


async def test_that_tool_decorator_can_set_overlap() -> None:
    @tool(overlap=ToolOverlap.NONE)
    def my_tool(context: ToolContext) -> ToolResult:
        return ToolResult({})

    assert my_tool.tool.overlap == ToolOverlap.NONE
