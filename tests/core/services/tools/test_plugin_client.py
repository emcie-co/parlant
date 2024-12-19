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

import asyncio
from contextlib import asynccontextmanager
import enum
from typing import AsyncIterator, Mapping, Optional, cast

from parlant.core.tools import ToolContext, ToolResult, ToolResultError
from parlant.core.services.tools.plugins import PluginServer, ToolEntry, tool
from lagom import Container
from pytest import fixture, raises
import pytest
from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.emission.event_buffer import EventBuffer, EventBufferFactory
from parlant.core.emissions import EventEmitter, EventEmitterFactory
from parlant.core.services.tools.plugins import PluginClient
from parlant.core.sessions import SessionId
from parlant.core.tools import ToolExecutionError


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
    return await container[AgentStore].create_agent(name="Test Agent")


@fixture
async def context(agent: Agent) -> ToolContext:
    return ToolContext(
        agent_id=agent.id,
        session_id="test_session",
        customer_id="test_customer",
    )


@asynccontextmanager
async def run_service_server(tools: list[ToolEntry]) -> AsyncIterator[PluginServer]:
    async with PluginServer(
        tools=tools,
        port=8091,
        host="127.0.0.1",
    ) as server:
        try:
            yield server
        finally:
            await server.shutdown()


def create_client(
    server: PluginServer,
    event_emitter_factory: EventEmitterFactory,
) -> PluginClient:
    return PluginClient(
        url=server.url,
        event_emitter_factory=event_emitter_factory,
        correlator=ContextualCorrelator(),
    )


async def test_that_a_plugin_with_no_configured_tools_returns_no_tools(
    container: Container,
) -> None:
    async with run_service_server([]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            tools = await client.list_tools()
            assert not tools


async def test_that_a_decorated_tool_can_be_called_directly(context: ToolContext) -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: Optional[int]) -> ToolResult:
        """My tool's description"""
        return ToolResult(arg_1 * (arg_2 or 0))

    assert my_tool(context, 2, None).data == 0
    assert my_tool(context, 2, 1).data == 2
    assert my_tool(context, 2, 2).data == 4
    assert my_tool(context, arg_1=2, arg_2=3).data == 6


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
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: Optional[int]) -> ToolResult:
        """My tool's description"""
        return ToolResult(arg_1 * (arg_2 or 0))

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            returned_tool = await client.read_tool(my_tool.tool.name)
            assert my_tool.tool == returned_tool


async def test_that_a_plugin_calls_a_tool(context: ToolContext, container: Container) -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: int) -> ToolResult:
        return ToolResult(arg_1 * arg_2)

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            result = await client.call_tool(
                my_tool.tool.name,
                context,
                arguments={"arg_1": 2, "arg_2": 4},
            )
            assert result.data == 8


async def test_that_a_plugin_calls_an_async_tool(
    context: ToolContext,
    container: Container,
) -> None:
    @tool
    async def my_tool(context: ToolContext, arg_1: int, arg_2: int) -> ToolResult:
        return ToolResult(arg_1 * arg_2)

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            result = await client.call_tool(
                my_tool.tool.name,
                context,
                arguments={"arg_1": 2, "arg_2": 4},
            )
            assert result.data == 8


async def test_that_a_plugin_tool_has_access_to_the_current_session_agent_and_customer(
    context: ToolContext,
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
                context,
                arguments={},
            )

            data = cast(Mapping[str, str], result.data)

            assert data["session_id"] == context.session_id
            assert data["agent_id"] == context.agent_id
            assert data["customer_id"] == context.customer_id


async def test_that_a_plugin_tool_can_emit_events(
    context: ToolContext,
    container: Container,
    agent: Agent,
) -> None:
    @tool
    async def my_tool(context: ToolContext) -> ToolResult:
        await context.emit_status("typing", {"tool": "my_tool"})
        await context.emit_message("Hello, cherry-pie!")
        await context.emit_message("How are you?")
        return ToolResult({"number": 123})

    buffers = SessionBuffers(container[AgentStore])

    async with run_service_server([my_tool]) as server:
        async with create_client(
            server,
            event_emitter_factory=buffers,
        ) as client:
            result = await client.call_tool(
                my_tool.tool.name,
                context,
                arguments={},
            )

            emitted_events = buffers.for_session[SessionId(context.session_id)].events

            assert len(emitted_events) == 3

            assert emitted_events[0].kind == "status"
            assert emitted_events[0].data == {"status": "typing", "data": {"tool": "my_tool"}}

            assert emitted_events[1].kind == "message"
            assert emitted_events[1].data == {
                "message": "Hello, cherry-pie!",
                "participant": {"id": agent.id, "display_name": agent.name},
            }

            assert emitted_events[2].kind == "message"
            assert emitted_events[2].data == {
                "message": "How are you?",
                "participant": {"id": agent.id, "display_name": agent.name},
            }

            assert result.data == {"number": 123}


async def test_that_a_plugin_tool_can_emit_events_and_ultimately_fail_with_an_error(
    context: ToolContext,
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
                    context,
                    arguments={},
                )

            emitted_events = buffers.for_session[SessionId(context.session_id)].events

            assert len(emitted_events) == 2

            assert emitted_events[0].kind == "message"
            assert emitted_events[0].data == {
                "message": "Hello, cherry-pie!",
                "participant": {"id": agent.id, "display_name": agent.name},
            }

            assert emitted_events[1].kind == "message"
            assert emitted_events[1].data == {
                "message": "How are you?",
                "participant": {"id": agent.id, "display_name": agent.name},
            }


async def test_that_a_plugin_tool_with_enum_parameter_can_be_called(
    context: ToolContext,
    container: Container,
    agent: Agent,
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
                context,
                arguments={"category": "category_a"},
            )

            assert result.data == "category_a"


async def test_that_a_plugin_calls_a_tool_with_an_optional_param(
    context: ToolContext, container: Container
) -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: Optional[int] = None) -> ToolResult:
        assert arg_2
        return ToolResult(arg_1 * arg_2)

    async with run_service_server([my_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            result = await client.call_tool(
                my_tool.tool.name,
                context,
                arguments={"arg_1": 2, "arg_2": 4},
            )
            assert result.data == 8


async def test_that_a_plugin_tool_that_returns_a_huge_payload_raises_an_error(
    context: ToolContext,
    container: Container,
    agent: Agent,
) -> None:
    @tool
    def huge_payload_tool(context: ToolContext) -> ToolResult:
        huge_payload = {f"key_{i}": "value" for i in range(10000)}
        return ToolResult({"size": len(huge_payload), "payload": huge_payload})

    async with run_service_server([huge_payload_tool]) as server:
        async with create_client(server, container[EventBufferFactory]) as client:
            tools = await client.list_tools()

            assert tools
            with raises(ToolResultError) as exc:
                await client.call_tool(huge_payload_tool.tool.name, context, arguments={})

            assert "Response exceeds 16KB limit" in str(exc.value)
