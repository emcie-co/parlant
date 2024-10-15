import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from emcie.common.tools import ToolContext, ToolResult
from emcie.common.plugin import PluginServer, ToolEntry, tool
from pytest import fixture
import pytest
from emcie.server.core.contextual_correlator import ContextualCorrelator
from emcie.server.core.emission.event_buffer import EventBuffer, EventBufferFactory
from emcie.server.core.emissions import EventEmitter, EventEmitterFactory
from emcie.server.core.services.tools.plugins import PluginClient
from emcie.server.core.sessions import SessionId
from emcie.server.core.tools import ToolExecutionError


class SessionBuffers(EventEmitterFactory):
    def __init__(self) -> None:
        self.for_session: dict[SessionId, EventBuffer] = {}

    def create_event_emitter(self, session_id: SessionId) -> EventEmitter:
        buffer = EventBuffer()
        self.for_session[session_id] = buffer
        return buffer


@fixture
def context() -> ToolContext:
    return ToolContext(session_id=SessionId("test_session"))


@asynccontextmanager
async def run_service_server(tools: list[ToolEntry]) -> AsyncIterator[PluginServer]:
    async with PluginServer(
        tools=tools,
        host="127.0.0.1",
    ) as server:
        try:
            yield server
        finally:
            await server.shutdown()


def create_client(
    server: PluginServer,
    event_emitter_factory: EventEmitterFactory = EventBufferFactory(),
) -> PluginClient:
    return PluginClient(
        url=server.url,
        event_emitter_factory=event_emitter_factory,
        correlator=ContextualCorrelator(),
    )


async def test_that_a_plugin_with_no_configured_tools_returns_no_tools() -> None:
    async with run_service_server([]) as server:
        async with create_client(server) as client:
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


async def test_that_a_plugin_with_one_configured_tool_returns_that_tool() -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: Optional[int]) -> ToolResult:
        """My tool's description"""
        return ToolResult(arg_1 * (arg_2 or 0))

    async with run_service_server([my_tool]) as server:
        async with create_client(server) as client:
            listed_tools = await client.list_tools()
            assert len(listed_tools) == 1
            assert my_tool.tool == listed_tools[0]


async def test_that_a_plugin_reads_a_tool() -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: Optional[int]) -> ToolResult:
        """My tool's description"""
        return ToolResult(arg_1 * (arg_2 or 0))

    async with run_service_server([my_tool]) as server:
        async with create_client(server) as client:
            returned_tool = await client.read_tool(my_tool.tool.id)
            assert my_tool.tool == returned_tool


async def test_that_a_plugin_calls_a_tool(context: ToolContext) -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: int) -> ToolResult:
        return ToolResult(arg_1 * arg_2)

    async with run_service_server([my_tool]) as server:
        async with create_client(server) as client:
            result = await client.call_tool(
                my_tool.tool.id,
                context,
                arguments={"arg_1": 2, "arg_2": 4},
            )
            assert result.data == 8


async def test_that_a_plugin_calls_an_async_tool(context: ToolContext) -> None:
    @tool
    async def my_tool(context: ToolContext, arg_1: int, arg_2: int) -> ToolResult:
        return ToolResult(arg_1 * arg_2)

    async with run_service_server([my_tool]) as server:
        async with create_client(server) as client:
            result = await client.call_tool(
                my_tool.tool.id,
                context,
                arguments={"arg_1": 2, "arg_2": 4},
            )
            assert result.data == 8


async def test_that_a_plugin_tool_has_access_to_the_current_session(
    context: ToolContext,
) -> None:
    @tool
    async def my_tool(context: ToolContext) -> ToolResult:
        return ToolResult(context.session_id)

    async with run_service_server([my_tool]) as server:
        async with create_client(server) as client:
            result = await client.call_tool(
                my_tool.tool.id,
                context,
                arguments={},
            )

            assert result.data == context.session_id


async def test_that_a_plugin_tool_can_emit_events(
    context: ToolContext,
) -> None:
    @tool
    async def my_tool(context: ToolContext) -> ToolResult:
        await context.emit_status("typing", {"tool": "my_tool"})
        await context.emit_message("Hello, cherry-pie!")
        await context.emit_message("How are you?")
        return ToolResult({"number": 123})

    buffers = SessionBuffers()

    async with run_service_server([my_tool]) as server:
        async with create_client(
            server,
            event_emitter_factory=buffers,
        ) as client:
            result = await client.call_tool(
                my_tool.tool.id,
                context,
                arguments={},
            )

            emitted_events = buffers.for_session[SessionId(context.session_id)].events

            assert len(emitted_events) == 3

            assert emitted_events[0].kind == "status"
            assert emitted_events[0].data == {"status": "typing", "data": {"tool": "my_tool"}}

            assert emitted_events[1].kind == "message"
            assert emitted_events[1].data == {"message": "Hello, cherry-pie!"}

            assert emitted_events[2].kind == "message"
            assert emitted_events[2].data == {"message": "How are you?"}

            assert result.data == {"number": 123}


async def test_that_a_plugin_tool_can_emit_events_and_ultimately_fail_with_an_error(
    context: ToolContext,
) -> None:
    @tool
    async def my_tool(context: ToolContext) -> ToolResult:
        await context.emit_message("Hello, cherry-pie!")
        await context.emit_message("How are you?")
        await asyncio.sleep(1)
        raise Exception("Tool failed")

    buffers = SessionBuffers()

    async with run_service_server([my_tool]) as server:
        async with create_client(
            server,
            event_emitter_factory=buffers,
        ) as client:
            with pytest.raises(ToolExecutionError):
                await client.call_tool(
                    my_tool.tool.id,
                    context,
                    arguments={},
                )

            emitted_events = buffers.for_session[SessionId(context.session_id)].events

            assert len(emitted_events) == 2

            assert emitted_events[0].kind == "message"
            assert emitted_events[0].data == {"message": "Hello, cherry-pie!"}

            assert emitted_events[1].kind == "message"
            assert emitted_events[1].data == {"message": "How are you?"}
