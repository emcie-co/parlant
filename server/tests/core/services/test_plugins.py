from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from emcie.common.tools import ToolContext, ToolResult
from emcie.common.plugin import PluginServer, ToolEntry, tool
from pytest import fixture
from emcie.server.core.services.plugins import PluginClient
from emcie.server.core.sessions import SessionId


@fixture
def context() -> ToolContext:
    return ToolContext(session_id=SessionId("test_session"))


@asynccontextmanager
async def run_plugin_server(tools: list[ToolEntry]) -> AsyncIterator[PluginServer]:
    async with PluginServer(name="test_plugin", tools=tools, host="127.0.0.1") as server:
        try:
            yield server
        finally:
            await server.shutdown()


async def test_that_a_plugin_with_no_configured_tools_returns_no_tools() -> None:
    async with run_plugin_server([]) as server:
        async with PluginClient(server.url) as client:
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

    async with run_plugin_server([my_tool]) as server:
        async with PluginClient(server.url) as client:
            listed_tools = await client.list_tools()
            assert len(listed_tools) == 1
            assert my_tool.tool == listed_tools[0]


async def test_that_a_plugin_reads_a_tool() -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: Optional[int]) -> ToolResult:
        """My tool's description"""
        return ToolResult(arg_1 * (arg_2 or 0))

    async with run_plugin_server([my_tool]) as server:
        async with PluginClient(server.url) as client:
            returned_tool = await client.read_tool(my_tool.tool.id)
            assert my_tool.tool == returned_tool


async def test_that_a_plugin_calls_a_tool(context: ToolContext) -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: int) -> ToolResult:
        return ToolResult(arg_1 * arg_2)

    async with run_plugin_server([my_tool]) as server:
        async with PluginClient(server.url) as client:
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

    async with run_plugin_server([my_tool]) as server:
        async with PluginClient(server.url) as client:
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

    async with run_plugin_server([my_tool]) as server:
        async with PluginClient(server.url) as client:
            result = await client.call_tool(
                my_tool.tool.id,
                context,
                arguments={},
            )

            assert result.data == context.session_id
