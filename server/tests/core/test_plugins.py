from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from emcie.common.plugin import PluginServer, ToolContext, ToolEntry, tool
from emcie.server.core.plugins import PluginClient


@asynccontextmanager
async def run_plugin_server(tools: list[ToolEntry]) -> AsyncIterator[PluginServer]:
    async with PluginServer(name="test_plugin", tools=tools, host="127.0.0.1") as server:
        try:
            yield server
        finally:
            await server.shutdown()


async def test_that_a_plugin_with_no_configured_tools_returns_no_tools() -> None:
    async with run_plugin_server([]) as server:
        async with PluginClient(server.host, server.port) as client:
            tools = await client.list_tools()
            assert not tools


async def test_that_a_plugin_with_one_configured_tool_returns_that_tool() -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: Optional[int]) -> int:
        """My tool's description"""
        return arg_1 * (arg_2 or 0)

    async with run_plugin_server([my_tool]) as server:
        async with PluginClient(server.host, server.port) as client:
            listed_tools = await client.list_tools()
            assert len(listed_tools) == 1
            assert my_tool.tool == listed_tools[0]


async def test_that_a_plugin_reads_a_tool() -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: Optional[int]) -> int:
        """My tool's description"""
        return arg_1 * (arg_2 or 0)

    async with run_plugin_server([my_tool]) as server:
        async with PluginClient(server.host, server.port) as client:
            returned_tool = await client.read_tool(my_tool.tool.id)
            assert my_tool.tool == returned_tool


async def test_that_a_plugin_calls_a_tool() -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: int) -> int:
        return arg_1 * arg_2

    async with run_plugin_server([my_tool]) as server:
        async with PluginClient(server.host, server.port) as client:
            result = await client.call_tool(
                my_tool.tool.id,
                arguments={"arg_1": 2, "arg_2": 4},
            )
            assert result == 8


async def test_that_a_plugin_calls_an_async_tool() -> None:
    @tool
    async def my_tool(context: ToolContext, arg_1: int, arg_2: int) -> int:
        return arg_1 * arg_2

    async with run_plugin_server([my_tool]) as server:
        async with PluginClient(server.host, server.port) as client:
            result = await client.call_tool(
                my_tool.tool.id,
                arguments={"arg_1": 2, "arg_2": 4},
            )
            assert result == 8
