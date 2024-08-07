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
