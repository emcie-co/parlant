from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator

from emcie.common.tools import Tool, ToolId
from emcie.common.plugin import PluginServer
from emcie.server.core.plugins import PluginClient


@asynccontextmanager
async def run_plugin_server(tools: list[Tool]) -> AsyncIterator[PluginServer]:
    async with PluginServer(tools, host="127.0.0.1") as server:
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
    tool = Tool(
        id=ToolId("my_tool"),
        creation_utc=datetime.now(),
        name="My Tool",
        description="",
        parameters={},
        required=[],
        consequential=False,
    )

    async with run_plugin_server([tool]) as server:
        async with PluginClient(server.host, server.port) as client:
            tools = await client.list_tools()
            assert len(tools) == 1
            assert tools[0] == tool
