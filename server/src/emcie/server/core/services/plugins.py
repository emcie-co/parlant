from __future__ import annotations
import dateutil.parser
from types import TracebackType
from typing import Optional, Sequence
import httpx
from urllib.parse import urljoin

from emcie.common.tools import Tool, ToolId, ToolResult, ToolContext
from emcie.server.core.tools import ToolExecutionError, ToolService


class PluginClient(ToolService):
    def __init__(self, url: str) -> None:
        self.url = url

    async def __aenter__(self) -> PluginClient:
        self._http_client = await httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(120),
        ).__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        await self._http_client.__aexit__(exc_type, exc_value, traceback)
        return False

    async def list_tools(self) -> Sequence[Tool]:
        response = await self._http_client.get(self._get_url("/tools"))
        content = response.json()
        return [
            Tool(
                id=t["id"],
                creation_utc=dateutil.parser.parse(t["creation_utc"]),
                name=t["name"],
                description=t["description"],
                parameters=t["parameters"],
                required=t["required"],
                consequential=t["consequential"],
            )
            for t in content["tools"]
        ]

    async def read_tool(self, tool_id: ToolId) -> Tool:
        response = await self._http_client.get(self._get_url(f"/tools/{tool_id}"))
        content = response.json()
        tool = content["tool"]
        return Tool(
            id=tool["id"],
            creation_utc=dateutil.parser.parse(tool["creation_utc"]),
            name=tool["name"],
            description=tool["description"],
            parameters=tool["parameters"],
            required=tool["required"],
            consequential=tool["consequential"],
        )

    async def call_tool(
        self,
        tool_id: ToolId,
        context: ToolContext,
        arguments: dict[str, object],
    ) -> ToolResult:
        response = await self._http_client.post(
            self._get_url(f"/tools/{tool_id}/calls"),
            json={
                "session_id": context.session_id,
                "arguments": arguments,
            },
        )

        if response.is_error:
            raise ToolExecutionError(tool_id)

        content = response.json()

        return ToolResult(**content["result"])

    def _get_url(self, path: str) -> str:
        return urljoin(f"{self.url}", path)
