from __future__ import annotations
import dateutil.parser
from types import TracebackType
from typing import Optional, Sequence, Type
import httpx
from urllib.parse import urljoin

from emcie.common.tools import Tool, ToolId, ToolParameter
from emcie.server.core.common import JSONSerializable
from emcie.server.core.tools import ToolService


class PluginClient(ToolService):
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    async def __aenter__(self) -> PluginClient:
        self._http_client = await httpx.AsyncClient().__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
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
        raise NotImplementedError()

    async def call_tool(
        self,
        tool_id: ToolId,
        parameters: dict[str, ToolParameter],
    ) -> JSONSerializable:
        raise NotImplementedError()

    def _get_url(self, path: str) -> str:
        return urljoin(f"http://{self.host}:{self.port}", path)
