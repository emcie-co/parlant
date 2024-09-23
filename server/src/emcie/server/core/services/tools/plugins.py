from __future__ import annotations
import json
import dateutil.parser
from types import TracebackType
from typing import Mapping, Optional, Sequence
import httpx
from urllib.parse import urljoin

from emcie.common.tools import Tool, ToolId, ToolResult, ToolContext
from emcie.server.core.common import JSONSerializable
from emcie.server.core.contextual_correlator import ContextualCorrelator
from emcie.server.core.emissions import EventEmitterFactory
from emcie.server.core.sessions import SessionId
from emcie.server.core.tools import ToolExecutionError, ToolService


class PluginClient(ToolService):
    def __init__(
        self,
        url: str,
        event_emitter_factory: EventEmitterFactory,
        correlator: ContextualCorrelator,
    ) -> None:
        self.url = url
        self._event_emitter_factory = event_emitter_factory
        self._correlator = correlator

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
        arguments: Mapping[str, JSONSerializable],
    ) -> ToolResult:
        try:
            async with self._http_client.stream(
                method="post",
                url=self._get_url(f"/tools/{tool_id}/calls"),
                json={
                    "session_id": context.session_id,
                    "arguments": arguments,
                },
            ) as response:
                if response.is_error:
                    raise ToolExecutionError(tool_id)

                event_emitter = self._event_emitter_factory.create_event_emitter(
                    session_id=SessionId(context.session_id),
                )

                async for chunk in response.aiter_text():
                    chunk_dict = json.loads(chunk)

                    if "data" and "metadata" in chunk_dict:
                        return ToolResult(
                            data=chunk_dict["data"],
                            metadata=chunk_dict["metadata"],
                        )
                    elif "message" in chunk_dict:
                        await event_emitter.emit_message_event(
                            correlation_id=self._correlator.correlation_id,
                            data={"message": chunk_dict["message"]},
                        )
                    elif "error" in chunk_dict:
                        raise ToolExecutionError(tool_id, chunk_dict["error"])
                    else:
                        raise ToolExecutionError(tool_id, f"Unexpected chunk dict: {chunk_dict}")
        except Exception as exc:
            raise ToolExecutionError(tool_id) from exc

        raise ToolExecutionError(tool_id, "Unexpected response (no result chunk)")

    def _get_url(self, path: str) -> str:
        return urljoin(f"{self.url}", path)
