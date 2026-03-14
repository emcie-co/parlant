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

from __future__ import annotations

from ast import literal_eval
from datetime import datetime, timezone
import json
from mailbox import FormatError
from mcp.types import Tool as McpTool
from types import TracebackType
from typing import Any, Sequence, Mapping, Optional, Literal, Callable, Awaitable, cast
from typing_extensions import override
import asyncio
import httpx

from fastmcp import FastMCP
from fastmcp.tools import Tool as FastMCPTool
from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport

from parlant.core.loggers import Logger
from parlant.core.tools import (
    Tool,
    ToolError,
    ToolOverlap,
    ToolParameterDescriptor,
    ToolParameterOptions,
    ToolResult,
    ToolContext,
    ToolService,
    ToolParameterType,
)
from parlant.core.common import JSONSerializable
from parlant.core.tracer import Tracer
from parlant.core.emissions import EventEmitterFactory

DEFAULT_MCP_PORT: int = 8181
DEFAULT_MCP_CONNECTION_ATTEMPTS: int = 3
DEFAULT_MCP_RETRY_DELAY_SECONDS: float = 0.5

StringBasedTypes = [
    "string",
    "enum",
    "date",
    "datetime",
    "timedelta",
    "path",
    "uuid",
]


class MCPToolServer:
    """This class is a wrapper around the FastMCP server, mainly to be used in testing the MCP client"""

    def __init__(
        self,
        tools: Sequence[Callable[..., Any]],
        port: int = DEFAULT_MCP_PORT,
        host: str = "0.0.0.0",
        server_data: Mapping[str, Any] = {},
        name: str = "",
        transport: Optional[Literal["stdio", "streamable-http", "sse"]] = "streamable-http",
    ) -> None:
        self._server: FastMCP[Any] = FastMCP(name=name)

        self._port = port

        if "://" in host:
            host = host.split("://")[1]
        self._host = host
        self.transport = transport
        for tool in tools:
            self._server.add_tool(FastMCPTool.from_function(tool))

    async def __aenter__(self) -> MCPToolServer:
        self._task = asyncio.create_task(
            self._server.run_async(
                transport=self.transport,
                host=self._host,
                port=self._port,
            )
        )

        start_timeout = 10
        sample_frequency = 0.1

        for _ in range(int(start_timeout / sample_frequency)):
            await asyncio.sleep(sample_frequency)

            if self.started():
                return self

        raise TimeoutError("MCP server failed to start within timeout period")

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        self._task.cancel()

        await asyncio.gather(self._task, return_exceptions=True)

        await asyncio.sleep(0.01)
        return False

    async def serve(self) -> None:
        await self._server.run_async(
            transport=self.transport,
            host=self._host,
            port=self._port,
        )

    async def shutdown(self) -> None:
        """At the time of creating this server, there is no graceful shutdown for the FactMCP http server"""
        if self.started() and hasattr(self._server, "server") and self._server.server:
            self._server.server.should_exit = True

    def started(self) -> bool:
        if hasattr(self._server, "_mcp_server") and self._server._mcp_server:
            return True
        return False

    def get_port(self) -> int:
        return self._port


class MCPToolClient(ToolService):
    def __init__(
        self,
        url: str,
        event_emitter_factory: EventEmitterFactory,
        logger: Logger,
        tracer: Tracer,
        port: int = DEFAULT_MCP_PORT,
    ) -> None:
        self._event_emitter_factory = event_emitter_factory
        self._logger = logger
        self._tracer = tracer
        self._client: Client[StreamableHttpTransport] | None = None
        self._client_lock = asyncio.Lock()
        if ":" in url[-6:]:
            parts = url.split(":")
            self.url = ":".join(parts[:-1])
            self.port = int(parts[-1])
        else:
            self.url = url
            self.port = port
        self.endpoint_url = f"{self.url}:{self.port}"

    async def __aenter__(self) -> MCPToolClient:
        await self._connect(force=True)
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        await self._disconnect(exc_type, exc_value, traceback)
        return False

    def _create_client(self) -> Client[StreamableHttpTransport]:
        return Client(StreamableHttpTransport(url=f"{self.url}:{self.port}/mcp"))

    def _is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected()

    def _is_reconnectable_exception(self, exc: Exception) -> bool:
        if isinstance(
            exc,
            (
                asyncio.TimeoutError,
                ConnectionError,
                httpx.HTTPError,
                httpx.TimeoutException,
                httpx.TransportError,
            ),
        ):
            return True

        if isinstance(exc, RuntimeError):
            message = str(exc).lower()
            reconnectable_markers = (
                "client is not connected",
                "session was closed unexpectedly",
                "closed unexpectedly",
            )
            return any(marker in message for marker in reconnectable_markers)

        return False

    async def _close_client(
        self,
        client: Client[StreamableHttpTransport],
        exc_type: Optional[type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        traceback: Optional[TracebackType] = None,
    ) -> None:
        try:
            await client.__aexit__(exc_type, exc_value, traceback)  # type: ignore[no-untyped-call]
        except RuntimeError:
            pass
        except Exception as exc:
            self._logger.warning(f"Failed to close MCP client cleanly: {exc}")

    async def _disconnect(
        self,
        exc_type: Optional[type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        traceback: Optional[TracebackType] = None,
    ) -> None:
        async with self._client_lock:
            if self._client:
                client = self._client
                self._client = None
                await self._close_client(client, exc_type, exc_value, traceback)

    async def _connect(self, force: bool = False) -> Client[StreamableHttpTransport]:
        async with self._client_lock:
            if not force and self._is_connected():
                assert self._client is not None
                return self._client

            if self._client:
                stale_client = self._client
                self._client = None
                await self._close_client(stale_client)

            last_error: Exception | None = None

            for attempt in range(1, DEFAULT_MCP_CONNECTION_ATTEMPTS + 1):
                client = self._create_client()

                try:
                    await asyncio.wait_for(client.__aenter__(), timeout=10.0)  # type: ignore[no-untyped-call]
                    self._client = client
                    return client
                except asyncio.TimeoutError:
                    last_error = ConnectionError(
                        f"Connection to MCP service at {self.url}:{self.port} timed out"
                    )
                except Exception as exc:
                    last_error = Exception(f"Failed to connect to MCP service: {str(exc)}")
                finally:
                    if self._client is None:
                        await self._close_client(client)

                if attempt < DEFAULT_MCP_CONNECTION_ATTEMPTS:
                    self._logger.warning(
                        f"MCP connection attempt {attempt} failed for {self.url}:{self.port}; retrying"
                    )
                    await asyncio.sleep(DEFAULT_MCP_RETRY_DELAY_SECONDS * attempt)

            assert last_error is not None
            raise last_error

    async def _with_reconnect(
        self,
        operation: Callable[[Client[StreamableHttpTransport]], Awaitable[Any]],
        *,
        retry_once: bool = True,
    ) -> Any:
        client = await self._connect()

        try:
            return await operation(client)
        except Exception as exc:
            if not retry_once or not self._is_reconnectable_exception(exc):
                raise

            self._logger.warning(
                f"MCP client session dropped for {self.url}:{self.port}; reconnecting and retrying"
            )
            client = await self._connect(force=True)
            return await operation(client)

    @override
    async def list_tools(self) -> Sequence[Tool]:
        try:
            tools = await self._with_reconnect(lambda client: client.list_tools())
            return [mcp_tool_to_parlant_tool(t) for t in tools]
        except Exception as e:
            raise ToolError(str(e))

    @override
    async def read_tool(self, name: str) -> Tool:
        try:
            tools = await self._with_reconnect(lambda client: client.list_tools())
            tool = next(t for t in tools if t.name == name)
            return mcp_tool_to_parlant_tool(tool)
        except Exception as e:
            raise ToolError(str(e))

    @override
    async def resolve_tool(
        self,
        name: str,
        context: ToolContext,
    ) -> Tool:
        return await self.read_tool(name)

    @override
    async def call_tool(
        self,
        name: str,
        context: ToolContext,
        arguments: Mapping[str, JSONSerializable],
    ) -> ToolResult:
        try:
            tool = await self.read_tool(name)
            arguments = prepare_tool_arguments(arguments, tool.parameters)
            client = await self._connect()
            result = await client.call_tool(name, dict(arguments))
            return ToolResult(data=mcp_result_to_tool_result_data(result))
        except Exception as e:
            raise ToolError(str(e))


# Partial mapping of mcp types to parlant types using fields "type" and "format"
mcp_parameter_type_map: dict[tuple[str, str | None], ToolParameterType] = {
    ("number", None): "number",
    ("integer", None): "integer",
    ("boolean", None): "boolean",
    ("string", None): "string",
    ("string", "date"): "date",
    ("string", "date-time"): "datetime",
    ("string", "duration"): "timedelta",
    ("string", "path"): "path",
    ("string", "uuid"): "uuid",
}


def mcp_tool_to_parlant_tool(mcp_tool: McpTool) -> Tool:
    parameters = {}
    for param in mcp_tool.inputSchema.get("properties", {}):
        parameters[param] = (
            mcp_parameter_to_parlant_parameter(param, mcp_tool.inputSchema),
            ToolParameterOptions(),
        )
    tool = Tool(
        name=mcp_tool.name,
        creation_utc=datetime.now(timezone.utc),
        description=(mcp_tool.description if mcp_tool.description else ""),
        metadata={},
        parameters=parameters,
        required=mcp_tool.inputSchema.get("required", []),
        consequential=True,
        overlap=ToolOverlap.ALWAYS,
    )
    return tool


def mcp_parameter_to_parlant_parameter(
    parameter_name: str, schema: dict[str, Any]
) -> ToolParameterDescriptor:
    mcp_param = schema["properties"][parameter_name]
    if "anyOf" in mcp_param:
        """ Union of types - currently only optional is supported"""
        mcp_param = resolve_optional(mcp_param["anyOf"])

    param_type: str | None = mcp_param.get("type", None)
    param_format: str | None = mcp_param.get("format", None)
    description = mcp_param.get("title") or mcp_param.get("description")

    if param_type is not None and (param_type, param_format) in mcp_parameter_type_map:
        """ basic types + easily serializable types """
        return ToolParameterDescriptor(
            type=mcp_parameter_type_map[(param_type, param_format)], description=description
        )

    if "enum" in mcp_param and param_type == "string":
        """ Literal (only string enums are supported) """
        return ToolParameterDescriptor(
            type="string", description=description, enum=mcp_param["enum"]
        )

    if "$ref" in mcp_param:
        """ Reference to another schema - enum and object references are supported"""
        def_ = resolve_ref(mcp_param["$ref"], schema)
        if _is_object_schema(def_):
            return ToolParameterDescriptor(type="string", description=description or "")
        return parse_enum_def(def_)

    if param_type == "array":
        """ Currently only lists and sets are supported """
        if "items" not in mcp_param:
            raise FormatError("Only lists and sets are supported collections")

        item_type, enum = parse_mcp_array_item(mcp_param["items"], schema)

        return ToolParameterDescriptor(
            type="array",
            item_type=item_type,
            **({"enum": enum} if enum is not None else {}),
            description=mcp_param.get("title", ""),
        )
    if _is_object_schema(mcp_param):
        return ToolParameterDescriptor(type="string", description=description or "")
    raise FormatError(f"Unsupported parameter type: {param_type} (parameter is {parameter_name})")


def parse_mcp_array_item(
    item_schema: dict[str, Any],
    root_schema: dict[str, Any],
) -> tuple[ToolParameterType, Sequence[str] | None]:
    if "$ref" in item_schema:
        def_ = resolve_ref(item_schema["$ref"], root_schema)
        if _is_object_schema(def_):
            return ("string", None)

        enum_desc = parse_enum_def(def_)
        return (enum_desc["type"], enum_desc["enum"])

    item_type = cast(str, item_schema.get("type"))
    item_format = cast(str, item_schema.get("format"))

    if _is_object_schema(item_schema):
        return ("string", None)

    key = (item_type, item_format)
    if key in mcp_parameter_type_map:
        return (mcp_parameter_type_map[cast(tuple[str, str | None], key)], None)

    raise FormatError(f"Unsupported array item type: {item_type}")


def _is_object_schema(schema_part: Mapping[str, Any]) -> bool:
    return schema_part.get("type") == "object" or "properties" in schema_part


def mcp_result_to_tool_result_data(result: Any) -> Any:
    raw_data = getattr(result, "data", None)
    if raw_data is not None:
        return _deserialize_mcp_data(raw_data)

    structured_content = getattr(result, "structuredContent", None)
    if structured_content is None:
        structured_content = getattr(result, "structured_content", None)
    if structured_content is not None:
        return structured_content

    text_blocks = [
        content.text for content in getattr(result, "content", []) if content.type == "text"
    ]

    if not text_blocks:
        return None

    parsed_blocks = [_deserialize_mcp_text(text) for text in text_blocks]

    if len(parsed_blocks) == 1:
        return parsed_blocks[0]

    return parsed_blocks


def _deserialize_mcp_text(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _deserialize_mcp_data(data: Any) -> Any:
    if isinstance(data, str):
        return _deserialize_mcp_text(data)
    return data


def resolve_ref(ref_: str, schema: dict[str, Any]) -> dict[str, Any]:
    if not ref_.startswith("#/"):
        raise FormatError(f"Invalid reference format: {ref_}")
    ref_ = ref_[2:]
    for part in ref_.split("/"):
        if part not in schema:
            raise FormatError(f"Reference #{ref_} not found in schema")
        schema = schema[part]
    return schema


def resolve_optional(schema: list[dict[str, Any]]) -> dict[str, bool]:
    if (
        len(schema) != 2
        or not (any(k.get("type") == "null" for k in schema))
        or all(k.get("type") == "null" for k in schema)
    ):
        raise FormatError("Union types are not supported, unless optional")
    return next(k for k in schema if k["type"] != "null")


def parse_enum_def(def_: dict[str, Any]) -> ToolParameterDescriptor:
    if "properties" in def_ or "enum" not in def_:
        raise FormatError("Only enum references are supported")
    if def_.get("type", None) != "string":
        raise FormatError("Only string enums are supported")
    description = def_.get("description", "")
    return ToolParameterDescriptor(
        type="string",
        description=description,
        enum=def_["enum"],
    )


def split_arg_list(argument: str | list[Any], item_type: str) -> list[str]:
    if isinstance(argument, list):
        return argument
    if item_type in StringBasedTypes:
        # literal_eval is used for protection against nesting of single/double quotes of str (and our enums are always strings)
        return list(literal_eval(argument))
    else:
        # Split list is used for most types so we won't have to rely on the LLM to provide pythonic syntax
        list_str = argument.strip()
        if list_str.startswith("[") and list_str.endswith("]"):
            return list_str[1:-1].split(", ")
        raise ValueError(f"Invalid list format for argument '{argument}'")


def prepare_tool_arguments(
    arguments: Mapping[str, JSONSerializable],
    parameters: dict[str, tuple[ToolParameterDescriptor, ToolParameterOptions]],
) -> Mapping[str, JSONSerializable]:
    fixed_args = dict(arguments)
    for arg in arguments:
        if arg not in parameters:
            raise ToolError(f"Argument '{arg}' not found in tool parameters")

        descriptor = parameters[arg][0]

        if descriptor["type"] == "array":
            arg_value = arguments[arg]
            if isinstance(arg_value, (str, list)):
                fixed_args[arg] = split_arg_list(arg_value, descriptor["item_type"])
            else:
                raise ToolError(
                    f"Argument '{arg}' must be a string or list for array type, got {type(arg_value).__name__}"
                )

    return fixed_args
