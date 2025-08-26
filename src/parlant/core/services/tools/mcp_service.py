# Copyright 2025 Emcie Co Ltd.
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
from mailbox import FormatError
from types import TracebackType
from typing import Any, Sequence, Mapping, Optional, Literal, Callable
from fastmcp.client.transports import StreamableHttpTransport
from typing_extensions import override
from fastmcp import FastMCP
from fastmcp.client import Client
from mcp.types import Tool as McpTool
import asyncio
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
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.emissions import EventEmitterFactory

DEFAULT_MCP_PORT: int = 8181

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

        self._server.settings.port = port

        if "://" in host:
            host = host.split("://")[1]
        self._server.settings.host = host
        self.transport = transport
        for tool in tools:
            self._server.add_tool(tool)

    async def __aenter__(self) -> MCPToolServer:
        self._task = asyncio.create_task(self._server.run_async(transport=self.transport))

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
        await self._server.run_async(transport=self.transport)

    async def shutdown(self) -> None:
        """At the time of creating this server, there is no graceful shutdown for the FactMCP http server"""
        if self.started() and hasattr(self._server, "server") and self._server.server:
            self._server.server.should_exit = True

    def started(self) -> bool:
        if hasattr(self._server, "_mcp_server") and self._server._mcp_server:
            return True
        return False

    def get_port(self) -> int:
        return self._server.settings.port


class MCPToolClient(ToolService):
    def __init__(
        self,
        url: str,
        event_emitter_factory: EventEmitterFactory,
        logger: Logger,
        correlator: ContextualCorrelator,
        port: int = DEFAULT_MCP_PORT,
    ) -> None:
        self._event_emitter_factory = event_emitter_factory
        self._logger = logger
        self._correlator = correlator
        if ":" in url[-6:]:
            parts = url.split(":")
            self.url = ":".join(parts[:-1])
            self.port = int(parts[-1])
        else:
            self.url = url
            self.port = port

    async def __aenter__(self) -> MCPToolClient:
        try:
            self._client = Client(StreamableHttpTransport(url=f"{self.url}:{self.port}/mcp"))
            await asyncio.wait_for(self._client.__aenter__(), timeout=10.0)  # type: ignore
            return self
        except asyncio.TimeoutError:
            raise ConnectionError(f"Connection to MCP service at {self.url}:{self.port} timed out")
        except Exception as e:
            raise Exception(f"Failed to connect to MCP service: {str(e)}")

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        if self._client:
            try:
                await self._client.__aexit__(exc_type, exc_value, traceback)  # type: ignore
            except RuntimeError:
                pass
        return False

    @override
    async def list_tools(self) -> Sequence[Tool]:
        try:
            if not self._client:
                raise ToolError("Client not initialized.")

            tools = await self._client.list_tools()
            return [mcp_tool_to_parlant_tool(t) for t in tools]
        except Exception as e:
            raise ToolError(str(e))

    @override
    async def read_tool(self, name: str) -> Tool:
        try:
            tools = await self._client.list_tools()
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
            result = await self._client.call_tool(name, dict(arguments))
            text = next((r.text for r in result if r.type == "text"), None)
            return ToolResult(data=text)
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
    for param in mcp_tool.inputSchema["properties"]:
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
        required=mcp_tool.inputSchema["required"],
        consequential=False,
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

    param_type = mcp_param.get("type", None)
    param_format = mcp_param.get("format", None)
    description = mcp_param.get("title", None)

    if (param_type, param_format) in mcp_parameter_type_map:
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
        """ Reference to another schema - currently only enum is supported"""
        def_ = resolve_ref(mcp_param["$ref"], schema)
        return parse_enum_def(def_)

    if param_type == "array":
        """ Currently only lists and sets are supported """
        if "items" not in mcp_param:
            raise FormatError("Only lists and sets are supported collections")

        enum_desc = None
        if "$ref" in mcp_param["items"]:
            """ Reference to another schema - currently only enum is supported"""
            def_ = resolve_ref(mcp_param["items"]["$ref"], schema)
            enum_desc = parse_enum_def(def_)

        return ToolParameterDescriptor(
            type="array",
            item_type=(
                enum_desc["type"]
                if enum_desc
                else mcp_parameter_type_map[(mcp_param["items"]["type"], None)]
            ),
            **({"enum": enum_desc["enum"]} if enum_desc is not None else {}),
            description=mcp_param.get("title", ""),
        )
    raise FormatError(f"Unsupported parameter type: {param_type} (parameter is {parameter_name})")


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
