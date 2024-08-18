from __future__ import annotations
from datetime import datetime, timezone
from openapi_parser import parse as parse_openapi_json
from openapi_parser.parser import (
    ContentType,
    DataType,
    Object,
    Operation,
    OperationMethod,
)
from types import TracebackType
from typing import Any, Awaitable, Callable, NamedTuple, Optional, Sequence, cast
import httpx

from emcie.common.tools import Tool, ToolId, ToolResult, ToolParameter, ToolParameterType
from emcie.server.core.common import JSONSerializable
from emcie.server.core.tools import ToolService


class _ToolSpec(NamedTuple):
    tool: Tool
    func: Callable[..., Awaitable[JSONSerializable]]


class OpenAPIClient(ToolService):
    def __init__(self, server_url: str, openapi_json: str) -> None:
        self.server_url = server_url
        self._tools = self._parse_tools(openapi_json)

    async def __aenter__(self) -> OpenAPIClient:
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

    def _parse_tools(self, openapi_json: str) -> dict[ToolId, _ToolSpec]:
        class ParseParametersResult(NamedTuple):
            parameters: dict[str, ToolParameter]
            required: list[str]

        def parse_parameters(operation: Operation) -> ParseParametersResult:
            result = ParseParametersResult(parameters={}, required=[])

            for parameter in operation.parameters:
                result.parameters[parameter.name] = {
                    "type": cast(ToolParameterType, parameter.schema.type.value),
                }

                if description := parameter.schema.description:
                    result.parameters[parameter.name]["description"] = description

                if enum := parameter.schema.enum:
                    result.parameters[parameter.name]["enum"] = enum

                if parameter.required:
                    result.required.append(parameter.name)

            if operation.request_body:
                assert (
                    len(operation.request_body.content) == 1
                ), "Only application/json is currently supported in OpenAPI services"

                assert (
                    operation.request_body.content[0].type == ContentType.JSON
                ), "Only application/json is currently supported in OpenAPI services"

                content = operation.request_body.content[0]

                if content.schema.type == DataType.OBJECT:
                    content_object = cast(Object, content.schema)

                    for property in content_object.properties:
                        result.parameters[property.name] = {
                            "type": cast(ToolParameterType, property.schema.type.value),
                        }

                        if description := property.schema.description:
                            result.parameters[property.name]["description"] = description

                        if enum := property.schema.enum:
                            result.parameters[property.name]["enum"] = enum

                        result.required.extend(content_object.required)
                else:
                    assert content.schema.title

                    parameter_name = content.schema.title.lower().replace(" ", "_")

                    result.parameters[parameter_name] = {
                        "type": cast(ToolParameterType, content.schema.type.value),
                    }

                    if description := content.schema.description:
                        result.parameters[parameter_name]["description"] = description

                    if enum := content.schema.enum:
                        result.parameters[parameter_name]["enum"] = enum

                    if operation.request_body.required:
                        result.required.append(parameter_name)

            return result

        tools = {}

        specification = parse_openapi_json(spec_string=openapi_json)

        for path in specification.paths:
            for operation in path.operations:
                assert operation.operation_id

                parse_result = parse_parameters(operation)

                tool = Tool(
                    id=ToolId(operation.operation_id),
                    creation_utc=datetime.now(timezone.utc),
                    name=operation.operation_id,
                    description=operation.description or "",
                    parameters=parse_result.parameters,
                    required=parse_result.required,
                    consequential=False,
                )

                async def tool_func(**kwargs: Any) -> JSONSerializable:
                    request_func = {
                        OperationMethod.HEAD: self._http_client.head,
                        OperationMethod.GET: self._http_client.get,
                        OperationMethod.POST: self._http_client.post,
                        OperationMethod.PUT: self._http_client.put,
                        OperationMethod.PATCH: self._http_client.patch,
                        OperationMethod.DELETE: self._http_client.delete,
                    }.get(operation.method)

                    return ""

                tools[tool.id] = _ToolSpec(tool=tool, func=tool_func)

        return tools

    async def list_tools(self) -> Sequence[Tool]:
        return [t.tool for t in self._tools.values()]

    async def read_tool(self, tool_id: ToolId) -> Tool:
        raise NotImplementedError()

    async def call_tool(
        self,
        tool_id: ToolId,
        arguments: dict[str, object],
    ) -> ToolResult:
        raise NotImplementedError()
