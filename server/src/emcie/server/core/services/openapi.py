from __future__ import annotations
from datetime import datetime, timezone
from functools import partial
import aiopenapi3  # type: ignore
import httpx
from openapi_parser import parse as parse_openapi_json
from openapi_parser.parser import (
    ContentType,
    DataType,
    Object,
    Operation,
)
from types import TracebackType
from typing import Any, Awaitable, Callable, NamedTuple, Optional, Sequence, cast

from emcie.common.tools import Tool, ToolId, ToolResult, ToolParameter, ToolParameterType
from emcie.server.core.tools import ToolService


class _ToolSpec(NamedTuple):
    tool: Tool
    func: Callable[..., Awaitable[ToolResult]]


class OpenAPIClient(ToolService):
    def __init__(self, server_url: str, openapi_json: str) -> None:
        self.server_url = server_url
        self.openapi_json = openapi_json
        self._tools = self._parse_tools(openapi_json)

    async def __aenter__(self) -> OpenAPIClient:
        class CustomClient(httpx.AsyncClient):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(
                    *args,
                    **{
                        **kwargs,
                        "timeout": httpx.Timeout(120),
                    },
                )

        self._openapi_client = aiopenapi3.OpenAPI.loads(
            url=self.server_url,
            data=self.openapi_json,
            session_factory=CustomClient,
        )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        return False

    def _parse_tools(self, openapi_json: str) -> dict[ToolId, _ToolSpec]:
        class ParameterSpecification(NamedTuple):
            query_parameters: dict[str, ToolParameter]
            body_parameters: dict[str, ToolParameter]
            required: list[str]

        def parse_parameters(operation: Operation) -> ParameterSpecification:
            result = ParameterSpecification(query_parameters={}, body_parameters={}, required=[])

            for parameter in operation.parameters:
                result.query_parameters[parameter.name] = {
                    "type": cast(ToolParameterType, parameter.schema.type.value),
                }

                if description := parameter.schema.description:
                    result.query_parameters[parameter.name]["description"] = description

                if enum := parameter.schema.enum:
                    result.query_parameters[parameter.name]["enum"] = enum

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

                assert (
                    content.schema.type == DataType.OBJECT
                ), "Only 'object' is supported as a schema type for request bodies in OpenAPI services"

                content_object = cast(Object, content.schema)

                for property in content_object.properties:
                    result.body_parameters[property.name] = {
                        "type": cast(ToolParameterType, property.schema.type.value),
                    }

                    if description := property.schema.description:
                        result.body_parameters[property.name]["description"] = description

                    if enum := property.schema.enum:
                        result.body_parameters[property.name]["enum"] = enum

                    result.required.extend(content_object.required)

            return result

        tools = {}

        specification = parse_openapi_json(spec_string=openapi_json)

        for path in specification.paths:
            for operation in path.operations:
                assert operation.operation_id

                parameter_spec = parse_parameters(operation)

                tool = Tool(
                    id=ToolId(operation.operation_id),
                    creation_utc=datetime.now(timezone.utc),
                    name=operation.operation_id,
                    description=operation.description or "",
                    parameters={
                        **parameter_spec.query_parameters,
                        **parameter_spec.body_parameters,
                    },
                    required=parameter_spec.required,
                    consequential=False,
                )

                async def tool_func(
                    url: str,
                    method: str,
                    parameter_spec: ParameterSpecification,
                    **parameters: Any,
                ) -> ToolResult:
                    request = self._openapi_client.createRequest((url, method))

                    query_parameters = {
                        k: v for k, v in parameters.items() if k in parameter_spec.query_parameters
                    }

                    body_parameters = {
                        k: v for k, v in parameters.items() if k in parameter_spec.body_parameters
                    }

                    response = await request(
                        parameters=query_parameters,
                        data=body_parameters,
                    )

                    data = response.model_dump()

                    return ToolResult(data=data)

                tools[tool.id] = _ToolSpec(
                    tool=tool,
                    func=partial(
                        tool_func,
                        path.url,
                        operation.method.value,
                        parameter_spec,
                    ),
                )

        return tools

    async def list_tools(self) -> Sequence[Tool]:
        return [t.tool for t in self._tools.values()]

    async def read_tool(self, tool_id: ToolId) -> Tool:
        return self._tools[tool_id].tool

    async def call_tool(
        self,
        tool_id: ToolId,
        arguments: dict[str, object],
    ) -> ToolResult:
        return await self._tools[tool_id].func(**arguments)
