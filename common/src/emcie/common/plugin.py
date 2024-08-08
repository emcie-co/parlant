from __future__ import annotations
import asyncio
from datetime import datetime, timezone
import inspect
from types import TracebackType
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    Sequence,
    TypedDict,
    Union,
    Unpack,
    get_args,
    overload,
)
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from emcie.common.tools import Tool, ToolId, ToolParameter, ToolParameterType
from emcie.common.types import JSONSerializable


class ToolContext:
    pass


ToolFunction = Union[
    Callable[[ToolContext], Any],
    Callable[[ToolContext, Any], Any],
    Callable[[ToolContext, Any, Any], Any],
    Callable[[ToolContext, Any, Any, Any], Any],
    Callable[[ToolContext, Any, Any, Any, Any], Any],
    Callable[[ToolContext, Any, Any, Any, Any, Any], Any],
    Callable[[ToolContext, Any, Any, Any, Any, Any, Any], Any],
    Callable[[ToolContext, Any, Any, Any, Any, Any, Any, Any], Any],
    Callable[[ToolContext, Any, Any, Any, Any, Any, Any, Any, Any], Any],
]


class ToolEntry(NamedTuple):
    tool: Tool
    function: ToolFunction


class _ToolDecoratorParams(TypedDict, total=False):
    id: str
    name: str
    consequential: bool


_ToolParameterType = Union[str, int, float, bool]


class _ResolvedToolParameterTyped(NamedTuple):
    t: type[_ToolParameterType]
    is_optional: bool


def _tool_decorator_impl(
    **kwargs: Unpack[_ToolDecoratorParams],
) -> Callable[[ToolFunction], ToolEntry]:
    def _resolve_param_type(param: inspect.Parameter) -> _ResolvedToolParameterTyped:
        if not param.annotation.__name__ == "Optional":
            return _ResolvedToolParameterTyped(t=param.annotation, is_optional=False)
        else:
            return _ResolvedToolParameterTyped(t=get_args(param.annotation)[0], is_optional=True)

    def _ensure_valid_tool_signature(func: ToolFunction) -> None:
        signature = inspect.signature(func)

        parameters = list(signature.parameters.values())

        assert (
            len(parameters) >= 1
        ), "A tool function must accept a parameter 'context: ToolContext'"

        assert (
            parameters[0].name == "context"
        ), "A tool function's first parameter must be 'context: ToolContext'"
        assert (
            parameters[0].annotation == ToolContext
        ), "A tool function's first parameter must be 'context: ToolContext'"

        assert (
            (signature.return_annotation in get_args(JSONSerializable))
            or signature.return_annotation == JSONSerializable
        ), "A tool function must return a JSON-serializable type"

        for param in parameters[1:]:
            param_type = _resolve_param_type(param)

            assert (
                param_type.t in get_args(_ToolParameterType)
            ), f"{param.name}: {param_type.t.__name__}: parameter type must be in {[t.__name__ for t in get_args(_ToolParameterType)]}"

    def _describe_parameters(func: ToolFunction) -> dict[str, ToolParameter]:
        type_to_param_type: dict[type[_ToolParameterType], ToolParameterType] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }

        parameters = list(inspect.signature(func).parameters.values())
        parameters = parameters[1:]  # Skip tool context parameter
        return {p.name: {"type": type_to_param_type[_resolve_param_type(p).t]} for p in parameters}

    def _find_required_params(func: ToolFunction) -> list[str]:
        parameters = list(inspect.signature(func).parameters.values())
        parameters = parameters[1:]  # Skip tool context parameter
        return [p.name for p in parameters if p.annotation.__name__ != "Optional"]

    def decorator(func: ToolFunction) -> ToolEntry:
        _ensure_valid_tool_signature(func)

        return ToolEntry(
            tool=Tool(
                id=ToolId(kwargs.get("id") or func.__qualname__),
                creation_utc=datetime.now(timezone.utc),
                name=kwargs.get("name") or func.__name__,
                description=func.__doc__ or "",
                parameters=_describe_parameters(func),
                required=_find_required_params(func),
                consequential=kwargs.get("consequential") or False,
            ),
            function=func,
        )

    return decorator


@overload
def tool(
    **kwargs: Unpack[_ToolDecoratorParams],
) -> Callable[[ToolFunction], ToolEntry]: ...


@overload
def tool(func: ToolFunction) -> ToolEntry: ...


def tool(
    func: ToolFunction | None = None,
    **kwargs: Unpack[_ToolDecoratorParams],
) -> ToolEntry | Callable[[ToolFunction], ToolEntry]:
    if func:
        return _tool_decorator_impl()(func)
    else:
        return _tool_decorator_impl(**kwargs)


class ListToolsResponse(BaseModel):
    tools: list[Tool]


class ReadToolResponse(BaseModel):
    tool: Tool


class CallToolRequest(BaseModel):
    arguments: dict[str, _ToolParameterType]


class CallToolResponse(BaseModel):
    result: object


class PluginServer:
    def __init__(
        self,
        name: str,
        tools: Sequence[ToolEntry],
        port: int = 8089,
        host: str = "0.0.0.0",
    ) -> None:
        self.name = name
        self.tools = {entry.tool.id: entry for entry in tools}
        self.host = host
        self.port = port

        self._server: Optional[uvicorn.Server] = None

    async def __aenter__(self) -> PluginServer:
        self._task = asyncio.create_task(self.serve())

        start_timeout = 5
        sample_frequency = 0.1

        for _ in range(int(start_timeout / sample_frequency)):
            await asyncio.sleep(sample_frequency)

            if self.started():
                return self

        raise TimeoutError()

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        try:
            await self._task
        except asyncio.CancelledError:
            pass

        return False

    async def serve(self) -> None:
        app = self._create_app()

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
        )

        self._server = uvicorn.Server(config)

        await self._server.serve()

    async def shutdown(self) -> None:
        self._task.cancel()

    def started(self) -> bool:
        if self._server:
            return self._server.started
        return False

    def _create_app(self) -> FastAPI:
        # TODO: Change uses of BaseModel to DefaultBaseModel

        app = FastAPI()

        @app.get("/tools")
        async def list_tools() -> ListToolsResponse:
            return ListToolsResponse(tools=[t.tool for t in self.tools.values()])

        @app.get("/tools/{tool_id}")
        async def read_tool(tool_id: ToolId) -> ReadToolResponse:
            return ReadToolResponse(tool=self.tools[tool_id].tool)

        @app.post("/tools/{tool_id}/calls")
        async def call_tool(
            tool_id: ToolId,
            request: CallToolRequest,
        ) -> CallToolResponse:
            stub_context = ToolContext()
            result = self.tools[tool_id].function(stub_context, **request.arguments)  # type: ignore
            return CallToolResponse(result=result)

        return app