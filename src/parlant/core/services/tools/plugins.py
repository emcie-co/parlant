# Copyright 2024 Emcie Co Ltd.
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
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import enum
import inspect
import json
import traceback
import dateutil.parser
from types import TracebackType
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    TypedDict,
    Union,
    get_args,
    overload,
)
from typing_extensions import Unpack, override
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
import httpx
from urllib.parse import urljoin

import uvicorn

from parlant.core.agents import AgentId
from parlant.core.tools import (
    Tool,
    ToolError,
    ToolParameter,
    ToolParameterType,
    ToolResult,
    ToolContext,
    EnumValueType,
    ToolResultError,
    normalize_tool_arguments,
)
from parlant.core.common import DefaultBaseModel, ItemNotFoundError, JSONSerializable, UniqueId
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.emissions import EventEmitterFactory
from parlant.core.sessions import SessionId, SessionStatus
from parlant.core.tools import ToolExecutionError, ToolService

ToolFunction = Union[
    Callable[
        [ToolContext],
        Union[ToolResult, Awaitable[ToolResult]],
    ],
    Callable[
        [ToolContext, Any],
        Union[ToolResult, Awaitable[ToolResult]],
    ],
    Callable[
        [ToolContext, Any, Any],
        Union[Awaitable[ToolResult], ToolResult],
    ],
    Callable[
        [ToolContext, Any, Any, Any],
        Union[ToolResult, Awaitable[ToolResult]],
    ],
    Callable[
        [ToolContext, Any, Any, Any, Any],
        Union[ToolResult, Awaitable[ToolResult]],
    ],
    Callable[
        [ToolContext, Any, Any, Any, Any, Any],
        Union[ToolResult, Awaitable[ToolResult]],
    ],
    Callable[
        [ToolContext, Any, Any, Any, Any, Any, Any],
        Union[ToolResult, Awaitable[ToolResult]],
    ],
    Callable[
        [ToolContext, Any, Any, Any, Any, Any, Any, Any],
        Union[ToolResult, Awaitable[ToolResult]],
    ],
    Callable[
        [ToolContext, Any, Any, Any, Any, Any, Any, Any, Any],
        Union[ToolResult, Awaitable[ToolResult]],
    ],
]


@dataclass(frozen=True)
class ToolEntry:
    tool: Tool
    function: ToolFunction

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)


class _ToolDecoratorParams(TypedDict, total=False):
    id: str
    name: str
    consequential: bool


_ToolParameterType = Union[str, int, float, bool]


class _ResolvedToolParameterTyped(NamedTuple):
    t: type[_ToolParameterType]
    is_optional: bool


def _resolve_param_type(param: inspect.Parameter) -> _ResolvedToolParameterTyped:
    try:
        if args := get_args(param.annotation):
            if getattr(param.annotation, "__name__", None) != "Optional":
                if len(args) != 2:
                    raise Exception()
                if type(None) not in args:
                    raise Exception()
                if all(t is None for t in args):
                    raise Exception()

            return _ResolvedToolParameterTyped(
                t=get_args(param.annotation)[0],
                is_optional=True,
            )
        else:
            return _ResolvedToolParameterTyped(
                t=param.annotation,
                is_optional=False,
            )
    except Exception:
        raise TypeError(f"Parameter type '{param.annotation}' is not supported in tool functions")


def _tool_decorator_impl(
    **kwargs: Unpack[_ToolDecoratorParams],
) -> Callable[[ToolFunction], ToolEntry]:
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
            signature.return_annotation == ToolResult
        ), "A tool function must return a ToolResult object"

        for param in parameters[1:]:
            param_type = _resolve_param_type(param)

            if param_type.t not in get_args(_ToolParameterType) and not issubclass(
                param_type.t, enum.Enum
            ):
                raise AssertionError(
                    f"{param.name}: {param_type.t.__name__}: parameter type must be in {[t.__name__ for t in get_args(_ToolParameterType)]} or be a valid Enum type"
                )

            if issubclass(param_type.t, enum.Enum):
                assert all(
                    type(e.value) in get_args(EnumValueType) for e in param_type.t
                ), f"{param.name}: {param_type.t.__name__}: Enum values must be in {[t.__name__ for t in get_args(EnumValueType)]}"

    def _describe_parameters(func: ToolFunction) -> dict[str, ToolParameter]:
        type_to_param_type: dict[type[_ToolParameterType], ToolParameterType] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }

        parameters = list(inspect.signature(func).parameters.values())
        parameters = parameters[1:]  # Skip tool context parameter

        param_descriptions = {}

        for p in parameters:
            param_type_info = _resolve_param_type(p)
            param_type = param_type_info.t

            if param_type in type_to_param_type:
                tool_param = ToolParameter(type=type_to_param_type[param_type])
            elif issubclass(param_type, enum.Enum):
                tool_param = ToolParameter(type="string", enum=[e.value for e in param_type])
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")

            param_descriptions[p.name] = tool_param

        return param_descriptions

    def _find_required_params(func: ToolFunction) -> list[str]:
        parameters = list(inspect.signature(func).parameters.values())
        parameters = parameters[1:]  # Skip tool context parameter
        resolved_params = {p.name: _resolve_param_type(p) for p in parameters}
        return [name for name, type in resolved_params.items() if not type.is_optional]

    def decorator(func: ToolFunction) -> ToolEntry:
        _ensure_valid_tool_signature(func)

        entry = ToolEntry(
            tool=Tool(
                creation_utc=datetime.now(timezone.utc),
                name=kwargs.get("name") or func.__name__,
                description=func.__doc__ or "",
                parameters=_describe_parameters(func),
                required=_find_required_params(func),
                consequential=kwargs.get("consequential") or False,
            ),
            function=func,
        )

        return entry

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


class ListToolsResponse(DefaultBaseModel):
    tools: list[Tool]


class ReadToolResponse(DefaultBaseModel):
    tool: Tool


class CallToolRequest(DefaultBaseModel):
    agent_id: str
    session_id: str
    customer_id: str
    arguments: dict[str, _ToolParameterType]


class PluginServer:
    def __init__(
        self,
        tools: Sequence[ToolEntry],
        port: int = 8089,
        host: str = "0.0.0.0",
        on_app_created: Callable[[FastAPI], Awaitable[FastAPI]] | None = None,
        plugin_data: Mapping[str, Any] = {},
        hosted: bool = False,
    ) -> None:
        self.tools = {entry.tool.name: entry for entry in tools}
        self.plugin_data = plugin_data
        self.host = host
        self.port = port
        self.hosted = hosted
        self.url = f"http://{self.host}:{self.port}"

        self._on_app_created = on_app_created

        self._server: uvicorn.Server | None = None

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

        if self._on_app_created:
            app = await self._on_app_created(app)

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="critical",
        )

        self._server = uvicorn.Server(config)

        if self.hosted:
            # Run without capturing signals.
            # This is because we're being hosted in another process
            # that has its own bookkeeping on signals.
            await self._server._serve()
        else:
            await self._server.serve()

    async def shutdown(self) -> None:
        if server := self._server:
            server.should_exit = True

    def started(self) -> bool:
        if self._server:
            return self._server.started
        return False

    def _create_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/tools")
        async def list_tools() -> ListToolsResponse:
            return ListToolsResponse(tools=[t.tool for t in self.tools.values()])

        @app.get("/tools/{name}")
        async def read_tool(name: str) -> ReadToolResponse:
            try:
                spec = self.tools[name]
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Tool: '{name}' does not exists",
                )
            return ReadToolResponse(tool=spec.tool)

        @app.post("/tools/{name}/calls")
        async def call_tool(
            name: str,
            request: CallToolRequest,
        ) -> StreamingResponse:
            try:
                self.tools[name]
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Tool: '{name}' does not exists",
                )

            end = asyncio.Event()
            chunks_received = asyncio.Semaphore(value=0)
            lock = asyncio.Lock()
            chunks: list[str] = []

            async def chunk_generator(
                result_future: Awaitable[ToolResult],
            ) -> AsyncIterator[str]:
                while True:
                    end_future = asyncio.ensure_future(end.wait())
                    chunks_received_future = asyncio.ensure_future(chunks_received.acquire())

                    await asyncio.wait(
                        [end_future, chunks_received_future],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if chunks_received_future.done():
                        async with lock:
                            next_chunk = chunks.pop(0)
                        yield next_chunk
                        # proceed to next potential acquire/end,
                        # skipping the end-check, otherwise
                        # we may skip emitted chunks.
                        continue
                    else:
                        # Release the acquire we performed to skip it
                        chunks_received.release()
                        await chunks_received_future

                    if end_future.done():
                        try:
                            result = await result_future

                            final_result_chunk = json.dumps(
                                {
                                    "data": result.data,
                                    "metadata": result.metadata,
                                    "control": result.control,
                                }
                            )

                            yield final_result_chunk
                        except Exception as exc:
                            yield json.dumps({"error": str(exc)})

                        return
                    else:
                        end_future.cancel()
                        await asyncio.gather(end_future, return_exceptions=True)

            async def emit_message(message: str) -> None:
                async with lock:
                    chunks.append(json.dumps({"message": message}))
                chunks_received.release()

            async def emit_status(
                status: SessionStatus,
                data: JSONSerializable,
            ) -> None:
                async with lock:
                    chunks.append(json.dumps({"status": status, "data": data}))
                chunks_received.release()

            context = ToolContext(
                agent_id=request.agent_id,
                session_id=request.session_id,
                customer_id=request.customer_id,
                emit_message=emit_message,
                emit_status=emit_status,
                plugin_data=self.plugin_data,
            )

            func = self.tools[name].function

            try:
                result = self.tools[name].function(
                    context,
                    **normalize_tool_arguments(
                        inspect.signature(func).parameters,
                        request.arguments,
                    ),
                )  # type: ignore
            except BaseException:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=traceback.format_exc(),
                )

            result_future: asyncio.Future[ToolResult]

            if inspect.isawaitable(result):
                result_future = asyncio.ensure_future(result)
            else:
                result_future = asyncio.Future[ToolResult]()
                result_future.set_result(result)

            result_future.add_done_callback(lambda _: end.set())

            return StreamingResponse(
                content=chunk_generator(result_future),
                media_type="text/plain",
            )

        return app


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

    @override
    async def list_tools(self) -> Sequence[Tool]:
        response = await self._http_client.get(self._get_url("/tools"))
        content = response.json()
        return [
            Tool(
                name=t["name"],
                creation_utc=dateutil.parser.parse(t["creation_utc"]),
                description=t["description"],
                parameters=t["parameters"],
                required=t["required"],
                consequential=t["consequential"],
            )
            for t in content["tools"]
        ]

    @override
    async def read_tool(self, name: str) -> Tool:
        response = await self._http_client.get(self._get_url(f"/tools/{name}"))
        if response.status_code == status.HTTP_404_NOT_FOUND:
            raise ItemNotFoundError(UniqueId(name))
        content = response.json()
        tool = content["tool"]
        return Tool(
            name=tool["name"],
            creation_utc=dateutil.parser.parse(tool["creation_utc"]),
            description=tool["description"],
            parameters=tool["parameters"],
            required=tool["required"],
            consequential=tool["consequential"],
        )

    @override
    async def call_tool(
        self,
        name: str,
        context: ToolContext,
        arguments: Mapping[str, JSONSerializable],
    ) -> ToolResult:
        try:
            async with self._http_client.stream(
                method="post",
                url=self._get_url(f"/tools/{name}/calls"),
                json={
                    "agent_id": context.agent_id,
                    "session_id": context.session_id,
                    "customer_id": context.customer_id,
                    "arguments": arguments,
                },
            ) as response:
                if response.status_code == status.HTTP_404_NOT_FOUND:
                    raise ItemNotFoundError(UniqueId(name))

                if response.is_error:
                    raise ToolExecutionError(
                        tool_name=name,
                        message=f"url='{self.url}', arguments='{arguments}'",
                    )

                event_emitter = await self._event_emitter_factory.create_event_emitter(
                    emitting_agent_id=AgentId(context.agent_id),
                    session_id=SessionId(context.session_id),
                )

                async for chunk in response.aiter_text():
                    if len(chunk) > (16 * 1024):
                        raise ToolResultError(
                            tool_name=name,
                            message=f"url='{self.url}', arguments='{arguments}', Response exceeds 16KB limit",
                        )

                    chunk_dict = json.loads(chunk)

                    if "data" and "metadata" in chunk_dict:
                        return ToolResult(
                            data=chunk_dict["data"],
                            metadata=chunk_dict["metadata"],
                            control=chunk_dict["control"],
                        )
                    elif "status" in chunk_dict:
                        await event_emitter.emit_status_event(
                            correlation_id=self._correlator.correlation_id,
                            data={
                                "status": chunk_dict["status"],
                                "data": chunk_dict.get("data", {}),
                            },
                        )
                    elif "message" in chunk_dict:
                        await event_emitter.emit_message_event(
                            correlation_id=self._correlator.correlation_id,
                            data=str(chunk_dict["message"]),
                        )
                    elif "error" in chunk_dict:
                        raise ToolExecutionError(
                            tool_name=name,
                            message=f"url='{self.url}', arguments='{arguments}', error: {chunk_dict['error']}",
                        )
                    else:
                        raise ToolResultError(
                            tool_name=name,
                            message=f"url='{self.url}', arguments='{arguments}', Unexpected chunk dict: {chunk_dict}",
                        )
        except ToolError as exc:
            raise exc
        except Exception as exc:
            raise ToolExecutionError(tool_name=name) from exc

        raise ToolExecutionError(
            tool_name=name,
            message=f"url='{self.url}', Unexpected response (no result chunk)",
        )

    def _get_url(self, path: str) -> str:
        return urljoin(f"{self.url}", path)
