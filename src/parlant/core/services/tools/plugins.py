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
import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timezone
import enum
import inspect
import json
import traceback
import dateutil.parser
from types import TracebackType
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    TypeAlias,
    TypedDict,
    Union,
    get_args,
    get_origin,
    overload,
)
from pydantic import BaseModel
from typing_extensions import Unpack, override
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import StreamingResponse
import httpx
from urllib.parse import urljoin

import uvicorn

from parlant.core.agents import AgentId
from parlant.core.loggers import Logger
from parlant.core.tools import (
    Tool,
    ToolError,
    ToolParameterDescriptor,
    ToolParameterOptions,
    ToolParameterType,
    ToolResult,
    ToolContext,
    ToolResultError,
    normalize_tool_arguments,
    validate_tool_arguments,
    ToolOverlap,
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
    name: str
    """Defines a custom name for the tool."""

    consequential: bool
    """Defines whether the tool is consequential or not."""

    metadata: Mapping[str, JSONSerializable]
    """Defines metadata for the tool, which can be used to provide additional information about the tool."""

    overlap: ToolOverlap
    """Defines how the tool overlaps with other tools. Defaults to ToolOverlap.AUTO."""


_ToolParameterType = Union[str, int, float, bool, date, datetime, list[Any], None]


class _ToolParameterInfo(NamedTuple):
    raw_type: type
    resolved_type: type[_ToolParameterType]
    options: Optional[ToolParameterOptions]
    is_optional: bool


def _resolve_param_info(param: inspect.Parameter) -> _ToolParameterInfo:
    try:
        parameter_type: type
        if get_origin(param.annotation) is list:
            # This way we handle typing.List[elem] as list[elem]
            elem_type = get_args(param.annotation)[0]
            parameter_type = list[elem_type]  # type: ignore[valid-type]
        else:
            parameter_type = param.annotation
        parameter_options: Optional[ToolParameterOptions] = None

        # If parameter has default then we'll consider it as optional (in terms of tool calling)
        if param.default is not inspect.Parameter.empty:
            has_default = True
        else:
            has_default = False

        # First thing, is our parameter annotated?
        if getattr(parameter_type, "__name__", None) == "Annotated":
            annotation_params = get_args(parameter_type)
            parameter_type = annotation_params[0]
            annotation_value = annotation_params[1]

            # Do we have a ToolParameterOptions to use here?
            # If so, let's unpack our parameter options from that.
            if isinstance(annotation_value, ToolParameterOptions):
                parameter_options = annotation_value

        # At this point—if needed—we've normalized an annotated
        # parameter to a non-annotated parameter.

        if args := get_args(parameter_type):
            # Okay, we're talking about a generic type.

            generic_type = getattr(parameter_type, "__name__", None)
            is_optional = False
            unpacked_type = None

            if generic_type == "Optional":
                is_optional = True
                unpacked_type = args[0]
            elif generic_type is None:
                # Assuming we encountered union syntax; i.e., `str | None`
                if len(args) != 2:
                    raise Exception()
                if type(None) not in args:
                    raise Exception()
                if all(t is None for t in args):
                    raise Exception()

                is_optional = True
                unpacked_type = next(t for t in args if t is not None)

            if not is_optional:
                # At this point, at least as far as our supported options,
                # we're expecting to see here a list[T] such that the type
                # is list and parameter type is T.
                if generic_type != "list":
                    raise Exception("Only `list` is supported as a generic container in parameters")

                return _ToolParameterInfo(
                    raw_type=parameter_type,
                    resolved_type=parameter_type,
                    options=parameter_options,
                    is_optional=has_default,
                )
            else:
                assert unpacked_type
                return _ToolParameterInfo(
                    raw_type=parameter_type,
                    resolved_type=unpacked_type,
                    options=parameter_options,
                    is_optional=True,
                )
        else:
            return _ToolParameterInfo(
                raw_type=parameter_type,
                resolved_type=parameter_type,
                options=parameter_options,
                is_optional=has_default,
            )
    except Exception:
        raise TypeError(f"Parameter type '{param.annotation}' is not supported in tool functions")


async def adapt_tool_arguments(
    parameters: Mapping[str, inspect.Parameter],
    arguments: Mapping[str, Any],
) -> Mapping[str, Any]:
    adapted_arguments = {}

    for name, argument in arguments.items():
        parameter_info = _resolve_param_info(parameters[name])

        if parameter_info.options and parameter_info.options.adapter:
            adapted_arguments[name] = await parameter_info.options.adapter(argument)
        else:
            adapted_arguments[name] = argument

    return adapted_arguments


async def _recompute_and_marshal_tool(
    tool: Tool, plugin_data: Mapping[str, Any], context: ToolContext
) -> Tool:
    """This function is specifically used to refresh some of the tool's
    details based on dynamic changes (e.g., updating parameter descriptors
    based on dynamically-generated enum choices)"""
    new_parameters = {}

    for name, (old_descriptor, options) in tool.parameters.items():
        new_descriptor = old_descriptor

        if options.choice_provider:
            args = {}
            for param_name in inspect.signature(options.choice_provider).parameters:
                # Tool context is identified by its type, all other parameters are taken by name from the plugin data
                if (
                    inspect.signature(options.choice_provider).parameters[param_name].annotation
                    is ToolContext
                ):
                    args[param_name] = context
                elif param_name in plugin_data:
                    args[param_name] = plugin_data[param_name]

            new_descriptor["enum"] = await options.choice_provider(**args)

        marshalled_options = ToolParameterOptions(
            hidden=options.hidden,
            source=options.source,
            description=options.description,
            significance=options.significance,
            examples=options.examples,
            display_name=options.display_name,
            precedence=options.precedence,
            adapter=None,
            choice_provider=None,
        )

        new_parameters[name] = (new_descriptor, marshalled_options)

    return Tool(
        name=tool.name,
        creation_utc=datetime.now(timezone.utc),
        description=tool.description,
        metadata=tool.metadata,
        parameters=new_parameters,
        required=tool.required,
        consequential=tool.consequential,
        overlap=tool.overlap,
    )


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
            param_info = _resolve_param_info(param)

            if issubclass(param_info.resolved_type, enum.Enum):
                assert all(
                    type(e.value) is str for e in param_info.resolved_type
                ), f"{param.name}: {param_info.resolved_type.__name__}: Enum values must be strings"

    def _describe_parameters(
        func: ToolFunction,
    ) -> dict[str, tuple[ToolParameterDescriptor, ToolParameterOptions]]:
        type_to_param_type: dict[type[_ToolParameterType], ToolParameterType] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            date: "date",
            datetime: "datetime",
        }

        parameters = list(inspect.signature(func).parameters.values())
        parameters = parameters[1:]  # Skip tool context parameter

        param_descriptors = {}

        for p in parameters:
            param_info = _resolve_param_info(p)
            param_type = param_info.resolved_type

            param_descriptor: ToolParameterDescriptor = {}

            if param_type in type_to_param_type:
                param_descriptor["type"] = type_to_param_type[param_type]
            elif issubclass(param_type, enum.Enum):
                param_descriptor["type"] = "string"
                param_descriptor["enum"] = [e.value for e in param_type]
            else:
                # Do a best-effort with the string type
                param_descriptor["type"] = "string"
                type_args = get_args(param_info.resolved_type)

                if len(type_args) > 0:
                    if param_info.resolved_type.__name__ != "list":
                        raise Exception(
                            "Only `list` is supported as a generic container in parameters"
                        )

                    list_item_type = type_args[0]

                    if list_item_type in type_to_param_type:
                        param_descriptor["type"] = "array"
                        param_descriptor["item_type"] = type_to_param_type[list_item_type]
                    elif issubclass(list_item_type, enum.Enum):
                        param_descriptor["type"] = "array"
                        param_descriptor["item_type"] = "string"
                        param_descriptor["enum"] = [e.value for e in list_item_type]
                elif issubclass(param_info.resolved_type, BaseModel):
                    param_descriptor["description"] = json.dumps(
                        {"json_schema": param_info.resolved_type.model_json_schema()}
                    )

            if options := param_info.options:
                if options.description:
                    param_descriptor["description"] = options.description
                if options.examples:
                    param_descriptor["examples"] = options.examples

            param_descriptors[p.name] = (
                param_descriptor,
                param_info.options or ToolParameterOptions(),
            )

        return param_descriptors

    def _find_required_params(func: ToolFunction) -> list[str]:
        parameters = list(inspect.signature(func).parameters.values())
        parameters = parameters[1:]  # Skip tool context parameter
        resolved_params = {p.name: _resolve_param_info(p) for p in parameters}
        return [name for name, type in resolved_params.items() if not type.is_optional]

    def decorator(func: ToolFunction) -> ToolEntry:
        _ensure_valid_tool_signature(func)

        entry = ToolEntry(
            tool=Tool(
                creation_utc=datetime.now(timezone.utc),
                name=kwargs.get("name", func.__name__),
                description=func.__doc__ or "",
                metadata=kwargs.get("metadata", {}),
                parameters=_describe_parameters(func),
                required=_find_required_params(func),
                consequential=kwargs.get("consequential", False),
                overlap=kwargs.get("overlap", ToolOverlap.AUTO),
            ),
            function=func,
        )

        return entry

    return decorator


@overload
def tool(
    **kwargs: Unpack[_ToolDecoratorParams],
) -> Callable[[ToolFunction], ToolEntry]:
    """Decorator for defining a tool function with metadata and options."""
    ...


@overload
def tool(func: ToolFunction) -> ToolEntry:
    """Decorator for defining a tool function with metadata and options."""
    ...


def tool(
    func: ToolFunction | None = None,
    **kwargs: Unpack[_ToolDecoratorParams],
) -> ToolEntry | Callable[[ToolFunction], ToolEntry]:
    """Decorator for defining a tool function with metadata and options."""

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


class _ToolResultShim(DefaultBaseModel):
    result: ToolResult


class ResolveToolRequest(DefaultBaseModel):
    agent_id: str
    session_id: str
    customer_id: str


ToolContextQuery: TypeAlias = Annotated[
    ResolveToolRequest,
    Query(
        description="The ids of a tool context",
        examples=[
            {"agent_id": "agent_id", "session_id": "session_id", "customer_id": "customer_id"}
        ],
    ),
]


class PluginServer:
    """A server that hosts tools, interfacing with a PluginClient in the form of a ToolService."""

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

    async def enable_tool(self, entry: ToolEntry) -> None:
        self.tools[entry.tool.name] = entry

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

        @app.get("/tools/{name}/resolve")
        async def resolve_tool(name: str, context: ToolContextQuery) -> ReadToolResponse:
            try:
                spec = self.tools[name]
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Tool: '{name}' does not exists",
                )

            tool = await _recompute_and_marshal_tool(
                spec.tool,
                self.plugin_data,
                ToolContext(context.agent_id, context.session_id, context.customer_id),
            )

            return ReadToolResponse(tool=tool)

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

                            final_result_chunk = _ToolResultShim(
                                result=ToolResult(
                                    data=result.data,
                                    metadata=result.metadata,
                                    control=result.control,
                                    canned_responses=result.canned_responses,
                                    canned_response_fields=result.canned_response_fields,
                                )
                            ).model_dump_json()

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

            async def emit_custom(data: JSONSerializable) -> None:
                async with lock:
                    chunks.append(json.dumps({"custom": data}))
                chunks_received.release()

            context = ToolContext(
                agent_id=request.agent_id,
                session_id=request.session_id,
                customer_id=request.customer_id,
                emit_message=emit_message,
                emit_status=emit_status,
                emit_custom=emit_custom,
                plugin_data=self.plugin_data,
            )

            func = self.tools[name].function

            try:
                tool_params = inspect.signature(func).parameters
                normalized_args = normalize_tool_arguments(tool_params, request.arguments)
                adapted_args = await adapt_tool_arguments(tool_params, normalized_args)

                result = self.tools[name].function(context, **adapted_args)  # type: ignore
            except BaseException as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=traceback.format_exception(exc),
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
        logger: Logger,
        correlator: ContextualCorrelator,
    ) -> None:
        self.url = url
        self._event_emitter_factory = event_emitter_factory
        self._logger = logger
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

    def _translate_parameters(
        self,
        parameters: dict[str, Any],
    ) -> dict[str, tuple[ToolParameterDescriptor, ToolParameterOptions]]:
        return {
            name: (
                descriptor,
                ToolParameterOptions(**options),
            )
            for name, (descriptor, options) in parameters.items()
        }

    @override
    async def list_tools(self) -> Sequence[Tool]:
        response = await self._http_client.get(self._get_url("/tools"))
        content = response.json()
        return [
            Tool(
                name=t["name"],
                creation_utc=dateutil.parser.parse(t["creation_utc"]),
                description=t["description"],
                metadata=t["metadata"],
                parameters=self._translate_parameters(t["parameters"]),
                required=t["required"],
                consequential=t["consequential"],
                overlap=ToolOverlap(t["overlap"]),
            )
            for t in content["tools"]
        ]

    @override
    async def read_tool(self, name: str) -> Tool:
        response = await self._http_client.get(self._get_url(f"/tools/{name}"))

        if response.status_code == status.HTTP_404_NOT_FOUND:
            raise ItemNotFoundError(UniqueId(name))
        if response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
            raise ToolError(name, "Failed to read tool from remote service")

        content = response.json()
        t = content["tool"]
        return Tool(
            name=t["name"],
            creation_utc=dateutil.parser.parse(t["creation_utc"]),
            description=t["description"],
            metadata=t["metadata"],
            parameters=self._translate_parameters(t["parameters"]),
            required=t["required"],
            consequential=t["consequential"],
            overlap=ToolOverlap(t["overlap"]),
        )

    @override
    async def resolve_tool(
        self,
        name: str,
        context: ToolContext,
    ) -> Tool:
        response = await self._http_client.get(
            self._get_url(f"/tools/{name}/resolve"),
            params={
                "agent_id": context.agent_id,
                "session_id": context.session_id,
                "customer_id": context.customer_id,
            },
        )

        if response.status_code == status.HTTP_404_NOT_FOUND:
            raise ItemNotFoundError(UniqueId(name))
        if response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
            raise ToolError(name, "Failed to read tool from remote service")

        content = response.json()
        t = content["tool"]
        return Tool(
            name=t["name"],
            creation_utc=dateutil.parser.parse(t["creation_utc"]),
            description=t["description"],
            metadata=t["metadata"],
            parameters=self._translate_parameters(t["parameters"]),
            required=t["required"],
            consequential=t["consequential"],
            overlap=ToolOverlap(t["overlap"]),
        )

    @override
    async def call_tool(
        self,
        name: str,
        context: ToolContext,
        arguments: Mapping[str, JSONSerializable],
    ) -> ToolResult:
        try:
            tool = await self.read_tool(name)
            validate_tool_arguments(tool, arguments)

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
                    err: ToolExecutionError

                    try:
                        detail = json.loads(await response.aread())["detail"]

                        self._logger.error(
                            f"[PluginClient] Tool call error (url={self.url}, tool={tool.name}):\n{detail}"
                        )

                        err = ToolExecutionError(
                            tool_name=name,
                            message=f"url='{self.url}', arguments='{arguments}', detail={detail}",
                        )
                    except Exception:
                        self._logger.error(
                            f"[PluginClient] Tool call error (url={self.url}, tool={tool.name})"
                        )

                        err = ToolExecutionError(
                            tool_name=name,
                            message=f"url='{self.url}', arguments='{arguments}'",
                        )

                    raise err

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

                    if "data" and "metadata" in chunk_dict.get("result", {}):
                        return _ToolResultShim.model_validate(chunk_dict).result
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
                    elif "custom" in chunk_dict:
                        await event_emitter.emit_custom_event(
                            correlation_id=self._correlator.correlation_id,
                            data=chunk_dict["custom"],
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
