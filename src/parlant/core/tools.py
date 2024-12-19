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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import enum
import importlib
import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    TypeAlias,
    Union,
)
from typing_extensions import override, TypedDict, NotRequired

from parlant.core.common import ItemNotFoundError, JSONSerializable, UniqueId

ToolParameterType = Literal[
    "string",
    "number",
    "integer",
    "boolean",
    "enum",
]

EnumValueType = Union[str, int]


class ToolParameter(TypedDict):
    type: ToolParameterType
    description: NotRequired[str]
    enum: NotRequired[list[EnumValueType]]


# These two aliases are redefined here to avoid a circular reference.
SessionStatus: TypeAlias = Literal["ready", "processing", "typing"]
SessionMode: TypeAlias = Literal["auto", "manual"]


class ToolContext:
    def __init__(
        self,
        agent_id: str,
        session_id: str,
        customer_id: str,
        emit_message: Optional[Callable[[str], Awaitable[None]]] = None,
        emit_status: Optional[
            Callable[
                [SessionStatus, JSONSerializable],
                Awaitable[None],
            ]
        ] = None,
    ) -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self.customer_id = customer_id
        self._emit_message = emit_message
        self._emit_status = emit_status

    async def emit_message(self, message: str) -> None:
        assert self._emit_message
        await self._emit_message(message)

    async def emit_status(
        self,
        status: SessionStatus,
        data: JSONSerializable,
    ) -> None:
        assert self._emit_status
        await self._emit_status(status, data)


class ControlOptions(TypedDict, total=False):
    mode: SessionMode


@dataclass(frozen=True)
class ToolResult:
    data: JSONSerializable
    metadata: Mapping[str, JSONSerializable] = field(default_factory=dict)
    control: ControlOptions = field(default_factory=lambda: ControlOptions())


@dataclass(frozen=True)
class Tool:
    name: str
    creation_utc: datetime
    description: str
    parameters: dict[str, ToolParameter]
    required: list[str]
    consequential: bool

    def __hash__(self) -> int:
        return hash(self.name)


class ToolId(NamedTuple):
    service_name: str
    tool_name: str

    @staticmethod
    def from_string(s: str) -> ToolId:
        parts = s.split(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid ToolId string format: '{s}'. Expected 'service_name:tool_name'."
            )
        return ToolId(service_name=parts[0], tool_name=parts[1])

    def to_string(self) -> str:
        return f"{self.service_name}:{self.tool_name}"


class ToolError(Exception):
    def __init__(
        self,
        tool_name: str,
        message: Optional[str] = None,
    ) -> None:
        if message:
            super().__init__(f"Tool error (tool='{tool_name}'): {message}")
        else:
            super().__init__(f"Tool error (tool='{tool_name}')")

        self.tool_name = tool_name


class ToolImportError(ToolError):
    pass


class ToolExecutionError(ToolError):
    pass


class ToolResultError(ToolError):
    pass


class ToolService(ABC):
    @abstractmethod
    async def list_tools(
        self,
    ) -> Sequence[Tool]: ...

    @abstractmethod
    async def read_tool(
        self,
        name: str,
    ) -> Tool: ...

    @abstractmethod
    async def call_tool(
        self,
        name: str,
        context: ToolContext,
        arguments: Mapping[str, JSONSerializable],
    ) -> ToolResult: ...


@dataclass(frozen=True)
class _LocalTool:
    name: str
    creation_utc: datetime
    module_path: str
    description: str
    parameters: dict[str, ToolParameter]
    required: list[str]
    consequential: bool


class LocalToolService(ToolService):
    def __init__(
        self,
    ) -> None:
        self._local_tools_by_name: dict[str, _LocalTool] = {}

    def _local_tool_to_tool(self, local_tool: _LocalTool) -> Tool:
        return Tool(
            creation_utc=local_tool.creation_utc,
            name=local_tool.name,
            description=local_tool.description,
            parameters=local_tool.parameters,
            required=local_tool.required,
            consequential=local_tool.consequential,
        )

    async def create_tool(
        self,
        name: str,
        module_path: str,
        description: str,
        parameters: Mapping[str, ToolParameter],
        required: Sequence[str],
        consequential: bool = False,
    ) -> Tool:
        creation_utc = datetime.now(timezone.utc)

        local_tool = _LocalTool(
            name=name,
            module_path=module_path,
            description=description,
            parameters=dict(parameters),
            creation_utc=creation_utc,
            required=list(required),
            consequential=consequential,
        )

        self._local_tools_by_name[name] = local_tool

        return self._local_tool_to_tool(local_tool)

    @override
    async def list_tools(
        self,
    ) -> Sequence[Tool]:
        return [self._local_tool_to_tool(t) for t in self._local_tools_by_name.values()]

    @override
    async def read_tool(
        self,
        name: str,
    ) -> Tool:
        try:
            return self._local_tool_to_tool(self._local_tools_by_name[name])
        except KeyError:
            raise ItemNotFoundError(item_id=UniqueId(name))

    @override
    async def call_tool(
        self,
        name: str,
        context: ToolContext,
        arguments: Mapping[str, JSONSerializable],
    ) -> ToolResult:
        _ = context

        try:
            local_tool = self._local_tools_by_name[name]
            module = importlib.import_module(local_tool.module_path)
            func = getattr(module, local_tool.name)
        except Exception as e:
            raise ToolImportError(name) from e

        try:
            func_params = inspect.signature(func).parameters
            result: ToolResult = func(**normalize_tool_arguments(func_params, arguments))

            if inspect.isawaitable(result):
                result = await result
        except Exception as e:
            raise ToolExecutionError(name) from e

        if not isinstance(result, ToolResult):
            raise ToolResultError(name, "Tool result is not an instance of ToolResult")

        return result


def normalize_tool_arguments(
    parameters: Mapping[str, inspect.Parameter],
    arguments: Mapping[str, Any],
) -> Any:
    return {
        param_name: normalize_tool_argument(parameters[param_name].annotation, argument)
        for param_name, argument in arguments.items()
    }


def normalize_tool_argument(parameter_type: Any, argument: Any) -> Any:
    if inspect.isclass(parameter_type) and issubclass(parameter_type, enum.Enum):
        return parameter_type(argument)
    return argument
