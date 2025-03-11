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
    get_args,
)
from pydantic import Field
from typing_extensions import override, TypedDict

from parlant.core.common import DefaultBaseModel, ItemNotFoundError, JSONSerializable, UniqueId

ToolParameterType = Literal[
    "array",
    "string",
    "number",
    "integer",
    "boolean",
]

EnumValueType = Union[str, int]


class ToolParameterDescriptor(TypedDict, total=False):
    type: ToolParameterType
    item_type: ToolParameterType
    enum: Sequence[EnumValueType]
    description: str
    examples: Sequence[str]


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
        plugin_data: Mapping[str, Any] = {},
    ) -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self.customer_id = customer_id
        self.plugin_data = plugin_data
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


class ToolParameterOptions(DefaultBaseModel):
    hidden: bool = Field(default=False)
    """If true, this parameter is not exposed in tool insights and message generation;
    meaning, agents would not be able to inform customers when it is missing and required."""

    source: Literal["any", "context", "customer"] = Field(default="any")
    """Describes what is the expected source for the argument. This can help agents understand
    whether to ask for it directly from the customer, or to seek it elsewhere in the context."""

    description: Optional[str] = Field(default=None)
    """A description of this parameter which should help agents understand how to extract arguments properly."""

    significance: Optional[str] = Field(default=None)
    """A description of the significance of this parameter for the tool call — why is it needed?"""

    examples: Sequence[Any] = Field(default_factory=list)
    """Examples of arguments which should help agents understand how to extract arguments properly."""

    adapter: Optional[Callable[[Any], Awaitable[Any]]] = Field(default=None, exclude=True)
    """A custom adapter function to convert the inferred value to a type."""

    choice_provider: Optional[Callable[[], Awaitable[list[str]]]] = Field(
        default=None, exclude=True
    )
    """A custom function to provide valid choicoes for the parameter's argument."""


@dataclass(frozen=True)
class Tool:
    name: str
    creation_utc: datetime
    description: str
    parameters: dict[str, tuple[ToolParameterDescriptor, ToolParameterOptions]]
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
    parameters: dict[str, ToolParameterDescriptor]
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
            parameters={
                name: (descriptor, ToolParameterOptions())
                for name, descriptor in local_tool.parameters.items()
            },
            required=local_tool.required,
            consequential=local_tool.consequential,
        )

    async def create_tool(
        self,
        name: str,
        module_path: str,
        description: str,
        parameters: Mapping[str, ToolParameterDescriptor],
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
            tool = await self.read_tool(name)
            validate_tool_arguments(tool, arguments)

            func_params = inspect.signature(func).parameters
            result: ToolResult = func(**normalize_tool_arguments(func_params, arguments))

            if inspect.isawaitable(result):
                result = await result
        except ToolError as e:
            raise e
        except Exception as e:
            raise ToolExecutionError(name) from e

        if not isinstance(result, ToolResult):
            raise ToolResultError(name, "Tool result is not an instance of ToolResult")

        return result


def validate_tool_arguments(
    tool: Tool,
    arguments: Mapping[str, Any],
) -> None:
    expected = set(tool.parameters.keys())
    received = set(arguments.keys())

    extra_args = received - expected

    missing_required = [p for p in tool.required if p not in arguments]

    if extra_args or missing_required:
        message = f"Argument mismatch.\n - Expected parameters: {sorted(expected)}"
        raise ToolExecutionError(message)

    type_map = {
        "string": (str,),
        "boolean": (
            str,
            bool,
        ),
        "integer": (
            str,
            int,
        ),
        "number": (str, int, float),
    }

    for param_name, arg_value in arguments.items():
        param_def, _ = tool.parameters[param_name]
        param_type = param_def["type"]

        values = [arg_value]

        if param_type == "array":
            values = arg_value

        for value in values:
            if allowed_values := param_def.get("enum", []):
                if value not in allowed_values:
                    message = (
                        f"Parameter '{param_name}' must be one of {allowed_values}, "
                        f"but got '{value}'."
                    )
                    raise ToolExecutionError(tool.name, message)
            else:
                expected_types = type_map.get(param_type)
                if expected_types is None:
                    raise ToolExecutionError(
                        tool.name, f"Parameter '{param_name}' has unknown type '{param_type}'"
                    )
                if value is not None and type(value) not in expected_types:
                    raise ToolExecutionError(
                        tool.name,
                        f"Parameter '{param_name}' must be of type {expected_types}, "
                        f"but got {type(value).__name__}: {value}",
                    )


def normalize_tool_arguments(
    parameters: Mapping[str, inspect.Parameter],
    arguments: Mapping[str, Any],
) -> Any:
    return {
        param_name: normalize_tool_argument(parameters[param_name].annotation, argument)
        for param_name, argument in arguments.items()
    }


def normalize_tool_argument(parameter_type: Any, argument: Any) -> Any:
    try:
        if getattr(parameter_type, "__name__", None) == "Annotated":
            parameter_type = get_args(parameter_type)[0]
        if (
            inspect.isclass(parameter_type)
            and issubclass(parameter_type, enum.Enum)
            or parameter_type in [str, int, float, bool]
        ):
            return parameter_type(argument)
        return argument
    except Exception as exc:
        raise ToolExecutionError(
            f"Failed to convert argument '{argument}' into a {parameter_type}"
        ) from exc
