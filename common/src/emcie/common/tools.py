from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Mapping, NewType, TypedDict, Union
from typing_extensions import NotRequired

from emcie.common.types.common import JSONSerializable


ToolId = NewType("ToolId", str)

ToolParameterType = Literal[
    "string",
    "number",
    "integer",
    "boolean",
]


class ToolParameter(TypedDict):
    type: ToolParameterType
    description: NotRequired[str]
    enum: NotRequired[list[Union[str, int, float, bool]]]


@dataclass(frozen=True)
class ToolContext:
    session_id: str


@dataclass(frozen=True)
class ToolResult:
    data: JSONSerializable
    metadata: Mapping[str, JSONSerializable] = field(default_factory=dict)


@dataclass(frozen=True)
class Tool:
    id: ToolId
    creation_utc: datetime
    name: str
    description: str
    parameters: dict[str, ToolParameter]
    required: list[str]
    consequential: bool

    def __hash__(self) -> int:
        return hash(self.id)
