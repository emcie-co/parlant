from dataclasses import dataclass
from datetime import datetime
from typing import Literal, NewType, TypedDict, Union
from typing_extensions import NotRequired


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
