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
from dataclasses import dataclass
from typing_extensions import Literal, TypedDict
import nanoid  # type: ignore
from pydantic import BaseModel, ConfigDict
import semver  # type: ignore
from typing import Any, Mapping, NewType, Optional, Sequence, TypeAlias, Union


AgentId = NewType("AgentId", str)
ContextVariableId = NewType("ContextVariableId", str)
ContextVariableValueId = NewType("ContextVariableValueId", str)
CustomerId = NewType("CustomerId", str)
GuidelineId = NewType("GuidelineId", str)
EventId = NewType("EventId", str)
SessionId = NewType("SessionId", str)
TermId = NewType("TermId", str)


class ContextVariable(TypedDict):
    id: ContextVariableId
    name: str
    description: Optional[str]
    key: str
    value: JSONSerializable


class ControlOptions(TypedDict, total=False):
    mode: SessionMode


@dataclass(frozen=True)
class GuidelineContent:
    condition: str
    action: str


class GuidelineProposition(TypedDict):
    guideline_id: GuidelineId
    condition: str
    action: str
    score: int
    rationale: str


class Term(TypedDict):
    id: TermId
    name: str
    description: str
    synonyms: list[str]


class ToolCall(TypedDict):
    tool_id: str
    arguments: Mapping[str, JSONSerializable]
    result: ToolResult


class ToolResult(TypedDict):
    data: JSONSerializable
    metadata: Mapping[str, JSONSerializable]
    control: ControlOptions


CoherenceCheckKind = Literal[
    "contradiction_with_existing_guideline", "contradiction_with_another_evaluated_guideline"
]
ConnectionPropositionKind = Literal[
    "connection_with_existing_guideline", "connection_with_another_evaluated_guideline"
]
ConsumerId: TypeAlias = Literal["client"]
"""In the future we may support multiple consumer IDs"""
EventSource: TypeAlias = Literal[
    "customer",
    "customer_ui",
    "human_agent",
    "human_agent_on_behalf_of_ai_agent",
    "ai_agent",
    "system",
]
EventKind: TypeAlias = Literal["message", "tool", "status", "custom"]
SessionMode: TypeAlias = Literal["auto", "manual"]
ToolServiceKind = Literal["openapi", "sdk", "local"]


def _without_dto_suffix(obj: Any, *args: Any) -> str:
    if isinstance(obj, str):
        name = obj
        if name.endswith("DTO"):
            return name[:-3]
        return name
    if isinstance(obj, type):
        name = obj.__name__
        if name.endswith("DTO"):
            return name[:-3]
        return name
    else:
        raise Exception("Invalid input to _without_dto_suffix()")


class DefaultBaseModel(BaseModel):
    """
    Base class for all Parlant Pydantic models.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        model_title_generator=_without_dto_suffix,
    )


JSONSerializable: TypeAlias = Union[
    str,
    int,
    float,
    bool,
    None,
    Mapping[str, "JSONSerializable"],
    Sequence["JSONSerializable"],
    Optional[str],
    Optional[int],
    Optional[float],
    Optional[bool],
    Optional[None],
    Optional[Mapping[str, "JSONSerializable"]],
    Optional[Sequence["JSONSerializable"]],
]

UniqueId = NewType("UniqueId", str)


class Version:
    String = NewType("String", str)

    @staticmethod
    def from_string(version_string: Version.String | str) -> Version:
        result = Version(major=0, minor=0, patch=0)
        result._v = semver.Version.parse(version_string)
        return result

    def __init__(
        self,
        major: int,
        minor: int,
        patch: int,
        prerelease: Optional[str] = None,
    ) -> None:
        self._v = semver.Version(
            major=major,
            minor=minor,
            patch=patch,
            prerelease=prerelease,
        )

    def to_string(self) -> Version.String:
        return Version.String(str(self._v))


SchemaVersion = NewType("SchemaVersion", int)
SCHEMA_VERSION_UNKNOWN = SchemaVersion(-1)
SCHEMA_VERSION_UNVERSIONED = SchemaVersion(0)


class ItemNotFoundError(Exception):
    def __init__(self, item_id: UniqueId, message: Optional[str] = None) -> None:
        if message:
            super().__init__(f"{message} (id='{item_id}')")
        else:
            super().__init__(f"Item '{item_id}' not found")


def generate_id() -> UniqueId:
    while True:
        new_id = nanoid.generate(size=10)
        if "-" not in (new_id[0], new_id[-1]) and "_" not in new_id:
            return UniqueId(new_id)
