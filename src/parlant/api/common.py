# Copyright 2026 Emcie Co Ltd.
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

from enum import Enum
from pydantic import Field
from typing import Annotated, Any, Mapping, TypeAlias

from parlant.core.common import DefaultBaseModel
from parlant.core.persistence.common import SortDirection
from parlant.core.groups import GroupId


def apigen_config(group_name: str, method_name: str) -> Mapping[str, Any]:
    return {
        "openapi_extra": {
            "x-fern-sdk-group-name": group_name,
            "x-fern-sdk-method-name": method_name,
        }
    }


ExampleJson: TypeAlias = dict[str, Any] | list[Any]
ExtraSchema: TypeAlias = dict[str, dict[str, Any]]


JSONSerializableDTO: TypeAlias = Annotated[
    Any,
    Field(
        examples=['"json_serializable_object"'],
        description="Any valid json",
    ),
]


def example_json_content(json_example: ExampleJson) -> ExtraSchema:
    return {"application/json": {"example": json_example}}


GroupIdField: TypeAlias = Annotated[
    GroupId,
    Field(
        description="Unique identifier for the group",
        examples=["group_123xyz", "group_premium42"],
    ),
]


GroupNameField: TypeAlias = Annotated[
    str,
    Field(
        description="Human-readable name for the group, used for display and organization",
        examples=["premium", "enterprise", "beta-tester"],
        min_length=1,
        max_length=50,
    ),
]

group_example: ExampleJson = {
    "id": "group_123xyz",
    "name": "premium",
    "creation_utc": "2024-03-24T12:00:00Z",
}


class GroupDTO(
    DefaultBaseModel,
    json_schema_extra={"example": group_example},
):
    """
    Represents a group in the system.

    Groups can be used to categorize and label various resources like customers, sessions,
    or content. They provide a flexible way to organize and filter data.
    """

    id: GroupIdField
    name: GroupNameField


class SortDirectionDTO(Enum):
    """The direction to sort results."""

    ASC = "asc"
    DESC = "desc"


def sort_direction_dto_to_sort_direction(
    dto: SortDirectionDTO,
) -> SortDirection:
    match dto:
        case SortDirectionDTO.ASC:
            return SortDirection.ASC
        case SortDirectionDTO.DESC:
            return SortDirection.DESC
        case _:
            raise ValueError(f"Unsupported sort direction: {dto}")
