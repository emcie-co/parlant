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

from datetime import datetime
from typing import Annotated, Optional, Sequence, TypeAlias
import dateutil
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import Field

from parlant.core.common import DefaultBaseModel
from parlant.core.fragments import FragmentId, FragmentStore, FragmentUpdateParams, FragmentField
from parlant.core.tags import TagId
from parlant.api.common import ExampleJson, apigen_config, example_json_content


API_GROUP = "fragments"


FragmentFieldNameField: TypeAlias = Annotated[
    str,
    Field(
        description="The name of the fragment field.",
        examples=["username", "location"],
        min_length=1,
    ),
]

FragmentFieldDescriptionField: TypeAlias = Annotated[
    str,
    Field(
        description="A description of the fragment field.",
        examples=["User's name", "Geographical location"],
        min_length=0,
    ),
]

FragmentFieldExampleField: TypeAlias = Annotated[
    str,
    Field(
        description="An example value for the fragment field.",
        examples=["Alice", "New York"],
        min_length=0,
    ),
]

fragment_field_example: ExampleJson = {
    "description": "An example value for the fragment field.",
    "examples": ["Alice", "New York"],
    "min_length": 1,
}


class FragmentFieldDTO(
    DefaultBaseModel,
    json_schema_extra={"example": fragment_field_example},
):
    name: FragmentFieldNameField
    description: FragmentFieldDescriptionField
    examples: list[FragmentFieldExampleField]


FragmentFieldSequenceField: TypeAlias = Annotated[
    Sequence[FragmentFieldDTO],
    Field(
        description="A sequence of fragment fields associated with the fragment.",
        examples=[
            [{"name": "username", "description": "User's name", "examples": ["Alice", "Bob"]}]
        ],
    ),
]

TagIdField: TypeAlias = Annotated[
    TagId,
    Field(
        description="Unique identifier for the tag",
        examples=["t9a8g703f4"],
    ),
]

TagIdSequenceField: TypeAlias = Annotated[
    Sequence[TagIdField],
    Field(
        description="Collection of tag IDs associated with the fragment.",
        examples=[["tag123", "tag456"], []],
    ),
]

FragmentIdField: TypeAlias = Annotated[
    FragmentId,
    Field(
        description="Unique identifier for the tag",
        examples=["t9a8g703f4"],
    ),
]

FragmentCreationUTCField: TypeAlias = Annotated[
    datetime,
    Field(
        description="UTC timestamp of when the fragment was created",
        examples=[dateutil.parser.parse("2024-03-24T12:00:00Z")],
    ),
]

FragmentValueField: TypeAlias = Annotated[
    str,
    Field(
        description="The textual content of the fragment.",
        examples=["Your account balance is {balance}", "the answer is {answer}"],
        min_length=1,
    ),
]

fragment_example: ExampleJson = {
    "id": "frag123",
    "creation_utc": "2024-03-24T12:00:00Z",
    "value": "Your account balance is {balance}",
    "fields": [{"name": "balance", "description": "Account's balance", "examples": [9000]}],
    "tags": ["private", "office"],
}


class FragmentDTO(
    DefaultBaseModel,
    json_schema_extra={"example": fragment_example},
):
    id: FragmentIdField
    creation_utc: FragmentCreationUTCField
    value: FragmentValueField
    fields: FragmentFieldSequenceField
    tags: TagIdSequenceField


fragment_creation_params_example: ExampleJson = {
    "value": "Your account balance is {balance}",
    "fields": [
        {
            "name": "balance",
            "description": "Account's balance",
            "examples": ["9000"],
        }
    ],
}


class FragmentCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": fragment_creation_params_example},
):
    """Parameters for creating a new fragment."""

    value: FragmentValueField
    fields: FragmentFieldSequenceField


FragmentTagUpdateAddField: TypeAlias = Annotated[
    Sequence[TagIdField],
    Field(
        description="Optional collection of tag ids to add to the fragment's tags",
    ),
]

FragmentTagUpdateRemoveField: TypeAlias = Annotated[
    Sequence[TagIdField],
    Field(
        description="Optional collection of tag ids to remove from the fragment's tags",
    ),
]

tags_update_params_example: ExampleJson = {
    "add": [
        "t9a8g703f4",
        "tag_456abc",
    ],
    "remove": [
        "tag_789def",
        "tag_012ghi",
    ],
}


class FragmentTagUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": tags_update_params_example},
):
    """
    Parameters for updating a fragment's tags.

    Allows adding new tags to and removing existing tags from a fragment.
    Both operations can be performed in a single request.
    """

    add: Optional[FragmentTagUpdateAddField] = None
    remove: Optional[FragmentTagUpdateRemoveField] = None


fragment_update_params_example: ExampleJson = {
    "value": "Your updated balance is {balance}",
    "fields": [
        {
            "name": "balance",
            "description": "Updated account balance",
            "examples": ["10000"],
        },
    ],
}


class FragmentUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": fragment_update_params_example},
):
    """Parameters for updating an existing fragment."""

    value: Optional[FragmentValueField] = None
    fields: Optional[FragmentFieldSequenceField] = None
    tags: Optional[FragmentTagUpdateParamsDTO] = None


def _dto_to_fragment_field(dto: FragmentFieldDTO) -> FragmentField:
    return FragmentField(
        name=dto.name,
        description=dto.description,
        examples=dto.examples,
    )


def _fragment_field_to_dto(fragment_field: FragmentField) -> FragmentFieldDTO:
    return FragmentFieldDTO(
        name=fragment_field.name,
        description=fragment_field.description,
        examples=fragment_field.examples,
    )


TagsQuery: TypeAlias = Annotated[
    Sequence[TagId],
    Query(description="Filter fragments by tags", examples=["tag1", "tag2"]),
]


def create_router(
    fragment_store: FragmentStore,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        operation_id="create_fragment",
        status_code=status.HTTP_201_CREATED,
        response_model=FragmentDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Fragment successfully created.",
                "content": example_json_content(fragment_example),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_fragment(
        params: FragmentCreationParamsDTO,
    ) -> FragmentDTO:
        fragment = await fragment_store.create_fragment(
            value=params.value,
            fields=[_dto_to_fragment_field(s) for s in params.fields],
        )

        return FragmentDTO(
            id=fragment.id,
            creation_utc=fragment.creation_utc,
            value=fragment.value,
            fields=[_fragment_field_to_dto(s) for s in fragment.fields],
            tags=fragment.tags,
        )

    @router.get(
        "/{fragment_id}",
        operation_id="read_fragment",
        response_model=FragmentDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Fragment details successfully retrieved. Returns the Fragment object.",
                "content": example_json_content(fragment_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Fragment not found. The specified fragment_id does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_fragment(
        fragment_id: FragmentIdField,
    ) -> FragmentDTO:
        """Retrieves details of a specific fragment by ID."""
        fragment = await fragment_store.read_fragment(fragment_id=fragment_id)

        return FragmentDTO(
            id=fragment.id,
            creation_utc=fragment.creation_utc,
            value=fragment.value,
            fields=[_fragment_field_to_dto(s) for s in fragment.fields],
            tags=fragment.tags,
        )

    @router.get(
        "",
        operation_id="list_fragments",
        response_model=Sequence[FragmentDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all fragments in the system",
                "content": example_json_content([fragment_example]),
            }
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_fragments(tags: TagsQuery = []) -> Sequence[FragmentDTO]:
        fragments = await fragment_store.list_fragments()

        return [
            FragmentDTO(
                id=f.id,
                creation_utc=f.creation_utc,
                value=f.value,
                fields=[_fragment_field_to_dto(s) for s in f.fields],
                tags=f.tags,
            )
            for f in fragments
            if (any(tag in f.tags for tag in tags) if tags else True)
        ]

    @router.patch(
        "/{fragment_id}",
        operation_id="update_fragment",
        response_model=FragmentDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Fragment successfully updated. Returns the updated Fragment object.",
                "content": example_json_content(fragment_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Fragment not found. The specified fragment_id does not exist"
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in update parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_fragment(
        fragment_id: FragmentIdField, params: FragmentUpdateParamsDTO
    ) -> FragmentDTO:
        """
        Updates an existing fragment's attributes.

        Only provided attributes will be updated; others remain unchanged.
        The fragment's ID and creation timestamp cannot be modified.
        Extra metadata and tags can be added or removed independently.
        """
        if params.fields and not params.value:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Fragment fields cannot be updated without providing a new value.",
            )

        if params.value:
            update_params: FragmentUpdateParams = {
                "value": params.value,
                "fields": (
                    [_dto_to_fragment_field(s) for s in params.fields] if params.fields else []
                ),
            }

            await fragment_store.update_fragment(fragment_id, update_params)

        if params.tags:
            if params.tags.add:
                for tag_id in params.tags.add:
                    await fragment_store.add_tag(fragment_id, tag_id)
            if params.tags.remove:
                for tag_id in params.tags.remove:
                    await fragment_store.remove_tag(fragment_id, tag_id)

        updated_fragment = await fragment_store.read_fragment(fragment_id)

        return FragmentDTO(
            id=updated_fragment.id,
            creation_utc=updated_fragment.creation_utc,
            value=updated_fragment.value,
            fields=[_fragment_field_to_dto(s) for s in updated_fragment.fields],
            tags=updated_fragment.tags,
        )

    @router.delete(
        "/{fragment_id}",
        operation_id="delete_fragment",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Fragment successfully deleted. No content returned."
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Fragment not found. The specified fragment_id does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_fragment(fragment_id: FragmentIdField) -> None:
        await fragment_store.delete_fragment(fragment_id)

    return router
