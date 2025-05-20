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

from datetime import datetime
from typing import Annotated, Optional, Sequence, TypeAlias
import dateutil
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import Field

from parlant.core.common import DefaultBaseModel
from parlant.core.utterances import (
    UtteranceId,
    UtteranceStore,
    UtteranceUpdateParams,
    UtteranceField,
)
from parlant.core.tags import TagId, TagStore
from parlant.api.common import ExampleJson, apigen_config, example_json_content


API_GROUP = "utterances"


UtteranceFieldNameField: TypeAlias = Annotated[
    str,
    Field(
        description="The name of the utterance field.",
        examples=["username", "location"],
        min_length=1,
    ),
]

UtteranceFieldDescriptionField: TypeAlias = Annotated[
    str,
    Field(
        description="A description of the utterance field.",
        examples=["User's name", "Geographical location"],
        min_length=0,
    ),
]

UtteranceFieldExampleField: TypeAlias = Annotated[
    str,
    Field(
        description="An example value for the utterance field.",
        examples=["Alice", "New York"],
        min_length=0,
    ),
]

utterance_field_example: ExampleJson = {
    "description": "An example value for the utterance field.",
    "examples": ["Alice", "New York"],
    "min_length": 1,
}


class UtteranceFieldDTO(
    DefaultBaseModel,
    json_schema_extra={"example": utterance_field_example},
):
    name: UtteranceFieldNameField
    description: UtteranceFieldDescriptionField
    examples: list[UtteranceFieldExampleField]


UtteranceFieldSequenceField: TypeAlias = Annotated[
    Sequence[UtteranceFieldDTO],
    Field(
        description="A sequence of utterance fields associated with the utterance.",
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
        description="Collection of tag IDs associated with the utterance.",
        examples=[["tag123", "tag456"], []],
    ),
]

UtteranceIdField: TypeAlias = Annotated[
    UtteranceId,
    Field(
        description="Unique identifier for the tag",
        examples=["t9a8g703f4"],
    ),
]

UtteranceCreationUTCField: TypeAlias = Annotated[
    datetime,
    Field(
        description="UTC timestamp of when the utterance was created",
        examples=[dateutil.parser.parse("2024-03-24T12:00:00Z")],
    ),
]

UtteranceValueField: TypeAlias = Annotated[
    str,
    Field(
        description="The textual content of the utterance.",
        examples=["Your account balance is {balance}", "the answer is {answer}"],
        min_length=1,
    ),
]

utterance_example: ExampleJson = {
    "id": "frag123",
    "creation_utc": "2024-03-24T12:00:00Z",
    "value": "Your account balance is {balance}",
    "fields": [{"name": "balance", "description": "Account's balance", "examples": [9000]}],
    "tags": ["private", "office"],
}


class UtteranceDTO(
    DefaultBaseModel,
    json_schema_extra={"example": utterance_example},
):
    id: UtteranceIdField
    creation_utc: UtteranceCreationUTCField
    value: UtteranceValueField
    fields: UtteranceFieldSequenceField
    tags: TagIdSequenceField


utterance_creation_params_example: ExampleJson = {
    "value": "Your account balance is {balance}",
    "fields": [
        {
            "name": "balance",
            "description": "Account's balance",
            "examples": ["9000"],
        }
    ],
}


class UtteranceCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": utterance_creation_params_example},
):
    """Parameters for creating a new utterance."""

    value: UtteranceValueField
    fields: UtteranceFieldSequenceField
    tags: Optional[TagIdSequenceField] = None


UtteranceTagUpdateAddField: TypeAlias = Annotated[
    Sequence[TagIdField],
    Field(
        description="Optional collection of tag ids to add to the utterance's tags",
    ),
]

UtteranceTagUpdateRemoveField: TypeAlias = Annotated[
    Sequence[TagIdField],
    Field(
        description="Optional collection of tag ids to remove from the utterance's tags",
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


class UtteranceTagUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": tags_update_params_example},
):
    """
    Parameters for updating an utterance's tags.

    Allows adding new tags to and removing existing tags from an utterance.
    Both operations can be performed in a single request.
    """

    add: Optional[UtteranceTagUpdateAddField] = None
    remove: Optional[UtteranceTagUpdateRemoveField] = None


utterance_update_params_example: ExampleJson = {
    "value": "Your updated balance is {balance}",
    "fields": [
        {
            "name": "balance",
            "description": "Updated account balance",
            "examples": ["10000"],
        },
    ],
}


class UtteranceUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": utterance_update_params_example},
):
    """Parameters for updating an existing utterance."""

    value: Optional[UtteranceValueField] = None
    fields: Optional[UtteranceFieldSequenceField] = None
    tags: Optional[UtteranceTagUpdateParamsDTO] = None


def _dto_to_utterance_field(dto: UtteranceFieldDTO) -> UtteranceField:
    return UtteranceField(
        name=dto.name,
        description=dto.description,
        examples=dto.examples,
    )


def _utterance_field_to_dto(utterance_field: UtteranceField) -> UtteranceFieldDTO:
    return UtteranceFieldDTO(
        name=utterance_field.name,
        description=utterance_field.description,
        examples=utterance_field.examples,
    )


TagsQuery: TypeAlias = Annotated[
    Sequence[TagId],
    Query(description="Filter utterances by tags", examples=["tag1", "tag2"]),
]


def create_router(
    utterance_store: UtteranceStore,
    tag_store: TagStore,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        operation_id="create_utterance",
        status_code=status.HTTP_201_CREATED,
        response_model=UtteranceDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Utterance successfully created.",
                "content": example_json_content(utterance_example),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_utterance(
        params: UtteranceCreationParamsDTO,
    ) -> UtteranceDTO:
        tags = []

        if params.tags:
            for tag_id in params.tags:
                _ = await tag_store.read_tag(tag_id=tag_id)

            tags = list(set(params.tags))

        utterance = await utterance_store.create_utterance(
            value=params.value,
            fields=[_dto_to_utterance_field(s) for s in params.fields],
            tags=tags or None,
        )

        return UtteranceDTO(
            id=utterance.id,
            creation_utc=utterance.creation_utc,
            value=utterance.value,
            fields=[_utterance_field_to_dto(s) for s in utterance.fields],
            tags=utterance.tags,
        )

    @router.get(
        "/{utterance_id}",
        operation_id="read_utterance",
        response_model=UtteranceDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Utterance details successfully retrieved. Returns the Utterance object.",
                "content": example_json_content(utterance_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Utterance not found. The specified utterance_id does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_utterance(
        utterance_id: UtteranceIdField,
    ) -> UtteranceDTO:
        """Retrieves details of a specific utterance by ID."""
        utterance = await utterance_store.read_utterance(utterance_id=utterance_id)

        return UtteranceDTO(
            id=utterance.id,
            creation_utc=utterance.creation_utc,
            value=utterance.value,
            fields=[_utterance_field_to_dto(s) for s in utterance.fields],
            tags=utterance.tags,
        )

    @router.get(
        "",
        operation_id="list_utterances",
        response_model=Sequence[UtteranceDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all utterances in the system",
                "content": example_json_content([utterance_example]),
            }
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_utterances(tags: TagsQuery = []) -> Sequence[UtteranceDTO]:
        if tags:
            utterances = await utterance_store.list_utterances(tags=tags)
        else:
            utterances = await utterance_store.list_utterances()

        return [
            UtteranceDTO(
                id=f.id,
                creation_utc=f.creation_utc,
                value=f.value,
                fields=[_utterance_field_to_dto(s) for s in f.fields],
                tags=f.tags,
            )
            for f in utterances
        ]

    @router.patch(
        "/{utterance_id}",
        operation_id="update_utterance",
        response_model=UtteranceDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Utterance successfully updated. Returns the updated Utterance object.",
                "content": example_json_content(utterance_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Utterance not found. The specified utterance_id does not exist"
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in update parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_utterance(
        utterance_id: UtteranceIdField, params: UtteranceUpdateParamsDTO
    ) -> UtteranceDTO:
        """
        Updates an existing utterance's attributes.

        Only provided attributes will be updated; others remain unchanged.
        The utterance's ID and creation timestamp cannot be modified.
        Extra metadata and tags can be added or removed independently.
        """
        if params.fields and not params.value:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Utterance fields cannot be updated without providing a new value.",
            )

        if params.value:
            update_params: UtteranceUpdateParams = {
                "value": params.value,
                "fields": (
                    [_dto_to_utterance_field(s) for s in params.fields] if params.fields else []
                ),
            }

            await utterance_store.update_utterance(utterance_id, update_params)

        if params.tags:
            if params.tags.add:
                for tag_id in params.tags.add:
                    _ = await tag_store.read_tag(tag_id=tag_id)
                    await utterance_store.upsert_tag(utterance_id, tag_id)
            if params.tags.remove:
                for tag_id in params.tags.remove:
                    await utterance_store.remove_tag(utterance_id, tag_id)

        updated_utterance = await utterance_store.read_utterance(utterance_id)

        return UtteranceDTO(
            id=updated_utterance.id,
            creation_utc=updated_utterance.creation_utc,
            value=updated_utterance.value,
            fields=[_utterance_field_to_dto(s) for s in updated_utterance.fields],
            tags=updated_utterance.tags,
        )

    @router.delete(
        "/{utterance_id}",
        operation_id="delete_utterance",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Utterance successfully deleted. No content returned."
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Utterance not found. The specified utterance_id does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_utterance(utterance_id: UtteranceIdField) -> None:
        await utterance_store.delete_utterance(utterance_id)

    return router
