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
from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import Field

from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.core.common import DefaultBaseModel
from parlant.core.canned_responses import (
    CannedResponseId,
    CannedResponseStore,
    CannedResponseUpdateParams,
    CannedResponseField,
)
from parlant.core.tags import TagId, TagStore
from parlant.api.common import ExampleJson, apigen_config, example_json_content


API_GROUP = "canned_responses"


CannedResponseFieldNameField: TypeAlias = Annotated[
    str,
    Field(
        description="The name of the canned response field.",
        examples=["username", "location"],
        min_length=1,
    ),
]

CannedResponseFieldDescriptionField: TypeAlias = Annotated[
    str,
    Field(
        description="A description of the canned response field.",
        examples=["User's name", "Geographical location"],
        min_length=0,
    ),
]

CannedResponseFieldExampleField: TypeAlias = Annotated[
    str,
    Field(
        description="An example value for the canned response field.",
        examples=["Alice", "New York"],
        min_length=0,
    ),
]

canned_response_field_example: ExampleJson = {
    "description": "An example value for the canned response field.",
    "examples": ["Alice", "New York"],
    "min_length": 1,
}


class CannedResponseFieldDTO(
    DefaultBaseModel,
    json_schema_extra={"example": canned_response_field_example},
):
    name: CannedResponseFieldNameField
    description: CannedResponseFieldDescriptionField
    examples: list[CannedResponseFieldExampleField]


CannedResponseFieldSequenceField: TypeAlias = Annotated[
    Sequence[CannedResponseFieldDTO],
    Field(
        description="A sequence of canned response fields associated with the canned response.",
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
        description="Collection of tag IDs associated with the canned response.",
        examples=[["tag123", "tag456"], []],
    ),
]

CannedResponseSignalSequenceField: TypeAlias = Annotated[
    Sequence[str],
    Field(
        description="A sequence of signals associated with the canned response, to help with filtering and matching.",
        examples=[
            ["What is your name?", "Where are you located?", "Let me know if I can help you."],
        ],
    ),
]

CannedResponseIdField: TypeAlias = Annotated[
    CannedResponseId,
    Field(
        description="Unique identifier for the tag",
        examples=["t9a8g703f4"],
    ),
]

CannedResponseCreationUTCField: TypeAlias = Annotated[
    datetime,
    Field(
        description="UTC timestamp of when the canned response was created",
        examples=[dateutil.parser.parse("2024-03-24T12:00:00Z")],
    ),
]

CannedResponseValueField: TypeAlias = Annotated[
    str,
    Field(
        description="The textual content of the canned response.",
        examples=["Your account balance is {balance}", "the answer is {answer}"],
        min_length=1,
    ),
]

canned_response_example: ExampleJson = {
    "id": "frag123",
    "creation_utc": "2024-03-24T12:00:00Z",
    "value": "Your account balance is {balance}",
    "fields": [{"name": "balance", "description": "Account's balance", "examples": [9000]}],
    "tags": ["private", "office"],
}


class CannedResponseDTO(
    DefaultBaseModel,
    json_schema_extra={"example": canned_response_example},
):
    id: CannedResponseIdField
    creation_utc: CannedResponseCreationUTCField
    value: CannedResponseValueField
    fields: CannedResponseFieldSequenceField
    tags: TagIdSequenceField
    signals: CannedResponseSignalSequenceField


canned_response_creation_params_example: ExampleJson = {
    "value": "Your account balance is {balance}",
    "fields": [
        {
            "name": "balance",
            "description": "Account's balance",
            "examples": ["9000"],
        }
    ],
}


class CannedResponseCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": canned_response_creation_params_example},
):
    """Parameters for creating a new canned response."""

    value: CannedResponseValueField
    fields: CannedResponseFieldSequenceField
    tags: Optional[TagIdSequenceField] = None
    signals: Optional[CannedResponseSignalSequenceField] = None


CannedResponseTagUpdateAddField: TypeAlias = Annotated[
    Sequence[TagIdField],
    Field(
        description="Optional collection of tag ids to add to the canned response's tags",
    ),
]

CannedResponseTagUpdateRemoveField: TypeAlias = Annotated[
    Sequence[TagIdField],
    Field(
        description="Optional collection of tag ids to remove from the canned response's tags",
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


class CannedResponseTagUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": tags_update_params_example},
):
    """
    Parameters for updating a canned response's tags.

    Allows adding new tags to and removing existing tags from a canned response.
    Both operations can be performed in a single request.
    """

    add: Optional[CannedResponseTagUpdateAddField] = None
    remove: Optional[CannedResponseTagUpdateRemoveField] = None


canned_response_update_params_example: ExampleJson = {
    "value": "Your updated balance is {balance}",
    "fields": [
        {
            "name": "balance",
            "description": "Updated account balance",
            "examples": ["10000"],
        },
    ],
}


class CannedResponseUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": canned_response_update_params_example},
):
    """Parameters for updating an existing canned response."""

    value: Optional[CannedResponseValueField] = None
    fields: Optional[CannedResponseFieldSequenceField] = None
    tags: Optional[CannedResponseTagUpdateParamsDTO] = None


def _dto_to_canned_response_field(dto: CannedResponseFieldDTO) -> CannedResponseField:
    return CannedResponseField(
        name=dto.name,
        description=dto.description,
        examples=dto.examples,
    )


def _canned_response_field_to_dto(
    canned_response_field: CannedResponseField,
) -> CannedResponseFieldDTO:
    return CannedResponseFieldDTO(
        name=canned_response_field.name,
        description=canned_response_field.description,
        examples=canned_response_field.examples,
    )


TagsQuery: TypeAlias = Annotated[
    Sequence[TagId],
    Query(description="Filter canned responses by tags", examples=["tag1", "tag2"]),
]


def create_router(
    authorization_policy: AuthorizationPolicy,
    canned_response_store: CannedResponseStore,
    tag_store: TagStore,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        operation_id="create_canned_response",
        status_code=status.HTTP_201_CREATED,
        response_model=CannedResponseDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "CannedResponse successfully created.",
                "content": example_json_content(canned_response_example),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_canned_response(
        request: Request,
        params: CannedResponseCreationParamsDTO,
    ) -> CannedResponseDTO:
        await authorization_policy.authorize(request, Operation.CREATE_CANNED_RESPONSE)

        tags = []

        if params.tags:
            for tag_id in params.tags:
                _ = await tag_store.read_tag(tag_id=tag_id)

            tags = list(set(params.tags))

        response = await canned_response_store.create_canned_response(
            value=params.value,
            fields=[_dto_to_canned_response_field(s) for s in params.fields],
            tags=tags or None,
            signals=params.signals or None,
        )

        return CannedResponseDTO(
            id=response.id,
            creation_utc=response.creation_utc,
            value=response.value,
            fields=[_canned_response_field_to_dto(s) for s in response.fields],
            tags=response.tags,
            signals=response.signals,
        )

    @router.get(
        "/{canned_response_id}",
        operation_id="read_canned_response",
        response_model=CannedResponseDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Canned response details successfully retrieved. Returns the CannedResponse object.",
                "content": example_json_content(canned_response_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Canned response not found. The specified canned_response_id does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_canned_response(
        request: Request,
        canned_response_id: CannedResponseIdField,
    ) -> CannedResponseDTO:
        """Retrieves details of a specific canned response by ID."""
        await authorization_policy.authorize(request, Operation.READ_CANNED_RESPONSE)

        response = await canned_response_store.read_canned_response(
            canned_response_id=canned_response_id
        )

        return CannedResponseDTO(
            id=response.id,
            creation_utc=response.creation_utc,
            value=response.value,
            fields=[_canned_response_field_to_dto(s) for s in response.fields],
            tags=response.tags,
            signals=response.signals,
        )

    @router.get(
        "",
        operation_id="list_canned_responses",
        response_model=Sequence[CannedResponseDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all canned responses in the system",
                "content": example_json_content([canned_response_example]),
            }
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_canned_responses(
        request: Request, tags: TagsQuery = []
    ) -> Sequence[CannedResponseDTO]:
        """Lists all canned responses, optionally filtered by tags."""
        await authorization_policy.authorize(request, Operation.LIST_CANNED_RESPONSES)

        if tags:
            responses = await canned_response_store.list_canned_responses(tags=tags)
        else:
            responses = await canned_response_store.list_canned_responses()

        return [
            CannedResponseDTO(
                id=f.id,
                creation_utc=f.creation_utc,
                value=f.value,
                fields=[_canned_response_field_to_dto(s) for s in f.fields],
                tags=f.tags,
                signals=f.signals,
            )
            for f in responses
        ]

    @router.patch(
        "/{canned_response_id}",
        operation_id="update_canned_response",
        response_model=CannedResponseDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Canned response successfully updated. Returns the updated CannedResponse object.",
                "content": example_json_content(canned_response_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "CannedResponse not found. The specified canned_response_id does not exist"
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in update parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_canned_response(
        request: Request,
        canned_response_id: CannedResponseIdField,
        params: CannedResponseUpdateParamsDTO,
    ) -> CannedResponseDTO:
        """
        Updates an existing canned response's attributes.

        Only provided attributes will be updated; others remain unchanged.
        The canned response's ID and creation timestamp cannot be modified.
        Extra metadata and tags can be added or removed independently.
        """
        await authorization_policy.authorize(request, Operation.UPDATE_CANNED_RESPONSE)

        if params.fields and not params.value:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="CannedResponse fields cannot be updated without providing a new value.",
            )

        if params.value:
            update_params: CannedResponseUpdateParams = {
                "value": params.value,
                "fields": (
                    [_dto_to_canned_response_field(s) for s in params.fields]
                    if params.fields
                    else []
                ),
            }

            await canned_response_store.update_canned_response(canned_response_id, update_params)

        if params.tags:
            if params.tags.add:
                for tag_id in params.tags.add:
                    _ = await tag_store.read_tag(tag_id=tag_id)
                    await canned_response_store.upsert_tag(canned_response_id, tag_id)
            if params.tags.remove:
                for tag_id in params.tags.remove:
                    await canned_response_store.remove_tag(canned_response_id, tag_id)

        updated_canrep = await canned_response_store.read_canned_response(canned_response_id)

        return CannedResponseDTO(
            id=updated_canrep.id,
            creation_utc=updated_canrep.creation_utc,
            value=updated_canrep.value,
            fields=[_canned_response_field_to_dto(s) for s in updated_canrep.fields],
            tags=updated_canrep.tags,
            signals=updated_canrep.signals,
        )

    @router.delete(
        "/{canned_response_id}",
        operation_id="delete_canned_response",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "CannedResponse successfully deleted. No content returned."
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "CannedResponse not found. The specified canned_response_id does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_canned_response(
        request: Request, canned_response_id: CannedResponseIdField
    ) -> None:
        await authorization_policy.authorize(request, Operation.DELETE_CANNED_RESPONSE)

        await canned_response_store.delete_canned_response(canned_response_id)

    return router
