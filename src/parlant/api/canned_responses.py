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

from datetime import datetime
from typing import Annotated, Sequence, TypeAlias
import dateutil
from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import Field

from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.core.app_modules.canned_responses import (
    CannedResponseTagUpdateParamsModel,
    CannedResponseMetadataUpdateParamsModel,
)
from parlant.core.application import Application
from parlant.core.common import DefaultBaseModel
from parlant.core.canned_responses import (
    CannedResponseId,
    CannedResponseField,
)
from parlant.core.tags import TagId
from parlant.api.common import ExampleJson, JSONSerializableDTO, apigen_config, example_json_content


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

CannedResponseMetadataField: TypeAlias = Annotated[
    dict[str, JSONSerializableDTO],
    Field(
        description="Additional metadata associated with the canned response.",
        examples=[{"category": "greeting", "priority": 1}],
    ),
]

CannedResponseMetadataUnsetField: TypeAlias = Annotated[
    Sequence[str],
    Field(
        description="Metadata keys to remove from the canned response",
        examples=[["old_key", "deprecated_field"]],
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
    "signals": ["What is your balance?", "How much money do I have?"],
    "metadata": {"category": "account", "priority": 1},
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
    metadata: CannedResponseMetadataField


canned_response_creation_params_example: ExampleJson = {
    "value": "Your account balance is {balance}",
    "fields": [
        {
            "name": "balance",
            "description": "Account's balance",
            "examples": ["9000"],
        }
    ],
    "metadata": {"category": "account", "priority": 1},
}


class CannedResponseCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": canned_response_creation_params_example},
):
    """Parameters for creating a new canned response."""

    value: CannedResponseValueField
    fields: CannedResponseFieldSequenceField
    tags: TagIdSequenceField | None = None
    signals: CannedResponseSignalSequenceField | None = None
    metadata: CannedResponseMetadataField | None = None


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

    add: CannedResponseTagUpdateAddField | None = None
    remove: CannedResponseTagUpdateRemoveField | None = None


canned_response_metadata_update_params_example: ExampleJson = {
    "set": {
        "category": "account",
        "priority": 2,
        "version": "1.1",
    },
    "unset": ["old_category", "deprecated_field"],
}


class CannedResponseMetadataUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": canned_response_metadata_update_params_example},
):
    """Parameters for updating the metadata of a canned response."""

    set: CannedResponseMetadataField | None = None
    unset: CannedResponseMetadataUnsetField | None = None


canned_response_update_params_example: ExampleJson = {
    "value": "Your updated balance is {balance}",
    "fields": [
        {
            "name": "balance",
            "description": "Updated account balance",
            "examples": ["10000"],
        },
    ],
    "metadata": {
        "set": {
            "category": "account",
            "priority": 2,
        },
        "unset": ["old_field"],
    },
}


class CannedResponseUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": canned_response_update_params_example},
):
    """Parameters for updating an existing canned response."""

    value: CannedResponseValueField | None = None
    fields: CannedResponseFieldSequenceField | None = None
    tags: CannedResponseTagUpdateParamsDTO | None = None
    metadata: CannedResponseMetadataUpdateParamsDTO | None = None


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
    app: Application,
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

        canrep = await app.canned_responses.create(
            value=params.value,
            fields=[_dto_to_canned_response_field(s) for s in params.fields],
            tags=params.tags or None,
            signals=params.signals or None,
            metadata=params.metadata or {},
        )

        return CannedResponseDTO(
            id=canrep.id,
            creation_utc=canrep.creation_utc,
            value=canrep.value,
            fields=[_canned_response_field_to_dto(s) for s in canrep.fields],
            tags=canrep.tags,
            signals=canrep.signals,
            metadata=canrep.metadata,
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

        canrep = await app.canned_responses.read(canned_response_id=canned_response_id)

        return CannedResponseDTO(
            id=canrep.id,
            creation_utc=canrep.creation_utc,
            value=canrep.value,
            fields=[_canned_response_field_to_dto(s) for s in canrep.fields],
            tags=canrep.tags,
            signals=canrep.signals,
            metadata=canrep.metadata,
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

        canreps = await app.canned_responses.find(tags=tags)

        return [
            CannedResponseDTO(
                id=f.id,
                creation_utc=f.creation_utc,
                value=f.value,
                fields=[_canned_response_field_to_dto(s) for s in f.fields],
                tags=f.tags,
                signals=f.signals,
                metadata=f.metadata,
            )
            for f in canreps
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
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
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
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="CannedResponse fields cannot be updated without providing a new value.",
            )

        metadata_params = None
        if params.metadata:
            metadata_params = CannedResponseMetadataUpdateParamsModel(
                set=params.metadata.set,
                unset=params.metadata.unset,
            )

        canrep = await app.canned_responses.update(
            canned_response_id=canned_response_id,
            value=params.value,
            fields=(
                [_dto_to_canned_response_field(s) for s in params.fields] if params.fields else []
            ),
            tags=CannedResponseTagUpdateParamsModel(add=params.tags.add, remove=params.tags.remove)
            if params.tags
            else None,
            metadata=metadata_params,
        )

        return CannedResponseDTO(
            id=canrep.id,
            creation_utc=canrep.creation_utc,
            value=canrep.value,
            fields=[_canned_response_field_to_dto(s) for s in canrep.fields],
            tags=canrep.tags,
            signals=canrep.signals,
            metadata=canrep.metadata,
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

        await app.canned_responses.delete(canned_response_id)

    return router
