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

from fastapi import APIRouter, HTTPException, Path, Query, Request, status
from typing import Annotated, Sequence, TypeAlias
from pydantic import Field

from parlant.api import common
from parlant.api.authorization import Operation, AuthorizationPolicy
from datetime import datetime
from parlant.api.common import apigen_config, ExampleJson
from parlant.core.app_modules.glossary import TermGroupsUpdateParamsModel
from parlant.core.agents import AgentId
from parlant.core.application import Application
from parlant.core.common import DefaultBaseModel
from parlant.core.glossary import TermId
from parlant.core.groups import GroupId

API_GROUP = "glossary"


TermNameField: TypeAlias = Annotated[
    str,
    Field(
        description="The name of the term, e.g., 'Gas' in blockchain.",
        examples=["Gas", "Token"],
        min_length=1,
        max_length=100,
    ),
]

TermDescriptionField: TypeAlias = Annotated[
    str,
    Field(
        description=("A detailed description of the term"),
        examples=[
            "Gas is a unit in Ethereum that measures the computational effort to execute transactions or smart contracts."
        ],
    ),
]

TermSynonymsField: TypeAlias = Annotated[
    Sequence[str],
    Field(
        description="A list of synonyms for the term, including alternate contexts if applicable.",
        examples=[["Execution Cost", "Blockchain Fuel"]],
    ),
]

term_creation_params_example: ExampleJson = {
    "name": "Gas",
    "description": "A unit in Ethereum that measures the computational effort to execute transactions or smart contracts",
    "synonyms": ["Transaction Fee", "Blockchain Fuel"],
}


TermIdPath: TypeAlias = Annotated[
    TermId,
    Path(
        description="Unique identifier for the term",
        examples=["term-eth01"],
    ),
]

TermAgentIdPath: TypeAlias = Annotated[
    AgentId,
    Path(
        description="Unique identifier for the agent associated with the term.",
        examples=["ag-123Txyz"],
    ),
]

TermGroupsField: TypeAlias = Annotated[
    list[GroupId],
    Field(
        description="List of group IDs associated with the term",
        examples=[["group1", "group2"]],
    ),
]

TermLastModifiedField: TypeAlias = Annotated[
    datetime,
    Field(
        description="UTC timestamp of the last modification to the term",
    ),
]


class TermCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": term_creation_params_example},
):
    """
    Parameters for creating a new glossary term.

    Use this model when adding new terms to an agent's glossary.
    """

    name: TermNameField
    description: TermDescriptionField
    synonyms: TermSynonymsField = []
    groups: TermGroupsField | None = None
    id: TermId | None = None


term_example: ExampleJson = {
    "id": "term-eth01",
    "name": "Gas",
    "description": "A unit in Ethereum that measures the computational effort to execute transactions or smart contracts",
    "synonyms": ["Transaction Fee", "Blockchain Fuel"],
    "groups": ["group1", "group2"],
}

term_update_params_example: ExampleJson = {
    "name": "Gas",
    "description": "A unit in Ethereum that measures the computational effort to execute transactions or smart contracts",
    "synonyms": ["Transaction Fee", "Blockchain Fuel"],
    "groups": {
        "add": ["group1", "group2"],
        "remove": ["group3", "group4"],
    },
}

term_groups_update_params_example: ExampleJson = {
    "add": [
        "t9a8g703f4",
        "group_456abc",
    ],
    "remove": [
        "group_789def",
        "group_012ghi",
    ],
}


class TermDTO(
    DefaultBaseModel,
    json_schema_extra={"example": term_example},
):
    """
    Represents a glossary term associated with an agent.

    Use this model for representing complete term information in API responses.
    """

    id: TermIdPath
    name: TermNameField
    description: TermDescriptionField
    synonyms: TermSynonymsField = []
    groups: TermGroupsField
    modified_utc: TermLastModifiedField


TermGroupsUpdateAddField: TypeAlias = Annotated[
    list[GroupId],
    Field(
        description="List of group IDs to add to the term",
        examples=[["group1", "group2"]],
    ),
]

TermGroupsUpdateRemoveField: TypeAlias = Annotated[
    list[GroupId],
    Field(
        description="List of group IDs to remove from the term",
        examples=[["group1", "group2"]],
    ),
]


class TermGroupsUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": term_groups_update_params_example},
):
    """
    Parameters for updating the groups of an existing glossary term.
    """

    add: TermGroupsUpdateAddField | None = None
    remove: TermGroupsUpdateRemoveField | None = None


class TermUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": term_update_params_example},
):
    """
    Parameters for updating an existing glossary term including groups.

    All fields are optional. Only the provided fields will be updated.
    """

    name: TermNameField | None = None
    description: TermDescriptionField | None = None
    synonyms: TermSynonymsField | None = None
    groups: TermGroupsUpdateParamsDTO | None = None


GroupIdQuery: TypeAlias = Annotated[
    GroupId | None,
    Query(
        description="Filter terms by group ID",
        examples=["group1", "group2"],
    ),
]


def create_router(
    authorization_policy: AuthorizationPolicy,
    app: Application,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_term",
        response_model=TermDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Term successfully created. Returns the complete term object including generated ID",
                "content": common.example_json_content(term_example),
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create_term"),
    )
    async def create_term(
        request: Request,
        params: TermCreationParamsDTO,
    ) -> TermDTO:
        """
        Creates a new term in the glossary.

        The term will be initialized with the provided name and description, and optional synonyms.
        A unique identifier will be automatically generated.

        Default behaviors:
        - `synonyms` defaults to an empty list if not provided
        """
        await authorization_policy.authorize(request, Operation.CREATE_TERM)

        try:
            term = await app.glossary.create(
                name=params.name,
                description=params.description,
                synonyms=params.synonyms,
                groups=params.groups,
                id=params.id,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e),
            )

        return TermDTO(
            id=term.id,
            name=term.name,
            description=term.description,
            synonyms=term.synonyms,
            groups=term.groups,
            modified_utc=term.modified_utc,
        )

    @router.get(
        "/{term_id}",
        operation_id="read_term",
        response_model=TermDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Term details successfully retrieved. Returns the complete term object",
                "content": common.example_json_content(term_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Term not found. The specified `term_id` does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve_term"),
    )
    async def read_term(
        request: Request,
        term_id: TermIdPath,
    ) -> TermDTO:
        """
        Retrieves details of a specific term by ID.
        """
        await authorization_policy.authorize(request, Operation.READ_TERM)

        term = await app.glossary.read(term_id=term_id)

        return TermDTO(
            id=term.id,
            name=term.name,
            description=term.description,
            synonyms=term.synonyms,
            groups=term.groups,
            modified_utc=term.modified_utc,
        )

    @router.get(
        "",
        operation_id="list_terms",
        response_model=Sequence[TermDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all terms in the glossary.",
                "content": common.example_json_content([term_example]),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list_terms"),
    )
    async def list_terms(
        request: Request,
        group_id: GroupIdQuery = None,
    ) -> Sequence[TermDTO]:
        """
        Retrieves a list of all terms in the glossary.

        Returns an empty list if no terms exist.
        Terms are returned in no guaranteed order.
        """
        await authorization_policy.authorize(request, Operation.LIST_TERMS)

        terms = await app.glossary.find(group_id)

        return [
            TermDTO(
                id=term.id,
                name=term.name,
                description=term.description,
                synonyms=term.synonyms,
                groups=term.groups,
                modified_utc=term.modified_utc,
            )
            for term in terms
        ]

    @router.patch(
        "/{term_id}",
        operation_id="update_term",
        response_model=TermDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Term successfully updated. Returns the updated term object",
                "content": common.example_json_content(term_update_params_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Term not found. The specified `term_id` does not exist"
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in update parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update_term"),
    )
    async def update_term(
        request: Request,
        term_id: TermIdPath,
        params: TermUpdateParamsDTO,
    ) -> TermDTO:
        """
        Updates an existing term's attributes in the glossary.

        Only the provided attributes will be updated; others will remain unchanged.
        The term's ID and creation timestamp cannot be modified.
        """
        await authorization_policy.authorize(request, Operation.UPDATE_TERM)

        term = await app.glossary.update(
            term_id=term_id,
            name=params.name,
            description=params.description,
            synonyms=params.synonyms,
            groups=TermGroupsUpdateParamsModel(
                add=params.groups.add,
                remove=params.groups.remove,
            )
            if params.groups
            else None,
        )

        return TermDTO(
            id=term.id,
            name=term.name,
            description=term.description,
            synonyms=term.synonyms,
            groups=term.groups,
            modified_utc=term.modified_utc,
        )

    @router.delete(
        "/{term_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_term",
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Term successfully deleted. No content returned"
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Term not found. The specified `term_id` does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="delete_term"),
    )
    async def delete_term(
        request: Request,
        term_id: TermIdPath,
    ) -> None:
        """
        Deletes a term from the glossary.

        Deleting a non-existent term will return 404.
        No content will be returned from a successful deletion.
        """
        await authorization_policy.authorize(request, Operation.DELETE_TERM)

        await app.glossary.delete(term_id=term_id)

    return router
