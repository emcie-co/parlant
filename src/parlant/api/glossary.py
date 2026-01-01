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
from parlant.api.common import apigen_config, ExampleJson
from parlant.core.app_modules.glossary import TermTagsUpdateParamsModel
from parlant.core.agents import AgentId
from parlant.core.application import Application
from parlant.core.common import DefaultBaseModel
from parlant.core.glossary import TermId
from parlant.core.tags import TagId

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

TermTagsField: TypeAlias = Annotated[
    list[TagId],
    Field(
        description="List of tag IDs associated with the term",
        examples=[["tag1", "tag2"]],
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
    tags: TermTagsField | None = None
    id: TermId | None = None


term_example: ExampleJson = {
    "id": "term-eth01",
    "name": "Gas",
    "description": "A unit in Ethereum that measures the computational effort to execute transactions or smart contracts",
    "synonyms": ["Transaction Fee", "Blockchain Fuel"],
    "tags": ["tag1", "tag2"],
}

term_update_params_example: ExampleJson = {
    "name": "Gas",
    "description": "A unit in Ethereum that measures the computational effort to execute transactions or smart contracts",
    "synonyms": ["Transaction Fee", "Blockchain Fuel"],
    "tags": {
        "add": ["tag1", "tag2"],
        "remove": ["tag3", "tag4"],
    },
}

term_tags_update_params_example: ExampleJson = {
    "add": [
        "t9a8g703f4",
        "tag_456abc",
    ],
    "remove": [
        "tag_789def",
        "tag_012ghi",
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
    tags: TermTagsField


TermTagsUpdateAddField: TypeAlias = Annotated[
    list[TagId],
    Field(
        description="List of tag IDs to add to the term",
        examples=[["tag1", "tag2"]],
    ),
]

TermTagsUpdateRemoveField: TypeAlias = Annotated[
    list[TagId],
    Field(
        description="List of tag IDs to remove from the term",
        examples=[["tag1", "tag2"]],
    ),
]


class TermTagsUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": term_tags_update_params_example},
):
    """
    Parameters for updating the tags of an existing glossary term.
    """

    add: TermTagsUpdateAddField | None = None
    remove: TermTagsUpdateRemoveField | None = None


class TermUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": term_update_params_example},
):
    """
    Parameters for updating an existing glossary term including tags.

    All fields are optional. Only the provided fields will be updated.
    """

    name: TermNameField | None = None
    description: TermDescriptionField | None = None
    synonyms: TermSynonymsField | None = None
    tags: TermTagsUpdateParamsDTO | None = None


TagIdQuery: TypeAlias = Annotated[
    TagId | None,
    Query(
        description="Filter terms by tag ID",
        examples=["tag1", "tag2"],
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
                tags=params.tags,
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
            tags=term.tags,
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
            tags=term.tags,
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
        tag_id: TagIdQuery = None,
    ) -> Sequence[TermDTO]:
        """
        Retrieves a list of all terms in the glossary.

        Returns an empty list if no terms exist.
        Terms are returned in no guaranteed order.
        """
        await authorization_policy.authorize(request, Operation.LIST_TERMS)

        terms = await app.glossary.find(tag_id)

        return [
            TermDTO(
                id=term.id,
                name=term.name,
                description=term.description,
                synonyms=term.synonyms,
                tags=term.tags,
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
            tags=TermTagsUpdateParamsModel(
                add=params.tags.add,
                remove=params.tags.remove,
            )
            if params.tags
            else None,
        )

        return TermDTO(
            id=term.id,
            name=term.name,
            description=term.description,
            synonyms=term.synonyms,
            tags=term.tags,
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
