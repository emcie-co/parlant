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

from fastapi import APIRouter, HTTPException, Path, Request, status
from pydantic import Field
from typing import Annotated, Any, Sequence, TypeAlias

from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import (
    ExampleJson,
    apigen_config,
    example_json_content,
)
from parlant.core.app_modules.playbooks import (
    PlaybookTagUpdateParamsModel,
    PlaybookDisabledRulesUpdateParamsModel,
)
from parlant.core.playbooks import DisabledRuleRef, PlaybookId
from parlant.core.application import Application
from parlant.core.common import DefaultBaseModel
from parlant.core.tags import TagId

API_GROUP = "playbooks"

PlaybookIdPath: TypeAlias = Annotated[
    PlaybookId,
    Path(
        description="Unique identifier for the playbook",
        examples=["pb_abc123"],
        min_length=1,
    ),
]

PlaybookNameField: TypeAlias = Annotated[
    str,
    Field(
        description="The display name of the playbook",
        examples=["Base Support", "Healthcare Playbook"],
        min_length=1,
        max_length=100,
    ),
]

PlaybookDescriptionField: TypeAlias = Annotated[
    str,
    Field(
        description="Detailed description of the playbook's purpose",
        examples=["Common support rules for all agents"],
    ),
]

PlaybookParentIdField: TypeAlias = Annotated[
    str,
    Field(
        description="ID of the parent playbook for inheritance",
        examples=["pb_parent123"],
    ),
]

PlaybookTagsField: TypeAlias = Annotated[
    list[TagId],
    Field(
        description="List of tag IDs associated with the playbook",
        examples=[["tag1", "tag2"]],
    ),
]

PlaybookDisabledRulesField: TypeAlias = Annotated[
    list[str],
    Field(
        description="List of disabled rule references (e.g., 'guideline:abc123')",
        examples=[["guideline:abc123", "term:xyz789"]],
    ),
]

playbook_example: ExampleJson = {
    "id": "pb_abc123",
    "name": "Healthcare Support",
    "description": "HIPAA-compliant healthcare support rules",
    "parent_id": "pb_base",
    "disabled_rules": ["guideline:xyz123"],
    "tags": ["tag1", "tag2"],
    "creation_utc": "2024-03-24T12:00:00Z",
}


class PlaybookDTO(
    DefaultBaseModel,
    json_schema_extra={"example": playbook_example},
):
    """
    A playbook is a reusable collection of behavioral rules that can be
    assigned to agents and supports single inheritance.
    """

    id: PlaybookIdPath
    name: PlaybookNameField
    description: PlaybookDescriptionField | None = None
    parent_id: PlaybookParentIdField | None = None
    disabled_rules: PlaybookDisabledRulesField = []
    tags: PlaybookTagsField = []


playbook_creation_params_example: ExampleJson = {
    "name": "Healthcare Support",
    "description": "HIPAA-compliant healthcare support rules",
    "parent_id": "pb_base",
    "tags": ["tag1", "tag2"],
}


class PlaybookCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": playbook_creation_params_example},
):
    """
    Parameters for creating a new playbook.
    """

    name: PlaybookNameField
    id: PlaybookIdPath | None = None
    description: PlaybookDescriptionField | None = None
    parent_id: PlaybookParentIdField | None = None
    tags: PlaybookTagsField | None = None


playbook_update_params_example: ExampleJson = {
    "name": "Updated Healthcare Support",
    "description": "Updated description",
}


class PlaybookUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": playbook_update_params_example},
):
    """
    Parameters for updating an existing playbook.
    """

    name: PlaybookNameField | None = None
    description: PlaybookDescriptionField | None = None
    parent_id: PlaybookParentIdField | None = None


tags_update_params_example: ExampleJson = {
    "add": ["tag1", "tag2"],
    "remove": ["tag3"],
}


class PlaybookTagUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": tags_update_params_example},
):
    """
    Parameters for updating a playbook's tags.
    """

    add: list[TagId] | None = None
    remove: list[TagId] | None = None


disabled_rules_update_params_example: ExampleJson = {
    "add": ["guideline:abc123", "term:xyz789"],
    "remove": ["guideline:old123"],
}


class PlaybookDisabledRulesUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": disabled_rules_update_params_example},
):
    """
    Parameters for updating a playbook's disabled rules.
    """

    add: list[str] | None = None
    remove: list[str] | None = None


def create_router(
    authorization_policy: AuthorizationPolicy,
    app: Application,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_playbook",
        response_model=PlaybookDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Playbook successfully created.",
                "content": example_json_content(playbook_example),
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_playbook(
        request: Request,
        params: PlaybookCreationParamsDTO,
    ) -> PlaybookDTO:
        """
        Creates a new playbook.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.CREATE_PLAYBOOK,
        )

        playbook = await app.playbooks.create(
            name=params.name,
            description=params.description,
            parent_id=PlaybookId(params.parent_id) if params.parent_id else None,
            tags=params.tags,
            id=PlaybookId(params.id) if params.id else None,
        )

        return PlaybookDTO(
            id=playbook.id,
            name=playbook.name,
            description=playbook.description,
            parent_id=playbook.parent_id,
            disabled_rules=list(playbook.disabled_rules),
            tags=list(playbook.tags),
            creation_utc=playbook.creation_utc,
        )

    @router.get(
        "",
        operation_id="list_playbooks",
        response_model=Sequence[PlaybookDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all playbooks",
                "content": example_json_content([playbook_example]),
            }
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_playbooks(request: Request) -> Sequence[PlaybookDTO]:
        """
        Retrieves a list of all playbooks.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.LIST_PLAYBOOKS,
        )

        playbooks = await app.playbooks.find()

        return [
            PlaybookDTO(
                id=p.id,
                name=p.name,
                description=p.description,
                parent_id=p.parent_id,
                disabled_rules=list(p.disabled_rules),
                tags=list(p.tags),
                creation_utc=p.creation_utc,
            )
            for p in playbooks
        ]

    @router.get(
        "/{playbook_id}",
        operation_id="read_playbook",
        response_model=PlaybookDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Playbook details successfully retrieved.",
                "content": example_json_content(playbook_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Playbook not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_playbook(
        request: Request,
        playbook_id: PlaybookIdPath,
    ) -> PlaybookDTO:
        """
        Retrieves details of a specific playbook.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.READ_PLAYBOOK,
        )

        playbook = await app.playbooks.read(playbook_id=playbook_id)

        return PlaybookDTO(
            id=playbook.id,
            name=playbook.name,
            description=playbook.description,
            parent_id=playbook.parent_id,
            disabled_rules=list(playbook.disabled_rules),
            tags=list(playbook.tags),
            creation_utc=playbook.creation_utc,
        )

    @router.patch(
        "/{playbook_id}",
        operation_id="update_playbook",
        response_model=PlaybookDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Playbook successfully updated.",
                "content": example_json_content(playbook_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Playbook not found"},
            status.HTTP_400_BAD_REQUEST: {
                "description": "Invalid update (e.g., circular inheritance)"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_playbook(
        request: Request,
        playbook_id: PlaybookIdPath,
        params: PlaybookUpdateParamsDTO,
    ) -> PlaybookDTO:
        """
        Updates an existing playbook.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.UPDATE_PLAYBOOK,
        )

        try:
            update_kwargs: dict[str, Any] = {
                "playbook_id": playbook_id,
                "name": params.name,
                "description": params.description,
            }

            # Only pass parent_id if it was explicitly set (even if to None/null)
            if "parent_id" in params.model_fields_set:
                update_kwargs["parent_id"] = (
                    PlaybookId(params.parent_id) if params.parent_id else None
                )

            playbook = await app.playbooks.update(**update_kwargs)  # type: ignore[arg-type]
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

        return PlaybookDTO(
            id=playbook.id,
            name=playbook.name,
            description=playbook.description,
            parent_id=playbook.parent_id,
            disabled_rules=list(playbook.disabled_rules),
            tags=list(playbook.tags),
            creation_utc=playbook.creation_utc,
        )

    @router.delete(
        "/{playbook_id}",
        operation_id="delete_playbook",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "Playbook successfully deleted."},
            status.HTTP_404_NOT_FOUND: {"description": "Playbook not found"},
            status.HTTP_400_BAD_REQUEST: {"description": "Cannot delete playbook with children"},
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_playbook(
        request: Request,
        playbook_id: PlaybookIdPath,
    ) -> None:
        """
        Deletes a playbook.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.DELETE_PLAYBOOK,
        )

        try:
            await app.playbooks.delete(playbook_id=playbook_id)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

    @router.patch(
        "/{playbook_id}/tags",
        operation_id="update_playbook_tags",
        response_model=PlaybookDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Playbook tags successfully updated.",
                "content": example_json_content(playbook_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Playbook not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="update_tags"),
    )
    async def update_playbook_tags(
        request: Request,
        playbook_id: PlaybookIdPath,
        params: PlaybookTagUpdateParamsDTO,
    ) -> PlaybookDTO:
        """
        Updates a playbook's tags.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.UPDATE_PLAYBOOK,
        )

        playbook = await app.playbooks.update(
            playbook_id=playbook_id,
            tags=PlaybookTagUpdateParamsModel(
                add=params.add,
                remove=params.remove,
            ),
        )

        return PlaybookDTO(
            id=playbook.id,
            name=playbook.name,
            description=playbook.description,
            parent_id=playbook.parent_id,
            disabled_rules=list(playbook.disabled_rules),
            tags=list(playbook.tags),
            creation_utc=playbook.creation_utc,
        )

    @router.patch(
        "/{playbook_id}/disabled-rules",
        operation_id="update_playbook_disabled_rules",
        response_model=PlaybookDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Playbook disabled rules successfully updated.",
                "content": example_json_content(playbook_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Playbook not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="update_disabled_rules"),
    )
    async def update_playbook_disabled_rules(
        request: Request,
        playbook_id: PlaybookIdPath,
        params: PlaybookDisabledRulesUpdateParamsDTO,
    ) -> PlaybookDTO:
        """
        Updates a playbook's disabled rules.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.UPDATE_PLAYBOOK,
        )

        playbook = await app.playbooks.update(
            playbook_id=playbook_id,
            disabled_rules=PlaybookDisabledRulesUpdateParamsModel(
                add=[DisabledRuleRef(r) for r in params.add] if params.add else None,
                remove=[DisabledRuleRef(r) for r in params.remove] if params.remove else None,
            ),
        )

        return PlaybookDTO(
            id=playbook.id,
            name=playbook.name,
            description=playbook.description,
            parent_id=playbook.parent_id,
            disabled_rules=list(playbook.disabled_rules),
            tags=list(playbook.tags),
            creation_utc=playbook.creation_utc,
        )

    @router.get(
        "/{playbook_id}/inheritance",
        operation_id="get_playbook_inheritance",
        response_model=Sequence[PlaybookDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "Inheritance chain from root to this playbook.",
                "content": example_json_content([playbook_example]),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Playbook not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="get_inheritance"),
    )
    async def get_playbook_inheritance(
        request: Request,
        playbook_id: PlaybookIdPath,
    ) -> Sequence[PlaybookDTO]:
        """
        Retrieves the inheritance chain for a playbook (root to child).
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.READ_PLAYBOOK,
        )

        chain = await app.playbooks.get_inheritance_chain(playbook_id=playbook_id)

        return [
            PlaybookDTO(
                id=p.id,
                name=p.name,
                description=p.description,
                parent_id=p.parent_id,
                disabled_rules=list(p.disabled_rules),
                tags=list(p.tags),
                creation_utc=p.creation_utc,
            )
            for p in chain
        ]

    return router
