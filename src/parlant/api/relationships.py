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

from typing import Sequence, Annotated, TypeAlias
from fastapi import APIRouter, HTTPException, Path, Query, Request, status

from parlant.api import common
from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import (
    ExampleJson,
    GuidelineDTO,
    GuidelineIdField,
    RelationshipDTO,
    RelationshipKindDTO,
    TagDTO,
    TagIdField,
    ToolIdDTO,
    apigen_config,
    tool_to_dto,
)
from parlant.core.app_modules.relationships import RelationshipModel
from parlant.core.application import Application
from parlant.core.common import DefaultBaseModel
from parlant.core.relationships import (
    RelationshipKind,
    RelationshipId,
)
from parlant.core.guidelines import GuidelineId
from parlant.core.tags import TagId
from parlant.api.common import relationship_example
from parlant.core.tools import ToolId

API_GROUP = "relationships"


relationship_creation_params_example: ExampleJson = {
    "source_guideline": "gid_123",
    "target_tag": "tid_456",
    "kind": "entailment",
}


relationship_creation_tool_example: ExampleJson = {
    "source_tool": {
        "service_name": "tool_service_name",
        "tool_name": "tool_name",
    },
    "target_tool": {
        "service_name": "tool_service_name",
        "tool_name": "tool_name",
    },
    "kind": "overlap",
}


class RelationshipCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={
        "example": relationship_creation_params_example,
        "tool_example": relationship_creation_tool_example,
    },
):
    source_guideline: GuidelineIdField | None = None
    source_tag: TagIdField | None = None
    source_tool: ToolIdDTO | None = None
    target_guideline: GuidelineIdField | None = None
    target_tag: TagIdField | None = None
    target_tool: ToolIdDTO | None = None
    kind: RelationshipKindDTO


GuidelineIdQuery: TypeAlias = Annotated[
    GuidelineId,
    Query(description="The ID of the guideline to list relationships for"),
]


TagIdQuery: TypeAlias = Annotated[
    TagId,
    Query(description="The ID of the tag to list relationships for"),
]


ToolIdQuery: TypeAlias = Annotated[
    str,
    Query(
        description="The ID of the tool to list relationships for. Format: service_name:tool_name"
    ),
]


IndirectQuery: TypeAlias = Annotated[
    bool,
    Query(description="Whether to include indirect relationships"),
]


RelationshipKindQuery: TypeAlias = Annotated[
    RelationshipKindDTO,
    Query(description="The kind of relationship to list"),
]


RelationshipIdPath: TypeAlias = Annotated[
    RelationshipId,
    Path(
        description="identifier of relationship",
        examples=[RelationshipId("gr_123")],
    ),
]


def _relationship_kind_to_dto(
    kind: RelationshipKind,
) -> RelationshipKindDTO:
    match kind:
        case RelationshipKind.ENTAILMENT:
            return RelationshipKindDTO.ENTAILMENT
        case RelationshipKind.PRIORITY:
            return RelationshipKindDTO.PRIORITY
        case RelationshipKind.DEPENDENCY:
            return RelationshipKindDTO.DEPENDENCY
        case RelationshipKind.DISAMBIGUATION:
            return RelationshipKindDTO.DISAMBIGUATION
        case RelationshipKind.REEVALUATION:
            return RelationshipKindDTO.REEVALUATION
        case RelationshipKind.OVERLAP:
            return RelationshipKindDTO.OVERLAP
        case _:
            raise ValueError(f"Invalid relationship kind: {kind.value}")


def _relationship_kind_dto_to_kind(
    dto: RelationshipKindDTO,
) -> RelationshipKind:
    match dto:
        case RelationshipKindDTO.ENTAILMENT:
            return RelationshipKind.ENTAILMENT
        case RelationshipKindDTO.PRIORITY:
            return RelationshipKind.PRIORITY
        case RelationshipKindDTO.DEPENDENCY:
            return RelationshipKind.DEPENDENCY
        case RelationshipKindDTO.DISAMBIGUATION:
            return RelationshipKind.DISAMBIGUATION
        case RelationshipKindDTO.REEVALUATION:
            return RelationshipKind.REEVALUATION
        case RelationshipKindDTO.OVERLAP:
            return RelationshipKind.OVERLAP
        case _:
            raise ValueError(f"Invalid relationship kind: {dto.value}")


def create_router(
    authorization_policy: AuthorizationPolicy,
    app: Application,
) -> APIRouter:
    def model_to_dto(
        model: RelationshipModel,
    ) -> RelationshipDTO:
        return RelationshipDTO(
            id=model.id,
            source_guideline=GuidelineDTO(
                id=model.source_guideline.id,
                condition=model.source_guideline.content.condition,
                action=model.source_guideline.content.action,
                enabled=model.source_guideline.enabled,
                tags=model.source_guideline.tags,
                metadata=model.source_guideline.metadata,
            )
            if model.source_guideline
            else None,
            source_tag=TagDTO(
                id=model.source_tag.id,
                name=model.source_tag.name,
            )
            if model.source_tag
            else None,
            target_guideline=GuidelineDTO(
                id=model.target_guideline.id,
                condition=model.target_guideline.content.condition,
                action=model.target_guideline.content.action,
                enabled=model.target_guideline.enabled,
                tags=model.target_guideline.tags,
                metadata=model.target_guideline.metadata,
            )
            if model.target_guideline
            else None,
            target_tag=TagDTO(
                id=model.target_tag.id,
                name=model.target_tag.name,
            )
            if model.target_tag
            else None,
            source_tool=tool_to_dto(model.source_tool) if model.source_tool else None,
            target_tool=tool_to_dto(model.target_tool) if model.target_tool else None,
            kind=_relationship_kind_to_dto(model.kind),
        )

    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_relationship",
        response_model=RelationshipDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Relationship successfully created. Returns the created relationship.",
                "content": common.example_json_content(relationship_example),
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_relationship(
        request: Request,
        params: RelationshipCreationParamsDTO,
    ) -> RelationshipDTO:
        """
        Create a relationship.

        A relationship is a relationship between a guideline and a tag.
        It can be created between a guideline and a tag, or between two guidelines, or between two tags.
        """
        await authorization_policy.authorize(
            request=request, operation=Operation.CREATE_RELATIONSHIP
        )

        if params.source_guideline and params.source_tag:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="A relationship cannot have both a source guideline and a source tag",
            )
        elif params.target_guideline and params.target_tag:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="A relationship cannot have both a target guideline and a target tag",
            )
        elif (
            params.source_guideline
            and params.target_guideline
            and params.source_guideline == params.target_guideline
        ) or (params.source_tag and params.target_tag and params.source_tag == params.target_tag):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="source and target cannot be the same entity",
            )

        model = await app.relationships.create(
            source_guideline=params.source_guideline,
            source_tag=params.source_tag,
            source_tool=ToolId(params.source_tool.service_name, params.source_tool.tool_name)
            if params.source_tool
            else None,
            target_guideline=params.target_guideline,
            target_tag=params.target_tag,
            target_tool=ToolId(params.target_tool.service_name, params.target_tool.tool_name)
            if params.target_tool
            else None,
            kind=_relationship_kind_dto_to_kind(params.kind),
        )

        return model_to_dto(model=model)

    @router.get(
        "",
        operation_id="list_relationships",
        response_model=Sequence[RelationshipDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "Relationships successfully retrieved. Returns a list of all relationships.",
                "content": common.example_json_content([relationship_example]),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_relationships(
        request: Request,
        kind: RelationshipKindQuery | None = None,
        indirect: IndirectQuery = True,
        guideline_id: GuidelineIdQuery | None = None,
        tag_id: TagIdQuery | None = None,
        tool_id: ToolIdQuery | None = None,
    ) -> Sequence[RelationshipDTO]:
        """
        List relationships.

        Either `guideline_id` or `tag_id` or `tool_id` must be provided.
        """
        await authorization_policy.authorize(
            request=request, operation=Operation.LIST_RELATIONSHIPS
        )

        if tool_id:
            service_name, tool_name = tool_id.split(":")
            t_id = ToolId(service_name=service_name, tool_name=tool_name)
        else:
            t_id = None

        models = await app.relationships.find(
            kind=_relationship_kind_dto_to_kind(kind) if kind else None,
            indirect=indirect,
            guideline_id=guideline_id,
            tag_id=tag_id,
            tool_id=t_id,
        )

        return [model_to_dto(model=model) for model in models]

    @router.get(
        "/{relationship_id}",
        operation_id="read_relationship",
        status_code=status.HTTP_200_OK,
        response_model=RelationshipDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Relationship successfully retrieved. Returns the requested relationship.",
                "content": common.example_json_content(relationship_example),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_relationship(
        request: Request,
        relationship_id: RelationshipIdPath,
    ) -> RelationshipDTO:
        """
        Read a relationship by ID.
        """
        await authorization_policy.authorize(request=request, operation=Operation.READ_RELATIONSHIP)

        model = await app.relationships.read(relationship_id=relationship_id)

        return model_to_dto(model=model)

    @router.delete(
        "/{relationship_id}",
        operation_id="delete_relationship",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "Relationship successfully deleted."},
            status.HTTP_404_NOT_FOUND: {"description": "Relationship not found."},
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_relationship(
        request: Request,
        relationship_id: RelationshipIdPath,
    ) -> None:
        """
        Delete a relationship by ID.
        """
        await authorization_policy.authorize(
            request=request, operation=Operation.DELETE_RELATIONSHIP
        )

        await app.relationships.delete(relationship_id=relationship_id)

    return router
