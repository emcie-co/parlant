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
    RuleDTO,
    RuleIdField,
    RelationshipDTO,
    RelationshipKindDTO,
    GroupDTO,
    GroupIdField,
    ToolIdDTO,
    apigen_config,
    effort_to_effort_dto,
    tool_to_dto,
)
from parlant.core.app_modules.relationships import RelationshipModel
from parlant.core.application import Application
from parlant.core.common import DefaultBaseModel
from parlant.core.relationships import (
    RelationshipKind,
    RelationshipId,
)
from parlant.core.rules import RuleId
from parlant.core.groups import GroupId
from parlant.api.common import relationship_example
from parlant.core.tools import ToolId

API_GROUP = "relationships"


relationship_creation_params_example: ExampleJson = {
    "source_rule": "gid_123",
    "target_group": "tid_456",
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
    source_rule: RuleIdField | None = None
    source_group: GroupIdField | None = None
    source_tool: ToolIdDTO | None = None
    target_rule: RuleIdField | None = None
    target_group: GroupIdField | None = None
    target_tool: ToolIdDTO | None = None
    kind: RelationshipKindDTO
    group_id: str | None = None


RuleIdQuery: TypeAlias = Annotated[
    RuleId,
    Query(description="The ID of the rule to list relationships for"),
]


GroupIdQuery: TypeAlias = Annotated[
    GroupId,
    Query(description="The ID of the group to list relationships for"),
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
        case RelationshipKind.DEPENDENCY_ANY:
            return RelationshipKindDTO.DEPENDENCY_ANY
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
        case RelationshipKindDTO.DEPENDENCY_ANY:
            return RelationshipKind.DEPENDENCY_ANY
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
            source_rule=RuleDTO(
                id=model.source_rule.id,
                condition=model.source_rule.content.condition,
                action=model.source_rule.content.action,
                enabled=model.source_rule.enabled,
                groups=model.source_rule.groups,
                metadata=model.source_rule.metadata,
                modified_utc=model.source_rule.modified_utc,
                effort=effort_to_effort_dto(model.source_rule.effort_lift)
                if model.source_rule.effort_lift
                else None,
                priority=model.source_rule.priority,
            )
            if model.source_rule
            else None,
            source_group=GroupDTO(
                id=model.source_group.id,
                name=model.source_group.name,
            )
            if model.source_group
            else None,
            target_rule=RuleDTO(
                id=model.target_rule.id,
                condition=model.target_rule.content.condition,
                action=model.target_rule.content.action,
                enabled=model.target_rule.enabled,
                groups=model.target_rule.groups,
                metadata=model.target_rule.metadata,
                modified_utc=model.target_rule.modified_utc,
                effort=effort_to_effort_dto(model.target_rule.effort_lift)
                if model.target_rule.effort_lift
                else None,
                priority=model.target_rule.priority,
            )
            if model.target_rule
            else None,
            target_group=GroupDTO(
                id=model.target_group.id,
                name=model.target_group.name,
            )
            if model.target_group
            else None,
            source_tool=tool_to_dto(model.source_tool) if model.source_tool else None,
            target_tool=tool_to_dto(model.target_tool) if model.target_tool else None,
            kind=_relationship_kind_to_dto(model.kind),
            group_id=model.group_id,
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

        A relationship is a relationship between a rule and a group.
        It can be created between a rule and a group, or between two rules, or between two groups.
        """
        await authorization_policy.authorize(
            request=request, operation=Operation.CREATE_RELATIONSHIP
        )

        if params.source_rule and params.source_group:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="A relationship cannot have both a source rule and a source group",
            )
        elif params.target_rule and params.target_group:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="A relationship cannot have both a target rule and a target group",
            )
        elif (
            params.source_rule and params.target_rule and params.source_rule == params.target_rule
        ) or (
            params.source_group
            and params.target_group
            and params.source_group == params.target_group
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="source and target cannot be the same entity",
            )

        model = await app.relationships.create(
            source_rule=params.source_rule,
            source_group=params.source_group,
            source_tool=ToolId(params.source_tool.service_name, params.source_tool.tool_name)
            if params.source_tool
            else None,
            target_rule=params.target_rule,
            target_group=params.target_group,
            target_tool=ToolId(params.target_tool.service_name, params.target_tool.tool_name)
            if params.target_tool
            else None,
            kind=_relationship_kind_dto_to_kind(params.kind),
            group_id=params.group_id,
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
        rule_id: RuleIdQuery | None = None,
        group_id: GroupIdQuery | None = None,
        tool_id: ToolIdQuery | None = None,
    ) -> Sequence[RelationshipDTO]:
        """
        List relationships.

        Either `rule_id` or `group_id` or `tool_id` must be provided.
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
            rule_id=rule_id,
            group_id=group_id,
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
