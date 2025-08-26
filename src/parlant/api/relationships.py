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

from itertools import chain
from typing import Optional, Sequence, cast, Annotated, TypeAlias
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
from parlant.core.agents import AgentId, AgentStore
from parlant.core.common import DefaultBaseModel
from parlant.core.journeys import JourneyId, JourneyNodeId, JourneyStore
from parlant.core.relationships import (
    RelationshipEntityId,
    RelationshipEntityKind,
    RelationshipKind,
    Relationship,
    RelationshipEntity,
    RelationshipId,
    RelationshipStore,
)
from parlant.core.guidelines import Guideline, GuidelineId, GuidelineStore
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tags import Tag, TagId, TagStore
from parlant.api.common import relationship_example
from parlant.core.tools import Tool, ToolId

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
    source_guideline: Optional[GuidelineIdField] = None
    source_tag: Optional[TagIdField] = None
    source_tool: Optional[ToolIdDTO] = None
    target_guideline: Optional[GuidelineIdField] = None
    target_tag: Optional[TagIdField] = None
    target_tool: Optional[ToolIdDTO] = None
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


def _get_relationship_entity(
    guideline_id: Optional[GuidelineId],
    tag_id: Optional[TagId],
    tool_id: Optional[ToolId],
) -> RelationshipEntity:
    if guideline_id:
        return RelationshipEntity(id=guideline_id, kind=RelationshipEntityKind.GUIDELINE)
    elif tag_id:
        return RelationshipEntity(id=tag_id, kind=RelationshipEntityKind.TAG)
    elif tool_id:
        return RelationshipEntity(id=tool_id, kind=RelationshipEntityKind.TOOL)
    else:
        raise ValueError("No entity provided")


async def _entity_id_to_tag(
    tag_store: TagStore,
    agent_store: AgentStore,
    journey_store: JourneyStore,
    tag_id: RelationshipEntityId | TagId | GuidelineId | ToolId,
) -> Tag:
    tag_id = cast(TagId, tag_id)

    if agent_id := Tag.extract_agent_id(tag_id):
        agent = await agent_store.read_agent(agent_id=cast(AgentId, agent_id))
        return Tag(
            id=tag_id,
            name=agent.name,
            creation_utc=agent.creation_utc,
        )
    elif journey_id := Tag.extract_journey_id(tag_id):
        journey = await journey_store.read_journey(journey_id=cast(JourneyId, journey_id))
        return Tag(
            id=tag_id,
            name=journey.title,
            creation_utc=journey.creation_utc,
        )
    elif journey_node_id := Tag.extract_journey_node_id(tag_id):
        journey_node = await journey_store.read_node(node_id=cast(JourneyNodeId, journey_node_id))
        return Tag(
            id=tag_id,
            name=str(journey_node.action),
            creation_utc=journey_node.creation_utc,
        )
    else:
        return await tag_store.read_tag(tag_id=tag_id)


def create_router(
    authorization_policy: AuthorizationPolicy,
    guideline_store: GuidelineStore,
    tag_store: TagStore,
    agent_store: AgentStore,
    journey_store: JourneyStore,
    relationship_store: RelationshipStore,
    service_registry: ServiceRegistry,
) -> APIRouter:
    async def relationship_to_dto(
        relationship: Relationship,
    ) -> RelationshipDTO:
        source_guideline = (
            await guideline_store.read_guideline(
                guideline_id=cast(GuidelineId, relationship.source.id)
            )
            if relationship.source.kind == RelationshipEntityKind.GUIDELINE
            else None
        )

        source_tag = (
            await _entity_id_to_tag(
                tag_store,
                agent_store,
                journey_store,
                relationship.source.id,
            )
            if relationship.source.kind == RelationshipEntityKind.TAG
            else None
        )

        target_guideline = (
            await guideline_store.read_guideline(
                guideline_id=cast(GuidelineId, relationship.target.id)
            )
            if relationship.target.kind == RelationshipEntityKind.GUIDELINE
            else None
        )

        target_tag = (
            await _entity_id_to_tag(
                tag_store,
                agent_store,
                journey_store,
                relationship.target.id,
            )
            if relationship.target.kind == RelationshipEntityKind.TAG
            else None
        )

        source_tool = (
            await (
                await service_registry.read_tool_service(
                    name=cast(ToolId, relationship.source.id).service_name
                )
            ).read_tool(name=cast(ToolId, relationship.source.id).tool_name)
            if relationship.source.kind == RelationshipEntityKind.TOOL
            else None
        )

        target_tool = (
            await (
                await service_registry.read_tool_service(
                    name=cast(ToolId, relationship.target.id).service_name
                )
            ).read_tool(name=cast(ToolId, relationship.target.id).tool_name)
            if relationship.target.kind == RelationshipEntityKind.TOOL
            else None
        )

        return RelationshipDTO(
            id=relationship.id,
            source_guideline=GuidelineDTO(
                id=cast(Guideline, source_guideline).id,
                condition=cast(Guideline, source_guideline).content.condition,
                action=cast(Guideline, source_guideline).content.action,
                enabled=cast(Guideline, source_guideline).enabled,
                tags=cast(Guideline, source_guideline).tags,
                metadata=cast(Guideline, source_guideline).metadata,
            )
            if relationship.source.kind == RelationshipEntityKind.GUIDELINE
            else None,
            source_tag=TagDTO(
                id=cast(Tag, source_tag).id,
                name=cast(Tag, source_tag).name,
            )
            if relationship.source.kind == RelationshipEntityKind.TAG
            else None,
            target_guideline=GuidelineDTO(
                id=cast(Guideline, target_guideline).id,
                condition=cast(Guideline, target_guideline).content.condition,
                action=cast(Guideline, target_guideline).content.action,
                enabled=cast(Guideline, target_guideline).enabled,
                tags=cast(Guideline, target_guideline).tags,
                metadata=cast(Guideline, target_guideline).metadata,
            )
            if relationship.target.kind == RelationshipEntityKind.GUIDELINE
            else None,
            target_tag=TagDTO(
                id=cast(Tag, target_tag).id,
                name=cast(Tag, target_tag).name,
            )
            if relationship.target.kind == RelationshipEntityKind.TAG
            else None,
            source_tool=tool_to_dto(cast(Tool, source_tool))
            if relationship.source.kind == RelationshipEntityKind.TOOL
            else None,
            target_tool=tool_to_dto(cast(Tool, target_tool))
            if relationship.target.kind == RelationshipEntityKind.TOOL
            else None,
            kind=_relationship_kind_to_dto(relationship.kind),
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
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
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
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="A relationship cannot have both a source guideline and a source tag",
            )
        elif params.target_guideline and params.target_tag:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="A relationship cannot have both a target guideline and a target tag",
            )
        elif (
            params.source_guideline
            and params.target_guideline
            and params.source_guideline == params.target_guideline
        ) or (params.source_tag and params.target_tag and params.source_tag == params.target_tag):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="source and target cannot be the same entity",
            )

        if params.source_guideline:
            await guideline_store.read_guideline(guideline_id=params.source_guideline)
        elif params.source_tag:
            await _entity_id_to_tag(
                tag_store,
                agent_store,
                journey_store,
                params.source_tag,
            )
        elif params.source_tool:
            service = await service_registry.read_tool_service(name=params.source_tool.service_name)
            _ = await service.read_tool(name=params.source_tool.tool_name)

        if params.target_guideline:
            await guideline_store.read_guideline(guideline_id=params.target_guideline)
        elif params.target_tag:
            await _entity_id_to_tag(
                tag_store,
                agent_store,
                journey_store,
                params.target_tag,
            )
        elif params.target_tool:
            service = await service_registry.read_tool_service(name=params.target_tool.service_name)
            _ = await service.read_tool(name=params.target_tool.tool_name)

        relationship = await relationship_store.create_relationship(
            source=_get_relationship_entity(
                params.source_guideline,
                params.source_tag,
                ToolId(params.source_tool.service_name, params.source_tool.tool_name)
                if params.source_tool
                else None,
            ),
            target=_get_relationship_entity(
                params.target_guideline,
                params.target_tag,
                ToolId(params.target_tool.service_name, params.target_tool.tool_name)
                if params.target_tool
                else None,
            ),
            kind=_relationship_kind_dto_to_kind(params.kind),
        )

        return await relationship_to_dto(relationship=relationship)

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
        kind: Optional[RelationshipKindQuery] = None,
        indirect: IndirectQuery = True,
        guideline_id: Optional[GuidelineIdQuery] = None,
        tag_id: Optional[TagIdQuery] = None,
        tool_id: Optional[ToolIdQuery] = None,
    ) -> Sequence[RelationshipDTO]:
        """
        List relationships.

        Either `guideline_id` or `tag_id` or `tool_id` must be provided.
        """
        await authorization_policy.authorize(
            request=request, operation=Operation.LIST_RELATIONSHIPS
        )

        if not guideline_id and not tag_id and not tool_id:
            relationships = await relationship_store.list_relationships(
                kind=_relationship_kind_dto_to_kind(kind) if kind else None,
                indirect=indirect,
            )

            return [
                await relationship_to_dto(relationship=relationship)
                for relationship in relationships
            ]

        entity_id: GuidelineId | TagId | ToolId
        if guideline_id:
            await guideline_store.read_guideline(guideline_id=guideline_id)
            entity_id = guideline_id
        elif tag_id:
            await _entity_id_to_tag(
                tag_store,
                agent_store,
                journey_store,
                tag_id,
            )
            entity_id = tag_id
        elif tool_id:
            service_name, tool_name = tool_id.split(":")
            service = await service_registry.read_tool_service(name=service_name)
            _ = await service.read_tool(name=tool_name)
            entity_id = ToolId(service_name, tool_name)
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Either `guideline_id` or `tag_id` or `tool_id` must be provided",
            )

        source_relationships = await relationship_store.list_relationships(
            kind=_relationship_kind_dto_to_kind(kind) if kind else None,
            source_id=entity_id,
            indirect=indirect,
        )

        target_relationships = await relationship_store.list_relationships(
            kind=_relationship_kind_dto_to_kind(kind) if kind else None,
            target_id=entity_id,
            indirect=indirect,
        )

        relationships = list(chain(source_relationships, target_relationships))

        return [
            await relationship_to_dto(relationship=relationship) for relationship in relationships
        ]

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

        relationship = await relationship_store.read_relationship(id=relationship_id)

        return await relationship_to_dto(relationship=relationship)

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

        await relationship_store.delete_relationship(id=relationship_id)

    return router
