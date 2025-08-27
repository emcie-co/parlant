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

from dataclasses import dataclass
from itertools import chain
from typing import Annotated, Optional, Sequence, TypeAlias, cast
from fastapi import APIRouter, HTTPException, Path, Request, status, Query
from pydantic import Field

from parlant.api import common
from parlant.api.authorization import Operation, AuthorizationPolicy
from parlant.api.common import (
    GuidelineDTO,
    GuidelineEnabledField,
    GuidelineIdField,
    GuidelineMetadataField,
    RelationshipDTO,
    GuidelineTagsField,
    RelationshipKindDTO,
    TagDTO,
    ToolIdDTO,
    apigen_config,
    guideline_dto_example,
)
from parlant.core.agents import AgentStore, AgentId
from parlant.core.common import (
    DefaultBaseModel,
)
from parlant.api.common import (
    ExampleJson,
    GuidelineConditionField,
    GuidelineActionField,
)

from parlant.core.journeys import JourneyId, JourneyStore
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipId,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.guidelines import (
    Guideline,
    GuidelineId,
    GuidelineStore,
    GuidelineUpdateParams,
)
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociationId,
    GuidelineToolAssociationStore,
)
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tags import TagId, TagStore, Tag
from parlant.core.tools import Tool, ToolId

API_GROUP = "guidelines"


GuidelineIdPath: TypeAlias = Annotated[
    GuidelineId,
    Path(
        description="Unique identifier for the guideline",
        examples=["IUCGT-l4pS"],
    ),
]


GuidelineToolAssociationIdField: TypeAlias = Annotated[
    GuidelineToolAssociationId,
    Field(
        description="Unique identifier for the association between a tool and a guideline",
        examples=["guid_tool_1"],
    ),
]


guideline_tool_association_example: ExampleJson = {
    "id": "gta_101xyz",
    "guideline_id": "guid_123xz",
    "tool_id": {"service_name": "pricing_service", "tool_name": "get_prices"},
}


class GuidelineToolAssociationDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_tool_association_example},
):
    """
    Represents an association between a Guideline and a Tool, enabling automatic tool invocation
    when the Guideline's conditions are met.
    """

    id: GuidelineToolAssociationIdField
    guideline_id: GuidelineIdField
    tool_id: ToolIdDTO


GuidelineConnectionAdditionSourceField: TypeAlias = Annotated[
    GuidelineId,
    Field(description="`id` of guideline that is source of this connection."),
]

GuidelineConnectionAdditionTargetField: TypeAlias = Annotated[
    GuidelineId,
    Field(description="`id` of guideline that is target of this connection."),
]


guideline_connection_addition_example: ExampleJson = {
    "source": "guid_123xz",
    "target": "guid_789yz",
}


guideline_tool_association_update_params_example: ExampleJson = {
    "add": [{"service_name": "pricing_service", "tool_name": "get_prices"}],
    "remove": [{"service_name": "old_service", "tool_name": "old_tool"}],
}


class GuidelineToolAssociationUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_tool_association_update_params_example},
):
    """Parameters for adding/removing tool associations."""

    add: Optional[Sequence[ToolIdDTO]] = None
    remove: Optional[Sequence[ToolIdDTO]] = None


@dataclass
class _GuidelineRelationship:
    """Represents a relationship between a guideline and another entity (guideline, tag, or tool)."""

    id: RelationshipId
    source: Guideline | Tag | Tool
    source_type: RelationshipEntityKind
    target: Guideline | Tag | Tool
    target_type: RelationshipEntityKind
    kind: RelationshipKind


async def _get_guideline_relationships_by_kind(
    guideline_store: GuidelineStore,
    tag_store: TagStore,
    relationship_store: RelationshipStore,
    entity_id: GuidelineId | TagId,
    kind: RelationshipKind,
    include_indirect: bool = True,
) -> Sequence[tuple[_GuidelineRelationship, bool]]:
    async def _get_entity(
        entity_id: GuidelineId | TagId,
        entity_type: RelationshipEntityKind,
    ) -> Guideline | Tag:
        if entity_type == RelationshipEntityKind.GUIDELINE:
            return await guideline_store.read_guideline(guideline_id=cast(GuidelineId, entity_id))
        elif entity_type == RelationshipEntityKind.TAG:
            return await tag_store.read_tag(tag_id=cast(TagId, entity_id))
        else:
            raise ValueError(f"Unsupported entity type: {entity_type}")

    relationships = []

    for r in chain(
        await relationship_store.list_relationships(
            kind=kind,
            indirect=include_indirect,
            source_id=entity_id,
        ),
        await relationship_store.list_relationships(
            kind=kind,
            indirect=include_indirect,
            target_id=entity_id,
        ),
    ):
        assert r.source.kind in (RelationshipEntityKind.GUIDELINE, RelationshipEntityKind.TAG)
        assert r.target.kind in (RelationshipEntityKind.GUIDELINE, RelationshipEntityKind.TAG)
        assert type(r.kind) is RelationshipKind

        relationships.append(
            _GuidelineRelationship(
                id=r.id,
                source=await _get_entity(cast(GuidelineId | TagId, r.source.id), r.source.kind),
                source_type=r.source.kind,
                target=await _get_entity(cast(GuidelineId | TagId, r.target.id), r.target.kind),
                target_type=r.target.kind,
                kind=r.kind,
            )
        )

    return [
        (
            r,
            entity_id
            not in [cast(Guideline | Tag, r.source).id, cast(Guideline | Tag, r.target).id],
        )
        for r in relationships
    ]


def _guideline_relationship_kind_to_dto(
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
        case _:
            raise ValueError(f"Invalid guideline relationship kind: {kind.value}")


async def _get_relationships(
    guideline_store: GuidelineStore,
    tag_store: TagStore,
    relationship_store: RelationshipStore,
    guideline_id: GuidelineId,
    include_indirect: bool = True,
) -> Sequence[tuple[_GuidelineRelationship, bool]]:
    return list(
        chain.from_iterable(
            [
                await _get_guideline_relationships_by_kind(
                    guideline_store=guideline_store,
                    tag_store=tag_store,
                    relationship_store=relationship_store,
                    entity_id=guideline_id,
                    kind=kind,
                    include_indirect=include_indirect,
                )
                for kind in list(RelationshipKind)
            ]
        )
    )


TagIdQuery: TypeAlias = Annotated[
    Optional[TagId],
    Query(
        description="The tag ID to filter guidelines by",
        examples=["tag:123"],
    ),
]


GuidelineTagsUpdateAddField: TypeAlias = Annotated[
    list[TagId],
    Field(
        description="List of tag IDs to add to the guideline",
        examples=[["tag1", "tag2"]],
    ),
]

GuidelineTagsUpdateRemoveField: TypeAlias = Annotated[
    list[TagId],
    Field(
        description="List of tag IDs to remove from the guideline",
        examples=[["tag1", "tag2"]],
    ),
]

guideline_tags_update_params_example: ExampleJson = {
    "add": [
        "tag1",
        "tag2",
    ],
    "remove": [
        "tag3",
        "tag4",
    ],
}


class GuidelineTagsUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_tags_update_params_example},
):
    """
    Parameters for updating the tags of an existing guideline.
    """

    add: Optional[GuidelineTagsUpdateAddField] = None
    remove: Optional[GuidelineTagsUpdateRemoveField] = None


TagIdField: TypeAlias = Annotated[
    TagId,
    Field(
        description="Unique identifier for the tag",
        examples=["t9a8g703f4"],
    ),
]

TagNameField: TypeAlias = Annotated[
    str,
    Field(
        description="Name of the tag",
        examples=["tag1"],
    ),
]

guideline_creation_params_example: ExampleJson = {
    "condition": "when the customer asks about pricing",
    "action": "provide current pricing information and mention any ongoing promotions",
    "enabled": False,
    "metadata": {"key1": "value1", "key2": "value2"},
}


class GuidelineCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_creation_params_example},
):
    """Parameters for creating a new guideline."""

    condition: GuidelineConditionField
    action: Optional[GuidelineActionField] = None
    metadata: Optional[GuidelineMetadataField] = None
    enabled: Optional[GuidelineEnabledField] = None
    tags: Optional[GuidelineTagsField] = None


GuidelineMetadataUnsetField: TypeAlias = Annotated[
    Sequence[str],
    Field(description="Metadata keys to remove from the guideline"),
]

guideline_metadata_update_params_example: ExampleJson = {
    "add": {
        "key1": "value1",
        "key2": "value2",
    },
    "remove": ["key3", "key4"],
}


class GuidelineMetadataUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_metadata_update_params_example},
):
    """Parameters for updating the metadata of a guideline."""

    set: Optional[GuidelineMetadataField] = None
    unset: Optional[GuidelineMetadataUnsetField] = None


guideline_update_params_example: ExampleJson = {
    "condition": "when the customer asks about pricing",
    "action": "provide current pricing information",
    "enabled": True,
    "tags": ["tag1", "tag2"],
    "metadata": {
        "add": {
            "key1": "value1",
            "key2": "value2",
        },
        "remove": ["key3", "key4"],
    },
    "tool_associations": {
        "add": [
            {
                "service_name": "new_service",
                "tool_name": "new_tool",
            }
        ],
        "remove": [
            {
                "service_name": "old_service",
                "tool_name": "old_tool",
            },
        ],
    },
}


class GuidelineUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_update_params_example},
):
    """Parameters for updating a guideline."""

    condition: Optional[GuidelineConditionField] = None
    action: Optional[GuidelineActionField] = None
    tool_associations: Optional[GuidelineToolAssociationUpdateParamsDTO] = None
    enabled: Optional[GuidelineEnabledField] = None
    tags: Optional[GuidelineTagsUpdateParamsDTO] = None
    metadata: Optional[GuidelineMetadataUpdateParamsDTO] = None


guideline_with_relationships_example: ExampleJson = {
    "guideline": {
        "id": "guid_123xz",
        "condition": "when the customer asks about pricing",
        "action": "provide current pricing information",
        "enabled": True,
        "tags": ["tag1", "tag2"],
    },
    "relationships": [
        {
            "id": "123",
            "source_guideline": {
                "id": "guid_123xz",
                "condition": "when the customer asks about pricing",
                "action": "provide current pricing information",
                "enabled": True,
                "tags": ["tag1", "tag2"],
            },
            "target_tag": {
                "id": "tid_456yz",
                "name": "tag1",
            },
            "indirect": False,
            "kind": "entailment",
        }
    ],
    "tool_associations": [
        {
            "id": "gta_101xyz",
            "guideline_id": "guid_123xz",
            "tool_id": {"service_name": "pricing_service", "tool_name": "get_prices"},
        }
    ],
}


class GuidelineWithRelationshipsAndToolAssociationsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_with_relationships_example},
):
    """A Guideline with its relationships and tool associations."""

    guideline: GuidelineDTO
    relationships: Sequence[RelationshipDTO]
    tool_associations: Sequence[GuidelineToolAssociationDTO]


def _guideline_relationship_to_dto(
    relationship: _GuidelineRelationship,
    indirect: bool,
) -> RelationshipDTO:
    if relationship.source_type == RelationshipEntityKind.GUIDELINE:
        rel_source_guideline = cast(Guideline, relationship.source)
    else:
        rel_source_tag = cast(Tag, relationship.source)

    if relationship.target_type == RelationshipEntityKind.GUIDELINE:
        rel_target_guideline = cast(Guideline, relationship.target)
    else:
        rel_target_tag = cast(Tag, relationship.target)

    return RelationshipDTO(
        id=relationship.id,
        source_guideline=GuidelineDTO(
            id=rel_source_guideline.id,
            condition=rel_source_guideline.content.condition,
            action=rel_source_guideline.content.action,
            enabled=rel_source_guideline.enabled,
            tags=rel_source_guideline.tags,
            metadata=rel_source_guideline.metadata,
        )
        if relationship.source_type == RelationshipEntityKind.GUIDELINE
        else None,
        source_tag=TagDTO(
            id=rel_source_tag.id,
            creation_utc=rel_source_tag.creation_utc,
            name=rel_source_tag.name,
        )
        if relationship.source_type == RelationshipEntityKind.TAG
        else None,
        target_guideline=GuidelineDTO(
            id=cast(Guideline | Tag, relationship.target).id,
            creation_utc=rel_target_guideline.creation_utc,
            condition=rel_target_guideline.content.condition,
            action=rel_target_guideline.content.action,
            enabled=rel_target_guideline.enabled,
            tags=rel_target_guideline.tags,
            metadata=rel_target_guideline.metadata,
        )
        if relationship.target_type == RelationshipEntityKind.GUIDELINE
        else None,
        target_tag=TagDTO(
            id=rel_target_tag.id,
            name=rel_target_tag.name,
        )
        if relationship.target_type == RelationshipEntityKind.TAG
        else None,
        indirect=indirect,
        kind=_guideline_relationship_kind_to_dto(relationship.kind),
    )


def create_router(
    authorization_policy: AuthorizationPolicy,
    guideline_store: GuidelineStore,
    relationship_store: RelationshipStore,
    service_registry: ServiceRegistry,
    guideline_tool_association_store: GuidelineToolAssociationStore,
    agent_store: AgentStore,
    tag_store: TagStore,
    journey_store: JourneyStore,
) -> APIRouter:
    """Creates a router for the guidelines API with tag-based paths."""
    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_guideline",
        response_model=GuidelineDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Guideline successfully created. Returns the created guideline.",
                "content": common.example_json_content(guideline_dto_example),
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_guideline(
        request: Request,
        params: GuidelineCreationParamsDTO,
    ) -> GuidelineDTO:
        """
        Creates a new guideline.

        See the [documentation](https://parlant.io/docs/concepts/customization/guidelines) for more information.
        """
        await authorization_policy.authorize(request=request, operation=Operation.CREATE_GUIDELINE)

        tags = []
        if params.tags:
            for tag_id in params.tags:
                if agent_id := Tag.extract_agent_id(tag_id):
                    _ = await agent_store.read_agent(agent_id=AgentId(agent_id))
                elif journey_id := Tag.extract_journey_id(tag_id):
                    _ = await journey_store.read_journey(journey_id=JourneyId(journey_id))
                else:
                    _ = await tag_store.read_tag(tag_id=tag_id)

            tags = list(set(params.tags))

        guideline = await guideline_store.create_guideline(
            condition=params.condition,
            action=params.action or None,
            metadata=params.metadata or {},
            enabled=params.enabled or True,
            tags=tags or None,
        )

        return GuidelineDTO(
            id=guideline.id,
            condition=guideline.content.condition,
            action=guideline.content.action,
            metadata=guideline.metadata,
            enabled=guideline.enabled,
            tags=guideline.tags,
        )

    @router.get(
        "",
        operation_id="list_guidelines",
        response_model=Sequence[GuidelineDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all guidelines for the specified tag or all guidelines if no tag is provided",
                "content": common.example_json_content([guideline_dto_example]),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_guidelines(
        request: Request,
        tag_id: TagIdQuery = None,
    ) -> Sequence[GuidelineDTO]:
        """
        Lists all guidelines for the specified tag or all guidelines if no tag is provided.

        Returns an empty list if no guidelines exist.
        Guidelines are returned in no guaranteed order.
        Does not include relationships or tool associations.
        """
        await authorization_policy.authorize(request=request, operation=Operation.LIST_GUIDELINES)

        if tag_id:
            guidelines = await guideline_store.list_guidelines(
                tags=[tag_id],
            )
        else:
            guidelines = await guideline_store.list_guidelines()

        return [
            GuidelineDTO(
                id=guideline.id,
                condition=guideline.content.condition,
                action=guideline.content.action,
                metadata=guideline.metadata,
                enabled=guideline.enabled,
                tags=guideline.tags,
            )
            for guideline in guidelines
        ]

    @router.get(
        "/{guideline_id}",
        operation_id="read_guideline",
        response_model=GuidelineWithRelationshipsAndToolAssociationsDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Guideline details successfully retrieved. Returns the complete guideline with its relationships and tool associations.",
                "content": common.example_json_content(guideline_with_relationships_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Guideline not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_guideline(
        request: Request,
        guideline_id: GuidelineIdPath,
    ) -> GuidelineWithRelationshipsAndToolAssociationsDTO:
        """
        Retrieves a specific guideline with all its relationships and tool associations.

        Returns both direct and indirect relationships between guidelines.
        Tool associations indicate which tools the guideline can use.
        """
        await authorization_policy.authorize(request=request, operation=Operation.READ_GUIDELINE)

        try:
            guideline = await guideline_store.read_guideline(guideline_id=guideline_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Guideline not found",
            )

        relationships = await _get_relationships(
            guideline_store=guideline_store,
            tag_store=tag_store,
            relationship_store=relationship_store,
            guideline_id=guideline_id,
            include_indirect=True,
        )

        return GuidelineWithRelationshipsAndToolAssociationsDTO(
            guideline=GuidelineDTO(
                id=guideline.id,
                condition=guideline.content.condition,
                action=guideline.content.action,
                metadata=guideline.metadata,
                enabled=guideline.enabled,
                tags=guideline.tags,
            ),
            relationships=[
                _guideline_relationship_to_dto(relationship, indirect)
                for relationship, indirect in relationships
            ],
            tool_associations=[
                GuidelineToolAssociationDTO(
                    id=a.id,
                    guideline_id=a.guideline_id,
                    tool_id=ToolIdDTO(
                        service_name=a.tool_id.service_name,
                        tool_name=a.tool_id.tool_name,
                    ),
                )
                for a in await guideline_tool_association_store.list_associations()
                if a.guideline_id == guideline_id
            ],
        )

    @router.patch(
        "/{guideline_id}",
        operation_id="update_guideline",
        response_model=GuidelineWithRelationshipsAndToolAssociationsDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Guideline successfully updated. Returns the updated guideline with its relationships and tool associations.",
                "content": common.example_json_content(guideline_with_relationships_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Guideline or referenced tool not found"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Invalid relationship rules or validation error in update parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_guideline(
        request: Request,
        guideline_id: GuidelineIdPath,
        params: GuidelineUpdateParamsDTO,
    ) -> GuidelineWithRelationshipsAndToolAssociationsDTO:
        """Updates a guideline's relationships and tool associations.

        Only provided attributes will be updated; others remain unchanged.

        Relationship rules:
        - A guideline cannot relate to itself
        - Only direct relationships can be removed
        - The relationship must specify this guideline as source or target

        Tool Association rules:
        - Tool services and tools must exist before creating associations

        Action with text can not be updated to None.
        """
        await authorization_policy.authorize(request=request, operation=Operation.UPDATE_GUIDELINE)

        _ = await guideline_store.read_guideline(guideline_id=guideline_id)

        if params.condition or params.action or params.enabled is not None:
            update_params: GuidelineUpdateParams = {}
            if params.condition:
                update_params["condition"] = params.condition
            if params.action:
                update_params["action"] = params.action
            if params.enabled is not None:
                update_params["enabled"] = params.enabled

            await guideline_store.update_guideline(
                guideline_id=guideline_id,
                params=GuidelineUpdateParams(**update_params),
            )

        if params.metadata:
            if params.metadata.set:
                for key, value in params.metadata.set.items():
                    await guideline_store.set_metadata(
                        guideline_id=guideline_id,
                        key=key,
                        value=value,
                    )

            if params.metadata.unset:
                for key in params.metadata.unset:
                    await guideline_store.unset_metadata(
                        guideline_id=guideline_id,
                        key=key,
                    )

        if params.tool_associations and params.tool_associations.add:
            for tool_id_dto in params.tool_associations.add:
                service_name = tool_id_dto.service_name
                tool_name = tool_id_dto.tool_name

                try:
                    service = await service_registry.read_tool_service(service_name)
                    _ = await service.read_tool(tool_name)
                except Exception:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Tool not found (service='{service_name}', tool='{tool_name}')",
                    )

                await guideline_tool_association_store.create_association(
                    guideline_id=guideline_id,
                    tool_id=ToolId(service_name=service_name, tool_name=tool_name),
                )

        if params.tool_associations and params.tool_associations.remove:
            associations = await guideline_tool_association_store.list_associations()

            for tool_id_dto in params.tool_associations.remove:
                if association := next(
                    (
                        assoc
                        for assoc in associations
                        if assoc.tool_id.service_name == tool_id_dto.service_name
                        and assoc.tool_id.tool_name == tool_id_dto.tool_name
                        and assoc.guideline_id == guideline_id
                    ),
                    None,
                ):
                    await guideline_tool_association_store.delete_association(association.id)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Tool association not found for service '{tool_id_dto.service_name}' and tool '{tool_id_dto.tool_name}'",
                    )

        if params.tags:
            if params.tags.add:
                for tag_id in params.tags.add:
                    if agent_id := Tag.extract_agent_id(tag_id):
                        _ = await agent_store.read_agent(agent_id=AgentId(agent_id))
                    elif journey_id := Tag.extract_journey_id(tag_id):
                        _ = await journey_store.read_journey(journey_id=JourneyId(journey_id))
                    else:
                        _ = await tag_store.read_tag(tag_id=tag_id)

                    await guideline_store.upsert_tag(
                        guideline_id=guideline_id,
                        tag_id=tag_id,
                    )

            if params.tags.remove:
                for tag_id in params.tags.remove:
                    await guideline_store.remove_tag(
                        guideline_id=guideline_id,
                        tag_id=tag_id,
                    )

        updated_guideline = await guideline_store.read_guideline(guideline_id=guideline_id)

        return GuidelineWithRelationshipsAndToolAssociationsDTO(
            guideline=GuidelineDTO(
                id=updated_guideline.id,
                condition=updated_guideline.content.condition,
                action=updated_guideline.content.action,
                metadata=updated_guideline.metadata,
                enabled=updated_guideline.enabled,
                tags=updated_guideline.tags,
            ),
            relationships=[
                _guideline_relationship_to_dto(relationship, indirect)
                for relationship, indirect in await _get_relationships(
                    guideline_store=guideline_store,
                    tag_store=tag_store,
                    relationship_store=relationship_store,
                    guideline_id=guideline_id,
                    include_indirect=True,
                )
            ],
            tool_associations=[
                GuidelineToolAssociationDTO(
                    id=a.id,
                    guideline_id=a.guideline_id,
                    tool_id=ToolIdDTO(
                        service_name=a.tool_id.service_name,
                        tool_name=a.tool_id.tool_name,
                    ),
                )
                for a in await guideline_tool_association_store.list_associations()
                if a.guideline_id == guideline_id
            ],
        )

    @router.delete(
        "/{guideline_id}",
        operation_id="delete_guideline",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Guideline successfully deleted. No content returned."
            },
            status.HTTP_404_NOT_FOUND: {"description": "Guideline not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_guideline(
        request: Request,
        guideline_id: GuidelineIdPath,
    ) -> None:
        await authorization_policy.authorize(request=request, operation=Operation.DELETE_GUIDELINE)

        guideline = await guideline_store.read_guideline(guideline_id=guideline_id)

        for r, _ in await _get_relationships(
            guideline_store=guideline_store,
            tag_store=tag_store,
            relationship_store=relationship_store,
            guideline_id=guideline_id,
            include_indirect=False,
        ):
            related_guideline = (
                r.target if cast(Guideline | Tag, r.source).id == guideline_id else r.source
            )
            if (
                isinstance(related_guideline, Guideline)
                and related_guideline.tags
                and not any(t in related_guideline.tags for t in guideline.tags)
            ):
                await relationship_store.delete_relationship(r.id)

        for associastion in await guideline_tool_association_store.list_associations():
            if associastion.guideline_id == guideline_id:
                await guideline_tool_association_store.delete_association(associastion.id)

        journeys = await journey_store.list_journeys()
        for journey in journeys:
            for condition in journey.conditions:
                if condition == guideline_id:
                    await journey_store.remove_condition(
                        journey_id=journey.id,
                        condition=condition,
                    )

        await guideline_store.delete_guideline(guideline_id=guideline_id)

    return router
