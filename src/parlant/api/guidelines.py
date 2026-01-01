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

from typing import Annotated, Sequence, TypeAlias, cast
from fastapi import APIRouter, HTTPException, Path, Request, status, Query
from pydantic import Field

from parlant.api import common
from parlant.api.authorization import Operation, AuthorizationPolicy
from parlant.api.common import (
    CompositionModeDTO,
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
    composition_mode_dto_to_composition_mode,
    composition_mode_to_composition_mode_dto,
    guideline_dto_example,
)
from parlant.core.app_modules.guidelines import (
    GuidelineMetadataUpdateParams,
    GuidelineRelationship,
    GuidelineTagsUpdateParams,
    GuidelineToolAssociationUpdateParams,
)
from parlant.core.application import Application
from parlant.core.common import (
    Criticality,
    DefaultBaseModel,
)
from parlant.api.common import (
    ExampleJson,
    GuidelineConditionField,
    GuidelineActionField,
)

from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipKind,
)
from parlant.core.guidelines import (
    Guideline,
    GuidelineId,
)
from parlant.core.guideline_tool_associations import GuidelineToolAssociationId
from parlant.core.tags import TagId, Tag
from parlant.core.tools import ToolId

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

    add: Sequence[ToolIdDTO] | None = None
    remove: Sequence[ToolIdDTO] | None = None


TagIdQuery: TypeAlias = Annotated[
    TagId | None,
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

    add: GuidelineTagsUpdateAddField | None = None
    remove: GuidelineTagsUpdateRemoveField | None = None


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
    "composition_mode": "strict_canned",
}


class GuidelineCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_creation_params_example},
):
    """Parameters for creating a new guideline."""

    id: GuidelineIdPath | None = None
    condition: GuidelineConditionField
    action: GuidelineActionField | None = None
    description: common.GuidelineDescriptionField | None = None
    criticality: common.CriticalityDTO | None = None
    metadata: GuidelineMetadataField | None = None
    enabled: GuidelineEnabledField | None = None
    tags: GuidelineTagsField | None = None
    composition_mode: CompositionModeDTO | None = None


GuidelineMetadataUnsetField: TypeAlias = Annotated[
    Sequence[str],
    Field(description="Metadata keys to remove from the guideline"),
]

guideline_metadata_update_params_example: ExampleJson = {
    "set": {
        "key1": "value1",
        "key2": "value2",
    },
    "unset": ["key3", "key4"],
}


class GuidelineMetadataUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_metadata_update_params_example},
):
    """Parameters for updating the metadata of a guideline."""

    set: GuidelineMetadataField | None = None
    unset: GuidelineMetadataUnsetField | None = None


guideline_update_params_example: ExampleJson = {
    "condition": "when the customer asks about pricing",
    "action": "provide current pricing information",
    "enabled": True,
    "tags": ["tag1", "tag2"],
    "metadata": {
        "set": {
            "key1": "value1",
            "key2": "value2",
        },
        "unset": ["key3", "key4"],
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

    condition: GuidelineConditionField | None = None
    action: GuidelineActionField | None = None
    description: common.GuidelineDescriptionField | None = None
    criticality: common.CriticalityDTO | None = None
    tool_associations: GuidelineToolAssociationUpdateParamsDTO | None = None
    enabled: GuidelineEnabledField | None = None
    tags: GuidelineTagsUpdateParamsDTO | None = None
    metadata: GuidelineMetadataUpdateParamsDTO | None = None
    composition_mode: CompositionModeDTO | None = None


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


def _criticality_to_dto(criticality: Criticality) -> common.CriticalityDTO:
    match criticality:
        case Criticality.LOW:
            return common.CriticalityDTO.LOW
        case Criticality.MEDIUM:
            return common.CriticalityDTO.MEDIUM
        case Criticality.HIGH:
            return common.CriticalityDTO.HIGH
        case _:
            raise ValueError(f"Invalid criticality: {criticality.value}")


def _criticality_from_dto(dto: common.CriticalityDTO) -> Criticality:
    match dto:
        case common.CriticalityDTO.LOW:
            return Criticality.LOW
        case common.CriticalityDTO.MEDIUM:
            return Criticality.MEDIUM
        case common.CriticalityDTO.HIGH:
            return Criticality.HIGH
        case _:
            raise ValueError(f"Invalid criticality DTO: {dto.value}")


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


def _guideline_relationship_to_dto(
    relationship: GuidelineRelationship,
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
            description=rel_source_guideline.content.description,
            criticality=_criticality_to_dto(rel_source_guideline.criticality),
            enabled=rel_source_guideline.enabled,
            tags=rel_source_guideline.tags,
            metadata=rel_source_guideline.metadata,
            composition_mode=composition_mode_to_composition_mode_dto(
                rel_source_guideline.composition_mode
            )
            if rel_source_guideline.composition_mode
            else None,
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
            description=rel_target_guideline.content.description,
            criticality=_criticality_to_dto(rel_target_guideline.criticality),
            enabled=rel_target_guideline.enabled,
            tags=rel_target_guideline.tags,
            metadata=rel_target_guideline.metadata,
            composition_mode=composition_mode_to_composition_mode_dto(
                rel_target_guideline.composition_mode
            )
            if rel_target_guideline.composition_mode
            else None,
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
    app: Application,
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
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
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

        The guideline will be initialized with the provided condition and optional action and settings.
        A unique identifier will be automatically generated unless a custom ID is provided.

        See the [documentation](https://parlant.io/docs/concepts/customization/guidelines) for more information.
        """
        await authorization_policy.authorize(request=request, operation=Operation.CREATE_GUIDELINE)

        try:
            guideline = await app.guidelines.create(
                condition=params.condition,
                action=params.action or None,
                description=params.description or None,
                criticality=_criticality_from_dto(params.criticality)
                if params.criticality
                else None,
                metadata=params.metadata or {},
                enabled=params.enabled or True,
                tags=params.tags,
                id=params.id,
                composition_mode=composition_mode_dto_to_composition_mode(params.composition_mode)
                if params.composition_mode
                else None,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(e),
            )

        return GuidelineDTO(
            id=guideline.id,
            condition=guideline.content.condition,
            action=guideline.content.action,
            description=guideline.content.description,
            criticality=_criticality_to_dto(guideline.criticality),
            metadata=guideline.metadata,
            enabled=guideline.enabled,
            tags=guideline.tags,
            composition_mode=composition_mode_to_composition_mode_dto(guideline.composition_mode)
            if guideline.composition_mode
            else None,
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

        guidelines = await app.guidelines.find(tag_id=tag_id)

        return [
            GuidelineDTO(
                id=guideline.id,
                condition=guideline.content.condition,
                action=guideline.content.action,
                description=guideline.content.description,
                criticality=_criticality_to_dto(guideline.criticality),
                metadata=guideline.metadata,
                enabled=guideline.enabled,
                tags=guideline.tags,
                composition_mode=composition_mode_to_composition_mode_dto(
                    guideline.composition_mode
                )
                if guideline.composition_mode
                else None,
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
            guideline = await app.guidelines.read(guideline_id=guideline_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Guideline not found",
            )

        relationships = await app.guidelines.find_relationships(
            guideline_id=guideline_id,
            include_indirect=True,
        )

        guideline_tool_associations = await app.guidelines.find_tool_associations(
            guideline_id=guideline_id
        )

        return GuidelineWithRelationshipsAndToolAssociationsDTO(
            guideline=GuidelineDTO(
                id=guideline.id,
                condition=guideline.content.condition,
                action=guideline.content.action,
                description=guideline.content.description,
                criticality=_criticality_to_dto(guideline.criticality),
                metadata=guideline.metadata,
                enabled=guideline.enabled,
                tags=guideline.tags,
                composition_mode=composition_mode_to_composition_mode_dto(
                    guideline.composition_mode
                )
                if guideline.composition_mode
                else None,
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
                for a in guideline_tool_associations
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
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
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

        updated_guideline = await app.guidelines.update(
            guideline_id=guideline_id,
            condition=params.condition,
            action=params.action,
            description=params.description,
            criticality=_criticality_from_dto(params.criticality) if params.criticality else None,
            tool_associations=GuidelineToolAssociationUpdateParams(
                add=[
                    ToolId(service_name=t.service_name, tool_name=t.tool_name)
                    for t in params.tool_associations.add
                ]
                if params.tool_associations.add
                else None,
                remove=[
                    ToolId(service_name=t.service_name, tool_name=t.tool_name)
                    for t in params.tool_associations.remove
                ]
                if params.tool_associations.remove
                else None,
            )
            if params.tool_associations
            else None,
            enabled=params.enabled,
            tags=GuidelineTagsUpdateParams(
                add=params.tags.add,
                remove=params.tags.remove,
            )
            if params.tags
            else None,
            metadata=GuidelineMetadataUpdateParams(
                set=params.metadata.set,
                unset=params.metadata.unset,
            )
            if params.metadata
            else None,
            composition_mode=composition_mode_dto_to_composition_mode(params.composition_mode)
            if params.composition_mode
            else None,
        )

        guideline_tool_associations = await app.guidelines.find_tool_associations(guideline_id)

        return GuidelineWithRelationshipsAndToolAssociationsDTO(
            guideline=GuidelineDTO(
                id=updated_guideline.id,
                condition=updated_guideline.content.condition,
                action=updated_guideline.content.action,
                description=updated_guideline.content.description,
                criticality=_criticality_to_dto(updated_guideline.criticality),
                metadata=updated_guideline.metadata,
                enabled=updated_guideline.enabled,
                tags=updated_guideline.tags,
                composition_mode=composition_mode_to_composition_mode_dto(
                    updated_guideline.composition_mode
                )
                if updated_guideline.composition_mode
                else None,
            ),
            relationships=[
                _guideline_relationship_to_dto(relationship, indirect)
                for relationship, indirect in await app.guidelines.find_relationships(
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
                for a in guideline_tool_associations
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

        await app.guidelines.delete(guideline_id=guideline_id)

    return router
