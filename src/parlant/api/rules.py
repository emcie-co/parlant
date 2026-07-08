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
    RuleDTO,
    RuleEnabledField,
    RuleIdField,
    RuleLabelsField,
    RuleMetadataField,
    RelationshipDTO,
    RuleGroupsField,
    RelationshipKindDTO,
    GroupDTO,
    ToolIdDTO,
    apigen_config,
    composition_mode_dto_to_composition_mode,
    composition_mode_to_composition_mode_dto,
    effort_dto_to_effort,
    effort_to_effort_dto,
    rule_dto_example,
)
from parlant.core.app_modules.rules import (
    RuleLabelsUpdateParams,
    RuleMetadataUpdateParams,
    RuleRelationship,
    RuleGroupsUpdateParams,
    RuleToolAssociationUpdateParams,
)
from parlant.core.application import Application
from parlant.core.common import (
    Weight,
    DefaultBaseModel,
)
from parlant.api.common import (
    ExampleJson,
    RuleConditionField,
    RuleActionField,
)

from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipKind,
)
from parlant.core.rules import (
    Rule,
    RuleId,
)
from parlant.core.rule_tool_associations import RuleToolAssociationId
from parlant.core.groups import GroupId, Group
from parlant.core.tools import ToolId

API_GROUP = "rules"


RuleIdPath: TypeAlias = Annotated[
    RuleId,
    Path(
        description="Unique identifier for the rule",
        examples=["IUCGT-l4pS"],
    ),
]


RuleToolAssociationIdField: TypeAlias = Annotated[
    RuleToolAssociationId,
    Field(
        description="Unique identifier for the association between a tool and a rule",
        examples=["guid_tool_1"],
    ),
]


rule_tool_association_example: ExampleJson = {
    "id": "gta_101xyz",
    "rule_id": "guid_123xz",
    "tool_id": {"service_name": "pricing_service", "tool_name": "get_prices"},
}


class RuleToolAssociationDTO(
    DefaultBaseModel,
    json_schema_extra={"example": rule_tool_association_example},
):
    """
    Represents an association between a Rule and a Tool, enabling automatic tool invocation
    when the Rule's conditions are met.
    """

    id: RuleToolAssociationIdField
    rule_id: RuleIdField
    tool_id: ToolIdDTO


RuleConnectionAdditionSourceField: TypeAlias = Annotated[
    RuleId,
    Field(description="`id` of rule that is source of this connection."),
]

RuleConnectionAdditionTargetField: TypeAlias = Annotated[
    RuleId,
    Field(description="`id` of rule that is target of this connection."),
]


rule_connection_addition_example: ExampleJson = {
    "source": "guid_123xz",
    "target": "guid_789yz",
}


rule_tool_association_update_params_example: ExampleJson = {
    "add": [{"service_name": "pricing_service", "tool_name": "get_prices"}],
    "remove": [{"service_name": "old_service", "tool_name": "old_tool"}],
}


class RuleToolAssociationUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": rule_tool_association_update_params_example},
):
    """Parameters for adding/removing tool associations."""

    add: Sequence[ToolIdDTO] | None = None
    remove: Sequence[ToolIdDTO] | None = None


GroupIdQuery: TypeAlias = Annotated[
    GroupId | None,
    Query(
        description="The group ID to filter rules by",
        examples=["group:123"],
    ),
]


RuleGroupsUpdateAddField: TypeAlias = Annotated[
    list[GroupId],
    Field(
        description="List of group IDs to add to the rule",
        examples=[["group1", "group2"]],
    ),
]

RuleGroupsUpdateRemoveField: TypeAlias = Annotated[
    list[GroupId],
    Field(
        description="List of group IDs to remove from the rule",
        examples=[["group1", "group2"]],
    ),
]

rule_groups_update_params_example: ExampleJson = {
    "add": [
        "group1",
        "group2",
    ],
    "remove": [
        "group3",
        "group4",
    ],
}


class RuleGroupsUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": rule_groups_update_params_example},
):
    """
    Parameters for updating the groups of an existing rule.
    """

    add: RuleGroupsUpdateAddField | None = None
    remove: RuleGroupsUpdateRemoveField | None = None


rule_labels_update_params_example: ExampleJson = {
    "upsert": ["vip", "priority"],
    "remove": ["old_label"],
}


class RuleLabelsUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": rule_labels_update_params_example},
):
    """
    Parameters for updating the labels of an existing rule.
    """

    upsert: RuleLabelsField | None = None
    remove: RuleLabelsField | None = None


GroupIdField: TypeAlias = Annotated[
    GroupId,
    Field(
        description="Unique identifier for the group",
        examples=["t9a8g703f4"],
    ),
]

GroupNameField: TypeAlias = Annotated[
    str,
    Field(
        description="Name of the group",
        examples=["group1"],
    ),
]

rule_creation_params_example: ExampleJson = {
    "condition": "when the customer asks about pricing",
    "action": "provide current pricing information and mention any ongoing promotions",
    "enabled": False,
    "metadata": {"key1": "value1", "key2": "value2"},
    "composition_mode": "strict_canned",
    "effort": "high",
    "labels": ["vip", "priority"],
}


class RuleCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": rule_creation_params_example},
):
    """Parameters for creating a new rule."""

    id: RuleIdPath | None = None
    condition: RuleConditionField
    action: RuleActionField | None = None
    description: common.RuleDescriptionField | None = None
    title: common.RuleTitleField | None = None
    criticality: common.WeightDTO | None = None
    metadata: RuleMetadataField | None = None
    enabled: RuleEnabledField | None = None
    groups: RuleGroupsField | None = None
    composition_mode: CompositionModeDTO | None = None
    effort: common.EffortDTO | None = None
    track: bool = True
    labels: RuleLabelsField | None = None
    priority: int = 0
    signals: common.RuleSignalsField = []
    anti_signals: common.RuleAntiSignalsField = []


RuleMetadataUnsetField: TypeAlias = Annotated[
    Sequence[str],
    Field(description="Metadata keys to remove from the rule"),
]

rule_metadata_update_params_example: ExampleJson = {
    "set": {
        "key1": "value1",
        "key2": "value2",
    },
    "unset": ["key3", "key4"],
}


class RuleMetadataUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": rule_metadata_update_params_example},
):
    """Parameters for updating the metadata of a rule."""

    set: RuleMetadataField | None = None
    unset: RuleMetadataUnsetField | None = None


rule_update_params_example: ExampleJson = {
    "condition": "when the customer asks about pricing",
    "action": "provide current pricing information",
    "enabled": True,
    "effort": "high",
    "groups": ["group1", "group2"],
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


class RuleUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": rule_update_params_example},
):
    """Parameters for updating a rule."""

    condition: RuleConditionField | None = None
    action: RuleActionField | None = None
    description: common.RuleDescriptionField | None = None
    title: common.RuleTitleField | None = None
    criticality: common.WeightDTO | None = None
    tool_associations: RuleToolAssociationUpdateParamsDTO | None = None
    enabled: RuleEnabledField | None = None
    groups: RuleGroupsUpdateParamsDTO | None = None
    metadata: RuleMetadataUpdateParamsDTO | None = None
    composition_mode: CompositionModeDTO | None = None
    effort: common.EffortDTO | None = None
    labels: RuleLabelsUpdateParamsDTO | None = None
    priority: int | None = None
    signals: common.RuleSignalsField | None = None
    anti_signals: common.RuleAntiSignalsField | None = None


rule_with_relationships_example: ExampleJson = {
    "rule": {
        "id": "guid_123xz",
        "condition": "when the customer asks about pricing",
        "action": "provide current pricing information",
        "enabled": True,
        "groups": ["group1", "group2"],
    },
    "relationships": [
        {
            "id": "123",
            "source_rule": {
                "id": "guid_123xz",
                "condition": "when the customer asks about pricing",
                "action": "provide current pricing information",
                "enabled": True,
                "groups": ["group1", "group2"],
            },
            "target_group": {
                "id": "tid_456yz",
                "name": "group1",
            },
            "indirect": False,
            "kind": "entailment",
        }
    ],
    "tool_associations": [
        {
            "id": "gta_101xyz",
            "rule_id": "guid_123xz",
            "tool_id": {"service_name": "pricing_service", "tool_name": "get_prices"},
        }
    ],
}


class RuleWithRelationshipsAndToolAssociationsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": rule_with_relationships_example},
):
    """A Rule with its relationships and tool associations."""

    rule: RuleDTO
    relationships: Sequence[RelationshipDTO]
    tool_associations: Sequence[RuleToolAssociationDTO]


def _criticality_to_dto(criticality: Weight) -> common.WeightDTO:
    match criticality:
        case Weight.LOW:
            return common.WeightDTO.LOW
        case Weight.MEDIUM:
            return common.WeightDTO.MEDIUM
        case Weight.HIGH:
            return common.WeightDTO.HIGH
        case _:
            raise ValueError(f"Invalid criticality: {criticality.value}")


def _criticality_from_dto(dto: common.WeightDTO) -> Weight:
    match dto:
        case common.WeightDTO.LOW:
            return Weight.LOW
        case common.WeightDTO.MEDIUM:
            return Weight.MEDIUM
        case common.WeightDTO.HIGH:
            return Weight.HIGH
        case _:
            raise ValueError(f"Invalid criticality DTO: {dto.value}")


def _rule_relationship_kind_to_dto(
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
            raise ValueError(f"Invalid rule relationship kind: {kind.value}")


def _rule_to_dto(rule: Rule) -> RuleDTO:
    return RuleDTO(
        id=rule.id,
        condition=rule.content.condition,
        action=rule.content.action,
        description=rule.content.description,
        title=rule.title,
        criticality=_criticality_to_dto(rule.weight),
        enabled=rule.enabled,
        groups=rule.groups,
        metadata=rule.metadata,
        modified_utc=rule.modified_utc,
        composition_mode=composition_mode_to_composition_mode_dto(rule.composition_mode)
        if rule.composition_mode
        else None,
        effort=effort_to_effort_dto(rule.effort_lift) if rule.effort_lift else None,
        track=rule.track,
        labels=rule.labels,
        priority=rule.priority,
        signals=rule.signals,
        anti_signals=rule.anti_signals,
    )


def _rule_relationship_to_dto(
    relationship: RuleRelationship,
    indirect: bool,
) -> RelationshipDTO:
    if relationship.source_type == RelationshipEntityKind.RULE:
        rel_source_rule = cast(Rule, relationship.source)
    else:
        rel_source_group = cast(Group, relationship.source)

    if relationship.target_type == RelationshipEntityKind.RULE:
        rel_target_rule = cast(Rule, relationship.target)
    else:
        rel_target_group = cast(Group, relationship.target)

    return RelationshipDTO(
        id=relationship.id,
        source_rule=_rule_to_dto(rel_source_rule)
        if relationship.source_type == RelationshipEntityKind.RULE
        else None,
        source_group=GroupDTO(
            id=rel_source_group.id,
            creation_utc=rel_source_group.creation_utc,
            name=rel_source_group.name,
        )
        if relationship.source_type.is_group
        else None,
        target_rule=_rule_to_dto(rel_target_rule)
        if relationship.target_type == RelationshipEntityKind.RULE
        else None,
        target_group=GroupDTO(
            id=rel_target_group.id,
            name=rel_target_group.name,
        )
        if relationship.target_type.is_group
        else None,
        indirect=indirect,
        kind=_rule_relationship_kind_to_dto(relationship.kind),
        group_id=relationship.group_id,
    )


def create_router(
    authorization_policy: AuthorizationPolicy,
    app: Application,
) -> APIRouter:
    """Creates a router for the rules API with group-based paths."""
    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_rule",
        response_model=RuleDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Rule successfully created. Returns the created rule.",
                "content": common.example_json_content(rule_dto_example),
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_rule(
        request: Request,
        params: RuleCreationParamsDTO,
    ) -> RuleDTO:
        """
        Creates a new rule.

        The rule will be initialized with the provided condition and optional action and settings.
        A unique identifier will be automatically generated unless a custom ID is provided.

        See the [documentation](https://parlant.io/docs/concepts/customization/rules) for more information.
        """
        await authorization_policy.authorize(request=request, operation=Operation.CREATE_RULE)

        try:
            rule = await app.rules.create(
                condition=params.condition,
                action=params.action or None,
                description=params.description or None,
                title=params.title or None,
                criticality=_criticality_from_dto(params.criticality)
                if params.criticality
                else None,
                metadata=params.metadata or {},
                enabled=params.enabled,
                groups=params.groups,
                id=params.id,
                composition_mode=composition_mode_dto_to_composition_mode(params.composition_mode)
                if params.composition_mode
                else None,
                effort=effort_dto_to_effort(params.effort) if params.effort else None,
                track=params.track,
                labels=params.labels,
                priority=params.priority,
                signals=params.signals,
                anti_signals=params.anti_signals,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(e),
            )

        return _rule_to_dto(rule)

    @router.get(
        "",
        operation_id="list_rules",
        response_model=Sequence[RuleDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all rules for the specified group or all rules if no group is provided",
                "content": common.example_json_content([rule_dto_example]),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_rules(
        request: Request,
        group_id: GroupIdQuery = None,
    ) -> Sequence[RuleDTO]:
        """
        Lists all rules for the specified group or all rules if no group is provided.

        Returns an empty list if no rules exist.
        Rules are returned in no guaranteed order.
        Does not include relationships or tool associations.
        """
        await authorization_policy.authorize(request=request, operation=Operation.LIST_RULES)

        rules = await app.rules.find(group_id=group_id)

        return [_rule_to_dto(rule) for rule in rules]

    @router.get(
        "/{rule_id}",
        operation_id="read_rule",
        response_model=RuleWithRelationshipsAndToolAssociationsDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Rule details successfully retrieved. Returns the complete rule with its relationships and tool associations.",
                "content": common.example_json_content(rule_with_relationships_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Rule not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_rule(
        request: Request,
        rule_id: RuleIdPath,
    ) -> RuleWithRelationshipsAndToolAssociationsDTO:
        """
        Retrieves a specific rule with all its relationships and tool associations.

        Returns both direct and indirect relationships between rules.
        Tool associations indicate which tools the rule can use.
        """
        await authorization_policy.authorize(request=request, operation=Operation.READ_RULE)

        try:
            rule = await app.rules.read(rule_id=rule_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Rule not found",
            )

        relationships = await app.rules.find_relationships(
            rule_id=rule_id,
            include_indirect=True,
        )

        rule_tool_associations = await app.rules.find_tool_associations(rule_id=rule_id)

        return RuleWithRelationshipsAndToolAssociationsDTO(
            rule=_rule_to_dto(rule),
            relationships=[
                _rule_relationship_to_dto(relationship, indirect)
                for relationship, indirect in relationships
            ],
            tool_associations=[
                RuleToolAssociationDTO(
                    id=a.id,
                    rule_id=a.rule_id,
                    tool_id=ToolIdDTO(
                        service_name=a.tool_id.service_name,
                        tool_name=a.tool_id.tool_name,
                    ),
                )
                for a in rule_tool_associations
            ],
        )

    @router.patch(
        "/{rule_id}",
        operation_id="update_rule",
        response_model=RuleWithRelationshipsAndToolAssociationsDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Rule successfully updated. Returns the updated rule with its relationships and tool associations.",
                "content": common.example_json_content(rule_with_relationships_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Rule or referenced tool not found"},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Invalid relationship rules or validation error in update parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_rule(
        request: Request,
        rule_id: RuleIdPath,
        params: RuleUpdateParamsDTO,
    ) -> RuleWithRelationshipsAndToolAssociationsDTO:
        """Updates a rule's relationships and tool associations.

        Only provided attributes will be updated; others remain unchanged.

        Relationship rules:
        - A rule cannot relate to itself
        - Only direct relationships can be removed
        - The relationship must specify this rule as source or target

        Tool Association rules:
        - Tool services and tools must exist before creating associations

        Action with text can not be updated to None.
        """
        await authorization_policy.authorize(request=request, operation=Operation.UPDATE_RULE)

        updated_rule = await app.rules.update(
            rule_id=rule_id,
            condition=params.condition,
            action=params.action,
            description=params.description,
            title=params.title,
            criticality=_criticality_from_dto(params.criticality) if params.criticality else None,
            tool_associations=RuleToolAssociationUpdateParams(
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
            groups=RuleGroupsUpdateParams(
                add=params.groups.add,
                remove=params.groups.remove,
            )
            if params.groups
            else None,
            metadata=RuleMetadataUpdateParams(
                set=params.metadata.set,
                unset=params.metadata.unset,
            )
            if params.metadata
            else None,
            composition_mode=composition_mode_dto_to_composition_mode(params.composition_mode)
            if params.composition_mode
            else None,
            effort=effort_dto_to_effort(params.effort) if params.effort else None,
            labels=RuleLabelsUpdateParams(
                upsert=params.labels.upsert,
                remove=params.labels.remove,
            )
            if params.labels
            else None,
            priority=params.priority,
            signals=params.signals,
            anti_signals=params.anti_signals,
        )

        rule_tool_associations = await app.rules.find_tool_associations(rule_id)

        return RuleWithRelationshipsAndToolAssociationsDTO(
            rule=_rule_to_dto(updated_rule),
            relationships=[
                _rule_relationship_to_dto(relationship, indirect)
                for relationship, indirect in await app.rules.find_relationships(
                    rule_id=rule_id,
                    include_indirect=True,
                )
            ],
            tool_associations=[
                RuleToolAssociationDTO(
                    id=a.id,
                    rule_id=a.rule_id,
                    tool_id=ToolIdDTO(
                        service_name=a.tool_id.service_name,
                        tool_name=a.tool_id.tool_name,
                    ),
                )
                for a in rule_tool_associations
            ],
        )

    @router.delete(
        "/{rule_id}",
        operation_id="delete_rule",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Rule successfully deleted. No content returned."
            },
            status.HTTP_404_NOT_FOUND: {"description": "Rule not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_rule(
        request: Request,
        rule_id: RuleIdPath,
    ) -> None:
        await authorization_policy.authorize(request=request, operation=Operation.DELETE_RULE)

        await app.rules.delete(rule_id=rule_id)

    return router
