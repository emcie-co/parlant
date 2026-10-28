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

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from parlant.core.agents import Effort
from parlant.core.rule_tool_associations import RuleToolAssociation
from parlant.core.app_modules.rules import (
    RuleLabelsUpdateParams,
    RuleMetadataUpdateParams,
    RuleModule,
    RuleRelationship,
    RuleToolAssociationUpdateParams,
)
from parlant.core.common import ItemNotFoundError, JSONSerializable, Weight
from parlant.core.groups import GroupId
from parlant.core.relationships import RelationshipEntityKind
from parlant.core.rules import Rule, RuleId
from parlant.core.tools import ToolId
from parlant.core.tunnel_handlers.common import TunnelHandlerSet
from parlant.core.tunnel_handlers.groups import serialize_group
from parlant.core.tunnel_handlers.services import serialize_tool


_PUBLIC_WEIGHTS: Mapping[str, Weight] = MappingProxyType(
    {weight.value: weight for weight in Weight}
)
_PUBLIC_WEIGHT_VALUES: Mapping[Weight, str] = MappingProxyType(
    {weight: value for value, weight in _PUBLIC_WEIGHTS.items()}
)

_PUBLIC_EFFORTS: Mapping[str, Effort] = MappingProxyType(
    {effort.value: effort for effort in Effort}
)
_PUBLIC_EFFORT_VALUES: Mapping[Effort, str] = MappingProxyType(
    {effort: value for value, effort in _PUBLIC_EFFORTS.items()}
)


class RuleTunnelHandlers:
    def __init__(self, rule_module: RuleModule) -> None:
        self._rule_module = rule_module

    def handler_set(self) -> TunnelHandlerSet:
        return TunnelHandlerSet(
            unary={
                "rules.create": self.create,
                "rules.list": self.list,
                "rules.retrieve": self.retrieve,
                "rules.update": self.update,
                "rules.delete": self.delete,
            },
            streams={},
        )

    async def create(self, params: dict[str, Any]) -> dict[str, Any]:
        condition = _parse_optional_string(params.get("condition"), field="condition")
        content = _parse_optional_string(params.get("content"), field="content")

        if not condition and not content:
            raise ValueError("A rule requires a condition, content, or both")

        rule = await self._rule_module.create(
            id=_parse_optional_rule_id(params.get("id"), field="id"),
            condition=condition,
            content=content,
            title=_parse_optional_string(params.get("title"), field="title"),
            weight=_parse_weight(params.get("weight")),
            metadata=_parse_metadata(params.get("metadata")),
            enabled=_parse_optional_bool(params.get("enabled"), field="enabled") is not False,
            groups=_parse_group_ids(params.get("groups"), field="groups"),
            effort=_parse_effort(params.get("effort")),
            labels=set(_parse_string_list(params.get("labels"), field="labels") or []),
            priority=_parse_priority(params.get("priority")),
            signals=_parse_string_list(params.get("signals"), field="signals") or [],
            anti_signals=_parse_string_list(params.get("anti_signals"), field="anti_signals") or [],
            message_review=_parse_optional_bool(
                params.get("message_review"), field="message_review"
            )
            or False,
            tool_review=_parse_optional_bool(params.get("tool_review"), field="tool_review")
            or False,
        )
        return {"rule": serialize_rule(rule)}

    async def list(self, params: dict[str, Any]) -> dict[str, Any]:
        rules = await self._rule_module.find(
            group_id=_parse_optional_group_id(params.get("group_id")),
        )
        return {"rules": [serialize_rule(rule) for rule in rules]}

    async def retrieve(self, params: dict[str, Any]) -> dict[str, Any]:
        rule_id = _require_rule_id(params.get("rule_id"))

        try:
            rule = await self._rule_module.read(rule_id=rule_id)
        except ItemNotFoundError:
            return {"rule": None, "relationships": [], "tool_associations": []}

        return await self._compound_response(rule)

    async def update(self, params: dict[str, Any]) -> dict[str, Any]:
        rule_id = _require_rule_id(params.get("rule_id"))

        rule = await self._rule_module.update(
            rule_id=rule_id,
            condition=_parse_optional_string(params.get("condition"), field="condition"),
            content=_parse_optional_string(params.get("content"), field="content"),
            title=_parse_optional_string(params.get("title"), field="title"),
            weight=_parse_optional_weight(params.get("weight")),
            tool_associations=_parse_tool_association_update(params.get("tool_associations")),
            enabled=_parse_optional_bool(params.get("enabled"), field="enabled"),
            groups=_parse_group_update(params.get("groups")),
            metadata=_parse_metadata_update(params.get("metadata")),
            effort=_parse_effort(params.get("effort")),
            labels=_parse_label_update(params.get("labels")),
            priority=_parse_optional_priority(params.get("priority")),
            signals=_parse_string_list(params.get("signals"), field="signals"),
            anti_signals=_parse_string_list(params.get("anti_signals"), field="anti_signals"),
            message_review=_parse_optional_bool(
                params.get("message_review"), field="message_review"
            ),
            tool_review=_parse_optional_bool(params.get("tool_review"), field="tool_review"),
        )
        return await self._compound_response(rule)

    async def delete(self, params: dict[str, Any]) -> dict[str, Any]:
        await self._rule_module.delete(rule_id=_require_rule_id(params.get("rule_id")))
        return {}

    async def _compound_response(self, rule: Rule) -> dict[str, Any]:
        relationships = await self._rule_module.find_relationships(
            rule_id=rule.id,
            include_indirect=True,
        )
        tool_associations = await self._rule_module.find_tool_associations(rule_id=rule.id)

        return {
            "rule": serialize_rule(rule),
            "relationships": [
                _serialize_rule_relationship(relationship, indirect)
                for relationship, indirect in relationships
            ],
            "tool_associations": [
                _serialize_tool_association(association) for association in tool_associations
            ],
        }


def serialize_rule(rule: Rule) -> dict[str, Any]:
    return {
        "id": str(rule.id),
        "condition": rule.condition,
        "content": rule.content,
        "title": rule.title,
        "weight": _PUBLIC_WEIGHT_VALUES[rule.weight],
        "enabled": rule.enabled,
        "groups": [str(group_id) for group_id in rule.groups],
        "metadata": dict(rule.metadata),
        "creation_utc": rule.creation_utc.isoformat(),
        "modified_utc": rule.modified_utc.isoformat(),
        "effort": _PUBLIC_EFFORT_VALUES[rule.effort] if rule.effort else None,
        "labels": sorted(rule.labels),
        "priority": rule.priority,
        "signals": list(rule.signals),
        "anti_signals": list(rule.anti_signals),
        "message_review": rule.message_review,
        "tool_review": rule.tool_review,
    }


def _serialize_rule_relationship(
    relationship: RuleRelationship,
    indirect: bool,
) -> dict[str, Any]:
    from parlant.core.tunnel_handlers.relationships import _PUBLIC_RELATIONSHIP_KIND_VALUES

    def serialize_entity(entity: Any, entity_type: RelationshipEntityKind) -> dict[str, Any]:
        if entity_type == RelationshipEntityKind.RULE:
            return {"rule": serialize_rule(entity)}
        if entity_type == RelationshipEntityKind.TOOL:
            return {"tool": serialize_tool(entity)}
        return {"group": serialize_group(entity)}

    return {
        "id": str(relationship.id),
        "source": serialize_entity(relationship.source, relationship.source_type),
        "target": serialize_entity(relationship.target, relationship.target_type),
        "kind": _PUBLIC_RELATIONSHIP_KIND_VALUES[relationship.kind],
        "group_id": relationship.group_id,
        "indirect": indirect,
    }


def _serialize_tool_association(association: RuleToolAssociation) -> dict[str, Any]:
    return {
        "id": str(association.id),
        "rule_id": str(association.rule_id),
        "tool_id": {
            "service_name": association.tool_id.service_name,
            "tool_name": association.tool_id.tool_name,
        },
    }


def _parse_optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value


def _require_rule_id(value: object) -> RuleId:
    if not isinstance(value, str) or not value:
        raise ValueError("rule_id must be a non-empty string")
    return RuleId(value)


def _parse_optional_rule_id(value: object, *, field: str) -> RuleId | None:
    parsed = _parse_optional_string(value, field=field)
    return RuleId(parsed) if parsed is not None else None


def _parse_optional_group_id(value: object) -> GroupId | None:
    parsed = _parse_optional_string(value, field="group_id")
    return GroupId(parsed) if parsed is not None else None


def _parse_weight(value: object) -> Weight:
    if value is None:
        return Weight.MEDIUM
    parsed = _parse_optional_weight(value)
    assert parsed is not None
    return parsed


def _parse_optional_weight(value: object) -> Weight | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in _PUBLIC_WEIGHTS:
        raise ValueError(f"weight must be one of {sorted(_PUBLIC_WEIGHTS)}")
    return _PUBLIC_WEIGHTS[value]


def _parse_effort(value: object) -> Effort | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in _PUBLIC_EFFORTS:
        raise ValueError(f"effort must be one of {list(_PUBLIC_EFFORTS)}")
    return _PUBLIC_EFFORTS[value]


def _parse_optional_bool(value: object, *, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean or null")
    return value


def _parse_metadata(value: object) -> Mapping[str, JSONSerializable]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object or null")
    return value


def _parse_string_list(value: object, *, field: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings or null")
    return value


def _parse_group_ids(value: object, *, field: str) -> list[GroupId] | None:
    values = _parse_string_list(value, field=field)
    if values is None:
        return None
    return [GroupId(group_id) for group_id in values]


def _parse_priority(value: object) -> int:
    parsed = _parse_optional_priority(value)
    return parsed if parsed is not None else 0


def _parse_optional_priority(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("priority must be an integer or null")
    return value


def _parse_metadata_update(value: object) -> RuleMetadataUpdateParams | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object or null")
    return RuleMetadataUpdateParams(
        set=_parse_metadata(value.get("set")),
        unset=_parse_string_list(value.get("unset"), field="metadata.unset") or [],
    )


def _parse_label_update(value: object) -> RuleLabelsUpdateParams | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("labels must be an object or null")
    return RuleLabelsUpdateParams(
        upsert=set(_parse_string_list(value.get("upsert"), field="labels.upsert") or []),
        remove=set(_parse_string_list(value.get("remove"), field="labels.remove") or []),
    )


def _parse_group_update(value: object) -> Any:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("groups must be an object or null")

    from parlant.core.app_modules.rules import RuleGroupsUpdateParams

    return RuleGroupsUpdateParams(
        add=_parse_group_ids(value.get("add"), field="groups.add"),
        remove=_parse_group_ids(value.get("remove"), field="groups.remove"),
    )


def _parse_tool_association_update(value: object) -> RuleToolAssociationUpdateParams | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("tool_associations must be an object or null")

    def parse_tool_ids(entries: object, *, field: str) -> list[ToolId] | None:
        if entries is None:
            return None
        if not isinstance(entries, list):
            raise ValueError(f"{field} must be a list of tool identifiers or null")
        tool_ids = []
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("service_name"), str)
                or not isinstance(entry.get("tool_name"), str)
            ):
                raise ValueError(f"{field} entries must carry service_name and tool_name strings")
            tool_ids.append(
                ToolId(service_name=entry["service_name"], tool_name=entry["tool_name"])
            )
        return tool_ids

    return RuleToolAssociationUpdateParams(
        add=parse_tool_ids(value.get("add"), field="tool_associations.add"),
        remove=parse_tool_ids(value.get("remove"), field="tool_associations.remove"),
    )
