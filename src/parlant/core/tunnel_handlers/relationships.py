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

from parlant.core.app_modules.relationships import RelationshipModel, RelationshipModule
from parlant.core.common import ItemNotFoundError
from parlant.core.groups import GroupId
from parlant.core.relationships import RelationshipId, RelationshipKind
from parlant.core.rules import RuleId
from parlant.core.tools import ToolId
from parlant.core.tunnel_handlers.common import TunnelHandlerSet
from parlant.core.tunnel_handlers.groups import serialize_group
from parlant.core.tunnel_handlers.rules import serialize_rule
from parlant.core.tunnel_handlers.services import serialize_tool


_PUBLIC_RELATIONSHIP_KINDS: Mapping[str, RelationshipKind] = MappingProxyType(
    {
        "entailment": RelationshipKind.ENTAILMENT,
        "priority": RelationshipKind.PRIORITY,
        "dependency": RelationshipKind.DEPENDENCY,
        "dependency_any": RelationshipKind.DEPENDENCY_ANY,
        "reevaluation": RelationshipKind.REEVALUATION,
        "overlap": RelationshipKind.OVERLAP,
    }
)
_PUBLIC_RELATIONSHIP_KIND_VALUES: Mapping[RelationshipKind, str] = MappingProxyType(
    {kind: value for value, kind in _PUBLIC_RELATIONSHIP_KINDS.items()}
)


class RelationshipTunnelHandlers:
    def __init__(self, relationship_module: RelationshipModule) -> None:
        self._relationship_module = relationship_module

    def handler_set(self) -> TunnelHandlerSet:
        return TunnelHandlerSet(
            unary={
                "relationships.create": self.create,
                "relationships.list": self.list,
                "relationships.retrieve": self.retrieve,
                "relationships.delete": self.delete,
            },
            streams={},
        )

    async def create(self, params: dict[str, Any]) -> dict[str, Any]:
        source_rule = _parse_optional_rule_id(params.get("source_rule"), field="source_rule")
        source_group = _parse_optional_group_id(params.get("source_group"), field="source_group")
        source_tool = _parse_optional_tool_input(params.get("source_tool"), field="source_tool")
        target_rule = _parse_optional_rule_id(params.get("target_rule"), field="target_rule")
        target_group = _parse_optional_group_id(params.get("target_group"), field="target_group")
        target_tool = _parse_optional_tool_input(params.get("target_tool"), field="target_tool")
        kind = _parse_relationship_kind(params.get("kind"))
        group_id = _parse_optional_string(params.get("group_id"), field="group_id")

        if source_rule and source_group:
            raise ValueError("A relationship cannot have both a source rule and a source group")
        if target_rule and target_group:
            raise ValueError("A relationship cannot have both a target rule and a target group")
        if (source_rule and target_rule and source_rule == target_rule) or (
            source_group and target_group and source_group == target_group
        ):
            raise ValueError("source and target cannot be the same entity")

        relationship = await self._relationship_module.create(
            source_rule=source_rule,
            source_group=source_group,
            source_tool=source_tool,
            target_rule=target_rule,
            target_group=target_group,
            target_tool=target_tool,
            kind=kind,
            group_id=group_id,
        )
        return {"relationship": _serialize_relationship(relationship)}

    async def list(self, params: dict[str, Any]) -> dict[str, Any]:
        relationships = await self._relationship_module.find(
            kind=_parse_optional_relationship_kind(params.get("kind")),
            indirect=_parse_indirect(params),
            rule_id=_parse_optional_rule_id(params.get("rule_id"), field="rule_id"),
            group_id=_parse_optional_group_id(params.get("group_id"), field="group_id"),
            tool_id=_parse_tool_filter(params.get("tool_id")),
        )
        return {
            "relationships": [
                _serialize_relationship(relationship) for relationship in relationships
            ]
        }

    async def retrieve(self, params: dict[str, Any]) -> dict[str, Any]:
        relationship_id = _require_relationship_id(params.get("relationship_id"))
        try:
            relationship = await self._relationship_module.read(relationship_id=relationship_id)
        except ItemNotFoundError:
            return {"relationship": None}
        return {"relationship": _serialize_relationship(relationship)}

    async def delete(self, params: dict[str, Any]) -> dict[str, Any]:
        await self._relationship_module.delete(
            relationship_id=_require_relationship_id(params.get("relationship_id"))
        )
        return {}


def _parse_optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value


def _parse_optional_rule_id(value: object, *, field: str) -> RuleId | None:
    parsed = _parse_optional_string(value, field=field)
    return RuleId(parsed) if parsed is not None else None


def _parse_optional_group_id(value: object, *, field: str) -> GroupId | None:
    parsed = _parse_optional_string(value, field=field)
    return GroupId(parsed) if parsed is not None else None


def _parse_optional_tool_input(value: object, *, field: str) -> ToolId | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a tool identifier object or null")
    service_name = value.get("service_name")
    tool_name = value.get("tool_name")
    if not isinstance(service_name, str):
        raise ValueError(f"{field}.service_name must be a string")
    if not isinstance(tool_name, str):
        raise ValueError(f"{field}.tool_name must be a string")
    return ToolId(service_name=service_name, tool_name=tool_name)


def _parse_relationship_kind(value: object) -> RelationshipKind:
    if not isinstance(value, str) or value not in _PUBLIC_RELATIONSHIP_KINDS:
        raise ValueError(f"Unsupported relationship kind: {value}")
    return _PUBLIC_RELATIONSHIP_KINDS[value]


def _parse_optional_relationship_kind(value: object) -> RelationshipKind | None:
    if value is None:
        return None
    return _parse_relationship_kind(value)


def _parse_indirect(params: dict[str, Any]) -> bool:
    if "indirect" not in params:
        return True
    value = params["indirect"]
    if not isinstance(value, bool):
        raise ValueError("indirect must be a boolean")
    return value


def _parse_tool_filter(value: object) -> ToolId | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or value.count(":") != 1:
        raise ValueError("tool_id must be a string containing exactly one colon")
    service_name, tool_name = value.split(":")
    return ToolId(service_name=service_name, tool_name=tool_name)


def _require_relationship_id(value: object) -> RelationshipId:
    if not isinstance(value, str) or not value:
        raise ValueError("relationship_id must be a non-empty string")
    return RelationshipId(value)


def _serialize_relationship(relationship: RelationshipModel) -> dict[str, Any]:
    return {
        "id": str(relationship.id),
        "source_rule": serialize_rule(relationship.source_rule)
        if relationship.source_rule
        else None,
        "source_group": serialize_group(relationship.source_group)
        if relationship.source_group
        else None,
        "source_tool": serialize_tool(relationship.source_tool)
        if relationship.source_tool
        else None,
        "target_rule": serialize_rule(relationship.target_rule)
        if relationship.target_rule
        else None,
        "target_group": serialize_group(relationship.target_group)
        if relationship.target_group
        else None,
        "target_tool": serialize_tool(relationship.target_tool)
        if relationship.target_tool
        else None,
        "kind": _PUBLIC_RELATIONSHIP_KIND_VALUES[relationship.kind],
        "group_id": relationship.group_id,
    }
