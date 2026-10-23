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

from typing import Any, cast

from croniter import croniter

from parlant.core.agents import AgentId
from parlant.core.app_modules.context_variables import (
    ContextVariableModule,
    ContextVariableTagsUpdateParams,
    ResolvedContextVariable,
)
from parlant.core.customers import CustomerId
from parlant.core.app_modules.groups import GroupModule
from parlant.core.common import JSONSerializable
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableId,
    ContextVariableValue,
)
from parlant.core.groups import GroupId
from parlant.core.tools import ToolId
from parlant.core.tunnel_handlers.common import TunnelHandlerSet


class ContextVariableTunnelHandlers:
    def __init__(
        self,
        context_variable_module: ContextVariableModule,
        group_module: GroupModule | None = None,
    ) -> None:
        self._context_variable_module = context_variable_module
        self._group_module = group_module

    def handler_set(self) -> TunnelHandlerSet:
        return TunnelHandlerSet(
            unary={
                "context_variables.create": self.create,
                "context_variables.list": self.list,
                "context_variables.retrieve": self.retrieve,
                "context_variables.resolve": self.resolve,
                "context_variables.update": self.update,
                "context_variables.delete": self.delete,
                "context_variables.delete_many": self.delete_many,
                "context_variables.list_values": self.list_values,
                "context_variables.retrieve_value": self.retrieve_value,
                "context_variables.update_value": self.update_value,
                "context_variables.delete_value": self.delete_value,
            },
            streams={},
        )

    async def resolve(self, params: dict[str, Any]) -> dict[str, Any]:
        """Every variable the engine would load for a customer, with the
        source that won -- customer, one of the customer's groups, the
        agent, or the global default. Read-only: stored values are reported
        as stored, never tool-refreshed.
        """
        customer_id = CustomerId(
            _require_nonempty_string(params.get("customer_id"), field="customer_id")
        )
        agent_id = AgentId(_require_nonempty_string(params.get("agent_id"), field="agent_id"))
        resolved = await self._context_variable_module.resolve_for_customer(
            customer_id=customer_id,
            agent_id=agent_id,
        )
        return {
            "resolved": [await self._serialize_resolved(entry) for entry in resolved],
        }

    async def _serialize_resolved(self, entry: ResolvedContextVariable) -> dict[str, Any]:
        source: dict[str, Any] = {"kind": entry.source.kind}
        if entry.source.kind == "group" and entry.source.group_id is not None:
            source["group_id"] = str(entry.source.group_id)
            # The drawer badge renders the group's NAME ("from group:
            # enterprise"), so the name travels with the id. A group that was
            # deleted after its value stuck around still resolves; its id is
            # the best name left.
            group_name = str(entry.source.group_id)
            if self._group_module is not None:
                try:
                    group = await self._group_module.read(entry.source.group_id)
                    group_name = group.name
                except Exception:  # noqa: BLE001 - the id is the fallback name
                    pass
            source["group_name"] = group_name
        return {
            "context_variable": _serialize_context_variable(entry.variable),
            "value": _serialize_context_variable_value(entry.value),
            "source": source,
        }

    async def create(self, params: dict[str, Any]) -> dict[str, Any]:
        context_variable = await self._context_variable_module.create(
            name=_require_nonempty_string(params.get("name"), field="name"),
            description=_parse_optional_string(params.get("description"), field="description"),
            tool_id=_parse_optional_tool_id(params.get("tool_id"), field="tool_id"),
            freshness_rules=_parse_freshness_rules(
                params.get("freshness_rules"),
                field="freshness_rules",
            ),
            groups=_parse_group_ids(params.get("groups"), field="groups"),
        )
        return {"context_variable": _serialize_context_variable(context_variable)}

    async def list(self, params: dict[str, Any]) -> dict[str, Any]:
        context_variables = await self._context_variable_module.find(
            group_id=_parse_optional_group_id(params.get("group_id"), field="group_id")
        )
        return {
            "context_variables": [
                _serialize_context_variable(context_variable)
                for context_variable in context_variables
            ]
        }

    async def retrieve(self, params: dict[str, Any]) -> dict[str, Any]:
        variable_id = _require_context_variable_id(
            params.get("variable_id"),
            field="variable_id",
        )
        context_variable = await self._context_variable_module.read(variable_id=variable_id)
        key_value_pairs = await self._context_variable_module.find_values(variable_id=variable_id)
        return {
            "context_variable": _serialize_context_variable(context_variable),
            "key_value_pairs": {
                key: _serialize_context_variable_value(value) for key, value in key_value_pairs
            },
        }

    async def update(self, params: dict[str, Any]) -> dict[str, Any]:
        context_variable = await self._context_variable_module.update(
            variable_id=_require_context_variable_id(
                params.get("variable_id"),
                field="variable_id",
            ),
            name=_parse_optional_nonempty_string(params.get("name"), field="name"),
            description=_parse_optional_string(params.get("description"), field="description"),
            tool_id=_parse_optional_tool_id(params.get("tool_id"), field="tool_id"),
            freshness_rules=_parse_freshness_rules(
                params.get("freshness_rules"),
                field="freshness_rules",
            ),
            groups=_parse_group_update(params.get("groups")),
        )
        return {"context_variable": _serialize_context_variable(context_variable)}

    async def delete(self, params: dict[str, Any]) -> dict[str, Any]:
        await self._context_variable_module.delete(
            variable_id=_require_context_variable_id(
                params.get("variable_id"),
                field="variable_id",
            )
        )
        return {}

    async def delete_many(self, params: dict[str, Any]) -> dict[str, Any]:
        await self._context_variable_module.delete_many(
            group_id=_parse_optional_group_id(params.get("group_id"), field="group_id")
        )
        return {}

    async def list_values(self, params: dict[str, Any]) -> dict[str, Any]:
        variable_id = _require_context_variable_id(
            params.get("variable_id"),
            field="variable_id",
        )
        values = await self._context_variable_module.find_values(variable_id=variable_id)
        return {
            "values": [
                {"key": key, "value": _serialize_context_variable_value(value)}
                for key, value in values
            ]
        }

    async def retrieve_value(self, params: dict[str, Any]) -> dict[str, Any]:
        variable_id, key = _parse_value_address(params)
        value = await self._context_variable_module.read_value(
            variable_id=variable_id,
            key=key,
        )
        return {"value": _serialize_context_variable_value(value) if value is not None else None}

    async def update_value(self, params: dict[str, Any]) -> dict[str, Any]:
        variable_id, key = _parse_value_address(params)
        if "data" not in params:
            raise ValueError("data is required")
        data = _require_json_serializable(params["data"], field="data")
        value = await self._context_variable_module.update_value(
            variable_id=variable_id,
            key=key,
            data=data,
        )
        return {"value": _serialize_context_variable_value(value)}

    async def delete_value(self, params: dict[str, Any]) -> dict[str, Any]:
        variable_id, key = _parse_value_address(params)
        value = await self._context_variable_module.read_value(
            variable_id=variable_id,
            key=key,
        )
        if value is None:
            raise ValueError(f"Value not found for variable '{variable_id}' and key '{key}'")
        await self._context_variable_module.delete_value(
            variable_id=variable_id,
            key=key,
        )
        return {}


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _require_nonempty_string(value: object, *, field: str) -> str:
    parsed = _require_string(value, field=field)
    if not parsed:
        raise ValueError(f"{field} must be a non-empty string")
    return parsed


def _parse_optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field=field)


def _parse_optional_nonempty_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, field=field)


def _require_context_variable_id(value: object, *, field: str) -> ContextVariableId:
    return ContextVariableId(_require_nonempty_string(value, field=field))


def _parse_optional_group_id(value: object, *, field: str) -> GroupId | None:
    if value is None:
        return None
    return GroupId(_require_string(value, field=field))


def _parse_string_list(value: object, *, field: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings or null")
    return list(value)


def _parse_group_ids(value: object, *, field: str) -> list[GroupId] | None:
    parsed = _parse_string_list(value, field=field)
    return [GroupId(group_id) for group_id in parsed] if parsed is not None else None


def _parse_optional_tool_id(value: object, *, field: str) -> ToolId | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a tool identifier object or null")
    service_name = _require_string(value.get("service_name"), field=f"{field}.service_name")
    tool_name = _require_string(value.get("tool_name"), field=f"{field}.tool_name")
    return ToolId(service_name=service_name, tool_name=tool_name)


def _parse_freshness_rules(value: object, *, field: str) -> str | None:
    parsed = _parse_optional_string(value, field=field)
    if parsed is None:
        return None
    try:
        croniter(parsed)
    except Exception as exc:
        raise ValueError(f"{field} contains an invalid cron expression") from exc
    return parsed


def _parse_group_update(value: object) -> ContextVariableTagsUpdateParams | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("groups must be an object or null")
    return ContextVariableTagsUpdateParams(
        add=_parse_group_ids(value.get("add"), field="groups.add"),
        remove=_parse_group_ids(value.get("remove"), field="groups.remove"),
    )


def _parse_value_address(params: dict[str, Any]) -> tuple[ContextVariableId, str]:
    return (
        _require_context_variable_id(params.get("variable_id"), field="variable_id"),
        _require_nonempty_string(params.get("key"), field="key"),
    )


def _require_json_serializable(value: object, *, field: str) -> JSONSerializable:
    if not _is_json_serializable(value):
        raise ValueError(f"{field} must be JSON serializable")
    return cast(JSONSerializable, value)


def _is_json_serializable(value: object) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_json_serializable(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_serializable(item) for key, item in value.items()
        )
    return False


def _serialize_context_variable(context_variable: ContextVariable) -> dict[str, Any]:
    return {
        "id": str(context_variable.id),
        "name": context_variable.name,
        "description": context_variable.description,
        "tool_id": (
            {
                "service_name": context_variable.tool_id.service_name,
                "tool_name": context_variable.tool_id.tool_name,
            }
            if context_variable.tool_id is not None
            else None
        ),
        "freshness_rules": context_variable.freshness_rules,
        "groups": [str(group_id) for group_id in context_variable.groups],
        "modified_utc": context_variable.modified_utc.isoformat(),
    }


def _serialize_context_variable_value(value: ContextVariableValue) -> dict[str, Any]:
    return {
        "id": str(value.id),
        "modified_utc": value.modified_utc.isoformat(),
        "data": value.data,
    }
