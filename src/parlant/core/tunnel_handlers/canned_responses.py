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

from parlant.core.app_modules.canned_responses import (
    CannedResponseGroupUpdateParamsModel,
    CannedResponseMetadataUpdateParamsModel,
    CannedResponseModule,
)
from parlant.core.canned_responses import CannedResponse, CannedResponseField, CannedResponseId
from parlant.core.common import ItemNotFoundError, JSONSerializable
from parlant.core.groups import GroupId
from parlant.core.tunnel_handlers.common import TunnelHandlerSet


class CannedResponseTunnelHandlers:
    def __init__(self, canned_response_module: CannedResponseModule) -> None:
        self._canned_response_module = canned_response_module

    def handler_set(self) -> TunnelHandlerSet:
        return TunnelHandlerSet(
            unary={
                "canned_responses.create": self.create,
                "canned_responses.list": self.list,
                "canned_responses.retrieve": self.retrieve,
                "canned_responses.update": self.update,
                "canned_responses.delete": self.delete,
            },
            streams={},
        )

    async def create(self, params: dict[str, Any]) -> dict[str, Any]:
        canned_response = await self._canned_response_module.create(
            value=_require_nonempty_string(params.get("value"), field="value"),
            fields=_parse_fields(params.get("fields"), field="fields") or [],
            signals=_parse_string_list(params.get("signals"), field="signals") or None,
            groups=_parse_group_ids(params.get("groups"), field="groups") or None,
            metadata=_parse_metadata(params.get("metadata"), field="metadata") or {},
            field_dependencies=_parse_string_list(
                params.get("field_dependencies"),
                field="field_dependencies",
            )
            or None,
            note=_parse_optional_string(
                params.get("note"),
                field="note",
            )
            or "",
        )
        return {"canned_response": _serialize_canned_response(canned_response)}

    async def list(self, params: dict[str, Any]) -> dict[str, Any]:
        groups = _require_group_ids(params["groups"], field="groups") if "groups" in params else []
        canned_responses = await self._canned_response_module.find(groups=groups)
        return {
            "canned_responses": [
                _serialize_canned_response(canned_response) for canned_response in canned_responses
            ]
        }

    async def retrieve(self, params: dict[str, Any]) -> dict[str, Any]:
        canned_response_id = _require_canned_response_id(
            params.get("canned_response_id"),
            field="canned_response_id",
        )
        try:
            canned_response = await self._canned_response_module.read(
                canned_response_id=canned_response_id
            )
        except ItemNotFoundError:
            return {"canned_response": None}
        return {"canned_response": _serialize_canned_response(canned_response)}

    async def update(self, params: dict[str, Any]) -> dict[str, Any]:
        canned_response_id = _require_canned_response_id(
            params.get("canned_response_id"),
            field="canned_response_id",
        )
        value = _parse_optional_nonempty_string(params.get("value"), field="value")
        fields = _parse_fields(params.get("fields"), field="fields") or []
        if fields and value is None:
            raise ValueError(
                "CannedResponse fields cannot be updated without providing a new value."
            )

        canned_response = await self._canned_response_module.update(
            canned_response_id=canned_response_id,
            value=value,
            fields=fields,
            groups=_parse_group_update(params.get("groups")),
            metadata=_parse_metadata_update(params.get("metadata")),
            note=_parse_optional_string(
                params.get("note"),
                field="note",
            ),
        )
        return {"canned_response": _serialize_canned_response(canned_response)}

    async def delete(self, params: dict[str, Any]) -> dict[str, Any]:
        await self._canned_response_module.delete(
            canned_response_id=_require_canned_response_id(
                params.get("canned_response_id"),
                field="canned_response_id",
            )
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


def _require_canned_response_id(value: object, *, field: str) -> CannedResponseId:
    return CannedResponseId(_require_nonempty_string(value, field=field))


def _parse_string_list(value: object, *, field: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings or null")
    return list(value)


def _require_group_ids(value: object, *, field: str) -> list[GroupId]:
    values = _parse_string_list(value, field=field)
    if values is None:
        raise ValueError(f"{field} must be a list of strings")
    return [GroupId(group_id) for group_id in values]


def _parse_group_ids(value: object, *, field: str) -> list[GroupId] | None:
    values = _parse_string_list(value, field=field)
    return [GroupId(group_id) for group_id in values] if values is not None else None


def _require_fields(value: object, *, field: str) -> list[CannedResponseField]:
    fields = _parse_fields(value, field=field)
    if fields is None:
        raise ValueError(f"{field} must be a list of canned response fields")
    return fields


def _parse_fields(value: object, *, field: str) -> list[CannedResponseField] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of canned response fields or null")

    fields: list[CannedResponseField] = []
    expected_keys = {"name", "description", "examples"}
    for index, item in enumerate(value):
        item_field = f"{field}[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{item_field} must contain exactly name, description, and examples")
        missing_keys = expected_keys - item.keys()
        if missing_keys:
            missing_key = sorted(missing_keys)[0]
            raise ValueError(f"{item_field}.{missing_key} is required")
        if item.keys() - expected_keys:
            raise ValueError(f"{item_field} must contain exactly name, description, and examples")
        fields.append(
            CannedResponseField(
                name=_require_nonempty_string(item.get("name"), field=f"{item_field}.name"),
                description=_require_string(
                    item.get("description"),
                    field=f"{item_field}.description",
                ),
                examples=_require_string_list(
                    item.get("examples"),
                    field=f"{item_field}.examples",
                ),
            )
        )
    return fields


def _require_string_list(value: object, *, field: str) -> list[str]:
    values = _parse_string_list(value, field=field)
    if values is None:
        raise ValueError(f"{field} must be a list of strings")
    return values


def _parse_metadata(
    value: object,
    *,
    field: str,
) -> dict[str, JSONSerializable] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object or null")
    if any(not _is_json_serializable(item) for item in value.values()):
        raise ValueError(f"{field} must be a JSON object or null")
    return dict(cast(dict[str, JSONSerializable], value))


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


def _parse_group_update(value: object) -> CannedResponseGroupUpdateParamsModel | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("groups must be an object or null")
    return CannedResponseGroupUpdateParamsModel(
        add=_parse_group_ids(value.get("add"), field="groups.add"),
        remove=_parse_group_ids(value.get("remove"), field="groups.remove"),
    )


def _parse_metadata_update(value: object) -> CannedResponseMetadataUpdateParamsModel | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object or null")
    return CannedResponseMetadataUpdateParamsModel(
        set=_parse_metadata(value.get("set"), field="metadata.set"),
        unset=_parse_string_list(value.get("unset"), field="metadata.unset"),
    )


def _serialize_canned_response(canned_response: CannedResponse) -> dict[str, Any]:
    return {
        "id": str(canned_response.id),
        "creation_utc": canned_response.creation_utc.isoformat(),
        "modified_utc": canned_response.modified_utc.isoformat(),
        "value": canned_response.value,
        "fields": [
            {
                "name": field.name,
                "description": field.description,
                "examples": list(field.examples),
            }
            for field in canned_response.fields
        ],
        "signals": list(canned_response.signals),
        "groups": [str(group_id) for group_id in canned_response.groups],
        "metadata": dict(canned_response.metadata),
        "field_dependencies": list(canned_response.field_dependencies),
        "note": canned_response.note,
    }
