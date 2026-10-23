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

from typing import Any

from parlant.core.app_modules.groups import GroupModule
from parlant.core.common import ItemNotFoundError
from parlant.core.groups import Group, GroupId, GroupUpdateParams
from parlant.core.tunnel_handlers.common import TunnelHandlerSet


class GroupTunnelHandlers:
    def __init__(self, group_module: GroupModule) -> None:
        self._group_module = group_module

    def handler_set(self) -> TunnelHandlerSet:
        return TunnelHandlerSet(
            unary={
                "groups.create": self.create,
                "groups.list": self.list,
                "groups.retrieve": self.retrieve,
                "groups.update": self.update,
                "groups.delete": self.delete,
            },
            streams={},
        )

    async def create(self, params: dict[str, Any]) -> dict[str, Any]:
        group = await self._group_module.create(name=_require_group_name(params.get("name")))
        return {"group": serialize_group(group)}

    async def list(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self._list(params=params, envelope="groups")

    async def retrieve(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self._retrieve(
            params=params,
            id_field="group_id",
            envelope="group",
        )

    async def update(self, params: dict[str, Any]) -> dict[str, Any]:
        group = await self._group_module.update(
            group_id=_require_group_id(params.get("group_id"), field="group_id"),
            params=GroupUpdateParams(name=_require_group_name(params.get("name"))),
        )
        return {"group": serialize_group(group)}

    async def delete(self, params: dict[str, Any]) -> dict[str, Any]:
        group_id = _require_group_id(params.get("group_id"), field="group_id")
        await self._group_module.delete(group_id=group_id)
        return {}

    async def _list(self, params: dict[str, Any], *, envelope: str) -> dict[str, Any]:
        groups = await self._group_module.find(name=_parse_name_filter(params.get("name")))
        return {envelope: [serialize_group(group) for group in groups]}

    async def _retrieve(
        self,
        params: dict[str, Any],
        *,
        id_field: str,
        envelope: str,
    ) -> dict[str, Any]:
        group_id = _require_group_id(params.get(id_field), field=id_field)
        try:
            group = await self._group_module.read(group_id=group_id)
        except ItemNotFoundError:
            return {envelope: None}
        return {envelope: serialize_group(group)}


def _require_group_name(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 50:
        raise ValueError("Group name must be a string between 1 and 50 characters")
    return value


def _parse_name_filter(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("name must be a string or null")
    return value


def _require_group_id(value: object, *, field: str) -> GroupId:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return GroupId(value)


def serialize_group(group: Group) -> dict[str, Any]:
    return {
        "id": str(group.id),
        "name": group.name,
    }
