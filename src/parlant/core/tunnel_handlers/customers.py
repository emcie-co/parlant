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

from datetime import datetime
from typing import Any

from parlant.core.app_modules.common import decode_cursor, encode_cursor
from parlant.core.app_modules.customers import (
    CustomerGroupUpdateParams,
    CustomerMetadataUpdateParams,
    CustomerModule,
)
from parlant.core.common import ItemNotFoundError
from parlant.core.customers import Customer, CustomerId, CustomerKind
from parlant.core.groups import GroupId
from parlant.core.persistence.common import Cursor, SortDirection
from parlant.core.tunnel_handlers.common import TunnelHandlerSet
from parlant.core.tunnel_handlers.sessions import _parse_sort_direction


class CustomerTunnelHandlers:
    def __init__(self, customer_module: CustomerModule) -> None:
        self._customer_module = customer_module

    def handler_set(self) -> TunnelHandlerSet:
        return TunnelHandlerSet(
            unary={
                "customers.create": self.create,
                "customers.list": self.list,
                "customers.retrieve": self.retrieve,
                "customers.update": self.update,
                "customers.delete": self.delete,
            },
            streams={},
        )

    async def create(self, params: dict[str, Any]) -> dict[str, Any]:
        customer = await self._customer_module.create(
            name=_require_customer_name(params.get("name")),
            kind=_parse_customer_kind(params.get("kind")) or CustomerKind.REAL,
            metadata=_parse_metadata(params.get("metadata"), field="metadata") or {},
            groups=_parse_group_ids(params.get("groups"), field="groups"),
            id=_parse_customer_id(params.get("id"), field="id"),
        )
        return {"customer": _serialize_customer(customer)}

    async def list(self, params: dict[str, Any]) -> dict[str, Any]:
        result = await self._customer_module.find(
            cursor=_parse_cursor(params.get("cursor")),
            limit=_parse_limit(params.get("limit")),
            sort_direction=_parse_customer_sort_direction(params.get("sort_direction")),
            min_modified_utc=_parse_min_modified_utc(params.get("min_modified_utc")),
            name_prefix=_parse_name_prefix(params.get("name_prefix")),
            kind=_parse_customer_kind(params.get("kind")),
        )
        return {
            "customers": [_serialize_customer(customer) for customer in result.items],
            "total_count": result.total_count,
            "has_more": result.has_more,
            "next_cursor": encode_cursor(result.next_cursor) if result.next_cursor else None,
        }

    async def retrieve(self, params: dict[str, Any]) -> dict[str, Any]:
        customer_id = _require_customer_id(params.get("customer_id"), field="customer_id")
        try:
            customer = await self._customer_module.read(customer_id=customer_id)
        except ItemNotFoundError:
            return {"customer": None}
        return {"customer": _serialize_customer(customer)}

    async def update(self, params: dict[str, Any]) -> dict[str, Any]:
        customer = await self._customer_module.update(
            customer_id=_require_customer_id(params.get("customer_id"), field="customer_id"),
            name=_parse_customer_name(params.get("name")),
            metadata=_parse_metadata_update(params.get("metadata")),
            groups=_parse_group_update(params.get("groups")),
        )
        return {"customer": _serialize_customer(customer)}

    async def delete(self, params: dict[str, Any]) -> dict[str, Any]:
        customer_id = _require_customer_id(params.get("customer_id"), field="customer_id")
        await self._customer_module.delete(customer_id=customer_id)
        return {}


def _parse_customer_kind(value: object) -> CustomerKind | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("kind must be real or persona")
    try:
        return CustomerKind(value)
    except ValueError:
        raise ValueError("kind must be real or persona")


def _parse_name_prefix(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("name_prefix must be a non-empty string")
    return value


def _require_customer_name(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 100:
        raise ValueError("Customer name must be a string between 1 and 100 characters")
    return value


def _parse_customer_name(value: object) -> str | None:
    if value is None:
        return None
    return _require_customer_name(value)


def _parse_customer_id(value: object, *, field: str) -> CustomerId | None:
    if value is None:
        return None
    return _require_customer_id(value, field=field)


def _require_customer_id(value: object, *, field: str) -> CustomerId:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return CustomerId(value)


def _parse_metadata(value: object, *, field: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object of string values or null")

    metadata: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"{field} must be an object of string values or null")
        metadata[key] = item
    return metadata


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


def _parse_metadata_update(value: object) -> CustomerMetadataUpdateParams | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object or null")
    return CustomerMetadataUpdateParams(
        set=_parse_metadata(value.get("set"), field="metadata.set"),
        unset=_parse_string_list(value.get("unset"), field="metadata.unset"),
    )


def _parse_group_update(value: object) -> CustomerGroupUpdateParams | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("groups must be an object or null")
    return CustomerGroupUpdateParams(
        add=_parse_group_ids(value.get("add"), field="groups.add"),
        remove=_parse_group_ids(value.get("remove"), field="groups.remove"),
    )


def _parse_limit(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    return value


def _parse_cursor(value: object) -> Cursor | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("cursor must be a string or null")
    return decode_cursor(value)


def _parse_min_modified_utc(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("min_modified_utc must be an ISO-8601 string or null")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("min_modified_utc must be an ISO-8601 string or null")


def _parse_customer_sort_direction(value: object) -> SortDirection | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Unsupported sort direction: {value}")
    return _parse_sort_direction(value)


def _serialize_customer(customer: Customer) -> dict[str, Any]:
    return {
        "id": str(customer.id),
        "creation_utc": customer.creation_utc.isoformat(),
        "modified_utc": customer.modified_utc.isoformat(),
        "name": customer.name,
        "kind": customer.kind.value,
        "metadata": dict(customer.metadata),
        "groups": [str(group_id) for group_id in customer.groups],
    }
