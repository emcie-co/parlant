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

from parlant.core.app_modules.glossary import GlossaryModule, TermGroupsUpdateParamsModel
from parlant.core.common import ItemNotFoundError
from parlant.core.glossary import Term, TermId
from parlant.core.groups import GroupId
from parlant.core.tunnel_handlers.common import TunnelHandlerSet


class GlossaryTunnelHandlers:
    def __init__(self, glossary_module: GlossaryModule) -> None:
        self._glossary_module = glossary_module

    def handler_set(self) -> TunnelHandlerSet:
        return TunnelHandlerSet(
            unary={
                "terms.create": self.create,
                "terms.list": self.list,
                "terms.retrieve": self.retrieve,
                "terms.update": self.update,
                "terms.delete": self.delete,
            },
            streams={},
        )

    async def create(self, params: dict[str, Any]) -> dict[str, Any]:
        term = await self._glossary_module.create(
            id=_parse_optional_term_id(params.get("id"), field="id"),
            name=_require_term_name(params.get("name"), field="name"),
            description=_require_string(params.get("description"), field="description"),
            synonyms=_parse_create_synonyms(params),
            groups=_parse_group_ids(params.get("groups"), field="groups"),
        )
        return {"term": _serialize_term(term)}

    async def list(self, params: dict[str, Any]) -> dict[str, Any]:
        terms = await self._glossary_module.find(
            group_id=_parse_optional_group_id(params.get("group_id"), field="group_id")
        )
        return {"terms": [_serialize_term(term) for term in terms]}

    async def retrieve(self, params: dict[str, Any]) -> dict[str, Any]:
        term_id = _require_term_id(params.get("term_id"), field="term_id")
        try:
            term = await self._glossary_module.read(term_id=term_id)
        except ItemNotFoundError:
            return {"term": None}
        return {"term": _serialize_term(term)}

    async def update(self, params: dict[str, Any]) -> dict[str, Any]:
        term = await self._glossary_module.update(
            term_id=_require_term_id(params.get("term_id"), field="term_id"),
            name=_parse_optional_term_name(params.get("name"), field="name"),
            description=_parse_optional_string(params.get("description"), field="description"),
            synonyms=_parse_string_list(params.get("synonyms"), field="synonyms"),
            groups=_parse_group_update(params.get("groups")),
        )
        return {"term": _serialize_term(term)}

    async def delete(self, params: dict[str, Any]) -> dict[str, Any]:
        await self._glossary_module.delete(
            term_id=_require_term_id(params.get("term_id"), field="term_id")
        )
        return {}


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _parse_optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field=field)


def _require_term_name(value: object, *, field: str) -> str:
    name = _require_string(value, field=field)
    if not 1 <= len(name) <= 100:
        raise ValueError(f"{field} must contain between 1 and 100 characters")
    return name


def _parse_optional_term_name(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_term_name(value, field=field)


def _parse_optional_term_id(value: object, *, field: str) -> TermId | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return TermId(value)


def _require_term_id(value: object, *, field: str) -> TermId:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return TermId(value)


def _parse_optional_group_id(value: object, *, field: str) -> GroupId | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return GroupId(value)


def _parse_string_list(value: object, *, field: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings or null")
    return value


def _require_string_list(value: object, *, field: str) -> list[str]:
    values = _parse_string_list(value, field=field)
    if values is None:
        raise ValueError(f"{field} must be a list of strings")
    return values


def _parse_create_synonyms(params: dict[str, Any]) -> list[str]:
    if "synonyms" not in params:
        return []
    return _require_string_list(params["synonyms"], field="synonyms")


def _parse_group_ids(value: object, *, field: str) -> list[GroupId] | None:
    values = _parse_string_list(value, field=field)
    return [GroupId(item) for item in values] if values is not None else None


def _parse_group_update(value: object) -> TermGroupsUpdateParamsModel | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("groups must be an object or null")
    return TermGroupsUpdateParamsModel(
        add=_parse_group_ids(value.get("add"), field="groups.add"),
        remove=_parse_group_ids(value.get("remove"), field="groups.remove"),
    )


def _serialize_term(term: Term) -> dict[str, Any]:
    return {
        "id": str(term.id),
        "name": term.name,
        "description": term.description,
        "synonyms": list(term.synonyms),
        "groups": [str(group_id) for group_id in term.groups],
        "modified_utc": term.modified_utc.isoformat(),
    }
