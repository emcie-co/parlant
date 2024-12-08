# Copyright 2024 Emcie Co Ltd.
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

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import NewType, Optional, Sequence, cast
from typing_extensions import TypedDict, override, Self
from datetime import datetime, timezone
from dataclasses import dataclass

from parlant.core.common import (
    ItemNotFoundError,
    JSONSerializable,
    UniqueId,
    Version,
    generate_id,
)
from parlant.core.persistence.document_database import (
    DocumentCollection,
    DocumentDatabase,
    ObjectId,
)
from parlant.core.tools import ToolId

ContextVariableId = NewType("ContextVariableId", str)
ContextVariableValueId = NewType("ContextVariableValueId", str)


@dataclass(frozen=True)
class ContextVariable:
    id: ContextVariableId
    name: str
    description: Optional[str]
    tool_id: Optional[ToolId]
    freshness_rules: Optional[str]
    """If None, the variable will only be updated on session creation"""


@dataclass(frozen=True)
class ContextVariableValue:
    id: ContextVariableValueId
    last_modified: datetime
    data: JSONSerializable


class ContextVariableUpdateParams(TypedDict, total=False):
    name: str
    description: Optional[str]
    tool_id: Optional[ToolId]
    freshness_rules: Optional[str]


class ContextVariableStore(ABC):
    GLOBAL_KEY = "__default"

    @abstractmethod
    async def create_variable(
        self,
        variable_set: str,
        name: str,
        description: Optional[str] = None,
        tool_id: Optional[ToolId] = None,
        freshness_rules: Optional[str] = None,
    ) -> ContextVariable: ...

    @abstractmethod
    async def update_variable(
        self,
        variable_set: str,
        id: ContextVariableId,
        params: ContextVariableUpdateParams,
    ) -> ContextVariable: ...

    @abstractmethod
    async def delete_variable(
        self,
        variable_set: str,
        id: ContextVariableId,
    ) -> None: ...

    @abstractmethod
    async def list_variables(
        self,
        variable_set: str,
    ) -> Sequence[ContextVariable]: ...

    @abstractmethod
    async def read_variable(
        self,
        variable_set: str,
        id: ContextVariableId,
    ) -> ContextVariable: ...

    @abstractmethod
    async def update_value(
        self,
        variable_set: str,
        variable_id: ContextVariableId,
        key: str,
        data: JSONSerializable,
    ) -> ContextVariableValue: ...

    @abstractmethod
    async def read_value(
        self,
        variable_set: str,
        variable_id: ContextVariableId,
        key: str,
    ) -> Optional[ContextVariableValue]: ...

    @abstractmethod
    async def delete_value(
        self,
        variable_set: str,
        variable_id: ContextVariableId,
        key: str,
    ) -> None: ...

    @abstractmethod
    async def list_values(
        self,
        variable_set: str,
        variable_id: ContextVariableId,
    ) -> Sequence[tuple[str, ContextVariableValue]]: ...


class _ContextVariableDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    variable_set: str
    name: str
    description: Optional[str]
    tool_id: Optional[str]
    freshness_rules: Optional[str]


class _ContextVariableValueDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    last_modified: str
    variable_set: str
    variable_id: ContextVariableId
    key: str
    data: JSONSerializable


class ContextVariableDocumentStore(ContextVariableStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(self, database: DocumentDatabase):
        self._database = database
        self._variable_collection: DocumentCollection[_ContextVariableDocument]
        self._value_collection: DocumentCollection[_ContextVariableValueDocument]

    async def __aenter__(self) -> Self:
        self._variable_collection = await self._database.get_or_create_collection(
            name="variables",
            schema=_ContextVariableDocument,
        )

        self._value_collection = await self._database.get_or_create_collection(
            name="values",
            schema=_ContextVariableValueDocument,
        )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    def _serialize_context_variable(
        self,
        context_variable: ContextVariable,
        variable_set: str,
    ) -> _ContextVariableDocument:
        return _ContextVariableDocument(
            id=ObjectId(context_variable.id),
            version=self.VERSION.to_string(),
            variable_set=variable_set,
            name=context_variable.name,
            description=context_variable.description,
            tool_id=context_variable.tool_id.to_string() if context_variable.tool_id else None,
            freshness_rules=context_variable.freshness_rules,
        )

    def _serialize_context_variable_value(
        self,
        context_variable_value: ContextVariableValue,
        variable_set: str,
        variable_id: ContextVariableId,
        key: str,
    ) -> _ContextVariableValueDocument:
        return _ContextVariableValueDocument(
            id=ObjectId(context_variable_value.id),
            version=self.VERSION.to_string(),
            last_modified=context_variable_value.last_modified.isoformat(),
            variable_set=variable_set,
            variable_id=variable_id,
            key=key,
            data=context_variable_value.data,
        )

    def _deserialize_context_variable(
        self,
        context_variable_document: _ContextVariableDocument,
    ) -> ContextVariable:
        return ContextVariable(
            id=ContextVariableId(context_variable_document["id"]),
            name=context_variable_document["name"],
            description=context_variable_document.get("description"),
            tool_id=ToolId.from_string(context_variable_document["tool_id"])
            if context_variable_document["tool_id"]
            else None,
            freshness_rules=context_variable_document["freshness_rules"],
        )

    def _deserialize_context_variable_value(
        self,
        context_variable_value_document: _ContextVariableValueDocument,
    ) -> ContextVariableValue:
        return ContextVariableValue(
            id=ContextVariableValueId(context_variable_value_document["id"]),
            last_modified=datetime.fromisoformat(context_variable_value_document["last_modified"]),
            data=context_variable_value_document["data"],
        )

    @override
    async def create_variable(
        self,
        variable_set: str,
        name: str,
        description: Optional[str] = None,
        tool_id: Optional[ToolId] = None,
        freshness_rules: Optional[str] = None,
    ) -> ContextVariable:
        context_variable = ContextVariable(
            id=ContextVariableId(generate_id()),
            name=name,
            description=description,
            tool_id=tool_id,
            freshness_rules=freshness_rules,
        )

        await self._variable_collection.insert_one(
            self._serialize_context_variable(context_variable, variable_set)
        )

        return context_variable

    @override
    async def update_variable(
        self,
        variable_set: str,
        id: ContextVariableId,
        params: ContextVariableUpdateParams,
    ) -> ContextVariable:
        variable_document = await self._variable_collection.find_one(
            filters={
                "id": {"$eq": id},
                "variable_set": {"$eq": variable_set},
            }
        )

        if not variable_document:
            raise ItemNotFoundError(item_id=UniqueId(id), message=f"variable_set={variable_set}")

        update_params = {
            **({"name": params["name"]} if "name" in params else {}),
            **({"description": params["description"]} if "description" in params else {}),
            **(
                {"tool_id": params["tool_id"].to_string()}
                if "tool_id" in params and params["tool_id"]
                else {}
            ),
            **(
                {
                    "freshness_rules": params["freshness_rules"]
                    if "freshness_rules" in params and params["freshness_rules"]
                    else {}
                }
            ),
        }

        result = await self._variable_collection.update_one(
            filters={
                "id": {"$eq": id},
                "variable_set": {"$eq": variable_set},
            },
            params=cast(_ContextVariableDocument, update_params),
        )

        assert result.updated_document

        return self._deserialize_context_variable(context_variable_document=result.updated_document)

    @override
    async def delete_variable(
        self,
        variable_set: str,
        id: ContextVariableId,
    ) -> None:
        variable_deletion_result = await self._variable_collection.delete_one(
            {
                "id": {"$eq": id},
                "variable_set": {"$eq": variable_set},
            }
        )
        if variable_deletion_result.deleted_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(id), message=f"variable_set={variable_set}")

        for k, _ in await self.list_values(variable_set=variable_set, variable_id=id):
            await self.delete_value(variable_set=variable_set, variable_id=id, key=k)

    @override
    async def list_variables(
        self,
        variable_set: str,
    ) -> Sequence[ContextVariable]:
        return [
            self._deserialize_context_variable(d)
            for d in await self._variable_collection.find({"variable_set": {"$eq": variable_set}})
        ]

    @override
    async def read_variable(
        self,
        variable_set: str,
        id: ContextVariableId,
    ) -> ContextVariable:
        variable_document = await self._variable_collection.find_one(
            {
                "variable_set": {"$eq": variable_set},
                "id": {"$eq": id},
            }
        )
        if not variable_document:
            raise ItemNotFoundError(
                item_id=UniqueId(id),
                message=f"variable_set={variable_set}",
            )

        return self._deserialize_context_variable(variable_document)

    @override
    async def update_value(
        self,
        variable_set: str,
        variable_id: ContextVariableId,
        key: str,
        data: JSONSerializable,
    ) -> ContextVariableValue:
        last_modified = datetime.now(timezone.utc)

        value = ContextVariableValue(
            id=ContextVariableValueId(generate_id()),
            last_modified=last_modified,
            data=data,
        )

        result = await self._value_collection.update_one(
            {
                "variable_set": {"$eq": variable_set},
                "variable_id": {"$eq": variable_id},
                "key": {"$eq": key},
            },
            self._serialize_context_variable_value(
                context_variable_value=value,
                variable_set=variable_set,
                variable_id=variable_id,
                key=key,
            ),
            upsert=True,
        )

        assert result.updated_document

        return value

    @override
    async def read_value(
        self,
        variable_set: str,
        variable_id: ContextVariableId,
        key: str,
    ) -> Optional[ContextVariableValue]:
        value_document = await self._value_collection.find_one(
            {
                "variable_set": {"$eq": variable_set},
                "variable_id": {"$eq": variable_id},
                "key": {"$eq": key},
            }
        )

        if not value_document:
            return None

        return self._deserialize_context_variable_value(value_document)

    @override
    async def delete_value(
        self,
        variable_set: str,
        variable_id: ContextVariableId,
        key: str,
    ) -> None:
        await self._value_collection.delete_one(
            {
                "variable_set": {"$eq": variable_set},
                "variable_id": {"$eq": variable_id},
                "key": {"$eq": key},
            }
        )

    @override
    async def list_values(
        self,
        variable_set: str,
        variable_id: ContextVariableId,
    ) -> Sequence[tuple[str, ContextVariableValue]]:
        return [
            (d["key"], self._deserialize_context_variable_value(d))
            for d in await self._value_collection.find(
                {
                    "variable_set": {"$eq": variable_set},
                    "variable_id": {"$eq": variable_id},
                }
            )
        ]
