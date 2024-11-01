from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal, NewType, Optional, Sequence, TypedDict
from datetime import datetime, timezone
from dataclasses import dataclass

from emcie.server.core.common import (
    ItemNotFoundError,
    JSONSerializable,
    UniqueId,
    Version,
    generate_id,
)
from emcie.server.core.persistence.document_database import (
    DocumentDatabase,
    ObjectId,
)
from emcie.server.core.tools import ToolId

ContextVariableId = NewType("ContextVariableId", str)
ContextVariableValueId = NewType("ContextVariableValueId", str)


@dataclass(frozen=True)
class FreshnessRules:
    """
    A data class representing the times at which the context variable should be considered fresh.
    """

    months: Optional[list[int]]
    days_of_month: Optional[list[int]]
    days_of_week: Optional[
        list[
            Literal[
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
            ]
        ]
    ]
    hours: Optional[list[int]]
    minutes: Optional[list[int]]
    seconds: Optional[list[int]]


@dataclass(frozen=True)
class ContextVariable:
    id: ContextVariableId
    name: str
    description: Optional[str]
    tool_id: Optional[ToolId]
    freshness_rules: Optional[FreshnessRules]
    """If None, the variable will only be updated on session creation"""


@dataclass(frozen=True)
class ContextVariableValue:
    id: ContextVariableValueId
    last_modified: datetime
    data: JSONSerializable


class ContextVariableStore(ABC):
    @abstractmethod
    async def create_variable(
        self,
        variable_set: str,
        name: str,
        description: Optional[str],
        tool_id: Optional[ToolId] = None,
        freshness_rules: Optional[FreshnessRules] = None,
    ) -> ContextVariable: ...

    @abstractmethod
    async def update_value(
        self,
        variable_set: str,
        key: str,
        variable_id: ContextVariableId,
        data: JSONSerializable,
    ) -> ContextVariableValue: ...

    @abstractmethod
    async def delete_variable(
        self,
        variable_set: str,
        id: ContextVariableId,
    ) -> Optional[ContextVariableId]: ...

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
    async def read_value(
        self,
        variable_set: str,
        key: str,
        variable_id: ContextVariableId,
    ) -> ContextVariableValue: ...

    @abstractmethod
    async def delete_value(
        self,
        variable_set: str,
        variable_id: ContextVariableId,
        key: str,
    ) -> Optional[ContextVariableValueId]: ...

    @abstractmethod
    async def list_values(
        self,
        variable_set: str,
        variable_id: ContextVariableId,
    ) -> Sequence[tuple[str, ContextVariableValue]]: ...


class _FreshnessRulesDocument(TypedDict, total=False):
    months: Optional[list[int]]
    days_of_month: Optional[list[int]]
    days_of_week: Optional[
        list[
            Literal[
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
            ]
        ]
    ]
    hours: Optional[list[int]]
    minutes: Optional[list[int]]
    seconds: Optional[list[int]]


class _ContextVariableDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    variable_set: str
    name: str
    description: Optional[str]
    tool_id: Optional[str]
    freshness_rules: Optional[_FreshnessRulesDocument]


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
        self._variable_collection = database.get_or_create_collection(
            name="variables",
            schema=_ContextVariableDocument,
        )

        self._value_collection = database.get_or_create_collection(
            name="values",
            schema=_ContextVariableValueDocument,
        )

    def _serialize_freshness_rules(
        self, freshness_rules: FreshnessRules
    ) -> _FreshnessRulesDocument:
        return _FreshnessRulesDocument(
            months=freshness_rules.months,
            days_of_month=freshness_rules.days_of_month,
            days_of_week=freshness_rules.days_of_week,
            hours=freshness_rules.hours,
            minutes=freshness_rules.minutes,
            seconds=freshness_rules.seconds,
        )

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
            freshness_rules=self._serialize_freshness_rules(context_variable.freshness_rules)
            if context_variable.freshness_rules
            else None,
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

    def _deserialize_freshness_rules(
        self,
        freshness_rules_document: _FreshnessRulesDocument,
    ) -> FreshnessRules:
        return FreshnessRules(
            months=freshness_rules_document["months"],
            days_of_month=freshness_rules_document["days_of_month"],
            days_of_week=freshness_rules_document["days_of_week"],
            hours=freshness_rules_document["hours"],
            minutes=freshness_rules_document["minutes"],
            seconds=freshness_rules_document["seconds"],
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
            freshness_rules=self._deserialize_freshness_rules(
                context_variable_document["freshness_rules"]
            )
            if context_variable_document["freshness_rules"]
            else None,
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

    async def create_variable(
        self,
        variable_set: str,
        name: str,
        description: Optional[str],
        tool_id: Optional[ToolId] = None,
        freshness_rules: Optional[FreshnessRules] = None,
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

    async def update_value(
        self,
        variable_set: str,
        key: str,
        variable_id: ContextVariableId,
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

    async def delete_variable(
        self,
        variable_set: str,
        id: ContextVariableId,
    ) -> Optional[ContextVariableId]:
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

        return (
            ContextVariableId(variable_deletion_result.deleted_document["id"])
            if variable_deletion_result.deleted_document
            else None
        )

    async def list_variables(
        self,
        variable_set: str,
    ) -> Sequence[ContextVariable]:
        return [
            self._deserialize_context_variable(d)
            for d in await self._variable_collection.find({"variable_set": {"$eq": variable_set}})
        ]

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

    async def read_value(
        self,
        variable_set: str,
        key: str,
        variable_id: ContextVariableId,
    ) -> ContextVariableValue:
        value_document = await self._value_collection.find_one(
            {
                "variable_set": {"$eq": variable_set},
                "variable_id": {"$eq": variable_id},
                "key": {"$eq": key},
            }
        )
        if not value_document:
            raise ItemNotFoundError(
                item_id=UniqueId(variable_id),
                message=f"variable_set={variable_set}, key={key}",
            )

        return self._deserialize_context_variable_value(value_document)

    async def delete_value(
        self,
        variable_set: str,
        variable_id: ContextVariableId,
        key: str,
    ) -> Optional[ContextVariableValueId]:
        result = await self._value_collection.delete_one(
            {
                "variable_set": {"$eq": variable_set},
                "variable_id": {"$eq": variable_id},
                "key": {"$eq": key},
            }
        )

        return (
            ContextVariableValueId(result.deleted_document["id"])
            if result.deleted_document
            else None
        )

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
