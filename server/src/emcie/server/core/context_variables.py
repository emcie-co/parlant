from abc import ABC, abstractmethod
from typing import Any, Literal, NewType, Optional, Sequence
from datetime import datetime, timezone
from dataclasses import dataclass

from emcie.server.base_models import DefaultBaseModel
from emcie.server.core import common
from emcie.server.core.tools import ToolId
from emcie.server.core.persistence.document_database import DocumentDatabase

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
    tool_id: ToolId
    freshness_rules: Optional[FreshnessRules]
    """If None, the variable will only be updated on session creation"""


@dataclass(frozen=True)
class ContextVariableValue:
    id: ContextVariableValueId
    variable_id: ContextVariableId
    last_modified: datetime
    data: common.JSONSerializable


class ContextVariableStore(ABC):
    @abstractmethod
    async def create_variable(
        self,
        variable_set: str,
        name: str,
        description: Optional[str],
        tool_id: ToolId,
        freshness_rules: Optional[FreshnessRules],
    ) -> ContextVariable: ...

    @abstractmethod
    async def update_value(
        self,
        variable_set: str,
        key: str,
        variable_id: ContextVariableId,
        data: common.JSONSerializable,
    ) -> ContextVariableValue: ...

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
    async def read_value(
        self,
        variable_set: str,
        key: str,
        variable_id: ContextVariableId,
    ) -> ContextVariableValue: ...


class ContextVariableDocumentStore(ContextVariableStore):
    class ContextVariableDocument(DefaultBaseModel):
        id: str
        variable_set: str
        name: str
        description: Optional[str] = None
        tool_id: ToolId
        freshness_rules: Optional[FreshnessRules]

    class ContextVariableValueDocument(DefaultBaseModel):
        id: str
        last_modified: datetime
        variable_set: str
        variable_id: ContextVariableId
        key: str
        data: dict[str, Any]

    def __init__(self, database: DocumentDatabase):
        self._variable_collection = database.get_or_create_collection(
            name="variables",
            schema=self.ContextVariableDocument,
        )
        self._value_collection = database.get_or_create_collection(
            name="values",
            schema=self.ContextVariableValueDocument,
        )

    async def create_variable(
        self,
        variable_set: str,
        name: str,
        description: Optional[str],
        tool_id: ToolId,
        freshness_rules: Optional[FreshnessRules],
    ) -> ContextVariable:
        variable_id = ContextVariableId(common.generate_id())

        await self._variable_collection.insert_one(
            {
                "id": variable_id,
                "variable_set": variable_set,
                "name": name,
                "description": description,
                "tool_id": tool_id,
                "freshness_rules": freshness_rules,
            },
        )
        return ContextVariable(
            id=ContextVariableId(variable_id),
            name=name,
            description=description,
            tool_id=tool_id,
            freshness_rules=freshness_rules,
        )

    async def update_value(
        self,
        variable_set: str,
        key: str,
        variable_id: ContextVariableId,
        data: common.JSONSerializable,
    ) -> ContextVariableValue:
        last_modified = datetime.now(timezone.utc)
        value_document_id = await self._value_collection.update_one(
            {
                "variable_set": {"$eq": variable_set},
                "variable_id": {"$eq": variable_id},
                "key": {"$eq": key},
            },
            {
                "id": common.generate_id(),
                "variable_set": variable_set,
                "variable_id": variable_id,
                "last_modified": last_modified,
                "data": data,
                "key": key,
            },
            upsert=True,
        )
        return ContextVariableValue(
            id=ContextVariableValueId(value_document_id),
            variable_id=variable_id,
            last_modified=last_modified,
            data=data,
        )

    async def delete_variable(
        self,
        variable_set: str,
        id: ContextVariableId,
    ) -> None:
        await self._variable_collection.delete_one(
            {
                "id": {"$eq": id},
                "variable_set": {"$eq": variable_set},
            }
        )
        await self._value_collection.delete_one(
            {
                "variable_id": {"$eq": id},
                "variable_set": {"$eq": variable_set},
            }
        )

    async def list_variables(
        self,
        variable_set: str,
    ) -> Sequence[ContextVariable]:
        return [
            ContextVariable(
                id=ContextVariableId(d["id"]),
                name=d["name"],
                description=d["description"],
                tool_id=d["tool_id"],
                freshness_rules=d["freshness_rules"],
            )
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
        return ContextVariable(
            id=ContextVariableId(variable_document["id"]),
            name=variable_document["name"],
            description=variable_document["description"],
            tool_id=variable_document["tool_id"],
            freshness_rules=variable_document["freshness_rules"],
        )

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
        return ContextVariableValue(
            id=ContextVariableValueId(value_document["id"]),
            variable_id=value_document["variable_id"],
            last_modified=value_document["last_modified"],
            data=value_document["data"],
        )
