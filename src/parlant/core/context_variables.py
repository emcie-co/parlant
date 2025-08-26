# Copyright 2025 Emcie Co Ltd.
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

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import (
    ItemNotFoundError,
    JSONSerializable,
    UniqueId,
    Version,
    IdGenerator,
    md5_checksum,
)
from parlant.core.persistence.common import ObjectId, Where
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
)
from parlant.core.persistence.document_database_helper import (
    DocumentMigrationHelper,
    DocumentStoreMigrationHelper,
)
from parlant.core.tags import TagId
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
    tags: Sequence[TagId]
    """If None, the variable will only be updated on session creation"""

    def __hash__(self) -> int:
        return hash(self.id)


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
    GLOBAL_KEY = "DEFAULT"

    @abstractmethod
    async def create_variable(
        self,
        name: str,
        description: Optional[str] = None,
        tool_id: Optional[ToolId] = None,
        freshness_rules: Optional[str] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> ContextVariable: ...

    @abstractmethod
    async def update_variable(
        self,
        id: ContextVariableId,
        params: ContextVariableUpdateParams,
    ) -> ContextVariable: ...

    @abstractmethod
    async def delete_variable(
        self,
        id: ContextVariableId,
    ) -> None: ...

    @abstractmethod
    async def list_variables(
        self,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Sequence[ContextVariable]: ...

    @abstractmethod
    async def read_variable(
        self,
        id: ContextVariableId,
    ) -> ContextVariable: ...

    @abstractmethod
    async def update_value(
        self,
        variable_id: ContextVariableId,
        key: str,
        data: JSONSerializable,
    ) -> ContextVariableValue: ...

    @abstractmethod
    async def read_value(
        self,
        variable_id: ContextVariableId,
        key: str,
    ) -> Optional[ContextVariableValue]: ...

    @abstractmethod
    async def delete_value(
        self,
        variable_id: ContextVariableId,
        key: str,
    ) -> None: ...

    @abstractmethod
    async def list_values(
        self,
        variable_id: ContextVariableId,
    ) -> Sequence[tuple[str, ContextVariableValue]]: ...

    @abstractmethod
    async def add_variable_tag(
        self,
        variable_id: ContextVariableId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> ContextVariable: ...

    @abstractmethod
    async def remove_variable_tag(
        self,
        variable_id: ContextVariableId,
        tag_id: TagId,
    ) -> ContextVariable: ...


class ContextVariableDocument_v0_1_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    variable_set: str
    name: str
    description: Optional[str]
    tool_id: Optional[str]
    freshness_rules: Optional[str]


class _ContextVariableDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    name: str
    description: Optional[str]
    tool_id: Optional[str]
    freshness_rules: Optional[str]


class _ContextVariableValueDocument_v0_1_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    last_modified: str
    variable_set: str
    variable_id: ContextVariableId
    key: str
    data: JSONSerializable


class _ContextVariableValueDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    last_modified: str
    variable_id: ContextVariableId
    key: str
    data: JSONSerializable


class ContextVariableTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    variable_id: ContextVariableId
    tag_id: TagId


class ContextVariableDocumentStore(ContextVariableStore):
    VERSION = Version.from_string("0.2.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        database: DocumentDatabase,
        allow_migration: bool = False,
    ):
        self._id_generator = id_generator

        self._database = database
        self._variable_collection: DocumentCollection[_ContextVariableDocument]
        self._variable_tag_association_collection: DocumentCollection[
            ContextVariableTagAssociationDocument
        ]
        self._value_collection: DocumentCollection[_ContextVariableValueDocument]
        self._allow_migration = allow_migration

        self._lock = ReaderWriterLock()

    async def _variable_document_loader(
        self, doc: BaseDocument
    ) -> Optional[_ContextVariableDocument]:
        async def v0_1_0_to_v0_2_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        return await DocumentMigrationHelper[_ContextVariableDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_2_0,
            },
        ).migrate(doc)

    async def _value_document_loader(
        self, doc: BaseDocument
    ) -> Optional[_ContextVariableValueDocument]:
        async def v0_1_0_to_v0_2_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(_ContextVariableValueDocument_v0_1_0, doc)
            return _ContextVariableValueDocument(
                id=d["id"],
                version=Version.String("0.2.0"),
                last_modified=d["last_modified"],
                variable_id=d["variable_id"],
                key=d["key"],
                data=d["data"],
            )

        return await DocumentMigrationHelper[_ContextVariableValueDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_2_0,
            },
        ).migrate(doc)

    async def _variable_tag_association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[ContextVariableTagAssociationDocument]:
        if doc["version"] == "0.1.0":
            doc = cast(ContextVariableTagAssociationDocument, doc)
            return ContextVariableTagAssociationDocument(
                id=doc["id"],
                version=Version.String("0.2.0"),
                creation_utc=doc["creation_utc"],
                variable_id=doc["variable_id"],
                tag_id=doc["tag_id"],
            )

        return cast(ContextVariableTagAssociationDocument, doc)

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
        ):
            self._variable_collection = await self._database.get_or_create_collection(
                name="variables",
                schema=_ContextVariableDocument,
                document_loader=self._variable_document_loader,
            )

            self._variable_tag_association_collection = (
                await self._database.get_or_create_collection(
                    name="variable_tag_associations",
                    schema=ContextVariableTagAssociationDocument,
                    document_loader=self._variable_tag_association_document_loader,
                )
            )

            self._value_collection = await self._database.get_or_create_collection(
                name="values",
                schema=_ContextVariableValueDocument,
                document_loader=self._value_document_loader,
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
    ) -> _ContextVariableDocument:
        return _ContextVariableDocument(
            id=ObjectId(context_variable.id),
            version=self.VERSION.to_string(),
            name=context_variable.name,
            description=context_variable.description,
            tool_id=context_variable.tool_id.to_string() if context_variable.tool_id else None,
            freshness_rules=context_variable.freshness_rules,
        )

    def _serialize_context_variable_value(
        self,
        context_variable_value: ContextVariableValue,
        variable_id: ContextVariableId,
        key: str,
    ) -> _ContextVariableValueDocument:
        return _ContextVariableValueDocument(
            id=ObjectId(context_variable_value.id),
            version=self.VERSION.to_string(),
            last_modified=context_variable_value.last_modified.isoformat(),
            variable_id=variable_id,
            key=key,
            data=context_variable_value.data,
        )

    async def _deserialize_context_variable(
        self,
        context_variable_document: _ContextVariableDocument,
    ) -> ContextVariable:
        tags = [
            d["tag_id"]
            for d in await self._variable_tag_association_collection.find(
                {"variable_id": {"$eq": context_variable_document["id"]}}
            )
        ]

        return ContextVariable(
            id=ContextVariableId(context_variable_document["id"]),
            name=context_variable_document["name"],
            description=context_variable_document.get("description"),
            tool_id=ToolId.from_string(context_variable_document["tool_id"])
            if context_variable_document["tool_id"]
            else None,
            freshness_rules=context_variable_document["freshness_rules"],
            tags=tags,
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
        name: str,
        description: Optional[str] = None,
        tool_id: Optional[ToolId] = None,
        freshness_rules: Optional[str] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> ContextVariable:
        async with self._lock.writer_lock:
            context_variable_checksum = md5_checksum(
                f"{name}{description}{tool_id}{freshness_rules}{tags}"
            )

            context_variable = ContextVariable(
                id=ContextVariableId(self._id_generator.generate(context_variable_checksum)),
                name=name,
                description=description,
                tool_id=tool_id,
                freshness_rules=freshness_rules,
                tags=tags or [],
            )

            await self._variable_collection.insert_one(
                self._serialize_context_variable(context_variable)
            )

            for tag_id in tags or []:
                tag_checksum = md5_checksum(f"{context_variable.id}{tag_id}")

                await self._variable_tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(self._id_generator.generate(tag_checksum)),
                        "version": self.VERSION.to_string(),
                        "creation_utc": datetime.now(timezone.utc).isoformat(),
                        "variable_id": context_variable.id,
                        "tag_id": tag_id,
                    }
                )

        return context_variable

    @override
    async def update_variable(
        self,
        id: ContextVariableId,
        params: ContextVariableUpdateParams,
    ) -> ContextVariable:
        async with self._lock.writer_lock:
            variable_document = await self._variable_collection.find_one(
                filters={
                    "id": {"$eq": id},
                }
            )

            if not variable_document:
                raise ItemNotFoundError(
                    item_id=UniqueId(id),
                )

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
                        else None
                    }
                ),
            }

            result = await self._variable_collection.update_one(
                filters={
                    "id": {"$eq": id},
                },
                params=cast(_ContextVariableDocument, update_params),
            )

        assert result.updated_document

        return await self._deserialize_context_variable(
            context_variable_document=result.updated_document
        )

    @override
    async def delete_variable(
        self,
        id: ContextVariableId,
    ) -> None:
        async with self._lock.writer_lock:
            variable_deletion_result = await self._variable_collection.delete_one(
                {
                    "id": {"$eq": id},
                }
            )
            if variable_deletion_result.deleted_count == 0:
                raise ItemNotFoundError(
                    item_id=UniqueId(id),
                )

            for doc in await self._variable_tag_association_collection.find(
                {
                    "variable_id": {"$eq": id},
                }
            ):
                await self._variable_tag_association_collection.delete_one(
                    {
                        "id": {"$eq": doc["id"]},
                    }
                )

            for k, _ in await self.list_values(variable_id=id):
                await self.delete_value(variable_id=id, key=k)

    @override
    async def list_variables(
        self,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Sequence[ContextVariable]:
        filters: Where = {}

        async with self._lock.reader_lock:
            if tags is not None:
                if len(tags) == 0:
                    variable_ids = {
                        doc["variable_id"]
                        for doc in await self._variable_tag_association_collection.find(filters={})
                    }
                    filters = (
                        {"$and": [{"id": {"$ne": id}} for id in variable_ids]}
                        if variable_ids
                        else {}
                    )
                else:
                    tag_filters: Where = {"$or": [{"tag_id": {"$eq": tag}} for tag in tags]}
                    tag_associations = await self._variable_tag_association_collection.find(
                        filters=tag_filters
                    )
                    variable_ids = {assoc["variable_id"] for assoc in tag_associations}

                    if not variable_ids:
                        return []

                    filters = {"$or": [{"id": {"$eq": id}} for id in variable_ids]}

            return [
                await self._deserialize_context_variable(d)
                for d in await self._variable_collection.find(filters=filters)
            ]

    @override
    async def read_variable(
        self,
        id: ContextVariableId,
    ) -> ContextVariable:
        async with self._lock.reader_lock:
            variable_document = await self._variable_collection.find_one(
                {
                    "id": {"$eq": id},
                }
            )

        if not variable_document:
            raise ItemNotFoundError(
                item_id=UniqueId(id),
            )

        return await self._deserialize_context_variable(context_variable_document=variable_document)

    @override
    async def update_value(
        self,
        variable_id: ContextVariableId,
        key: str,
        data: JSONSerializable,
    ) -> ContextVariableValue:
        async with self._lock.writer_lock:
            last_modified = datetime.now(timezone.utc)

            value_checksum = md5_checksum(f"{variable_id}{key}{data}")

            value = ContextVariableValue(
                id=ContextVariableValueId(self._id_generator.generate(value_checksum)),
                last_modified=last_modified,
                data=data,
            )

            result = await self._value_collection.update_one(
                {
                    "variable_id": {"$eq": variable_id},
                    "key": {"$eq": key},
                },
                self._serialize_context_variable_value(
                    context_variable_value=value,
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
        variable_id: ContextVariableId,
        key: str,
    ) -> Optional[ContextVariableValue]:
        async with self._lock.reader_lock:
            value_document = await self._value_collection.find_one(
                {
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
        variable_id: ContextVariableId,
        key: str,
    ) -> None:
        async with self._lock.writer_lock:
            await self._value_collection.delete_one(
                {
                    "variable_id": {"$eq": variable_id},
                    "key": {"$eq": key},
                }
            )

    @override
    async def list_values(
        self,
        variable_id: ContextVariableId,
    ) -> Sequence[tuple[str, ContextVariableValue]]:
        async with self._lock.reader_lock:
            return [
                (d["key"], self._deserialize_context_variable_value(d))
                for d in await self._value_collection.find(
                    {
                        "variable_id": {"$eq": variable_id},
                    }
                )
            ]

    @override
    async def add_variable_tag(
        self,
        variable_id: ContextVariableId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> ContextVariable:
        async with self._lock.writer_lock:
            variable = await self.read_variable(id=variable_id)

            if tag_id in variable.tags:
                return variable

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_checksum = md5_checksum(f"{variable_id}{tag_id}")

            association_document: ContextVariableTagAssociationDocument = {
                "id": ObjectId(self._id_generator.generate(association_checksum)),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "variable_id": variable_id,
                "tag_id": tag_id,
            }

            _ = await self._variable_tag_association_collection.insert_one(
                document=association_document
            )

            variable_document = await self._variable_collection.find_one(
                {"id": {"$eq": variable_id}}
            )

        if not variable_document:
            raise ItemNotFoundError(item_id=UniqueId(variable_id))

        return await self._deserialize_context_variable(context_variable_document=variable_document)

    @override
    async def remove_variable_tag(
        self,
        variable_id: ContextVariableId,
        tag_id: TagId,
    ) -> ContextVariable:
        async with self._lock.writer_lock:
            delete_result = await self._variable_tag_association_collection.delete_one(
                {
                    "variable_id": {"$eq": variable_id},
                    "tag_id": {"$eq": tag_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(tag_id))

            variable_document = await self._variable_collection.find_one(
                {"id": {"$eq": variable_id}}
            )

        if not variable_document:
            raise ItemNotFoundError(item_id=UniqueId(variable_id))

        return await self._deserialize_context_variable(context_variable_document=variable_document)
