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

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from typing import NewType, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self


from parlant.core.async_utils import ReaderWriterLock, safe_gather
from parlant.core.common import ItemNotFoundError, try_or_none, IdGenerator, UniqueId
from parlant.core.persistence.common import ObjectId, Where
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentCollection,
    DocumentDatabase,
)
from parlant.core.common import Version
from parlant.core.persistence.document_database_helper import (
    DocumentMigrationHelper,
    DocumentStoreMigrationHelper,
)

GroupId = NewType("GroupId", str)

_BUILT_IN_GROUP_CREATION_TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Group:
    id: GroupId
    creation_utc: datetime
    modified_utc: datetime
    name: str


class GroupIds:
    @staticmethod
    def preamble() -> GroupId:
        return GroupId("__preamble__")

    @staticmethod
    def for_agent_id(agent_id: str) -> GroupId:
        return GroupId(f"agent:{agent_id}")

    @staticmethod
    def extract_agent_id(group_id: GroupId) -> Optional[str]:
        if not group_id.startswith("agent:"):
            return None

        return str(group_id.split(":")[1])

    @staticmethod
    def for_journey_id(journey_id: str) -> GroupId:
        return GroupId(f"journey:{journey_id}")

    @staticmethod
    def extract_journey_id(group_id: GroupId) -> Optional[str]:
        if not group_id.startswith("journey:"):
            return None

        return str(group_id.split(":")[1])

    @staticmethod
    def for_journey_node_id(journey_node_id: str) -> GroupId:
        return GroupId(f"journey_node:{journey_node_id}")

    @staticmethod
    def extract_journey_node_id(group_id: GroupId) -> Optional[str]:
        if not group_id.startswith("journey_node:"):
            return None

        return str(group_id.split(":")[1])

    @staticmethod
    def for_rule_id(rule_id: str) -> GroupId:
        return GroupId(f"rule:{rule_id}")

    @staticmethod
    def extract_rule_id(group_id: GroupId) -> Optional[str]:
        if not group_id.startswith("rule:"):
            return None

        return str(group_id.split(":")[1])


class GroupUpdateParams(TypedDict, total=False):
    name: str


class GroupStore(ABC):
    @abstractmethod
    async def create_group(
        self,
        name: str,
        creation_utc: Optional[datetime] = None,
        id: Optional[GroupId] = None,
    ) -> Group: ...

    @abstractmethod
    async def read_group(
        self,
        group_id: GroupId,
    ) -> Group: ...

    @abstractmethod
    async def update_group(
        self,
        group_id: GroupId,
        params: GroupUpdateParams,
    ) -> Group: ...

    @abstractmethod
    async def list_groups(
        self,
        name: Optional[str] = None,
    ) -> Sequence[Group]: ...

    @abstractmethod
    async def delete_group(
        self,
        group_id: GroupId,
    ) -> None: ...


class _GroupDocument_v0_1_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    name: str


class _GroupDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    last_modified: str
    name: str


class GroupDocumentStore(GroupStore):
    VERSION = Version.from_string("0.2.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        database: DocumentDatabase,
        allow_migration: bool = False,
        collections_prefix: str | None = None,
    ) -> None:
        self._id_generator = id_generator

        self._database = database
        self._collection: DocumentCollection[_GroupDocument]
        self._allow_migration = allow_migration
        self._collections_prefix = collections_prefix
        self._lock = ReaderWriterLock()

    async def _document_loader(self, doc: BaseDocument) -> Optional[_GroupDocument]:
        async def v0_1_0_to_v0_2_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(_GroupDocument_v0_1_0, doc)
            return _GroupDocument(
                id=d["id"],
                version=Version.String("0.2.0"),
                creation_utc=d["creation_utc"],
                last_modified=d["creation_utc"],
                name=d["name"],
            )

        return await DocumentMigrationHelper[_GroupDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_2_0,
            },
        ).migrate(doc)

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
            collections_prefix=self._collections_prefix,
        ):
            self._collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_groups" if self._collections_prefix else "groups",
                schema=_GroupDocument,
                document_loader=self._document_loader,
            )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    def _serialize(
        self,
        group: Group,
    ) -> _GroupDocument:
        return _GroupDocument(
            id=ObjectId(group.id),
            version=self.VERSION.to_string(),
            creation_utc=group.creation_utc.isoformat(),
            last_modified=group.modified_utc.isoformat(),
            name=group.name,
        )

    def _deserialize(self, document: _GroupDocument) -> Group:
        return Group(
            id=GroupId(document["id"]),
            creation_utc=datetime.fromisoformat(document["creation_utc"]),
            modified_utc=datetime.fromisoformat(document["last_modified"]),
            name=document["name"],
        )

    @override
    async def create_group(
        self,
        name: str,
        creation_utc: Optional[datetime] = None,
        id: Optional[GroupId] = None,
    ) -> Group:
        async with self._lock.writer_lock:
            existing = await self._collection.find({"name": {"$eq": name}})
            if existing:
                raise ValueError(f"Group with name '{name}' already exists")

            creation_utc = creation_utc or datetime.now(timezone.utc)

            if id is not None:
                group_id = id
                existing_by_id = await self._collection.find_one(filters={"id": {"$eq": group_id}})
                if existing_by_id:
                    raise ValueError(f"Group with id '{group_id}' already exists")
            else:
                group_id = GroupId(self._id_generator.generate(f"{name}"))

            group = Group(
                id=group_id,
                creation_utc=creation_utc,
                modified_utc=creation_utc,
                name=name,
            )
            await self._collection.insert_one(self._serialize(group))

        return group

    @override
    async def read_group(
        self,
        group_id: GroupId,
    ) -> Group:
        async with self._lock.reader_lock:
            document = await self._collection.find_one({"id": {"$eq": group_id}})

        if not document:
            raise ItemNotFoundError(item_id=UniqueId(group_id))

        return self._deserialize(document)

    @override
    async def update_group(
        self,
        group_id: GroupId,
        params: GroupUpdateParams,
    ) -> Group:
        async with self._lock.writer_lock:
            tag_document = await self._collection.find_one(filters={"id": {"$eq": group_id}})

            if not tag_document:
                raise ItemNotFoundError(item_id=UniqueId(group_id))

            result = await self._collection.update_one(
                filters={"id": {"$eq": group_id}},
                params={
                    "name": params["name"],
                    "last_modified": datetime.now(timezone.utc).isoformat(),
                },
            )

        assert result.updated_document

        return self._deserialize(document=result.updated_document)

    @override
    async def list_groups(
        self,
        name: Optional[str] = None,
    ) -> Sequence[Group]:
        filters: Where = {}

        if name is not None:
            filters = {"name": {"$eq": name}}

        async with self._lock.reader_lock:
            return [self._deserialize(doc) for doc in await self._collection.find(filters)]

    @override
    async def delete_group(
        self,
        group_id: GroupId,
    ) -> None:
        async with self._lock.writer_lock:
            result = await self._collection.delete_one({"id": {"$eq": group_id}})

        if result.deleted_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(group_id))


class CompositeGroupStore(GroupStore):
    def __init__(
        self,
        writable_store: GroupStore,
        readable_stores: Sequence[GroupStore],
    ) -> None:
        self._writable_store = writable_store
        self._readable_stores = readable_stores
        self._all_stores: Sequence[GroupStore] = [writable_store, *readable_stores]

    @override
    async def create_group(
        self,
        name: str,
        creation_utc: Optional[datetime] = None,
        id: Optional[GroupId] = None,
    ) -> Group:
        return await self._writable_store.create_group(name, creation_utc, id=id)

    @override
    async def read_group(
        self,
        group_id: GroupId,
    ) -> Group:
        results = await safe_gather(
            *[try_or_none(store.read_group(group_id)) for store in self._all_stores]
        )
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise ItemNotFoundError(item_id=UniqueId(group_id))
        return result

    @override
    async def update_group(
        self,
        group_id: GroupId,
        params: GroupUpdateParams,
    ) -> Group:
        return await self._writable_store.update_group(group_id, params)

    @override
    async def list_groups(
        self,
        name: Optional[str] = None,
    ) -> Sequence[Group]:
        results = await safe_gather(*[store.list_groups(name=name) for store in self._all_stores])
        return list(chain.from_iterable(results))

    @override
    async def delete_group(
        self,
        group_id: GroupId,
    ) -> None:
        return await self._writable_store.delete_group(group_id)
