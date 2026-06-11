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

TagId = NewType("TagId", str)

_BUILT_IN_TAG_CREATION_TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Tag:
    id: TagId
    creation_utc: datetime
    modified_utc: datetime
    name: str

    @staticmethod
    def preamble() -> Tag:
        return Tag(
            id=TagId("__preamble__"),
            name="__preamble__",
            creation_utc=_BUILT_IN_TAG_CREATION_TIME,
            modified_utc=_BUILT_IN_TAG_CREATION_TIME,
        )

    @staticmethod
    def for_agent_id(agent_id: str) -> Tag:
        return Tag(
            id=TagId(f"agent:{agent_id}"),
            name=f"agent:{agent_id}",
            creation_utc=_BUILT_IN_TAG_CREATION_TIME,
            modified_utc=_BUILT_IN_TAG_CREATION_TIME,
        )

    @staticmethod
    def extract_agent_id(tag_id: TagId) -> Optional[str]:
        if not tag_id.startswith("agent:"):
            return None

        return str(tag_id.split(":")[1])

    @staticmethod
    def for_journey_id(journey_id: str) -> Tag:
        return Tag(
            id=TagId(f"journey:{journey_id}"),
            name=f"journey:{journey_id}",
            creation_utc=_BUILT_IN_TAG_CREATION_TIME,
            modified_utc=_BUILT_IN_TAG_CREATION_TIME,
        )

    @staticmethod
    def extract_journey_id(tag_id: TagId) -> Optional[str]:
        if not tag_id.startswith("journey:"):
            return None

        return str(tag_id.split(":")[1])

    @staticmethod
    def for_journey_node_id(journey_node_id: str) -> Tag:
        return Tag(
            id=TagId(f"journey_node:{journey_node_id}"),
            name=f"journey_node:{journey_node_id}",
            creation_utc=_BUILT_IN_TAG_CREATION_TIME,
            modified_utc=_BUILT_IN_TAG_CREATION_TIME,
        )

    @staticmethod
    def extract_journey_node_id(tag_id: TagId) -> Optional[str]:
        if not tag_id.startswith("journey_node:"):
            return None

        return str(tag_id.split(":")[1])

    @staticmethod
    def for_guideline_id(guideline_id: str) -> Tag:
        return Tag(
            id=TagId(f"guideline:{guideline_id}"),
            name=f"guideline:{guideline_id}",
            creation_utc=_BUILT_IN_TAG_CREATION_TIME,
            modified_utc=_BUILT_IN_TAG_CREATION_TIME,
        )

    @staticmethod
    def extract_guideline_id(tag_id: TagId) -> Optional[str]:
        if not tag_id.startswith("guideline:"):
            return None

        return str(tag_id.split(":")[1])


class TagUpdateParams(TypedDict, total=False):
    name: str


class TagStore(ABC):
    @abstractmethod
    async def create_tag(
        self,
        name: str,
        creation_utc: Optional[datetime] = None,
        id: Optional[TagId] = None,
    ) -> Tag: ...

    @abstractmethod
    async def read_tag(
        self,
        tag_id: TagId,
    ) -> Tag: ...

    @abstractmethod
    async def update_tag(
        self,
        tag_id: TagId,
        params: TagUpdateParams,
    ) -> Tag: ...

    @abstractmethod
    async def list_tags(
        self,
        name: Optional[str] = None,
    ) -> Sequence[Tag]: ...

    @abstractmethod
    async def delete_tag(
        self,
        tag_id: TagId,
    ) -> None: ...


class _TagDocument_v0_1_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    name: str


class _TagDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    last_modified: str
    name: str


class TagDocumentStore(TagStore):
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
        self._collection: DocumentCollection[_TagDocument]
        self._allow_migration = allow_migration
        self._collections_prefix = collections_prefix
        self._lock = ReaderWriterLock()

    async def _document_loader(self, doc: BaseDocument) -> Optional[_TagDocument]:
        async def v0_1_0_to_v0_2_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(_TagDocument_v0_1_0, doc)
            return _TagDocument(
                id=d["id"],
                version=Version.String("0.2.0"),
                creation_utc=d["creation_utc"],
                last_modified=d["creation_utc"],
                name=d["name"],
            )

        return await DocumentMigrationHelper[_TagDocument](
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
                name=f"{self._collections_prefix}_tags" if self._collections_prefix else "tags",
                schema=_TagDocument,
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
        tag: Tag,
    ) -> _TagDocument:
        return _TagDocument(
            id=ObjectId(tag.id),
            version=self.VERSION.to_string(),
            creation_utc=tag.creation_utc.isoformat(),
            last_modified=tag.modified_utc.isoformat(),
            name=tag.name,
        )

    def _deserialize(self, document: _TagDocument) -> Tag:
        return Tag(
            id=TagId(document["id"]),
            creation_utc=datetime.fromisoformat(document["creation_utc"]),
            modified_utc=datetime.fromisoformat(document["last_modified"]),
            name=document["name"],
        )

    @override
    async def create_tag(
        self,
        name: str,
        creation_utc: Optional[datetime] = None,
        id: Optional[TagId] = None,
    ) -> Tag:
        async with self._lock.writer_lock:
            existing = await self._collection.find({"name": {"$eq": name}})
            if existing:
                raise ValueError(f"Tag with name '{name}' already exists")

            creation_utc = creation_utc or datetime.now(timezone.utc)

            if id is not None:
                tag_id = id
                existing_by_id = await self._collection.find_one(filters={"id": {"$eq": tag_id}})
                if existing_by_id:
                    raise ValueError(f"Tag with id '{tag_id}' already exists")
            else:
                tag_id = TagId(self._id_generator.generate(f"{name}"))

            tag = Tag(
                id=tag_id,
                creation_utc=creation_utc,
                modified_utc=creation_utc,
                name=name,
            )
            await self._collection.insert_one(self._serialize(tag))

        return tag

    @override
    async def read_tag(
        self,
        tag_id: TagId,
    ) -> Tag:
        async with self._lock.reader_lock:
            document = await self._collection.find_one({"id": {"$eq": tag_id}})

        if not document:
            raise ItemNotFoundError(item_id=UniqueId(tag_id))

        return self._deserialize(document)

    @override
    async def update_tag(
        self,
        tag_id: TagId,
        params: TagUpdateParams,
    ) -> Tag:
        async with self._lock.writer_lock:
            tag_document = await self._collection.find_one(filters={"id": {"$eq": tag_id}})

            if not tag_document:
                raise ItemNotFoundError(item_id=UniqueId(tag_id))

            result = await self._collection.update_one(
                filters={"id": {"$eq": tag_id}},
                params={
                    "name": params["name"],
                    "last_modified": datetime.now(timezone.utc).isoformat(),
                },
            )

        assert result.updated_document

        return self._deserialize(document=result.updated_document)

    @override
    async def list_tags(
        self,
        name: Optional[str] = None,
    ) -> Sequence[Tag]:
        filters: Where = {}

        if name is not None:
            filters = {"name": {"$eq": name}}

        async with self._lock.reader_lock:
            return [self._deserialize(doc) for doc in await self._collection.find(filters)]

    @override
    async def delete_tag(
        self,
        tag_id: TagId,
    ) -> None:
        async with self._lock.writer_lock:
            result = await self._collection.delete_one({"id": {"$eq": tag_id}})

        if result.deleted_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(tag_id))


class CompositeTagStore(TagStore):
    def __init__(
        self,
        writable_store: TagStore,
        readable_stores: Sequence[TagStore],
    ) -> None:
        self._writable_store = writable_store
        self._readable_stores = readable_stores
        self._all_stores: Sequence[TagStore] = [writable_store, *readable_stores]

    @override
    async def create_tag(
        self,
        name: str,
        creation_utc: Optional[datetime] = None,
        id: Optional[TagId] = None,
    ) -> Tag:
        return await self._writable_store.create_tag(name, creation_utc, id=id)

    @override
    async def read_tag(
        self,
        tag_id: TagId,
    ) -> Tag:
        results = await safe_gather(
            *[try_or_none(store.read_tag(tag_id)) for store in self._all_stores]
        )
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise ItemNotFoundError(item_id=UniqueId(tag_id))
        return result

    @override
    async def update_tag(
        self,
        tag_id: TagId,
        params: TagUpdateParams,
    ) -> Tag:
        return await self._writable_store.update_tag(tag_id, params)

    @override
    async def list_tags(
        self,
        name: Optional[str] = None,
    ) -> Sequence[Tag]:
        results = await safe_gather(*[store.list_tags(name=name) for store in self._all_stores])
        return list(chain.from_iterable(results))

    @override
    async def delete_tag(
        self,
        tag_id: TagId,
    ) -> None:
        return await self._writable_store.delete_tag(tag_id)
