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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import NewType, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import ItemNotFoundError, generate_id, UniqueId
from parlant.core.persistence.common import MigrationRequiredError, ObjectId, VersionMismatchError
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentCollection,
    DocumentDatabase,
)
from parlant.core.common import Version

TagId = NewType("TagId", str)


@dataclass(frozen=True)
class Tag:
    id: TagId
    creation_utc: datetime
    name: str


class TagUpdateParams(TypedDict, total=False):
    name: str


class TagStore(ABC):
    @abstractmethod
    async def create_tag(
        self,
        name: str,
        creation_utc: Optional[datetime] = None,
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
    ) -> Sequence[Tag]: ...

    @abstractmethod
    async def delete_tag(
        self,
        tag_id: TagId,
    ) -> None: ...


class _TagDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    name: str


class _MetadataDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String


class TagDocumentStore(TagStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(self, database: DocumentDatabase, migrate: bool = True) -> None:
        self._database = database
        self._collection: DocumentCollection[_TagDocument]
        self._meta_collection: DocumentCollection[_MetadataDocument]

        self._migrate = migrate

        self._lock = ReaderWriterLock()

    async def _meta_document_loader(self, doc: BaseDocument) -> Optional[_MetadataDocument]:
        if doc["version"] == "0.1.0":
            return cast(_MetadataDocument, doc)

        if not self._migrate:
            raise VersionMismatchError(
                f"Version mismatch in 'TagDocumentStore': Expected '{self.VERSION.to_string()}', but got '{doc['version']}'."
            )

        return None

    async def _document_loader(self, doc: BaseDocument) -> Optional[_TagDocument]:
        if doc["version"] == "0.1.0":
            return cast(_TagDocument, doc)
        return None

    async def __aenter__(self) -> Self:
        self._meta_collection = await self._database.get_or_create_collection(
            name="metadata",
            schema=_MetadataDocument,
            document_loader=self._meta_document_loader,
        )

        async with self._lock.reader_lock:
            existing_meta = await self._meta_collection.find_one({})
            if not existing_meta:
                if not self._migrate:
                    raise MigrationRequiredError(
                        "Migration is required to proceed with initialization."
                    )

        self._collection = await self._database.get_or_create_collection(
            name="tags",
            schema=_TagDocument,
            document_loader=self._document_loader,
        )

        async with self._lock.writer_lock:
            if not existing_meta:
                meta_document = _MetadataDocument(
                    id=ObjectId(generate_id()),
                    version=TagDocumentStore.VERSION.to_string(),
                )
                await self._meta_collection.insert_one(meta_document)

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
            name=tag.name,
        )

    def _deserialize(self, document: _TagDocument) -> Tag:
        return Tag(
            id=TagId(document["id"]),
            creation_utc=datetime.fromisoformat(document["creation_utc"]),
            name=document["name"],
        )

    @override
    async def create_tag(
        self,
        name: str,
        creation_utc: Optional[datetime] = None,
    ) -> Tag:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            tag = Tag(id=TagId(generate_id()), creation_utc=creation_utc, name=name)
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
                params={"name": params["name"]},
            )

        assert result.updated_document

        return self._deserialize(document=result.updated_document)

    @override
    async def list_tags(
        self,
    ) -> Sequence[Tag]:
        async with self._lock.reader_lock:
            return [self._deserialize(doc) for doc in await self._collection.find({})]

    @override
    async def delete_tag(
        self,
        tag_id: TagId,
    ) -> None:
        async with self._lock.writer_lock:
            result = await self._collection.delete_one({"id": {"$eq": tag_id}})

        if result.deleted_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(tag_id))
