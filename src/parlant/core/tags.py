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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import NewType, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self


from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import ItemNotFoundError, IdGenerator, UniqueId
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentCollection,
    DocumentDatabase,
)
from parlant.core.common import Version
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper

TagId = NewType("TagId", str)


@dataclass(frozen=True)
class Tag:
    id: TagId
    creation_utc: datetime
    name: str

    @staticmethod
    def preamble() -> TagId:
        return TagId("__preamble__")

    @staticmethod
    def for_agent_id(agent_id: str) -> TagId:
        return TagId(f"agent:{agent_id}")

    @staticmethod
    def extract_agent_id(tag_id: TagId) -> Optional[str]:
        if not tag_id.startswith("agent:"):
            return None

        return str(tag_id.split(":")[1])

    @staticmethod
    def for_journey_id(journey_id: str) -> TagId:
        return TagId(f"journey:{journey_id}")

    @staticmethod
    def extract_journey_id(tag_id: TagId) -> Optional[str]:
        if not tag_id.startswith("journey:"):
            return None

        return str(tag_id.split(":")[1])

    @staticmethod
    def for_journey_node_id(journey_node_id: str) -> TagId:
        return TagId(f"journey_node:{journey_node_id}")

    @staticmethod
    def extract_journey_node_id(tag_id: TagId) -> Optional[str]:
        if not tag_id.startswith("journey_node:"):
            return None

        return str(tag_id.split(":")[1])

    @staticmethod
    def for_guideline_id(guideline_id: str) -> TagId:
        return TagId(f"guideline:{guideline_id}")

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


class TagDocumentStore(TagStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        database: DocumentDatabase,
        allow_migration: bool = False,
    ) -> None:
        self._id_generator = id_generator

        self._database = database
        self._collection: DocumentCollection[_TagDocument]
        self._allow_migration = allow_migration
        self._lock = ReaderWriterLock()

    async def _document_loader(self, doc: BaseDocument) -> Optional[_TagDocument]:
        if doc["version"] == "0.1.0":
            return cast(_TagDocument, doc)
        return None

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
        ):
            self._collection = await self._database.get_or_create_collection(
                name="tags",
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

            tag_checksum = f"{name}"

            tag = Tag(
                id=TagId(self._id_generator.generate(tag_checksum)),
                creation_utc=creation_utc,
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
