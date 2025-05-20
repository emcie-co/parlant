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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import NewType, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper
from parlant.core.tags import TagId
from parlant.core.common import ItemNotFoundError, UniqueId, Version, generate_id
from parlant.core.persistence.common import ObjectId, Where
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
)

UtteranceId = NewType("UtteranceId", str)


@dataclass(frozen=True)
class UtteranceField:
    name: str
    description: str
    examples: list[str]


@dataclass(frozen=True)
class Utterance:
    TRANSIENT_ID = UtteranceId("<transient>")
    INVALID_ID = UtteranceId("<invalid>")

    id: UtteranceId
    creation_utc: datetime
    value: str
    fields: Sequence[UtteranceField]
    tags: Sequence[TagId]


class UtteranceUpdateParams(TypedDict, total=True):
    value: str
    fields: Sequence[UtteranceField]


class UtteranceStore(ABC):
    @abstractmethod
    async def create_utterance(
        self,
        value: str,
        fields: Sequence[UtteranceField],
        creation_utc: Optional[datetime] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Utterance: ...

    @abstractmethod
    async def read_utterance(
        self,
        utterance_id: UtteranceId,
    ) -> Utterance: ...

    @abstractmethod
    async def update_utterance(
        self,
        utterance_id: UtteranceId,
        params: UtteranceUpdateParams,
    ) -> Utterance: ...

    @abstractmethod
    async def delete_utterance(
        self,
        utterance_id: UtteranceId,
    ) -> None: ...

    @abstractmethod
    async def list_utterances(
        self,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Sequence[Utterance]: ...

    @abstractmethod
    async def upsert_tag(
        self,
        utterance_id: UtteranceId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_tag(
        self,
        utterance_id: UtteranceId,
        tag_id: TagId,
    ) -> None: ...


class _UtteranceFieldDocument(TypedDict):
    name: str
    description: str
    examples: list[str]


class _UtteranceDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    value: str
    fields: Sequence[_UtteranceFieldDocument]


class _UtteranceTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    utterance_id: UtteranceId
    tag_id: TagId


class UtteranceDocumentStore(UtteranceStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(self, database: DocumentDatabase, allow_migration: bool = False) -> None:
        self._database = database
        self._utterances_collection: DocumentCollection[_UtteranceDocument]
        self._utterance_tag_association_collection: DocumentCollection[
            _UtteranceTagAssociationDocument
        ]
        self._allow_migration = allow_migration
        self._lock = ReaderWriterLock()

    async def _document_loader(self, doc: BaseDocument) -> Optional[_UtteranceDocument]:
        if doc["version"] == "0.1.0":
            return cast(_UtteranceDocument, doc)

        return None

    async def _association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[_UtteranceTagAssociationDocument]:
        if doc["version"] == "0.1.0":
            return cast(_UtteranceTagAssociationDocument, doc)
        return None

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
        ):
            self._utterances_collection = await self._database.get_or_create_collection(
                name="utterances",
                schema=_UtteranceDocument,
                document_loader=self._document_loader,
            )

            self._utterance_tag_association_collection = (
                await self._database.get_or_create_collection(
                    name="utterance_tag_associations",
                    schema=_UtteranceTagAssociationDocument,
                    document_loader=self._association_document_loader,
                )
            )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        return False

    def _serialize_utterance(self, utterance: Utterance) -> _UtteranceDocument:
        return _UtteranceDocument(
            id=ObjectId(utterance.id),
            version=self.VERSION.to_string(),
            creation_utc=utterance.creation_utc.isoformat(),
            value=utterance.value,
            fields=[
                {"name": s.name, "description": s.description, "examples": s.examples}
                for s in utterance.fields
            ],
        )

    async def _deserialize_utterance(self, utterance_document: _UtteranceDocument) -> Utterance:
        tags = [
            doc["tag_id"]
            for doc in await self._utterance_tag_association_collection.find(
                {"utterance_id": {"$eq": utterance_document["id"]}}
            )
        ]

        return Utterance(
            id=UtteranceId(utterance_document["id"]),
            creation_utc=datetime.fromisoformat(utterance_document["creation_utc"]),
            value=utterance_document["value"],
            fields=[
                UtteranceField(name=d["name"], description=d["description"], examples=d["examples"])
                for d in utterance_document["fields"]
            ],
            tags=tags,
        )

    @override
    async def create_utterance(
        self,
        value: str,
        fields: Sequence[UtteranceField],
        creation_utc: Optional[datetime] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Utterance:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            utterance = Utterance(
                id=UtteranceId(generate_id()),
                value=value,
                fields=fields,
                creation_utc=creation_utc,
                tags=tags or [],
            )

            await self._utterances_collection.insert_one(
                document=self._serialize_utterance(utterance=utterance)
            )

            for tag_id in tags or []:
                await self._utterance_tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(generate_id()),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "utterance_id": utterance.id,
                        "tag_id": tag_id,
                    }
                )

        return utterance

    @override
    async def read_utterance(
        self,
        utterance_id: UtteranceId,
    ) -> Utterance:
        async with self._lock.reader_lock:
            utterance_document = await self._utterances_collection.find_one(
                filters={"id": {"$eq": utterance_id}}
            )

        if not utterance_document:
            raise ItemNotFoundError(item_id=UniqueId(utterance_id))

        return await self._deserialize_utterance(utterance_document)

    @override
    async def update_utterance(
        self,
        utterance_id: UtteranceId,
        params: UtteranceUpdateParams,
    ) -> Utterance:
        async with self._lock.writer_lock:
            utterance_document = await self._utterances_collection.find_one(
                filters={"id": {"$eq": utterance_id}}
            )

            if not utterance_document:
                raise ItemNotFoundError(item_id=UniqueId(utterance_id))

            result = await self._utterances_collection.update_one(
                filters={"id": {"$eq": utterance_id}},
                params={
                    "value": params["value"],
                    "fields": [
                        {"name": s.name, "description": s.description, "examples": s.examples}
                        for s in params["fields"]
                    ],
                },
            )

        assert result.updated_document

        return await self._deserialize_utterance(utterance_document=result.updated_document)

    async def list_utterances(
        self,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Sequence[Utterance]:
        filters: Where = {}

        async with self._lock.reader_lock:
            if tags is not None:
                if len(tags) == 0:
                    utterance_ids = {
                        doc["utterance_id"]
                        for doc in await self._utterance_tag_association_collection.find(filters={})
                    }
                    filters = (
                        {"$and": [{"id": {"$ne": id}} for id in utterance_ids]}
                        if utterance_ids
                        else {}
                    )
                else:
                    tag_filters: Where = {"$or": [{"tag_id": {"$eq": tag}} for tag in tags]}
                    tag_associations = await self._utterance_tag_association_collection.find(
                        filters=tag_filters
                    )
                    utterance_ids = {assoc["utterance_id"] for assoc in tag_associations}

                    if not utterance_ids:
                        return []

                    filters = {"$or": [{"id": {"$eq": id}} for id in utterance_ids]}

            return [
                await self._deserialize_utterance(d)
                for d in await self._utterances_collection.find(filters=filters)
            ]

    @override
    async def delete_utterance(
        self,
        utterance_id: UtteranceId,
    ) -> None:
        async with self._lock.writer_lock:
            result = await self._utterances_collection.delete_one({"id": {"$eq": utterance_id}})

        if result.deleted_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(utterance_id))

    @override
    async def upsert_tag(
        self,
        utterance_id: UtteranceId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            utterance = await self.read_utterance(utterance_id)

            if tag_id in utterance.tags:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_document: _UtteranceTagAssociationDocument = {
                "id": ObjectId(generate_id()),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "utterance_id": utterance_id,
                "tag_id": tag_id,
            }

            _ = await self._utterance_tag_association_collection.insert_one(
                document=association_document
            )

            utterance_document = await self._utterances_collection.find_one(
                {"id": {"$eq": utterance_id}}
            )

        if not utterance_document:
            raise ItemNotFoundError(item_id=UniqueId(utterance_id))

        return True

    @override
    async def remove_tag(
        self,
        utterance_id: UtteranceId,
        tag_id: TagId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._utterance_tag_association_collection.delete_one(
                {
                    "utterance_id": {"$eq": utterance_id},
                    "tag_id": {"$eq": tag_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(tag_id))

            utterance_document = await self._utterances_collection.find_one(
                {"id": {"$eq": utterance_id}}
            )

        if not utterance_document:
            raise ItemNotFoundError(item_id=UniqueId(utterance_id))
