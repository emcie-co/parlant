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
from itertools import chain
import json
from typing import Any, Awaitable, Callable, NewType, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self, Required

from parlant.core import async_utils
from parlant.core.async_utils import ReaderWriterLock
from parlant.core.nlp.embedding import Embedder, EmbedderFactory
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper
from parlant.core.persistence.vector_database import (
    SimilarDocumentResult,
    VectorCollection,
    VectorDatabase,
)
from parlant.core.persistence.vector_database_helper import (
    VectorDocumentStoreMigrationHelper,
    calculate_min_vectors_for_max_item_count,
    query_chunks,
)
from parlant.core.tags import TagId
from parlant.core.common import ItemNotFoundError, UniqueId, Version, generate_id, md5_checksum
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
    queries: Sequence[str]
    tags: Sequence[TagId]

    def __hash__(self) -> int:
        return hash(self.id)


class UtteranceUpdateParams(TypedDict, total=False):
    value: str
    fields: Sequence[UtteranceField]
    queries: Sequence[str]


class UtteranceStore(ABC):
    @abstractmethod
    async def create_utterance(
        self,
        value: str,
        fields: Optional[Sequence[UtteranceField]] = None,
        queries: Optional[Sequence[str]] = None,
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
    async def find_relevant_utterances(
        self,
        query: str,
        available_utterances: Sequence[Utterance],
        max_count: int,
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


class UtteranceDocument_v0_1_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    value: str
    fields: Sequence[_UtteranceFieldDocument]


class _UtteranceDocument_v0_2_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    content: str
    checksum: Required[str]
    value: str
    fields: str


class _UtteranceDocument(TypedDict, total=False):
    id: ObjectId
    utterance_id: ObjectId
    version: Version.String
    creation_utc: str
    content: str
    checksum: Required[str]
    value: str
    fields: str
    queries: str


class UtteranceTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    utterance_id: UtteranceId
    tag_id: TagId


class UtteranceVectorStore(UtteranceStore):
    VERSION = Version.from_string("0.3.0")

    def __init__(
        self,
        vector_db: VectorDatabase,
        document_db: DocumentDatabase,
        embedder_type_provider: Callable[[], Awaitable[type[Embedder]]],
        embedder_factory: EmbedderFactory,
        allow_migration: bool = True,
    ) -> None:
        self._vector_db = vector_db
        self._database = document_db

        self._utterances_collection: VectorCollection[_UtteranceDocument]
        self._utterance_tag_association_collection: DocumentCollection[
            UtteranceTagAssociationDocument
        ]
        self._allow_migration = allow_migration
        self._lock = ReaderWriterLock()
        self._embedder_factory = embedder_factory
        self._embedder_type_provider = embedder_type_provider
        self._embedder: Embedder

    async def _document_loader(self, doc: BaseDocument) -> Optional[_UtteranceDocument]:
        if doc["version"] == "0.1.0":
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        if doc["version"] == "0.2.0":
            doc = cast(_UtteranceDocument_v0_2_0, doc)

            return _UtteranceDocument(
                id=ObjectId(doc["id"]),
                utterance_id=ObjectId(doc["id"]),
                version=self.VERSION.to_string(),
                creation_utc=doc["creation_utc"],
                content=doc["content"],
                checksum=doc["checksum"],
                value=doc["value"],
                fields=doc["fields"],
            )

        if doc["version"] == "0.3.0":
            return cast(_UtteranceDocument, doc)

        return None

    async def _association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[UtteranceTagAssociationDocument]:
        if doc["version"] == "0.1.0":
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        if doc["version"] in ["0.2.0", "0.3.0"]:
            return cast(UtteranceTagAssociationDocument, doc)

        return None

    async def __aenter__(self) -> Self:
        embedder_type = await self._embedder_type_provider()

        self._embedder = self._embedder_factory.create_embedder(embedder_type)

        async with VectorDocumentStoreMigrationHelper(
            store=self,
            database=self._vector_db,
            allow_migration=self._allow_migration,
        ):
            self._utterances_collection = await self._vector_db.get_or_create_collection(
                name="utterances",
                schema=_UtteranceDocument,
                embedder_type=embedder_type,
                document_loader=self._document_loader,
            )

        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
        ):
            self._utterance_tag_association_collection = (
                await self._database.get_or_create_collection(
                    name="utterance_tag_associations",
                    schema=UtteranceTagAssociationDocument,
                    document_loader=self._association_document_loader,
                )
            )

        async with VectorDocumentStoreMigrationHelper(
            store=self,
            database=self._vector_db,
            allow_migration=self._allow_migration,
        ):
            self._utterances_collection = await self._vector_db.get_or_create_collection(
                name="utterances",
                schema=_UtteranceDocument,
                embedder_type=embedder_type,
                document_loader=self._document_loader,
            )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        return False

    def _serialize_utterance(self, utterance: Utterance, content: str) -> _UtteranceDocument:
        return _UtteranceDocument(
            id=ObjectId(generate_id()),
            utterance_id=ObjectId(utterance.id),
            version=self.VERSION.to_string(),
            creation_utc=utterance.creation_utc.isoformat(),
            content=content,
            checksum=md5_checksum(content),
            value=utterance.value,
            fields=json.dumps(
                [
                    {"name": s.name, "description": s.description, "examples": s.examples}
                    for s in utterance.fields
                ]
            ),
            queries=json.dumps(list(utterance.queries)),
        )

    async def _deserialize_utterance(self, utterance_document: _UtteranceDocument) -> Utterance:
        tags = [
            doc["tag_id"]
            for doc in await self._utterance_tag_association_collection.find(
                {"utterance_id": {"$eq": utterance_document["utterance_id"]}}
            )
        ]

        return Utterance(
            id=UtteranceId(utterance_document["utterance_id"]),
            creation_utc=datetime.fromisoformat(utterance_document["creation_utc"]),
            value=utterance_document["value"],
            fields=[
                UtteranceField(name=d["name"], description=d["description"], examples=d["examples"])
                for d in json.loads(utterance_document["fields"])
            ],
            tags=tags,
            queries=json.loads(utterance_document.get("queries", "[]")),
        )

    def _list_utterance_contents(self, utterance: Utterance) -> list[str]:
        return [utterance.value, *utterance.queries]

    async def _insert_utterance(self, utterance: Utterance) -> _UtteranceDocument:
        insertion_tasks = []

        for content in self._list_utterance_contents(utterance):
            doc = self._serialize_utterance(utterance, content)
            insertion_tasks.append(self._utterances_collection.insert_one(document=doc))

        await async_utils.safe_gather(*insertion_tasks)

        return doc

    @override
    async def create_utterance(
        self,
        value: str,
        fields: Optional[Sequence[UtteranceField]] = None,
        queries: Optional[Sequence[str]] = None,
        creation_utc: Optional[datetime] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Utterance:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            utterance = Utterance(
                id=UtteranceId(generate_id()),
                value=value,
                fields=fields or [],
                creation_utc=creation_utc,
                tags=tags or [],
                queries=queries or [],
            )

            await self._insert_utterance(utterance)

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
                filters={"utterance_id": {"$eq": utterance_id}}
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
        # async with self._lock.writer_lock:
        #    utterance_document = await self._utterances_collection.find_one(
        #        filters={"id": {"$eq": utterance_id}}
        #    )

        #    if not utterance_document:
        #        raise ItemNotFoundError(item_id=UniqueId(utterance_id))

        #    result = await self._utterances_collection.update_one(
        #        filters={"id": {"$eq": utterance_id}},
        #        params={
        #            "value": params["value"],
        #            "fields": json.dumps(
        #                [
        #                    {"name": s.name, "description": s.description, "examples": s.examples}
        #                    for s in params["fields"]
        #                ]
        #            ),
        #            "content": content,
        #            "checksum": md5_checksum(content),
        #        },
        #    )

        # assert result.updated_document

        # return await self._deserialize_utterance(utterance_document=result.updated_document)

        async with self._lock.writer_lock:
            all_docs = await self._utterances_collection.find(
                filters={"utterance_id": {"$eq": utterance_id}}
            )

            if not all_docs:
                raise ItemNotFoundError(item_id=UniqueId(utterance_id))

            existing_value = await self._deserialize_utterance(all_docs[0])

            for doc in all_docs:
                await self._utterances_collection.delete_one(filters={"id": {"$eq": doc["id"]}})

            value = params.get("value", existing_value.value)
            fields = params.get("fields", existing_value.fields)
            queries = params.get("queries", existing_value.queries)

            utterance = Utterance(
                id=UtteranceId(utterance_id),
                creation_utc=datetime.fromisoformat(doc["creation_utc"]),
                value=value,
                fields=fields,
                queries=queries,
                tags=existing_value.tags,
            )

            doc = await self._insert_utterance(utterance)

        return await self._deserialize_utterance(doc)

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
                        {"$and": [{"utterance_id": {"$ne": id}} for id in utterance_ids]}
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

                    filters = {"$or": [{"utterance_id": {"$eq": id}} for id in utterance_ids]}

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
            utterance_documents = await self._utterances_collection.find(
                {"utterance_id": {"$eq": utterance_id}}
            )

            if not utterance_documents:
                raise ItemNotFoundError(item_id=UniqueId(utterance_id))

            tasks: list[Awaitable[Any]] = [
                self._utterances_collection.delete_one({"id": {"$eq": utterance_document["id"]}})
                for utterance_document in utterance_documents
            ]

            tasks += [
                self._utterance_tag_association_collection.delete_one(
                    {"utterance_id": {"$eq": d["utterance_id"]}}
                )
                for d in utterance_documents
            ]

            await async_utils.safe_gather(*tasks)

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

            association_document: UtteranceTagAssociationDocument = {
                "id": ObjectId(generate_id()),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "utterance_id": utterance_id,
                "tag_id": tag_id,
            }

            _ = await self._utterance_tag_association_collection.insert_one(
                document=association_document
            )

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

    @override
    async def find_relevant_utterances(
        self,
        query: str,
        available_utterances: Sequence[Utterance],
        max_count: int,
    ) -> Sequence[Utterance]:
        if not available_utterances:
            return []

        async with self._lock.reader_lock:
            queries = await query_chunks(query, self._embedder)
            filters: Where = {"utterance_id": {"$in": [str(c.id) for c in available_utterances]}}

            tasks = [
                self._utterances_collection.find_similar_documents(
                    filters=filters,
                    query=q,
                    k=calculate_min_vectors_for_max_item_count(
                        items=available_utterances,
                        count_item_vectors=lambda c: len(self._list_utterance_contents(c)),
                        max_items_to_return=max_count,
                    ),
                )
                for q in queries
            ]

        all_sdocs = chain.from_iterable(await async_utils.safe_gather(*tasks))

        unique_sdocs: dict[str, SimilarDocumentResult[_UtteranceDocument]] = {}

        for similar_doc in all_sdocs:
            if (
                similar_doc.document["utterance_id"] not in unique_sdocs
                or unique_sdocs[similar_doc.document["utterance_id"]].distance
                > similar_doc.distance
            ):
                unique_sdocs[similar_doc.document["utterance_id"]] = similar_doc

            if len(unique_sdocs) >= max_count:
                break

        top_results = sorted(unique_sdocs.values(), key=lambda r: r.distance)[:max_count]

        return [await self._deserialize_utterance(r.document) for r in top_results]
