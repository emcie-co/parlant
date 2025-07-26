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
from parlant.core.persistence.document_database_helper import (
    DocumentMigrationHelper,
    DocumentStoreMigrationHelper,
)
from parlant.core.persistence.vector_database import (
    SimilarDocumentResult,
    VectorCollection,
    VectorDatabase,
    BaseDocument as VectorDocument,
)
from parlant.core.persistence.vector_database_helper import (
    VectorDocumentStoreMigrationHelper,
    VectorDocumentMigrationHelper,
    calculate_min_vectors_for_max_item_count,
    query_chunks,
)
from parlant.core.tags import TagId
from parlant.core.common import ItemNotFoundError, UniqueId, Version, IdGenerator, md5_checksum
from parlant.core.persistence.common import ObjectId, Where
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
)

CannedResponseId = NewType("CannedResponseId", str)


@dataclass(frozen=True)
class CannedResponseField:
    name: str
    description: str
    examples: list[str]


@dataclass(frozen=True)
class CannedResponse:
    TRANSIENT_ID = CannedResponseId("<transient>")
    INVALID_ID = CannedResponseId("<invalid>")

    id: CannedResponseId
    creation_utc: datetime
    value: str
    fields: Sequence[CannedResponseField]
    signals: Sequence[str]
    tags: Sequence[TagId]

    def __hash__(self) -> int:
        return hash(self.id)


class CannedResponseUpdateParams(TypedDict, total=False):
    value: str
    fields: Sequence[CannedResponseField]
    signals: Sequence[str]


class CannedResponseStore(ABC):
    @abstractmethod
    async def create_can_rep(
        self,
        value: str,
        fields: Optional[Sequence[CannedResponseField]] = None,
        signals: Optional[Sequence[str]] = None,
        creation_utc: Optional[datetime] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> CannedResponse: ...

    @abstractmethod
    async def read_can_rep(
        self,
        can_rep_id: CannedResponseId,
    ) -> CannedResponse: ...

    @abstractmethod
    async def update_can_rep(
        self,
        can_rep_id: CannedResponseId,
        params: CannedResponseUpdateParams,
    ) -> CannedResponse: ...

    @abstractmethod
    async def delete_can_rep(
        self,
        can_rep_id: CannedResponseId,
    ) -> None: ...

    @abstractmethod
    async def list_can_reps(
        self,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Sequence[CannedResponse]: ...

    @abstractmethod
    async def find_relevant_can_reps(
        self,
        query: str,
        available_can_reps: Sequence[CannedResponse],
        max_count: int,
    ) -> Sequence[CannedResponse]: ...

    @abstractmethod
    async def upsert_tag(
        self,
        can_rep_id: CannedResponseId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_tag(
        self,
        can_rep_id: CannedResponseId,
        tag_id: TagId,
    ) -> None: ...


class _CannedResponseFieldDocument(TypedDict):
    name: str
    description: str
    examples: list[str]


class UtteranceDocument_v0_1_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    value: str
    fields: Sequence[_CannedResponseFieldDocument]


class UtteranceDocument_v0_2_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    content: str
    checksum: Required[str]
    value: str
    fields: str


class UtteranceDocument_v0_3_0(TypedDict, total=False):
    id: ObjectId
    utterance_id: ObjectId
    version: Version.String
    creation_utc: str
    content: str
    checksum: Required[str]
    value: str
    fields: str
    queries: str


class CannedResponseDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    value: str
    fields: str
    signals: Sequence[str]


class CannedResponseVectorDocument(TypedDict, total=False):
    id: ObjectId
    can_rep_id: ObjectId
    version: Version.String
    content: str
    checksum: Required[str]


class UtteranceTagAssociationDocument_v0_3_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    utterance_id: CannedResponseId
    tag_id: TagId


class CannedResponseTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    can_rep_id: CannedResponseId
    tag_id: TagId


class CannedResponseVectorStore(CannedResponseStore):
    VERSION = Version.from_string("0.4.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        vector_db: VectorDatabase,
        document_db: DocumentDatabase,
        embedder_type_provider: Callable[[], Awaitable[type[Embedder]]],
        embedder_factory: EmbedderFactory,
        allow_migration: bool = True,
    ) -> None:
        self._id_generator = id_generator

        self._vector_db = vector_db
        self._database = document_db

        self._can_reps_vector_collection: VectorCollection[CannedResponseVectorDocument]
        self._can_reps_collection: DocumentCollection[CannedResponseDocument]
        self._response_tag_association_collection: DocumentCollection[
            CannedResponseTagAssociationDocument
        ]
        self._allow_migration = allow_migration
        self._lock = ReaderWriterLock()
        self._embedder_factory = embedder_factory
        self._embedder_type_provider = embedder_type_provider
        self._embedder: Embedder

    async def _vector_document_loader(
        self, doc: VectorDocument
    ) -> Optional[CannedResponseVectorDocument]:
        async def v0_1_0_to_v0_4_0(doc: VectorDocument) -> Optional[VectorDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        return await VectorDocumentMigrationHelper[CannedResponseVectorDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_4_0,
                "0.2.0": v0_1_0_to_v0_4_0,
                "0.3.0": v0_1_0_to_v0_4_0,
            },
        ).migrate(doc)

    async def _document_loader(self, doc: BaseDocument) -> Optional[CannedResponseDocument]:
        async def v0_1_0_to_v0_4_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        return await DocumentMigrationHelper[CannedResponseDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_4_0,
                "0.2.0": v0_1_0_to_v0_4_0,
                "0.3.0": v0_1_0_to_v0_4_0,
            },
        ).migrate(doc)

    async def _association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[CannedResponseTagAssociationDocument]:
        async def v0_1_0_to_v0_2_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        async def v0_2_0_to_v0_3_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(CannedResponseTagAssociationDocument, doc)

            return CannedResponseTagAssociationDocument(
                id=doc["id"],
                version=Version.String("0.3.0"),
                creation_utc=doc["creation_utc"],
                can_rep_id=CannedResponseId(doc["can_rep_id"]),
                tag_id=TagId(doc["tag_id"]),
            )

        async def v0_3_0_to_v0_4_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(CannedResponseTagAssociationDocument, doc)

            return CannedResponseTagAssociationDocument(
                id=doc["id"],
                version=Version.String("0.4.0"),
                creation_utc=doc["creation_utc"],
                can_rep_id=CannedResponseId(doc["can_rep_id"]),
                tag_id=TagId(doc["tag_id"]),
            )

        return await DocumentMigrationHelper[CannedResponseTagAssociationDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_2_0,
                "0.2.0": v0_2_0_to_v0_3_0,
                "0.3.0": v0_3_0_to_v0_4_0,
            },
        ).migrate(doc)

    async def __aenter__(self) -> Self:
        embedder_type = await self._embedder_type_provider()

        self._embedder = self._embedder_factory.create_embedder(embedder_type)

        async with VectorDocumentStoreMigrationHelper(
            store=self,
            database=self._vector_db,
            allow_migration=self._allow_migration,
        ):
            self._can_reps_vector_collection = await self._vector_db.get_or_create_collection(
                name="canned_responses",
                schema=CannedResponseVectorDocument,
                embedder_type=embedder_type,
                document_loader=self._vector_document_loader,
            )

        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
        ):
            self._can_reps_collection = await self._database.get_or_create_collection(
                name="canned_responses",
                schema=CannedResponseDocument,
                document_loader=self._document_loader,
            )

            self._response_tag_association_collection = (
                await self._database.get_or_create_collection(
                    name="canned_response_tag_associations",
                    schema=CannedResponseTagAssociationDocument,
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

    def _serialize_can_rep(
        self,
        response: CannedResponse,
    ) -> CannedResponseDocument:
        return CannedResponseDocument(
            id=ObjectId(response.id),
            version=self.VERSION.to_string(),
            creation_utc=response.creation_utc.isoformat(),
            value=response.value,
            fields=json.dumps(
                [
                    {"name": s.name, "description": s.description, "examples": s.examples}
                    for s in response.fields
                ]
            ),
            signals=response.signals,
        )

    async def _deserialize_can_rep(
        self, response_document: CannedResponseDocument
    ) -> CannedResponse:
        tags = [
            doc["tag_id"]
            for doc in await self._response_tag_association_collection.find(
                {"can_rep_id": {"$eq": response_document["id"]}}
            )
        ]

        return CannedResponse(
            id=CannedResponseId(response_document["id"]),
            creation_utc=datetime.fromisoformat(response_document["creation_utc"]),
            value=response_document["value"],
            fields=[
                CannedResponseField(
                    name=d["name"], description=d["description"], examples=d["examples"]
                )
                for d in json.loads(response_document["fields"])
            ],
            tags=tags,
            signals=response_document["signals"],
        )

    def _list_can_rep_contents(self, response: CannedResponse) -> list[str]:
        return [response.value, *response.signals]

    async def _insert_can_rep(self, response: CannedResponse) -> CannedResponseDocument:
        insertion_tasks = []

        for content in self._list_can_rep_contents(response):
            vec_doc = CannedResponseVectorDocument(
                id=ObjectId(response.id),
                can_rep_id=ObjectId(response.id),
                version=self.VERSION.to_string(),
                content=content,
                checksum=md5_checksum(content),
            )

            insertion_tasks.append(self._can_reps_vector_collection.insert_one(document=vec_doc))

        await async_utils.safe_gather(*insertion_tasks)

        doc = self._serialize_can_rep(response)
        await self._can_reps_collection.insert_one(document=doc)

        return doc

    @override
    async def create_can_rep(
        self,
        value: str,
        fields: Optional[Sequence[CannedResponseField]] = None,
        signals: Optional[Sequence[str]] = None,
        creation_utc: Optional[datetime] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> CannedResponse:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            response_checksum = md5_checksum(f"{value}{fields}")
            can_rep_id = CannedResponseId(self._id_generator.generate(response_checksum))

            response = CannedResponse(
                id=can_rep_id,
                value=value,
                fields=fields or [],
                creation_utc=creation_utc,
                tags=tags or [],
                signals=signals or [],
            )

            await self._insert_can_rep(response)

            for tag_id in tags or []:
                tag_checksum = md5_checksum(f"{response.id}{tag_id}")

                await self._response_tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(self._id_generator.generate(tag_checksum)),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "can_rep_id": response.id,
                        "tag_id": tag_id,
                    }
                )

        return response

    @override
    async def read_can_rep(
        self,
        can_rep_id: CannedResponseId,
    ) -> CannedResponse:
        async with self._lock.reader_lock:
            response_document = await self._can_reps_collection.find_one(
                filters={"id": {"$eq": can_rep_id}}
            )

        if not response_document:
            raise ItemNotFoundError(item_id=UniqueId(can_rep_id))

        return await self._deserialize_can_rep(response_document)

    @override
    async def update_can_rep(
        self,
        can_rep_id: CannedResponseId,
        params: CannedResponseUpdateParams,
    ) -> CannedResponse:
        async with self._lock.writer_lock:
            doc = await self._can_reps_collection.find_one(filters={"id": {"$eq": can_rep_id}})
            all_vector_docs = await self._can_reps_vector_collection.find(
                filters={"can_rep_id": {"$eq": can_rep_id}}
            )

            if not doc:
                raise ItemNotFoundError(item_id=UniqueId(can_rep_id))

            existing_value = await self._deserialize_can_rep(doc)

            for v_doc in all_vector_docs:
                await self._can_reps_collection.delete_one(filters={"id": {"$eq": v_doc["id"]}})

            value = params.get("value", existing_value.value)
            fields = params.get("fields", existing_value.fields)
            signals = params.get("signals", existing_value.signals)

            response = CannedResponse(
                id=CannedResponseId(can_rep_id),
                creation_utc=datetime.fromisoformat(doc["creation_utc"]),
                value=value,
                fields=fields,
                signals=signals,
                tags=existing_value.tags,
            )

            doc = await self._insert_can_rep(response)

        return await self._deserialize_can_rep(doc)

    async def list_can_reps(
        self,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Sequence[CannedResponse]:
        filters: Where = {}

        async with self._lock.reader_lock:
            if tags is not None:
                if len(tags) == 0:
                    can_rep_ids = {
                        doc["can_rep_id"]
                        for doc in await self._response_tag_association_collection.find(filters={})
                    }
                    filters = (
                        {"$and": [{"id": {"$ne": id}} for id in can_rep_ids]} if can_rep_ids else {}
                    )
                else:
                    tag_filters: Where = {"$or": [{"tag_id": {"$eq": tag}} for tag in tags]}
                    tag_associations = await self._response_tag_association_collection.find(
                        filters=tag_filters
                    )
                    can_rep_ids = {assoc["can_rep_id"] for assoc in tag_associations}

                    if not can_rep_ids:
                        return []

                    filters = {"$or": [{"id": {"$eq": id}} for id in can_rep_ids]}

            can_reps = await self._can_reps_collection.find(filters=filters)

            return [await self._deserialize_can_rep(d) for d in can_reps]

    @override
    async def delete_can_rep(
        self,
        can_rep_id: CannedResponseId,
    ) -> None:
        async with self._lock.writer_lock:
            tasks: list[Awaitable[Any]] = [
                self._can_reps_collection.delete_one({"id": {"$eq": can_rep_id}})
            ]

            response_vector_documents = await self._can_reps_vector_collection.find(
                {"can_rep_id": {"$eq": can_rep_id}}
            )

            tasks += [
                self._can_reps_vector_collection.delete_one({"id": {"$eq": doc["id"]}})
                for doc in response_vector_documents
            ]

            tag_docs = await self._response_tag_association_collection.find(
                {"can_rep_id": {"$eq": can_rep_id}}
            )

            tasks += [
                self._response_tag_association_collection.delete_one(
                    {"can_rep_id": {"$eq": d["can_rep_id"]}}
                )
                for d in tag_docs
            ]

            await async_utils.safe_gather(*tasks)

    @override
    async def upsert_tag(
        self,
        can_rep_id: CannedResponseId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            response = await self.read_can_rep(can_rep_id)

            if tag_id in response.tags:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_checksum = md5_checksum(f"{can_rep_id}{tag_id}")

            association_document: CannedResponseTagAssociationDocument = {
                "id": ObjectId(self._id_generator.generate(association_checksum)),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "can_rep_id": can_rep_id,
                "tag_id": tag_id,
            }

            _ = await self._response_tag_association_collection.insert_one(
                document=association_document
            )

        return True

    @override
    async def remove_tag(
        self,
        can_rep_id: CannedResponseId,
        tag_id: TagId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._response_tag_association_collection.delete_one(
                {
                    "can_rep_id": {"$eq": can_rep_id},
                    "tag_id": {"$eq": tag_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(tag_id))

    @override
    async def find_relevant_can_reps(
        self,
        query: str,
        available_can_reps: Sequence[CannedResponse],
        max_count: int,
    ) -> Sequence[CannedResponse]:
        if not available_can_reps:
            return []

        async with self._lock.reader_lock:
            queries = await query_chunks(query, self._embedder)
            filters: Where = {"can_rep_id": {"$in": [str(c.id) for c in available_can_reps]}}

            tasks = [
                self._can_reps_vector_collection.find_similar_documents(
                    filters=filters,
                    query=q,
                    k=calculate_min_vectors_for_max_item_count(
                        items=available_can_reps,
                        count_item_vectors=lambda c: len(self._list_can_rep_contents(c)),
                        max_items_to_return=max_count,
                    ),
                )
                for q in queries
            ]

        all_sdocs = chain.from_iterable(await async_utils.safe_gather(*tasks))

        unique_sdocs: dict[str, SimilarDocumentResult[CannedResponseVectorDocument]] = {}

        for similar_doc in all_sdocs:
            if (
                similar_doc.document["can_rep_id"] not in unique_sdocs
                or unique_sdocs[similar_doc.document["can_rep_id"]].distance > similar_doc.distance
            ):
                unique_sdocs[similar_doc.document["can_rep_id"]] = similar_doc

            if len(unique_sdocs) >= max_count:
                break

        top_results = sorted(unique_sdocs.values(), key=lambda r: r.distance)[:max_count]

        return [
            await self._deserialize_can_rep(d)
            for d in await self._can_reps_collection.find(
                {"id": {"$in": [r.document["can_rep_id"] for r in top_results]}}
            )
        ]
