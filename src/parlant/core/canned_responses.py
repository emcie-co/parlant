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
import jinja2
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
    @staticmethod
    def create_transient(
        value: str,
    ) -> CannedResponse:
        return CannedResponse(
            id=CannedResponse.TRANSIENT_ID,
            value=value,
            fields=[],
            creation_utc=datetime.now(),
            tags=[],
            signals=[],
        )

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


@dataclass(frozen=True)
class CannedResponseRelevantResult:
    canned_response: CannedResponse
    score: float


class CannedResponseUpdateParams(TypedDict, total=False):
    value: str
    fields: Sequence[CannedResponseField]
    signals: Sequence[str]


class CannedResponseStore(ABC):
    @abstractmethod
    async def create_canned_response(
        self,
        value: str,
        fields: Optional[Sequence[CannedResponseField]] = None,
        signals: Optional[Sequence[str]] = None,
        creation_utc: Optional[datetime] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> CannedResponse: ...

    @abstractmethod
    async def read_canned_response(
        self,
        canned_response_id: CannedResponseId,
    ) -> CannedResponse: ...

    @abstractmethod
    async def update_canned_response(
        self,
        canned_response_id: CannedResponseId,
        params: CannedResponseUpdateParams,
    ) -> CannedResponse: ...

    @abstractmethod
    async def delete_canned_response(
        self,
        canned_response_id: CannedResponseId,
    ) -> None: ...

    @abstractmethod
    async def list_canned_responses(
        self,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Sequence[CannedResponse]: ...

    @abstractmethod
    async def find_relevant_canned_responses(
        self,
        query: str,
        available_canned_responses: Sequence[CannedResponse],
        max_count: int,
    ) -> Sequence[CannedResponseRelevantResult]: ...

    @abstractmethod
    async def upsert_tag(
        self,
        canned_response_id: CannedResponseId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_tag(
        self,
        canned_response_id: CannedResponseId,
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
    canned_response_id: ObjectId
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
    canned_response_id: CannedResponseId
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

        self._canreps_vector_collection: VectorCollection[CannedResponseVectorDocument]
        self._canreps_collection: DocumentCollection[CannedResponseDocument]
        self._canrep_tag_association_collection: DocumentCollection[
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
                canned_response_id=CannedResponseId(doc["canned_response_id"]),
                tag_id=TagId(doc["tag_id"]),
            )

        async def v0_3_0_to_v0_4_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(CannedResponseTagAssociationDocument, doc)

            return CannedResponseTagAssociationDocument(
                id=doc["id"],
                version=Version.String("0.4.0"),
                creation_utc=doc["creation_utc"],
                canned_response_id=CannedResponseId(doc["canned_response_id"]),
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
            self._canreps_vector_collection = await self._vector_db.get_or_create_collection(
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
            self._canreps_collection = await self._database.get_or_create_collection(
                name="canned_responses",
                schema=CannedResponseDocument,
                document_loader=self._document_loader,
            )

            self._canrep_tag_association_collection = await self._database.get_or_create_collection(
                name="canned_response_tag_associations",
                schema=CannedResponseTagAssociationDocument,
                document_loader=self._association_document_loader,
            )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        return False

    def _serialize_canned_response(
        self,
        canned_response_id: CannedResponse,
    ) -> CannedResponseDocument:
        return CannedResponseDocument(
            id=ObjectId(canned_response_id.id),
            version=self.VERSION.to_string(),
            creation_utc=canned_response_id.creation_utc.isoformat(),
            value=canned_response_id.value,
            fields=json.dumps(
                [
                    {"name": s.name, "description": s.description, "examples": s.examples}
                    for s in canned_response_id.fields
                ]
            ),
            signals=canned_response_id.signals,
        )

    async def _deserialize_canned_response(
        self, canned_response_document: CannedResponseDocument
    ) -> CannedResponse:
        tags = [
            doc["tag_id"]
            for doc in await self._canrep_tag_association_collection.find(
                {"canned_response_id": {"$eq": canned_response_document["id"]}}
            )
        ]

        return CannedResponse(
            id=CannedResponseId(canned_response_document["id"]),
            creation_utc=datetime.fromisoformat(canned_response_document["creation_utc"]),
            value=canned_response_document["value"],
            fields=[
                CannedResponseField(
                    name=d["name"], description=d["description"], examples=d["examples"]
                )
                for d in json.loads(canned_response_document["fields"])
            ],
            tags=tags,
            signals=canned_response_document["signals"],
        )

    def _list_canned_response_contents(self, canned_response: CannedResponse) -> list[str]:
        return [canned_response.value, *canned_response.signals]

    async def _insert_canned_response(
        self, canned_response: CannedResponse
    ) -> CannedResponseDocument:
        insertion_tasks = []

        for content in self._list_canned_response_contents(canned_response):
            vec_doc = CannedResponseVectorDocument(
                id=ObjectId(canned_response.id),
                canned_response_id=ObjectId(canned_response.id),
                version=self.VERSION.to_string(),
                content=content,
                checksum=md5_checksum(content),
            )

            insertion_tasks.append(self._canreps_vector_collection.insert_one(document=vec_doc))

        await async_utils.safe_gather(*insertion_tasks)

        doc = self._serialize_canned_response(canned_response)
        await self._canreps_collection.insert_one(document=doc)

        return doc

    @override
    async def create_canned_response(
        self,
        value: str,
        fields: Optional[Sequence[CannedResponseField]] = None,
        signals: Optional[Sequence[str]] = None,
        creation_utc: Optional[datetime] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> CannedResponse:
        self._validate_template(value)

        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            canrep_checksum = md5_checksum(f"{value}{fields}")
            canrep_id = CannedResponseId(self._id_generator.generate(canrep_checksum))

            canrep = CannedResponse(
                id=canrep_id,
                value=value,
                fields=fields or [],
                creation_utc=creation_utc,
                tags=tags or [],
                signals=signals or [],
            )

            await self._insert_canned_response(canrep)

            for tag_id in tags or []:
                tag_checksum = md5_checksum(f"{canrep.id}{tag_id}")

                await self._canrep_tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(self._id_generator.generate(tag_checksum)),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "canned_response_id": canrep.id,
                        "tag_id": tag_id,
                    }
                )

        return canrep

    def _validate_template(self, template: str) -> None:
        try:
            jinja2.Environment().parse(template)
        except jinja2.exceptions.TemplateSyntaxError as e:
            raise ValueError(f"Invalid Jinja2 template: '{template}': {e}")

    @override
    async def read_canned_response(
        self,
        canned_response_id: CannedResponseId,
    ) -> CannedResponse:
        async with self._lock.reader_lock:
            canned_response_document = await self._canreps_collection.find_one(
                filters={"id": {"$eq": canned_response_id}}
            )

        if not canned_response_document:
            raise ItemNotFoundError(item_id=UniqueId(canned_response_id))

        return await self._deserialize_canned_response(canned_response_document)

    @override
    async def update_canned_response(
        self,
        canned_response_id: CannedResponseId,
        params: CannedResponseUpdateParams,
    ) -> CannedResponse:
        async with self._lock.writer_lock:
            doc = await self._canreps_collection.find_one(
                filters={"id": {"$eq": canned_response_id}}
            )
            all_vector_docs = await self._canreps_vector_collection.find(
                filters={"canned_response_id": {"$eq": canned_response_id}}
            )

            if not doc:
                raise ItemNotFoundError(item_id=UniqueId(canned_response_id))

            existing_value = await self._deserialize_canned_response(doc)

            for v_doc in all_vector_docs:
                await self._canreps_collection.delete_one(filters={"id": {"$eq": v_doc["id"]}})

            value = params.get("value", existing_value.value)
            fields = params.get("fields", existing_value.fields)
            signals = params.get("signals", existing_value.signals)

            canrep = CannedResponse(
                id=CannedResponseId(canned_response_id),
                creation_utc=datetime.fromisoformat(doc["creation_utc"]),
                value=value,
                fields=fields,
                signals=signals,
                tags=existing_value.tags,
            )

            doc = await self._insert_canned_response(canrep)

        return await self._deserialize_canned_response(doc)

    async def list_canned_responses(
        self,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Sequence[CannedResponse]:
        filters: Where = {}

        async with self._lock.reader_lock:
            if tags is not None:
                if len(tags) == 0:
                    canrep_ids = {
                        doc["canned_response_id"]
                        for doc in await self._canrep_tag_association_collection.find(filters={})
                    }
                    filters = (
                        {"$and": [{"id": {"$ne": id}} for id in canrep_ids]} if canrep_ids else {}
                    )
                else:
                    tag_filters: Where = {"$or": [{"tag_id": {"$eq": tag}} for tag in tags]}
                    tag_associations = await self._canrep_tag_association_collection.find(
                        filters=tag_filters
                    )
                    canrep_ids = {assoc["canned_response_id"] for assoc in tag_associations}

                    if not canrep_ids:
                        return []

                    filters = {"$or": [{"id": {"$eq": id}} for id in canrep_ids]}

            canreps = await self._canreps_collection.find(filters=filters)

            return [await self._deserialize_canned_response(d) for d in canreps]

    @override
    async def delete_canned_response(
        self,
        canned_response_id: CannedResponseId,
    ) -> None:
        async with self._lock.writer_lock:
            tasks: list[Awaitable[Any]] = [
                self._canreps_collection.delete_one({"id": {"$eq": canned_response_id}})
            ]

            response_vector_documents = await self._canreps_vector_collection.find(
                {"canned_response_id": {"$eq": canned_response_id}}
            )

            tasks += [
                self._canreps_vector_collection.delete_one({"id": {"$eq": doc["id"]}})
                for doc in response_vector_documents
            ]

            tag_docs = await self._canrep_tag_association_collection.find(
                {"canned_response_id": {"$eq": canned_response_id}}
            )

            tasks += [
                self._canrep_tag_association_collection.delete_one(
                    {"canned_response_id": {"$eq": d["canned_response_id"]}}
                )
                for d in tag_docs
            ]

            await async_utils.safe_gather(*tasks)

    @override
    async def upsert_tag(
        self,
        canned_response_id: CannedResponseId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            canrep = await self.read_canned_response(canned_response_id)

            if tag_id in canrep.tags:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_checksum = md5_checksum(f"{canned_response_id}{tag_id}")

            association_document: CannedResponseTagAssociationDocument = {
                "id": ObjectId(self._id_generator.generate(association_checksum)),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "canned_response_id": canned_response_id,
                "tag_id": tag_id,
            }

            _ = await self._canrep_tag_association_collection.insert_one(
                document=association_document
            )

        return True

    @override
    async def remove_tag(
        self,
        canned_response_id: CannedResponseId,
        tag_id: TagId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._canrep_tag_association_collection.delete_one(
                {
                    "canned_response_id": {"$eq": canned_response_id},
                    "tag_id": {"$eq": tag_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(tag_id))

    @override
    async def find_relevant_canned_responses(
        self,
        query: str,
        available_canned_responses: Sequence[CannedResponse],
        max_count: int,
    ) -> Sequence[CannedResponseRelevantResult]:
        if not available_canned_responses:
            return []

        async with self._lock.reader_lock:
            queries = await query_chunks(query, self._embedder)
            filters: Where = {
                "canned_response_id": {"$in": [str(c.id) for c in available_canned_responses]}
            }

            tasks = [
                self._canreps_vector_collection.find_similar_documents(
                    filters=filters,
                    query=q,
                    k=calculate_min_vectors_for_max_item_count(
                        items=available_canned_responses,
                        count_item_vectors=lambda c: len(self._list_canned_response_contents(c)),
                        max_items_to_return=max_count,
                    ),
                )
                for q in queries
            ]

        all_sdocs = chain.from_iterable(await async_utils.safe_gather(*tasks))

        unique_sdocs: dict[str, SimilarDocumentResult[CannedResponseVectorDocument]] = {}

        for similar_doc in all_sdocs:
            if (
                similar_doc.document["canned_response_id"] not in unique_sdocs
                or unique_sdocs[similar_doc.document["canned_response_id"]].distance
                > similar_doc.distance
            ):
                unique_sdocs[similar_doc.document["canned_response_id"]] = similar_doc

            if len(unique_sdocs) >= max_count:
                break

        top_results = sorted(unique_sdocs.values(), key=lambda r: r.distance)[:max_count]

        return [
            CannedResponseRelevantResult(
                canned_response=await self._deserialize_canned_response(d),
                score=(1 - unique_sdocs[d["id"]].distance),
            )
            for d in await self._canreps_collection.find(
                {"id": {"$in": [r.document["canned_response_id"] for r in top_results]}}
            )
        ]
