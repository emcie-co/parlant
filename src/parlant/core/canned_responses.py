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
import json
from typing import Any, Awaitable, Mapping, NewType, Optional, Sequence
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from typing_extensions import override, TypedDict, Self, Required

from parlant.core import async_utils
from parlant.core.async_utils import ReaderWriterLock, safe_gather
from parlant.core.nlp.embedding import Embedder, EmbedderProvider
from parlant.core.persistence.document_database_helper import (
    DocumentMigrationHelper,
    DocumentStoreMigrationHelper,
)
from parlant.core.persistence.vector_database import (
    SimilarDocumentResult,
    VectorCollection,
    VectorCollectionIndex,
    VectorDatabase,
    BaseDocument as VectorDocument,
)
from parlant.core.persistence.vector_database_helper import (
    VectorDocumentStoreMigrationHelper,
    VectorDocumentMigrationHelper,
    calculate_min_vectors_for_max_item_count,
    query_chunks,
)
from parlant.core.groups import GroupId
from parlant.core.common import (
    ItemNotFoundError,
    try_or_none,
    JSONSerializable,
    UniqueId,
    Version,
    IdGenerator,
    xxh3_checksum,
)
from parlant.core.persistence.common import ObjectId, Where
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
    group_ids_by_entity,
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
        now = datetime.now()
        return CannedResponse(
            id=CannedResponse.TRANSIENT_ID,
            value=value,
            fields=[],
            creation_utc=now,
            modified_utc=now,
            groups=[],
            signals=[],
            metadata={},
            field_dependencies=[],
        )

    TRANSIENT_ID = CannedResponseId("<transient>")
    INVALID_ID = CannedResponseId("<invalid>")

    id: CannedResponseId
    creation_utc: datetime
    modified_utc: datetime
    value: str
    fields: Sequence[CannedResponseField]
    signals: Sequence[str]
    metadata: Mapping[str, JSONSerializable]
    groups: Sequence[GroupId]
    field_dependencies: Sequence[str]
    # Free-text guidance about when/how this response applies. Surfaced to the
    # engine as "Notes" and available for analytics, but never rendered into
    # the response text itself.
    note: str = ""

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True)
class CannedResponseRelevanceResult:
    canned_response: CannedResponse
    score: float


class CannedResponseUpdateParams(TypedDict, total=False):
    value: str
    fields: Sequence[CannedResponseField]
    signals: Sequence[str]
    metadata: Mapping[str, JSONSerializable]
    field_dependencies: Sequence[str]
    note: str


class CannedResponseStore(ABC):
    @abstractmethod
    async def create_canned_response(
        self,
        value: str,
        fields: Optional[Sequence[CannedResponseField]] = None,
        signals: Optional[Sequence[str]] = None,
        creation_utc: Optional[datetime] = None,
        metadata: Mapping[str, JSONSerializable] = {},
        groups: Optional[Sequence[GroupId]] = None,
        field_dependencies: Optional[Sequence[str]] = None,
        note: str = "",
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
        groups: Optional[Sequence[GroupId]] = None,
    ) -> Sequence[CannedResponse]: ...

    @abstractmethod
    async def filter_relevant_canned_responses(
        self,
        query: str,
        available_canned_responses: Sequence[CannedResponse],
        max_count: int,
    ) -> Sequence[CannedResponseRelevanceResult]: ...

    @abstractmethod
    async def upsert_group(
        self,
        canned_response_id: CannedResponseId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_group(
        self,
        canned_response_id: CannedResponseId,
        group_id: GroupId,
    ) -> None: ...


class _CannedResponseFieldDocument(TypedDict):
    name: str
    description: str
    examples: list[str]


class CannedResponseDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    last_modified: str
    value: str
    fields: str
    signals: Sequence[str]
    metadata: Mapping[str, JSONSerializable]
    field_dependencies: Sequence[str]
    note: str


class CannedResponseVectorDocument(TypedDict, total=False):
    id: ObjectId
    canned_response_id: ObjectId
    version: Version.String
    content: str
    checksum: Required[str]


class CannedResponseGroupAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    canned_response_id: CannedResponseId
    group_id: GroupId


class CannedResponseVectorStore(CannedResponseStore):
    VERSION = Version.from_string("0.8.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        vector_db: VectorDatabase,
        document_db: DocumentDatabase,
        embedder_provider: EmbedderProvider,
        allow_migration: bool = True,
        collections_prefix: str | None = None,
    ) -> None:
        self._id_generator = id_generator

        self._vector_db = vector_db
        self._database = document_db

        self._canreps_vector_collection: VectorCollection[CannedResponseVectorDocument]
        self._canreps_collection: DocumentCollection[CannedResponseDocument]
        self._canrep_tag_association_collection: DocumentCollection[
            CannedResponseGroupAssociationDocument
        ]
        self._allow_migration = allow_migration
        self._collections_prefix = collections_prefix
        self._lock = ReaderWriterLock()
        self._embedder_provider = embedder_provider
        self._embedder: Embedder

    async def _vector_document_loader(
        self, doc: VectorDocument
    ) -> Optional[CannedResponseVectorDocument]:
        # Fresh data baseline: no historical versions. Register converters here
        # when the document schema next changes.
        return await VectorDocumentMigrationHelper[CannedResponseVectorDocument](
            self,
            {},
        ).migrate(doc)

    async def _document_loader(self, doc: BaseDocument) -> Optional[CannedResponseDocument]:
        # Fresh data baseline: no historical versions. Register converters here
        # when the document schema next changes.
        return await DocumentMigrationHelper[CannedResponseDocument](
            self,
            {},
        ).migrate(doc)

    async def _association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[CannedResponseGroupAssociationDocument]:
        # Fresh data baseline: no historical versions. Register converters here
        # when the document schema next changes.
        return await DocumentMigrationHelper[CannedResponseGroupAssociationDocument](
            self,
            {},
        ).migrate(doc)

    async def __aenter__(self) -> Self:
        self._embedder = await self._embedder_provider()

        async with VectorDocumentStoreMigrationHelper(
            store=self,
            database=self._vector_db,
            allow_migration=self._allow_migration,
        ):
            self._canreps_vector_collection = await self._vector_db.get_or_create_collection(
                name=f"{self._collections_prefix}_canned_responses"
                if self._collections_prefix
                else "canned_responses",
                schema=CannedResponseVectorDocument,
                embedder=self._embedder,
                document_loader=self._vector_document_loader,
            )
            await self._canreps_vector_collection.ensure_indexes(
                [VectorCollectionIndex(field="canned_response_id")]
            )

        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
            collections_prefix=self._collections_prefix,
        ):
            self._canreps_collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_canned_responses"
                if self._collections_prefix
                else "canned_responses",
                schema=CannedResponseDocument,
                document_loader=self._document_loader,
            )

            self._canrep_tag_association_collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_canned_response_tag_associations"
                if self._collections_prefix
                else "canned_response_tag_associations",
                schema=CannedResponseGroupAssociationDocument,
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
            last_modified=canned_response_id.modified_utc.isoformat(),
            value=canned_response_id.value,
            fields=json.dumps(
                [
                    {"name": s.name, "description": s.description, "examples": s.examples}
                    for s in canned_response_id.fields
                ]
            ),
            signals=canned_response_id.signals,
            metadata=canned_response_id.metadata,
            field_dependencies=canned_response_id.field_dependencies,
            note=canned_response_id.note,
        )

    async def _deserialize_canned_response(
        self, canned_response_document: CannedResponseDocument
    ) -> CannedResponse:
        groups = [
            doc["group_id"]
            for doc in await self._canrep_tag_association_collection.find(
                {"canned_response_id": {"$eq": canned_response_document["id"]}}
            )
        ]

        return self._to_canned_response(canned_response_document, groups)

    async def _deserialize_many(
        self,
        documents: Sequence[CannedResponseDocument],
    ) -> Sequence[CannedResponse]:
        """Reconstruct a whole page of entities with one association query."""
        group_ids = await group_ids_by_entity(
            self._canrep_tag_association_collection,
            entity_field="canned_response_id",
            value_field="group_id",
            entity_ids=[d["id"] for d in documents],
        )

        return [self._to_canned_response(d, group_ids[d["id"]]) for d in documents]

    def _to_canned_response(
        self,
        canned_response_document: CannedResponseDocument,
        group_ids: Sequence[str],
    ) -> CannedResponse:
        return CannedResponse(
            id=CannedResponseId(canned_response_document["id"]),
            creation_utc=datetime.fromisoformat(canned_response_document["creation_utc"]),
            modified_utc=datetime.fromisoformat(
                canned_response_document.get(
                    "last_modified", canned_response_document["creation_utc"]
                )
            ),
            value=canned_response_document["value"],
            fields=[
                CannedResponseField(
                    name=d["name"], description=d["description"], examples=d["examples"]
                )
                for d in json.loads(canned_response_document["fields"])
            ],
            metadata=canned_response_document["metadata"],
            groups=[GroupId(group_id) for group_id in group_ids],
            signals=canned_response_document["signals"],
            field_dependencies=canned_response_document.get("field_dependencies", []),
            note=canned_response_document.get("note", ""),
        )

    def _list_canned_response_contents(self, canned_response: CannedResponse) -> list[str]:
        return [canned_response.value, *canned_response.signals]

    async def _insert_canned_response(
        self,
        canned_response: CannedResponse,
    ) -> CannedResponseDocument:
        insertion_tasks = []

        for content in self._list_canned_response_contents(canned_response):
            vec_doc = CannedResponseVectorDocument(
                id=ObjectId(canned_response.id),
                canned_response_id=ObjectId(canned_response.id),
                version=self.VERSION.to_string(),
                content=content,
                checksum=xxh3_checksum(content),
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
        metadata: Mapping[str, JSONSerializable] = {},
        groups: Optional[Sequence[GroupId]] = None,
        field_dependencies: Optional[Sequence[str]] = None,
        note: str = "",
    ) -> CannedResponse:
        self._validate_template(value)

        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            canrep_checksum = xxh3_checksum(f"{value}{fields}")
            canrep_id = CannedResponseId(self._id_generator.generate(canrep_checksum))

            canrep = CannedResponse(
                id=canrep_id,
                value=value,
                fields=fields or [],
                creation_utc=creation_utc,
                modified_utc=creation_utc,
                metadata=metadata,
                groups=groups or [],
                signals=signals or [],
                field_dependencies=field_dependencies or [],
                note=note,
            )

            await self._insert_canned_response(canrep)

            for group_id in groups or []:
                tag_checksum = xxh3_checksum(f"{canrep.id}{group_id}")

                await self._canrep_tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(self._id_generator.generate(tag_checksum)),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "canned_response_id": canrep.id,
                        "group_id": group_id,
                    }
                )

        return canrep

    def _validate_template(self, template: str) -> None:
        # Parse with the same sandboxed environment used to render, so validation
        # and rendering agree on what is a legal template.
        try:
            ImmutableSandboxedEnvironment().parse(template)
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
        if "value" in params:
            self._validate_template(params["value"])

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
                await self._canreps_vector_collection.delete_one(
                    filters={"id": {"$eq": v_doc["id"]}}
                )

            # Delete the existing main document
            await self._canreps_collection.delete_one(filters={"id": {"$eq": canned_response_id}})

            canrep = CannedResponse(
                id=CannedResponseId(canned_response_id),
                creation_utc=datetime.fromisoformat(doc["creation_utc"]),
                modified_utc=datetime.now(timezone.utc),
                value=params.get("value", existing_value.value),
                fields=params.get("fields", existing_value.fields),
                signals=params.get("signals", existing_value.signals),
                metadata=params.get("metadata", existing_value.metadata),
                groups=existing_value.groups,
                field_dependencies=params.get(
                    "field_dependencies", existing_value.field_dependencies
                ),
                note=params.get("note", existing_value.note),
            )

            doc = await self._insert_canned_response(canrep)

        return await self._deserialize_canned_response(doc)

    async def list_canned_responses(
        self,
        groups: Optional[Sequence[GroupId]] = None,
    ) -> Sequence[CannedResponse]:
        filters: Where = {}

        async with self._lock.reader_lock:
            if groups is not None:
                if len(groups) == 0:
                    canrep_ids = {
                        doc["canned_response_id"]
                        for doc in await self._canrep_tag_association_collection.find(filters={})
                    }
                    filters = (
                        {"$and": [{"id": {"$ne": id}} for id in canrep_ids]} if canrep_ids else {}
                    )
                else:
                    tag_filters: Where = {"$or": [{"group_id": {"$eq": group}} for group in groups]}
                    tag_associations = await self._canrep_tag_association_collection.find(
                        filters=tag_filters
                    )
                    canrep_ids = {assoc["canned_response_id"] for assoc in tag_associations}

                    if not canrep_ids:
                        return []

                    filters = {"$or": [{"id": {"$eq": id}} for id in canrep_ids]}

            canreps = await self._canreps_collection.find(filters=filters)

            return await self._deserialize_many(list(canreps))

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
    async def upsert_group(
        self,
        canned_response_id: CannedResponseId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            canrep = await self.read_canned_response(canned_response_id)

            if group_id in canrep.groups:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_checksum = xxh3_checksum(f"{canned_response_id}{group_id}")

            association_document: CannedResponseGroupAssociationDocument = {
                "id": ObjectId(self._id_generator.generate(association_checksum)),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "canned_response_id": canned_response_id,
                "group_id": group_id,
            }

            _ = await self._canrep_tag_association_collection.insert_one(
                document=association_document
            )

        return True

    @override
    async def remove_group(
        self,
        canned_response_id: CannedResponseId,
        group_id: GroupId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._canrep_tag_association_collection.delete_one(
                {
                    "canned_response_id": {"$eq": canned_response_id},
                    "group_id": {"$eq": group_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(group_id))

    @override
    async def filter_relevant_canned_responses(
        self,
        query: str,
        available_canned_responses: Sequence[CannedResponse],
        max_count: int,
    ) -> Sequence[CannedResponseRelevanceResult]:
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
                    hints={"group": "canned_responses"},
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

        canrep_docs: dict[str, CannedResponseDocument] = {
            d["id"]: d
            for d in await self._canreps_collection.find(
                {"id": {"$in": [r.document["canned_response_id"] for r in top_results]}}
            )
        }

        # Pair each vector hit with its own document before deserializing, so a
        # hit whose canned response is no longer stored drops out with its own
        # score rather than shifting every later score onto the wrong response.
        matched = [
            (canrep_docs[canned_response_id], vector_doc)
            for vector_doc in top_results
            if (canned_response_id := vector_doc.document["canned_response_id"]) in canrep_docs
        ]

        canned_responses = await self._deserialize_many([doc for doc, _ in matched])

        return [
            CannedResponseRelevanceResult(
                canned_response=canned_response,
                score=1.0 - vector_doc.distance,
            )
            for canned_response, (_, vector_doc) in zip(canned_responses, matched)
        ]


class CompositeCannedResponseStore(CannedResponseStore):
    def __init__(
        self,
        writable_store: CannedResponseStore,
        readable_stores: Sequence[CannedResponseStore],
    ) -> None:
        self._writable_store = writable_store
        self._readable_stores = readable_stores
        self._all_stores: Sequence[CannedResponseStore] = [writable_store, *readable_stores]

    @override
    async def create_canned_response(
        self,
        value: str,
        fields: Optional[Sequence[CannedResponseField]] = None,
        signals: Optional[Sequence[str]] = None,
        creation_utc: Optional[datetime] = None,
        metadata: Mapping[str, JSONSerializable] = {},
        groups: Optional[Sequence[GroupId]] = None,
        field_dependencies: Optional[Sequence[str]] = None,
        note: str = "",
    ) -> CannedResponse:
        return await self._writable_store.create_canned_response(
            value=value,
            fields=fields,
            signals=signals,
            creation_utc=creation_utc,
            metadata=metadata,
            groups=groups,
            field_dependencies=field_dependencies,
            note=note,
        )

    @override
    async def read_canned_response(
        self,
        canned_response_id: CannedResponseId,
    ) -> CannedResponse:
        results = await safe_gather(
            *[
                try_or_none(store.read_canned_response(canned_response_id))
                for store in self._all_stores
            ]
        )
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise ItemNotFoundError(item_id=UniqueId(canned_response_id))
        return result

    @override
    async def update_canned_response(
        self,
        canned_response_id: CannedResponseId,
        params: CannedResponseUpdateParams,
    ) -> CannedResponse:
        return await self._writable_store.update_canned_response(canned_response_id, params)

    @override
    async def delete_canned_response(
        self,
        canned_response_id: CannedResponseId,
    ) -> None:
        return await self._writable_store.delete_canned_response(canned_response_id)

    @override
    async def list_canned_responses(
        self,
        groups: Optional[Sequence[GroupId]] = None,
    ) -> Sequence[CannedResponse]:
        results = await safe_gather(
            *[store.list_canned_responses(groups=groups) for store in self._all_stores]
        )
        return list(chain.from_iterable(results))

    @override
    async def filter_relevant_canned_responses(
        self,
        query: str,
        available_canned_responses: Sequence[CannedResponse],
        max_count: int,
    ) -> Sequence[CannedResponseRelevanceResult]:
        return await self._writable_store.filter_relevant_canned_responses(
            query, available_canned_responses, max_count
        )

    @override
    async def upsert_group(
        self,
        canned_response_id: CannedResponseId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        return await self._writable_store.upsert_group(canned_response_id, group_id, creation_utc)

    @override
    async def remove_group(
        self,
        canned_response_id: CannedResponseId,
        group_id: GroupId,
    ) -> None:
        return await self._writable_store.remove_group(canned_response_id, group_id)
