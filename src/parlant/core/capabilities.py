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

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from typing import Awaitable, Callable, NewType, Optional, Sequence, TypedDict, cast
from typing_extensions import override, Self, Required

from parlant.core import async_utils
from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import ItemNotFoundError, Version, IdGenerator, UniqueId, md5_checksum
from parlant.core.persistence.common import ObjectId, Where
from parlant.core.nlp.embedding import Embedder, EmbedderFactory
from parlant.core.persistence.vector_database import (
    BaseDocument as VectorBaseDocument,
    SimilarDocumentResult,
    VectorCollection,
    VectorDatabase,
)
from parlant.core.persistence.vector_database_helper import (
    VectorDocumentStoreMigrationHelper,
    calculate_min_vectors_for_max_item_count,
    query_chunks,
)
from parlant.core.persistence.document_database import (
    DocumentCollection,
    DocumentDatabase,
    BaseDocument,
)
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper
from parlant.core.tags import TagId


CapabilityId = NewType("CapabilityId", str)


@dataclass(frozen=True)
class Capability:
    id: CapabilityId
    creation_utc: datetime
    title: str
    description: str
    signals: Sequence[str]
    tags: list[TagId]

    def __hash__(self) -> int:
        return hash(self.id)


class CapabilityUpdateParams(TypedDict, total=False):
    title: str
    description: str
    signals: Sequence[str]


class CapabilityStore:
    @abstractmethod
    async def create_capability(
        self,
        title: str,
        description: str,
        creation_utc: Optional[datetime] = None,
        signals: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Capability: ...

    @abstractmethod
    async def update_capability(
        self,
        capability_id: CapabilityId,
        params: CapabilityUpdateParams,
    ) -> Capability: ...

    @abstractmethod
    async def read_capability(
        self,
        capability_id: CapabilityId,
    ) -> Capability: ...

    @abstractmethod
    async def list_capabilities(
        self,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Sequence[Capability]: ...

    @abstractmethod
    async def delete_capability(
        self,
        capability_id: CapabilityId,
    ) -> None: ...

    @abstractmethod
    async def find_relevant_capabilities(
        self,
        query: str,
        available_capabilities: Sequence[Capability],
        max_count: int,
    ) -> Sequence[Capability]: ...

    @abstractmethod
    async def upsert_tag(
        self,
        capability_id: CapabilityId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_tag(
        self,
        capability_id: CapabilityId,
        tag_id: TagId,
    ) -> None: ...


class CapabilityDocument_v0_1_0(TypedDict, total=False):
    id: ObjectId
    capability_id: ObjectId
    version: Version.String
    creation_utc: str
    content: str
    checksum: Required[str]
    title: str
    description: str
    queries: str


class CapabilityVectorDocument(TypedDict, total=False):
    id: ObjectId
    capability_id: ObjectId
    version: Version.String
    content: str
    checksum: Required[str]


class CapabilityDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    title: str
    description: str
    signals: Sequence[str]


class CapabilityTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    capability_id: CapabilityId
    tag_id: TagId


class CapabilityVectorStore(CapabilityStore):
    VERSION = Version.from_string("0.2.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        vector_db: VectorDatabase,
        document_db: DocumentDatabase,
        embedder_type_provider: Callable[[], Awaitable[type[Embedder]]],
        embedder_factory: EmbedderFactory,
        allow_migration: bool = True,
    ):
        self._id_generator = id_generator

        self._vector_db = vector_db
        self._document_db = document_db
        self._allow_migration = allow_migration
        self._vector_collection: VectorCollection[CapabilityVectorDocument]
        self._collection: DocumentCollection[CapabilityDocument]
        self._tag_association_collection: DocumentCollection[CapabilityTagAssociationDocument]

        self._embedder_factory = embedder_factory
        self._embedder_type_provider = embedder_type_provider
        self._embedder: Embedder

        self._lock = ReaderWriterLock()

    async def _vector_document_loader(
        self, doc: VectorBaseDocument
    ) -> Optional[CapabilityVectorDocument]:
        if doc["version"] == self.VERSION.to_string():
            return cast(CapabilityVectorDocument, doc)
        return None

    async def _document_loader(self, doc: BaseDocument) -> Optional[CapabilityDocument]:
        if doc["version"] == self.VERSION.to_string():
            return cast(CapabilityDocument, doc)
        return None

    async def _association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[CapabilityTagAssociationDocument]:
        if doc["version"] == self.VERSION.to_string():
            return cast(CapabilityTagAssociationDocument, doc)
        return None

    async def __aenter__(self) -> Self:
        embedder_type = await self._embedder_type_provider()
        self._embedder = self._embedder_factory.create_embedder(embedder_type)

        async with VectorDocumentStoreMigrationHelper(
            store=self,
            database=self._vector_db,
            allow_migration=self._allow_migration,
        ):
            self._vector_collection = await self._vector_db.get_or_create_collection(
                name="capabilities",
                schema=CapabilityVectorDocument,
                embedder_type=embedder_type,
                document_loader=self._vector_document_loader,
            )

        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._document_db,
            allow_migration=self._allow_migration,
        ):
            self._collection = await self._document_db.get_or_create_collection(
                name="capabilities",
                schema=CapabilityDocument,
                document_loader=self._document_loader,
            )

            self._tag_association_collection = await self._document_db.get_or_create_collection(
                name="capability_tags",
                schema=CapabilityTagAssociationDocument,
                document_loader=self._association_document_loader,
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
        capability: Capability,
    ) -> CapabilityDocument:
        return CapabilityDocument(
            id=ObjectId(capability.id),
            version=self.VERSION.to_string(),
            creation_utc=capability.creation_utc.isoformat(),
            title=capability.title,
            description=capability.description,
            signals=capability.signals,
        )

    async def _deserialize(self, doc: CapabilityDocument) -> Capability:
        tags = [
            d["tag_id"]
            for d in await self._tag_association_collection.find(
                {"capability_id": {"$eq": doc["id"]}}
            )
        ]

        return Capability(
            id=CapabilityId(doc["id"]),
            creation_utc=datetime.fromisoformat(doc["creation_utc"]),
            title=doc["title"],
            description=doc["description"],
            signals=doc["signals"],
            tags=tags,
        )

    def _list_capability_contents(self, capability: Capability) -> list[str]:
        return [f"{capability.title}: {capability.description}"] + list(capability.signals)

    async def _insert_capability(self, capability: Capability) -> CapabilityDocument:
        insertion_tasks = []

        for content in self._list_capability_contents(capability):
            doc_id = self._id_generator.generate(md5_checksum(content))

            vec_doc = CapabilityVectorDocument(
                id=ObjectId(doc_id),
                capability_id=ObjectId(capability.id),
                version=self.VERSION.to_string(),
                content=content,
                checksum=md5_checksum(content),
            )

            insertion_tasks.append(self._vector_collection.insert_one(document=vec_doc))

        await async_utils.safe_gather(*insertion_tasks)

        doc = self._serialize(capability)
        await self._collection.insert_one(document=doc)

        return doc

    @override
    async def create_capability(
        self,
        title: str,
        description: str,
        creation_utc: Optional[datetime] = None,
        signals: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Capability:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            signals = list(signals) if signals else []
            tags = list(tags) if tags else []

            capability_checksum = md5_checksum(f"{title}{description}{signals}{tags}")

            capability_id = CapabilityId(self._id_generator.generate(capability_checksum))
            capability = Capability(
                id=capability_id,
                creation_utc=creation_utc,
                title=title,
                description=description,
                signals=signals,
                tags=tags,
            )

            await self._insert_capability(capability)

            for tag_id in tags:
                tag_checksum = md5_checksum(f"{capability_id}{tag_id}")

                await self._tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(self._id_generator.generate(tag_checksum)),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "capability_id": capability.id,
                        "tag_id": tag_id,
                    }
                )

        return capability

    @override
    async def update_capability(
        self,
        capability_id: CapabilityId,
        params: CapabilityUpdateParams,
    ) -> Capability:
        async with self._lock.writer_lock:
            all_docs = await self._collection.find(filters={"id": {"$eq": capability_id}})

            if not all_docs:
                raise ItemNotFoundError(item_id=UniqueId(capability_id))

            for doc in all_docs:
                await self._collection.delete_one(filters={"id": {"$eq": doc["id"]}})

            title = params.get("title", doc["title"])
            description = params.get("description", doc["description"])
            signals = params.get("signals", cast(Sequence[str], list(doc["signals"])))

            capability = Capability(
                id=capability_id,
                creation_utc=datetime.fromisoformat(all_docs[0]["creation_utc"]),
                title=title,
                description=description,
                signals=signals,
                tags=[],
            )

            doc = await self._insert_capability(capability)

        return await self._deserialize(doc)

    @override
    async def read_capability(
        self,
        capability_id: CapabilityId,
    ) -> Capability:
        async with self._lock.reader_lock:
            doc = await self._collection.find_one(filters={"id": {"$eq": capability_id}})

        if not doc:
            raise ItemNotFoundError(item_id=UniqueId(capability_id))

        return await self._deserialize(doc)

    @override
    async def list_capabilities(
        self,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Sequence[Capability]:
        filters: Where = {}
        async with self._lock.reader_lock:
            if tags is not None:
                if len(tags) == 0:
                    capability_ids = {
                        doc["id"] for doc in await self._tag_association_collection.find(filters={})
                    }

                    if not capability_ids:
                        filters = {}

                    elif len(capability_ids) == 1:
                        filters = {"id": {"$ne": capability_ids.pop()}}

                    else:
                        filters = {"$and": [{"id": {"$ne": id}} for id in capability_ids]}

                else:
                    tag_filters: Where = {"$or": [{"tag_id": {"$eq": tag}} for tag in tags]}
                    tag_associations = await self._tag_association_collection.find(
                        filters=tag_filters
                    )

                    capability_ids = {
                        ObjectId(assoc["capability_id"]) for assoc in tag_associations
                    }
                    if not capability_ids:
                        return []

                    if len(capability_ids) == 1:
                        filters = {"id": {"$eq": capability_ids.pop()}}

                    else:
                        filters = {"$or": [{"id": {"$eq": id}} for id in capability_ids]}

            docs = {}
            for d in await self._collection.find(filters=filters):
                if d["id"] not in docs:
                    docs[d["id"]] = d

            return [await self._deserialize(d) for d in docs.values()]

    @override
    async def delete_capability(
        self,
        capability_id: CapabilityId,
    ) -> None:
        async with self._lock.writer_lock:
            docs = await self._collection.find(filters={"id": {"$eq": capability_id}})

            tag_associations = await self._tag_association_collection.find(
                filters={"capability_id": {"$eq": capability_id}}
            )

            if not docs:
                raise ItemNotFoundError(item_id=UniqueId(capability_id))

            for doc in docs:
                await self._collection.delete_one(filters={"id": {"$eq": doc["id"]}})

            for tag_assoc in tag_associations:
                await self._tag_association_collection.delete_one(
                    filters={"id": {"$eq": tag_assoc["id"]}}
                )

    @override
    async def find_relevant_capabilities(
        self,
        query: str,
        available_capabilities: Sequence[Capability],
        max_count: int,
    ) -> Sequence[Capability]:
        if not available_capabilities:
            return []

        async with self._lock.reader_lock:
            queries = await query_chunks(query, self._embedder)
            filters: Where = {"capability_id": {"$in": [str(c.id) for c in available_capabilities]}}

            tasks = [
                self._vector_collection.find_similar_documents(
                    filters=filters,
                    query=q,
                    k=calculate_min_vectors_for_max_item_count(
                        items=available_capabilities,
                        count_item_vectors=lambda c: len(self._list_capability_contents(c)),
                        max_items_to_return=max_count,
                    ),
                )
                for q in queries
            ]

        all_sdocs = chain.from_iterable(await async_utils.safe_gather(*tasks))

        unique_sdocs: dict[str, SimilarDocumentResult[CapabilityVectorDocument]] = {}

        for similar_doc in all_sdocs:
            if (
                similar_doc.document["capability_id"] not in unique_sdocs
                or unique_sdocs[similar_doc.document["capability_id"]].distance
                > similar_doc.distance
            ):
                unique_sdocs[similar_doc.document["capability_id"]] = similar_doc

            if len(unique_sdocs) >= max_count:
                break

        capability_ids = [
            r.document["capability_id"]
            for r in sorted(unique_sdocs.values(), key=lambda r: r.distance)[:max_count]
        ]

        return [
            await self._deserialize(doc)
            for doc in await self._collection.find(
                filters={"id": {"$in": [ObjectId(cid) for cid in capability_ids]}}
            )
        ]

    @override
    async def upsert_tag(
        self,
        capability_id: CapabilityId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            capability = await self.read_capability(capability_id)

            if tag_id in capability.tags:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            tag_checksum = md5_checksum(f"{capability_id}{tag_id}")

            assoc_doc: CapabilityTagAssociationDocument = {
                "id": ObjectId(self._id_generator.generate(tag_checksum)),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "capability_id": capability_id,
                "tag_id": tag_id,
            }

            _ = await self._tag_association_collection.insert_one(document=assoc_doc)
            doc = await self._collection.find_one({"id": {"$eq": capability_id}})

        if not doc:
            raise ItemNotFoundError(item_id=UniqueId(capability_id))

        return True

    @override
    async def remove_tag(
        self,
        capability_id: CapabilityId,
        tag_id: TagId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._tag_association_collection.delete_one(
                {
                    "capability_id": {"$eq": capability_id},
                    "tag_id": {"$eq": tag_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(tag_id))

            doc = await self._collection.find_one({"id": {"$eq": capability_id}})

        if not doc:
            raise ItemNotFoundError(item_id=UniqueId(capability_id))
