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
import json
from typing import Awaitable, Callable, NewType, Optional, Sequence, TypedDict, cast
from typing_extensions import override, Self, Required

from parlant.core import async_utils
from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import ItemNotFoundError, Version, generate_id, UniqueId, md5_checksum
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
    queries: Sequence[str]
    tags: list[TagId]

    def __hash__(self) -> int:
        return hash(self.id)


class CapabilityUpdateParams(TypedDict, total=False):
    title: str
    description: str
    queries: Sequence[str]


class CapabilityStore:
    @abstractmethod
    async def create_capability(
        self,
        title: str,
        description: str,
        creation_utc: Optional[datetime] = None,
        queries: Optional[Sequence[str]] = None,
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


class _CapabilityDocument(TypedDict, total=False):
    id: ObjectId
    capability_id: ObjectId
    version: Version.String
    creation_utc: str
    content: str
    checksum: Required[str]
    title: str
    description: str
    queries: str


class _CapabilityTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    capability_id: CapabilityId
    tag_id: TagId


class CapabilityVectorStore(CapabilityStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        vector_db: VectorDatabase,
        document_db: DocumentDatabase,
        embedder_type_provider: Callable[[], Awaitable[type[Embedder]]],
        embedder_factory: EmbedderFactory,
        allow_migration: bool = True,
    ):
        self._vector_db = vector_db
        self._document_db = document_db
        self._allow_migration = allow_migration
        self._embedder_factory = embedder_factory
        self._embedder_type_provider = embedder_type_provider
        self._lock = ReaderWriterLock()
        self._collection: VectorCollection[_CapabilityDocument]
        self._tag_association_collection: DocumentCollection[_CapabilityTagAssociationDocument]
        self._embedder: Embedder

    async def _document_loader(self, doc: VectorBaseDocument) -> Optional[_CapabilityDocument]:
        if doc["version"] == self.VERSION.to_string():
            return cast(_CapabilityDocument, doc)
        return None

    async def _association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[_CapabilityTagAssociationDocument]:
        if doc["version"] == self.VERSION.to_string():
            return cast(_CapabilityTagAssociationDocument, doc)
        return None

    async def __aenter__(self) -> Self:
        embedder_type = await self._embedder_type_provider()
        self._embedder = self._embedder_factory.create_embedder(embedder_type)

        async with VectorDocumentStoreMigrationHelper(
            store=self,
            database=self._vector_db,
            allow_migration=self._allow_migration,
        ):
            self._collection = await self._vector_db.get_or_create_collection(
                name="capabilities",
                schema=_CapabilityDocument,
                embedder_type=embedder_type,
                document_loader=self._document_loader,
            )

        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._document_db,
            allow_migration=self._allow_migration,
        ):
            self._tag_association_collection = await self._document_db.get_or_create_collection(
                name="capability_tags",
                schema=_CapabilityTagAssociationDocument,
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

    @staticmethod
    def assemble_content(title: str, description: str, queries: Sequence[str]) -> str:
        content = f"{title}: {description}"
        if queries:
            content += "\nQueries: " + "; ".join(queries)
        return content

    def _serialize(
        self,
        capability: Capability,
        content: str,
    ) -> _CapabilityDocument:
        return _CapabilityDocument(
            id=ObjectId(generate_id()),
            capability_id=ObjectId(capability.id),
            version=self.VERSION.to_string(),
            creation_utc=capability.creation_utc.isoformat(),
            title=capability.title,
            description=capability.description,
            queries=json.dumps(list(capability.queries)),
            content=content,
            checksum=md5_checksum(content),
        )

    async def _deserialize(self, doc: _CapabilityDocument) -> Capability:
        tags = [
            d["tag_id"]
            for d in await self._tag_association_collection.find(
                {"capability_id": {"$eq": doc["capability_id"]}}
            )
        ]

        return Capability(
            id=CapabilityId(doc["capability_id"]),
            creation_utc=datetime.fromisoformat(doc["creation_utc"]),
            title=doc["title"],
            description=doc["description"],
            queries=json.loads(doc["queries"]),
            tags=tags,
        )

    def _list_capability_contents(self, capability: Capability) -> list[str]:
        return [f"{capability.title}: {capability.description}"] + list(capability.queries)

    async def _insert_capability(self, capability: Capability) -> _CapabilityDocument:
        insertion_tasks = []

        for content in self._list_capability_contents(capability):
            doc = self._serialize(capability, content)
            insertion_tasks.append(self._collection.insert_one(document=doc))

        await async_utils.safe_gather(*insertion_tasks)

        return doc

    @override
    async def create_capability(
        self,
        title: str,
        description: str,
        creation_utc: Optional[datetime] = None,
        queries: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Capability:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            queries = list(queries) if queries else []
            tags = list(tags) if tags else []

            capability_id = CapabilityId(generate_id())
            capability = Capability(
                id=capability_id,
                creation_utc=creation_utc,
                title=title,
                description=description,
                queries=queries,
                tags=tags,
            )

            await self._insert_capability(capability)

            for tag in tags:
                await self._tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(generate_id()),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "capability_id": capability.id,
                        "tag_id": tag,
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
            all_docs = await self._collection.find(
                filters={"capability_id": {"$eq": capability_id}}
            )

            if not all_docs:
                raise ItemNotFoundError(item_id=UniqueId(capability_id))

            for doc in all_docs:
                await self._collection.delete_one(filters={"id": {"$eq": doc["id"]}})

            title = params.get("title", doc["title"])
            description = params.get("description", doc["description"])
            queries = params.get("queries", cast(Sequence[str], list(json.loads(doc["queries"]))))

            capability = Capability(
                id=capability_id,
                creation_utc=datetime.fromisoformat(all_docs[0]["creation_utc"]),
                title=title,
                description=description,
                queries=queries,
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
            doc = await self._collection.find_one(filters={"capability_id": {"$eq": capability_id}})

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
                        doc["capability_id"]
                        for doc in await self._tag_association_collection.find(filters={})
                    }

                    if not capability_ids:
                        filters = {}

                    elif len(capability_ids) == 1:
                        filters = {"capability_id": {"$ne": capability_ids.pop()}}

                    else:
                        filters = {
                            "$and": [{"capability_id": {"$ne": id}} for id in capability_ids]
                        }

                else:
                    tag_filters: Where = {"$or": [{"tag_id": {"$eq": tag}} for tag in tags]}
                    tag_associations = await self._tag_association_collection.find(
                        filters=tag_filters
                    )

                    capability_ids = {assoc["capability_id"] for assoc in tag_associations}
                    if not capability_ids:
                        return []

                    if len(capability_ids) == 1:
                        filters = {"capability_id": {"$eq": capability_ids.pop()}}

                    else:
                        filters = {"$or": [{"capability_id": {"$eq": id}} for id in capability_ids]}

            docs = {}
            for d in await self._collection.find(filters=filters):
                if d["capability_id"] not in docs:
                    docs[d["capability_id"]] = d

            return [await self._deserialize(d) for d in docs.values()]

    @override
    async def delete_capability(
        self,
        capability_id: CapabilityId,
    ) -> None:
        async with self._lock.writer_lock:
            docs = await self._collection.find(filters={"capability_id": {"$eq": capability_id}})

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
                self._collection.find_similar_documents(
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

        unique_sdocs: dict[str, SimilarDocumentResult[_CapabilityDocument]] = {}

        for similar_doc in all_sdocs:
            if (
                similar_doc.document["capability_id"] not in unique_sdocs
                or unique_sdocs[similar_doc.document["capability_id"]].distance
                > similar_doc.distance
            ):
                unique_sdocs[similar_doc.document["capability_id"]] = similar_doc

            if len(unique_sdocs) >= max_count:
                break

        top_results = sorted(unique_sdocs.values(), key=lambda r: r.distance)[:max_count]

        return [await self._deserialize(r.document) for r in top_results]

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

            assoc_doc: _CapabilityTagAssociationDocument = {
                "id": ObjectId(generate_id()),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "capability_id": capability_id,
                "tag_id": tag_id,
            }

            _ = await self._tag_association_collection.insert_one(document=assoc_doc)
            doc = await self._collection.find_one({"capability_id": {"$eq": capability_id}})

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

            doc = await self._collection.find_one({"capability_id": {"$eq": capability_id}})

        if not doc:
            raise ItemNotFoundError(item_id=UniqueId(capability_id))
