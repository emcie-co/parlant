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
    VectorCollection,
    VectorDatabase,
)
from parlant.core.persistence.vector_database_helper import (
    VectorDocumentMigrationHelper,
    VectorDocumentStoreMigrationHelper,
    query_chunks,
)
from parlant.core.persistence.document_database import (
    DocumentCollection,
    DocumentDatabase,
    BaseDocument,
)
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper
from parlant.core.tags import TagId


TermId = NewType("TermId", str)


@dataclass(frozen=True)
class Term:
    id: TermId
    creation_utc: datetime
    name: str
    description: str
    synonyms: list[str]
    tags: list[TagId]

    def __repr__(self) -> str:
        term_string = f"Name: '{self.name}', Description: {self.description}"
        if self.synonyms:
            term_string += f", Synonyms: {', '.join(self.synonyms)}"
        return term_string

    def __hash__(self) -> int:
        return hash(self.id)


class TermUpdateParams(TypedDict, total=False):
    name: str
    description: str
    synonyms: Sequence[str]


class GlossaryStore:
    @abstractmethod
    async def create_term(
        self,
        name: str,
        description: str,
        creation_utc: Optional[datetime] = None,
        synonyms: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Term: ...

    @abstractmethod
    async def update_term(
        self,
        term_id: TermId,
        params: TermUpdateParams,
    ) -> Term: ...

    @abstractmethod
    async def read_term(
        self,
        term_id: TermId,
    ) -> Term: ...

    @abstractmethod
    async def list_terms(
        self,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Sequence[Term]: ...

    @abstractmethod
    async def delete_term(
        self,
        term_id: TermId,
    ) -> None: ...

    @abstractmethod
    async def find_relevant_terms(
        self,
        query: str,
        available_terms: Sequence[Term],
        max_terms: int = 20,
    ) -> Sequence[Term]: ...

    @abstractmethod
    async def upsert_tag(
        self,
        term_id: TermId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_tag(
        self,
        term_id: TermId,
        tag_id: TagId,
    ) -> None: ...


class TermDocument_v0_1_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    content: str
    checksum: Required[str]
    term_set: str
    creation_utc: str
    name: str
    description: str
    synonyms: Optional[str]


class _TermDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    content: str
    checksum: Required[str]
    creation_utc: str
    name: str
    description: str
    synonyms: Optional[str]


class TermTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    term_id: TermId
    tag_id: TagId


class GlossaryVectorStore(GlossaryStore):
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

        self._collection: VectorCollection[_TermDocument]
        self._association_collection: DocumentCollection[TermTagAssociationDocument]

        self._allow_migration = allow_migration

        self._embedder_factory = embedder_factory
        self._embedder_type_provider = embedder_type_provider
        self._embedder: Embedder

        self._lock = ReaderWriterLock()

    async def _document_loader(self, document: VectorBaseDocument) -> Optional[_TermDocument]:
        async def v0_1_0_to_v0_2_0(document: VectorBaseDocument) -> Optional[VectorBaseDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        return await VectorDocumentMigrationHelper[_TermDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_2_0,
            },
        ).migrate(document)

    async def _association_document_loader(
        self, document: BaseDocument
    ) -> Optional[TermTagAssociationDocument]:
        return cast(TermTagAssociationDocument, document)

    async def __aenter__(self) -> Self:
        embedder_type = await self._embedder_type_provider()

        self._embedder = self._embedder_factory.create_embedder(embedder_type)

        async with VectorDocumentStoreMigrationHelper(
            store=self,
            database=self._vector_db,
            allow_migration=self._allow_migration,
        ):
            self._collection = await self._vector_db.get_or_create_collection(
                name="glossary",
                schema=_TermDocument,
                embedder_type=embedder_type,
                document_loader=self._document_loader,
            )

        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._document_db,
            allow_migration=self._allow_migration,
        ):
            self._association_collection = await self._document_db.get_or_create_collection(
                name="glossary_tags",
                schema=TermTagAssociationDocument,
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
        term: Term,
        content: str,
        checksum: str,
    ) -> _TermDocument:
        return _TermDocument(
            id=ObjectId(term.id),
            version=self.VERSION.to_string(),
            content=content,
            checksum=checksum,
            creation_utc=term.creation_utc.isoformat(),
            name=term.name,
            description=term.description,
            synonyms=(", ").join(term.synonyms) if term.synonyms is not None else "",
        )

    async def _deserialize(self, term_document: _TermDocument) -> Term:
        tags = await self._association_collection.find(
            filters={"term_id": {"$eq": term_document["id"]}}
        )

        return Term(
            id=TermId(term_document["id"]),
            creation_utc=datetime.fromisoformat(term_document["creation_utc"]),
            name=term_document["name"],
            description=term_document["description"],
            synonyms=term_document["synonyms"].split(", ") if term_document["synonyms"] else [],
            tags=[TagId(t["tag_id"]) for t in tags],
        )

    @override
    async def create_term(
        self,
        name: str,
        description: str,
        creation_utc: Optional[datetime] = None,
        synonyms: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Term:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            content = self._assemble_term_content(
                name=name,
                description=description,
                synonyms=synonyms,
            )

            term_checksum = md5_checksum(f"{name}{description}{synonyms}")

            term = Term(
                id=TermId(self._id_generator.generate(term_checksum)),
                creation_utc=creation_utc,
                name=name,
                description=description,
                synonyms=list(synonyms) if synonyms else [],
                tags=list(tags) if tags else [],
            )

            await self._collection.insert_one(
                document=self._serialize(
                    term=term,
                    content=content,
                    checksum=md5_checksum(content),
                )
            )

            for tag_id in tags or []:
                tag_checksum = md5_checksum(f"{term.id}{tag_id}")

                await self._association_collection.insert_one(
                    document={
                        "id": ObjectId(self._id_generator.generate(tag_checksum)),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "term_id": term.id,
                        "tag_id": tag_id,
                    }
                )
        return term

    @override
    async def read_term(
        self,
        term_id: TermId,
    ) -> Term:
        async with self._lock.reader_lock:
            term_document = await self._collection.find_one(filters={"id": {"$eq": term_id}})

        if not term_document:
            raise ItemNotFoundError(item_id=UniqueId(term_id))

        return await self._deserialize(term_document=term_document)

    @override
    async def update_term(
        self,
        term_id: TermId,
        params: TermUpdateParams,
    ) -> Term:
        async with self._lock.writer_lock:
            document_to_update = await self._collection.find_one(filters={"id": {"$eq": term_id}})

            if not document_to_update:
                raise ItemNotFoundError(item_id=UniqueId(term_id))

            assert "name" in document_to_update
            assert "description" in document_to_update
            assert "synonyms" in document_to_update

            name = params.get("name", document_to_update["name"])
            description = params.get("description", document_to_update["description"])
            synonyms = params.get("synonyms", document_to_update["synonyms"])

            content = self._assemble_term_content(
                name=name,
                description=description,
                synonyms=synonyms,
            )

            update_result = await self._collection.update_one(
                filters={"id": {"$eq": term_id}},
                params={
                    "content": content,
                    "name": name,
                    "description": description,
                    "synonyms": ", ".join(synonyms) if synonyms else "",
                    "checksum": md5_checksum(content),
                },
            )

        assert update_result.updated_document

        return await self._deserialize(term_document=update_result.updated_document)

    @override
    async def list_terms(
        self,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Sequence[Term]:
        filters: Where = {}

        async with self._lock.reader_lock:
            if tags is not None:
                if len(tags) == 0:
                    term_ids = {
                        doc["term_id"]
                        for doc in await self._association_collection.find(filters={})
                    }
                    if not term_ids:
                        filters = {}
                    elif len(term_ids) == 1:
                        filters = {"id": {"$ne": term_ids.pop()}}
                    else:
                        filters = {"$and": [{"id": {"$ne": id}} for id in term_ids]}

                else:
                    tag_filters: Where = {"$or": [{"tag_id": {"$eq": tag}} for tag in tags]}
                    tag_associations = await self._association_collection.find(filters=tag_filters)
                    term_ids = {assoc["term_id"] for assoc in tag_associations}

                    if not term_ids:
                        return []

                    if len(term_ids) == 1:
                        filters = {"id": {"$eq": term_ids.pop()}}
                    else:
                        filters = {"$or": [{"id": {"$eq": id}} for id in term_ids]}

            return [
                await self._deserialize(d) for d in await self._collection.find(filters=filters)
            ]

    @override
    async def delete_term(
        self,
        term_id: TermId,
    ) -> None:
        async with self._lock.writer_lock:
            term_document = await self._collection.find_one(filters={"id": {"$eq": term_id}})
            term_tag_associations = await self._association_collection.find(
                filters={"term_id": {"$eq": term_id}}
            )

            if not term_document:
                raise ItemNotFoundError(item_id=UniqueId(term_id))

            await self._collection.delete_one(filters={"id": {"$eq": term_id}})
            for tag_association in term_tag_associations:
                await self._association_collection.delete_one(
                    filters={"id": {"$eq": tag_association["id"]}}
                )

    @override
    async def find_relevant_terms(
        self,
        query: str,
        available_terms: Sequence[Term],
        max_terms: int = 20,
    ) -> Sequence[Term]:
        if not available_terms:
            return []

        async with self._lock.reader_lock:
            queries = await query_chunks(query, self._embedder)

            filters: Where = {"id": {"$in": [str(t.id) for t in available_terms]}}

            tasks = [
                self._collection.find_similar_documents(
                    filters=filters,
                    query=q,
                    k=max_terms,
                )
                for q in queries
            ]

        all_results = chain.from_iterable(await async_utils.safe_gather(*tasks))
        unique_results = list(set(all_results))
        top_results = sorted(unique_results, key=lambda r: r.distance)[:max_terms]

        return [await self._deserialize(r.document) for r in top_results]

    def _assemble_term_content(
        self,
        name: str,
        description: str,
        synonyms: Optional[Sequence[str]],
    ) -> str:
        content = f"{name}"

        if synonyms:
            content += f", {', '.join(synonyms)}"

        content += f": {description}"

        return content

    async def upsert_tag(
        self,
        term_id: TermId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            term = await self.read_term(term_id)

            if tag_id in term.tags:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_checksum = md5_checksum(f"{term_id}{tag_id}")

            association_document: TermTagAssociationDocument = {
                "id": ObjectId(self._id_generator.generate(association_checksum)),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "term_id": term_id,
                "tag_id": tag_id,
            }

            _ = await self._association_collection.insert_one(document=association_document)

            term_document = await self._collection.find_one({"id": {"$eq": term_id}})

        if not term_document:
            raise ItemNotFoundError(item_id=UniqueId(term_id))

        return True

    @override
    async def remove_tag(
        self,
        term_id: TermId,
        tag_id: TagId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._association_collection.delete_one(
                {
                    "term_id": {"$eq": term_id},
                    "tag_id": {"$eq": tag_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(tag_id))

            term_document = await self._collection.find_one({"id": {"$eq": term_id}})

        if not term_document:
            raise ItemNotFoundError(item_id=UniqueId(term_id))
