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

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from typing import NewType, Optional, Sequence, TypedDict, cast
from typing_extensions import override, Self, Required

from parlant.core import async_utils
from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import ItemNotFoundError, Version, generate_id, UniqueId, md5_checksum
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
    synonyms: Sequence[str]
    tags: Sequence[TagId]

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
        tags: Optional[Sequence[TagId]] = None,
    ) -> Sequence[Term]: ...

    @abstractmethod
    async def add_tag(
        self,
        term_id: TermId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> Term: ...

    @abstractmethod
    async def remove_tag(
        self,
        term_id: TermId,
        tag_id: TagId,
    ) -> Term: ...


class _TermDocument_v0_1_0(TypedDict, total=False):
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


class _TermTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    term_id: TermId
    tag_id: TagId


class GlossaryVectorStore(GlossaryStore):
    VERSION = Version.from_string("0.2.0")

    def __init__(
        self,
        vector_db: VectorDatabase,
        document_db: DocumentDatabase,
        embedder_type: type[Embedder],
        embedder_factory: EmbedderFactory,
        allow_migration: bool = True,
    ):
        self._vector_db = vector_db
        self._document_db = document_db

        self._collection: VectorCollection[_TermDocument]
        self._association_collection: DocumentCollection[_TermTagAssociationDocument]

        self._allow_migration = allow_migration
        self._embedder = embedder_factory.create_embedder(embedder_type)
        self._embedder_type = embedder_type

        self._lock = ReaderWriterLock()

    async def _document_loader(self, document: VectorBaseDocument) -> Optional[_TermDocument]:
        async def v0_1_0_to_v_0_2_0(document: VectorBaseDocument) -> Optional[VectorBaseDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        return await VectorDocumentMigrationHelper[_TermDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v_0_2_0,
            },
        ).migrate(document)

    async def _association_document_loader(
        self, document: BaseDocument
    ) -> Optional[_TermTagAssociationDocument]:
        return cast(_TermTagAssociationDocument, document)

    async def __aenter__(self) -> Self:
        async with VectorDocumentStoreMigrationHelper(
            store=self,
            database=self._vector_db,
            allow_migration=self._allow_migration,
        ):
            self._collection = await self._vector_db.get_or_create_collection(
                name="glossary",
                schema=_TermDocument,
                embedder_type=self._embedder_type,
                document_loader=self._document_loader,
            )
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._document_db,
            allow_migration=self._allow_migration,
        ):
            self._association_collection = await self._document_db.get_or_create_collection(
                name="glossary_tag_associations",
                schema=_TermTagAssociationDocument,
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

            term = Term(
                id=TermId(generate_id()),
                creation_utc=creation_utc,
                name=name,
                description=description,
                synonyms=list(synonyms) if synonyms else [],
                tags=tags or [],
            )

            await self._collection.insert_one(
                document=self._serialize(
                    term=term,
                    content=content,
                    checksum=md5_checksum(content),
                )
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
                    filters = {"$and": [{"id": {"$ne": id}} for id in term_ids]} if term_ids else {}
                else:
                    tag_filters: Where = {"$or": [{"tag_id": {"$eq": tag}} for tag in tags]}
                    tag_associations = await self._association_collection.find(filters=tag_filters)
                    term_ids = {assoc["term_id"] for assoc in tag_associations}

                    if not term_ids:
                        return []

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

    async def _query_chunks(self, query: str) -> list[str]:
        max_length = self._embedder.max_tokens // 5
        total_token_count = await self._embedder.tokenizer.estimate_token_count(query)

        words = query.split()
        total_word_count = len(words)

        tokens_per_word = total_token_count / total_word_count

        words_per_chunk = max(int(max_length / tokens_per_word), 1)

        chunks = []
        for i in range(0, total_word_count, words_per_chunk):
            chunk_words = words[i : i + words_per_chunk]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)

        return [
            text if await self._embedder.tokenizer.estimate_token_count(text) else ""
            for text in chunks
        ]

    @override
    async def find_relevant_terms(
        self,
        query: str,
        tags: Optional[Sequence[TagId]] = None,
        max_terms: int = 20,
    ) -> Sequence[Term]:
        async with self._lock.reader_lock:
            queries = await self._query_chunks(query)

            filters: Where = {}

            term_ids = await self.list_terms(tags=tags)
            if term_ids:
                filters = {"id": {"$in": [str(id) for id in term_ids]}}

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

    async def add_tag(
        self,
        term_id: TermId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> Term:
        async with self._lock.writer_lock:
            term = await self.read_term(term_id)

            if tag_id in term.tags:
                return term

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_document: _TermTagAssociationDocument = {
                "id": ObjectId(generate_id()),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "term_id": term_id,
                "tag_id": tag_id,
            }

            _ = await self._association_collection.insert_one(document=association_document)

            term_document = await self._collection.find_one({"id": {"$eq": term_id}})

        if not term_document:
            raise ItemNotFoundError(item_id=UniqueId(term_id))

        return await self._deserialize(term_document=term_document)

    @override
    async def remove_tag(
        self,
        term_id: TermId,
        tag_id: TagId,
    ) -> Term:
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

        return await self._deserialize(term_document=term_document)
