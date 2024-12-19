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
from typing import NewType, Optional, Sequence, TypedDict
from typing_extensions import override, Self

from parlant.core import async_utils
from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import ItemNotFoundError, Version, generate_id, UniqueId
from parlant.core.persistence.common import ObjectId
from parlant.core.nlp.embedding import Embedder, EmbedderFactory
from parlant.core.persistence.vector_database import VectorCollection, VectorDatabase


TermId = NewType("TermId", str)


@dataclass(frozen=True)
class Term:
    id: TermId
    creation_utc: datetime
    name: str
    description: str
    synonyms: list[str]

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
        term_set: str,
        name: str,
        description: str,
        creation_utc: Optional[datetime] = None,
        synonyms: Optional[Sequence[str]] = None,
    ) -> Term: ...

    @abstractmethod
    async def update_term(
        self,
        term_set: str,
        term_id: TermId,
        params: TermUpdateParams,
    ) -> Term: ...

    @abstractmethod
    async def read_term(
        self,
        term_set: str,
        term_id: TermId,
    ) -> Term: ...

    @abstractmethod
    async def list_terms(
        self,
        term_set: str,
    ) -> Sequence[Term]: ...

    @abstractmethod
    async def delete_term(
        self,
        term_set: str,
        term_id: TermId,
    ) -> None: ...

    @abstractmethod
    async def find_relevant_terms(
        self,
        term_set: str,
        query: str,
    ) -> Sequence[Term]: ...


class _TermDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    term_set: str
    creation_utc: str
    name: str
    description: str
    synonyms: Optional[str]
    content: str


class GlossaryVectorStore(GlossaryStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        vector_db: VectorDatabase,
        embedder_type: type[Embedder],
        embedder_factory: EmbedderFactory,
    ):
        self._vector_db = vector_db
        self._collection: VectorCollection[_TermDocument]
        self._embedder = embedder_factory.create_embedder(embedder_type)
        self._embedder_type = embedder_type

        self._lock = ReaderWriterLock()

    async def __aenter__(self) -> Self:
        self._collection = await self._vector_db.get_or_create_collection(
            name="glossary",
            schema=_TermDocument,
            embedder_type=self._embedder_type,
        )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    def _serialize(self, term: Term, term_set: str, content: str) -> _TermDocument:
        return _TermDocument(
            id=ObjectId(term.id),
            version=self.VERSION.to_string(),
            term_set=term_set,
            creation_utc=term.creation_utc.isoformat(),
            name=term.name,
            description=term.description,
            synonyms=(", ").join(term.synonyms) if term.synonyms is not None else "",
            content=content,
        )

    def _deserialize(self, term_document: _TermDocument) -> Term:
        return Term(
            id=TermId(term_document["id"]),
            creation_utc=datetime.fromisoformat(term_document["creation_utc"]),
            name=term_document["name"],
            description=term_document["description"],
            synonyms=term_document["synonyms"].split(", ") if term_document["synonyms"] else [],
        )

    @override
    async def create_term(
        self,
        term_set: str,
        name: str,
        description: str,
        creation_utc: Optional[datetime] = None,
        synonyms: Optional[Sequence[str]] = None,
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
            )

            await self._collection.insert_one(document=self._serialize(term, term_set, content))

        return term

    @override
    async def update_term(
        self,
        term_set: str,
        term_id: TermId,
        params: TermUpdateParams,
    ) -> Term:
        async with self._lock.writer_lock:
            document_to_update = await self._collection.find_one(
                {"$and": [{"term_set": {"$eq": term_set}}, {"id": {"$eq": term_id}}]}
            )

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
                filters={"$and": [{"term_set": {"$eq": term_set}}, {"id": {"$eq": term_id}}]},
                params={
                    "content": content,
                    "name": name,
                    "description": description,
                    "synonyms": ", ".join(synonyms) if synonyms else "",
                },
            )

        assert update_result.updated_document

        return self._deserialize(term_document=update_result.updated_document)

    @override
    async def read_term(
        self,
        term_set: str,
        term_id: TermId,
    ) -> Term:
        async with self._lock.reader_lock:
            term_document = await self._collection.find_one(
                filters={"$and": [{"term_set": {"$eq": term_set}}, {"id": {"$eq": term_id}}]}
            )

        if not term_document:
            raise ItemNotFoundError(item_id=UniqueId(term_id), message=f"term_set={term_set}")

        return self._deserialize(term_document=term_document)

    @override
    async def list_terms(
        self,
        term_set: str,
    ) -> Sequence[Term]:
        async with self._lock.reader_lock:
            return [
                self._deserialize(term_document=d)
                for d in await self._collection.find(filters={"term_set": {"$eq": term_set}})
            ]

    @override
    async def delete_term(
        self,
        term_set: str,
        term_id: TermId,
    ) -> None:
        async with self._lock.writer_lock:
            term_document = await self._collection.find_one(
                filters={"$and": [{"term_set": {"$eq": term_set}}, {"id": {"$eq": term_id}}]}
            )

            if not term_document:
                raise ItemNotFoundError(item_id=UniqueId(term_id))

            await self._collection.delete_one(
                filters={"$and": [{"term_set": {"$eq": term_set}}, {"id": {"$eq": term_id}}]}
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
        term_set: str,
        query: str,
        max_terms: int = 20,
    ) -> Sequence[Term]:
        async with self._lock.reader_lock:
            queries = await self._query_chunks(query)

            tasks = [
                self._collection.find_similar_documents(
                    filters={"term_set": {"$eq": term_set}},
                    query=q,
                    k=max_terms,
                )
                for q in queries
            ]

        all_results = chain.from_iterable(await async_utils.safe_gather(*tasks))
        unique_results = list(set(all_results))
        top_results = sorted(unique_results, key=lambda r: r.distance)[:max_terms]

        return [self._deserialize(r.document) for r in top_results]

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
