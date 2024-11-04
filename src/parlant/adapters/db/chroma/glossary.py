import asyncio
from datetime import datetime, timezone
from itertools import chain
from typing import Optional, Sequence, TypedDict

from parlant.adapters.db.chroma.database import ChromaDatabase
from parlant.core.common import (
    ItemNotFoundError,
    UniqueId,
    Version,
    generate_id,
)
from parlant.core.nlp.embedding import Embedder
from parlant.core.persistence.document_database import ObjectId
from parlant.core.glossary import Term, TermId, TermUpdateParams, GlossaryStore


class _TermDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    term_set: str
    creation_utc: str
    name: str
    description: str
    synonyms: Optional[str]
    content: str


class GlossaryChromaStore(GlossaryStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        chroma_db: ChromaDatabase,
        embedder_type: type[Embedder],
        n_results: int = 20,
    ):
        self._collection = chroma_db.get_or_create_collection(
            name="glossary",
            schema=_TermDocument,
            embedder_type=embedder_type,
        )
        self._n_results = n_results
        self._embedder = embedder_type()

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

    async def create_term(
        self,
        term_set: str,
        name: str,
        description: str,
        creation_utc: Optional[datetime] = None,
        synonyms: Optional[Sequence[str]] = None,
    ) -> Term:
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

    async def update_term(
        self,
        term_set: str,
        term_id: TermId,
        params: TermUpdateParams,
    ) -> Term:
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
            filters={"$and": [{"term_set": {"$eq": term_set}}, {"name": {"$eq": name}}]},
            params={
                "content": content,
                "name": name,
                "description": description,
                "synonyms": ", ".join(synonyms) if synonyms else "",
            },
        )

        assert update_result.updated_document

        return self._deserialize(term_document=update_result.updated_document)

    async def read_term(
        self,
        term_set: str,
        term_id: TermId,
    ) -> Term:
        term_document = await self._collection.find_one(
            filters={"$and": [{"term_set": {"$eq": term_set}}, {"id": {"$eq": term_id}}]}
        )
        if not term_document:
            raise ItemNotFoundError(item_id=UniqueId(term_id), message=f"term_set={term_set}")

        return self._deserialize(term_document=term_document)

    async def list_terms(
        self,
        term_set: str,
    ) -> Sequence[Term]:
        return [
            self._deserialize(term_document=d)
            for d in await self._collection.find(filters={"term_set": {"$eq": term_set}})
        ]

    async def delete_term(
        self,
        term_set: str,
        term_id: TermId,
    ) -> TermId:
        term_document = await self._collection.find_one(
            filters={"$and": [{"term_set": {"$eq": term_set}}, {"id": {"$eq": term_id}}]}
        )

        if not term_document:
            raise ItemNotFoundError(item_id=UniqueId(term_id))

        await self._collection.delete_one(
            filters={"$and": [{"term_set": {"$eq": term_set}}, {"id": {"$eq": term_id}}]}
        )

        return TermId(term_document["id"])

    async def _chunk_query(self, query: str) -> list[str]:
        max_length = self._embedder.max_tokens // 10

        list_of_lines = []

        while len(query) > max_length:
            line_length = query[:max_length].rfind(" ")
            list_of_lines.append(query[: line_length if line_length != -1 else max_length])
            query = query[line_length + 1 :]

        list_of_lines.append(query)
        return list_of_lines

    async def find_relevant_terms(
        self,
        term_set: str,
        query: str,
    ) -> Sequence[Term]:
        queries = await self._chunk_query(query)

        tasks = [
            self._collection.find_similar_documents(
                filters={"term_set": {"$eq": term_set}},
                query=q,
                k=self._n_results,
            )
            for q in queries
        ]

        all_results = sorted(
            list(set(chain.from_iterable(await asyncio.gather(*tasks)))),
            key=lambda doc_result: doc_result.distance,
        )

        return [self._deserialize(r.document) for r in all_results[: self._n_results]]

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
