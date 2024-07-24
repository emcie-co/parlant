from dataclasses import dataclass
from datetime import datetime, timezone
from typing import NewType, Optional, Sequence

from emcie.server.base_models import DefaultBaseModel
from emcie.server.core import common
from emcie.server.core.persistence.document_database import DocumentDatabase, VectorCollection

TermId = NewType("TermId", str)


@dataclass(frozen=True)
class Term:
    id: TermId
    creation_utc: datetime
    name: str
    description: str
    synonyms: Optional[Sequence[str]]

    def __repr__(self) -> str:
        term_string = f"{self.name}: {self.description}"
        if self.synonyms:
            term_string += f"\n\tSysnonyms: {','.join(self.synonyms)}"
        return term_string


class TerminologyStore:
    class TermDocument(DefaultBaseModel):
        id: TermId
        term_set: str
        creation_utc: datetime
        name: str
        content: str
        description: str
        synonyms: Optional[str]

    def __init__(
        self,
        vector_db: DocumentDatabase,
    ):
        self._collection: VectorCollection = vector_db.get_or_create_collection(
            name="terminology",
            schema=self.TermDocument,
        )
        self._n_results = 20

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

        document = {
            "id": common.generate_id(),
            "term_set": term_set,
            "content": content,
            "name": name,
            "description": description,
            "creation_utc": creation_utc,
            "synonyms": ", ".join(synonyms) if synonyms else "",
        }

        term_id = await self._collection.insert_one(document=document)

        return Term(
            id=term_id,
            creation_utc=creation_utc,
            name=name,
            description=description,
            synonyms=synonyms,
        )

    async def update_term(
        self,
        term_set: str,
        name: str,
        description: str,
        synonyms: Sequence[str],
    ) -> Term:
        filters = {"term_set": term_set, "term_name": name}

        content = self._assemble_term_content(
            name=name,
            description=description,
            synonyms=synonyms,
        )

        updated_document = {
            "id": common.generate_id(),
            "term_set": term_set,
            "content": content,
            "name": name,
            "description": description,
            "synonyms": ", ".join(synonyms) if synonyms else "",
        }

        term_id = await self._collection.update_one(
            filters=filters,
            updated_document=updated_document,
        )

        term_doc = await self._collection.find_one({"id": term_id})

        return Term(
            id=term_id,
            creation_utc=term_doc["creation_utc"],
            name=name,
            description=description,
            synonyms=synonyms,
        )

    async def read_term(
        self,
        term_set: str,
        name: str,
    ) -> Term:
        filters = {"term_set": term_set, "term_name": name}

        term_document = self._collection.find_one(filters=filters)

        return Term(
            id=term_document["term_id"],
            creation_utc=term_document["creation_utc"],
            name=term_document["name"],
            description=term_document["description"],
            synonyms=term_document["synonyms"],
        )

    async def list_terms(
        self,
        term_set: str,
    ) -> Sequence[Term]:
        return [
            Term(
                id=d["id"],
                creation_utc=d["creation_utc"],
                name=d["name"],
                description=d["description"],
                synonyms=d["synonyms"],
            )
            for d in await self._collection.find(filters={"term_set": term_set})
        ]

    async def find_relevant_terms(
        self,
        term_set: str,
        query: str,
    ) -> Sequence[Term]:
        return [
            Term(
                id=d["id"],
                creation_utc=d["creation_utc"],
                name=d["name"],
                description=d["description"],
                synonyms=d["synonyms"],
            )
            for d in await self._collection.find_similar_documents(
                filters={"term_set": term_set},
                query=query,
                k=self._n_results,
            )
        ]

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
