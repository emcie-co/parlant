import heapq
from typing import List, Dict, Iterable, NewType, Optional, Tuple, Union
from tinydb.storages import MemoryStorage
from pydantic import BaseModel
from tinydb import TinyDB, table

from emcie.server.embedders import EmbedderModel, OpenAIEmbedder


class RagDocument(BaseModel):
    id: str
    metadata: Optional[dict] = None
    document: str
    vector: List[float]


class RagStore:
    def __init__(
        self,
        embedder: Optional[EmbedderModel] = None,
        db: Optional[TinyDB] = None,
    ) -> None:
        self.db = db or TinyDB(storage=MemoryStorage)
        self.embedder = embedder or OpenAIEmbedder()

    async def upsert(self, document: dict) -> Dict:
        document_vector = await self.embedder.get_embedding(document["document"])
        full_document = {
            **{"document_vector": document_vector},
            **document,
        }
        self.db.upsert(
            table.Document(
                full_document,
                doc_id=document["id"],
            )
        )
        return document

    async def query(self, query: str, k: int = 3) -> Iterable[Dict]:
        query_embedding = await self.embedder.get_embedding(query)
        docs_with_distance = [
            {
                **doc,
                **{
                    "distance": self.embedder.get_similarity(
                        query_embedding, doc["document_vector"]
                    )
                },
            }
            for doc in self.get_all_documents()
        ]
        top_k = heapq.nlargest(
            k,
            docs_with_distance,
            key=lambda doc: 1 - doc["distance"],
        )

        return sorted(top_k, key=lambda doc: doc["distance"])

    def get_all_documents(self) -> Iterable[Dict] | None:
        return self.db.all()
