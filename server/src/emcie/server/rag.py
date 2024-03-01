from typing import List, Dict, Iterable, NewType, Optional, Tuple, Union
from tinydb.storages import MemoryStorage
import chromadb
from pydantic import BaseModel
from tinydb import TinyDB, table, where


class RagDocument(BaseModel):
    id: str
    metadata: Optional[dict] = None
    document: str


class RagStore:
    def __init__(self, db: Optional[TinyDB] = None) -> None:
        self.rag = chromadb.Client()
        self.collection = self.rag.create_collection("documents")
        self.db = db or TinyDB(storage=MemoryStorage)

    def upsert(self, document: dict, retrain_in_case_of_update: bool = True) -> Dict:
        self.collection.upsert(
            documents=[document["document"]],
            metadatas=[document["metadata"]],
            ids=[document["id"]],
        )
        return document

    def query(self, query: str, k: int = 3) -> Iterable[Dict]:
        query_result = self.collection.query(query_texts=[query], n_results=k)
        result = []
        for i in range(len(query_result["ids"][0])):
            result.append(
                {
                    "id": query_result["ids"][0][i],
                    "metadata": query_result["metadatas"][0][i],
                    "document": query_result["documents"][0][i],
                }
            )
        return result

    def get_all_documents(self) -> Iterable[Dict] | None:
        documents = self._get_documents(self.collection.get())
        result = []
        for i in range(len(documents["ids"])):
            result.append(
                {
                    "id": documents["ids"][i],
                    "metadata": documents["metadatas"][i],
                    "document": documents["documents"][i],
                }
            )
        return result
