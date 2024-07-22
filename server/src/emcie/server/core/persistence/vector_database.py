from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
import json
from loguru import logger
import os
from pathlib import Path
from typing import Any, Sequence
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from emcie.server.core.persistence.common import ObjectId


class VectorDatabase(ABC):

    @abstractmethod
    async def create_collection(
        self,
        name: str,
    ) -> VectorCollection:
        """
        Creates a new collection with the given name and schema and returns the collection.
        """
        ...

    @abstractmethod
    async def get_collection(
        self,
        name: str,
    ) -> VectorCollection:
        """
        Retrieves an existing collection by its name.
        """
        ...

    @abstractmethod
    async def get_or_create_collection(self, name: str,) -> VectorCollection:
        """
        Get or create a new collection
        """
        ...

    @abstractmethod
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        """
        Deletes a collection by its name.
        """
        ...


class VectorCollection(ABC):

    @abstractmethod
    async def add_vector(
        self,
        vector: Sequence[float],
        metadata: dict[str, Any],
    ) -> str:
        """
        Adds a new vector to the collection and returns its identifier.
        """
        ...

    @abstractmethod
    async def update_vector(
        self,
        vector_id: str,
        vector: Sequence[float],
        metadata: dict[str, Any],
    ) -> None:
        """
        Updates an existing vector in the collection.
        """
        ...

    @abstractmethod
    async def get_vector(
        self,
        vector_id: str,
    ) -> dict[str, Any]:
        """
        Retrieves a vector and its metadata by its identifier.
        """
        ...

    @abstractmethod
    async def list_vectors(self) -> Sequence[dict[str, Any]]:
        """
        Lists all vectors and their metadata in the collection.
        """
        ...

    @abstractmethod
    async def find_similar_vectors(
        self,
        vector: Sequence[float],
        k: int,
    ) -> Sequence[dict[str, Any]]:
        """
        Finds the k most similar vectors to the given vector in the collection.
        """
        ...

    @abstractmethod
    async def delete_vector(
        self,
        vector_id: str,
    ) -> None:
        """
        Deletes a vector by its identifier from the collection.
        """
        ...

    @abstractmethod
    async def flush(self) -> None:
        """
        Flushes any buffered operations to the database.
        """
        ...


class ChromaDatabase(VectorDatabase):

    def __init__(self, dir_path: Path) -> None:
        self._chroma_client = chromadb.PersistentClient(str(dir_path))
        self._embedding_function = OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-large",
        )
        self.collections: dict[str, chromadb.Collection] = {
            collection.name: collection
            for collection in self._chroma_client.list_collections()
        }
    
    async def create_collection(
        self,
        name: str,
    ) -> ChromaCollection:
        if name in self.collections:
            raise ValueError(f'Collection "{name}" already exists.')
        logger.debug(f'Creating chromadb collection "{name}"')
        chroma_collection = self._chroma_client.create_collection(
            name=name,
            embedding_function=self._embedding_function,
        )
        new_collection = ChromaCollection(chroma_collection)
        self.collections[name] = new_collection
        return new_collection

    async def get_collection(
        self,
        name: str,
    ) -> ChromaCollection:
        if name not in self.collections:
            raise ValueError(f'ChromaDB collection "{name}" not found.')
        return self.collections[name]
    
    async def get_or_create_collection(
        self,
        name: str,
    ) -> VectorCollection:
        try:
            return self.get_collection(name=name)
        except ValueError:
            return self.create_collection(name=name)

    async def delete_collection(
        self,
        name: str,
    ) -> None:
        if name not in self.collections:
            raise ValueError(f'Collection "{name}" not found.')
        del self.collections[name]


class ChromaCollection(VectorCollection):

    def __init__(
        self,
        collection: chromadb.Collection,
    ):
        self._collection = collection
        self._lock = asyncio.Lock()

    async def add_vector(
        self,
        document_id: ObjectId,
        document: str,
        metadata: dict[str, Any],
    ) -> ObjectId:
        with self._lock:
            self._collection.add(
                ids=[document_id],
                documents=[document],
                metadatas=[
                    metadata
                ],
            )
        return document_id

    async def update_vector(
        self,
        document_id: ObjectId,
        document: str,
        metadata: dict[str, Any],
    ) -> ObjectId:
        with self._lock:
            self._collection.update(
                ids=[document_id], 
                documents=[document], 
                metadatas=[
                    metadata
                ],
            )
        return document_id

    async def get_vector(
        self,
        document_id: ObjectId,
    ) -> dict[str, Any]:
        with self._lock:
            return self._collection.get(ids=[document_id])

    async def list_documents(self) -> Sequence[dict[str, Any]]:
        return self._collection.get()

    async def find_similar_documents(
        self,
        query: str,
        k: int,
    ) -> Sequence[dict[str, Any]]:
        with self._lock:
            docs = self._collection.query(query_texts=query, n_results=k)

        if docs:
            logger.debug(f"Similar documents found: {json.dumps(docs["metadatas"], indent=2,)}")

        return docs["metadatas"]

    async def delete_vector(
        self,
        document_id: ObjectId,
    ) -> None:
        self._collection.delete(ids=[document_id])
