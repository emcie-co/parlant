from __future__ import annotations
import asyncio
import importlib
import json
import operator
from loguru import logger
import os
from pathlib import Path
from typing import Any, Mapping, Sequence, Type, cast
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction  # type: ignore

from emcie.server.base_models import DefaultBaseModel
from emcie.server.core.persistence.common import ObjectId, Where
from emcie.server.core.persistence.document_database import DocumentCollection, DocumentDatabase


class ChromaDatabase(DocumentDatabase):

    def __init__(self, dir_path: Path) -> None:
        self._chroma_client = chromadb.PersistentClient(str(dir_path))
        self._embedding_function = OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-large",
        )
        self._collections: dict[str, ChromaCollection] = self._load_chromadb_collections()

    def _load_chromadb_collections(self) -> dict[str, ChromaCollection]:
        collections: dict[str, ChromaCollection] = {}
        for chromadb_collection in self._chroma_client.list_collections():
            collections[chromadb_collection.name] = ChromaCollection(
                chromadb_collection=chromadb_collection,
                name=chromadb_collection.name,
                schema=operator.attrgetter(chromadb_collection.metadata["model_path"])(
                    importlib.import_module(chromadb_collection.metadata["module_path"])
                ),
            )
        return collections

    def create_collection(
        self,
        name: str,
        schema: Type[DefaultBaseModel],
    ) -> ChromaCollection:
        if name in self._collections:
            raise ValueError(f'Collection "{name}" already exists.')
        logger.debug(f'Creating chromadb collection "{name}"')
        new_collection = ChromaCollection(
            chromadb_collection=self._chroma_client.create_collection(
                name=name,
                embedding_function=self._embedding_function,
                metadata={
                    "module_path": schema.__module__,
                    "model_path": schema.__qualname__,
                },
            ),
            name=name,
            schema=schema,
        )
        self._collections[name] = new_collection
        return new_collection

    def get_collection(
        self,
        name: str,
    ) -> ChromaCollection:
        if collection := self._collections.get(name):
            return collection
        raise ValueError(f'ChromaDB collection "{name}" not found.')

    def get_or_create_collection(
        self,
        name: str,
        schema: Type[DefaultBaseModel],
    ) -> ChromaCollection:
        if collection := self._collections.get(name):
            return collection
        logger.debug(f'Creating chromadb collection "{name}"')
        new_collection = ChromaCollection(
            chromadb_collection=self._chroma_client.create_collection(
                name=name,
                embedding_function=self._embedding_function,
                metadata={
                    "module_path": schema.__module__,
                    "model_path": schema.__qualname__,
                },
            ),
            name=name,
            schema=schema,
        )
        self._collections[name] = new_collection
        return new_collection

    def delete_collection(
        self,
        name: str,
    ) -> None:
        if name not in self._collections:
            raise ValueError(f'Collection "{name}" not found.')
        self._chroma_client.delete_collection(name=name)
        del self._collections[name]


class ChromaCollection(DocumentCollection):

    def __init__(
        self,
        chromadb_collection: chromadb.Collection,
        name: str,
        schema: Type[DefaultBaseModel],
    ) -> None:
        self._name = name
        self._schema = schema
        self._lock = asyncio.Lock()
        self._chroma_collection = chromadb_collection

    async def find(
        self,
        filters: Where,
    ) -> Sequence[Mapping[str, Any]]:
        if metadatas := self._chroma_collection.get(where=cast(chromadb.Where, filters))[
            "metadatas"
        ]:
            return [{k: v for k, v in m.items()} for m in metadatas]
        return []

    async def find_one(
        self,
        filters: Where,
    ) -> Mapping[str, Any]:
        if metadatas := self._chroma_collection.get(where=cast(chromadb.Where, filters))[
            "metadatas"
        ]:
            return {k: v for k, v in metadatas[0].items()}

        raise ValueError("No document found matching the provided filters.")

    async def insert_one(
        self,
        document: Mapping[str, Any],
    ) -> ObjectId:
        async with self._lock:
            self._chroma_collection.add(
                ids=[document["id"]],
                documents=[document["content"]],
                metadatas=[self._schema(**document).model_dump(mode="json")],
            )

        document_id: ObjectId = document["id"]
        return document_id

    async def update_one(
        self,
        filters: Where,
        updated_document: Mapping[str, Any],
        upsert: bool = False,
    ) -> ObjectId:
        document_id: ObjectId

        async with self._lock:
            if docs := self._chroma_collection.get(where=cast(chromadb.Where, filters))[
                "metadatas"
            ]:
                document_id = ObjectId(str(docs[0]["id"]))

                self._chroma_collection.update(
                    ids=[document_id],
                    documents=[updated_document["content"]],
                    metadatas=[
                        {**docs[0], **self._schema(**updated_document).model_dump(mode="json")}
                    ],
                )
                return document_id

            elif upsert:
                document_id = updated_document["id"]

                self._chroma_collection.add(
                    ids=[updated_document["id"]],
                    documents=[updated_document["content"]],
                    metadatas=[self._schema(**updated_document).model_dump(mode="json")],
                )
                return document_id

            raise ValueError("No document found matching the provided filters.")

    async def delete_one(
        self,
        filters: Where,
    ) -> None:
        async with self._lock:
            self._chroma_collection.delete(where=cast(chromadb.Where, filters))

    async def find_similar_documents(
        self,
        filters: Where,
        query: str,
        k: int,
    ) -> Sequence[Mapping[str, Any]]:
        docs = self._chroma_collection.query(
            where=cast(chromadb.Where, filters),
            query_texts=[query],
            n_results=k,
        )

        if metadatas := docs["metadatas"]:
            logger.debug(f"Similar documents found: {json.dumps(metadatas[0], indent=2)}")

            return [{k: v for k, v in m.items()} for m in metadatas[0]]

        raise ValueError("No similar documents found.")
