from __future__ import annotations
import asyncio
import importlib
import json
import operator
from pathlib import Path
from typing import Generic, Optional, Sequence, TypeVar, cast

import chromadb

from emcie.server.core.generation.embedders import Embedder, EmbedderFactory
from emcie.server.core.persistence.common import (
    BaseDocument,
    Where,
)
from emcie.server.core.persistence.document_database import DeleteResult, InsertResult, UpdateResult
from emcie.server.logger import Logger


class ChromaDocument(BaseDocument):
    content: str


TChromaDocument = TypeVar("TChromaDocument", bound=ChromaDocument)


class ChromaDatabase:
    def __init__(
        self,
        logger: Logger,
        dir_path: Path,
        embedder_factory: EmbedderFactory,
    ) -> None:
        self.logger = logger
        self._embedder_factory = embedder_factory

        self._chroma_client = chromadb.PersistentClient(str(dir_path))
        self._collections: dict[str, ChromaCollection[ChromaDocument]] = (
            self._load_chromadb_collections()
        )

    def _load_chromadb_collections(self) -> dict[str, ChromaCollection[ChromaDocument]]:
        collections: dict[str, ChromaCollection[ChromaDocument]] = {}
        for chromadb_collection in self._chroma_client.list_collections():
            embedder_module = importlib.import_module(
                chromadb_collection.metadata["embedder_module_path"]
            )
            embedder_type = getattr(
                embedder_module,
                chromadb_collection.metadata["embedder_type_path"],
            )
            embedder = self._embedder_factory.create_embedder(embedder_type)

            chroma_collection = self._chroma_client.get_collection(
                name=chromadb_collection.name,
                embedding_function=None,
            )

            collections[chromadb_collection.name] = ChromaCollection(
                logger=self.logger,
                chromadb_collection=chroma_collection,
                name=chromadb_collection.name,
                schema=operator.attrgetter(chromadb_collection.metadata["schema_model_path"])(
                    importlib.import_module(chromadb_collection.metadata["schema_module_path"])
                ),
                embedder=embedder,
            )
        return collections

    def create_collection(
        self,
        name: str,
        schema: type[TChromaDocument],
        embedder_type: type[Embedder],
    ) -> ChromaCollection[TChromaDocument]:
        if name in self._collections:
            raise ValueError(f'Collection "{name}" already exists.')

        assert issubclass(schema, ChromaDocument)

        self._collections[name] = ChromaCollection(
            self.logger,
            chromadb_collection=self._chroma_client.create_collection(
                name=name,
                metadata={
                    "schema_module_path": schema.__module__,
                    "schema_model_path": schema.__qualname__,
                    "embedder_module_path": embedder_type.__module__,
                    "embedder_type_path": embedder_type.__qualname__,
                },
                embedding_function=None,
            ),
            name=name,
            schema=schema,
            embedder=self._embedder_factory.create_embedder(embedder_type),
        )

        return cast(ChromaCollection[TChromaDocument], self._collections[name])

    def get_collection(
        self,
        name: str,
    ) -> ChromaCollection[TChromaDocument]:
        if collection := self._collections.get(name):
            return cast(ChromaCollection[TChromaDocument], collection)

        raise ValueError(f'ChromaDB collection "{name}" not found.')

    def get_or_create_collection(
        self,
        name: str,
        schema: type[TChromaDocument],
        embedder_type: type[Embedder],
    ) -> ChromaCollection[TChromaDocument]:
        if collection := self._collections.get(name):
            assert schema == collection._schema
            return cast(ChromaCollection[TChromaDocument], collection)

        assert issubclass(schema, ChromaDocument)

        self._collections[name] = ChromaCollection(
            self.logger,
            chromadb_collection=self._chroma_client.create_collection(
                name=name,
                metadata={
                    "schema_module_path": schema.__module__,
                    "schema_model_path": schema.__qualname__,
                    "embedder_module_path": embedder_type.__module__,
                    "embedder_type_path": embedder_type.__qualname__,
                },
                embedding_function=None,
            ),
            name=name,
            schema=schema,
            embedder=self._embedder_factory.create_embedder(embedder_type),
        )

        return cast(ChromaCollection[TChromaDocument], self._collections[name])

    def delete_collection(
        self,
        name: str,
    ) -> None:
        if name not in self._collections:
            raise ValueError(f'Collection "{name}" not found.')
        self._chroma_client.delete_collection(name=name)
        del self._collections[name]


class ChromaCollection(Generic[TChromaDocument]):
    def __init__(
        self,
        logger: Logger,
        chromadb_collection: chromadb.Collection,
        name: str,
        schema: type[TChromaDocument],
        embedder: Embedder,
    ) -> None:
        self.logger = logger
        self._name = name
        self._schema = schema
        self._embedder = embedder

        self._lock = asyncio.Lock()
        self._chroma_collection = chromadb_collection

    async def find(
        self,
        filters: Where,
    ) -> Sequence[TChromaDocument]:
        if metadatas := self._chroma_collection.get(where=cast(chromadb.Where, filters))[
            "metadatas"
        ]:
            return [self._schema.model_validate(m) for m in metadatas]

        return []

    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TChromaDocument]:
        if metadatas := self._chroma_collection.get(where=cast(chromadb.Where, filters))[
            "metadatas"
        ]:
            return self._schema.model_validate({k: v for k, v in metadatas[0].items()})

        return None

    async def insert_one(
        self,
        document: TChromaDocument,
    ) -> InsertResult:
        embeddings = list((await self._embedder.embed([document.content])).vectors)

        async with self._lock:
            self._chroma_collection.add(
                ids=[document.id],
                documents=[document.content],
                metadatas=[document.model_dump(mode="json")],
                embeddings=embeddings,
            )

        return InsertResult(acknowledged=True)

    async def update_one(
        self,
        filters: Where,
        updated_document: TChromaDocument,
        upsert: bool = False,
    ) -> UpdateResult[TChromaDocument]:
        embeddings = list((await self._embedder.embed([updated_document.content])).vectors)

        async with self._lock:
            if self._chroma_collection.get(where=cast(chromadb.Where, filters))["metadatas"]:
                self._chroma_collection.update(
                    ids=[updated_document.id],
                    documents=[updated_document.content],
                    metadatas=[updated_document.model_dump(mode="json")],
                    embeddings=embeddings,
                )

                return UpdateResult(
                    acknowledged=True,
                    matched_count=1,
                    modified_count=1,
                    updated_document=updated_document,
                )

            elif upsert:
                self._chroma_collection.add(
                    ids=[updated_document.id],
                    documents=[updated_document.content],
                    metadatas=[updated_document.model_dump(mode="json")],
                    embeddings=embeddings,
                )

                return UpdateResult(
                    acknowledged=True,
                    matched_count=0,
                    modified_count=0,
                    updated_document=updated_document,
                )

            return UpdateResult(
                acknowledged=True,
                matched_count=0,
                modified_count=0,
                updated_document=None,
            )

    async def delete_one(
        self,
        filters: Where,
    ) -> DeleteResult[TChromaDocument]:
        async with self._lock:
            if docs := self._chroma_collection.get(where=cast(chromadb.Where, filters))[
                "metadatas"
            ]:
                if len(docs) > 1:
                    raise ValueError(
                        f"ChromaCollection delete_one: detected more than one document with filters '{filters}'. Aborting..."
                    )
                deleted_document = docs[0]

                self._chroma_collection.delete(where=cast(chromadb.Where, filters))

                return DeleteResult(
                    deleted_count=1,
                    acknowledged=True,
                    deleted_document=self._schema.model_validate(deleted_document),
                )

            return DeleteResult(
                acknowledged=True,
                deleted_count=0,
                deleted_document=None,
            )

    async def find_similar_documents(
        self,
        filters: Where,
        query: str,
        k: int,
    ) -> Sequence[TChromaDocument]:
        query_embeddings = list((await self._embedder.embed([query])).vectors)

        docs = self._chroma_collection.query(
            where=cast(chromadb.Where, filters),
            query_embeddings=query_embeddings,
            n_results=k,
        )

        if metadatas := docs["metadatas"]:
            self.logger.debug(f"Similar documents found: {json.dumps(metadatas[0], indent=2)}")

            return [self._schema.model_validate(m) for m in metadatas[0]]

        return []
