from __future__ import annotations
import asyncio
import importlib
import json
import operator
from pathlib import Path
import threading
from typing import Any, Coroutine, Generic, Optional, Sequence, Type, TypeVar, cast

import chromadb
from chromadb.api.types import Embeddable

from emcie.server.core.generation.embedders import Embedder
from emcie.server.core.persistence.common import (
    BaseDocument,
    Where,
)
from emcie.server.core.persistence.document_database import DeleteResult, InsertResult, UpdateResult
from emcie.server.logger import Logger


class ChromaDocument(BaseDocument):
    content: str


TChromaDocument = TypeVar("TChromaDocument", bound=ChromaDocument)


class EmbeddingFunctionAdapter(chromadb.EmbeddingFunction[Embeddable]):
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def __call__(self, input: Embeddable) -> chromadb.Embeddings:
        async def call_embedder() -> Sequence[Sequence[float]]:
            return (await self.embedder.embed([str(i) for i in input])).vectors

        def run_coroutine_in_thread(
            coro: Coroutine[Any, Any, Sequence[Sequence[float]]],
        ) -> Sequence[Sequence[float]]:
            result_container = []
            exception_container = []

            def target() -> None:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(coro)
                    loop.close()
                    result_container.append(result)
                except Exception as e:
                    exception_container.append(e)

            thread = threading.Thread(target=target)
            thread.start()
            thread.join()

            if exception_container:
                raise exception_container[0]
            return result_container[0]

        vectors = run_coroutine_in_thread(call_embedder())

        return [v for v in vectors]


class ChromaDatabase:
    def __init__(self, logger: Logger, dir_path: Path) -> None:
        self.logger = logger

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
            embedding_function = EmbeddingFunctionAdapter(embedder_type(logger=self.logger))

            chroma_collection = self._chroma_client.get_collection(
                name=chromadb_collection.name,
                embedding_function=embedding_function,
            )

            collections[chromadb_collection.name] = ChromaCollection(
                logger=self.logger,
                chromadb_collection=chroma_collection,
                name=chromadb_collection.name,
                schema=operator.attrgetter(chromadb_collection.metadata["schema_model_path"])(
                    importlib.import_module(chromadb_collection.metadata["schema_module_path"])
                ),
            )
        return collections

    def create_collection(
        self,
        name: str,
        schema: Type[TChromaDocument],
        embedder_type: Type[Embedder],
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
                embedding_function=EmbeddingFunctionAdapter(embedder_type(logger=self.logger)),
            ),
            name=name,
            schema=schema,
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
        schema: Type[TChromaDocument],
        embedder_type: Type[Embedder],
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
                embedding_function=EmbeddingFunctionAdapter(embedder_type(logger=self.logger)),
            ),
            name=name,
            schema=schema,
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
        schema: Type[TChromaDocument],
    ) -> None:
        self.logger = logger
        self._name = name
        self._schema = schema

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
        async with self._lock:
            self._chroma_collection.add(
                ids=[document.id],
                documents=[document.content],
                metadatas=[document.model_dump(mode="json")],
            )

        return InsertResult(acknowledged=True)

    async def update_one(
        self,
        filters: Where,
        updated_document: TChromaDocument,
        upsert: bool = False,
    ) -> UpdateResult[TChromaDocument]:
        async with self._lock:
            if self._chroma_collection.get(where=cast(chromadb.Where, filters))["metadatas"]:
                self._chroma_collection.update(
                    ids=[updated_document.id],
                    documents=[updated_document.content],
                    metadatas=[updated_document.model_dump(mode="json")],
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
        docs = self._chroma_collection.query(
            where=cast(chromadb.Where, filters),
            query_texts=[query],
            n_results=k,
        )

        if metadatas := docs["metadatas"]:
            self.logger.debug(f"Similar documents found: {json.dumps(metadatas[0], indent=2)}")

            return [self._schema.model_validate(m) for m in metadatas[0]]

        return []
