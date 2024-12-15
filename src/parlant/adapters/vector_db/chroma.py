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

from __future__ import annotations
import importlib
import json
import operator
from pathlib import Path
from typing import Generic, Optional, Sequence, cast
from typing_extensions import override, Self
import chromadb

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.logging import Logger
from parlant.core.nlp.embedding import Embedder, EmbedderFactory
from parlant.core.persistence.common import Where, ensure_is_total
from parlant.core.persistence.vector_database import (
    BaseDocument,
    DeleteResult,
    InsertResult,
    SimilarDocumentResult,
    UpdateResult,
    VectorCollection,
    VectorDatabase,
    TDocument,
)


class ChromaDatabase(VectorDatabase):
    def __init__(
        self,
        logger: Logger,
        dir_path: Path,
        embedder_factory: EmbedderFactory,
    ) -> None:
        self._dir_path = dir_path
        self._logger = logger
        self._embedder_factory = embedder_factory

        self._chroma_client: chromadb.api.ClientAPI
        self._collections: dict[str, ChromaCollection[BaseDocument]] = {}

    async def __aenter__(self) -> Self:
        self._chroma_client = chromadb.PersistentClient(str(self._dir_path))
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

            self._collections[chromadb_collection.name] = ChromaCollection(
                logger=self._logger,
                chromadb_collection=chroma_collection,
                name=chromadb_collection.name,
                schema=operator.attrgetter(chromadb_collection.metadata["schema_model_path"])(
                    importlib.import_module(chromadb_collection.metadata["schema_module_path"])
                ),
                embedder=embedder,
            )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
    ) -> ChromaCollection[TDocument]:
        if name in self._collections:
            raise ValueError(f'Collection "{name}" already exists.')

        self._collections[name] = ChromaCollection(
            self._logger,
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

        return cast(ChromaCollection[TDocument], self._collections[name])

    @override
    async def get_collection(
        self,
        name: str,
    ) -> ChromaCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(ChromaCollection[TDocument], collection)

        raise ValueError(f'ChromaDB collection "{name}" not found.')

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
    ) -> ChromaCollection[TDocument]:
        if collection := self._collections.get(name):
            assert schema == collection._schema
            return cast(ChromaCollection[TDocument], collection)

        self._collections[name] = ChromaCollection(
            self._logger,
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

        return cast(ChromaCollection[TDocument], self._collections[name])

    @override
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        if name not in self._collections:
            raise ValueError(f'Collection "{name}" not found.')
        self._chroma_client.delete_collection(name=name)
        del self._collections[name]


class ChromaCollection(Generic[TDocument], VectorCollection[TDocument]):
    def __init__(
        self,
        logger: Logger,
        chromadb_collection: chromadb.Collection,
        name: str,
        schema: type[TDocument],
        embedder: Embedder,
    ) -> None:
        self._logger = logger
        self._name = name
        self._schema = schema
        self._embedder = embedder

        self._lock = ReaderWriterLock()
        self._chroma_collection = chromadb_collection

    @override
    async def find(
        self,
        filters: Where,
    ) -> Sequence[TDocument]:
        async with self._lock.reader_lock:
            if metadatas := self._chroma_collection.get(
                where=cast(chromadb.Where, filters) or None
            )["metadatas"]:
                return [cast(TDocument, m) for m in metadatas]

        return []

    @override
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]:
        async with self._lock.reader_lock:
            if metadatas := self._chroma_collection.get(
                where=cast(chromadb.Where, filters) or None
            )["metadatas"]:
                return cast(TDocument, {k: v for k, v in metadatas[0].items()})

        return None

    @override
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult:
        ensure_is_total(document, self._schema)

        embeddings = list((await self._embedder.embed([document["content"]])).vectors)

        async with self._lock.writer_lock:
            self._chroma_collection.add(
                ids=[document["id"]],
                documents=[document["content"]],
                metadatas=[cast(chromadb.Metadata, document)],
                embeddings=embeddings,
            )

        return InsertResult(acknowledged=True)

    @override
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        async with self._lock.writer_lock:
            if docs := self._chroma_collection.get(where=cast(chromadb.Where, filters) or None)[
                "metadatas"
            ]:
                doc = docs[0]

                if "content" in params:
                    embeddings = list((await self._embedder.embed([params["content"]])).vectors)
                    document = params["content"]
                else:
                    embeddings = list((await self._embedder.embed([str(doc["content"])])).vectors)
                    document = str(doc["content"])

                updated_document = {**doc, **params}

                self._chroma_collection.update(
                    ids=[str(doc["id"])],
                    documents=[document],
                    metadatas=[cast(chromadb.Metadata, updated_document)],
                    embeddings=embeddings,  # type: ignore
                )

                return UpdateResult(
                    acknowledged=True,
                    matched_count=1,
                    modified_count=1,
                    updated_document=cast(TDocument, updated_document),
                )

            elif upsert:
                ensure_is_total(params, self._schema)

                embeddings = list((await self._embedder.embed([params["content"]])).vectors)

                self._chroma_collection.add(
                    ids=[params["id"]],
                    documents=[params["content"]],
                    metadatas=[cast(chromadb.Metadata, params)],
                    embeddings=embeddings,
                )

                return UpdateResult(
                    acknowledged=True,
                    matched_count=0,
                    modified_count=0,
                    updated_document=params,
                )

            return UpdateResult(
                acknowledged=True,
                matched_count=0,
                modified_count=0,
                updated_document=None,
            )

    @override
    async def delete_one(
        self,
        filters: Where,
    ) -> DeleteResult[TDocument]:
        async with self._lock.writer_lock:
            if docs := self._chroma_collection.get(where=cast(chromadb.Where, filters) or None)[
                "metadatas"
            ]:
                if len(docs) > 1:
                    raise ValueError(
                        f"ChromaCollection delete_one: detected more than one document with filters '{filters}'. Aborting..."
                    )
                deleted_document = docs[0]

                self._chroma_collection.delete(where=cast(chromadb.Where, filters) or None)

                return DeleteResult(
                    deleted_count=1,
                    acknowledged=True,
                    deleted_document=cast(TDocument, deleted_document),
                )

            return DeleteResult(
                acknowledged=True,
                deleted_count=0,
                deleted_document=None,
            )

    @override
    async def find_similar_documents(
        self,
        filters: Where,
        query: str,
        k: int,
    ) -> Sequence[SimilarDocumentResult[TDocument]]:
        async with self._lock.reader_lock:
            query_embeddings = list((await self._embedder.embed([query])).vectors)

            docs = self._chroma_collection.query(
                where=cast(chromadb.Where, filters) or None,
                query_embeddings=query_embeddings,
                n_results=k,
            )

            if not docs["metadatas"]:
                return []

            self._logger.debug(
                f"Similar documents found: {json.dumps(docs['metadatas'][0], indent=2)}"
            )

            assert docs["distances"]
            return [
                SimilarDocumentResult(document=cast(TDocument, m), distance=d)
                for m, d in zip(docs["metadatas"][0], docs["distances"][0])
            ]
