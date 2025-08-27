# Copyright 2025 Emcie Co Ltd.
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
import json
from pathlib import Path
from typing import Awaitable, Callable, Generic, Optional, Sequence, cast
from typing_extensions import override, Self
import chromadb
from chromadb.api.collection_configuration import (
    CreateCollectionConfiguration,
    CreateHNSWConfiguration,
)


from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import JSONSerializable
from parlant.core.loggers import Logger
from parlant.core.nlp.embedding import (
    Embedder,
    EmbedderFactory,
    EmbeddingCacheProvider,
    NoOpEmbedder,
)
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
    identity_loader,
)


class ChromaDatabase(VectorDatabase):
    def __init__(
        self,
        logger: Logger,
        dir_path: Path,
        embedder_factory: EmbedderFactory,
        embedding_cache_provider: EmbeddingCacheProvider,
    ) -> None:
        self._dir_path = dir_path
        self._logger = logger
        self._embedder_factory = embedder_factory

        self.chroma_client: chromadb.api.ClientAPI
        self._collections: dict[str, ChromaCollection[BaseDocument]] = {}

        self._embedding_cache_provider = embedding_cache_provider

    async def __aenter__(self) -> Self:
        self.chroma_client = chromadb.PersistentClient(str(self._dir_path))
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    def format_collection_name(
        self,
        name: str,
        embedder_type: type[Embedder],
    ) -> str:
        return f"{name}_{embedder_type.__name__}"

    # Loads documents from unembedded collection, migrates them if needed, and ensures embedded collection is in sync
    async def _load_collection_documents(
        self,
        embedded_collection: chromadb.Collection,
        unembedded_collection: chromadb.Collection,
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> chromadb.Collection:
        failed_migrations: list[BaseDocument] = []
        embedder = self._embedder_factory.create_embedder(embedder_type)

        unembedded_docs = unembedded_collection.get()["metadatas"]
        indexing_required = False

        if unembedded_docs:
            for doc in unembedded_docs:
                prospective_doc = cast(BaseDocument, doc)
                try:
                    if loaded_doc := await document_loader(prospective_doc):
                        if loaded_doc != prospective_doc:
                            unembedded_collection.update(
                                ids=[prospective_doc["id"]],
                                documents=[loaded_doc["content"]],
                                metadatas=[cast(chromadb.Metadata, loaded_doc)],
                                embeddings=[0],
                            )
                            indexing_required = True
                    else:
                        self._logger.warning(f'Failed to load document "{doc}"')
                        unembedded_collection.delete(where={"id": prospective_doc["id"]})
                        failed_migrations.append(prospective_doc)

                except Exception as e:
                    self._logger.error(f"Failed to load document '{doc}'. error: {e}.")
                    failed_migrations.append(prospective_doc)

            # Store failed migrations in a separate collection for debugging
            if failed_migrations:
                failed_migrations_collection = await self.get_or_create_collection(
                    "failed_migrations",
                    BaseDocument,
                    NoOpEmbedder,
                    identity_loader,
                )

                for failed_doc in failed_migrations:
                    failed_migrations_collection.embedded_collection.add(
                        ids=[failed_doc["id"]],
                        documents=[failed_doc["content"]],
                        metadatas=[cast(chromadb.Metadata, failed_doc)],
                        embeddings=[0],
                    )

        if (
            indexing_required
            or unembedded_collection.metadata["version"] != embedded_collection.metadata["version"]
        ):
            await self._index_collection(embedded_collection, unembedded_collection, embedder)

        return embedded_collection

    # Syncs embedded collection with unembedded collection
    async def _index_collection(
        self,
        collection: chromadb.Collection,
        unembedded_collection: chromadb.Collection,
        embedder: Embedder,
    ) -> None:
        if docs := unembedded_collection.get()["metadatas"]:
            unembedded_docs_by_id = {doc["id"]: doc for doc in docs}

        # Remove docs from embedded collection that no longer exist in unembedded
        # Update embeddings for changed docs
        if docs := collection.get()["metadatas"]:
            for doc in docs:
                if doc["id"] not in unembedded_docs_by_id:
                    collection.delete(where={"id": cast(str, doc["id"])})
                else:
                    if doc["checksum"] != unembedded_docs_by_id[doc["id"]]["checksum"]:
                        embeddings = list(
                            (
                                await embedder.embed(
                                    [cast(str, unembedded_docs_by_id[doc["id"]]["content"])]
                                )
                            ).vectors
                        )

                        collection.update(
                            ids=[str(doc["id"])],
                            documents=[cast(str, unembedded_docs_by_id[doc["id"]]["content"])],
                            metadatas=unembedded_docs_by_id[doc["id"]],
                            embeddings=embeddings,
                        )
                    unembedded_docs_by_id.pop(doc["id"])

        # Add new docs from unembedded to embedded collection
        for doc in unembedded_docs_by_id.values():
            collection.add(
                ids=[str(doc["id"])],
                documents=[cast(str, doc["content"])],
                metadatas=[doc],
                embeddings=list((await embedder.embed([cast(str, doc["content"])])).vectors),
            )

        collection.metadata.update({"version": unembedded_collection.metadata["version"]})

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
    ) -> ChromaCollection[TDocument]:
        if name in self._collections:
            raise ValueError(f'Collection "{name}" already exists.')

        embedded_collection = self.chroma_client.create_collection(
            name=self.format_collection_name(name, embedder_type),
            metadata={"version": 1},
            embedding_function=None,
            configuration=CreateCollectionConfiguration(
                hnsw=CreateHNSWConfiguration(space="cosine")
            ),
        )

        unembedded_collection = self.chroma_client.create_collection(
            name=f"{name}_unembedded",
            metadata={"version": 1},
            embedding_function=None,
        )

        self._collections[name] = ChromaCollection(
            self._logger,
            embedded_collection=embedded_collection,
            unembedded_collection=unembedded_collection,
            name=name,
            schema=schema,
            embedder=self._embedder_factory.create_embedder(embedder_type),
            embedding_cache_provider=self._embedding_cache_provider,
            version=1,
        )

        return cast(ChromaCollection[TDocument], self._collections[name])

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> ChromaCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(ChromaCollection[TDocument], collection)

        # Find unembedded collection first which acts as the SSOT.
        # Check if we have a corresponding embedded collection for the embedder type.
        # Whether we find an existing embedded collection or create a new one,
        # we reindex and sync it with the unembedded collection to ensure consistency
        elif unembedded_collection := next(
            (
                col
                for col in self.chroma_client.list_collections()
                if col.name == f"{name}_unembedded"
            ),
            None,
        ):
            embedded_collection = next(
                (
                    col
                    for col in self.chroma_client.list_collections()
                    if col.name == self.format_collection_name(name, embedder_type)
                ),
                None,
            ) or self.chroma_client.create_collection(
                name=self.format_collection_name(name, embedder_type),
                metadata={"version": 1},
                embedding_function=None,
                configuration=CreateCollectionConfiguration(
                    hnsw=CreateHNSWConfiguration(space="cosine")
                ),
            )

            await self._index_collection(
                collection=embedded_collection,
                unembedded_collection=unembedded_collection,
                embedder=self._embedder_factory.create_embedder(embedder_type),
            )

            self._collections[name] = ChromaCollection(
                self._logger,
                embedded_collection=await self._load_collection_documents(
                    embedded_collection=embedded_collection,
                    unembedded_collection=unembedded_collection,
                    embedder_type=embedder_type,
                    document_loader=document_loader,
                ),
                unembedded_collection=unembedded_collection,
                name=name,
                schema=schema,
                embedder=self._embedder_factory.create_embedder(embedder_type),
                embedding_cache_provider=self._embedding_cache_provider,
                version=1,
            )
            return cast(ChromaCollection[TDocument], self._collections[name])

        raise ValueError(f'ChromaDB collection "{name}" not found.')

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> ChromaCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(ChromaCollection[TDocument], collection)

        # Get or create unembedded collection for storing raw documents
        # Then get or create embedded collection for storing embeddings
        # Load and migrate documents from unembedded collection, then reindex embedded collection to ensure it is in sync
        unembedded_collection = next(
            (
                col
                for col in self.chroma_client.list_collections()
                if col.name == f"{name}_unembedded"
            ),
            None,
        ) or self.chroma_client.create_collection(
            name=f"{name}_unembedded",
            metadata={"version": 1},
            embedding_function=None,
        )

        embedded_collection = next(
            (
                col
                for col in self.chroma_client.list_collections()
                if col.name == self.format_collection_name(name, embedder_type)
            ),
            None,
        ) or self.chroma_client.create_collection(
            name=self.format_collection_name(name, embedder_type),
            metadata={"version": 1},
            configuration=CreateCollectionConfiguration(
                hnsw=CreateHNSWConfiguration(space="cosine")
            ),
        )

        self._collections[name] = ChromaCollection(
            self._logger,
            embedded_collection=await self._load_collection_documents(
                embedded_collection=embedded_collection,
                unembedded_collection=unembedded_collection,
                embedder_type=embedder_type,
                document_loader=document_loader,
            ),
            unembedded_collection=unembedded_collection,
            name=name,
            schema=schema,
            embedder=self._embedder_factory.create_embedder(embedder_type),
            embedding_cache_provider=self._embedding_cache_provider,
            version=1,
        )

        return cast(ChromaCollection[TDocument], self._collections[name])

    @override
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        if name not in self._collections:
            raise ValueError(f'Collection "{name}" not found.')

        self.chroma_client.delete_collection(name=name)
        self.chroma_client.delete_collection(name=f"{name}_unembedded")
        del self._collections[name]

    @override
    async def upsert_metadata(
        self,
        key: str,
        value: JSONSerializable,
    ) -> None:
        if metadata_collection := next(
            (col for col in self.chroma_client.list_collections() if col.name == "metadata"),
            None,
        ):
            pass
        else:
            metadata_collection = self.chroma_client.create_collection(
                name="metadata",
                embedding_function=None,
            )

        if metadatas := metadata_collection.get()["metadatas"]:
            document = cast(dict[str, JSONSerializable], metadatas[0])
            document[key] = value

            metadata_collection.update(
                ids=["__metadata__"],
                documents=["__metadata__"],
                metadatas=[cast(chromadb.Metadata, document)],
                embeddings=[0],
            )
        else:
            document = {key: value}

            metadata_collection.add(
                ids=["__metadata__"],
                documents=["__metadata__"],
                metadatas=[cast(chromadb.Metadata, document)],
                embeddings=[0],
            )

    @override
    async def remove_metadata(
        self,
        key: str,
    ) -> None:
        if metadata_collection := next(
            (col for col in self.chroma_client.list_collections() if col.name == "metadata"),
            None,
        ):
            if metadatas := metadata_collection.get()["metadatas"]:
                document = cast(dict[str, JSONSerializable], metadatas[0])
                document.pop(key)

                metadata_collection.update(
                    ids=["__metadata__"],
                    documents=["__metadata__"],
                    metadatas=[cast(chromadb.Metadata, document)],
                    embeddings=[0],
                )
            else:
                raise ValueError(f'Metadata with key "{key}" not found.')
        else:
            raise ValueError("Metadata collection not found.")

    @override
    async def read_metadata(
        self,
    ) -> dict[str, JSONSerializable]:
        if metadata_collection := next(
            (col for col in self.chroma_client.list_collections() if col.name == "metadata"),
            None,
        ):
            if metadatas := metadata_collection.get()["metadatas"]:
                return cast(dict[str, JSONSerializable], metadatas[0])
            else:
                return {}
        else:
            return {}


class ChromaCollection(Generic[TDocument], VectorCollection[TDocument]):
    def __init__(
        self,
        logger: Logger,
        embedded_collection: chromadb.Collection,
        unembedded_collection: chromadb.Collection,
        name: str,
        schema: type[TDocument],
        embedder: Embedder,
        embedding_cache_provider: EmbeddingCacheProvider,
        version: int,
    ) -> None:
        self._logger = logger
        self._name = name
        self._schema = schema
        self._embedder = embedder
        self._embedding_cache_provider = embedding_cache_provider
        self._version = version

        self._lock = ReaderWriterLock()
        self._unembedded_collection = unembedded_collection
        self.embedded_collection = embedded_collection

    @override
    async def find(
        self,
        filters: Where,
    ) -> Sequence[TDocument]:
        async with self._lock.reader_lock:
            if metadatas := self.embedded_collection.get(
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
            if metadatas := self.embedded_collection.get(
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

        if e := await self._embedding_cache_provider().get(
            embedder_type=type(self._embedder),
            texts=[document["content"]],
        ):
            embeddings = list(e.vectors)
        else:
            embeddings = list((await self._embedder.embed([document["content"]])).vectors)
            await self._embedding_cache_provider().set(
                embedder_type=type(self._embedder),
                texts=[document["content"]],
                vectors=embeddings,
            )

        async with self._lock.writer_lock:
            self._version += 1

            self._unembedded_collection.add(
                ids=[document["id"]],
                documents=[document["content"]],
                metadatas=[cast(chromadb.Metadata, document)],
                embeddings=[0],
            )

            self._unembedded_collection.modify(
                metadata={**self._unembedded_collection.metadata, **{"version": self._version}}
            )

            self.embedded_collection.add(
                ids=[document["id"]],
                documents=[document["content"]],
                metadatas=[cast(chromadb.Metadata, document)],
                embeddings=embeddings,
            )
            self.embedded_collection.modify(
                metadata={**self.embedded_collection.metadata, **{"version": self._version}}
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
            if docs := self.embedded_collection.get(where=cast(chromadb.Where, filters) or None)[
                "metadatas"
            ]:
                doc = docs[0]

                if "content" in params:
                    content = params["content"]
                    document = params["content"]
                else:
                    content = str(doc["content"])
                    document = str(doc["content"])

                if e := await self._embedding_cache_provider().get(
                    embedder_type=type(self._embedder),
                    texts=[content],
                ):
                    embeddings = list(e.vectors)
                else:
                    embeddings = list((await self._embedder.embed([content])).vectors)
                    await self._embedding_cache_provider().set(
                        embedder_type=type(self._embedder),
                        texts=[content],
                        vectors=embeddings,
                    )

                updated_document = {**doc, **params}

                self._version += 1

                self._unembedded_collection.update(
                    ids=[str(doc["id"])],
                    documents=[document],
                    metadatas=[cast(chromadb.Metadata, updated_document)],
                    embeddings=[0],
                )
                self._unembedded_collection.modify(
                    metadata={**self._unembedded_collection.metadata, **{"version": self._version}}
                )

                self.embedded_collection.update(
                    ids=[str(doc["id"])],
                    documents=[document],
                    metadatas=[cast(chromadb.Metadata, updated_document)],
                    embeddings=embeddings,  # type: ignore
                )
                self.embedded_collection.modify(
                    metadata={**self.embedded_collection.metadata, **{"version": self._version}}
                )

                return UpdateResult(
                    acknowledged=True,
                    matched_count=1,
                    modified_count=1,
                    updated_document=cast(TDocument, updated_document),
                )

            elif upsert:
                ensure_is_total(params, self._schema)

                if e := await self._embedding_cache_provider().get(
                    embedder_type=type(self._embedder),
                    texts=[params["content"]],
                ):
                    embeddings = list(e.vectors)
                else:
                    embeddings = list((await self._embedder.embed([params["content"]])).vectors)
                    await self._embedding_cache_provider().set(
                        embedder_type=type(self._embedder),
                        texts=[params["content"]],
                        vectors=embeddings,
                    )

                self._version += 1

                self._unembedded_collection.add(
                    ids=[params["id"]],
                    documents=[params["content"]],
                    metadatas=[cast(chromadb.Metadata, params)],
                    embeddings=[0],
                )
                self._unembedded_collection.modify(
                    metadata={**self._unembedded_collection.metadata, **{"version": self._version}}
                )

                self.embedded_collection.add(
                    ids=[params["id"]],
                    documents=[params["content"]],
                    metadatas=[cast(chromadb.Metadata, params)],
                    embeddings=embeddings,
                )
                self.embedded_collection.modify(
                    metadata={**self.embedded_collection.metadata, **{"version": self._version}}
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
            if docs := self.embedded_collection.get(where=cast(chromadb.Where, filters) or None)[
                "metadatas"
            ]:
                if len(docs) > 1:
                    raise ValueError(
                        f"ChromaCollection delete_one: detected more than one document with filters '{filters}'. Aborting..."
                    )
                deleted_document = docs[0]

                self._version += 1

                self._unembedded_collection.delete(where=cast(chromadb.Where, filters) or None)
                self._unembedded_collection.modify(
                    metadata={**self._unembedded_collection.metadata, **{"version": self._version}}
                )

                self.embedded_collection.delete(where=cast(chromadb.Where, filters) or None)
                self.embedded_collection.modify(
                    metadata={**self.embedded_collection.metadata, **{"version": self._version}}
                )

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

            docs = self.embedded_collection.query(
                where=cast(chromadb.Where, filters) or None,
                query_embeddings=query_embeddings,
                n_results=k,
            )

            if not docs["metadatas"]:
                return []

            self._logger.trace(
                f"Similar documents found\n{json.dumps(docs['metadatas'][0], indent=2)}"
            )

            assert docs["distances"]
            return [
                SimilarDocumentResult(document=cast(TDocument, m), distance=d)
                for m, d in zip(docs["metadatas"][0], docs["distances"][0])
            ]
