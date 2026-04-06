# Copyright 2026 Emcie Co Ltd.
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

# Requires MongoDB Atlas with vector search support.
# Atlas builds the index asynchronously after creation.

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable, Generic, Mapping, Optional, Sequence, cast

from pymongo import AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.operations import SearchIndexModel
from typing_extensions import Self, override

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import JSONSerializable
from parlant.core.loggers import Logger
from parlant.core.nlp.embedding import (
    Embedder,
    EmbedderFactory,
    EmbeddingCacheProvider,
)
from parlant.core.persistence.common import Where, ensure_is_total
from parlant.core.persistence.vector_database import (
    BaseDocument,
    BaseVectorCollection,
    DeleteResult,
    InsertResult,
    SimilarDocumentResult,
    TDocument,
    UpdateResult,
    VectorDatabase,
)
from parlant.core.tracer import Tracer

VECTOR_INDEX_NAME = "vector_index"
_METADATA_COLLECTION_NAME = "_vector_metadata"
_INDEX_READY_POLL_INTERVAL = 0.5
_INDEX_READY_TIMEOUT = 60.0


class MongoVectorDatabase(VectorDatabase):
    def __init__(
        self,
        mongo_client: AsyncMongoClient[Any],
        database_name: str,
        logger: Logger,
        tracer: Tracer,
        embedder_factory: EmbedderFactory,
        embedding_cache_provider: EmbeddingCacheProvider,
    ) -> None:
        self._mongo_client = mongo_client
        self._database_name = database_name
        self._logger = logger
        self._tracer = tracer
        self._embedder_factory = embedder_factory
        self._embedding_cache_provider = embedding_cache_provider

        self._database: AsyncDatabase[Any] = mongo_client[database_name]
        self._collections: dict[str, MongoVectorCollection[BaseDocument]] = {}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        self._collections.clear()

    def _format_collection_name(
        self,
        name: str,
        embedder_type: type[Embedder],
    ) -> str:
        return f"{name}_{embedder_type.__name__}"

    async def _ensure_collection_exists(
        self,
        collection_name: str,
    ) -> None:
        """Ensure the MongoDB collection exists by creating it if needed."""
        existing = await self._database.list_collection_names()
        if collection_name not in existing:
            await self._database.create_collection(collection_name)

    async def _ensure_vector_index(
        self,
        mongo_collection: AsyncCollection[Any],
        embedder: Embedder,
    ) -> None:
        """Create a vector search index on the collection if it doesn't exist."""
        # Ensure the collection exists in MongoDB before creating a search index
        await self._ensure_collection_exists(mongo_collection.name)

        # Check if the index already exists
        existing_indexes: list[Mapping[str, Any]] = []
        cursor = await mongo_collection.list_search_indexes()
        async for index in cursor:
            existing_indexes.append(index)

        for index in existing_indexes:
            if index.get("name") == VECTOR_INDEX_NAME:
                return

        search_index_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "__embedding__",
                        "numDimensions": embedder.dimensions,
                        "similarity": "cosine",
                    },
                    {
                        "type": "filter",
                        "path": "id",
                    },
                    {
                        "type": "filter",
                        "path": "version",
                    },
                    {
                        "type": "filter",
                        "path": "name",
                    },
                    {
                        "type": "filter",
                        "path": "checksum",
                    },
                ]
            },
            name=VECTOR_INDEX_NAME,
            type="vectorSearch",
        )

        await mongo_collection.create_search_index(search_index_model)
        self._logger.info(
            f"Created vector search index '{VECTOR_INDEX_NAME}' "
            f"on collection '{mongo_collection.name}'"
        )

        # Wait for the index to be ready
        await self._wait_for_index_ready(mongo_collection)

    async def _wait_for_index_ready(
        self,
        mongo_collection: AsyncCollection[Any],
    ) -> None:
        """Poll until the vector search index is queryable."""
        elapsed = 0.0
        while elapsed < _INDEX_READY_TIMEOUT:
            index_cursor = await mongo_collection.list_search_indexes()
            async for index in index_cursor:
                if index.get("name") == VECTOR_INDEX_NAME and index.get("queryable") is True:
                    return
            await asyncio.sleep(_INDEX_READY_POLL_INTERVAL)
            elapsed += _INDEX_READY_POLL_INTERVAL

        self._logger.warning(
            f"Vector search index '{VECTOR_INDEX_NAME}' on '{mongo_collection.name}' "
            f"did not become ready within {_INDEX_READY_TIMEOUT}s"
        )

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
    ) -> MongoVectorCollection[TDocument]:
        if name in self._collections:
            raise ValueError(f'Collection "{name}" already exists.')

        embedder = self._embedder_factory.create_embedder(embedder_type)
        collection_name = self._format_collection_name(name, embedder_type)

        mongo_collection = self._database[collection_name]
        await self._ensure_vector_index(mongo_collection, embedder)

        collection = MongoVectorCollection(
            logger=self._logger,
            tracer=self._tracer,
            mongo_collection=mongo_collection,
            name=name,
            schema=schema,
            embedder=embedder,
            embedding_cache_provider=self._embedding_cache_provider,
        )

        self._collections[name] = collection  # type: ignore[assignment]
        return collection  # type: ignore[return-value]

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> MongoVectorCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(MongoVectorCollection[TDocument], collection)

        embedder = self._embedder_factory.create_embedder(embedder_type)
        collection_name = self._format_collection_name(name, embedder_type)

        # Check if the collection exists in MongoDB
        existing_collections = await self._database.list_collection_names()
        if collection_name not in existing_collections:
            raise ValueError(f'Mongo vector collection "{name}" not found.')

        mongo_collection = self._database[collection_name]
        await self._ensure_vector_index(mongo_collection, embedder)

        # Run document loader on existing documents for migration
        await self._migrate_documents(mongo_collection, embedder, document_loader)

        collection = MongoVectorCollection(
            logger=self._logger,
            tracer=self._tracer,
            mongo_collection=mongo_collection,
            name=name,
            schema=schema,
            embedder=embedder,
            embedding_cache_provider=self._embedding_cache_provider,
        )

        self._collections[name] = collection  # type: ignore[assignment]
        return collection  # type: ignore[return-value]

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> MongoVectorCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(MongoVectorCollection[TDocument], collection)

        embedder = self._embedder_factory.create_embedder(embedder_type)
        collection_name = self._format_collection_name(name, embedder_type)

        mongo_collection = self._database[collection_name]
        await self._ensure_vector_index(mongo_collection, embedder)

        # Run document loader on existing documents for migration
        await self._migrate_documents(mongo_collection, embedder, document_loader)

        collection = MongoVectorCollection(
            logger=self._logger,
            tracer=self._tracer,
            mongo_collection=mongo_collection,
            name=name,
            schema=schema,
            embedder=embedder,
            embedding_cache_provider=self._embedding_cache_provider,
        )

        self._collections[name] = collection  # type: ignore[assignment]
        return collection  # type: ignore[return-value]

    async def _migrate_documents(
        self,
        mongo_collection: AsyncCollection[Any],
        embedder: Embedder,
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> None:
        """Run document_loader on all existing documents, updating or removing as needed."""
        cursor = mongo_collection.find({})
        async for raw_doc in cursor:
            mongo_id = raw_doc.pop("_id", None)
            # Strip embedding from the document before passing to loader
            raw_doc.pop("__embedding__", None)

            prospective_doc = cast(BaseDocument, raw_doc)
            try:
                loaded_doc = await document_loader(prospective_doc)
                if loaded_doc:
                    if loaded_doc != prospective_doc:
                        # Re-embed if content changed
                        content = loaded_doc.get("content", "")
                        embeddings = list((await embedder.embed([str(content)])).vectors)
                        update_doc = dict(loaded_doc)
                        update_doc["__embedding__"] = list(embeddings[0])
                        await mongo_collection.replace_one(
                            {"_id": mongo_id},
                            update_doc,
                        )
                else:
                    self._logger.warning(f'Failed to load document "{prospective_doc}"')
                    await mongo_collection.delete_one({"_id": mongo_id})
            except Exception as e:
                self._logger.error(f"Failed to load document '{prospective_doc}' with error: {e}.")

    @override
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        if name not in self._collections:
            raise ValueError(f'Collection "{name}" not found.')

        collection = self._collections[name]
        await collection._mongo_collection.drop()
        del self._collections[name]

    @override
    async def upsert_metadata(
        self,
        key: str,
        value: JSONSerializable,
    ) -> None:
        metadata_collection = self._database[_METADATA_COLLECTION_NAME]
        await metadata_collection.update_one(
            {"_id": "metadata"},
            {"$set": {key: value}},
            upsert=True,
        )

    @override
    async def remove_metadata(
        self,
        key: str,
    ) -> None:
        metadata_collection = self._database[_METADATA_COLLECTION_NAME]
        await metadata_collection.update_one(
            {"_id": "metadata"},
            {"$unset": {key: ""}},
        )

    @override
    async def read_metadata(
        self,
    ) -> Mapping[str, JSONSerializable]:
        metadata_collection = self._database[_METADATA_COLLECTION_NAME]
        doc = await metadata_collection.find_one({"_id": "metadata"})
        if doc is None:
            return {}

        result = dict(doc)
        result.pop("_id", None)
        return result


class MongoVectorCollection(Generic[TDocument], BaseVectorCollection[TDocument]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        mongo_collection: AsyncCollection[Any],
        name: str,
        schema: type[TDocument],
        embedder: Embedder,
        embedding_cache_provider: EmbeddingCacheProvider,
    ) -> None:
        super().__init__(tracer)

        self._logger = logger
        self._name = name
        self._schema = schema
        self._embedder = embedder
        self._embedding_cache_provider = embedding_cache_provider
        self._mongo_collection = mongo_collection
        self._lock = ReaderWriterLock()

    def _strip_internal_fields(self, doc: dict[str, Any]) -> TDocument:
        """Remove MongoDB internal fields and embedding from a document."""
        doc.pop("_id", None)
        doc.pop("__embedding__", None)
        return cast(TDocument, doc)

    async def _get_embedding(self, content: str) -> list[float]:
        """Get embedding for content, using cache if available."""
        if e := await self._embedding_cache_provider().get(
            embedder_type=type(self._embedder),
            texts=[content],
        ):
            return list(e.vectors[0])

        embeddings = list((await self._embedder.embed([content])).vectors)
        await self._embedding_cache_provider().set(
            embedder_type=type(self._embedder),
            texts=[content],
            vectors=list(embeddings),
        )
        return list(embeddings[0])

    @override
    async def find(
        self,
        filters: Where,
    ) -> Sequence[TDocument]:
        async with self._lock.reader_lock:
            mongo_filter = dict(filters) if filters else {}

            docs: list[TDocument] = []
            async for raw_doc in self._mongo_collection.find(mongo_filter):
                docs.append(self._strip_internal_fields(raw_doc))

            return docs

    @override
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]:
        async with self._lock.reader_lock:
            mongo_filter = dict(filters) if filters else {}

            raw_doc = await self._mongo_collection.find_one(mongo_filter)
            if raw_doc is None:
                return None

            return self._strip_internal_fields(raw_doc)

    @override
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult:
        ensure_is_total(document, self._schema)

        embedding = await self._get_embedding(document["content"])

        async with self._lock.writer_lock:
            mongo_doc = dict(document)
            mongo_doc["__embedding__"] = embedding

            await self._mongo_collection.insert_one(mongo_doc)

        return InsertResult(acknowledged=True)

    @override
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        async with self._lock.writer_lock:
            mongo_filter = dict(filters) if filters else {}

            existing = await self._mongo_collection.find_one(mongo_filter)

            if existing:
                doc = dict(existing)
                doc.pop("_id", None)
                doc.pop("__embedding__", None)

                content = params.get("content", doc.get("content", ""))
                embedding = await self._get_embedding(str(content))

                updated_doc = {**doc, **params}
                updated_doc["__embedding__"] = embedding

                await self._mongo_collection.replace_one(
                    {"_id": existing["_id"]},
                    updated_doc,
                )

                updated_doc.pop("_id", None)
                updated_doc.pop("__embedding__", None)

                return UpdateResult(
                    acknowledged=True,
                    matched_count=1,
                    modified_count=1,
                    updated_document=cast(TDocument, updated_doc),
                )

            elif upsert:
                ensure_is_total(params, self._schema)

                embedding = await self._get_embedding(params["content"])

                mongo_doc = dict(params)
                mongo_doc["__embedding__"] = embedding

                await self._mongo_collection.insert_one(mongo_doc)

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
            mongo_filter = dict(filters) if filters else {}

            existing = await self._mongo_collection.find_one(mongo_filter)

            if existing:
                await self._mongo_collection.delete_one({"_id": existing["_id"]})

                return DeleteResult(
                    deleted_count=1,
                    acknowledged=True,
                    deleted_document=self._strip_internal_fields(existing),
                )

            return DeleteResult(
                acknowledged=True,
                deleted_count=0,
                deleted_document=None,
            )

    @override
    async def do_find_similar_documents(
        self,
        filters: Where,
        query: str,
        k: int,
        hints: Mapping[str, Any] = {},
    ) -> Sequence[SimilarDocumentResult[TDocument]]:
        async with self._lock.reader_lock:
            query_embeddings = list((await self._embedder.embed([query], hints)).vectors)

            if not query_embeddings or len(query_embeddings[0]) == 0:
                self._logger.warning(f"Empty embedding generated for query: {query}")
                return []

            # Build the $vectorSearch stage
            vector_search_stage: dict[str, Any] = {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "__embedding__",
                    "queryVector": list(query_embeddings[0]),
                    "numCandidates": max(k * 10, 100),
                    "limit": k,
                }
            }

            # Add pre-filter if filters are provided
            mongo_filter = dict(filters) if filters else {}
            if mongo_filter:
                vector_search_stage["$vectorSearch"]["filter"] = mongo_filter

            pipeline: list[dict[str, Any]] = [
                vector_search_stage,
                {
                    "$addFields": {
                        "search_score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]

            results: list[SimilarDocumentResult[TDocument]] = []
            agg_cursor = await self._mongo_collection.aggregate(pipeline)
            async for raw_doc in agg_cursor:
                score = raw_doc.pop("search_score", 0.0)
                doc = self._strip_internal_fields(raw_doc)
                results.append(
                    SimilarDocumentResult(
                        document=doc,
                        distance=1.0 - float(score),
                    )
                )

            self._logger.trace(
                f"Similar documents found\n{json.dumps([dict(r.document) for r in results], indent=2)}"
            )

            return results
