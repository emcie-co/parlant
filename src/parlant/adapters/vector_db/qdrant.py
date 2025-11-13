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
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Awaitable, Callable, Generic, Optional, Sequence, cast
from typing_extensions import override, Self
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Filter, FieldCondition, Range, MatchValue, MatchAny


from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import JSONSerializable
from parlant.core.loggers import Logger
from parlant.core.nlp.embedding import (
    Embedder,
    EmbedderFactory,
    EmbeddingCacheProvider,
    NullEmbedder,
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


def _string_id_to_int(doc_id: str) -> int:
    """Convert a string ID to an integer for Qdrant point IDs."""
    # Use hash to convert string to integer
    # Take absolute value and use modulo to ensure it fits in int64 range
    hash_value = int(hashlib.sha256(doc_id.encode()).hexdigest()[:15], 16)
    # Ensure it's within safe int64 range (Qdrant supports int64)
    return hash_value % (2**63 - 1)


def _convert_where_to_qdrant_filter(where: Where) -> Optional[Filter]:
    """Convert a Where filter to a Qdrant Filter."""
    if not where:
        return None

    # Handle logical operators
    if "$and" in where:
        conditions = []
        for sub_filter in where["$and"]:
            qdrant_filter = _convert_where_to_qdrant_filter(sub_filter)
            if qdrant_filter:
                conditions.append(qdrant_filter)
        if conditions:
            return Filter(must=conditions)
        return None

    if "$or" in where:
        conditions = []
        for sub_filter in where["$or"]:
            qdrant_filter = _convert_where_to_qdrant_filter(sub_filter)
            if qdrant_filter:
                conditions.append(qdrant_filter)
        if conditions:
            return Filter(should=conditions)
        return None

    # Handle field conditions
    conditions = []
    for field_name, field_filter in where.items():
        if isinstance(field_filter, dict):
            for operator, filter_value in field_filter.items():
                if operator == "$eq":
                    conditions.append(
                        FieldCondition(key=field_name, match=MatchValue(value=filter_value))
                    )
                elif operator == "$ne":
                    # Qdrant doesn't have $ne, so we use must_not
                    conditions.append(
                        FieldCondition(key=field_name, match=MatchValue(value=filter_value))
                    )
                    return Filter(must_not=[FieldCondition(key=field_name, match=MatchValue(value=filter_value))])
                elif operator == "$gt":
                    conditions.append(
                        FieldCondition(key=field_name, range=Range(gt=filter_value))
                    )
                elif operator == "$gte":
                    conditions.append(
                        FieldCondition(key=field_name, range=Range(gte=filter_value))
                    )
                elif operator == "$lt":
                    conditions.append(
                        FieldCondition(key=field_name, range=Range(lt=filter_value))
                    )
                elif operator == "$lte":
                    conditions.append(
                        FieldCondition(key=field_name, range=Range(lte=filter_value))
                    )
                elif operator == "$in":
                    conditions.append(
                        FieldCondition(key=field_name, match=MatchAny(any=list(filter_value)))
                    )
                elif operator == "$nin":
                    # Qdrant doesn't have $nin, so we use must_not with MatchAny
                    return Filter(
                        must_not=[
                            FieldCondition(key=field_name, match=MatchAny(any=list(filter_value)))
                        ]
                    )

    if conditions:
        return Filter(must=conditions)
    return None


class QdrantDatabase(VectorDatabase):
    def __init__(
        self,
        logger: Logger,
        path: Optional[Path] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedder_factory: EmbedderFactory = None,
        embedding_cache_provider: EmbeddingCacheProvider = None,
    ) -> None:
        self._path = path
        self._url = url
        self._api_key = api_key
        self._logger = logger
        self._embedder_factory = embedder_factory

        self.qdrant_client: Optional[QdrantClient] = None
        self._collections: dict[str, QdrantCollection[BaseDocument]] = {}

        self._embedding_cache_provider = embedding_cache_provider

    async def __aenter__(self) -> Self:
        if self._path:
            # On Windows, retry if the storage folder is locked (from previous instance)
            # This handles cases where a previous instance hasn't fully released file locks
            max_retries = 5 if sys.platform == "win32" else 1
            for attempt in range(max_retries):
                try:
                    self.qdrant_client = QdrantClient(path=str(self._path))
                    break
                except RuntimeError as e:
                    if "already accessed" in str(e) and attempt < max_retries - 1:
                        import asyncio
                        # Exponential backoff: 0.05s, 0.1s, 0.15s, 0.2s, 0.25s
                        delay = 0.05 * (attempt + 1)
                        await asyncio.sleep(delay)
                        continue
                    raise
        elif self._url:
            self.qdrant_client = QdrantClient(url=self._url, api_key=self._api_key)
        else:
            # Default to in-memory for testing
            self.qdrant_client = QdrantClient(":memory:")
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        # Close collections first to release any resources
        self._collections.clear()

        # Close Qdrant client to release file locks (important on Windows)
        if self.qdrant_client is not None:
            try:
                # Explicitly close the client to release file locks and resources
                # This is critical on Windows where file locks can persist
                # The close() method releases all file handles and locks
                self.qdrant_client.close()
            except AttributeError:
                # If close() doesn't exist (shouldn't happen, but be safe)
                pass
            except Exception as e:
                # Log but don't fail if close() raises an exception
                self._logger.warning(f"Error closing Qdrant client: {e}")
            finally:
                # Clear the reference and force garbage collection
                # This ensures all Python references are released
                client = self.qdrant_client
                self.qdrant_client = None
                del client
                # Only force GC on Windows where file locks are more persistent
                if sys.platform == "win32":
                    gc.collect()
                    # On Windows, file locks may take a moment to be released by the OS
                    # Even after close(), Windows may need a brief moment to release locks
                    import asyncio
                    await asyncio.sleep(0.05)  # Minimal delay for Windows file lock release

    def format_collection_name(
        self,
        name: str,
        embedder_type: type[Embedder],
    ) -> str:
        return f"{name}_{embedder_type.__name__}"

    def _ensure_payload_index(self, collection_name: str, field_name: str) -> None:
        """Ensure a payload index exists for a field."""
        try:
            # Check if index exists
            collection_info = self.qdrant_client.get_collection(collection_name)
            existing_indexes = collection_info.payload_schema or {}

            # Create index if it doesn't exist
            if field_name not in existing_indexes:
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
        except Exception:
            # Try to create index anyway (might fail if it exists)
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # Index might already exist or creation failed

    # Loads documents from unembedded collection, migrates them if needed, and ensures embedded collection is in sync
    async def _load_collection_documents(
        self,
        embedded_collection_name: str,
        unembedded_collection_name: str,
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> str:
        failed_migrations: list[BaseDocument] = []
        embedder = self._embedder_factory.create_embedder(embedder_type)

        # Get all points from unembedded collection
        unembedded_points = self.qdrant_client.scroll(
            collection_name=unembedded_collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )[0]

        indexing_required = False

        if unembedded_points:
            for point in unembedded_points:
                prospective_doc = cast(BaseDocument, point.payload)
                try:
                    if loaded_doc := await document_loader(prospective_doc):
                        if loaded_doc != prospective_doc:
                            # Update the unembedded collection
                            self.qdrant_client.upsert(
                                collection_name=unembedded_collection_name,
                                points=[
                                    models.PointStruct(
                                        id=point.id,
                                        vector=[],
                                        payload=cast(dict, loaded_doc),
                                    )
                                ],
                            )
                            indexing_required = True
                    else:
                        self._logger.warning(f'Failed to load document "{prospective_doc}"')
                        self.qdrant_client.delete(
                            collection_name=unembedded_collection_name,
                            points_selector=models.PointIdsList(
                                points=[point.id],
                            ),
                        )
                        failed_migrations.append(prospective_doc)

                except Exception as e:
                    self._logger.error(f"Failed to load document '{prospective_doc}'. error: {e}.")
                    failed_migrations.append(prospective_doc)

            # Store failed migrations in a separate collection for debugging
            if failed_migrations:
                failed_migrations_collection = await self.get_or_create_collection(
                    "failed_migrations",
                    BaseDocument,
                    NullEmbedder,
                    identity_loader,
                )

                for failed_doc in failed_migrations:
                    point_id = _string_id_to_int(str(failed_doc["id"]))
                    self.qdrant_client.upsert(
                        collection_name=failed_migrations_collection._unembedded_collection_name,
                        points=[
                            models.PointStruct(
                                id=point_id,
                                vector=[0],
                                payload=cast(dict, failed_doc),
                            )
                        ],
                    )

        # Get version from special version point in collections
        unembedded_version = await self._get_collection_version(unembedded_collection_name)
        embedded_version = await self._get_collection_version(embedded_collection_name)

        if indexing_required or unembedded_version != embedded_version:
            await self._index_collection(embedded_collection_name, unembedded_collection_name, embedder)

        return embedded_collection_name

    async def _get_collection_version(self, collection_name: str) -> int:
        """Get version from a special version point in the collection."""
        version_point_id = _string_id_to_int("__version__")
        try:
            points = self.qdrant_client.retrieve(
                collection_name=collection_name,
                ids=[version_point_id],
                with_payload=True,
            )
            if points and points[0].payload:
                return cast(int, points[0].payload.get("version", 1))
        except Exception:
            pass
        return 1

    async def _set_collection_version(self, collection_name: str, version: int) -> None:
        """Set version in a special version point in the collection."""
        version_point_id = _string_id_to_int("__version__")

        # Get collection info to determine vector size
        collection_info = self.qdrant_client.get_collection(collection_name)
        vector_size = collection_info.config.params.vectors.size

        # Create zero vector of correct size
        zero_vector = [0.0] * vector_size

        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=version_point_id,
                    vector=zero_vector,
                    payload={"version": version},
                )
            ],
        )

    # Syncs embedded collection with unembedded collection
    async def _index_collection(
        self,
        embedded_collection_name: str,
        unembedded_collection_name: str,
        embedder: Embedder,
    ) -> None:
        # Get all points from unembedded collection
        unembedded_points = self.qdrant_client.scroll(
            collection_name=unembedded_collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )[0]

        # Map by document ID (string) from payload, not point ID (integer)
        # Filter out version points and other special points
        unembedded_docs_by_id = {
            point.payload["id"]: point 
            for point in unembedded_points 
            if "id" in point.payload and point.payload["id"] != "__version__"
        }

        # Get all points from embedded collection
        embedded_points = self.qdrant_client.scroll(
            collection_name=embedded_collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=True,
        )[0]

        # Map by document ID (string) from payload, not point ID (integer)
        # Filter out version points and other special points
        embedded_docs_by_id = {
            point.payload["id"]: point 
            for point in embedded_points 
            if "id" in point.payload and point.payload["id"] != "__version__"
        }

        # Remove docs from embedded collection that no longer exist in unembedded
        # Update embeddings for changed docs
        for doc_id, embedded_point in embedded_docs_by_id.items():
            if doc_id not in unembedded_docs_by_id:
                self.qdrant_client.delete(
                    collection_name=embedded_collection_name,
                    points_selector=models.PointIdsList(points=[embedded_point.id]),
                )
            else:
                unembedded_point = unembedded_docs_by_id[doc_id]
                unembedded_doc = unembedded_point.payload
                if embedded_point.payload.get("checksum") != unembedded_doc.get("checksum"):
                    embeddings = list(
                        (
                            await embedder.embed(
                                [cast(str, unembedded_doc["content"])]
                            )
                        ).vectors
                    )

                    self.qdrant_client.upsert(
                        collection_name=embedded_collection_name,
                        points=[
                            models.PointStruct(
                                id=embedded_point.id,  # Keep existing point ID
                                vector=embeddings[0],
                                payload=cast(dict, unembedded_doc),
                            )
                        ],
                    )
                unembedded_docs_by_id.pop(doc_id)

        # Add new docs from unembedded to embedded collection
        for doc_id, doc in unembedded_docs_by_id.items():
            embeddings = list(
                (await embedder.embed([cast(str, doc["content"])])).vectors
            )

            # Convert string ID to integer for Qdrant
            point_id = _string_id_to_int(str(doc_id))

            self.qdrant_client.upsert(
                collection_name=embedded_collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embeddings[0],
                        payload=cast(dict, doc),
                    )
                ],
            )

        # Update version in unembedded collection
        unembedded_version = await self._get_collection_version(unembedded_collection_name)
        await self._set_collection_version(unembedded_collection_name, unembedded_version)
        await self._set_collection_version(embedded_collection_name, unembedded_version)

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
    ) -> QdrantCollection[TDocument]:
        if name in self._collections:
            raise ValueError(f'Collection "{name}" already exists.')

        embedder = self._embedder_factory.create_embedder(embedder_type)
        vector_size = embedder.dimensions

        embedded_collection_name = self.format_collection_name(name, embedder_type)
        unembedded_collection_name = f"{name}_unembedded"

        # Create embedded collection
        self.qdrant_client.create_collection(
            collection_name=embedded_collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

        # Create unembedded collection (with empty vectors for metadata storage)
        self.qdrant_client.create_collection(
            collection_name=unembedded_collection_name,
            vectors_config=models.VectorParams(
                size=1,  # Minimal size for unembedded collection
                distance=models.Distance.COSINE,
            ),
        )

        # Ensure payload indexes exist
        self._ensure_payload_index(embedded_collection_name, "id")
        self._ensure_payload_index(unembedded_collection_name, "id")

        collection = QdrantCollection(
            self._logger,
            qdrant_client=self.qdrant_client,
            embedded_collection_name=embedded_collection_name,
            unembedded_collection_name=unembedded_collection_name,
            name=name,
            schema=schema,
            embedder=embedder,
            embedding_cache_provider=self._embedding_cache_provider,
            version=1,
        )
        collection._database = self
        self._collections[name] = collection

        return cast(QdrantCollection[TDocument], self._collections[name])

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> QdrantCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(QdrantCollection[TDocument], collection)

        # Find unembedded collection first which acts as the SSOT.
        unembedded_collection_name = f"{name}_unembedded"
        embedded_collection_name = self.format_collection_name(name, embedder_type)

        # Check if collections exist
        collections = self.qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]

        if unembedded_collection_name in collection_names:
            if embedded_collection_name not in collection_names:
                # Create embedded collection if it doesn't exist
                embedder = self._embedder_factory.create_embedder(embedder_type)
                self.qdrant_client.create_collection(
                    collection_name=embedded_collection_name,
                    vectors_config=models.VectorParams(
                        size=embedder.dimensions,
                        distance=models.Distance.COSINE,
                    ),
                )
                # Ensure payload index exists
                self._ensure_payload_index(embedded_collection_name, "id")

            await self._index_collection(
                embedded_collection_name=embedded_collection_name,
                unembedded_collection_name=unembedded_collection_name,
                embedder=self._embedder_factory.create_embedder(embedder_type),
            )

            collection = QdrantCollection(
                self._logger,
                qdrant_client=self.qdrant_client,
                embedded_collection_name=await self._load_collection_documents(
                    embedded_collection_name=embedded_collection_name,
                    unembedded_collection_name=unembedded_collection_name,
                    embedder_type=embedder_type,
                    document_loader=document_loader,
                ),
                unembedded_collection_name=unembedded_collection_name,
                name=name,
                schema=schema,
                embedder=self._embedder_factory.create_embedder(embedder_type),
                embedding_cache_provider=self._embedding_cache_provider,
                version=1,
            )
            collection._database = self
            self._collections[name] = collection
            return cast(QdrantCollection[TDocument], self._collections[name])

        raise ValueError(f'Qdrant collection "{name}" not found.')

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> QdrantCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(QdrantCollection[TDocument], collection)

        embedder = self._embedder_factory.create_embedder(embedder_type)
        vector_size = embedder.dimensions

        embedded_collection_name = self.format_collection_name(name, embedder_type)
        unembedded_collection_name = f"{name}_unembedded"

        # Get or create collections
        collections = self.qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]

        if unembedded_collection_name not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=unembedded_collection_name,
                vectors_config=models.VectorParams(
                    size=1,  # Minimal size for unembedded collection
                    distance=models.Distance.COSINE,
                ),
            )
        if embedded_collection_name not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=embedded_collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

        # Ensure payload indexes exist for both collections
        self._ensure_payload_index(unembedded_collection_name, "id")
        self._ensure_payload_index(embedded_collection_name, "id")

        collection = QdrantCollection(
            self._logger,
            qdrant_client=self.qdrant_client,
            embedded_collection_name=await self._load_collection_documents(
                embedded_collection_name=embedded_collection_name,
                unembedded_collection_name=unembedded_collection_name,
                embedder_type=embedder_type,
                document_loader=document_loader,
            ),
            unembedded_collection_name=unembedded_collection_name,
            name=name,
            schema=schema,
            embedder=embedder,
            embedding_cache_provider=self._embedding_cache_provider,
            version=1,
        )
        collection._database = self
        self._collections[name] = collection

        return cast(QdrantCollection[TDocument], self._collections[name])

    @override
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        if name not in self._collections:
            raise ValueError(f'Collection "{name}" not found.')

        embedded_collection_name = self.format_collection_name(
            name, type(self._collections[name]._embedder)
        )
        unembedded_collection_name = f"{name}_unembedded"

        self.qdrant_client.delete_collection(collection_name=embedded_collection_name)
        self.qdrant_client.delete_collection(collection_name=unembedded_collection_name)
        del self._collections[name]

    @override
    async def upsert_metadata(
        self,
        key: str,
        value: JSONSerializable,
    ) -> None:
        metadata_collection_name = "metadata"

        # Check if metadata collection exists
        collections = self.qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]

        if metadata_collection_name not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=metadata_collection_name,
                vectors_config=models.VectorParams(
                    size=1,
                    distance=models.Distance.COSINE,
                ),
            )

        # Get existing metadata
        points = self.qdrant_client.scroll(
            collection_name=metadata_collection_name,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )[0]

        if points:
            document = cast(dict[str, JSONSerializable], points[0].payload)
            document[key] = value

            self.qdrant_client.upsert(
                collection_name=metadata_collection_name,
                points=[
                    models.PointStruct(
                        id=points[0].id,
                        vector=[0],
                        payload=cast(dict, document),
                    )
                ],
            )
        else:
            document = {key: value}

            metadata_point_id = _string_id_to_int("__metadata__")
            self.qdrant_client.upsert(
                collection_name=metadata_collection_name,
                points=[
                    models.PointStruct(
                        id=metadata_point_id,
                        vector=[0],
                        payload=cast(dict, document),
                    )
                ],
            )

    @override
    async def remove_metadata(
        self,
        key: str,
    ) -> None:
        metadata_collection_name = "metadata"

        collections = self.qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]

        if metadata_collection_name in collection_names:
            points = self.qdrant_client.scroll(
                collection_name=metadata_collection_name,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )[0]

            if points:
                document = cast(dict[str, JSONSerializable], points[0].payload)
                document.pop(key)

                self.qdrant_client.upsert(
                    collection_name=metadata_collection_name,
                    points=[
                        models.PointStruct(
                            id=points[0].id,
                            vector=[0],
                            payload=cast(dict, document),
                        )
                    ],
                )
            else:
                raise ValueError(f'Metadata with key "{key}" not found.')
        else:
            raise ValueError("Metadata collection not found.")

    @override
    async def read_metadata(
        self,
    ) -> dict[str, JSONSerializable]:
        metadata_collection_name = "metadata"

        collections = self.qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]

        if metadata_collection_name in collection_names:
            points = self.qdrant_client.scroll(
                collection_name=metadata_collection_name,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )[0]

            if points:
                return cast(dict[str, JSONSerializable], points[0].payload)
            else:
                return {}
        else:
            return {}


class QdrantCollection(Generic[TDocument], VectorCollection[TDocument]):
    def __init__(
        self,
        logger: Logger,
        qdrant_client: QdrantClient,
        embedded_collection_name: str,
        unembedded_collection_name: str,
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
        self._unembedded_collection_name = unembedded_collection_name
        self.embedded_collection_name = embedded_collection_name
        self.qdrant_client = qdrant_client
        self._database: Optional[QdrantDatabase] = None  # Reference to parent database for version methods

    @override
    async def find(
        self,
        filters: Where,
    ) -> Sequence[TDocument]:
        async with self._lock.reader_lock:
            # Ensure indexes exist for filtering
            if filters:
                self._database._ensure_payload_index(self.embedded_collection_name, "id")

            qdrant_filter = _convert_where_to_qdrant_filter(filters)

            try:
                points = self.qdrant_client.scroll(
                    collection_name=self.embedded_collection_name,
                    scroll_filter=qdrant_filter,
                    limit=10000,
                    with_payload=True,
                    with_vectors=False,
                )[0]
            except Exception:
                # If filter fails due to missing index, scroll all and filter in memory
                if qdrant_filter:
                    all_points = self.qdrant_client.scroll(
                        collection_name=self.embedded_collection_name,
                        limit=10000,
                        with_payload=True,
                        with_vectors=False,
                    )[0]
                    # Filter in memory
                    from parlant.core.persistence.common import matches_filters
                    points = [p for p in all_points if matches_filters(filters, p.payload)]
                else:
                    points = []

            return [cast(TDocument, point.payload) for point in points]

    @override
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]:
        async with self._lock.reader_lock:
            # Ensure indexes exist for filtering
            if filters:
                self._database._ensure_payload_index(self.embedded_collection_name, "id")

            qdrant_filter = _convert_where_to_qdrant_filter(filters)

            try:
                points = self.qdrant_client.scroll(
                    collection_name=self.embedded_collection_name,
                    scroll_filter=qdrant_filter,
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )[0]
            except Exception:
                # If filter fails due to missing index, scroll all and filter in memory
                if qdrant_filter:
                    all_points = self.qdrant_client.scroll(
                        collection_name=self.embedded_collection_name,
                        limit=10000,
                        with_payload=True,
                        with_vectors=False,
                    )[0]
                    # Filter in memory
                    from parlant.core.persistence.common import matches_filters
                    points = [p for p in all_points if matches_filters(filters, p.payload)][:1]
                else:
                    points = []

            if points:
                return cast(TDocument, points[0].payload)

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

            # Convert string ID to integer for Qdrant
            point_id = _string_id_to_int(str(document["id"]))

            # Insert into unembedded collection
            self.qdrant_client.upsert(
                collection_name=self._unembedded_collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=[0],
                        payload=cast(dict, document),
                    )
                ],
            )

            # Insert into embedded collection
            self.qdrant_client.upsert(
                collection_name=self.embedded_collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embeddings[0],
                        payload=cast(dict, document),
                    )
                ],
            )

            # Update version in both collections
            if self._database:
                await self._database._set_collection_version(self._unembedded_collection_name, self._version)
                await self._database._set_collection_version(self.embedded_collection_name, self._version)

        return InsertResult(acknowledged=True)

    @override
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        async with self._lock.writer_lock:
            qdrant_filter = _convert_where_to_qdrant_filter(filters)

            points = self.qdrant_client.scroll(
                collection_name=self.embedded_collection_name,
                scroll_filter=qdrant_filter,
                limit=1,
                with_payload=True,
                with_vectors=True,
            )[0]

            if points:
                point = points[0]
                doc = cast(dict, point.payload)

                if "content" in params:
                    content = params["content"]
                else:
                    content = str(doc["content"])

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

                # Update unembedded collection
                self.qdrant_client.upsert(
                    collection_name=self._unembedded_collection_name,
                    points=[
                        models.PointStruct(
                            id=point.id,  # point.id is already an integer
                            vector=[0],
                            payload=cast(dict, updated_document),
                        )
                    ],
                )

                # Update embedded collection
                self.qdrant_client.upsert(
                    collection_name=self.embedded_collection_name,
                    points=[
                        models.PointStruct(
                            id=point.id,  # point.id is already an integer
                            vector=embeddings[0],
                            payload=cast(dict, updated_document),
                        )
                    ],
                )

                # Update version in both collections
                if self._database:
                    await self._database._set_collection_version(self._unembedded_collection_name, self._version)
                    await self._database._set_collection_version(self.embedded_collection_name, self._version)

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

                # Convert string ID to integer for Qdrant
                point_id = _string_id_to_int(str(params["id"]))

                # Insert into unembedded collection
                self.qdrant_client.upsert(
                    collection_name=self._unembedded_collection_name,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=[0],
                            payload=cast(dict, params),
                        )
                    ],
                )

                # Insert into embedded collection
                self.qdrant_client.upsert(
                    collection_name=self.embedded_collection_name,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=embeddings[0],
                            payload=cast(dict, params),
                        )
                    ],
                )

                # Update version in both collections
                if self._database:
                    await self._database._set_collection_version(self._unembedded_collection_name, self._version)
                    await self._database._set_collection_version(self.embedded_collection_name, self._version)

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
            qdrant_filter = _convert_where_to_qdrant_filter(filters)

            points = self.qdrant_client.scroll(
                collection_name=self.embedded_collection_name,
                scroll_filter=qdrant_filter,
                limit=2,  # Check for more than one
                with_payload=True,
                with_vectors=False,
            )[0]

            if len(points) > 1:
                raise ValueError(
                    f"QdrantCollection delete_one: detected more than one document with filters '{filters}'. Aborting..."
                )

            if points:
                deleted_document = cast(TDocument, points[0].payload)
                point_id = points[0].id

                self._version += 1

                # Delete from unembedded collection
                self.qdrant_client.delete(
                    collection_name=self._unembedded_collection_name,
                    points_selector=models.PointIdsList(points=[point_id]),
                )

                # Delete from embedded collection
                self.qdrant_client.delete(
                    collection_name=self.embedded_collection_name,
                    points_selector=models.PointIdsList(points=[point_id]),
                )

                # Update version in both collections
                if self._database:
                    await self._database._set_collection_version(self._unembedded_collection_name, self._version)
                    await self._database._set_collection_version(self.embedded_collection_name, self._version)

                return DeleteResult(
                    deleted_count=1,
                    acknowledged=True,
                    deleted_document=deleted_document,
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
            qdrant_filter = _convert_where_to_qdrant_filter(filters)

            search_results = self.qdrant_client.search(
                collection_name=self.embedded_collection_name,
                query_vector=query_embeddings[0],
                query_filter=qdrant_filter,
                limit=k,
            )

            if not search_results:
                return []

            self._logger.trace(
                f"Similar documents found\n{json.dumps([r.payload for r in search_results], indent=2)}"
            )

            return [
                SimilarDocumentResult(
                    document=cast(TDocument, result.payload),
                    distance=1.0 - result.score,  # Convert similarity to distance
                )
                for result in search_results
            ]

