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

# Maintainer: Sifat Hasan <sihabhossan633@gmail.com>

from __future__ import annotations
import asyncio
import json
from typing import Any, Awaitable, Callable, Generic, Optional, Sequence, TypeVar, cast
from typing_extensions import override, Self
from pinecone import Pinecone, ServerlessSpec  # type: ignore[import-untyped]
from pinecone.exceptions import PineconeException  # type: ignore[import-untyped]


class _EmptyQueryResults:
    """Empty query results for fallback when Pinecone query fails."""
    matches: list[Any] = []


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


T = TypeVar("T")


async def _retry_on_timeout_async(
    operation: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    logger: Optional[Logger] = None,
) -> T:
    """
    Retry an async operation on timeout errors with exponential backoff.

    Args:
        operation: The async operation to retry (callable that returns Awaitable[T])
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        logger: Optional logger for warning messages

    Returns:
        The result of the operation

    Raises:
        The last exception if all retries fail
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries):
        try:
            return await operation()
        except (PineconeException, Exception) as e:
            # Check if it's a timeout error
            error_str = str(e).lower()
            is_timeout = (
                "timeout" in error_str
                or "read operation timed out" in error_str
                or "readtimeout" in error_str
            )

            if is_timeout and attempt < max_retries - 1:
                delay = base_delay * (2**attempt)  # Exponential backoff: 1s, 2s, 4s
                if logger:
                    logger.warning(
                        f"Pinecone operation timed out (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay}s..."
                    )
                await asyncio.sleep(delay)
                last_exception = e
                continue
            else:
                # Not a timeout or out of retries
                raise

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed unexpectedly")


def _extract_field_names_from_where(where: Where, field_names: set[str]) -> None:
    """Recursively extract all field names from a Where filter."""
    if not where:
        return

    # Handle logical operators
    if "$and" in where:
        for sub_filter in where["$and"]:
            if isinstance(sub_filter, dict):
                _extract_field_names_from_where(sub_filter, field_names)
        return

    if "$or" in where:
        for sub_filter in where["$or"]:
            if isinstance(sub_filter, dict):
                _extract_field_names_from_where(sub_filter, field_names)
        return

    # Handle field conditions
    for field_name, field_filter in where.items():
        if isinstance(field_filter, dict):
            # This is a field with operators
            field_names.add(field_name)
            # Recursively check nested filters (for complex nested structures)
            for operator, filter_value in field_filter.items():
                if operator in ["$and", "$or"] and isinstance(filter_value, list):
                    for nested_filter in filter_value:
                        if isinstance(nested_filter, dict):
                            _extract_field_names_from_where(nested_filter, field_names)


def _build_collection_filter(
    collection_name: str,
    collection_type: str,
    embedder_type_name: str,
    additional_filter: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a Pinecone filter that includes collection metadata filtering."""
    collection_filter = {
        "$and": [
            {"_parlant_collection": {"$eq": collection_name}},
            {"_parlant_type": {"$eq": collection_type}},
            {"_parlant_embedder": {"$eq": embedder_type_name}},
        ]
    }

    if additional_filter:
        collection_filter["$and"].append(additional_filter)

    return collection_filter


def _make_vector_id(collection_name: str, collection_type: str, doc_id: str) -> str:
    """Create a unique vector ID that includes collection information."""
    return f"{collection_name}_{collection_type}_{doc_id}"


def _convert_where_to_pinecone_filter(where: Where) -> Optional[dict[str, Any]]:
    """Convert a Where filter to a Pinecone filter."""
    if not where:
        return None

    # Handle logical operators
    if "$and" in where:
        and_conditions: list[dict[str, Any]] = []
        for sub_filter in where["$and"]:
            if isinstance(sub_filter, dict):
                pinecone_filter = _convert_where_to_pinecone_filter(sub_filter)
                if pinecone_filter:
                    and_conditions.append(pinecone_filter)
        if and_conditions:
            return {"$and": and_conditions}
        return None

    if "$or" in where:
        or_conditions: list[dict[str, Any]] = []
        for sub_filter in where["$or"]:
            if isinstance(sub_filter, dict):
                pinecone_filter = _convert_where_to_pinecone_filter(sub_filter)
                if pinecone_filter:
                    or_conditions.append(pinecone_filter)
        if or_conditions:
            return {"$or": or_conditions}
        return None

    # Handle field conditions
    field_conditions: dict[str, Any] = {}
    for field_name, field_filter in where.items():
        if isinstance(field_filter, dict):
            for operator, filter_value in field_filter.items():
                if operator == "$eq":
                    field_conditions[field_name] = {"$eq": filter_value}
                elif operator == "$ne":
                    field_conditions[field_name] = {"$ne": filter_value}
                elif operator == "$gt":
                    field_conditions[field_name] = {"$gt": filter_value}
                elif operator == "$gte":
                    field_conditions[field_name] = {"$gte": filter_value}
                elif operator == "$lt":
                    field_conditions[field_name] = {"$lt": filter_value}
                elif operator == "$lte":
                    field_conditions[field_name] = {"$lte": filter_value}
                elif operator == "$in":
                    field_conditions[field_name] = {"$in": list(filter_value)}
                elif operator == "$nin":
                    field_conditions[field_name] = {"$nin": list(filter_value)}

    if field_conditions:
        # If multiple field conditions, wrap in $and
        if len(field_conditions) > 1:
            return {"$and": [{k: v} for k, v in field_conditions.items()]}
        # Single condition
        return field_conditions
    return None


class PineconeDatabase(VectorDatabase):
    def __init__(
        self,
        logger: Logger,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        embedder_factory: Optional[EmbedderFactory] = None,
        embedding_cache_provider: Optional[EmbeddingCacheProvider] = None,
    ) -> None:
        self._api_key = api_key
        self._index_name = index_name
        self._logger = logger
        self._embedder_factory = embedder_factory

        self.pinecone_client: Optional[Pinecone] = None
        self._index: Optional[Any] = None  # Pinecone Index object
        self._index_dimension: int = (
            1536  # Default dimension, will be updated when index is created
        )
        self._collections: dict[str, PineconeCollection[BaseDocument]] = {}

        self._embedding_cache_provider = embedding_cache_provider

    async def __aenter__(self) -> Self:
        import os

        if self._api_key:
            self.pinecone_client = Pinecone(api_key=self._api_key)
        else:
            # Try to get from environment variable
            api_key = os.getenv("PINECONE_API_KEY")
            if api_key:
                self.pinecone_client = Pinecone(api_key=api_key)
            else:
                raise ValueError(
                    "Pinecone API key must be provided either as parameter or PINECONE_API_KEY environment variable"
                )

        # Get index name from parameter, env var, or default
        if not self._index_name:
            self._index_name = os.getenv("PINECONE_INDEX_NAME", "parlant-pineconedb")

        # Get embedder dimensions from embedder factory if available
        # Otherwise use a default dimension (1536 for OpenAI embeddings)
        default_dimension = 1536
        if self._embedder_factory:
            try:
                # Try to get a sample embedder to determine dimension
                # This is a best-effort approach
                sample_embedder = self._embedder_factory.create_embedder(NullEmbedder)
                if hasattr(sample_embedder, "dimensions"):
                    default_dimension = sample_embedder.dimensions
            except Exception:
                pass  # Use default if we can't determine

        # Check if index exists, create if not
        indexes = await asyncio.to_thread(self.pinecone_client.list_indexes)
        index_names = list(indexes) if isinstance(indexes, list) else [idx.name for idx in indexes]

        if self._index_name not in index_names:
            try:
                await asyncio.to_thread(
                    self.pinecone_client.create_index,
                    name=self._index_name,
                    dimension=default_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                self._logger.info(
                    f"Created Pinecone index: {self._index_name} with dimension {default_dimension}"
                )
            except Exception as e:
                # Index might have been created by another process
                if "already exists" not in str(e).lower():
                    raise
        else:
            self._logger.info(f"Using existing Pinecone index: {self._index_name}")

        # Get the index
        self._index = self.pinecone_client.Index(self._index_name)

        # Set the index dimension (use default for now)
        # In practice, all vectors in the index must have the same dimension
        # which should match the embedder dimension
        self._index_dimension = default_dimension

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        # Close collections first to release any resources
        self._collections.clear()

        # Pinecone client doesn't need explicit close, but clear reference
        self._index = None
        self.pinecone_client = None

    def format_collection_name(
        self,
        name: str,
        embedder_type: type[Embedder],
    ) -> str:
        return f"{name}_{embedder_type.__name__}"

    # Loads documents from unembedded collection, migrates them if needed, and ensures embedded collection is in sync
    async def _load_collection_documents(
        self,
        embedded_collection_name: str,
        unembedded_collection_name: str,
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> str:
        assert self.pinecone_client is not None, "Pinecone client must be initialized"
        assert self._index is not None, "Pinecone index must be initialized"
        assert self._embedder_factory is not None, "Embedder factory must be provided"
        failed_migrations: list[BaseDocument] = []
        embedder = self._embedder_factory.create_embedder(embedder_type)
        embedder_type_name = embedder_type.__name__

        # Extract collection name from unembedded_collection_name (remove "_unembedded" suffix)
        collection_name = unembedded_collection_name.replace("_unembedded", "")

        # Get all vectors from unembedded collection using query with metadata filter
        unembedded_filter = _build_collection_filter(
            collection_name=collection_name,
            collection_type="unembedded",
            embedder_type_name=embedder_type_name,
        )

        try:
            unembedded_results = await asyncio.to_thread(
                self._index.query,
                vector=[0.0] * embedder.dimensions,  # Dummy vector for fetching all
                top_k=10000,
                filter=unembedded_filter,
                include_metadata=True,
            )
        except Exception:
            # Fallback: create empty results
            unembedded_results = _EmptyQueryResults()

        indexing_required = False

        if unembedded_results.matches:
            for match in unembedded_results.matches:
                # Remove internal Parlant metadata fields
                doc_metadata = {
                    k: v for k, v in match.metadata.items() if not k.startswith("_parlant_")
                }
                prospective_doc = cast(BaseDocument, doc_metadata)
                try:
                    if loaded_doc := await document_loader(prospective_doc):
                        if loaded_doc != prospective_doc:
                            # Update the unembedded collection
                            doc_with_metadata = dict(cast(dict[str, Any], loaded_doc))
                            doc_with_metadata["_parlant_collection"] = collection_name
                            doc_with_metadata["_parlant_type"] = "unembedded"
                            doc_with_metadata["_parlant_embedder"] = embedder_type_name

                            await asyncio.to_thread(
                                self._index.upsert,
                                vectors=[
                                    {
                                        "id": match.id,
                                        "values": [0.0] * embedder.dimensions,
                                        "metadata": doc_with_metadata,
                                    }
                                ],
                            )
                            indexing_required = True
                    else:
                        self._logger.warning(f'Failed to load document "{prospective_doc}"')
                        await asyncio.to_thread(self._index.delete, ids=[match.id])
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
                    # Use the collection interface consistently instead of direct Pinecone operations
                    await failed_migrations_collection.insert_one(failed_doc)

        # Get version from special version point in collections
        unembedded_version = await self._get_collection_version(unembedded_collection_name)
        embedded_version = await self._get_collection_version(embedded_collection_name)

        if indexing_required or unembedded_version != embedded_version:
            await self._index_collection(
                embedded_collection_name, unembedded_collection_name, embedder
            )

        return embedded_collection_name

    async def _get_collection_version(self, collection_name: str) -> int:
        """Get version from metadata collection."""
        assert self.pinecone_client is not None, "Pinecone client must be initialized"
        version_key = f"{collection_name}_version"
        try:
            metadata = await self.read_metadata()
            return cast(int, metadata.get(version_key, 1))
        except Exception:
            return 1

    async def _set_collection_version(self, collection_name: str, version: int) -> None:
        """Set version in metadata collection."""
        assert self.pinecone_client is not None, "Pinecone client must be initialized"
        version_key = f"{collection_name}_version"
        await self.upsert_metadata(version_key, version)

    # Syncs embedded collection with unembedded collection
    async def _index_collection(
        self,
        embedded_collection_name: str,
        unembedded_collection_name: str,
        embedder: Embedder,
    ) -> None:
        assert self.pinecone_client is not None, "Pinecone client must be initialized"
        assert self._index is not None, "Pinecone index must be initialized"

        embedder_type_name = type(embedder).__name__
        # Extract collection name from unembedded_collection_name (remove "_unembedded" suffix)
        collection_name = unembedded_collection_name.replace("_unembedded", "")

        # Get all vectors from unembedded collection using query with metadata filter
        unembedded_filter = _build_collection_filter(
            collection_name=collection_name,
            collection_type="unembedded",
            embedder_type_name=embedder_type_name,
        )

        try:
            unembedded_results = await asyncio.to_thread(
                self._index.query,
                vector=[0.0] * embedder.dimensions,  # Dummy vector for fetching all
                top_k=10000,
                filter=unembedded_filter,
                include_metadata=True,
            )
        except Exception:
            # Fallback: create empty results
            unembedded_results = _EmptyQueryResults()

        # Map by document ID (string) from metadata (remove Parlant internal fields)
        unembedded_docs_by_id = {}
        for match in unembedded_results.matches:
            if match.metadata is not None and "id" in match.metadata:
                doc_metadata = {
                    k: v for k, v in match.metadata.items() if not k.startswith("_parlant_")
                }
                doc_id = cast(str, doc_metadata["id"])
                unembedded_docs_by_id[doc_id] = (match, doc_metadata)

        # Get all vectors from embedded collection using query with metadata filter
        embedded_filter = _build_collection_filter(
            collection_name=collection_name,
            collection_type="embedded",
            embedder_type_name=embedder_type_name,
        )

        try:
            embedded_results = await asyncio.to_thread(
                self._index.query,
                vector=[0.0] * embedder.dimensions,  # Dummy vector for fetching all
                top_k=10000,
                filter=embedded_filter,
                include_metadata=True,
            )
        except Exception:
            # Fallback: create empty results
            embedded_results = _EmptyQueryResults()

        # Map by document ID (string) from metadata (remove Parlant internal fields)
        embedded_docs_by_id = {}
        for match in embedded_results.matches:
            if match.metadata is not None and "id" in match.metadata:
                doc_metadata = {
                    k: v for k, v in match.metadata.items() if not k.startswith("_parlant_")
                }
                doc_id = cast(str, doc_metadata["id"])
                embedded_docs_by_id[doc_id] = (match, doc_metadata)

        # Remove docs from embedded collection that no longer exist in unembedded
        # Update embeddings for changed docs
        for doc_id, (embedded_match, embedded_doc_metadata) in embedded_docs_by_id.items():
            if doc_id not in unembedded_docs_by_id:
                await asyncio.to_thread(self._index.delete, ids=[embedded_match.id])
            else:
                unembedded_match, unembedded_doc = unembedded_docs_by_id[doc_id]
                # Only recompute embeddings if checksum changed
                if embedded_doc_metadata.get("checksum") != unembedded_doc.get("checksum"):
                    embeddings = list(
                        (await embedder.embed([cast(str, unembedded_doc["content"])])).vectors
                    )
                    if not embeddings or len(embeddings[0]) == 0:
                        self._logger.warning(
                            f"Empty embedding for document {doc_id}, skipping sync"
                        )
                        continue
                    vector = embeddings[0]
                else:
                    # Use existing vector if checksum hasn't changed
                    vector = embedded_match.values

                # Add Parlant metadata
                doc_with_metadata = dict(unembedded_doc)
                doc_with_metadata["_parlant_collection"] = collection_name
                doc_with_metadata["_parlant_type"] = "embedded"
                doc_with_metadata["_parlant_embedder"] = embedder_type_name

                await asyncio.to_thread(
                    self._index.upsert,
                    vectors=[
                        {
                            "id": embedded_match.id,
                            "values": vector,
                            "metadata": doc_with_metadata,
                        }
                    ],
                )
                unembedded_docs_by_id.pop(doc_id)

        # Add new docs from unembedded to embedded collection
        for doc_id, (unembedded_match, unembedded_doc) in unembedded_docs_by_id.items():
            embeddings = list(
                (await embedder.embed([cast(str, unembedded_doc["content"])])).vectors
            )

            if not embeddings or len(embeddings[0]) == 0:
                self._logger.warning(f"Empty embedding for document {doc_id}, skipping")
                continue

            # Create embedded vector ID
            embedded_vector_id = _make_vector_id(collection_name, "embedded", doc_id)

            # Add Parlant metadata
            doc_with_metadata = dict(unembedded_doc)
            doc_with_metadata["_parlant_collection"] = collection_name
            doc_with_metadata["_parlant_type"] = "embedded"
            doc_with_metadata["_parlant_embedder"] = embedder_type_name

            await asyncio.to_thread(
                self._index.upsert,
                vectors=[
                    {
                        "id": embedded_vector_id,
                        "values": embeddings[0],
                        "metadata": doc_with_metadata,
                    }
                ],
            )

        # Update version in both collections
        unembedded_version = await self._get_collection_version(unembedded_collection_name)
        await self._set_collection_version(unembedded_collection_name, unembedded_version)
        await self._set_collection_version(embedded_collection_name, unembedded_version)

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
    ) -> PineconeCollection[TDocument]:
        assert self.pinecone_client is not None, "Pinecone client must be initialized"
        assert self._index is not None, "Pinecone index must be initialized"
        assert self._embedder_factory is not None, "Embedder factory must be provided"
        assert self._embedding_cache_provider is not None, (
            "Embedding cache provider must be provided"
        )
        if name in self._collections:
            raise ValueError(f'Collection "{name}" already exists.')

        embedder = self._embedder_factory.create_embedder(embedder_type)

        # Check if index dimension matches - if not, we may need to handle this
        # For now, we'll assume the index was created with the correct dimension
        # In production, you might want to check and update the index dimension

        embedded_collection_name = self.format_collection_name(name, embedder_type)
        unembedded_collection_name = f"{name}_unembedded"

        collection = PineconeCollection(
            self._logger,
            pinecone_index=self._index,
            embedded_collection_name=embedded_collection_name,
            unembedded_collection_name=unembedded_collection_name,
            name=name,
            schema=schema,
            embedder=embedder,
            embedding_cache_provider=self._embedding_cache_provider,
            version=1,
        )
        collection._database = self
        self._collections[name] = collection  # type: ignore[assignment]

        return collection  # type: ignore[return-value]

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> PineconeCollection[TDocument]:
        assert self.pinecone_client is not None, "Pinecone client must be initialized"
        assert self._index is not None, "Pinecone index must be initialized"
        assert self._embedder_factory is not None, "Embedder factory must be provided"
        assert self._embedding_cache_provider is not None, (
            "Embedding cache provider must be provided"
        )
        if collection := self._collections.get(name):
            return cast(PineconeCollection[TDocument], collection)

        self._embedder_factory.create_embedder(embedder_type)
        self.format_collection_name(name, embedder_type)

        # Check if collection exists by querying for any documents
        # We'll use get_or_create_collection which handles this better
        return await self.get_or_create_collection(
            name=name,
            schema=schema,
            embedder_type=embedder_type,
            document_loader=document_loader,
        )

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> PineconeCollection[TDocument]:
        assert self.pinecone_client is not None, "Pinecone client must be initialized"
        assert self._index is not None, "Pinecone index must be initialized"
        assert self._embedder_factory is not None, "Embedder factory must be provided"
        assert self._embedding_cache_provider is not None, (
            "Embedding cache provider must be provided"
        )
        if collection := self._collections.get(name):
            return cast(PineconeCollection[TDocument], collection)

        embedder = self._embedder_factory.create_embedder(embedder_type)
        embedded_collection_name = self.format_collection_name(name, embedder_type)
        unembedded_collection_name = f"{name}_unembedded"

        # Collections are stored in the single index using metadata
        # No need to create separate indexes - just create the collection object
        collection = PineconeCollection(
            self._logger,
            pinecone_index=self._index,
            embedded_collection_name=embedded_collection_name,
            unembedded_collection_name=unembedded_collection_name,
            name=name,
            schema=schema,
            embedder=embedder,
            embedding_cache_provider=self._embedding_cache_provider,
            version=1,
        )
        collection._database = self
        self._collections[name] = collection  # type: ignore[assignment]

        return collection  # type: ignore[return-value]

    @override
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        assert self.pinecone_client is not None, "Pinecone client must be initialized"
        assert self._index is not None, "Pinecone index must be initialized"
        if name not in self._collections:
            raise ValueError(f'Collection "{name}" not found.')

        collection = self._collections[name]
        embedder_type_name = type(collection._embedder).__name__

        # Delete all vectors for this collection using metadata filter
        # Query all vectors for this collection
        collection_filter = _build_collection_filter(
            collection_name=name,
            collection_type="embedded",  # Get embedded vectors
            embedder_type_name=embedder_type_name,
        )

        try:
            # Get all vectors for this collection
            results = await asyncio.to_thread(
                self._index.query,
                vector=[0.0] * collection._embedder.dimensions,
                top_k=10000,
                filter=collection_filter,
                include_metadata=True,
            )

            # Also get unembedded vectors
            unembedded_filter = _build_collection_filter(
                collection_name=name,
                collection_type="unembedded",
                embedder_type_name=embedder_type_name,
            )
            unembedded_results = await asyncio.to_thread(
                self._index.query,
                vector=[0.0] * collection._embedder.dimensions,
                top_k=10000,
                filter=unembedded_filter,
                include_metadata=True,
            )

            # Collect all vector IDs to delete
            vector_ids = [match.id for match in results.matches]
            vector_ids.extend([match.id for match in unembedded_results.matches])

            if vector_ids:
                await asyncio.to_thread(self._index.delete, ids=vector_ids)
        except Exception as e:
            self._logger.warning(f"Error deleting collection vectors: {e}")

        del self._collections[name]

    @override
    async def upsert_metadata(
        self,
        key: str,
        value: JSONSerializable,
    ) -> None:
        assert self.pinecone_client is not None, "Pinecone client must be initialized"
        assert self._index is not None, "Pinecone index must be initialized"

        # Metadata is stored in the same index with special collection name
        metadata_collection_name = "__metadata__"
        metadata_vector_id = "__metadata__"

        # Query for existing metadata
        metadata_filter = _build_collection_filter(
            collection_name=metadata_collection_name,
            collection_type="metadata",
            embedder_type_name="metadata",
        )

        try:
            results = await asyncio.to_thread(
                self._index.query,
                vector=[0.0] * self._index_dimension,
                top_k=1,
                filter=metadata_filter,
                include_metadata=True,
            )
        except Exception:
            results = None

        if results and results.matches:
            # Remove internal Parlant metadata fields
            document = {
                k: v
                for k, v in results.matches[0].metadata.items()
                if not k.startswith("_parlant_")
            }
            document[key] = value

            # Add back Parlant metadata
            metadata_doc = dict(document)
            metadata_doc["_parlant_collection"] = metadata_collection_name
            metadata_doc["_parlant_type"] = "metadata"
            metadata_doc["_parlant_embedder"] = "metadata"

            await asyncio.to_thread(
                self._index.upsert,
                vectors=[
                    {
                        "id": metadata_vector_id,
                        "values": [0.0] * self._index_dimension,
                        "metadata": cast(dict[str, Any], metadata_doc),
                    }
                ],
            )
        else:
            document = {key: value}

            # Add Parlant metadata
            metadata_doc = dict(document)
            metadata_doc["_parlant_collection"] = metadata_collection_name
            metadata_doc["_parlant_type"] = "metadata"
            metadata_doc["_parlant_embedder"] = "metadata"

            await asyncio.to_thread(
                self._index.upsert,
                vectors=[
                    {
                        "id": metadata_vector_id,
                        "values": [0.0] * self._index_dimension,
                        "metadata": cast(dict[str, Any], metadata_doc),
                    }
                ],
            )

    @override
    async def remove_metadata(
        self,
        key: str,
    ) -> None:
        assert self.pinecone_client is not None, "Pinecone client must be initialized"
        assert self._index is not None, "Pinecone index must be initialized"

        metadata_collection_name = "__metadata__"
        metadata_vector_id = "__metadata__"

        # Query for existing metadata
        metadata_filter = _build_collection_filter(
            collection_name=metadata_collection_name,
            collection_type="metadata",
            embedder_type_name="metadata",
        )

        try:
            results = await asyncio.to_thread(
                self._index.query,
                vector=[0.0] * self._index_dimension,
                top_k=1,
                filter=metadata_filter,
                include_metadata=True,
            )
        except Exception:
            results = None

        if results and results.matches:
            # Remove internal Parlant metadata fields
            document = {
                k: v
                for k, v in results.matches[0].metadata.items()
                if not k.startswith("_parlant_")
            }
            if key in document:
                document.pop(key)

                # Add back Parlant metadata
                metadata_doc = dict(document)
                metadata_doc["_parlant_collection"] = metadata_collection_name
                metadata_doc["_parlant_type"] = "metadata"
                metadata_doc["_parlant_embedder"] = "metadata"

                await asyncio.to_thread(
                    self._index.upsert,
                    vectors=[
                        {
                            "id": metadata_vector_id,
                            "values": [0.0] * self._index_dimension,
                            "metadata": cast(dict[str, Any], metadata_doc),
                        }
                    ],
                )
            else:
                raise ValueError(f'Metadata with key "{key}" not found.')
        else:
            raise ValueError(f'Metadata with key "{key}" not found.')

    @override
    async def read_metadata(
        self,
    ) -> dict[str, JSONSerializable]:
        assert self.pinecone_client is not None, "Pinecone client must be initialized"
        assert self._index is not None, "Pinecone index must be initialized"

        metadata_collection_name = "__metadata__"

        # Query for existing metadata
        metadata_filter = _build_collection_filter(
            collection_name=metadata_collection_name,
            collection_type="metadata",
            embedder_type_name="metadata",
        )

        try:
            results = await asyncio.to_thread(
                self._index.query,
                vector=[0.0] * self._index_dimension,
                top_k=1,
                filter=metadata_filter,
                include_metadata=True,
            )
        except Exception:
            results = None

        if results and results.matches:
            # Remove internal Parlant metadata fields
            document = {
                k: v
                for k, v in results.matches[0].metadata.items()
                if not k.startswith("_parlant_")
            }
            return cast(dict[str, JSONSerializable], document)
        else:
            return {}


class PineconeCollection(Generic[TDocument], VectorCollection[TDocument]):
    def __init__(
        self,
        logger: Logger,
        pinecone_index: Any,  # Pinecone Index object
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
        self._pinecone_index = pinecone_index
        self._embedder_type_name = type(embedder).__name__
        self._database: Optional[PineconeDatabase] = (
            None  # Reference to parent database for version methods
        )

    @override
    async def find(
        self,
        filters: Where,
    ) -> Sequence[TDocument]:
        async with self._lock.reader_lock:
            pinecone_filter = _convert_where_to_pinecone_filter(filters)

            # Build collection filter with metadata
            collection_filter = _build_collection_filter(
                collection_name=self._name,
                collection_type="embedded",
                embedder_type_name=self._embedder_type_name,
                additional_filter=pinecone_filter,
            )

            try:
                # Query with dummy vector to fetch all, then filter
                results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=[0.0] * self._embedder.dimensions,
                    top_k=10000,
                    filter=collection_filter,
                    include_metadata=True,
                )

                # Extract document metadata (remove Parlant internal fields)
                documents = []
                for match in results.matches:
                    if match.metadata is not None:
                        # Remove internal Parlant metadata fields
                        doc_metadata = {
                            k: v for k, v in match.metadata.items() if not k.startswith("_parlant_")
                        }
                        if filters:
                            from parlant.core.persistence.common import matches_filters

                            if matches_filters(filters, doc_metadata):
                                documents.append(cast(TDocument, doc_metadata))
                        else:
                            documents.append(cast(TDocument, doc_metadata))

                return documents
            except Exception:
                # If filter fails, query all and filter in memory
                all_results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=[0.0] * self._embedder.dimensions,
                    top_k=10000,
                    include_metadata=True,
                )
                # Filter in memory
                from parlant.core.persistence.common import matches_filters

                documents = []
                for match in all_results.matches:
                    if match.metadata is not None:
                        # Check collection metadata
                        if (
                            match.metadata.get("_parlant_collection") == self._name
                            and match.metadata.get("_parlant_type") == "embedded"
                            and match.metadata.get("_parlant_embedder") == self._embedder_type_name
                        ):
                            # Remove internal Parlant metadata fields
                            doc_metadata = {
                                k: v
                                for k, v in match.metadata.items()
                                if not k.startswith("_parlant_")
                            }
                            if not filters or matches_filters(filters, doc_metadata):
                                documents.append(cast(TDocument, doc_metadata))

                return documents

    @override
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]:
        async with self._lock.reader_lock:
            pinecone_filter = _convert_where_to_pinecone_filter(filters)

            # Build collection filter with metadata
            collection_filter = _build_collection_filter(
                collection_name=self._name,
                collection_type="embedded",
                embedder_type_name=self._embedder_type_name,
                additional_filter=pinecone_filter,
            )

            try:
                results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=[0.0] * self._embedder.dimensions,
                    top_k=1,
                    filter=collection_filter,
                    include_metadata=True,
                )
            except Exception:
                # If filter fails, query all and filter in memory
                all_results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=[0.0] * self._embedder.dimensions,
                    top_k=10000,
                    include_metadata=True,
                )
                # Filter in memory
                from parlant.core.persistence.common import matches_filters

                results_matches = []
                for match in all_results.matches:
                    if match.metadata is not None:
                        # Check collection metadata
                        if (
                            match.metadata.get("_parlant_collection") == self._name
                            and match.metadata.get("_parlant_type") == "embedded"
                            and match.metadata.get("_parlant_embedder") == self._embedder_type_name
                        ):
                            # Remove internal Parlant metadata fields
                            doc_metadata = {
                                k: v
                                for k, v in match.metadata.items()
                                if not k.startswith("_parlant_")
                            }
                            if not filters or matches_filters(filters, doc_metadata):
                                results_matches.append(match)
                                break  # Only need one

                # Create a mock results object
                class MockResults:
                    matches = results_matches[:1]

                results = MockResults()

            if results.matches:
                match = results.matches[0]
                if match.metadata is not None:
                    # Remove internal Parlant metadata fields
                    doc_metadata = {
                        k: v for k, v in match.metadata.items() if not k.startswith("_parlant_")
                    }
                    return cast(TDocument, doc_metadata)

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

        if not embeddings or len(embeddings[0]) == 0:
            raise ValueError(
                f"Empty embedding generated for document content: {document['content'][:50]}..."
            )

        async with self._lock.writer_lock:
            self._version += 1

            # Create vector IDs with collection prefix
            doc_id = str(document["id"])
            unembedded_vector_id = _make_vector_id(self._name, "unembedded", doc_id)
            embedded_vector_id = _make_vector_id(self._name, "embedded", doc_id)

            # Add Parlant internal metadata to document
            doc_metadata = dict(cast(dict[str, Any], document))
            doc_metadata["_parlant_collection"] = self._name
            doc_metadata["_parlant_embedder"] = self._embedder_type_name

            # Insert into unembedded collection with retry on timeout
            unembedded_metadata = {**doc_metadata, "_parlant_type": "unembedded"}
            await _retry_on_timeout_async(
                lambda: asyncio.to_thread(
                    self._pinecone_index.upsert,
                    vectors=[
                        {
                            "id": unembedded_vector_id,
                            "values": [0.0] * self._embedder.dimensions,
                            "metadata": unembedded_metadata,
                        }
                    ],
                ),
                max_retries=3,
                logger=self._logger,
            )

            # Insert into embedded collection with retry on timeout
            embedded_metadata = {**doc_metadata, "_parlant_type": "embedded"}
            await _retry_on_timeout_async(
                lambda: asyncio.to_thread(
                    self._pinecone_index.upsert,
                    vectors=[
                        {
                            "id": embedded_vector_id,
                            "values": embeddings[0],
                            "metadata": embedded_metadata,
                        }
                    ],
                ),
                max_retries=3,
                logger=self._logger,
            )

            # Update version in both collections
            if self._database:
                await self._database._set_collection_version(
                    self._unembedded_collection_name, self._version
                )
                await self._database._set_collection_version(
                    self.embedded_collection_name, self._version
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
            pinecone_filter = _convert_where_to_pinecone_filter(filters)

            # Build collection filter with metadata
            collection_filter = _build_collection_filter(
                collection_name=self._name,
                collection_type="embedded",
                embedder_type_name=self._embedder_type_name,
                additional_filter=pinecone_filter,
            )

            try:
                results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=[0.0] * self._embedder.dimensions,
                    top_k=1,
                    filter=collection_filter,
                    include_metadata=True,
                )
            except Exception:
                # If filter fails, query all and filter in memory
                all_results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=[0.0] * self._embedder.dimensions,
                    top_k=10000,
                    include_metadata=True,
                )
                # Filter in memory
                from parlant.core.persistence.common import matches_filters

                results_matches = []
                for match in all_results.matches:
                    if match.metadata is not None:
                        # Check collection metadata
                        if (
                            match.metadata.get("_parlant_collection") == self._name
                            and match.metadata.get("_parlant_type") == "embedded"
                            and match.metadata.get("_parlant_embedder") == self._embedder_type_name
                        ):
                            # Remove internal Parlant metadata fields
                            doc_metadata = {
                                k: v
                                for k, v in match.metadata.items()
                                if not k.startswith("_parlant_")
                            }
                            if not filters or matches_filters(filters, doc_metadata):
                                results_matches.append(match)
                                break  # Only need one

                # Create a mock results object
                class MockResults:
                    matches = results_matches[:1]

                results = MockResults()

            if results.matches:
                match = results.matches[0]
                # Remove internal Parlant metadata fields to get document
                doc_metadata = {
                    k: v for k, v in match.metadata.items() if not k.startswith("_parlant_")
                }
                doc = cast(dict[str, Any], doc_metadata)

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

                if not embeddings or len(embeddings[0]) == 0:
                    raise ValueError(f"Empty embedding generated for content: {content[:50]}...")

                updated_document = {**doc, **params}

                self._version += 1

                # Extract document ID from vector ID or metadata
                doc_id = str(
                    updated_document.get(
                        "id", match.id.split("_")[-1] if "_" in match.id else match.id
                    )
                )
                unembedded_vector_id = _make_vector_id(self._name, "unembedded", doc_id)
                embedded_vector_id = _make_vector_id(self._name, "embedded", doc_id)

                # Add Parlant internal metadata
                updated_metadata = dict(updated_document)
                updated_metadata["_parlant_collection"] = self._name
                updated_metadata["_parlant_embedder"] = self._embedder_type_name

                # Update unembedded collection with retry on timeout
                unembedded_metadata = {**updated_metadata, "_parlant_type": "unembedded"}
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        self._pinecone_index.upsert,
                        vectors=[
                            {
                                "id": unembedded_vector_id,
                                "values": [0.0] * self._embedder.dimensions,
                                "metadata": unembedded_metadata,
                            }
                        ],
                    ),
                    max_retries=3,
                    logger=self._logger,
                )

                # Update embedded collection with retry on timeout
                embedded_metadata = {**updated_metadata, "_parlant_type": "embedded"}
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        self._pinecone_index.upsert,
                        vectors=[
                            {
                                "id": embedded_vector_id,
                                "values": embeddings[0],
                                "metadata": embedded_metadata,
                            }
                        ],
                    ),
                    max_retries=3,
                    logger=self._logger,
                )

                # Update version in both collections
                if self._database:
                    await self._database._set_collection_version(
                        self._unembedded_collection_name, self._version
                    )
                    await self._database._set_collection_version(
                        self.embedded_collection_name, self._version
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

                if not embeddings or len(embeddings[0]) == 0:
                    raise ValueError(
                        f"Empty embedding generated for content: {params['content'][:50] if 'content' in params else 'N/A'}..."
                    )

                self._version += 1

                # Create vector IDs with collection prefix
                doc_id = str(params["id"])
                unembedded_vector_id = _make_vector_id(self._name, "unembedded", doc_id)
                embedded_vector_id = _make_vector_id(self._name, "embedded", doc_id)

                # Add Parlant internal metadata
                params_metadata = dict(cast(dict[str, Any], params))
                params_metadata["_parlant_collection"] = self._name
                params_metadata["_parlant_embedder"] = self._embedder_type_name

                # Insert into unembedded collection with retry on timeout
                unembedded_metadata = {**params_metadata, "_parlant_type": "unembedded"}
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        self._pinecone_index.upsert,
                        vectors=[
                            {
                                "id": unembedded_vector_id,
                                "values": [0.0] * self._embedder.dimensions,
                                "metadata": unembedded_metadata,
                            }
                        ],
                    ),
                    max_retries=3,
                    logger=self._logger,
                )

                # Insert into embedded collection with retry on timeout
                embedded_metadata = {**params_metadata, "_parlant_type": "embedded"}
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        self._pinecone_index.upsert,
                        vectors=[
                            {
                                "id": embedded_vector_id,
                                "values": embeddings[0],
                                "metadata": embedded_metadata,
                            }
                        ],
                    ),
                    max_retries=3,
                    logger=self._logger,
                )

                # Update version in both collections
                if self._database:
                    await self._database._set_collection_version(
                        self._unembedded_collection_name, self._version
                    )
                    await self._database._set_collection_version(
                        self.embedded_collection_name, self._version
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
            pinecone_filter = _convert_where_to_pinecone_filter(filters)

            # Build collection filter with metadata
            collection_filter = _build_collection_filter(
                collection_name=self._name,
                collection_type="embedded",
                embedder_type_name=self._embedder_type_name,
                additional_filter=pinecone_filter,
            )

            try:
                results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=[0.0] * self._embedder.dimensions,
                    top_k=2,  # Check for more than one
                    filter=collection_filter,
                    include_metadata=True,
                )
            except Exception:
                # If filter fails, query all and filter in memory
                all_results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=[0.0] * self._embedder.dimensions,
                    top_k=10000,
                    include_metadata=True,
                )
                # Filter in memory
                from parlant.core.persistence.common import matches_filters

                results_matches = []
                for match in all_results.matches:
                    if match.metadata is not None:
                        # Check collection metadata
                        if (
                            match.metadata.get("_parlant_collection") == self._name
                            and match.metadata.get("_parlant_type") == "embedded"
                            and match.metadata.get("_parlant_embedder") == self._embedder_type_name
                        ):
                            # Remove internal Parlant metadata fields
                            doc_metadata = {
                                k: v
                                for k, v in match.metadata.items()
                                if not k.startswith("_parlant_")
                            }
                            if not filters or matches_filters(filters, doc_metadata):
                                results_matches.append(match)

                results_matches = results_matches[:2]

                # Create a mock results object
                class MockResults:
                    matches = results_matches

                results = MockResults()

            if len(results.matches) > 1:
                raise ValueError(
                    f"PineconeCollection delete_one: detected more than one document with filters '{filters}'. Aborting..."
                )

            if results.matches:
                match = results.matches[0]
                # Remove internal Parlant metadata fields to get document
                doc_metadata = {
                    k: v for k, v in match.metadata.items() if not k.startswith("_parlant_")
                }
                deleted_document = cast(TDocument, doc_metadata)

                # Extract document ID from vector ID or metadata
                doc_id = str(
                    deleted_document.get(
                        "id", match.id.split("_")[-1] if "_" in match.id else match.id
                    )
                )
                unembedded_vector_id = _make_vector_id(self._name, "unembedded", doc_id)
                embedded_vector_id = _make_vector_id(self._name, "embedded", doc_id)

                self._version += 1

                # Delete from unembedded and embedded collections
                await asyncio.to_thread(
                    self._pinecone_index.delete, ids=[unembedded_vector_id, embedded_vector_id]
                )

                # Update version in both collections
                if self._database:
                    await self._database._set_collection_version(
                        self._unembedded_collection_name, self._version
                    )
                    await self._database._set_collection_version(
                        self.embedded_collection_name, self._version
                    )

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
            pinecone_filter = _convert_where_to_pinecone_filter(filters)

            if not query_embeddings or len(query_embeddings[0]) == 0:
                self._logger.warning(f"Empty embedding generated for query: {query}")
                return []

            # Build collection filter with metadata
            collection_filter = _build_collection_filter(
                collection_name=self._name,
                collection_type="embedded",
                embedder_type_name=self._embedder_type_name,
                additional_filter=pinecone_filter,
            )

            try:
                search_results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=list(query_embeddings[0]),
                    top_k=k,
                    filter=collection_filter,
                    include_metadata=True,
                )
            except Exception:
                # If filter fails, query without filter and filter in memory
                all_results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=list(query_embeddings[0]),
                    top_k=k * 2,  # Get more to filter
                    include_metadata=True,
                )
                # Filter in memory
                from parlant.core.persistence.common import matches_filters

                search_results_matches = []
                for match in all_results.matches:
                    if match.metadata is not None:
                        # Check collection metadata
                        if (
                            match.metadata.get("_parlant_collection") == self._name
                            and match.metadata.get("_parlant_type") == "embedded"
                            and match.metadata.get("_parlant_embedder") == self._embedder_type_name
                        ):
                            # Remove internal Parlant metadata fields
                            doc_metadata = {
                                k: v
                                for k, v in match.metadata.items()
                                if not k.startswith("_parlant_")
                            }
                            if not filters or matches_filters(filters, doc_metadata):
                                search_results_matches.append(match)

                search_results_matches = search_results_matches[:k]

                # Create a mock results object
                class MockResults:
                    matches = search_results_matches

                search_results = MockResults()

            if not search_results.matches:
                return []

            self._logger.trace(
                f"Similar documents found\n{json.dumps([r.metadata for r in search_results.matches], indent=2)}"
            )

            results = []
            for result in search_results.matches:
                if result.metadata is not None:
                    # Remove internal Parlant metadata fields
                    doc_metadata = {
                        k: v for k, v in result.metadata.items() if not k.startswith("_parlant_")
                    }
                    results.append(
                        SimilarDocumentResult(
                            document=cast(TDocument, doc_metadata),
                            distance=1.0 - result.score,  # Convert similarity to distance
                        )
                    )

            return results
