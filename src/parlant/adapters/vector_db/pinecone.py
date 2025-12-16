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
import asyncio
import json
from typing import Any, Awaitable, Callable, Generic, Optional, Sequence, TypeVar, cast
from typing_extensions import override, Self
from pinecone import Pinecone, ServerlessSpec  # type: ignore[import-untyped]
from pinecone.exceptions import PineconeException  # type: ignore[import-untyped]
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


class _EmptyQueryResults:
    """Empty query results for fallback when Pinecone query fails."""

    matches: list[Any] = []


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


def _make_namespace(
    collection_name: str,
    embedder_type_name: str,
    collection_type: str,
) -> str:
    """Create a Pinecone namespace for a collection.

    Pinecone uses namespaces to separate collections within a single index.
    Format: {collection_name}_{embedder_type}_{embedded|unembedded}
    """
    return f"{collection_name}_{embedder_type_name}_{collection_type}"


def _make_vector_id(doc_id: str) -> str:
    """Create a vector ID from document ID.

    With namespaces, we don't need to include collection info in the ID.
    """
    return str(doc_id)


def _create_metadata_vector(dimension: int) -> list[float]:
    """
    Create a small non-zero vector for metadata storage.
    Pinecone requires at least one non-zero value in dense vectors.
    We use a small constant value (1e-6) in the first dimension to satisfy this requirement.
    """
    vector = [0.0] * dimension
    if dimension > 0:
        vector[0] = 1e-6  # Small non-zero value to satisfy Pinecone's requirement
    return vector


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
        self._index_dimension: int = 3072  # Default dimension (text-embedding-3-large), will be updated when index is created
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

        # Default to 3072 for text-embedding-3-large (most common modern embedding model)
        # We can't determine the actual embedder dimensions at initialization time
        # since different collections may use different embedders
        # The index dimension will be validated when collections are created
        default_dimension = 3072

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

        # Get the actual dimension from the existing index if available
        # This ensures we use the correct dimension even if index was created with different dimension
        try:
            # Get index info from Pinecone client
            index_info = await asyncio.to_thread(
                self.pinecone_client.describe_index, self._index_name
            )
            if hasattr(index_info, "dimension") and index_info.dimension:
                self._index_dimension = index_info.dimension
                self._logger.info(f"Detected existing index dimension: {self._index_dimension}")
            else:
                self._index_dimension = default_dimension
        except Exception as e:
            # If we can't get the dimension, use the default
            self._logger.warning(
                f"Could not determine index dimension, using default {default_dimension}: {e}"
            )
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

        # Create namespace names for Pinecone
        unembedded_namespace = _make_namespace(collection_name, embedder_type_name, "unembedded")
        _make_namespace(collection_name, embedder_type_name, "embedded")

        # Get all vectors from unembedded namespace using query
        try:
            unembedded_results = await asyncio.to_thread(
                self._index.query,
                vector=[0.0] * embedder.dimensions,  # Dummy vector for fetching all
                top_k=10000,
                namespace=unembedded_namespace,
                include_metadata=True,
            )
        except Exception:
            # Fallback: create empty results
            unembedded_results = _EmptyQueryResults()

        indexing_required = False

        if unembedded_results.matches:
            for match in unembedded_results.matches:
                # Metadata no longer needs Parlant internal fields since we use namespaces
                doc_metadata = cast(BaseDocument, match.metadata or {})
                prospective_doc = doc_metadata
                try:
                    if loaded_doc := await document_loader(prospective_doc):
                        if loaded_doc != prospective_doc:
                            # Update the unembedded namespace
                            # Use metadata vector (with small non-zero value) for unembedded vectors
                            await asyncio.to_thread(
                                self._index.upsert,
                                vectors=[
                                    {
                                        "id": match.id,
                                        "values": _create_metadata_vector(embedder.dimensions),
                                        "metadata": cast(dict[str, Any], loaded_doc),
                                    }
                                ],
                                namespace=unembedded_namespace,
                            )
                            indexing_required = True
                    else:
                        self._logger.warning(f'Failed to load document "{prospective_doc}"')
                        await asyncio.to_thread(
                            self._index.delete, ids=[match.id], namespace=unembedded_namespace
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

        # Create namespace names for Pinecone
        unembedded_namespace = _make_namespace(collection_name, embedder_type_name, "unembedded")
        embedded_namespace = _make_namespace(collection_name, embedder_type_name, "embedded")

        # Get all vectors from unembedded namespace
        try:
            unembedded_results = await asyncio.to_thread(
                self._index.query,
                vector=[0.0] * embedder.dimensions,  # Dummy vector for fetching all
                top_k=10000,
                namespace=unembedded_namespace,
                include_metadata=True,
            )
        except Exception:
            # Fallback: create empty results
            unembedded_results = _EmptyQueryResults()

        # Map by document ID (string) from metadata
        unembedded_docs_by_id = {}
        for match in unembedded_results.matches:
            if match.metadata is not None and "id" in match.metadata:
                doc_metadata = cast(dict[str, Any], match.metadata)
                doc_id = cast(str, doc_metadata["id"])
                unembedded_docs_by_id[doc_id] = (match, doc_metadata)

        # Get all vectors from embedded namespace
        try:
            embedded_results = await asyncio.to_thread(
                self._index.query,
                vector=[0.0] * embedder.dimensions,  # Dummy vector for fetching all
                top_k=10000,
                namespace=embedded_namespace,
                include_metadata=True,
            )
        except Exception:
            # Fallback: create empty results
            embedded_results = _EmptyQueryResults()

        # Map by document ID (string) from metadata
        embedded_docs_by_id = {}
        for match in embedded_results.matches:
            if match.metadata is not None and "id" in match.metadata:
                doc_metadata = cast(dict[str, Any], match.metadata)
                doc_id = cast(str, doc_metadata["id"])
                embedded_docs_by_id[doc_id] = (match, doc_metadata)

        # Remove docs from embedded namespace that no longer exist in unembedded
        # Update embeddings for changed docs
        for doc_id, (embedded_match, embedded_doc_metadata) in embedded_docs_by_id.items():
            if doc_id not in unembedded_docs_by_id:
                await asyncio.to_thread(
                    self._index.delete, ids=[embedded_match.id], namespace=embedded_namespace
                )
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

                await asyncio.to_thread(
                    self._index.upsert,
                    vectors=[
                        {
                            "id": embedded_match.id,
                            "values": vector,
                            "metadata": unembedded_doc,
                        }
                    ],
                    namespace=embedded_namespace,
                )
                unembedded_docs_by_id.pop(doc_id)

        # Add new docs from unembedded to embedded namespace
        for doc_id, (unembedded_match, unembedded_doc) in unembedded_docs_by_id.items():
            embeddings = list(
                (await embedder.embed([cast(str, unembedded_doc["content"])])).vectors
            )

            if not embeddings or len(embeddings[0]) == 0:
                self._logger.warning(f"Empty embedding for document {doc_id}, skipping")
                continue

            # Use document ID directly as vector ID (namespaces provide isolation)
            vector_id = _make_vector_id(doc_id)

            await asyncio.to_thread(
                self._index.upsert,
                vectors=[
                    {
                        "id": vector_id,
                        "values": embeddings[0],
                        "metadata": unembedded_doc,
                    }
                ],
                namespace=embedded_namespace,
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

        # Validate that embedder dimensions match index dimensions
        if embedder.dimensions != self._index_dimension:
            raise ValueError(
                f"Embedder dimensions ({embedder.dimensions}) do not match Pinecone index dimensions ({self._index_dimension}). "
                f"Please recreate the index with dimension {embedder.dimensions} or use a different embedder."
            )

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

        # Validate that embedder dimensions match index dimensions
        if embedder.dimensions != self._index_dimension:
            raise ValueError(
                f"Embedder dimensions ({embedder.dimensions}) do not match Pinecone index dimensions ({self._index_dimension}). "
                f"Please recreate the index with dimension {embedder.dimensions} or use a different embedder."
            )

        # Load and sync existing documents if collections exist
        # This ensures existing documents are loaded and synced properly
        embedded_collection_name = await self._load_collection_documents(
            embedded_collection_name=embedded_collection_name,
            unembedded_collection_name=unembedded_collection_name,
            embedder_type=embedder_type,
            document_loader=document_loader,
        )

        # Get the actual version from metadata (defaults to 1 if collection is new)
        collection_version = await self._get_collection_version(embedded_collection_name)

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
            version=collection_version,
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

        # Delete all vectors for this collection using namespaces
        # Get namespace names
        embedded_namespace = _make_namespace(name, embedder_type_name, "embedded")
        unembedded_namespace = _make_namespace(name, embedder_type_name, "unembedded")

        try:
            # Get all vectors from embedded namespace
            embedded_results = await asyncio.to_thread(
                self._index.query,
                vector=[0.0] * collection._embedder.dimensions,
                top_k=10000,
                namespace=embedded_namespace,
                include_metadata=True,
            )

            # Get all vectors from unembedded namespace
            unembedded_results = await asyncio.to_thread(
                self._index.query,
                vector=[0.0] * collection._embedder.dimensions,
                top_k=10000,
                namespace=unembedded_namespace,
                include_metadata=True,
            )

            # Collect all vector IDs to delete
            embedded_vector_ids = [match.id for match in embedded_results.matches]
            unembedded_vector_ids = [match.id for match in unembedded_results.matches]

            # Delete from both namespaces
            if embedded_vector_ids:
                await asyncio.to_thread(
                    self._index.delete, ids=embedded_vector_ids, namespace=embedded_namespace
                )
            if unembedded_vector_ids:
                await asyncio.to_thread(
                    self._index.delete, ids=unembedded_vector_ids, namespace=unembedded_namespace
                )
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

        # Metadata is stored in a special namespace
        metadata_namespace = "__metadata__"
        metadata_vector_id = "__metadata__"

        # Query for existing metadata in namespace
        try:
            results = await asyncio.to_thread(
                self._index.query,
                vector=_create_metadata_vector(self._index_dimension),
                top_k=1,
                namespace=metadata_namespace,
                include_metadata=True,
            )
        except Exception:
            results = None

        metadata_vector = _create_metadata_vector(self._index_dimension)

        if results and results.matches:
            # Get existing metadata
            document = cast(dict[str, Any], results.matches[0].metadata or {})
            document[key] = value

            await asyncio.to_thread(
                self._index.upsert,
                vectors=[
                    {
                        "id": metadata_vector_id,
                        "values": metadata_vector,
                        "metadata": document,
                    }
                ],
                namespace=metadata_namespace,
            )
        else:
            document = {key: value}

            await asyncio.to_thread(
                self._index.upsert,
                vectors=[
                    {
                        "id": metadata_vector_id,
                        "values": metadata_vector,
                        "metadata": document,
                    }
                ],
                namespace=metadata_namespace,
            )

    @override
    async def remove_metadata(
        self,
        key: str,
    ) -> None:
        assert self.pinecone_client is not None, "Pinecone client must be initialized"
        assert self._index is not None, "Pinecone index must be initialized"

        metadata_namespace = "__metadata__"
        metadata_vector_id = "__metadata__"

        # Query for existing metadata in namespace
        try:
            results = await asyncio.to_thread(
                self._index.query,
                vector=_create_metadata_vector(self._index_dimension),
                top_k=1,
                namespace=metadata_namespace,
                include_metadata=True,
            )
        except Exception:
            results = None

        if results and results.matches:
            document = cast(dict[str, Any], results.matches[0].metadata or {})
            if key in document:
                document.pop(key)

                metadata_vector = _create_metadata_vector(self._index_dimension)
                await asyncio.to_thread(
                    self._index.upsert,
                    vectors=[
                        {
                            "id": metadata_vector_id,
                            "values": metadata_vector,
                            "metadata": document,
                        }
                    ],
                    namespace=metadata_namespace,
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

        metadata_namespace = "__metadata__"

        # Query for existing metadata in namespace
        try:
            results = await asyncio.to_thread(
                self._index.query,
                vector=_create_metadata_vector(self._index_dimension),
                top_k=1,
                namespace=metadata_namespace,
                include_metadata=True,
            )
        except Exception:
            results = None

        if results and results.matches:
            document = cast(dict[str, Any], results.matches[0].metadata or {})
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

        # Compute namespace names for Pinecone (namespaces provide collection isolation)
        self._embedded_namespace = _make_namespace(name, self._embedder_type_name, "embedded")
        self._unembedded_namespace = _make_namespace(name, self._embedder_type_name, "unembedded")

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

            try:
                # Query with dummy vector to fetch all, then filter by namespace
                results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=[0.0] * self._embedder.dimensions,
                    top_k=10000,
                    filter=pinecone_filter,
                    namespace=self._embedded_namespace,
                    include_metadata=True,
                )

                # Extract document metadata
                documents = []
                for match in results.matches:
                    if match.metadata is not None:
                        doc_metadata = cast(dict[str, Any], match.metadata)
                        if filters:
                            from parlant.core.persistence.common import matches_filters

                            if matches_filters(filters, doc_metadata):
                                documents.append(cast(TDocument, doc_metadata))
                        else:
                            documents.append(cast(TDocument, doc_metadata))

                return documents
            except Exception:
                # If filter fails, query all in namespace and filter in memory
                all_results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=[0.0] * self._embedder.dimensions,
                    top_k=10000,
                    namespace=self._embedded_namespace,
                    include_metadata=True,
                )
                # Filter in memory
                from parlant.core.persistence.common import matches_filters

                documents = []
                for match in all_results.matches:
                    if match.metadata is not None:
                        doc_metadata = cast(dict[str, Any], match.metadata)
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

            try:
                results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=[0.0] * self._embedder.dimensions,
                    top_k=1,
                    filter=pinecone_filter,
                    namespace=self._embedded_namespace,
                    include_metadata=True,
                )
            except Exception:
                # If filter fails, query all in namespace and filter in memory
                all_results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=[0.0] * self._embedder.dimensions,
                    top_k=10000,
                    namespace=self._embedded_namespace,
                    include_metadata=True,
                )
                # Filter in memory
                from parlant.core.persistence.common import matches_filters

                results_matches = []
                for match in all_results.matches:
                    if match.metadata is not None:
                        doc_metadata = cast(dict[str, Any], match.metadata)
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
                    doc_metadata = cast(dict[str, Any], match.metadata)
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

            # Use document ID directly as vector ID (namespaces provide isolation)
            doc_id = str(document["id"])
            vector_id = _make_vector_id(doc_id)

            # Document metadata (no need for Parlant internal fields with namespaces)
            doc_metadata = dict(cast(dict[str, Any], document))

            # Insert into unembedded namespace with retry on timeout
            # Use metadata vector (with small non-zero value) for unembedded vectors
            # Pinecone requires at least one non-zero value in dense vectors
            await _retry_on_timeout_async(
                lambda: asyncio.to_thread(
                    self._pinecone_index.upsert,
                    vectors=[
                        {
                            "id": vector_id,
                            "values": _create_metadata_vector(self._embedder.dimensions),
                            "metadata": doc_metadata,
                        }
                    ],
                    namespace=self._unembedded_namespace,
                ),
                max_retries=3,
                logger=self._logger,
            )

            # Insert into embedded namespace with retry on timeout
            await _retry_on_timeout_async(
                lambda: asyncio.to_thread(
                    self._pinecone_index.upsert,
                    vectors=[
                        {
                            "id": vector_id,
                            "values": embeddings[0],
                            "metadata": doc_metadata,
                        }
                    ],
                    namespace=self._embedded_namespace,
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

            try:
                results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=[0.0] * self._embedder.dimensions,
                    top_k=1,
                    filter=pinecone_filter,
                    namespace=self._embedded_namespace,
                    include_metadata=True,
                )
            except Exception:
                # If filter fails, query all in namespace and filter in memory
                all_results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=[0.0] * self._embedder.dimensions,
                    top_k=10000,
                    namespace=self._embedded_namespace,
                    include_metadata=True,
                )
                # Filter in memory
                from parlant.core.persistence.common import matches_filters

                results_matches = []
                for match in all_results.matches:
                    if match.metadata is not None:
                        doc_metadata = cast(dict[str, Any], match.metadata)
                        if not filters or matches_filters(filters, doc_metadata):
                            results_matches.append(match)
                            break  # Only need one

                # Create a mock results object
                class MockResults:
                    matches = results_matches[:1]

                results = MockResults()

            if results.matches:
                match = results.matches[0]
                # Get document from metadata
                doc = cast(dict[str, Any], match.metadata or {})

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

                # Use document ID directly as vector ID (namespaces provide isolation)
                doc_id = str(updated_document.get("id", match.id))
                vector_id = _make_vector_id(doc_id)

                # Document metadata (no need for Parlant internal fields with namespaces)
                updated_metadata = dict(updated_document)

                # Update unembedded namespace with retry on timeout
                # Use metadata vector (with small non-zero value) for unembedded vectors
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        self._pinecone_index.upsert,
                        vectors=[
                            {
                                "id": vector_id,
                                "values": _create_metadata_vector(self._embedder.dimensions),
                                "metadata": updated_metadata,
                            }
                        ],
                        namespace=self._unembedded_namespace,
                    ),
                    max_retries=3,
                    logger=self._logger,
                )

                # Update embedded namespace with retry on timeout
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        self._pinecone_index.upsert,
                        vectors=[
                            {
                                "id": vector_id,
                                "values": embeddings[0],
                                "metadata": updated_metadata,
                            }
                        ],
                        namespace=self._embedded_namespace,
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

                # Use document ID directly as vector ID (namespaces provide isolation)
                doc_id = str(params["id"])
                vector_id = _make_vector_id(doc_id)

                # Document metadata (no need for Parlant internal fields with namespaces)
                params_metadata = dict(cast(dict[str, Any], params))

                # Insert into unembedded namespace with retry on timeout
                # Use metadata vector (with small non-zero value) for unembedded vectors
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        self._pinecone_index.upsert,
                        vectors=[
                            {
                                "id": vector_id,
                                "values": _create_metadata_vector(self._embedder.dimensions),
                                "metadata": params_metadata,
                            }
                        ],
                        namespace=self._unembedded_namespace,
                    ),
                    max_retries=3,
                    logger=self._logger,
                )

                # Insert into embedded namespace with retry on timeout
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        self._pinecone_index.upsert,
                        vectors=[
                            {
                                "id": vector_id,
                                "values": embeddings[0],
                                "metadata": params_metadata,
                            }
                        ],
                        namespace=self._embedded_namespace,
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

            try:
                results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=[0.0] * self._embedder.dimensions,
                    top_k=2,  # Check for more than one
                    filter=pinecone_filter,
                    namespace=self._embedded_namespace,
                    include_metadata=True,
                )
            except Exception:
                # If filter fails, query all in namespace and filter in memory
                all_results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=[0.0] * self._embedder.dimensions,
                    top_k=10000,
                    namespace=self._embedded_namespace,
                    include_metadata=True,
                )
                # Filter in memory
                from parlant.core.persistence.common import matches_filters

                results_matches = []
                for match in all_results.matches:
                    if match.metadata is not None:
                        doc_metadata = cast(dict[str, Any], match.metadata)
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
                # Get document from metadata
                deleted_document = cast(TDocument, match.metadata or {})

                # Use document ID directly as vector ID (namespaces provide isolation)
                doc_id = str(deleted_document.get("id", match.id))
                vector_id = _make_vector_id(doc_id)

                self._version += 1

                # Delete from unembedded and embedded namespaces
                await asyncio.to_thread(
                    self._pinecone_index.delete,
                    ids=[vector_id],
                    namespace=self._unembedded_namespace,
                )
                await asyncio.to_thread(
                    self._pinecone_index.delete,
                    ids=[vector_id],
                    namespace=self._embedded_namespace,
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

            try:
                search_results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=list(query_embeddings[0]),
                    top_k=k,
                    filter=pinecone_filter,
                    namespace=self._embedded_namespace,
                    include_metadata=True,
                )
            except Exception:
                # If filter fails, query without filter in namespace and filter in memory
                all_results = await asyncio.to_thread(
                    self._pinecone_index.query,
                    vector=list(query_embeddings[0]),
                    top_k=k * 2,  # Get more to filter
                    namespace=self._embedded_namespace,
                    include_metadata=True,
                )
                # Filter in memory
                from parlant.core.persistence.common import matches_filters

                search_results_matches = []
                for match in all_results.matches:
                    if match.metadata is not None:
                        doc_metadata = cast(dict[str, Any], match.metadata)
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
                    doc_metadata = cast(dict[str, Any], result.metadata)
                    results.append(
                        SimilarDocumentResult(
                            document=cast(TDocument, doc_metadata),
                            distance=1.0 - result.score,  # Convert similarity to distance
                        )
                    )

            return results
