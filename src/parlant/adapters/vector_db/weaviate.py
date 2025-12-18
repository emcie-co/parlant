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
import gc
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Generic, Optional, Sequence, TypeVar, cast
from typing_extensions import override, Self
from urllib.parse import urlparse
import weaviate  # type: ignore[import-untyped]
from weaviate.classes.init import Auth  # type: ignore[import-untyped]
from weaviate.classes.query import Filter, MetadataQuery  # type: ignore[import-untyped]
from weaviate.classes.config import Property, DataType, VectorDistances, Configure  # type: ignore[import-untyped]
from weaviate.exceptions import WeaviateBaseError, UnexpectedStatusCodeError  # type: ignore[import-untyped]

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
    """Retry an async operation on timeout errors with exponential backoff."""
    last_exception: Exception | None = None
    total_attempts = max_retries + 1

    for attempt in range(total_attempts):
        try:
            return await operation()
        except (WeaviateBaseError, Exception) as e:
            error_str = str(e).lower()
            # Check for retryable errors: timeouts, connection issues, network errors
            is_retryable = (
                "timeout" in error_str
                or "read operation timed out" in error_str
                or "readtimeout" in error_str
                or "connection" in error_str
                or "network" in error_str
                or "refused" in error_str
                or "temporarily unavailable" in error_str
                or "service unavailable" in error_str
            )

            if is_retryable and attempt < total_attempts - 1:
                delay = base_delay * (2**attempt)
                if logger:
                    logger.warning(
                        f"Weaviate operation failed (attempt {attempt + 1}/{total_attempts}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                await asyncio.sleep(delay)
                last_exception = e
                continue
            else:
                # For non-retryable errors or final attempt, raise immediately
                if logger and attempt == total_attempts - 1:
                    logger.error(f"Weaviate operation failed after {total_attempts} attempts: {e}")
                raise

    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed unexpectedly")


def _string_id_to_uuid(doc_id: str) -> str:
    """Convert a string ID to a UUID-like string for Weaviate object IDs."""
    hash_value = hashlib.sha256(doc_id.encode()).hexdigest()
    return f"{hash_value[:8]}-{hash_value[8:12]}-{hash_value[12:16]}-{hash_value[16:20]}-{hash_value[20:32]}"


def _normalize_datetime(value: Any) -> Any:
    """Normalize datetime values to ISO format strings recursively."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [_normalize_datetime(item) for item in value]
    if isinstance(value, dict):
        return {k: _normalize_datetime(v) for k, v in value.items()}
    return value


def _to_weaviate_properties(doc: dict[str, Any]) -> dict[str, Any]:
    """Convert document from API format (id) to Weaviate format (item_id)."""
    result = {}
    for key, value in doc.items():
        if key == "id":
            result["item_id"] = value
        else:
            result[key] = _normalize_datetime(value)
    return result


def _from_weaviate_properties(properties: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert document from Weaviate format (item_id) to API format (id)."""
    if properties is None:
        return None
    result = {}
    for key, value in properties.items():
        if key == "item_id":
            result["id"] = value
        else:
            result[key] = _normalize_datetime(value)
    return result


def _convert_where_to_weaviate_filter(where: Where) -> Optional[Filter]:
    """Convert a Where filter to a Weaviate Filter."""
    if not where:
        return None

    if "$and" in where:
        filters = []
        for sub_filter in where["$and"]:
            if isinstance(sub_filter, dict):
                wf = _convert_where_to_weaviate_filter(sub_filter)
                if wf:
                    filters.append(wf)
        if filters:
            return Filter.all_of(filters)
        return None

    if "$or" in where:
        filters = []
        for sub_filter in where["$or"]:
            if isinstance(sub_filter, dict):
                wf = _convert_where_to_weaviate_filter(sub_filter)
                if wf:
                    filters.append(wf)
        if filters:
            return Filter.any_of(filters)
        return None

    conditions = []
    for field_name, field_filter in where.items():
        # Convert 'id' to 'item_id' for Weaviate
        weaviate_field_name = "item_id" if field_name == "id" else field_name
        
        if isinstance(field_filter, dict):
            for operator, filter_value in field_filter.items():
                if operator == "$eq":
                    conditions.append(Filter.by_property(weaviate_field_name).equal(filter_value))
                elif operator == "$ne":
                    conditions.append(Filter.by_property(weaviate_field_name).not_equal(filter_value))
                elif operator == "$gt":
                    conditions.append(Filter.by_property(weaviate_field_name).greater_than(filter_value))
                elif operator == "$gte":
                    conditions.append(Filter.by_property(weaviate_field_name).greater_or_equal(filter_value))
                elif operator == "$lt":
                    conditions.append(Filter.by_property(weaviate_field_name).less_than(filter_value))
                elif operator == "$lte":
                    conditions.append(Filter.by_property(weaviate_field_name).less_or_equal(filter_value))
                elif operator == "$in":
                    conditions.append(Filter.by_property(weaviate_field_name).contains_any(list(filter_value)))
                elif operator == "$nin":
                    conditions.append(Filter.by_property(weaviate_field_name).not_contains_any(list(filter_value)))

    if conditions:
        if len(conditions) == 1:
            return conditions[0]
        return Filter.all_of(conditions)
    return None


def _handle_collection_exists_error(e: WeaviateBaseError) -> None:
    """Handle 'already exists' errors when creating collections."""
    error_str = str(e).lower()
    if hasattr(e, 'message'):
        error_str += ' ' + str(e.message).lower()
    if hasattr(e, 'response') and hasattr(e.response, 'json'):
        try:
            response_body = e.response.json()
            if isinstance(response_body, dict) and 'error' in response_body:
                error_str += ' ' + str(response_body['error']).lower()
        except Exception:
            pass
    if (
        "already exists" in error_str
        or "422" in error_str
        or ("class name" in error_str and "already exists" in error_str)
    ):
        return
    raise


def _find_metadata_collection_name(collection_list: list[str], preferred_name: str = "metadata") -> str:
    """Find metadata collection name case-insensitively."""
    # Try exact match first
    if preferred_name in collection_list:
        return preferred_name
    # Try case-insensitive match
    for coll_name in collection_list:
        if coll_name.lower() == preferred_name.lower():
            return coll_name
    return preferred_name  # Return preferred name if not found (will be created)


class WeaviateDatabase(VectorDatabase):
    def __init__(
        self,
        logger: Logger,
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        grpc_port: Optional[int] = None,
        embedder_factory: Optional[EmbedderFactory] = None,
        embedding_cache_provider: Optional[EmbeddingCacheProvider] = None,
    ) -> None:
        self._path = path
        self._url = url
        self._api_key = api_key
        self._grpc_port = grpc_port or 50051
        self._logger = logger
        self._embedder_factory = embedder_factory
        self._embedding_cache_provider = embedding_cache_provider
        self.weaviate_client: Optional[Any] = None
        self._collections: dict[str, Any] = {}
        self._cleanup_done: bool = False  # Flag to make cleanup idempotent

    async def __aenter__(self) -> Self:
        max_retries = 5 if sys.platform == "win32" else 1
        for attempt in range(max_retries):
            try:
                if self._url and self._api_key:
                    parsed = urlparse(self._url)
                    cluster_name = parsed.hostname or self._url
                    self._logger.info(f"Connecting to Weaviate cluster: {cluster_name}")
                    self.weaviate_client = await asyncio.to_thread(
                        weaviate.connect_to_weaviate_cloud,
                        cluster_url=self._url,
                        auth_credentials=Auth.api_key(self._api_key),
                    )
                    # Verify connection is ready
                    if not self.weaviate_client.is_ready():
                        raise ConnectionError("Weaviate client is not ready after connection")
                    self._logger.info(f"Successfully connected to Weaviate cluster: {cluster_name}")
                elif self._url:
                    parsed = urlparse(self._url)
                    host = parsed.hostname or "localhost"
                    port = parsed.port or 8080
                    self._logger.info(f"Connecting to Weaviate local instance: {host}:{port}")
                    self.weaviate_client = await asyncio.to_thread(
                        weaviate.connect_to_local,
                        host=host,
                        port=port,
                        grpc_port=self._grpc_port,
                    )
                    # Verify connection is ready
                    if not self.weaviate_client.is_ready():
                        raise ConnectionError("Weaviate client is not ready after connection")
                    self._logger.info(f"Successfully connected to Weaviate local instance: {host}:{port}")
                else:
                    raise ValueError("Either url or path must be provided")
                break
            except Exception as e:
                error_msg = str(e).lower()
                is_connection_error = (
                    "connection" in error_msg
                    or "timeout" in error_msg
                    or "refused" in error_msg
                    or "network" in error_msg
                    or "auth" in error_msg
                    or "unauthorized" in error_msg
                )
                if is_connection_error and attempt < max_retries - 1:
                    delay = 0.05 * (attempt + 1)
                    self._logger.warning(
                        f"Connection attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                self._logger.error(f"Failed to connect to Weaviate after {attempt + 1} attempts: {e}")
                raise
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        # Make cleanup idempotent - safe to call multiple times
        if not hasattr(self, '_cleanup_done'):
            self._cleanup_done = False
        
        if self._cleanup_done:
            return
        
        try:
            self._collections.clear()
            
            client = self.weaviate_client
            if client is not None:
                self.weaviate_client = None  # Clear reference immediately to prevent reuse
                # Try to close with a very short timeout
                # If it doesn't complete quickly, we'll just skip it and let GC handle it
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(client.close),
                        timeout=0.1,  # Very short timeout - if it hangs, we skip it immediately
                    )
                except asyncio.TimeoutError:
                    # Timeout - just continue, Python GC will clean up the client
                    pass
                except (AttributeError, RuntimeError, OSError) as e:
                    # AttributeError: close() method doesn't exist
                    # RuntimeError: client already closed or event loop issues
                    # OSError: connection/socket errors during close
                    # These are non-fatal - client might already be closed
                    pass
                except Exception as e:
                    # Log but don't fail - cleanup should never block
                    pass
                finally:
                    # Always clean up references immediately, even if close() failed or timed out
                    # This ensures we don't hold references that might prevent garbage collection
                    try:
                        # Try to close underlying HTTP connection pools if they exist
                        # The Weaviate client may use httpx which has connection pools
                        if hasattr(client, '_connection') and hasattr(client._connection, 'close'):
                            try:
                                client._connection.close()
                            except Exception:
                                pass
                        if hasattr(client, '_http_client') and hasattr(client._http_client, 'close'):
                            try:
                                client._http_client.close()
                            except Exception:
                                pass
                        del client
                    except Exception:
                        pass  # Ignore errors during deletion
                    # Force garbage collection to release any lingering connections/threads
                    # Multiple passes help clean up SSL connections and thread pools on Windows
                    try:
                        gc.collect()
                        gc.collect()  # Second pass for stubborn references
                        if sys.platform == "win32":
                            gc.collect()  # Third pass on Windows for SSL connections
                    except Exception:
                        pass  # Ignore GC errors
            
            self._cleanup_done = True
        except Exception as e:
            self._logger.error(f"Fatal error during WeaviateDatabase cleanup: {e}", exc_info=True)
            # Don't raise - cleanup should never fail, just log the error
            self._cleanup_done = True  # Mark as done even on error to prevent retry loops

    def format_collection_name(self, name: str, embedder_type: type[Embedder]) -> str:
        return f"{name}_{embedder_type.__name__}"

    async def _get_collection_version(self, collection_name: str) -> int:
        """Get version from metadata collection."""
        try:
            metadata = await self.read_metadata()
            return cast(int, metadata.get(f"{collection_name}_version", 1))
        except Exception:
            return 1

    async def _set_collection_version(self, collection_name: str, version: int) -> None:
        """Set version in metadata collection."""
        await self.upsert_metadata(f"{collection_name}_version", version)

    async def _load_collection_documents(
        self,
        embedded_collection_name: str,
        unembedded_collection_name: str,
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> str:
        assert self.weaviate_client is not None
        assert self._embedder_factory is not None
        failed_migrations: list[BaseDocument] = []
        embedder = self._embedder_factory.create_embedder(embedder_type)

        try:
            unembedded_collection = await _retry_on_timeout_async(
                lambda: asyncio.to_thread(
            self.weaviate_client.collections.get,
            unembedded_collection_name
                ),
                max_retries=3,
                logger=self._logger,
            )
        except Exception as e:
            self._logger.error(
                f"Failed to get unembedded collection '{unembedded_collection_name}': {e}. "
                f"Collection may not exist or connection failed."
            )
            # If collection doesn't exist, return the embedded collection name anyway
            # The collection will be created when needed
            return embedded_collection_name

        try:
            def _fetch_objects():
                return unembedded_collection.query.fetch_objects(limit=10000).objects
            
            unembedded_objects = await _retry_on_timeout_async(
                lambda: asyncio.to_thread(_fetch_objects),
                max_retries=3,
                logger=self._logger,
            )
        except Exception as e:
            self._logger.warning(
                f"Failed to fetch objects from unembedded collection '{unembedded_collection_name}': {e}. "
                f"Continuing without loading existing documents."
            )
            unembedded_objects = []

        indexing_required = False

        if unembedded_objects:
            self._logger.info(
                f"Loading {len(unembedded_objects)} documents from collection '{unembedded_collection_name}'"
            )
            for obj in unembedded_objects:
                if obj.properties is None:
                    continue
                prospective_doc = cast(BaseDocument, _from_weaviate_properties(obj.properties))
                if prospective_doc is None:
                    continue
                try:
                    if loaded_doc := await document_loader(prospective_doc):
                        if loaded_doc != prospective_doc:
                            try:
                                await _retry_on_timeout_async(
                                    lambda: asyncio.to_thread(
                                        unembedded_collection.data.update,
                                        uuid=obj.uuid,
                                        properties=_to_weaviate_properties(cast(dict[str, Any], loaded_doc)),
                                    ),
                                    max_retries=3,
                                    logger=self._logger,
                                )
                                indexing_required = True
                            except Exception as e:
                                self._logger.warning(
                                    f"Failed to update document '{prospective_doc.get('id', 'unknown')}' in unembedded collection: {e}"
                                )
                    else:
                        self._logger.warning(f'Document loader returned None for document "{prospective_doc.get("id", "unknown")}"')
                        try:
                            await asyncio.to_thread(unembedded_collection.data.delete_by_id, obj.uuid)
                        except Exception as e:
                            self._logger.warning(f"Failed to delete document with uuid {obj.uuid}: {e}")
                        failed_migrations.append(prospective_doc)
                except Exception as e:
                    self._logger.error(f"Failed to load document '{prospective_doc.get('id', 'unknown')}': {e}")
                    failed_migrations.append(prospective_doc)

            if failed_migrations:
                self._logger.info(f"Storing {len(failed_migrations)} failed migrations")
                try:
                    failed_collection = await self.get_or_create_collection(
                        "failed_migrations",
                        BaseDocument,
                        NullEmbedder,
                        identity_loader,
                    )
                    for failed_doc in failed_migrations:
                        try:
                            await failed_collection.insert_one(failed_doc)
                        except Exception as e:
                            self._logger.warning(f"Failed to store failed migration document: {e}")
                except Exception as e:
                    self._logger.warning(f"Failed to create or access failed_migrations collection: {e}")

        unembedded_version = await self._get_collection_version(unembedded_collection_name)
        embedded_version = await self._get_collection_version(embedded_collection_name)

        if indexing_required or unembedded_version != embedded_version:
            self._logger.info(
                f"Indexing collection '{embedded_collection_name}' "
                f"(indexing_required={indexing_required}, "
                f"unembedded_version={unembedded_version}, embedded_version={embedded_version})"
            )
            try:
                await self._index_collection(
                    embedded_collection_name, unembedded_collection_name, embedder
                )
            except Exception as e:
                self._logger.error(
                    f"Failed to index collection '{embedded_collection_name}': {e}. "
                    f"Collection will continue to work but may have stale data."
                )
                # Don't raise - allow collection to be used even if indexing fails

        return embedded_collection_name

    async def _index_collection(
        self,
        embedded_collection_name: str,
        unembedded_collection_name: str,
        embedder: Embedder,
    ) -> None:
        assert self.weaviate_client is not None

        try:
            unembedded_collection = await _retry_on_timeout_async(
                lambda: asyncio.to_thread(
            self.weaviate_client.collections.get,
            unembedded_collection_name
                ),
                max_retries=3,
                logger=self._logger,
            )
        except Exception as e:
            self._logger.error(f"Failed to get unembedded collection '{unembedded_collection_name}' for indexing: {e}")
            raise

        try:
            def _fetch_unembedded_objects():
                return unembedded_collection.query.fetch_objects(limit=10000).objects
            
            unembedded_objects = await _retry_on_timeout_async(
                lambda: asyncio.to_thread(_fetch_unembedded_objects),
                max_retries=3,
                logger=self._logger,
            )
        except Exception as e:
            self._logger.error(f"Failed to fetch objects from unembedded collection for indexing: {e}")
            raise

        unembedded_docs_by_id = {
            cast(str, obj.properties["item_id"]): obj
            for obj in unembedded_objects
            if obj.properties is not None and "item_id" in obj.properties
        }

        try:
            embedded_collection = await _retry_on_timeout_async(
                lambda: asyncio.to_thread(
            self.weaviate_client.collections.get,
            embedded_collection_name
                ),
                max_retries=3,
                logger=self._logger,
            )
        except Exception as e:
            self._logger.error(f"Failed to get embedded collection '{embedded_collection_name}' for indexing: {e}")
            raise

        try:
            def _fetch_embedded_objects():
                return embedded_collection.query.fetch_objects(limit=10000).objects
            
            embedded_objects = await _retry_on_timeout_async(
                lambda: asyncio.to_thread(_fetch_embedded_objects),
                max_retries=3,
                logger=self._logger,
            )
        except Exception as e:
            self._logger.error(f"Failed to fetch objects from embedded collection for indexing: {e}")
            raise

        embedded_docs_by_id = {
            cast(str, obj.properties["item_id"]): obj
            for obj in embedded_objects
            if obj.properties is not None and "item_id" in obj.properties
        }

        async def _get_embedding_vector(
            doc_id: str,
            content: str,
            embedded_obj: Any,
            force_recompute: bool = False,
        ) -> Optional[list[float]]:
            """Get embedding vector for a document, reusing existing if valid."""
            if not force_recompute:
                # Try to reuse existing vector if available
                if embedded_obj.vector and isinstance(embedded_obj.vector, dict):
                    default_vector = embedded_obj.vector.get("default")
                    if isinstance(default_vector, (list, tuple)) and len(default_vector) > 0:
                        if isinstance(default_vector[0], (int, float)):
                            return cast(list[float], list(default_vector))
            
            # Generate new embedding
            try:
                embeddings = list((await embedder.embed([content])).vectors)
                if not embeddings or len(embeddings[0]) == 0:
                    self._logger.warning(f"Empty embedding for document {doc_id}, skipping sync")
                    return None
                return embeddings[0]
            except Exception as e:
                self._logger.error(f"Failed to generate embedding for document {doc_id}: {e}")
                return None

        for doc_id, embedded_obj in embedded_docs_by_id.items():
            if doc_id not in unembedded_docs_by_id:
                await asyncio.to_thread(embedded_collection.data.delete_by_id, embedded_obj.uuid)
            else:
                unembedded_obj = unembedded_docs_by_id[doc_id]
                unembedded_doc = _from_weaviate_properties(unembedded_obj.properties)
                if unembedded_doc is None:
                    continue

                # Check if we need to recompute embedding (checksum changed)
                embedded_props = embedded_obj.properties
                force_recompute = (
                    embedded_props is None
                    or embedded_props.get("checksum") != unembedded_doc.get("checksum")
                )

                vector = await _get_embedding_vector(
                    doc_id=doc_id,
                    content=cast(str, unembedded_doc["content"]),
                    embedded_obj=embedded_obj,
                    force_recompute=force_recompute,
                )
                
                if vector is None:
                        continue

                normalized_doc = _to_weaviate_properties(unembedded_doc)
                await asyncio.to_thread(
                    embedded_collection.data.update,
                    uuid=embedded_obj.uuid,
                    properties=normalized_doc,
                    vector=vector,
                )
            unembedded_docs_by_id.pop(doc_id)

        for doc_id, unembedded_obj in unembedded_docs_by_id.items():
            if unembedded_obj.properties is None:
                continue
            doc_dict = _from_weaviate_properties(unembedded_obj.properties)
            if doc_dict is None:
                continue
            
            # Generate embedding for new document
            try:
                embeddings = list((await embedder.embed([cast(str, doc_dict["content"])])).vectors)
                if not embeddings or len(embeddings[0]) == 0:
                    self._logger.warning(f"Empty embedding for document {doc_id}, skipping")
                    continue
                vector = embeddings[0]
            except Exception as e:
                self._logger.error(f"Failed to generate embedding for document {doc_id}: {e}")
                continue

            object_uuid = _string_id_to_uuid(str(doc_id))
            normalized_doc = _to_weaviate_properties(doc_dict)
            await asyncio.to_thread(
                embedded_collection.data.insert,
                uuid=object_uuid,
                properties=normalized_doc,
                vector=vector,
            )

        unembedded_version = await self._get_collection_version(unembedded_collection_name)
        await self._set_collection_version(unembedded_collection_name, unembedded_version)
        await self._set_collection_version(embedded_collection_name, unembedded_version)
    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
    ) -> Any:
        assert self.weaviate_client is not None
        assert self._embedder_factory is not None
        assert self._embedding_cache_provider is not None
        if name in self._collections:
            raise ValueError(f'Collection "{name}" already exists.')

        embedder = self._embedder_factory.create_embedder(embedder_type)
        embedded_collection_name = self.format_collection_name(name, embedder_type)
        unembedded_collection_name = f"{name}_unembedded"

        def _create_embedded() -> None:
            try:
                self.weaviate_client.collections.create(
                    name=embedded_collection_name,
                    vector_config=Configure.Vectors.self_provided(
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE,
                        ),
                    ),
                    properties=[
                        Property(name="item_id", data_type=DataType.TEXT),
                        Property(name="version", data_type=DataType.TEXT),
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="checksum", data_type=DataType.TEXT),
                    ],
                )
            except WeaviateBaseError as e:
                _handle_collection_exists_error(e)

        def _create_unembedded() -> None:
            try:
                self.weaviate_client.collections.create(
                    name=unembedded_collection_name,
                    vector_config=Configure.Vectors.self_provided(
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE,
                        ),
                    ),
                    properties=[
                        Property(name="item_id", data_type=DataType.TEXT),
                        Property(name="version", data_type=DataType.TEXT),
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="checksum", data_type=DataType.TEXT),
                    ],
                )
            except WeaviateBaseError as e:
                _handle_collection_exists_error(e)

        await asyncio.to_thread(_create_embedded)
        await asyncio.to_thread(_create_unembedded)

        collection = WeaviateCollection(
            self._logger,
            weaviate_client=self.weaviate_client,
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
        return collection

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> Any:
        assert self.weaviate_client is not None
        assert self._embedder_factory is not None
        assert self._embedding_cache_provider is not None
        if collection := self._collections.get(name):
            return cast(Any, collection)

        unembedded_collection_name = f"{name}_unembedded"
        embedded_collection_name = self.format_collection_name(name, embedder_type)

        collections = await asyncio.to_thread(self.weaviate_client.collections.list_all)
        collection_list = list(collections)

        if unembedded_collection_name in collection_list:
            if embedded_collection_name not in collection_list:
                embedder = self._embedder_factory.create_embedder(embedder_type)
                def _create_embedded() -> None:
                    try:
                        self.weaviate_client.collections.create(
                            name=embedded_collection_name,
                            vector_config=Configure.Vectors.self_provided(
                                vector_index_config=Configure.VectorIndex.hnsw(
                                    distance_metric=VectorDistances.COSINE,
                                ),
                            ),
                            properties=[
                                Property(name="item_id", data_type=DataType.TEXT),
                                Property(name="version", data_type=DataType.TEXT),
                                Property(name="content", data_type=DataType.TEXT),
                                Property(name="checksum", data_type=DataType.TEXT),
                            ],
                        )
                    except WeaviateBaseError as e:
                        _handle_collection_exists_error(e)
                await asyncio.to_thread(_create_embedded)
                await self._index_collection(
                    embedded_collection_name=embedded_collection_name,
                    unembedded_collection_name=unembedded_collection_name,
                    embedder=embedder,
                )

            try:
                loaded_embedded_collection_name = await self._load_collection_documents(
                    embedded_collection_name=embedded_collection_name,
                    unembedded_collection_name=unembedded_collection_name,
                    embedder_type=embedder_type,
                    document_loader=document_loader,
                )
            except Exception as e:
                self._logger.error(
                    f"Error loading collection documents for '{name}': {e}. "
                    f"Collection will be created but may not have all documents loaded."
                )
                # Continue anyway - use the embedded collection name we expected
                loaded_embedded_collection_name = embedded_collection_name

            collection = WeaviateCollection(
                self._logger,
                weaviate_client=self.weaviate_client,
                embedded_collection_name=loaded_embedded_collection_name,
                unembedded_collection_name=unembedded_collection_name,
                name=name,
                schema=schema,
                embedder=self._embedder_factory.create_embedder(embedder_type),
                embedding_cache_provider=self._embedding_cache_provider,
                version=1,
            )
            collection._database = self
            self._collections[name] = collection
            return collection

        raise ValueError(f'Weaviate collection "{name}" not found.')

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> Any:
        assert self.weaviate_client is not None
        assert self._embedder_factory is not None
        assert self._embedding_cache_provider is not None
        if collection := self._collections.get(name):
            return cast(Any, collection)

        embedder = self._embedder_factory.create_embedder(embedder_type)
        embedded_collection_name = self.format_collection_name(name, embedder_type)
        unembedded_collection_name = f"{name}_unembedded"

        collections = await asyncio.to_thread(self.weaviate_client.collections.list_all)
        collection_list = list(collections)

        if unembedded_collection_name not in collection_list:
            def _create_unembedded() -> None:
                try:
                    self.weaviate_client.collections.create(
                        name=unembedded_collection_name,
                        vector_config=Configure.Vectors.self_provided(
                            vector_index_config=Configure.VectorIndex.hnsw(
                                distance_metric=VectorDistances.COSINE,
                            ),
                        ),
                        properties=[
                            Property(name="item_id", data_type=DataType.TEXT),
                            Property(name="version", data_type=DataType.TEXT),
                            Property(name="content", data_type=DataType.TEXT),
                            Property(name="checksum", data_type=DataType.TEXT),
                        ],
                    )
                except WeaviateBaseError as e:
                    _handle_collection_exists_error(e)
            await asyncio.to_thread(_create_unembedded)

        if embedded_collection_name not in collection_list:
            def _create_embedded() -> None:
                try:
                    self.weaviate_client.collections.create(
                        name=embedded_collection_name,
                        vector_config=Configure.Vectors.self_provided(
                            vector_index_config=Configure.VectorIndex.hnsw(
                                distance_metric=VectorDistances.COSINE,
                            ),
                        ),
                        properties=[
                            Property(name="item_id", data_type=DataType.TEXT),
                            Property(name="version", data_type=DataType.TEXT),
                            Property(name="content", data_type=DataType.TEXT),
                            Property(name="checksum", data_type=DataType.TEXT),
                        ],
                    )
                except WeaviateBaseError as e:
                    _handle_collection_exists_error(e)
            await asyncio.to_thread(_create_embedded)

        try:
            loaded_embedded_collection_name = await self._load_collection_documents(
                embedded_collection_name=embedded_collection_name,
                unembedded_collection_name=unembedded_collection_name,
                embedder_type=embedder_type,
                document_loader=document_loader,
            )
        except Exception as e:
            self._logger.error(
                f"Error loading collection documents for '{name}': {e}. "
                f"Collection will be created but may not have all documents loaded."
            )
            # Continue anyway - use the embedded collection name we expected
            loaded_embedded_collection_name = embedded_collection_name

        try:
            collection = WeaviateCollection(
                self._logger,
                weaviate_client=self.weaviate_client,
                embedded_collection_name=loaded_embedded_collection_name,
                unembedded_collection_name=unembedded_collection_name,
                name=name,
                schema=schema,
                embedder=embedder,
                embedding_cache_provider=self._embedding_cache_provider,
                version=1,
            )
            collection._database = self
            self._collections[name] = collection
            return collection
        except Exception as e:
            raise

    @override
    async def delete_collection(self, name: str) -> None:
        assert self.weaviate_client is not None
        if name not in self._collections:
            raise ValueError(f'Collection "{name}" not found.')

        embedded_collection_name = self.format_collection_name(
            name, type(self._collections[name]._embedder)
        )
        unembedded_collection_name = f"{name}_unembedded"

        await asyncio.to_thread(
            self.weaviate_client.collections.delete,
            embedded_collection_name
        )
        await asyncio.to_thread(
            self.weaviate_client.collections.delete,
            unembedded_collection_name
        )
        del self._collections[name]

    @override
    async def upsert_metadata(self, key: str, value: JSONSerializable) -> None:
        assert self.weaviate_client is not None
        metadata_collection_name = "metadata"

        collections = await asyncio.to_thread(self.weaviate_client.collections.list_all)
        collection_list = list(collections)

        # Find metadata collection case-insensitively (handle existing "Metadata" vs "metadata")
        existing_metadata_collection = _find_metadata_collection_name(collection_list, metadata_collection_name)
        if existing_metadata_collection.lower() == metadata_collection_name.lower() and existing_metadata_collection in collection_list:
            metadata_collection_name = existing_metadata_collection  # Use existing case
        elif metadata_collection_name not in collection_list:
            def _create_metadata() -> None:
                try:
                    self.weaviate_client.collections.create(
                        name=metadata_collection_name,
                        vector_config=Configure.Vectors.self_provided(
                            vector_index_config=Configure.VectorIndex.hnsw(
                                distance_metric=VectorDistances.COSINE,
                            ),
                        ),
                        properties=[
                            Property(name="key", data_type=DataType.TEXT),
                            Property(name="value", data_type=DataType.TEXT),
                        ],
                    )
                except WeaviateBaseError as e:
                    _handle_collection_exists_error(e)
            # Add timeout to prevent hanging when creating metadata collection
            await asyncio.wait_for(
                asyncio.to_thread(_create_metadata),
                timeout=10.0  # 10 second timeout for creating collection
            )

        try:
            # Add timeout to prevent hanging when reconnecting to existing clusters
            metadata_collection = await asyncio.wait_for(
                asyncio.to_thread(
                    self.weaviate_client.collections.get,
                    metadata_collection_name
                ),
                timeout=10.0  # 10 second timeout for getting collection
            )
            metadata_uuid = _string_id_to_uuid("__metadata__")

            try:
                # Add timeout to prevent hanging when fetching metadata
                obj = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: metadata_collection.query.fetch_object_by_id(metadata_uuid)
                    ),
                    timeout=10.0  # 10 second timeout for fetching object
                )
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
                # Timeout, cancellation, or other error - treat as object not found, will create new one
                obj = None

            if obj is not None:
                # Read existing metadata from the "value" property as JSON
                existing_metadata_str = obj.properties.get("value", "{}") if obj.properties else "{}"
                try:
                    document = json.loads(existing_metadata_str) if isinstance(existing_metadata_str, str) else (existing_metadata_str or {})
                except (json.JSONDecodeError, TypeError) as e:
                    document = {}
                document[key] = _normalize_datetime(value)
                json_value = json.dumps(document, default=str)
                # Store entire metadata dictionary as JSON string in "value" property
                # Add timeout to prevent hanging when reconnecting to existing clusters
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            metadata_collection.data.update,
                            uuid=obj.uuid,
                            properties={"key": "__metadata__", "value": json_value},
                        ),
                        timeout=10.0  # 10 second timeout for metadata update
                    )
                    # Update succeeded - return early, don't try to insert
                    return
                except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                    self._logger.warning(f"Metadata update for key '{key}' timed out or was cancelled, but continuing")
                    # Don't raise - allow startup to continue even if metadata update fails
                    return
                except UnexpectedStatusCodeError as e:
                    # Check if it's an "already exists" error (422 status code) - treat as success
                    error_str = str(e).lower()
                    if "422" in error_str and "already exists" in error_str:
                        self._logger.debug(f"Metadata for key '{key}' already exists, treating as success")
                        return
                    # For other errors, log and continue
                    self._logger.warning(f"Failed to update metadata for key '{key}': {e}")
                    return
                except Exception as e:
                    # Log but don't raise for non-critical metadata updates during startup
                    self._logger.warning(f"Failed to update metadata for key '{key}': {e}")
                    return
            else:
                document = {key: _normalize_datetime(value)}
                # Add timeout to prevent hanging when reconnecting to existing clusters
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            metadata_collection.data.insert,
                            uuid=metadata_uuid,
                            properties={"key": "__metadata__", "value": json.dumps(document, default=str)},
                            vector=[0.0],
                        ),
                        timeout=10.0  # 10 second timeout for metadata insert
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                    self._logger.warning(f"Metadata insert for key '{key}' timed out or was cancelled, but continuing")
                    # Don't raise - allow startup to continue even if metadata insert fails
                except UnexpectedStatusCodeError as e:
                    # Check if it's an "already exists" error (422 status code) - treat as success (idempotent)
                    error_str = str(e).lower()
                    if "422" in error_str and "already exists" in error_str:
                        self._logger.debug(f"Metadata for key '{key}' already exists, treating as success")
                    else:
                        # Other 422 errors should still be logged
                        self._logger.warning(f"Failed to insert metadata for key '{key}': {e}")
                except Exception as e:
                    # Log but don't raise for non-critical metadata inserts during startup
                    self._logger.warning(f"Failed to insert metadata for key '{key}': {e}")
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            # Don't raise for timeouts/cancellations - allow startup to continue
            self._logger.warning(f"Metadata upsert for key '{key}' timed out or was cancelled, but continuing")
        except Exception as e:
            # For other errors, log but don't raise to allow startup to continue
            self._logger.warning(f"Metadata upsert for key '{key}' failed: {e}")

    @override
    async def remove_metadata(self, key: str) -> None:
        assert self.weaviate_client is not None
        metadata_collection_name = "metadata"

        collections = await asyncio.to_thread(self.weaviate_client.collections.list_all)
        collection_list = list(collections)
        
        # Find metadata collection case-insensitively (handle existing "Metadata" vs "metadata")
        metadata_collection_name = _find_metadata_collection_name(collection_list, metadata_collection_name)

        if metadata_collection_name not in collection_list:
            raise ValueError(f'Metadata with key "{key}" not found.')

        metadata_collection = await asyncio.to_thread(
            self.weaviate_client.collections.get,
            metadata_collection_name
        )
        metadata_uuid = _string_id_to_uuid("__metadata__")

        obj = await asyncio.to_thread(
            lambda: metadata_collection.query.fetch_object_by_id(metadata_uuid)
        )

        if obj is not None and obj.properties:
            document = cast(dict[str, JSONSerializable], _from_weaviate_properties(obj.properties) or {})
            if key not in document:
                raise ValueError(f'Metadata with key "{key}" not found.')
            document.pop(key)
            await asyncio.to_thread(
                metadata_collection.data.update,
                uuid=obj.uuid,
                properties=_to_weaviate_properties(document),
            )
        else:
            raise ValueError(f'Metadata with key "{key}" not found.')

    @override
    async def read_metadata(self) -> dict[str, JSONSerializable]:
        assert self.weaviate_client is not None
        metadata_collection_name = "metadata"

        try:
            collections = await asyncio.to_thread(self.weaviate_client.collections.list_all)
            collection_list = list(collections)
        except Exception as e:
            return {}
        # Find metadata collection case-insensitively (handle existing "Metadata" vs "metadata")
        original_name = metadata_collection_name
        metadata_collection_name = _find_metadata_collection_name(collection_list, metadata_collection_name)

        if metadata_collection_name in collection_list:
            try:
                metadata_collection = await asyncio.to_thread(
                    self.weaviate_client.collections.get,
                    metadata_collection_name
                )
                metadata_uuid = _string_id_to_uuid("__metadata__")
                try:
                    obj = await asyncio.to_thread(
                        lambda: metadata_collection.query.fetch_object_by_id(metadata_uuid)
                    )
                    if obj and obj.properties:
                        # Read metadata from "value" property as JSON string
                        metadata_str = obj.properties.get("value", "{}")
                        try:
                            result = json.loads(metadata_str) if isinstance(metadata_str, str) else (metadata_str or {})
                            return cast(dict[str, JSONSerializable], result)
                        except (json.JSONDecodeError, TypeError):
                            return {}
                except Exception:
                    pass
            except Exception:
                pass

        return {}


class WeaviateCollection(Generic[TDocument], VectorCollection[TDocument]):
    def __init__(
        self,
        logger: Logger,
        weaviate_client: Any,
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
        self.weaviate_client = weaviate_client
        self._database: Optional[WeaviateDatabase] = None

    @override
    async def find(self, filters: Where) -> Sequence[TDocument]:
        async with self._lock.reader_lock:
            weaviate_filter = _convert_where_to_weaviate_filter(filters)
            collection = await asyncio.to_thread(
                self.weaviate_client.collections.get,
                self.embedded_collection_name
            )

            try:
                if weaviate_filter:
                    objects = await asyncio.to_thread(
                        lambda: collection.query.fetch_objects(limit=10000, filters=weaviate_filter).objects
                    )
                else:
                    objects = await asyncio.to_thread(
                        lambda: collection.query.fetch_objects(limit=10000).objects
                    )
            except WeaviateBaseError:
                # If filter fails (e.g., property not in schema), fetch all and filter in memory
                if weaviate_filter:
                    all_objects = await asyncio.to_thread(
                        lambda: collection.query.fetch_objects(limit=10000).objects
                    )
                    from parlant.core.persistence.common import matches_filters
                    objects = [
                        obj for obj in all_objects
                        if obj.properties is not None and matches_filters(filters, _from_weaviate_properties(obj.properties) or {})
                    ]
                else:
                    objects = []

            return [
                cast(TDocument, _from_weaviate_properties(obj.properties))
                for obj in objects
                if obj.properties is not None
            ]

    @override
    async def find_one(self, filters: Where) -> Optional[TDocument]:
        async with self._lock.reader_lock:
            weaviate_filter = _convert_where_to_weaviate_filter(filters)
            collection = await asyncio.to_thread(
                self.weaviate_client.collections.get,
                self.embedded_collection_name
            )

            try:
                if weaviate_filter:
                    objects = await asyncio.to_thread(
                        lambda: collection.query.fetch_objects(limit=1, filters=weaviate_filter).objects
                    )
                else:
                    objects = await asyncio.to_thread(
                        lambda: collection.query.fetch_objects(limit=1).objects
                    )
            except WeaviateBaseError:
                # If filter fails (e.g., property not in schema), fetch all and filter in memory
                if weaviate_filter:
                    all_objects = await asyncio.to_thread(
                        lambda: collection.query.fetch_objects(limit=10000).objects
                    )
                    from parlant.core.persistence.common import matches_filters
                    objects = [
                        obj for obj in all_objects
                        if obj.properties is not None and matches_filters(filters, _from_weaviate_properties(obj.properties) or {})
                    ][:1]
                else:
                    objects = []

            if objects:
                return cast(TDocument, _from_weaviate_properties(objects[0].properties))
            return None

    @override
    async def insert_one(self, document: TDocument) -> InsertResult:
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
            object_uuid = _string_id_to_uuid(str(document["id"]))

            unembedded_collection = await asyncio.to_thread(
                self.weaviate_client.collections.get,
                self._unembedded_collection_name
            )
            # Insert into unembedded collection - handle "already exists" as success (idempotent)
            try:
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        unembedded_collection.data.insert,
                        uuid=object_uuid,
                        properties=_to_weaviate_properties(cast(dict[str, Any], document)),
                        vector=[0.0],
                    ),
                    max_retries=3,
                    logger=self._logger,
                )
            except UnexpectedStatusCodeError as e:
                # Check if it's an "already exists" error (422 status code)
                error_str = str(e).lower()
                if "422" in error_str and "already exists" in error_str:
                    # Document already exists - treat as success (idempotent operation)
                    self._logger.debug(f"Document with id '{document['id']}' already exists in unembedded collection, skipping insert")
                else:
                    # Other 422 errors should still be raised
                    raise

            embedded_collection = await asyncio.to_thread(
                self.weaviate_client.collections.get,
                self.embedded_collection_name
            )
            # Insert into embedded collection - handle "already exists" as success (idempotent)
            try:
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        embedded_collection.data.insert,
                        uuid=object_uuid,
                        properties=_to_weaviate_properties(cast(dict[str, Any], document)),
                        vector=embeddings[0],
                    ),
                    max_retries=3,
                    logger=self._logger,
                )
            except UnexpectedStatusCodeError as e:
                # Check if it's an "already exists" error (422 status code)
                error_str = str(e).lower()
                if "422" in error_str and "already exists" in error_str:
                    # Document already exists - treat as success (idempotent operation)
                    self._logger.debug(f"Document with id '{document['id']}' already exists in embedded collection, skipping insert")
                else:
                    # Other 422 errors should still be raised
                    raise

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
            weaviate_filter = _convert_where_to_weaviate_filter(filters)
            collection = await asyncio.to_thread(
                self.weaviate_client.collections.get,
                self.embedded_collection_name
            )

            try:
                objects = await asyncio.to_thread(
                    lambda: collection.query.fetch_objects(limit=1, filters=weaviate_filter).objects
                )
            except WeaviateBaseError:
                # If filter fails (e.g., property not in schema), fetch all and filter in memory
                if weaviate_filter:
                    all_objects = await asyncio.to_thread(
                        lambda: collection.query.fetch_objects(limit=10000).objects
                    )
                    from parlant.core.persistence.common import matches_filters
                    objects = [
                        obj for obj in all_objects
                        if obj.properties is not None and matches_filters(filters, _from_weaviate_properties(obj.properties) or {})
                    ][:1]
                else:
                    objects = []

            if objects:
                obj = objects[0]
                doc = cast(dict[str, Any], _from_weaviate_properties(obj.properties))
                if doc is None:
                    doc = {}

                if "content" in params:
                    content = params["content"]
                else:
                    content = str(doc.get("content", ""))

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

                unembedded_collection = await asyncio.to_thread(
                    self.weaviate_client.collections.get,
                    self._unembedded_collection_name
                )
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        unembedded_collection.data.update,
                        uuid=obj.uuid,
                        properties=_to_weaviate_properties(updated_document),
                    ),
                    max_retries=3,
                    logger=self._logger,
                )

                embedded_collection = await asyncio.to_thread(
                    self.weaviate_client.collections.get,
                    self.embedded_collection_name
                )
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        embedded_collection.data.update,
                        uuid=obj.uuid,
                        properties=_to_weaviate_properties(updated_document),
                        vector=embeddings[0],
                    ),
                    max_retries=3,
                    logger=self._logger,
                )

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
                object_uuid = _string_id_to_uuid(str(params["id"]))

                unembedded_collection = await asyncio.to_thread(
                    self.weaviate_client.collections.get,
                    self._unembedded_collection_name
                )
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        unembedded_collection.data.insert,
                        uuid=object_uuid,
                        properties=_to_weaviate_properties(cast(dict[str, Any], params)),
                        vector=[0.0],
                    ),
                    max_retries=3,
                    logger=self._logger,
                )

                embedded_collection = await asyncio.to_thread(
                    self.weaviate_client.collections.get,
                    self.embedded_collection_name
                )
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        embedded_collection.data.insert,
                        uuid=object_uuid,
                        properties=_to_weaviate_properties(cast(dict[str, Any], params)),
                        vector=embeddings[0],
                    ),
                    max_retries=3,
                    logger=self._logger,
                )

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
    async def delete_one(self, filters: Where) -> DeleteResult[TDocument]:
        async with self._lock.writer_lock:
            weaviate_filter = _convert_where_to_weaviate_filter(filters)
            collection = await asyncio.to_thread(
                self.weaviate_client.collections.get,
                self.embedded_collection_name
            )

            try:
                objects = await asyncio.to_thread(
                    lambda: collection.query.fetch_objects(limit=2, filters=weaviate_filter).objects
                )
            except WeaviateBaseError:
                # If filter fails (e.g., property not in schema), fetch all and filter in memory
                if weaviate_filter:
                    all_objects = await asyncio.to_thread(
                        lambda: collection.query.fetch_objects(limit=10000).objects
                    )
                    from parlant.core.persistence.common import matches_filters
                    objects = [
                        obj for obj in all_objects
                        if obj.properties is not None and matches_filters(filters, _from_weaviate_properties(obj.properties) or {})
                    ][:2]
                else:
                    objects = []

            if len(objects) > 1:
                raise ValueError(
                    f"WeaviateCollection delete_one: detected more than one document with filters '{filters}'. Aborting..."
                )

            if objects:
                obj = objects[0]
                deleted_document = cast(TDocument, _from_weaviate_properties(obj.properties))
                self._version += 1

                unembedded_collection = await asyncio.to_thread(
                    self.weaviate_client.collections.get,
                    self._unembedded_collection_name
                )
                await asyncio.to_thread(
                    unembedded_collection.data.delete_by_id,
                    obj.uuid
                )

                embedded_collection = await asyncio.to_thread(
                    self.weaviate_client.collections.get,
                    self.embedded_collection_name
                )
                await asyncio.to_thread(
                    embedded_collection.data.delete_by_id,
                    obj.uuid
                )

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
            weaviate_filter = _convert_where_to_weaviate_filter(filters)

            if not query_embeddings or len(query_embeddings[0]) == 0:
                self._logger.warning(f"Empty embedding generated for query: {query}")
                return []

            collection = await asyncio.to_thread(
                self.weaviate_client.collections.get,
                self.embedded_collection_name
            )
            search_results = await asyncio.to_thread(
                lambda: collection.query.near_vector(
                    near_vector=query_embeddings[0],
                    limit=k,
                    filters=weaviate_filter,
                    return_metadata=MetadataQuery(distance=True),
                ).objects
            )

            if not search_results:
                return []

            serializable_results = [
                _from_weaviate_properties(r.properties) or {}
                for r in search_results
                if r.properties is not None
            ]
            self._logger.trace(
                f"Similar documents found\n{json.dumps(serializable_results, indent=2, default=str)}"
            )

            return [
                SimilarDocumentResult(
                    document=cast(TDocument, _from_weaviate_properties(result.properties)),
                    distance=1.0 - (result.metadata.distance if hasattr(result.metadata, 'distance') and result.metadata.distance is not None else 0.0),
                )
                for result in search_results
                if result.properties is not None
            ]
