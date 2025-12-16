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
from datetime import datetime
from typing import Any, Awaitable, Callable, Generic, Optional, Sequence, TypeVar, cast
from typing_extensions import override, Self
import weaviate  # type: ignore[import-untyped]
from weaviate.classes.query import Filter  # type: ignore[import-untyped]
from weaviate.classes.config import Property, DataType, VectorDistances, Configure  # type: ignore[import-untyped]
from weaviate.exceptions import WeaviateBaseError  # type: ignore[import-untyped]


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
        max_retries: Maximum number of retry attempts (total attempts will be max_retries + 1)
        base_delay: Base delay in seconds for exponential backoff
        logger: Optional logger for warning messages

    Returns:
        The result of the operation

    Raises:
        The last exception if all retries fail
    """
    last_exception: Exception | None = None
    total_attempts = max_retries + 1  # Initial attempt + retries

    for attempt in range(total_attempts):
        try:
            return await operation()
        except (WeaviateBaseError, Exception) as e:
            # Check if it's a timeout error
            error_str = str(e).lower()
            is_timeout = (
                "timeout" in error_str
                or "read operation timed out" in error_str
                or "readtimeout" in error_str
                or "connection" in error_str
            )

            # Only retry if it's a timeout and we haven't exhausted all attempts
            # attempt is 0-indexed, so attempt < total_attempts - 1 means we have more attempts left
            if is_timeout and attempt < total_attempts - 1:
                delay = base_delay * (2**attempt)  # Exponential backoff: 1s, 2s, 4s
                if logger:
                    logger.warning(
                        f"Weaviate operation timed out (attempt {attempt + 1}/{total_attempts}). "
                        f"Retrying in {delay}s..."
                    )
                await asyncio.sleep(delay)
                last_exception = e
                continue
            else:
                # Not a timeout or out of retries - raise the exception
                if is_timeout and logger:
                    logger.error(
                        f"Weaviate operation timed out on final attempt ({attempt + 1}/{total_attempts}). "
                        f"Giving up."
                    )
                raise

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed unexpectedly")


def _string_id_to_uuid(doc_id: str) -> str:
    """Convert a string ID to a UUID-like string for Weaviate object IDs."""
    # Weaviate accepts UUID strings, so we'll generate a deterministic UUID from the string ID
    # Using SHA256 to create a deterministic UUID
    hash_value = hashlib.sha256(doc_id.encode()).hexdigest()
    # Format as UUID: 8-4-4-4-12
    return f"{hash_value[:8]}-{hash_value[8:12]}-{hash_value[12:16]}-{hash_value[16:20]}-{hash_value[20:32]}"


def _handle_collection_exists_error(e: WeaviateBaseError) -> None:
    """Handle 'already exists' errors when creating collections.
    
    If the collection already exists, ignore the error. Otherwise, re-raise it.
    """
    error_str = str(e).lower()
    # Check error message attribute if available
    if hasattr(e, 'message'):
        error_msg = str(e.message).lower()
        error_str += ' ' + error_msg
    # Check response body if available (for UnexpectedStatusCodeError)
    if hasattr(e, 'response') and hasattr(e.response, 'json'):
        try:
            response_body = e.response.json()
            if isinstance(response_body, dict) and 'error' in response_body:
                error_str += ' ' + str(response_body['error']).lower()
        except Exception:
            pass
    # Check for various "already exists" indicators
    if (
        "already exists" in error_str
        or "422" in error_str
        or ("class name" in error_str and "already exists" in error_str)
    ):
        # Collection already exists, which is fine - ignore the error
        return
    # Re-raise if it's a different error
    raise


def _convert_doc_to_weaviate_properties(doc: dict[str, Any]) -> dict[str, Any]:
    """Convert document properties from API format (with 'id') to Weaviate format (with 'item_id').
    
    Weaviate reserves 'id' as a property name, so we use 'item_id' internally.
    Also converts datetime objects to ISO format strings for proper storage.
    """
    weaviate_doc = {}
    for key, value in doc.items():
        if key == "id":
            weaviate_doc["item_id"] = value
        elif isinstance(value, datetime):
            # Convert datetime objects to ISO format strings
            weaviate_doc[key] = value.isoformat()
        else:
            weaviate_doc[key] = value
    return weaviate_doc


def _normalize_datetime_value(value: Any) -> Any:
    """Normalize datetime values to ISO format strings.
    
    Handles various datetime representations that might come from Weaviate:
    - datetime objects -> ISO string
    - Already ISO strings -> keep as string
    - None -> None
    - Lists -> recursively normalize items
    - Dicts -> recursively normalize values
    - Other types -> return as-is
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        # If it's already a string, keep as-is
        return value
    if isinstance(value, list):
        # Recursively normalize list items
        return [_normalize_datetime_value(item) for item in value]
    if isinstance(value, dict):
        # Recursively normalize dict values
        return {k: _normalize_datetime_value(v) for k, v in value.items()}
    # For any other type, return as-is
    return value


def _convert_weaviate_properties_to_doc(properties: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert Weaviate properties (with 'item_id') back to API format (with 'id').
    
    Weaviate reserves 'id' as a property name, so we use 'item_id' internally.
    Also ensures datetime fields are returned as ISO format strings (not datetime objects).
    Handles cases where Weaviate might return datetime objects or other types.
    """
    if properties is None:
        return None
    doc = {}
    for key, value in properties.items():
        if key == "item_id":
            doc["id"] = value
        else:
            # Normalize datetime values for all fields
            # This handles fields like 'creation_utc' that should be ISO strings
            normalized_value = _normalize_datetime_value(value)
            doc[key] = normalized_value
    return doc


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


def _convert_field_name_to_weaviate(field_name: str) -> str:
    """Convert API field name to Weaviate field name.
    
    Weaviate reserves 'id' as a property name, so we use 'item_id' internally.
    """
    if field_name == "id":
        return "item_id"
    return field_name


def _convert_where_to_weaviate_filter(where: Where) -> Any:  # type: ignore[type-arg]
    """Convert a Where filter to a Weaviate Filter."""
    if not where:
        return None

    # Handle logical operators
    if "$and" in where:
        and_conditions: list[Any] = []  # type: ignore[type-arg]
        for sub_filter in where["$and"]:
            if isinstance(sub_filter, dict):
                weaviate_filter = _convert_where_to_weaviate_filter(sub_filter)
                if weaviate_filter:
                    and_conditions.append(weaviate_filter)
        if and_conditions:
            # Weaviate uses Filter.all_of for AND
            return Filter.all_of(and_conditions)  # type: ignore[arg-type]
        return None

    if "$or" in where:
        or_conditions: list[Any] = []  # type: ignore[type-arg]
        for sub_filter in where["$or"]:
            if isinstance(sub_filter, dict):
                weaviate_filter = _convert_where_to_weaviate_filter(sub_filter)
                if weaviate_filter:
                    or_conditions.append(weaviate_filter)
        if or_conditions:
            # Weaviate uses Filter.any_of for OR
            return Filter.any_of(or_conditions)  # type: ignore[arg-type]
        return None

    # Handle field conditions
    field_conditions: list[Any] = []  # type: ignore[type-arg]
    for field_name, field_filter in where.items():
        if isinstance(field_filter, dict):
            # Convert field name from API format (id) to Weaviate format (item_id)
            weaviate_field_name = _convert_field_name_to_weaviate(field_name)
            for operator, filter_value in field_filter.items():
                if operator == "$eq":
                    field_conditions.append(Filter.by_property(weaviate_field_name).equal(filter_value))  # type: ignore[arg-type]
                elif operator == "$ne":
                    field_conditions.append(Filter.by_property(weaviate_field_name).not_equal(filter_value))  # type: ignore[arg-type]
                elif operator == "$gt":
                    field_conditions.append(
                        Filter.by_property(weaviate_field_name).greater_than(filter_value)  # type: ignore[arg-type]
                    )
                elif operator == "$gte":
                    field_conditions.append(
                        Filter.by_property(weaviate_field_name).greater_or_equal(filter_value)  # type: ignore[arg-type]
                    )
                elif operator == "$lt":
                    field_conditions.append(Filter.by_property(weaviate_field_name).less_than(filter_value))  # type: ignore[arg-type]
                elif operator == "$lte":
                    field_conditions.append(
                        Filter.by_property(weaviate_field_name).less_or_equal(filter_value)  # type: ignore[arg-type]
                    )
                elif operator == "$in":
                    # Weaviate uses contains_any for $in
                    field_conditions.append(
                        Filter.by_property(weaviate_field_name).contains_any(list(filter_value))  # type: ignore[arg-type]
                    )
                elif operator == "$nin":
                    # Weaviate doesn't have direct $nin, use not_equal for each value
                    # For multiple values, we need to use Filter.all_of with not_equal
                    not_conditions = [
                        Filter.by_property(weaviate_field_name).not_equal(val)
                        for val in list(filter_value)  # type: ignore[arg-type]
                    ]
                    if not_conditions:
                        field_conditions.append(Filter.all_of(not_conditions))  # type: ignore[arg-type]

    if field_conditions:
        return Filter.all_of(field_conditions)  # type: ignore[arg-type]
    return None


class WeaviateDatabase(VectorDatabase):
    def __init__(
        self,
        logger: Logger,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedder_factory: Optional[EmbedderFactory] = None,
        embedding_cache_provider: Optional[EmbeddingCacheProvider] = None,
        grpc_port: Optional[int] = None,
    ) -> None:
        # Normalize URL - ensure it has a scheme
        if url:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if not parsed.scheme:
                # If no scheme, assume http://
                url = f"http://{url}"
        else:
            url = "http://localhost:8080"
        
        self._url = url
        self._api_key = api_key
        self._grpc_port = grpc_port or 50051  # Default gRPC port
        self._logger = logger
        self._embedder_factory = embedder_factory

        self.weaviate_client: Optional[weaviate.WeaviateClient] = None
        self._collections: dict[str, WeaviateCollection[BaseDocument]] = {}

        self._embedding_cache_provider = embedding_cache_provider

    async def __aenter__(self) -> Self:
        # On Windows, retry if connection fails (from previous instance)
        max_retries = 5 if sys.platform == "win32" else 1
        for attempt in range(max_retries):
            try:
                from urllib.parse import urlparse
                parsed = urlparse(self._url)
                
                if self._api_key:
                    # Cloud connection - validate URL doesn't look like localhost
                    if parsed.hostname in ("localhost", "127.0.0.1") or not parsed.hostname:
                        raise ValueError(
                            f"Invalid Weaviate Cloud URL: {self._url}. "
                            "Cloud URLs should not point to localhost when api_key is provided."
                        )
                    
                    # Extract cluster name from URL for logging
                    cluster_name = parsed.hostname or self._url
                    self._logger.info(f"Connecting to Weaviate cluster: {cluster_name}")
                    
                    self.weaviate_client = await asyncio.to_thread(
                        weaviate.connect_to_weaviate_cloud,
                        cluster_url=self._url,
                        auth_credentials=weaviate.auth.AuthApiKey(self._api_key),
                    )
                else:
                    # Local connection - parse host and port properly
                    host = parsed.hostname or "localhost"
                    port = parsed.port or 8080
                    self._logger.info(f"Connecting to Weaviate local instance: {host}:{port} (gRPC: {self._grpc_port})")
                    
                    self.weaviate_client = await asyncio.to_thread(
                        weaviate.connect_to_local,
                        host=host,
                        port=port,
                        grpc_port=self._grpc_port,
                    )
                break
            except Exception as e:
                if "connection" in str(e).lower() and attempt < max_retries - 1:
                    # Exponential backoff: 0.05s, 0.1s, 0.15s, 0.2s, 0.25s
                    delay = 0.05 * (attempt + 1)
                    await asyncio.sleep(delay)
                    continue
                raise
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        # Close collections first to release any resources
        self._collections.clear()

        # Close Weaviate client to release connections
        # Similar to Qdrant: simple, direct approach
        if self.weaviate_client is not None:
            try:
                # Close the client - wrap in to_thread since it's synchronous
                # Use a short timeout to prevent hanging, but keep it simple
                await asyncio.wait_for(
                    asyncio.to_thread(self.weaviate_client.close),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                # If close() hangs, log and continue - don't block shutdown
                self._logger.warning("Weaviate client close() timed out, forcing cleanup")
            except AttributeError:
                # If close() doesn't exist (shouldn't happen, but be safe)
                pass
            except Exception as e:
                # Log but don't fail if close() raises an exception
                self._logger.warning(f"Error closing Weaviate client: {e}")
            finally:
                # Clear the reference and force garbage collection
                # This ensures all Python references are released
                client = self.weaviate_client
                self.weaviate_client = None
                del client
                # Only force GC on Windows where connections are more persistent
                if sys.platform == "win32":
                    gc.collect()
                    # On Windows, connections may take a moment to be released by the OS
                    # Even after close(), Windows may need a brief moment to release sockets
                    await asyncio.sleep(0.05)  # Minimal delay for Windows file lock release

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
        assert self.weaviate_client is not None, "Weaviate client must be initialized"
        assert self._embedder_factory is not None, "Embedder factory must be provided"
        failed_migrations: list[BaseDocument] = []
        embedder = self._embedder_factory.create_embedder(embedder_type)

        # Get all objects from unembedded collection
        unembedded_collection = self.weaviate_client.collections.get(unembedded_collection_name)
        unembedded_objects = await asyncio.to_thread(
            lambda: unembedded_collection.query.fetch_objects(limit=10000).objects
        )

        indexing_required = False

        if unembedded_objects:
            for obj in unembedded_objects:
                # Convert Weaviate properties (item_id) back to API format (id)
                weaviate_props = obj.properties
                if weaviate_props is not None:
                    prospective_doc = cast(BaseDocument, _convert_weaviate_properties_to_doc(weaviate_props))
                else:
                    continue
                try:
                    if loaded_doc := await document_loader(prospective_doc):
                        if loaded_doc != prospective_doc:
                            # Update the unembedded collection
                            await asyncio.to_thread(
                                unembedded_collection.data.update,
                                uuid=obj.uuid,
                                properties=_convert_doc_to_weaviate_properties(cast(dict[str, Any], loaded_doc)),
                            )
                            indexing_required = True
                    else:
                        self._logger.warning(f'Failed to load document "{prospective_doc}"')
                        await asyncio.to_thread(unembedded_collection.data.delete_by_id, obj.uuid)
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
        assert self.weaviate_client is not None, "Weaviate client must be initialized"
        version_key = f"{collection_name}_version"
        try:
            metadata = await self.read_metadata()
            return cast(int, metadata.get(version_key, 1))
        except Exception:
            return 1

    async def _set_collection_version(self, collection_name: str, version: int) -> None:
        """Set version in metadata collection."""
        assert self.weaviate_client is not None, "Weaviate client must be initialized"
        version_key = f"{collection_name}_version"
        await self.upsert_metadata(version_key, version)

    # Syncs embedded collection with unembedded collection
    async def _index_collection(
        self,
        embedded_collection_name: str,
        unembedded_collection_name: str,
        embedder: Embedder,
    ) -> None:
        assert self.weaviate_client is not None, "Weaviate client must be initialized"
        # Get all objects from unembedded collection
        unembedded_collection = self.weaviate_client.collections.get(unembedded_collection_name)
        unembedded_objects = await asyncio.to_thread(
            lambda: unembedded_collection.query.fetch_objects(limit=10000).objects
        )

        # Map by document ID (string) from properties
        unembedded_docs_by_id = {
            cast(str, obj.properties["item_id"]): obj
            for obj in unembedded_objects
            if obj.properties is not None and "item_id" in obj.properties
        }

        # Get all objects from embedded collection
        embedded_collection = self.weaviate_client.collections.get(embedded_collection_name)
        embedded_objects = await asyncio.to_thread(
            lambda: embedded_collection.query.fetch_objects(limit=10000).objects
        )

        # Map by document ID (string) from properties
        embedded_docs_by_id = {
            cast(str, obj.properties["item_id"]): obj
            for obj in embedded_objects
            if obj.properties is not None and "item_id" in obj.properties
        }

        # Remove docs from embedded collection that no longer exist in unembedded
        # Update embeddings for changed docs
        for doc_id, embedded_obj in embedded_docs_by_id.items():
            if doc_id not in unembedded_docs_by_id:
                await asyncio.to_thread(embedded_collection.data.delete_by_id, embedded_obj.uuid)
            else:
                unembedded_obj = unembedded_docs_by_id[doc_id]
                unembedded_doc = unembedded_obj.properties
                if unembedded_doc is not None and embedded_obj.properties is not None:
                    # Only recompute embeddings if checksum changed
                    if embedded_obj.properties.get("checksum") != unembedded_doc.get("checksum"):
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
                        existing_vector: Optional[Sequence[float]] = None
                        if embedded_obj.vector and isinstance(embedded_obj.vector, dict):
                            default_vector = embedded_obj.vector.get("default")
                            if (
                                isinstance(default_vector, (list, tuple))
                                and len(default_vector) > 0
                            ):
                                # Check if it's a list of floats (not list of lists)
                                if isinstance(default_vector[0], (int, float)):
                                    existing_vector = cast(list[float], list(default_vector))
                        if existing_vector is None:
                            # Recompute if vector is missing
                            embeddings = list(
                                (
                                    await embedder.embed([cast(str, unembedded_doc["content"])])
                                ).vectors
                            )
                            if not embeddings or len(embeddings[0]) == 0:
                                self._logger.warning(
                                    f"Empty embedding for document {doc_id}, skipping sync"
                                )
                                continue
                            existing_vector = embeddings[0]
                        vector = existing_vector

                    # Normalize datetime fields in the document (already in Weaviate format)
                    normalized_doc = {}
                    for k, v in unembedded_doc.items():
                        normalized_doc[k] = _normalize_datetime_value(v)
                    
                    await asyncio.to_thread(
                        embedded_collection.data.update,
                        uuid=embedded_obj.uuid,
                        properties=cast(dict[str, Any], normalized_doc),
                        vector=vector,
                    )
                unembedded_docs_by_id.pop(doc_id)

        # Add new docs from unembedded to embedded collection
        for doc_id, unembedded_obj in unembedded_docs_by_id.items():
            doc = unembedded_obj.properties
            if doc is None:
                continue
            doc_dict = doc
            embeddings = list((await embedder.embed([cast(str, doc_dict["content"])])).vectors)

            if not embeddings or len(embeddings[0]) == 0:
                self._logger.warning(f"Empty embedding for document {doc_id}, skipping")
                continue

            # Convert string ID to UUID for Weaviate
            object_uuid = _string_id_to_uuid(str(doc_id))

            # Normalize datetime fields in the document (already in Weaviate format)
            normalized_doc = {}
            for k, v in doc_dict.items():
                normalized_doc[k] = _normalize_datetime_value(v)

            await asyncio.to_thread(
                embedded_collection.data.insert,
                uuid=object_uuid,
                properties=cast(dict[str, Any], normalized_doc),
                vector=embeddings[0],
            )

        # Update version in both collections to reflect reindexing
        unembedded_version = await self._get_collection_version(unembedded_collection_name)
        # Increment version to indicate that reindexing has occurred
        new_version = unembedded_version + 1
        await self._set_collection_version(unembedded_collection_name, new_version)
        await self._set_collection_version(embedded_collection_name, new_version)

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
    ) -> WeaviateCollection[TDocument]:
        assert self.weaviate_client is not None, "Weaviate client must be initialized"
        assert self._embedder_factory is not None, "Embedder factory must be provided"
        assert self._embedding_cache_provider is not None, (
            "Embedding cache provider must be provided"
        )
        if name in self._collections:
            raise ValueError(f'Collection "{name}" already exists.')

        embedder = self._embedder_factory.create_embedder(embedder_type)

        embedded_collection_name = self.format_collection_name(name, embedder_type)
        unembedded_collection_name = f"{name}_unembedded"

        # Create embedded collection
        def _create_embedded_collection() -> None:
            assert self.weaviate_client is not None
            try:
                self.weaviate_client.collections.create(  # type: ignore[misc]
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

        await asyncio.to_thread(_create_embedded_collection)

        # Create unembedded collection (with minimal vector size for metadata storage)
        def _create_unembedded_collection() -> None:
            assert self.weaviate_client is not None
            try:
                self.weaviate_client.collections.create(  # type: ignore[misc]
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

        await asyncio.to_thread(_create_unembedded_collection)

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
        self._collections[name] = collection  # type: ignore[assignment]

        return collection  # type: ignore[return-value]

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> WeaviateCollection[TDocument]:
        assert self.weaviate_client is not None, "Weaviate client must be initialized"
        assert self._embedder_factory is not None, "Embedder factory must be provided"
        assert self._embedding_cache_provider is not None, (
            "Embedding cache provider must be provided"
        )
        if collection := self._collections.get(name):
            return cast(WeaviateCollection[TDocument], collection)

        # Find unembedded collection first which acts as the SSOT.
        unembedded_collection_name = f"{name}_unembedded"
        embedded_collection_name = self.format_collection_name(name, embedder_type)

        # Check if collections exist
        collections = self.weaviate_client.collections.list_all()

        if unembedded_collection_name in collections:
            if embedded_collection_name not in collections:
                # Create embedded collection if it doesn't exist
                self._embedder_factory.create_embedder(embedder_type)

                def _create_embedded_collection_in_get() -> None:
                    assert self.weaviate_client is not None
                    try:
                        self.weaviate_client.collections.create(  # type: ignore[misc]
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

                await asyncio.to_thread(_create_embedded_collection_in_get)

            await self._index_collection(
                embedded_collection_name=embedded_collection_name,
                unembedded_collection_name=unembedded_collection_name,
                embedder=self._embedder_factory.create_embedder(embedder_type),
            )

            collection = WeaviateCollection(
                self._logger,
                weaviate_client=self.weaviate_client,
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
            self._collections[name] = collection  # type: ignore[assignment]
            return collection  # type: ignore[return-value]

        raise ValueError(f'Weaviate collection "{name}" not found.')

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> WeaviateCollection[TDocument]:
        assert self.weaviate_client is not None, "Weaviate client must be initialized"
        assert self._embedder_factory is not None, "Embedder factory must be provided"
        assert self._embedding_cache_provider is not None, (
            "Embedding cache provider must be provided"
        )
        if collection := self._collections.get(name):
            return cast(WeaviateCollection[TDocument], collection)

        embedder = self._embedder_factory.create_embedder(embedder_type)

        embedded_collection_name = self.format_collection_name(name, embedder_type)
        unembedded_collection_name = f"{name}_unembedded"

        # Get or create collections
        collections = self.weaviate_client.collections.list_all()

        if unembedded_collection_name not in collections:

            def _create_unembedded_collection_in_get() -> None:
                assert self.weaviate_client is not None
                try:
                    self.weaviate_client.collections.create(  # type: ignore[misc]
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

            await asyncio.to_thread(_create_unembedded_collection_in_get)
        if embedded_collection_name not in collections:

            def _create_embedded_collection_in_get_or_create() -> None:
                assert self.weaviate_client is not None
                try:
                    self.weaviate_client.collections.create(  # type: ignore[misc]
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

            await asyncio.to_thread(_create_embedded_collection_in_get_or_create)

        collection = WeaviateCollection(
            self._logger,
            weaviate_client=self.weaviate_client,
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
        self._collections[name] = collection  # type: ignore[assignment]

        return collection  # type: ignore[return-value]

    @override
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        assert self.weaviate_client is not None, "Weaviate client must be initialized"
        if name not in self._collections:
            raise ValueError(f'Collection "{name}" not found.')

        embedded_collection_name = self.format_collection_name(
            name, type(self._collections[name]._embedder)
        )
        unembedded_collection_name = f"{name}_unembedded"

        await asyncio.to_thread(self.weaviate_client.collections.delete, embedded_collection_name)
        await asyncio.to_thread(self.weaviate_client.collections.delete, unembedded_collection_name)
        del self._collections[name]

    @override
    async def upsert_metadata(
        self,
        key: str,
        value: JSONSerializable,
    ) -> None:
        assert self.weaviate_client is not None, "Weaviate client must be initialized"
        metadata_collection_name = "metadata"

        # Check if metadata collection exists and create if needed
        collections = self.weaviate_client.collections.list_all()

        if metadata_collection_name not in collections:
            # Try to create the collection, but handle the case where it already exists
            # (e.g., due to race condition or case sensitivity)
            # Use dynamic schema (no explicit properties) to allow arbitrary key-value pairs
            def _create_metadata_collection() -> None:
                assert self.weaviate_client is not None
                try:
                    self.weaviate_client.collections.create(  # type: ignore[misc]
                        name=metadata_collection_name,
                        vector_config=Configure.Vectors.self_provided(
                            vector_index_config=Configure.VectorIndex.hnsw(
                                distance_metric=VectorDistances.COSINE,
                            ),
                        ),
                        # No explicit properties - use dynamic schema for singleton metadata object
                    )
                except WeaviateBaseError as e:
                    _handle_collection_exists_error(e)

            await asyncio.to_thread(_create_metadata_collection)

        # Get existing metadata using deterministic UUID
        metadata_collection = self.weaviate_client.collections.get(metadata_collection_name)
        metadata_uuid = _string_id_to_uuid("__metadata__")
        
        # FIX: Use query.fetch_object_by_id instead of data.get_by_id
        # Weaviate v4 returns None if not found, it does not raise an exception usually.
        obj = await asyncio.to_thread(
            lambda: metadata_collection.query.fetch_object_by_id(metadata_uuid)
        )

        if obj is not None:
            # Object exists, update it
            # Convert properties to ensure datetime fields are strings
            document = cast(dict[str, JSONSerializable], _convert_weaviate_properties_to_doc(obj.properties) or {})
            # Normalize the value being set (in case it's a datetime)
            normalized_value = _normalize_datetime_value(value) if isinstance(value, datetime) else value
            document[key] = normalized_value
            
            await asyncio.to_thread(
                metadata_collection.data.update,
                uuid=metadata_uuid,
                properties=cast(dict[str, Any], document),
            )
        else:
            # Object does not exist, insert it
            # Normalize the value being set (in case it's a datetime)
            normalized_value = _normalize_datetime_value(value) if isinstance(value, datetime) else value
            document = {key: normalized_value}
            try:
                await asyncio.to_thread(
                    metadata_collection.data.insert,
                    uuid=metadata_uuid,
                    properties=cast(dict[str, Any], document),
                    vector=[0.0],  # Dummy vector
                )
            except WeaviateBaseError as e:
                # Handle race condition where it was created between the fetch and insert
                error_str = str(e).lower()
                if "already exists" in error_str or "422" in error_str:
                    # Retry as update
                    await asyncio.to_thread(
                        metadata_collection.data.update,
                        uuid=metadata_uuid,
                        properties=cast(dict[str, Any], document),
                    )
                else:
                    raise

    @override
    async def remove_metadata(
        self,
        key: str,
    ) -> None:
        assert self.weaviate_client is not None, "Weaviate client must be initialized"
        metadata_collection_name = "metadata"

        collections = self.weaviate_client.collections.list_all()

        if metadata_collection_name not in collections:
            raise ValueError(f'Metadata with key "{key}" not found.')

        metadata_collection = self.weaviate_client.collections.get(metadata_collection_name)
        metadata_uuid = _string_id_to_uuid("__metadata__")

        # FIX: Use fetch_object_by_id for consistency and safety
        obj = await asyncio.to_thread(
            lambda: metadata_collection.query.fetch_object_by_id(metadata_uuid)
        )

        if obj is not None and obj.properties:
            # Convert properties to ensure datetime fields are strings
            document = cast(dict[str, JSONSerializable], _convert_weaviate_properties_to_doc(obj.properties) or {})
            if key not in document:
                raise ValueError(f'Metadata with key "{key}" not found.')
            document.pop(key)

            await asyncio.to_thread(
                metadata_collection.data.update,
                uuid=metadata_uuid,
                properties=cast(dict[str, Any], document),
            )
        else:
            raise ValueError(f'Metadata with key "{key}" not found.')

    @override
    async def read_metadata(
        self,
    ) -> dict[str, JSONSerializable]:
        assert self.weaviate_client is not None, "Weaviate client must be initialized"
        metadata_collection_name = "metadata"

        collections = self.weaviate_client.collections.list_all()

        if metadata_collection_name in collections:
            metadata_collection = self.weaviate_client.collections.get(metadata_collection_name)
            metadata_uuid = _string_id_to_uuid("__metadata__")
            
            # FIX: Use query.fetch_object_by_id instead of data.get_by_id
            try:
                obj = await asyncio.to_thread(
                    lambda: metadata_collection.query.fetch_object_by_id(metadata_uuid)
                )
                if obj and obj.properties:
                    # Convert properties to ensure datetime fields are strings
                    return cast(dict[str, JSONSerializable], _convert_weaviate_properties_to_doc(obj.properties) or {})
                else:
                    return {}
            except Exception:
                # If fetch fails, return empty dict
                return {}
        else:
            return {}


class WeaviateCollection(Generic[TDocument], VectorCollection[TDocument]):
    def __init__(
        self,
        logger: Logger,
        weaviate_client: weaviate.WeaviateClient,
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
        self._database: Optional[WeaviateDatabase] = (
            None  # Reference to parent database for version methods
        )

    @override
    async def find(
        self,
        filters: Where,
    ) -> Sequence[TDocument]:
        async with self._lock.reader_lock:
            weaviate_filter = _convert_where_to_weaviate_filter(filters)

            # Don't silently fall back to fetching all objects - let errors propagate
            collection = self.weaviate_client.collections.get(self.embedded_collection_name)
            if weaviate_filter:
                objects = await asyncio.to_thread(
                    lambda: collection.query.fetch_objects(
                        limit=10000, filters=weaviate_filter
                    ).objects
                )
            else:
                objects = await asyncio.to_thread(
                    lambda: collection.query.fetch_objects(limit=10000).objects
                )

            return [
                cast(TDocument, _convert_weaviate_properties_to_doc(obj.properties))
                for obj in objects
                if obj.properties is not None
            ]

    @override
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]:
        async with self._lock.reader_lock:
            weaviate_filter = _convert_where_to_weaviate_filter(filters)

            # Don't silently fall back to fetching all objects - let errors propagate
            collection = self.weaviate_client.collections.get(self.embedded_collection_name)
            if weaviate_filter:
                objects = await asyncio.to_thread(
                    lambda: collection.query.fetch_objects(
                        limit=1, filters=weaviate_filter
                    ).objects
                )
            else:
                objects = await asyncio.to_thread(
                    lambda: collection.query.fetch_objects(limit=1).objects
                )

            if objects and objects[0].properties is not None:
                converted_props = _convert_weaviate_properties_to_doc(objects[0].properties)
                if converted_props is not None:
                    return cast(TDocument, converted_props)

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

            # Convert string ID to UUID for Weaviate
            object_uuid = _string_id_to_uuid(str(document["id"]))

            # Insert into unembedded collection with retry on timeout
            unembedded_collection = self.weaviate_client.collections.get(
                self._unembedded_collection_name
            )
            await _retry_on_timeout_async(
                lambda: asyncio.to_thread(
                    unembedded_collection.data.insert,
                    uuid=object_uuid,
                    properties=_convert_doc_to_weaviate_properties(cast(dict[str, Any], document)),
                    vector=[0.0],  # Dummy vector
                ),
                max_retries=3,
                logger=self._logger,
            )

            # Insert into embedded collection with retry on timeout
            embedded_collection = self.weaviate_client.collections.get(
                self.embedded_collection_name
            )
            await _retry_on_timeout_async(
                lambda: asyncio.to_thread(
                    embedded_collection.data.insert,
                    uuid=object_uuid,
                    properties=_convert_doc_to_weaviate_properties(cast(dict[str, Any], document)),
                    vector=embeddings[0],
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
            weaviate_filter = _convert_where_to_weaviate_filter(filters)

            collection = self.weaviate_client.collections.get(self.embedded_collection_name)
            objects = await asyncio.to_thread(
                lambda: collection.query.fetch_objects(limit=1, filters=weaviate_filter).objects
            )

            if objects:
                obj = objects[0]
                # Convert Weaviate properties (item_id) back to API format (id)
                weaviate_doc = cast(dict[str, Any], obj.properties)
                doc = _convert_weaviate_properties_to_doc(weaviate_doc)
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

                # Update unembedded collection with retry on timeout
                unembedded_collection = self.weaviate_client.collections.get(
                    self._unembedded_collection_name
                )
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        unembedded_collection.data.update,
                        uuid=obj.uuid,
                        properties=_convert_doc_to_weaviate_properties(updated_document),
                    ),
                    max_retries=3,
                    logger=self._logger,
                )

                # Update embedded collection with retry on timeout
                embedded_collection = self.weaviate_client.collections.get(
                    self.embedded_collection_name
                )
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        embedded_collection.data.update,
                        uuid=obj.uuid,
                        properties=_convert_doc_to_weaviate_properties(updated_document),
                        vector=embeddings[0],
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

                # Convert string ID to UUID for Weaviate
                object_uuid = _string_id_to_uuid(str(params["id"]))

                # Insert into unembedded collection with retry on timeout
                unembedded_collection = self.weaviate_client.collections.get(
                    self._unembedded_collection_name
                )
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        unembedded_collection.data.insert,
                        uuid=object_uuid,
                        properties=_convert_doc_to_weaviate_properties(cast(dict[str, Any], params)),
                        vector=[0.0],  # Dummy vector
                    ),
                    max_retries=3,
                    logger=self._logger,
                )

                # Insert into embedded collection with retry on timeout
                embedded_collection = self.weaviate_client.collections.get(
                    self.embedded_collection_name
                )
                await _retry_on_timeout_async(
                    lambda: asyncio.to_thread(
                        embedded_collection.data.insert,
                        uuid=object_uuid,
                        properties=_convert_doc_to_weaviate_properties(cast(dict[str, Any], params)),
                        vector=embeddings[0],
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
            weaviate_filter = _convert_where_to_weaviate_filter(filters)

            collection = self.weaviate_client.collections.get(self.embedded_collection_name)
            objects = await asyncio.to_thread(
                lambda: collection.query.fetch_objects(limit=2, filters=weaviate_filter).objects
            )

            if len(objects) > 1:
                raise ValueError(
                    f"WeaviateCollection delete_one: detected more than one document with filters '{filters}'. Aborting..."
                )

            if objects:
                weaviate_props = objects[0].properties
                converted_props = _convert_weaviate_properties_to_doc(weaviate_props)
                if converted_props is None:
                    raise ValueError("Failed to convert Weaviate properties to document format")
                deleted_document = cast(TDocument, converted_props)
                object_uuid = objects[0].uuid

                self._version += 1

                # Delete from unembedded collection
                unembedded_collection = self.weaviate_client.collections.get(
                    self._unembedded_collection_name
                )
                await asyncio.to_thread(unembedded_collection.data.delete_by_id, object_uuid)

                # Delete from embedded collection
                embedded_collection = self.weaviate_client.collections.get(
                    self.embedded_collection_name
                )
                await asyncio.to_thread(embedded_collection.data.delete_by_id, object_uuid)

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
            weaviate_filter = _convert_where_to_weaviate_filter(filters)

            if not query_embeddings or len(query_embeddings[0]) == 0:
                self._logger.warning(f"Empty embedding generated for query: {query}")
                return []

            collection = self.weaviate_client.collections.get(self.embedded_collection_name)
            search_results = await asyncio.to_thread(
                lambda: collection.query.near_vector(
                    near_vector=query_embeddings[0],
                    limit=k,
                    filters=weaviate_filter,
                    return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
                ).objects
            )

            if not search_results:
                return []

            # Convert properties to JSON-serializable format for logging
            # This ensures datetime objects are converted to strings
            serializable_results = [
                _convert_weaviate_properties_to_doc(r.properties) or {}
                for r in search_results
                if r.properties is not None
            ]
            self._logger.trace(
                f"Similar documents found\n{json.dumps(serializable_results, indent=2, default=str)}"
            )

            return [
                SimilarDocumentResult(
                    document=cast(TDocument, _convert_weaviate_properties_to_doc(result.properties)),
                    distance=result.metadata.distance
                    if result.metadata and result.metadata.distance
                    else 0.0,
                )
                for result in search_results
                if result.properties is not None
            ]
