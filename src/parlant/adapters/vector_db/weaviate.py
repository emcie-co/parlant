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
from urllib.parse import urlparse
import weaviate  # type: ignore[import-untyped]
from weaviate.classes.query import Filter, MetadataQuery  # type: ignore[import-untyped]
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
    """Retry an async operation on timeout errors with exponential backoff."""
    last_exception: Exception | None = None
    total_attempts = max_retries + 1

    for attempt in range(total_attempts):
        try:
            return await operation()
        except (WeaviateBaseError, Exception) as e:
            error_str = str(e).lower()
            is_timeout = (
                "timeout" in error_str
                or "read operation timed out" in error_str
                or "readtimeout" in error_str
                or "connection" in error_str
            )

            if is_timeout and attempt < total_attempts - 1:
                delay = base_delay * (2**attempt)
                if logger:
                    logger.warning(
                        f"Weaviate operation timed out (attempt {attempt + 1}/{total_attempts}). "
                        f"Retrying in {delay}s..."
                    )
                await asyncio.sleep(delay)
                last_exception = e
                continue
            else:
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
                        auth_credentials=weaviate.auth.AuthApiKey(self._api_key),
                    )
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
                else:
                    raise ValueError("Either url or path must be provided")
                break
            except Exception as e:
                if "connection" in str(e).lower() and attempt < max_retries - 1:
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
        self._collections.clear()
        if self.weaviate_client is not None:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self.weaviate_client.close),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                self._logger.warning("Weaviate client close() timed out, forcing cleanup")
            except (AttributeError, Exception) as e:
                if not isinstance(e, AttributeError):
                    self._logger.warning(f"Error closing Weaviate client: {e}")
            finally:
                client = self.weaviate_client
                self.weaviate_client = None
                del client
                if sys.platform == "win32":
                    gc.collect()
                    await asyncio.sleep(0.05)

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

        unembedded_collection = await asyncio.to_thread(
            self.weaviate_client.collections.get,
            unembedded_collection_name
        )
        unembedded_objects = await asyncio.to_thread(
            lambda: unembedded_collection.query.fetch_objects(limit=10000).objects
        )

        indexing_required = False

        if unembedded_objects:
            for obj in unembedded_objects:
                if obj.properties is None:
                    continue
                prospective_doc = cast(BaseDocument, _from_weaviate_properties(obj.properties))
                if prospective_doc is None:
                    continue
                try:
                    if loaded_doc := await document_loader(prospective_doc):
                        if loaded_doc != prospective_doc:
                            await asyncio.to_thread(
                                unembedded_collection.data.update,
                                uuid=obj.uuid,
                                properties=_to_weaviate_properties(cast(dict[str, Any], loaded_doc)),
                            )
                            indexing_required = True
                    else:
                        self._logger.warning(f'Failed to load document "{prospective_doc}"')
                        await asyncio.to_thread(unembedded_collection.data.delete_by_id, obj.uuid)
                        failed_migrations.append(prospective_doc)
                except Exception as e:
                    self._logger.error(f"Failed to load document '{prospective_doc}'. error: {e}.")
                    failed_migrations.append(prospective_doc)

            if failed_migrations:
                failed_collection = await self.get_or_create_collection(
                    "failed_migrations",
                    BaseDocument,
                    NullEmbedder,
                    identity_loader,
                )
                for failed_doc in failed_migrations:
                    await failed_collection.insert_one(failed_doc)

        unembedded_version = await self._get_collection_version(unembedded_collection_name)
        embedded_version = await self._get_collection_version(embedded_collection_name)

        if indexing_required or unembedded_version != embedded_version:
            await self._index_collection(
                embedded_collection_name, unembedded_collection_name, embedder
            )

        return embedded_collection_name

    async def _index_collection(
        self,
        embedded_collection_name: str,
        unembedded_collection_name: str,
        embedder: Embedder,
    ) -> None:
        assert self.weaviate_client is not None

        unembedded_collection = await asyncio.to_thread(
            self.weaviate_client.collections.get,
            unembedded_collection_name
        )
        unembedded_objects = await asyncio.to_thread(
            lambda: unembedded_collection.query.fetch_objects(limit=10000).objects
        )

        unembedded_docs_by_id = {
            cast(str, obj.properties["item_id"]): obj
            for obj in unembedded_objects
            if obj.properties is not None and "item_id" in obj.properties
        }

        embedded_collection = await asyncio.to_thread(
            self.weaviate_client.collections.get,
            embedded_collection_name
        )
        embedded_objects = await asyncio.to_thread(
            lambda: embedded_collection.query.fetch_objects(limit=10000).objects
        )

        embedded_docs_by_id = {
            cast(str, obj.properties["item_id"]): obj
            for obj in embedded_objects
            if obj.properties is not None and "item_id" in obj.properties
        }

        for doc_id, embedded_obj in embedded_docs_by_id.items():
            if doc_id not in unembedded_docs_by_id:
                await asyncio.to_thread(embedded_collection.data.delete_by_id, embedded_obj.uuid)
            else:
                unembedded_obj = unembedded_docs_by_id[doc_id]
                unembedded_doc = _from_weaviate_properties(unembedded_obj.properties)
                if unembedded_doc is None:
                    continue

                embedded_props = embedded_obj.properties
                if embedded_props and embedded_props.get("checksum") != unembedded_doc.get("checksum"):
                    embeddings = list((await embedder.embed([cast(str, unembedded_doc["content"])])).vectors)
                    if not embeddings or len(embeddings[0]) == 0:
                        self._logger.warning(f"Empty embedding for document {doc_id}, skipping sync")
                        continue
                    vector = embeddings[0]
                else:
                    if embedded_obj.vector and isinstance(embedded_obj.vector, dict):
                        default_vector = embedded_obj.vector.get("default")
                        if isinstance(default_vector, (list, tuple)) and len(default_vector) > 0:
                            if isinstance(default_vector[0], (int, float)):
                                vector = cast(list[float], list(default_vector))
                            else:
                                embeddings = list((await embedder.embed([cast(str, unembedded_doc["content"])])).vectors)
                                if not embeddings or len(embeddings[0]) == 0:
                                    continue
                                vector = embeddings[0]
                        else:
                            embeddings = list((await embedder.embed([cast(str, unembedded_doc["content"])])).vectors)
                            if not embeddings or len(embeddings[0]) == 0:
                                continue
                            vector = embeddings[0]
                    else:
                        embeddings = list((await embedder.embed([cast(str, unembedded_doc["content"])])).vectors)
                        if not embeddings or len(embeddings[0]) == 0:
                            continue
                        vector = embeddings[0]

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
            embeddings = list((await embedder.embed([cast(str, doc_dict["content"])])).vectors)
            if not embeddings or len(embeddings[0]) == 0:
                self._logger.warning(f"Empty embedding for document {doc_id}, skipping")
                continue

            object_uuid = _string_id_to_uuid(str(doc_id))
            normalized_doc = _to_weaviate_properties(doc_dict)
            await asyncio.to_thread(
                embedded_collection.data.insert,
                uuid=object_uuid,
                properties=normalized_doc,
                vector=embeddings[0],
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
        self._collections[name] = collection
        return collection

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

        if metadata_collection_name not in collection_list:
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
            await asyncio.to_thread(_create_metadata)

        metadata_collection = await asyncio.to_thread(
            self.weaviate_client.collections.get,
            metadata_collection_name
        )
        metadata_uuid = _string_id_to_uuid("__metadata__")

        obj = await asyncio.to_thread(
            lambda: metadata_collection.query.fetch_object_by_id(metadata_uuid)
        )

        if obj is not None:
            document = cast(dict[str, JSONSerializable], _from_weaviate_properties(obj.properties) or {})
            document[key] = _normalize_datetime(value)
            await asyncio.to_thread(
                metadata_collection.data.update,
                uuid=obj.uuid,
                properties=_to_weaviate_properties(document),
            )
        else:
            document = {key: _normalize_datetime(value)}
            await asyncio.to_thread(
                metadata_collection.data.insert,
                uuid=metadata_uuid,
                properties=_to_weaviate_properties(document),
                vector=[0.0],
            )

    @override
    async def remove_metadata(self, key: str) -> None:
        assert self.weaviate_client is not None
        metadata_collection_name = "metadata"

        collections = await asyncio.to_thread(self.weaviate_client.collections.list_all)
        collection_list = list(collections)

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

        collections = await asyncio.to_thread(self.weaviate_client.collections.list_all)
        collection_list = list(collections)

        if metadata_collection_name in collection_list:
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
                    return cast(dict[str, JSONSerializable], _from_weaviate_properties(obj.properties) or {})
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

            embedded_collection = await asyncio.to_thread(
                self.weaviate_client.collections.get,
                self.embedded_collection_name
            )
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
