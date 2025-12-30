# Copyright 2025 Werktøj ApS
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
import os
from typing import Any, Awaitable, Callable, Optional, Sequence, cast
from typing_extensions import override, Self

from parlant.core.common import JSONSerializable
from parlant.core.loggers import Logger
from parlant.core.nlp.embedding import (
    Embedder,
    EmbedderFactory,
    EmbeddingCacheProvider,
    EmbeddingResult,
    NoOpEmbedder,
)
from parlant.core.persistence.common import (
    Where,
    WhereExpression,
    LogicalOperator,
    LiteralValue,
    ensure_is_total,
)
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

try:
    from elasticsearch import AsyncElasticsearch
    from elasticsearch.helpers import async_bulk
    from elasticsearch.exceptions import (
        NotFoundError,
        RequestError,
        ConnectionError as ESConnectionError,
        TransportError,
        ConnectionTimeout,
        ConflictError,
    )
except ImportError:
    raise ImportError(
        "elasticsearch package is required for ElasticsearchVectorDatabase. "
        "Install it with: pip install elasticsearch"
    )


def get_elasticsearch_index_prefix_from_env() -> str:
    """
    Get the Elasticsearch index prefix from environment variables.

    Environment variables:
        ELASTICSEARCH__INDEX_PREFIX: Prefix for all indices (default: parlant)

    Returns:
        Index prefix string
    """
    return os.environ.get("ELASTICSEARCH__INDEX_PREFIX", "parlant")


def get_elasticsearch_index_settings_from_env() -> dict[str, Any]:
    """
    Get Elasticsearch index settings from environment variables.

    Environment variables:
        ELASTICSEARCH__NUMBER_OF_SHARDS: Number of primary shards (default: 1)
        ELASTICSEARCH__NUMBER_OF_REPLICAS: Number of replica shards (default: 0)
        ELASTICSEARCH__REFRESH_INTERVAL: Index refresh interval (default: 5s)
        ELASTICSEARCH__CODEC: Index codec for compression (default: best_compression)
        ELASTICSEARCH__ENABLE_QUERY_CACHE: Enable query caching (default: true)

    Returns:
        Dictionary of index settings
    """
    number_of_shards: int = int(os.environ.get("ELASTICSEARCH__NUMBER_OF_SHARDS", "1"))
    number_of_replicas: int = int(os.environ.get("ELASTICSEARCH__NUMBER_OF_REPLICAS", "0"))
    refresh_interval: str = os.environ.get("ELASTICSEARCH__REFRESH_INTERVAL", "5s")
    codec: str = os.environ.get("ELASTICSEARCH__CODEC", "best_compression")
    enable_query_cache: bool = os.environ.get(
        "ELASTICSEARCH__ENABLE_QUERY_CACHE", "true"
    ).lower() in (
        "true",
        "1",
        "yes",
    )

    return {
        "number_of_shards": number_of_shards,
        "number_of_replicas": number_of_replicas,
        "index.refresh_interval": refresh_interval,
        "index.codec": codec,
        "index.queries.cache.enabled": enable_query_cache,
        # Additional best-practice settings
        "index.mapping.total_fields.limit": 2000,
        "index.max_result_window": 10000,
        "index.search.idle.after": "30s",
        "index.translog.durability": "async",
        "index.translog.sync_interval": "5s",
    }


def create_elasticsearch_client_from_env() -> AsyncElasticsearch:
    """
    Create an AsyncElasticsearch client using environment variables.

    Environment variables:
        ELASTICSEARCH__HOST: Elasticsearch host (default: localhost)
        ELASTICSEARCH__PORT: Elasticsearch port (default: 9200)
        ELASTICSEARCH__USERNAME: Elasticsearch username (optional)
        ELASTICSEARCH__PASSWORD: Elasticsearch password (optional)
        ELASTICSEARCH__USE_SSL: Use SSL connection (default: false)
        ELASTICSEARCH__VERIFY_CERTS: Verify SSL certificates (default: false)
        ELASTICSEARCH__TIMEOUT: Request timeout in seconds (default: 30)
        ELASTICSEARCH__MAX_RETRIES: Maximum number of retries (default: 3)
        ELASTICSEARCH__RETRY_ON_TIMEOUT: Retry on timeout (default: true)

    Returns:
        Configured AsyncElasticsearch client
    """
    host: str = os.environ.get("ELASTICSEARCH__HOST", "localhost")
    port: int = int(os.environ.get("ELASTICSEARCH__PORT", "9200"))
    username: Optional[str] = os.environ.get("ELASTICSEARCH__USERNAME")
    password: Optional[str] = os.environ.get("ELASTICSEARCH__PASSWORD")
    use_ssl: bool = os.environ.get("ELASTICSEARCH__USE_SSL", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    verify_certs: bool = os.environ.get("ELASTICSEARCH__VERIFY_CERTS", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    timeout: int = int(os.environ.get("ELASTICSEARCH__TIMEOUT", "30"))
    max_retries: int = int(os.environ.get("ELASTICSEARCH__MAX_RETRIES", "3"))
    retry_on_timeout: bool = os.environ.get("ELASTICSEARCH__RETRY_ON_TIMEOUT", "true").lower() in (
        "true",
        "1",
        "yes",
    )

    # Build the connection URL
    scheme: str = "https" if use_ssl else "http"
    if username and password:
        url: str = f"{scheme}://{username}:{password}@{host}:{port}"
    else:
        url: str = f"{scheme}://{host}:{port}"

    return AsyncElasticsearch(
        hosts=[url],
        request_timeout=timeout,
        max_retries=max_retries,
        retry_on_timeout=retry_on_timeout,
        verify_certs=verify_certs,
        # Performance optimizations for ES8
        http_compress=True,
        connections_per_node=10,  # ES8: replaces deprecated maxsize parameter
    )


class ElasticsearchVectorDatabase(VectorDatabase):
    """
    An Elasticsearch implementation of the VectorDatabase interface.

    This adapter uses Elasticsearch's dense_vector field type and kNN search
    capabilities to provide efficient vector storage and similarity search.

    Features:
    - Dense vector storage with configurable similarity metrics
    - kNN search for vector similarity queries
    - Metadata storage and filtering
    - Automatic embedding generation and caching
    - Migration support for schema changes
    - Dual-index architecture (embedded + unembedded) for data integrity
    """

    def __init__(
        self,
        elasticsearch_client: AsyncElasticsearch,
        index_prefix: str,
        logger: Logger,
        embedder_factory: EmbedderFactory,
        embedding_cache_provider: EmbeddingCacheProvider,
    ) -> None:
        """
        Initialize the Elasticsearch vector database.

        Args:
            elasticsearch_client: Configured AsyncElasticsearch client
            index_prefix: Prefix for all Elasticsearch indices created by this database
            logger: Logger instance for error reporting
            embedder_factory: Factory for creating embedder instances
            embedding_cache_provider: Provider for embedding caches
        """
        self.elasticsearch_client: AsyncElasticsearch = elasticsearch_client
        self.index_prefix: str = index_prefix
        self._logger: Logger = logger
        self._embedder_factory: EmbedderFactory = embedder_factory
        self._embedding_cache_provider: EmbeddingCacheProvider = embedding_cache_provider
        self._collections: dict[str, ElasticsearchVectorCollection[BaseDocument]] = {}
        self._metadata_index: str = f"{index_prefix}_vecdb_metadata"
        self._version: int = 1

    def _get_embedded_index_name(self, collection_name: str, embedder_type: type[Embedder]) -> str:
        """Generate the full Elasticsearch index name for an embedded vector collection."""
        return f"{self.index_prefix}_vecdb_{collection_name}_{embedder_type.__name__}".lower()

    def _get_unembedded_index_name(self, collection_name: str) -> str:
        """Generate the full Elasticsearch index name for an unembedded collection (source of truth)."""
        return f"{self.index_prefix}_vecdb_{collection_name}_unembedded".lower()

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        # Create index template for consistent mappings
        await self._create_index_template()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        """Exit async context manager."""
        # Close Elasticsearch client
        await self.elasticsearch_client.close()

    async def _create_index_template(self) -> None:
        """Create index template for consistent mapping across collections."""
        template_name: str = f"{self.index_prefix}_vecdb_template"

        template: dict[str, Any] = {
            "index_patterns": [f"{self.index_prefix}_vecdb_*"],
            "template": {
                "settings": get_elasticsearch_index_settings_from_env(),
                "mappings": {
                    "dynamic_templates": [
                        {
                            "metadata_strings_as_keywords": {
                                "path_match": "metadata.*",
                                "match_mapping_type": "string",
                                "mapping": {
                                    "type": "keyword",
                                    "ignore_above": 256,
                                    "fields": {"text": {"type": "text", "analyzer": "standard"}},
                                },
                            }
                        }
                    ]
                },
            },
        }

        try:
            await self.elasticsearch_client.indices.put_index_template(
                name=template_name, body=template
            )
            self._logger.info(f"Created index template: {template_name}")
        except Exception as e:
            self._logger.warning(f"Failed to create index template: {e}")

    async def health_check(self) -> dict[str, Any]:
        """Check Elasticsearch cluster health and connection."""
        try:
            health: dict[str, Any] = await self.elasticsearch_client.cluster.health()
            info: dict[str, Any] = await self.elasticsearch_client.info()
            indices: list[dict[str, Any]] = await self.elasticsearch_client.cat.indices(
                format="json"
            )

            return {
                "status": health["status"],
                "cluster_name": health["cluster_name"],
                "number_of_nodes": health["number_of_nodes"],
                "elasticsearch_version": info["version"]["number"],
                "indices_count": len(indices),
            }
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return {"status": "red", "error": str(e)}

    async def _ensure_metadata_index_exists(self) -> None:
        """Ensure the metadata index exists."""
        try:
            exists: bool = await self.elasticsearch_client.indices.exists(
                index=self._metadata_index
            )
            if not exists:
                index_settings: dict[str, Any] = get_elasticsearch_index_settings_from_env()

                mappings_def: dict[str, Any] = {
                    "properties": {
                        "key": {"type": "keyword"},
                        "value": {"type": "object", "enabled": False},
                    }
                }
                settings_def: dict[str, Any] = {
                    "number_of_shards": index_settings["number_of_shards"],
                    "number_of_replicas": index_settings["number_of_replicas"],
                }
                await self.elasticsearch_client.indices.create(
                    index=self._metadata_index, mappings=mappings_def, settings=settings_def
                )
                self._logger.info(f"Created Elasticsearch metadata index: {self._metadata_index}")
        except RequestError as e:
            self._logger.error(f"Failed to create metadata index {self._metadata_index}: {e}")
            raise

    async def _ensure_unembedded_index_exists(self, index_name: str) -> None:
        """
        Ensure the unembedded index exists (source of truth for raw documents).

        Args:
            index_name: Name of the unembedded Elasticsearch index
        """
        try:
            exists: bool = await self.elasticsearch_client.indices.exists(index=index_name)
            if not exists:
                index_settings: dict[str, Any] = get_elasticsearch_index_settings_from_env()

                mappings_def: dict[str, Any] = {
                    "properties": {
                        "id": {"type": "keyword"},
                        "version": {"type": "keyword"},
                        "content": {"type": "text", "analyzer": "standard"},
                        "checksum": {"type": "keyword"},
                        "metadata": {"type": "object", "dynamic": True},
                    }
                }

                await self.elasticsearch_client.indices.create(
                    index=index_name, mappings=mappings_def, settings=index_settings
                )
                self._logger.info(f"Created unembedded Elasticsearch index: {index_name}")
        except RequestError as e:
            self._logger.error(f"Failed to create unembedded index {index_name}: {e}")
            raise

    async def _ensure_embedded_index_exists(
        self, index_name: str, embedder_type: type[Embedder]
    ) -> None:
        """
        Ensure the embedded vector index exists with optimized settings for vector search.

        Args:
            index_name: Name of the Elasticsearch index
            embedder_type: Embedder type to determine vector dimensions
        """
        try:
            exists: bool = await self.elasticsearch_client.indices.exists(index=index_name)
            if not exists:
                embedder: Embedder = self._embedder_factory.create_embedder(embedder_type)

                # Choose similarity metric based on embedder type
                similarity_metric: str = "l2_norm" if embedder_type == NoOpEmbedder else "cosine"

                index_settings: dict[str, Any] = get_elasticsearch_index_settings_from_env()

                mappings_def: dict[str, Any] = {
                    "dynamic_templates": [
                        {
                            "metadata_strings": {
                                "match_mapping_type": "string",
                                "path_match": "metadata.*",
                                "mapping": {"type": "keyword"},
                            }
                        }
                    ],
                    "properties": {
                        "id": {"type": "keyword"},
                        "version": {"type": "keyword"},
                        "content": {"type": "text", "analyzer": "standard"},
                        "checksum": {"type": "keyword"},
                        "content_vector": {
                            "type": "dense_vector",
                            "dims": embedder.dimensions,
                            "index": True,
                            "similarity": similarity_metric,
                        },
                        "metadata": {"type": "object", "dynamic": True},
                    },
                }
                await self.elasticsearch_client.indices.create(
                    index=index_name, mappings=mappings_def, settings=index_settings
                )
                self._logger.info(f"Created embedded Elasticsearch vector index: {index_name}")
        except RequestError as e:
            self._logger.error(f"Failed to create embedded index {index_name}: {e}")
            raise

    async def _load_collection_documents(
        self,
        embedded_index_name: str,
        unembedded_index_name: str,
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> None:
        """
        Load documents from unembedded index, migrate them if needed, and sync with embedded index.

        Args:
            embedded_index_name: Name of the embedded vector index
            unembedded_index_name: Name of the unembedded source index
            embedder_type: Type of embedder for this collection
            document_loader: Function to transform raw documents
        """
        failed_migrations: list[BaseDocument] = []
        embedder: Embedder = self._embedder_factory.create_embedder(embedder_type)

        # Get all documents from unembedded index
        try:
            response: dict[str, Any] = await self.elasticsearch_client.search(
                index=unembedded_index_name,
                query={"match_all": {}},
                size=10000,
            )
            unembedded_docs: list[dict[str, Any]] = [
                hit["_source"] for hit in response["hits"]["hits"]
            ]
        except NotFoundError:
            unembedded_docs: list[dict[str, Any]] = []
        except Exception as e:
            self._logger.error(f"Failed to load unembedded documents: {e}")
            unembedded_docs: list[dict[str, Any]] = []

        indexing_required: bool = False

        if unembedded_docs:
            for doc in unembedded_docs:
                # Merge metadata back into document
                if "metadata" in doc:
                    metadata: dict[str, Any] = doc.pop("metadata")
                    doc.update(metadata)

                prospective_doc: BaseDocument = cast(BaseDocument, doc)
                try:
                    if loaded_doc := await document_loader(prospective_doc):
                        if loaded_doc != prospective_doc:
                            # Document was transformed, update unembedded index
                            prepared_doc: dict[str, Any] = self._prepare_document_for_indexing(
                                loaded_doc
                            )
                            await self.elasticsearch_client.index(
                                index=unembedded_index_name,
                                id=prospective_doc["id"],
                                document=prepared_doc,
                                refresh="wait_for",
                            )
                            indexing_required = True
                    else:
                        self._logger.warning(f'Failed to load document "{doc}"')
                        await self.elasticsearch_client.delete(
                            index=unembedded_index_name,
                            id=prospective_doc["id"],
                            refresh="wait_for",
                        )
                        failed_migrations.append(prospective_doc)

                except Exception as e:
                    self._logger.error(f"Failed to load document '{doc}'. error: {e}.")
                    failed_migrations.append(prospective_doc)

            # Store failed migrations
            if failed_migrations:
                failed_migrations_collection: ElasticsearchVectorCollection[
                    BaseDocument
                ] = await self.get_or_create_collection(
                    "failed_migrations", BaseDocument, NoOpEmbedder, identity_loader
                )

                for failed_doc in failed_migrations:
                    await failed_migrations_collection.insert_one(failed_doc)

        # Sync embedded index with unembedded index
        if indexing_required:
            await self._index_collection(embedded_index_name, unembedded_index_name, embedder)

    async def _index_collection(
        self,
        embedded_index_name: str,
        unembedded_index_name: str,
        embedder: Embedder,
    ) -> None:
        """
        Sync embedded collection with unembedded collection.

        Args:
            embedded_index_name: Name of the embedded vector index
            unembedded_index_name: Name of the unembedded source index
            embedder: Embedder instance for generating vectors
        """
        # Get all unembedded documents
        try:
            response: dict[str, Any] = await self.elasticsearch_client.search(
                index=unembedded_index_name,
                query={"match_all": {}},
                size=10000,
            )
            unembedded_docs_list: list[dict[str, Any]] = [
                hit["_source"] for hit in response["hits"]["hits"]
            ]
            unembedded_docs_by_id: dict[str, dict[str, Any]] = {
                doc["id"]: doc for doc in unembedded_docs_list
            }
        except NotFoundError:
            unembedded_docs_by_id: dict[str, dict[str, Any]] = {}

        # Get all embedded documents
        try:
            response: dict[str, Any] = await self.elasticsearch_client.search(
                index=embedded_index_name,
                query={"match_all": {}},
                size=10000,
                source={"excludes": ["content_vector"]},
            )
            embedded_docs: list[dict[str, Any]] = [
                hit["_source"] for hit in response["hits"]["hits"]
            ]
        except NotFoundError:
            embedded_docs: list[dict[str, Any]] = []

        # Remove docs from embedded that no longer exist in unembedded
        # Update embeddings for changed docs
        for doc in embedded_docs:
            doc_id: str = doc["id"]
            if doc_id not in unembedded_docs_by_id:
                await self.elasticsearch_client.delete(
                    index=embedded_index_name, id=doc_id, refresh="false"
                )
            else:
                if doc.get("checksum") != unembedded_docs_by_id[doc_id].get("checksum"):
                    # Document changed, re-embed
                    content: str = cast(str, unembedded_docs_by_id[doc_id]["content"])
                    embeddings: list[float] = list((await embedder.embed([content])).vectors[0])

                    prepared_doc: dict[str, Any] = unembedded_docs_by_id[doc_id].copy()
                    prepared_doc["content_vector"] = embeddings

                    await self.elasticsearch_client.index(
                        index=embedded_index_name,
                        id=doc_id,
                        document=prepared_doc,
                        refresh="false",
                    )
                unembedded_docs_by_id.pop(doc_id)

        # Add new docs from unembedded to embedded
        for doc in unembedded_docs_by_id.values():
            content: str = cast(str, doc["content"])
            embeddings: list[float] = list((await embedder.embed([content])).vectors[0])

            prepared_doc: dict[str, Any] = doc.copy()
            prepared_doc["content_vector"] = embeddings

            await self.elasticsearch_client.index(
                index=embedded_index_name,
                id=doc["id"],
                document=prepared_doc,
                refresh="false",
            )

        # Refresh once after all operations
        await self.elasticsearch_client.indices.refresh(index=embedded_index_name)

    def _prepare_document_for_indexing(self, document: TDocument) -> dict[str, Any]:
        """
        Prepare a document for indexing by separating core fields from metadata.

        Args:
            document: The document to prepare

        Returns:
            Dictionary ready for Elasticsearch indexing
        """
        core_fields: set[str] = {"id", "version", "content", "checksum"}

        if hasattr(document, "items"):
            doc_items: list[tuple[str, Any]] = list(document.items())
        else:
            doc_items: list[tuple[str, Any]] = [(k, document[k]) for k in document.keys()]

        core_data: dict[str, Any] = {k: v for k, v in doc_items if k in core_fields}
        metadata: dict[str, Any] = {k: v for k, v in doc_items if k not in core_fields}

        es_document: dict[str, Any] = core_data.copy()
        if metadata:
            es_document["metadata"] = metadata

        return es_document

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
    ) -> ElasticsearchVectorCollection[TDocument]:
        """
        Create a new vector collection (Elasticsearch indices).

        Args:
            name: Collection name
            schema: Document schema type
            embedder_type: Type of embedder for generating vectors

        Returns:
            New ElasticsearchVectorCollection instance
        """
        if name in self._collections:
            raise ValueError(f'Collection "{name}" already exists.')

        embedded_index_name: str = self._get_embedded_index_name(name, embedder_type)
        unembedded_index_name: str = self._get_unembedded_index_name(name)

        await self._ensure_unembedded_index_exists(unembedded_index_name)
        await self._ensure_embedded_index_exists(embedded_index_name, embedder_type)

        embedder: Embedder = self._embedder_factory.create_embedder(embedder_type)

        collection: ElasticsearchVectorCollection[TDocument] = ElasticsearchVectorCollection(
            database=self,
            name=name,
            embedded_index_name=embedded_index_name,
            unembedded_index_name=unembedded_index_name,
            schema=schema,
            embedder=embedder,
        )

        self._collections[name] = collection
        return cast(ElasticsearchVectorCollection[TDocument], collection)

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> ElasticsearchVectorCollection[TDocument]:
        """
        Get an existing vector collection.

        Args:
            name: Collection name
            schema: Document schema type
            embedder_type: Type of embedder for generating vectors
            document_loader: Function to transform raw documents

        Returns:
            Existing ElasticsearchVectorCollection instance

        Raises:
            ValueError: If collection doesn't exist
        """
        if collection := self._collections.get(name):
            return cast(ElasticsearchVectorCollection[TDocument], collection)

        unembedded_index_name: str = self._get_unembedded_index_name(name)
        embedded_index_name: str = self._get_embedded_index_name(name, embedder_type)

        try:
            unembedded_exists: bool = await self.elasticsearch_client.indices.exists(
                index=unembedded_index_name
            )
            if unembedded_exists:
                # Ensure embedded index exists
                embedded_exists: bool = await self.elasticsearch_client.indices.exists(
                    index=embedded_index_name
                )
                if not embedded_exists:
                    await self._ensure_embedded_index_exists(embedded_index_name, embedder_type)

                # Load and migrate documents
                await self._load_collection_documents(
                    embedded_index_name,
                    unembedded_index_name,
                    embedder_type,
                    document_loader,
                )

                # Index collection to sync
                embedder: Embedder = self._embedder_factory.create_embedder(embedder_type)
                await self._index_collection(embedded_index_name, unembedded_index_name, embedder)

                collection: ElasticsearchVectorCollection[TDocument] = (
                    ElasticsearchVectorCollection(
                        database=self,
                        name=name,
                        embedded_index_name=embedded_index_name,
                        unembedded_index_name=unembedded_index_name,
                        schema=schema,
                        embedder=embedder,
                    )
                )

                self._collections[name] = collection
                return cast(ElasticsearchVectorCollection[TDocument], collection)

        except Exception as e:
            self._logger.error(f"Failed to get collection {name}: {e}")

        raise ValueError(f'Collection "{name}" does not exist')

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> ElasticsearchVectorCollection[TDocument]:
        """
        Get an existing collection or create a new one.

        Args:
            name: Collection name
            schema: Document schema type
            embedder_type: Type of embedder for generating vectors
            document_loader: Function to transform raw documents

        Returns:
            ElasticsearchVectorCollection instance
        """
        if collection := self._collections.get(name):
            assert schema == collection._schema
            return cast(ElasticsearchVectorCollection[TDocument], collection)

        try:
            return await self.get_collection(name, schema, embedder_type, document_loader)
        except ValueError:
            return await self.create_collection(name, schema, embedder_type)

    @override
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        """
        Delete a collection (drop both embedded and unembedded Elasticsearch indices).

        Args:
            name: Collection name

        Raises:
            ValueError: If collection doesn't exist
        """
        pattern: str = f"{self.index_prefix}_{name}_*"

        try:
            indices_response: list[dict[str, Any]] = await self.elasticsearch_client.cat.indices(
                index=pattern, format="json", h="index"
            )

            if indices_response:
                for index_info in indices_response:
                    index_name: str = index_info["index"]
                    await self.elasticsearch_client.indices.delete(index=index_name)
                    self._logger.info(f"Deleted Elasticsearch index: {index_name}")

                if name in self._collections:
                    del self._collections[name]
                return

        except Exception as e:
            self._logger.error(f"Failed to delete collection {name}: {e}")
            raise

        raise ValueError(f'Collection "{name}" does not exist')

    @override
    async def upsert_metadata(
        self,
        key: str,
        value: JSONSerializable,
    ) -> None:
        """Store metadata in the metadata index."""
        await self._ensure_metadata_index_exists()

        try:
            await self.elasticsearch_client.index(
                index=self._metadata_index,
                id=key,
                document={"key": key, "value": value},
                refresh="wait_for",
            )
        except Exception as e:
            self._logger.error(f"Failed to upsert metadata {key}: {e}")
            raise

    @override
    async def remove_metadata(
        self,
        key: str,
    ) -> None:
        """Remove metadata from the metadata index."""
        try:
            await self.elasticsearch_client.delete(index=self._metadata_index, id=key)
        except NotFoundError:
            pass
        except Exception as e:
            self._logger.error(f"Failed to remove metadata {key}: {e}")
            raise

    @override
    async def read_metadata(
        self,
    ) -> dict[str, JSONSerializable]:
        """Read all metadata from the metadata index."""
        # Ensure metadata index exists before reading
        await self._ensure_metadata_index_exists()

        try:
            response: dict[str, Any] = await self.elasticsearch_client.search(
                index=self._metadata_index,
                query={"match_all": {}},
                size=10000,
            )

            metadata: dict[str, JSONSerializable] = {}
            for hit in response["hits"]["hits"]:
                source: dict[str, Any] = hit["_source"]
                metadata[source["key"]] = source["value"]

            return metadata

        except NotFoundError:
            # Index doesn't exist yet - return empty metadata
            return {}
        except Exception as e:
            self._logger.error(f"Failed to read metadata: {e}")
            raise


class ElasticsearchVectorCollection(VectorCollection[TDocument]):
    """
    An Elasticsearch implementation of VectorCollection.

    This class provides vector document operations (CRUD + similarity search)
    backed by dual Elasticsearch indices (embedded + unembedded).
    """

    def __init__(
        self,
        database: ElasticsearchVectorDatabase,
        name: str,
        embedded_index_name: str,
        unembedded_index_name: str,
        schema: type[TDocument],
        embedder: Embedder,
        refresh_policy: str = "wait_for",
    ) -> None:
        """
        Initialize the Elasticsearch vector collection.

        Args:
            database: Parent ElasticsearchVectorDatabase instance
            name: Collection name
            embedded_index_name: Elasticsearch embedded vector index name
            unembedded_index_name: Elasticsearch unembedded index name (source of truth)
            schema: Document schema type
            embedder: Embedder instance for generating vectors
            refresh_policy: Refresh policy for write operations
        """
        self._database: ElasticsearchVectorDatabase = database
        self._name: str = name
        self._embedded_index_name: str = embedded_index_name
        self._unembedded_index_name: str = unembedded_index_name
        self._schema: type[TDocument] = schema
        self._embedder: Embedder = embedder
        self._refresh_policy: str = refresh_policy
        self._field_cache: dict[str, str] = {}
        self._version: int = 1

    @property
    def elasticsearch_client(self) -> AsyncElasticsearch:
        """Get the Elasticsearch client from the parent database."""
        return self._database.elasticsearch_client

    @property
    def similarity_metric(self) -> str:
        """Get the similarity metric for this collection."""
        return "l2_norm" if isinstance(self._embedder, NoOpEmbedder) else "cosine"

    async def _get_field_mapping(self, field_name: str) -> str:
        """
        Get the correct field path for querying using Field Capabilities API.

        Args:
            field_name: Base field name

        Returns:
            Optimal field path (either field_name or field_name.keyword)
        """
        if field_name not in self._field_cache:
            try:
                response: dict[str, Any] = await self.elasticsearch_client.field_caps(
                    index=self._embedded_index_name, fields=[field_name, f"{field_name}.keyword"]
                )
                # Prefer .keyword for exact matches if available
                if f"{field_name}.keyword" in response.get("fields", {}):
                    self._field_cache[field_name] = f"{field_name}.keyword"
                else:
                    self._field_cache[field_name] = field_name
            except Exception:
                self._field_cache[field_name] = field_name

        return self._field_cache[field_name]

    def _validate_vector_dimensions(self, vector: list[float], operation_name: str) -> None:
        """
        Validate that vector dimensions match the expected size.

        Args:
            vector: The vector to validate
            operation_name: Name of the operation for error messages

        Raises:
            ValueError: If vector dimensions do not match
        """
        if len(vector) != self._embedder.dimensions:
            raise ValueError(
                f"{operation_name} failed: Vector dimension mismatch. "
                f"Expected {self._embedder.dimensions} dimensions, "
                f"got {len(vector)} dimensions."
            )

    def _prepare_document_for_indexing(self, document: TDocument) -> dict[str, Any]:
        """
        Prepare a document for indexing by separating core fields from metadata.

        Args:
            document: The document to prepare

        Returns:
            Dictionary ready for Elasticsearch indexing
        """
        return self._database._prepare_document_for_indexing(document)

    async def _get_content_vector(self, content: str) -> list[float]:
        """
        Get the vector embedding for content with dimension validation, using cache if available.

        Args:
            content: Text content to embed

        Returns:
            Vector embedding as list of floats
        """
        if e := await self._database._embedding_cache_provider().get(
            embedder_type=type(self._embedder),
            texts=[content],
        ):
            vector: list[float] = list(e.vectors[0])
        else:
            embedding_result: EmbeddingResult = await self._embedder.embed([content])
            vector: list[float] = embedding_result.vectors[0]

            await self._database._embedding_cache_provider().set(
                embedder_type=type(self._embedder),
                texts=[content],
                vectors=[vector],
            )

        self._validate_vector_dimensions(vector, "Embedding generation")
        return vector

    async def _calculate_optimal_num_candidates(self, k: int) -> int:
        """
        Calculate optimal num_candidates based on index size.

        Args:
            k: Number of results requested

        Returns:
            Optimal num_candidates value (always >= k)
        """
        try:
            count: dict[str, Any] = await self.elasticsearch_client.count(
                index=self._embedded_index_name
            )
            doc_count: int = count["count"]

            if doc_count < 1000:
                return max(k, min(doc_count, k * 10))
            elif doc_count < 10000:
                return max(k, min(doc_count, k * 50))
            else:
                return max(k, min(doc_count, max(k * 100, 10000)))
        except Exception:
            return max(k * 50, 100, k)

    def _convert_score_to_distance(self, score: float) -> float:
        """
        Convert ES similarity score to distance based on similarity metric.

        Args:
            score: Elasticsearch similarity score

        Returns:
            Distance value (lower = more similar)
        """
        if self.similarity_metric == "cosine":
            return max(0.0, min(2.0, 1.0 - score))
        elif self.similarity_metric == "dot_product":
            return 1.0 / (1.0 + max(0.0, score))
        elif self.similarity_metric == "l2_norm":
            return max(0.0, score)
        else:
            return score

    def _translate_where_to_elasticsearch_query(self, where: Where) -> dict[str, Any]:
        """
        Translate a Where filter to an Elasticsearch query.

        Args:
            where: Where filter specification

        Returns:
            Elasticsearch query dictionary
        """
        if not where:
            return {"match_all": {}}

        if next(iter(where.keys())) in ("$and", "$or"):
            logical_op: LogicalOperator = cast(LogicalOperator, where)

            if "$and" in logical_op:
                sub_queries: list[dict[str, Any]] = [
                    self._translate_where_to_elasticsearch_query(sub_filter)
                    for sub_filter in logical_op["$and"]
                ]
                return {"bool": {"must": sub_queries}}

            elif "$or" in logical_op:
                sub_queries: list[dict[str, Any]] = [
                    self._translate_where_to_elasticsearch_query(sub_filter)
                    for sub_filter in logical_op["$or"]
                ]
                return {"bool": {"should": sub_queries}}

        field_filters: WhereExpression = cast(WhereExpression, where)
        must_clauses: list[dict[str, Any]] = []

        core_keyword_fields: set[str] = {"id", "version", "checksum"}
        content_field: str = "content"

        for field_name, field_filter in field_filters.items():
            # Determine field path
            if field_name in core_keyword_fields:
                es_field: str = field_name
            elif field_name == content_field:
                es_field: str = field_name
            else:
                es_field: str = f"metadata.{field_name}"

            for operator, filter_value in field_filter.items():
                if operator == "$eq":
                    must_clauses.append({"term": {es_field: filter_value}})
                elif operator == "$ne":
                    must_clauses.append({"bool": {"must_not": {"term": {es_field: filter_value}}}})
                elif operator == "$gt":
                    must_clauses.append({"range": {es_field: {"gt": filter_value}}})
                elif operator == "$gte":
                    must_clauses.append({"range": {es_field: {"gte": filter_value}}})
                elif operator == "$lt":
                    must_clauses.append({"range": {es_field: {"lt": filter_value}}})
                elif operator == "$lte":
                    must_clauses.append({"range": {es_field: {"lte": filter_value}}})
                elif operator == "$in":
                    filter_values: list[LiteralValue] = cast(list[LiteralValue], filter_value)
                    must_clauses.append({"terms": {es_field: filter_values}})
                elif operator == "$nin":
                    filter_values: list[LiteralValue] = cast(list[LiteralValue], filter_value)
                    must_clauses.append(
                        {"bool": {"must_not": {"terms": {es_field: filter_values}}}}
                    )

        if len(must_clauses) == 1:
            return must_clauses[0]
        elif len(must_clauses) > 1:
            return {"bool": {"must": must_clauses}}
        else:
            return {"match_all": {}}

    @override
    async def find(
        self,
        filters: Where,
    ) -> Sequence[TDocument]:
        """
        Find documents matching the given filters.

        Args:
            filters: Where filter specification

        Returns:
            Sequence of matching documents
        """
        query: dict[str, Any] = self._translate_where_to_elasticsearch_query(filters)

        try:
            response: dict[str, Any] = await self.elasticsearch_client.search(
                index=self._embedded_index_name,
                query=query,
                size=10000,
                source={"excludes": ["content_vector"]},
            )

            documents: list[TDocument] = []
            for hit in response["hits"]["hits"]:
                doc: dict[str, Any] = hit["_source"]
                if "metadata" in doc:
                    metadata: dict[str, Any] = doc.pop("metadata")
                    doc.update(metadata)
                documents.append(cast(TDocument, doc))

            return documents
        except NotFoundError:
            return []
        except Exception as e:
            self._database._logger.error(
                f"Search failed in vector index {self._embedded_index_name}: {e}"
            )
            return []

    @override
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]:
        """
        Find the first document matching the given filters.

        Args:
            filters: Where filter specification

        Returns:
            First matching document or None
        """
        query: dict[str, Any] = self._translate_where_to_elasticsearch_query(filters)

        try:
            response: dict[str, Any] = await self.elasticsearch_client.search(
                index=self._embedded_index_name,
                query=query,
                size=1,
                source={"excludes": ["content_vector"]},
            )

            hits: list[dict[str, Any]] = response["hits"]["hits"]
            if hits:
                doc: dict[str, Any] = hits[0]["_source"]
                if "metadata" in doc:
                    metadata: dict[str, Any] = doc.pop("metadata")
                    doc.update(metadata)
                return cast(TDocument, doc)
            return None
        except NotFoundError:
            return None
        except Exception as e:
            self._database._logger.error(
                f"Find one failed in vector index {self._embedded_index_name}: {e}"
            )
            return None

    async def _insert_with_retry(self, document: TDocument, max_retries: int = 3) -> InsertResult:
        """
        Insert a single document with retry logic for transient failures.

        Args:
            document: Document to insert
            max_retries: Maximum number of retry attempts

        Returns:
            Insert operation result
        """
        ensure_is_total(document, self._schema)
        retry_delay: int = 1

        for attempt in range(max_retries):
            try:
                es_document: dict[str, Any] = self._prepare_document_for_indexing(document)

                if "content" in es_document and es_document["content"]:
                    content_vector: list[float] = await self._get_content_vector(
                        es_document["content"]
                    )
                    es_document["content_vector"] = content_vector

                doc_id: Optional[str] = (
                    document.get("id")
                    if hasattr(document, "get")
                    else document["id"]
                    if "id" in document
                    else None
                )

                self._version += 1

                # Insert into unembedded index (source of truth)
                if doc_id:
                    await self.elasticsearch_client.index(
                        index=self._unembedded_index_name,
                        id=doc_id,
                        document=self._prepare_document_for_indexing(document),
                        refresh=self._refresh_policy,
                    )

                    # Insert into embedded index with vector
                    await self.elasticsearch_client.index(
                        index=self._embedded_index_name,
                        id=doc_id,
                        document=es_document,
                        refresh=self._refresh_policy,
                    )
                else:
                    await self.elasticsearch_client.index(
                        index=self._unembedded_index_name,
                        document=self._prepare_document_for_indexing(document),
                        refresh=self._refresh_policy,
                    )
                    await self.elasticsearch_client.index(
                        index=self._embedded_index_name,
                        document=es_document,
                        refresh=self._refresh_policy,
                    )

                return InsertResult(acknowledged=True)

            except ConnectionTimeout:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))
                    continue
                raise

            except ConflictError:
                return InsertResult(acknowledged=True)

            except Exception as e:
                self._database._logger.error(
                    f"Insert failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    return InsertResult(acknowledged=False)
                await asyncio.sleep(retry_delay)

        return InsertResult(acknowledged=False)

    @override
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult:
        """
        Insert a single document with vector embedding.

        Args:
            document: Document to insert

        Returns:
            Insert operation result
        """
        return await self._insert_with_retry(document)

    async def insert_many(
        self, documents: Sequence[TDocument], chunk_size: int = 500
    ) -> InsertResult:
        """
        Bulk insert documents with automatic chunking.

        Args:
            documents: Sequence of documents to insert
            chunk_size: Number of documents per bulk request

        Returns:
            Insert operation result
        """

        async def doc_generator():
            for doc in documents:
                ensure_is_total(doc, self._schema)
                es_doc: dict[str, Any] = self._prepare_document_for_indexing(doc)

                if "content" in es_doc and es_doc["content"]:
                    es_doc["content_vector"] = await self._get_content_vector(es_doc["content"])

                doc_id: Optional[str] = doc.get("id") if hasattr(doc, "get") else doc.get("id")

                # Yield for unembedded index
                yield {
                    "_index": self._unembedded_index_name,
                    "_id": doc_id,
                    "_source": self._prepare_document_for_indexing(doc),
                }

                # Yield for embedded index
                yield {"_index": self._embedded_index_name, "_id": doc_id, "_source": es_doc}

        try:
            await async_bulk(
                self.elasticsearch_client,
                doc_generator(),
                chunk_size=chunk_size,
                refresh=self._refresh_policy,
            )
            return InsertResult(acknowledged=True)
        except Exception as e:
            self._database._logger.error(f"Bulk insert failed: {e}")
            return InsertResult(acknowledged=False)

    @override
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        """
        Update the first document matching the given filters.

        Args:
            filters: Where filter specification
            params: Update parameters
            upsert: Whether to insert if no document matches

        Returns:
            Update operation result
        """
        existing_doc: Optional[TDocument] = await self.find_one(filters)

        if existing_doc:
            updated_doc: TDocument = cast(TDocument, {**existing_doc, **params})

            try:
                es_document: dict[str, Any] = self._prepare_document_for_indexing(updated_doc)

                if "content" in params and params["content"]:
                    content_vector: list[float] = await self._get_content_vector(params["content"])
                    es_document["content_vector"] = content_vector
                elif "content" in es_document and es_document["content"]:
                    content_vector: list[float] = await self._get_content_vector(
                        es_document["content"]
                    )
                    es_document["content_vector"] = content_vector

                doc_id: Optional[str] = (
                    existing_doc.get("id")
                    if hasattr(existing_doc, "get")
                    else existing_doc["id"]
                    if "id" in existing_doc
                    else None
                )

                if doc_id:
                    self._version += 1

                    # Update unembedded index
                    await self.elasticsearch_client.index(
                        index=self._unembedded_index_name,
                        id=doc_id,
                        document=self._prepare_document_for_indexing(updated_doc),
                        refresh=self._refresh_policy,
                    )

                    # Update embedded index
                    await self.elasticsearch_client.index(
                        index=self._embedded_index_name,
                        id=doc_id,
                        document=es_document,
                        refresh=self._refresh_policy,
                    )

                    return UpdateResult(
                        acknowledged=True,
                        matched_count=1,
                        modified_count=1,
                        updated_document=updated_doc,
                    )
                else:
                    self._database._logger.error("Cannot update document without ID")
                    return UpdateResult(
                        acknowledged=False,
                        matched_count=0,
                        modified_count=0,
                        updated_document=None,
                    )
            except Exception as e:
                self._database._logger.error(
                    f"Update failed in vector index {self._embedded_index_name}: {e}"
                )
                return UpdateResult(
                    acknowledged=False,
                    matched_count=1,
                    modified_count=0,
                    updated_document=None,
                )
        elif upsert:
            ensure_is_total(document=params, schema=self._schema)
            await self.insert_one(params)
            return UpdateResult(
                acknowledged=True,
                matched_count=0,
                modified_count=0,
                updated_document=params,
            )
        else:
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
        """
        Delete the first document matching the given filters.

        Args:
            filters: Where filter specification

        Returns:
            Delete operation result
        """
        existing_doc: Optional[TDocument] = await self.find_one(filters)

        if existing_doc:
            deleted_doc: TDocument = cast(TDocument, existing_doc)
            try:
                doc_id: Optional[str] = (
                    deleted_doc.get("id")
                    if hasattr(deleted_doc, "get")
                    else deleted_doc["id"]
                    if "id" in deleted_doc
                    else None
                )
                if doc_id:
                    self._version += 1

                    # Delete from unembedded index
                    await self.elasticsearch_client.delete(
                        index=self._unembedded_index_name,
                        id=doc_id,
                        refresh=self._refresh_policy,
                    )

                    # Delete from embedded index
                    await self.elasticsearch_client.delete(
                        index=self._embedded_index_name,
                        id=doc_id,
                        refresh=self._refresh_policy,
                    )

                    return DeleteResult(
                        acknowledged=True,
                        deleted_count=1,
                        deleted_document=deleted_doc,
                    )
                else:
                    self._database._logger.error("Cannot delete document without ID")
                    return DeleteResult(
                        acknowledged=False,
                        deleted_count=0,
                        deleted_document=None,
                    )
            except Exception as e:
                self._database._logger.error(
                    f"Delete failed in vector index {self._embedded_index_name}: {e}"
                )
                return DeleteResult(
                    acknowledged=False,
                    deleted_count=0,
                    deleted_document=None,
                )
        else:
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
        """
        Find documents similar to the given query using vector similarity search.

        Args:
            filters: Where filter specification for filtering results
            query: Query text to find similar documents for
            k: Number of similar documents to return

        Returns:
            Sequence of similar documents with similarity scores
        """
        try:
            query_vector: list[float] = await self._get_content_vector(query)
            self._validate_vector_dimensions(query_vector, "Vector similarity search")

            num_candidates: int = await self._calculate_optimal_num_candidates(k)

            # Use dedicated kNN endpoint when no filters
            if not filters or filters == {}:
                response: dict[str, Any] = await self.elasticsearch_client.knn_search(
                    index=self._embedded_index_name,
                    knn={
                        "field": "content_vector",
                        "query_vector": query_vector,
                        "k": k,
                        "num_candidates": num_candidates,
                    },
                    source={"excludes": ["content_vector"]},
                )
            else:
                # Use regular search with knn + filter
                filter_query: dict[str, Any] = self._translate_where_to_elasticsearch_query(filters)
                response: dict[str, Any] = await self.elasticsearch_client.search(
                    index=self._embedded_index_name,
                    knn={
                        "field": "content_vector",
                        "query_vector": query_vector,
                        "k": k,
                        "num_candidates": num_candidates,
                        "filter": filter_query,
                    },
                    size=k,
                    source={"excludes": ["content_vector"]},
                )

            results: list[SimilarDocumentResult[TDocument]] = []
            for hit in response["hits"]["hits"]:
                doc: dict[str, Any] = hit["_source"]
                if "metadata" in doc:
                    metadata: dict[str, Any] = doc.pop("metadata")
                    doc.update(metadata)

                distance: float = self._convert_score_to_distance(hit["_score"])
                similarity_result: SimilarDocumentResult[TDocument] = SimilarDocumentResult(
                    document=cast(TDocument, doc), distance=distance
                )
                results.append(similarity_result)

            return results

        except NotFoundError:
            self._database._logger.info(
                f"Index {self._embedded_index_name} not found, returning empty results"
            )
            return []
        except ESConnectionError as e:
            self._database._logger.error(
                f"Connection error during vector search in {self._embedded_index_name}: {e}"
            )
            raise
        except RequestError as e:
            self._database._logger.error(
                f"Invalid request during vector search in {self._embedded_index_name}: {e.info}"
            )
            return []
        except TransportError as e:
            self._database._logger.error(
                f"Transport error during vector search in {self._embedded_index_name}: {e}"
            )
            return []
        except ValueError as e:
            self._database._logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            self._database._logger.error(
                f"Unexpected error during vector search in {self._embedded_index_name}: {e}",
                exc_info=True,
            )
            return []

    async def hybrid_search(
        self,
        query: str,
        text_weight: float = 0.3,
        vector_weight: float = 0.7,
        k: int = 10,
        filters: Where = None,
    ) -> Sequence[SimilarDocumentResult[TDocument]]:
        """
        Perform hybrid search combining text and vector similarity.

        Args:
            query: Query text
            text_weight: Weight for text matching (0.0-1.0)
            vector_weight: Weight for vector similarity (0.0-1.0)
            k: Number of results to return
            filters: Optional filters

        Returns:
            Sequence of similar documents with scores
        """
        query_vector: list[float] = await self._get_content_vector(query)

        search_query: dict[str, Any] = {
            "bool": {"should": [{"match": {"content": {"query": query, "boost": text_weight}}}]}
        }

        if filters:
            filter_query: dict[str, Any] = self._translate_where_to_elasticsearch_query(filters)
            search_query["bool"]["filter"] = filter_query

        response: dict[str, Any] = await self.elasticsearch_client.search(
            index=self._embedded_index_name,
            query=search_query,
            knn={
                "field": "content_vector",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": await self._calculate_optimal_num_candidates(k),
                "boost": vector_weight,
            },
            size=k,
            source={"excludes": ["content_vector"]},
        )

        results: list[SimilarDocumentResult[TDocument]] = []
        for hit in response["hits"]["hits"]:
            doc: dict[str, Any] = hit["_source"]
            if "metadata" in doc:
                metadata: dict[str, Any] = doc.pop("metadata")
                doc.update(metadata)

            distance: float = self._convert_score_to_distance(hit["_score"])
            results.append(SimilarDocumentResult(document=cast(TDocument, doc), distance=distance))

        return results

    async def find_paginated(
        self, filters: Where, page_size: int = 100, search_after: Optional[list] = None
    ) -> tuple[Sequence[TDocument], Optional[list]]:
        """
        Find documents with efficient pagination using search_after.

        Args:
            filters: Where filter specification
            page_size: Number of documents per page
            search_after: Pagination cursor from previous request

        Returns:
            Tuple of (documents, next_search_after_cursor)
        """
        query: dict[str, Any] = self._translate_where_to_elasticsearch_query(filters)

        search_params: dict[str, Any] = {
            "index": self._embedded_index_name,
            "query": query,
            "size": page_size,
            "sort": [{"id": "asc"}],  # Use document id field instead of _id (avoids fielddata requirement)
            "source": {"excludes": ["content_vector"]},
        }

        if search_after:
            search_params["search_after"] = search_after

        response: dict[str, Any] = await self.elasticsearch_client.search(**search_params)

        documents: list[TDocument] = []
        last_sort: Optional[list] = None

        for hit in response["hits"]["hits"]:
            doc: dict[str, Any] = hit["_source"]
            if "metadata" in doc:
                metadata: dict[str, Any] = doc.pop("metadata")
                doc.update(metadata)
            documents.append(cast(TDocument, doc))
            last_sort: Optional[list] = hit["sort"]

        return documents, last_sort
