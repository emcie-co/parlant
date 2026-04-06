# Copyright 2025 Werktøj Aps
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
import os
from typing import Any, Awaitable, Callable, Optional, Sequence, cast
from typing_extensions import override, Self

from parlant.core.persistence.common import (
    Cursor,
    ObjectId,
    SortDirection,
    Where,
    WhereExpression,
    LogicalOperator,
    ensure_is_total,
)
from parlant.core.persistence.document_database import (
    BaseDocument,
    DeleteResult,
    DocumentCollection,
    DocumentDatabase,
    FindResult,
    InsertResult,
    TDocument,
    UpdateResult,
    identity_loader,
)
from parlant.core.loggers import Logger

try:
    from elasticsearch import AsyncElasticsearch
    from elasticsearch.exceptions import NotFoundError, RequestError
except ImportError:
    raise ImportError(
        "elasticsearch package is required for ElasticsearchDocumentDatabase. "
        "Install it with: pip install elasticsearch"
    )


def _get_elasticsearch_connection_url() -> str:
    """Build Elasticsearch connection URL from environment variables."""
    host: str = os.environ.get("ELASTICSEARCH__HOST", "localhost")
    port: int = int(os.environ.get("ELASTICSEARCH__PORT", "9200"))
    username: Optional[str] = os.environ.get("ELASTICSEARCH__USERNAME")
    password: Optional[str] = os.environ.get("ELASTICSEARCH__PASSWORD")
    use_ssl: bool = os.environ.get("ELASTICSEARCH__USE_SSL", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    scheme: str = "https" if use_ssl else "http"
    if username and password:
        return f"{scheme}://{username}:{password}@{host}:{port}"
    return f"{scheme}://{host}:{port}"


def create_elasticsearch_document_client_from_env() -> AsyncElasticsearch:
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

    return AsyncElasticsearch(
        hosts=[_get_elasticsearch_connection_url()],
        request_timeout=timeout,
        max_retries=max_retries,
        retry_on_timeout=retry_on_timeout,
        verify_certs=verify_certs,
        http_compress=True,
        connections_per_node=10,
    )


def get_elasticsearch_document_index_settings_from_env() -> dict[str, Any]:
    """
    Get Elasticsearch index settings for document storage from environment variables.

    Environment variables:
        ELASTICSEARCH__NUMBER_OF_SHARDS: Number of primary shards (default: 1)
        ELASTICSEARCH__NUMBER_OF_REPLICAS: Number of replica shards (default: 0)
        ELASTICSEARCH__REFRESH_INTERVAL: Index refresh interval (default: 1s)
        ELASTICSEARCH__CODEC: Index codec for compression (default: best_compression)

    Returns:
        Dictionary of index settings
    """
    number_of_shards: int = int(os.environ.get("ELASTICSEARCH__NUMBER_OF_SHARDS", "1"))
    number_of_replicas: int = int(os.environ.get("ELASTICSEARCH__NUMBER_OF_REPLICAS", "0"))
    refresh_interval: str = os.environ.get("ELASTICSEARCH__REFRESH_INTERVAL", "1s")
    codec: str = os.environ.get("ELASTICSEARCH__CODEC", "best_compression")

    return {
        "number_of_shards": number_of_shards,
        "number_of_replicas": number_of_replicas,
        "index.refresh_interval": refresh_interval,
        "index.codec": codec,
        "index.mapping.total_fields.limit": 2000,
        "index.max_result_window": 10000,
        "index.translog.durability": "request",  # Safer for document storage
    }


def get_elasticsearch_document_index_prefix_from_env() -> str:
    """
    Get Elasticsearch index prefix for document storage from environment variables.

    Environment variables:
        ELASTICSEARCH__INDEX_PREFIX: Index prefix (default: parlant)

    Returns:
        Index prefix string
    """
    return os.environ.get("ELASTICSEARCH__INDEX_PREFIX", "parlant")


class ElasticsearchDocumentDatabase(DocumentDatabase):
    """
    An Elasticsearch implementation of the DocumentDatabase interface.

    This adapter allows using Elasticsearch as the persistence layer for Parlant documents.
    Collections are mapped to Elasticsearch indices with appropriate mappings.
    """

    def __init__(
        self,
        elasticsearch_client: AsyncElasticsearch,
        index_prefix: str,
        logger: Logger,
        store_context: str | None = None,
    ) -> None:
        """
        Initialize the Elasticsearch document database.

        Args:
            elasticsearch_client: Configured AsyncElasticsearch client
            index_prefix: Prefix for all Elasticsearch indices created by this database
            logger: Logger instance for error reporting
            store_context: Optional store context for metadata collection naming (e.g., "session", "customer")
        """
        self.elasticsearch_client = elasticsearch_client
        self.index_prefix = index_prefix
        self._logger = logger
        self._collections: dict[str, ElasticsearchDocumentCollection[BaseDocument]] = {}
        # Use store context for metadata naming (each store gets its own metadata collection)
        self._metadata_suffix = store_context

    def _get_index_name(self, collection_name: str) -> str:
        """Generate the full Elasticsearch index name for a collection.

        Elasticsearch requires index names to be lowercase, so we normalize the name here.
        """
        return f"{self.index_prefix}_{collection_name}".lower()

    async def _ensure_index_exists(self, index_name: str, schema: type[TDocument]) -> None:
        """
        Ensure the Elasticsearch index exists with appropriate mappings.

        Args:
            index_name: Name of the Elasticsearch index
            schema: Document schema type for mapping generation
        """
        try:
            exists = await self.elasticsearch_client.indices.exists(index=index_name)
            if exists:
                # Check if the existing mapping is correct, and update if needed
                await self._ensure_correct_mapping(index_name)
            else:
                # Create index with comprehensive mapping for common fields
                mapping = {
                    "mappings": {
                        "properties": {
                            "id": {"type": "keyword"},
                            "version": {"type": "keyword"},
                            # Session and event related fields
                            "session_id": {"type": "keyword"},
                            "event_id": {"type": "keyword"},
                            "agent_id": {"type": "keyword"},
                            "customer_id": {"type": "keyword"},
                            "correlation_id": {"type": "keyword"},
                            "source": {"type": "keyword"},
                            "kind": {"type": "keyword"},
                            # Context variable fields
                            "variable_id": {"type": "keyword"},
                            "key": {"type": "keyword"},
                            "tool_id": {"type": "keyword"},
                            # Guideline fields
                            "guideline_id": {"type": "keyword"},
                            "tag_id": {"type": "keyword"},
                            # Other common ID fields
                            "evaluation_id": {"type": "keyword"},
                            # Embedding cache fields (store without indexing)
                            "vectors": {"type": "object", "enabled": False},
                            # Text fields that should be both searchable and exact-match
                            "name": {
                                "type": "text",
                                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                            },
                            "condition": {
                                "type": "text",
                                "fields": {"keyword": {"type": "keyword", "ignore_above": 1024}},
                            },
                            "action": {
                                "type": "text",
                                "fields": {"keyword": {"type": "keyword", "ignore_above": 1024}},
                            },
                        }
                    },
                    "settings": {
                        "number_of_replicas": 0,  # Single-node deployment
                        "number_of_shards": 1,
                    },
                }
                await self.elasticsearch_client.indices.create(index=index_name, body=mapping)
                self._logger.info(f"Created Elasticsearch index: {index_name}")
        except RequestError as e:
            self._logger.error(f"Failed to create index {index_name}: {e}")
            raise
        except Exception as e:
            self._logger.warning(f"Failed to create index {index_name}: {e}")
            # Don't raise - allow graceful degradation

    async def _ensure_correct_mapping(self, index_name: str) -> None:
        """
        Ensure existing index has correct field mappings, especially for keyword fields.

        Args:
            index_name: Name of the Elasticsearch index to check
        """
        try:
            # Get current mapping
            mapping_response = await self.elasticsearch_client.indices.get_mapping(index=index_name)
            current_mapping = mapping_response[index_name]["mappings"]

            # Check if critical fields have wrong mappings
            properties = current_mapping.get("properties", {})

            # Fields that must be keywords
            critical_keyword_fields = [
                "id",
                "version",
                "session_id",
                "event_id",
                "agent_id",
                "customer_id",
                "correlation_id",
                "source",
                "kind",
                "variable_id",
                "key",
                "tool_id",
                "guideline_id",
                "tag_id",
                "evaluation_id",
            ]

            needs_update = False
            for field in critical_keyword_fields:
                if field in properties:
                    field_mapping = properties[field]
                    # If field is mapped as text instead of keyword, we need to update
                    if field_mapping.get("type") == "text" and "keyword" not in field_mapping.get(
                        "fields", {}
                    ):
                        needs_update = True
                        self._logger.warning(
                            f"Field '{field}' in index '{index_name}' has incorrect mapping: {field_mapping}"
                        )
                        break

            if needs_update:
                # Update mapping to add keyword subfields where needed
                mapping_update = {"properties": {}}

                for field in critical_keyword_fields:
                    if field in properties and properties[field].get("type") == "text":
                        # Add keyword subfield to existing text field
                        mapping_update["properties"][field] = {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                        }

                if mapping_update["properties"]:
                    await self.elasticsearch_client.indices.put_mapping(
                        index=index_name, body=mapping_update
                    )
                    self._logger.info(
                        f"Updated mapping for index '{index_name}' to add keyword subfields"
                    )

        except Exception as e:
            self._logger.warning(f"Failed to check/update mapping for index {index_name}: {e}")
            # Don't raise - this is a best-effort operation

    async def load_documents_with_loader(
        self,
        name: str,
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
        documents: Sequence[BaseDocument] | None = None,
    ) -> Sequence[TDocument]:
        """
        Load documents from Elasticsearch using the provided document loader.

        Args:
            name: Collection name
            document_loader: Function to transform raw documents
            documents: Optional list of specific documents to load

        Returns:
            Sequence of loaded documents
        """
        data: list[TDocument] = []
        failed_migrations: list[BaseDocument] = []

        if documents is not None:
            # Load specific documents
            collection_documents = documents
        else:
            # Load all documents from the index
            index_name = self._get_index_name(name)
            try:
                response = await self.elasticsearch_client.search(
                    index=index_name,
                    body={
                        "query": {"match_all": {}},
                        "size": 10000,
                    },  # Adjust size as needed
                )
                collection_documents = []
                for hit in response["hits"]["hits"]:
                    doc = hit["_source"]
                    # Ensure the Elasticsearch _id is available as 'id' field
                    if "id" not in doc:
                        doc["id"] = hit["_id"]
                    collection_documents.append(doc)
            except NotFoundError:
                # Index doesn't exist yet
                collection_documents = []
            except Exception as e:
                self._logger.error(f"Failed to search index {index_name}: {e}")
                collection_documents = []

        for doc in collection_documents:
            try:
                if loaded_doc := await document_loader(doc):
                    data.append(loaded_doc)
                else:
                    self._logger.warning(f'Failed to load document "{doc}"')
                    failed_migrations.append(doc)
            except Exception as e:
                self._logger.error(
                    f"Failed to load document '{doc}' with error: {e}. Added to failed migrations collection."
                )
                failed_migrations.append(doc)

        if failed_migrations:
            failed_migrations_collection = await self.get_or_create_collection(
                "failed_migrations", BaseDocument, identity_loader
            )

            for doc in failed_migrations:
                await failed_migrations_collection.insert_one(doc)

        return data

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> ElasticsearchDocumentCollection[TDocument]:
        """
        Create a new collection (Elasticsearch index).

        Args:
            name: Collection name
            schema: Document schema type

        Returns:
            New ElasticsearchDocumentCollection instance
        """
        index_name = self._get_index_name(name)
        await self._ensure_index_exists(index_name, schema)

        collection = ElasticsearchDocumentCollection(
            database=self,
            name=name,
            index_name=index_name,
            schema=schema,
        )

        self._collections[name] = collection
        return cast(ElasticsearchDocumentCollection[TDocument], collection)

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> ElasticsearchDocumentCollection[TDocument]:
        """
        Get an existing collection, applying document migrations like MongoDB.

        Args:
            name: Collection name
            schema: Document schema type
            document_loader: Function to transform raw documents

        Returns:
            Existing ElasticsearchDocumentCollection instance

        Raises:
            ValueError: If collection doesn't exist
        """
        if collection := self._collections.get(name):
            return cast(ElasticsearchDocumentCollection[TDocument], collection)

        index_name = self._get_index_name(name)
        try:
            exists = await self.elasticsearch_client.indices.exists(index=index_name)
            if not exists:
                raise ValueError(f'Collection "{name}" does not exist')

            # Load existing documents and apply migrations (like MongoDB does)
            failed_migrations_collection_name = f"{name}_failed_migrations"

            # Check if we have a failed migrations index and clean it up
            failed_migrations_index = self._get_index_name(failed_migrations_collection_name)
            try:
                if await self.elasticsearch_client.indices.exists(index=failed_migrations_index):
                    self._logger.info(f"Deleting old `{failed_migrations_collection_name}` index")
                    await self.elasticsearch_client.indices.delete(index=failed_migrations_index)
            except Exception:
                pass  # Ignore cleanup errors

            # Get all existing documents
            try:
                response = await self.elasticsearch_client.search(
                    index=index_name,
                    body={"query": {"match_all": {}}, "size": 10000},
                )
                existing_documents = []
                for hit in response["hits"]["hits"]:
                    doc = hit["_source"]
                    # Ensure the Elasticsearch _id is available as 'id' field
                    if "id" not in doc:
                        doc["id"] = hit["_id"]
                    existing_documents.append(doc)
            except Exception as e:
                self._logger.error(f"Failed to load existing documents from {index_name}: {e}")
                existing_documents = []

            failed_migration_collection: Optional[ElasticsearchDocumentCollection[TDocument]] = None

            # Process each document with the loader (migration)
            self._logger.info(f"Processing {len(existing_documents)} documents for migration")
            for doc in existing_documents:
                self._logger.info(f"Processing document: {doc}")
                try:
                    loaded_doc = await document_loader(doc)
                    self._logger.info(f"Document loader returned: {loaded_doc}")
                    if loaded_doc:
                        # Update the document in place (like MongoDB's replace_one)
                        doc_id = doc.get("id")
                        if doc_id:
                            await self.elasticsearch_client.index(
                                index=index_name, id=doc_id, body=loaded_doc, refresh=True
                            )
                        continue

                    # Document failed to load, move to failed migrations
                    if failed_migration_collection is None:
                        self._logger.warning(
                            f"Creating: `{failed_migrations_collection_name}` index to store failed migrations..."
                        )
                        failed_migration_collection = await self.create_collection(
                            failed_migrations_collection_name, schema
                        )

                    self._logger.warning(f'Failed to load document "{doc}"')
                    try:
                        # Try to insert the document as-is first
                        await failed_migration_collection.insert_one(doc)
                        self._logger.info(
                            "Successfully inserted failed document into failed migrations collection"
                        )
                    except Exception as insert_error:
                        # If schema validation fails, insert directly into Elasticsearch bypassing schema validation
                        self._logger.warning(
                            f"Schema validation failed for failed migration, inserting directly: {insert_error}"
                        )
                        try:
                            # Insert directly into Elasticsearch without schema validation
                            doc_id = doc.get("id", f"failed_{hash(str(doc))}")
                            await self.elasticsearch_client.index(
                                index=f"{self.index_prefix}_{failed_migrations_collection_name}",
                                id=doc_id,
                                body=doc,
                                refresh=True,
                            )
                            self._logger.info(
                                "Successfully inserted failed document directly into Elasticsearch"
                            )
                        except Exception as direct_insert_error:
                            self._logger.error(
                                f"Failed to insert document directly into Elasticsearch: {direct_insert_error}"
                            )
                    self._logger.info(f"Document ID from doc: {doc.get('id')}")

                    # Delete the failed document from the main index
                    doc_id = doc.get("id")
                    self._logger.info(
                        f"Attempting to delete document with ID: {doc_id} (type: {type(doc_id)})"
                    )
                    if doc_id:
                        try:
                            # Check if document exists before deleting
                            exists = await self.elasticsearch_client.exists(
                                index=index_name, id=doc_id
                            )
                            self._logger.info(f"Document exists in Elasticsearch: {exists}")

                            delete_result = await self.elasticsearch_client.delete(
                                index=index_name, id=doc_id, refresh=True
                            )
                            self._logger.info(f"Delete result: {delete_result}")
                            self._logger.info(
                                f"Deleted failed migration document with ID: {doc_id}"
                            )
                        except Exception as e:
                            self._logger.error(
                                f"Failed to delete document {doc_id} from main index: {e}"
                            )
                            pass  # Ignore delete errors for failed migrations

                except Exception as e:
                    if failed_migration_collection is None:
                        self._logger.warning(
                            f"Creating: `{failed_migrations_collection_name}` index to store failed migrations..."
                        )
                        failed_migration_collection = await self.create_collection(
                            failed_migrations_collection_name, schema
                        )

                    self._logger.error(
                        f"Failed to load document '{doc}' with error: {e}. Added to `{failed_migrations_collection_name}` index."
                    )
                    await failed_migration_collection.insert_one(doc)

                    # Delete the failed document from the main index
                    doc_id = doc.get("id")
                    if doc_id:
                        try:
                            await self.elasticsearch_client.delete(
                                index=index_name, id=doc_id, refresh=True
                            )
                            self._logger.info(
                                f"Deleted failed migration document with ID: {doc_id}"
                            )
                        except Exception as e:
                            self._logger.error(
                                f"Failed to delete document {doc_id} from main index: {e}"
                            )
                            pass  # Ignore delete errors for failed migrations

            # Create and cache the collection
            collection = ElasticsearchDocumentCollection(
                database=self,
                name=name,
                index_name=index_name,
                schema=schema,
            )
            self._collections[name] = collection
            return cast(ElasticsearchDocumentCollection[TDocument], collection)

        except ValueError:
            raise  # Re-raise collection not found errors
        except Exception as e:
            self._logger.error(f"Failed to get collection {name}: {e}")
            raise ValueError(f'Collection "{name}" does not exist')

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> ElasticsearchDocumentCollection[TDocument]:
        """
        Get an existing collection or create a new one (matches MongoDB behavior).

        Args:
            name: Collection name
            schema: Document schema type
            document_loader: Function to transform raw documents

        Returns:
            ElasticsearchDocumentCollection instance
        """

        # Each store has its own database instance with unique metadata collection
        if name == "metadata" and self._metadata_suffix:
            # Store-specific metadata for separate databases
            name = f"metadata_{self._metadata_suffix}"

        # Ensure the index exists with correct mapping before any operations
        index_name = self._get_index_name(name)
        await self._ensure_index_exists(index_name, schema)

        # Try to get existing collection first (with migrations)
        try:
            return await self.get_collection(name, schema, document_loader)
        except ValueError:
            # Collection doesn't exist, create it
            return await self.create_collection(name, schema)

    @override
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        """
        Delete a collection (drop Elasticsearch index).

        Args:
            name: Collection name

        Raises:
            ValueError: If collection doesn't exist
        """
        index_name = self._get_index_name(name)

        try:
            exists = await self.elasticsearch_client.indices.exists(index=index_name)
            if exists:
                await self.elasticsearch_client.indices.delete(index=index_name)
                if name in self._collections:
                    del self._collections[name]
                self._logger.info(f"Deleted Elasticsearch index: {index_name}")
                return
        except Exception as e:
            self._logger.error(f"Failed to delete index {index_name}: {e}")
            raise

        raise ValueError(f'Collection "{name}" does not exist')

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        """Exit async context manager."""
        return False


def _translate_where_to_elasticsearch_query(where: Where) -> dict[str, Any]:
    """
    Translate a Where filter to an Elasticsearch query.

    Args:
        where: Where filter specification

    Returns:
        Elasticsearch query dictionary
    """
    if not where:
        return {"match_all": {}}

    # Check if this is a logical operator
    if next(iter(where.keys())) in ("$and", "$or"):
        logical_op = cast(LogicalOperator, where)

        if "$and" in logical_op:
            sub_queries = [
                _translate_where_to_elasticsearch_query(sub_filter)
                for sub_filter in logical_op["$and"]
            ]
            return {"bool": {"must": sub_queries}}

        elif "$or" in logical_op:
            sub_queries = [
                _translate_where_to_elasticsearch_query(sub_filter)
                for sub_filter in logical_op["$or"]
            ]
            return {"bool": {"should": sub_queries}}

    # Handle field-level filters
    field_filters = cast(WhereExpression, where)
    must_clauses = []

    # Fields that are mapped as keywords (exact match)
    keyword_fields = {
        "id",
        "version",
        "session_id",
        "event_id",
        "agent_id",
        "customer_id",
        "correlation_id",
        "source",
        "kind",
        "variable_id",
        "key",
        "tool_id",
        "guideline_id",
        "tag_id",
        "evaluation_id",
    }

    # Fields that have both text and keyword subfields
    text_with_keyword_fields = {"name", "condition", "action"}

    for field_name, field_filter in field_filters.items():
        for operator, filter_value in field_filter.items():
            if operator == "$eq":
                if field_name in keyword_fields:
                    # Try direct keyword field first, fallback to .keyword subfield
                    # This handles both explicit keyword mapping and auto-created text+keyword mapping
                    must_clauses.append(
                        {
                            "bool": {
                                "should": [
                                    {"term": {field_name: filter_value}},
                                    {"term": {f"{field_name}.keyword": filter_value}},
                                ]
                            }
                        }
                    )
                elif field_name in text_with_keyword_fields and isinstance(filter_value, str):
                    # Use .keyword subfield for exact matches
                    must_clauses.append({"term": {f"{field_name}.keyword": filter_value}})
                else:
                    # Default to term query
                    must_clauses.append({"term": {field_name: filter_value}})
            elif operator == "$ne":
                if field_name in keyword_fields:
                    # Try direct keyword field first, fallback to .keyword subfield
                    must_clauses.append(
                        {
                            "bool": {
                                "must_not": {
                                    "bool": {
                                        "should": [
                                            {"term": {field_name: filter_value}},
                                            {"term": {f"{field_name}.keyword": filter_value}},
                                        ]
                                    }
                                }
                            }
                        }
                    )
                elif field_name in text_with_keyword_fields and isinstance(filter_value, str):
                    # Use .keyword subfield for exact matches
                    must_clauses.append(
                        {"bool": {"must_not": {"term": {f"{field_name}.keyword": filter_value}}}}
                    )
                else:
                    # Default to term query
                    must_clauses.append(
                        {"bool": {"must_not": {"term": {field_name: filter_value}}}}
                    )
            elif operator == "$gt":
                must_clauses.append({"range": {field_name: {"gt": filter_value}}})
            elif operator == "$gte":
                must_clauses.append({"range": {field_name: {"gte": filter_value}}})
            elif operator == "$lt":
                must_clauses.append({"range": {field_name: {"lt": filter_value}}})
            elif operator == "$lte":
                must_clauses.append({"range": {field_name: {"lte": filter_value}}})
            elif operator == "$in":
                must_clauses.append({"terms": {field_name: filter_value}})
            elif operator == "$nin":
                must_clauses.append({"bool": {"must_not": {"terms": {field_name: filter_value}}}})

    if len(must_clauses) == 1:
        return must_clauses[0]
    elif len(must_clauses) > 1:
        return {"bool": {"must": must_clauses}}
    else:
        return {"match_all": {}}


class ElasticsearchDocumentCollection(DocumentCollection[TDocument]):
    """
    An Elasticsearch implementation of DocumentCollection.

    This class provides document operations (CRUD) backed by an Elasticsearch index.
    """

    def __init__(
        self,
        database: ElasticsearchDocumentDatabase,
        name: str,
        index_name: str,
        schema: type[TDocument],
    ) -> None:
        """
        Initialize the Elasticsearch document collection.

        Args:
            database: Parent ElasticsearchDocumentDatabase instance
            name: Collection name
            index_name: Elasticsearch index name
            schema: Document schema type
        """
        self._database = database
        self._name = name
        self._index_name = index_name
        self._schema = schema

    @property
    def elasticsearch_client(self) -> AsyncElasticsearch:
        """Get the Elasticsearch client from the parent database."""
        return self._database.elasticsearch_client

    @override
    async def find(
        self,
        filters: Where,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
    ) -> FindResult[TDocument]:
        """Find documents with cursor-based pagination.

        Results are sorted by creation_utc with id as tiebreaker.

        Args:
            filters: Where filter specification.
            limit: Maximum number of documents to return per page.
            cursor: Pagination cursor from a previous query.
            sort_direction: Sort order (ASC or DESC). Defaults to ASC.

        Returns:
            FindResult containing documents and pagination metadata.
        """
        query = _translate_where_to_elasticsearch_query(filters)
        sort_direction = sort_direction or SortDirection.ASC

        # Build the Elasticsearch query with cursor conditions
        if cursor is not None:
            cursor_query = self._build_cursor_query(cursor=cursor, sort_direction=sort_direction)
            # Combine the original query with cursor conditions
            if query.get("match_all"):
                query = cursor_query
            else:
                query = {"bool": {"must": [query, cursor_query]}}

        # Build sort specification
        sort_order = "desc" if sort_direction == SortDirection.DESC else "asc"
        sort_spec = [
            {"creation_utc": {"order": sort_order}},
            {"id": {"order": sort_order}},  # Tiebreaker
        ]

        # Fetch limit + 1 to check if there are more results
        search_size = (limit + 1) if limit else 10000

        try:
            response = await self.elasticsearch_client.search(
                index=self._index_name,
                body={"query": query, "sort": sort_spec, "size": search_size},
            )

            documents: list[TDocument] = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                # Ensure the Elasticsearch _id is available as 'id' field
                if "id" not in doc:
                    doc["id"] = hit["_id"]
                documents.append(cast(TDocument, doc))

            # Calculate pagination metadata
            has_more = False
            next_cursor = None
            total_count = len(documents)

            if limit and len(documents) > limit:
                has_more = True
                documents = documents[:limit]  # Remove the extra item

                # Create cursor from the last item
                if documents:
                    last_item = documents[-1]
                    next_cursor = Cursor(
                        creation_utc=str(last_item.get("creation_utc", "")),
                        id=ObjectId(str(last_item.get("id", ""))),
                    )

            return FindResult(
                items=documents,
                total_count=len(documents),
                has_more=has_more,
                next_cursor=next_cursor,
            )
        except NotFoundError:
            # Index doesn't exist
            return FindResult(items=[], total_count=0, has_more=False, next_cursor=None)
        except Exception as e:
            self._database._logger.error(f"Search failed in index {self._index_name}: {e}")
            return FindResult(items=[], total_count=0, has_more=False, next_cursor=None)

    def _build_cursor_query(
        self,
        cursor: Cursor,
        sort_direction: SortDirection,
    ) -> dict[str, Any]:
        """Build an Elasticsearch query for cursor-based pagination.

        Args:
            cursor: The pagination cursor.
            sort_direction: The sort direction.

        Returns:
            Elasticsearch query dict for cursor conditions.
        """
        if sort_direction == SortDirection.DESC:
            # For descending order, get items with smaller creation_utc or same creation_utc with smaller id
            return {
                "bool": {
                    "should": [
                        {"range": {"creation_utc": {"lt": cursor.creation_utc}}},
                        {
                            "bool": {
                                "must": [
                                    {"term": {"creation_utc": cursor.creation_utc}},
                                    {"range": {"id": {"lt": cursor.id}}},
                                ]
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            }
        else:
            # For ascending order, get items with larger creation_utc or same creation_utc with larger id
            return {
                "bool": {
                    "should": [
                        {"range": {"creation_utc": {"gt": cursor.creation_utc}}},
                        {
                            "bool": {
                                "must": [
                                    {"term": {"creation_utc": cursor.creation_utc}},
                                    {"range": {"id": {"gt": cursor.id}}},
                                ]
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            }

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
        query = _translate_where_to_elasticsearch_query(filters)

        try:
            response = await self.elasticsearch_client.search(
                index=self._index_name, body={"query": query, "size": 1}
            )

            hits = response["hits"]["hits"]
            if hits:
                doc = hits[0]["_source"]
                # Ensure the Elasticsearch _id is available as 'id' field
                if "id" not in doc:
                    doc["id"] = hits[0]["_id"]

                # Handle invalid version strings in metadata collections
                if "metadata" in self._index_name and "version" in doc:
                    version_str = doc.get("version")
                    if version_str and isinstance(version_str, str):
                        try:
                            # Try to parse the version to see if it's valid
                            from parlant.core.common import Version

                            Version.from_string(version_str)
                        except ValueError:
                            # Invalid version string - convert to a valid but old version
                            # This will trigger MigrationRequired in the framework
                            doc = dict(doc)  # Make a copy
                            doc["version"] = "0.0.1"  # Very old version that will trigger migration

                return cast(TDocument, doc)
            return None
        except NotFoundError:
            # Index doesn't exist
            return None
        except Exception as e:
            self._database._logger.error(f"Find one failed in index {self._index_name}: {e}")
            return None

    @override
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult:
        """
        Insert a single document.

        Args:
            document: Document to insert

        Returns:
            Insert operation result
        """
        ensure_is_total(document, self._schema)

        try:
            # Use the document ID as the Elasticsearch document ID
            doc_id = document.get("id")
            if doc_id:
                await self.elasticsearch_client.index(
                    index=self._index_name, id=doc_id, body=document, refresh=True
                )
            else:
                await self.elasticsearch_client.index(
                    index=self._index_name, body=document, refresh=True
                )

            return InsertResult(acknowledged=True)
        except Exception as e:
            self._database._logger.error(f"Insert failed in index {self._index_name}: {e}")
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
        # First, find the document to update
        existing_doc = await self.find_one(filters)

        if existing_doc:
            # Update the existing document
            updated_doc = cast(TDocument, {**existing_doc, **params})

            try:
                doc_id = existing_doc.get("id")
                if doc_id:
                    await self.elasticsearch_client.index(
                        index=self._index_name,
                        id=doc_id,
                        body=updated_doc,
                        refresh=True,
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
                self._database._logger.error(f"Update failed in index {self._index_name}: {e}")
                return UpdateResult(
                    acknowledged=False,
                    matched_count=1,
                    modified_count=0,
                    updated_document=None,
                )
        elif upsert:
            # Insert new document
            await self.insert_one(params)
            return UpdateResult(
                acknowledged=True,
                matched_count=0,
                modified_count=0,
                updated_document=params,
            )
        else:
            # No document found and upsert is False
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
        # First, find the document to delete
        existing_doc = await self.find_one(filters)

        if existing_doc:
            try:
                doc_id = existing_doc.get("id")
                if doc_id:
                    await self.elasticsearch_client.delete(
                        index=self._index_name,
                        id=doc_id,
                        refresh=True,  # Ensure immediate visibility
                    )

                    return DeleteResult(
                        deleted_count=1,
                        acknowledged=True,
                        deleted_document=cast(TDocument, existing_doc),
                    )
                else:
                    self._database._logger.error("Cannot delete document without ID")
                    return DeleteResult(
                        acknowledged=False,
                        deleted_count=0,
                        deleted_document=None,
                    )
            except NotFoundError:
                # Document was already deleted
                return DeleteResult(
                    acknowledged=True,
                    deleted_count=0,
                    deleted_document=None,
                )
            except Exception as e:
                self._database._logger.error(f"Delete failed in index {self._index_name}: {e}")
                return DeleteResult(
                    acknowledged=False,
                    deleted_count=0,
                    deleted_document=None,
                )
        else:
            # Document not found
            return DeleteResult(
                acknowledged=True,
                deleted_count=0,
                deleted_document=None,
            )
