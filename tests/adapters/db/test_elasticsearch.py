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

import uuid
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import AsyncIterator, Iterator, Optional, TypedDict, cast
import numpy as np
from typing_extensions import Required
from lagom import Container
from pytest import fixture, raises, mark

from parlant.adapters.nlp.openai_service import OpenAITextEmbedding3Large
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.adapters.vector_db.elasticsearch import (
    ElasticsearchVectorCollection,
    ElasticsearchVectorDatabase,
    create_elasticsearch_client_from_env,
)
from parlant.core.agents import AgentStore, AgentId
from parlant.core.common import IdGenerator, Version, md5_checksum
from parlant.core.glossary import GlossaryVectorStore
from parlant.core.nlp.embedding import Embedder, EmbedderFactory, NullEmbedder, NullEmbeddingCache
from parlant.core.loggers import Logger
from parlant.core.nlp.service import NLPService
from parlant.core.persistence.common import MigrationRequired, ObjectId
from parlant.core.persistence.vector_database import BaseDocument
from parlant.core.tags import Tag, TagId
from tests.test_utilities import SyncAwaiter


async def _openai_embedder_type_provider() -> type[Embedder]:
    return OpenAITextEmbedding3Large


async def _null_embedder_type_provider() -> type[Embedder]:
    return NullEmbedder


class _TestDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    content: str
    checksum: Required[str]
    name: str


@dataclass(frozen=True)
class _TestContext:
    home_dir: Path
    container: Container


@fixture
def agent_id(
    container: Container,
    sync_await: SyncAwaiter,
) -> AgentId:
    store = container[AgentStore]
    agent = sync_await(store.create_agent(name="test-agent", max_engine_iterations=2))
    return agent.id


@fixture
def context(container: Container) -> Iterator[_TestContext]:
    with tempfile.TemporaryDirectory() as home_dir:
        home_dir_path = Path(home_dir)
        yield _TestContext(
            container=container,
            home_dir=home_dir_path,
        )


@fixture
def doc_version() -> Version.String:
    return Version.from_string("0.1.0").to_string()


@fixture
def test_prefix() -> str:
    """Generate unique prefix for each test to avoid isolation issues."""
    unique_id = uuid.uuid4().hex[:8]
    return f"test_parlant_{unique_id}"


async def cleanup_test_indices(prefix: str) -> None:
    """Helper function to cleanup all test indices and templates."""
    es_client = create_elasticsearch_client_from_env()
    try:
        # Cleanup vector database indices (with _vecdb_ infix)
        all_indices = await es_client.indices.get(index=f"{prefix}_vecdb_*")
        for index_name in all_indices.keys():
            try:
                await es_client.indices.delete(index=index_name)
            except Exception:
                pass

        # Cleanup vector database template
        try:
            await es_client.indices.delete_index_template(name=f"{prefix}_vecdb_template")
        except Exception:
            pass  # Template doesn't exist

    except Exception:
        pass  # No indices to clean
    finally:
        await es_client.close()


@fixture(scope="function", autouse=True)
async def cleanup_elasticsearch_indices(test_prefix: str) -> AsyncIterator[None]:
    """Automatically cleanup test indices before and after each test."""
    # Cleanup before test
    await cleanup_test_indices(test_prefix)
    yield
    # Cleanup after test
    await cleanup_test_indices(test_prefix)


@fixture
async def elasticsearch_database(
    context: _TestContext,
    test_prefix: str,
) -> AsyncIterator[ElasticsearchVectorDatabase]:
    """Create Elasticsearch database with proper cleanup after each test."""
    async with create_database(context, test_prefix) as es_database:
        yield es_database


def create_database(context: _TestContext, index_prefix: str) -> ElasticsearchVectorDatabase:
    """Create Elasticsearch database with test-specific configuration."""
    es_client = create_elasticsearch_client_from_env()

    return ElasticsearchVectorDatabase(
        elasticsearch_client=es_client,
        index_prefix=index_prefix,
        logger=context.container[Logger],
        embedder_factory=EmbedderFactory(context.container),
        embedding_cache_provider=NullEmbeddingCache,
    )


@fixture
async def elasticsearch_collection(
    elasticsearch_database: ElasticsearchVectorDatabase,
) -> AsyncIterator[ElasticsearchVectorCollection[_TestDocument]]:
    """Create and cleanup test collection."""
    collection = await elasticsearch_database.get_or_create_collection(
        "test_collection",
        _TestDocument,
        embedder_type=OpenAITextEmbedding3Large,
        document_loader=_identity_loader,
    )
    yield collection
    # Cleanup: delete collection after test
    try:
        await elasticsearch_database.delete_collection("test_collection")
    except Exception:
        pass  # Already deleted or doesn't exist


async def _identity_loader(doc: BaseDocument) -> _TestDocument:
    """Identity document loader for tests."""
    return cast(_TestDocument, doc)


# ============================================================================
# BASIC CRUD TESTS
# ============================================================================


@mark.asyncio
async def test_that_a_document_can_be_found_based_on_a_metadata_field(
    elasticsearch_collection: ElasticsearchVectorCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    """Test finding documents by ID and metadata fields."""
    doc = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    # Insert document
    insert_result = await elasticsearch_collection.insert_one(doc)
    assert insert_result.acknowledged

    # Find by ID
    find_by_id_result = await elasticsearch_collection.find({"id": {"$eq": "1"}})
    assert len(find_by_id_result) == 1
    assert find_by_id_result[0] == doc

    # Find one by ID
    find_one_result = await elasticsearch_collection.find_one({"id": {"$eq": "1"}})
    assert find_one_result == doc

    # Find by name (metadata field)
    find_by_name_result = await elasticsearch_collection.find({"name": {"$eq": "test name"}})
    assert len(find_by_name_result) == 1
    assert find_by_name_result[0] == doc

    # Find by non-existing name
    find_by_not_existing_name_result = await elasticsearch_collection.find(
        {"name": {"$eq": "not existing"}}
    )
    assert len(find_by_not_existing_name_result) == 0


@mark.asyncio
async def test_that_update_one_without_upsert_updates_existing_document(
    elasticsearch_collection: ElasticsearchVectorCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    """Test updating an existing document without upsert."""
    document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    await elasticsearch_collection.insert_one(document)

    updated_document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="new name",
        checksum=md5_checksum("test content"),
    )

    update_result = await elasticsearch_collection.update_one(
        {"name": {"$eq": "test name"}},
        updated_document,
        upsert=False,
    )

    assert update_result.acknowledged
    assert update_result.matched_count == 1
    assert update_result.modified_count == 1
    assert update_result.updated_document == updated_document

    # Verify old name no longer exists
    result = await elasticsearch_collection.find({"name": {"$eq": "test name"}})
    assert len(result) == 0

    # Verify new name exists
    result = await elasticsearch_collection.find({"name": {"$eq": "new name"}})
    assert len(result) == 1
    assert result[0] == updated_document


@mark.asyncio
async def test_that_update_one_without_upsert_and_no_preexisting_document_with_same_id_does_not_insert(
    elasticsearch_collection: ElasticsearchVectorCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    """Test that update without upsert doesn't insert new documents."""
    updated_document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    result = await elasticsearch_collection.update_one(
        {"name": {"$eq": "new name"}},
        updated_document,
        upsert=False,
    )

    assert result.acknowledged
    assert result.matched_count == 0
    assert result.modified_count == 0
    assert result.updated_document is None
    assert 0 == len(await elasticsearch_collection.find({}))


@mark.asyncio
async def test_that_update_one_with_upsert_and_no_preexisting_document_with_same_id_does_insert_new_document(
    elasticsearch_collection: ElasticsearchVectorCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    """Test that update with upsert creates new documents."""
    updated_document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    update_result = await elasticsearch_collection.update_one(
        {"name": {"$eq": "test name"}},
        updated_document,
        upsert=True,
    )

    assert update_result.acknowledged
    assert update_result.matched_count == 0
    assert update_result.modified_count == 0
    assert update_result.updated_document == updated_document

    result = await elasticsearch_collection.find({"name": {"$eq": "test name"}})
    assert len(result) == 1
    assert result[0] == updated_document


@mark.asyncio
async def test_delete_one(
    elasticsearch_collection: ElasticsearchVectorCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    """Test deleting a document."""
    document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    await elasticsearch_collection.insert_one(document)

    # Verify document exists
    result = await elasticsearch_collection.find({"id": {"$eq": "1"}})
    assert len(result) == 1

    # Delete document
    deleted_result = await elasticsearch_collection.delete_one({"id": {"$eq": "1"}})

    assert deleted_result.acknowledged
    assert deleted_result.deleted_count == 1
    assert deleted_result.deleted_document is not None
    assert deleted_result.deleted_document["id"] == ObjectId("1")

    # Verify document no longer exists
    result = await elasticsearch_collection.find({"id": {"$eq": "1"}})
    assert len(result) == 0


# ============================================================================
# VECTOR SIMILARITY SEARCH TESTS
# ============================================================================


@mark.asyncio
async def test_find_similar_documents(
    elasticsearch_collection: ElasticsearchVectorCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    """Test vector similarity search returns most similar documents."""
    apple_document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="apple",
        name="Apple",
        checksum=md5_checksum("apple"),
    )

    banana_document = _TestDocument(
        id=ObjectId("2"),
        version=doc_version,
        content="banana",
        name="Banana",
        checksum=md5_checksum("banana"),
    )

    cherry_document = _TestDocument(
        id=ObjectId("3"),
        version=doc_version,
        content="cherry",
        name="Cherry",
        checksum=md5_checksum("cherry"),
    )

    # Insert all documents
    await elasticsearch_collection.insert_one(apple_document)
    await elasticsearch_collection.insert_one(banana_document)
    await elasticsearch_collection.insert_one(cherry_document)
    await elasticsearch_collection.insert_one(
        _TestDocument(
            id=ObjectId("4"),
            version=doc_version,
            content="date",
            name="Date",
            checksum=md5_checksum("date"),
        )
    )
    await elasticsearch_collection.insert_one(
        _TestDocument(
            id=ObjectId("5"),
            version=doc_version,
            content="elderberry",
            name="Elderberry",
            checksum=md5_checksum("elderberry"),
        )
    )

    query = "apple banana cherry"
    k = 3

    similar_results = await elasticsearch_collection.find_similar_documents({}, query, k)
    result = [s.document for s in similar_results]

    assert len(result) == 3
    assert apple_document in result
    assert banana_document in result
    assert cherry_document in result

    # Verify distances are valid
    for sim_result in similar_results:
        assert sim_result.distance >= 0.0
        assert sim_result.distance <= 2.0  # Max cosine distance


@mark.asyncio
async def test_find_similar_documents_with_filters(
    elasticsearch_collection: ElasticsearchVectorCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    """Test vector similarity search with metadata filters."""
    # Insert documents with different categories
    await elasticsearch_collection.insert_one(
        _TestDocument(
            id=ObjectId("1"),
            version=doc_version,
            content="red apple fruit",
            name="Red Apple",
            checksum=md5_checksum("red apple fruit"),
        )
    )
    await elasticsearch_collection.insert_one(
        _TestDocument(
            id=ObjectId("2"),
            version=doc_version,
            content="yellow banana fruit",
            name="Yellow Banana",
            checksum=md5_checksum("yellow banana fruit"),
        )
    )
    await elasticsearch_collection.insert_one(
        _TestDocument(
            id=ObjectId("3"),
            version=doc_version,
            content="apple computer",
            name="Apple Computer",
            checksum=md5_checksum("apple computer"),
        )
    )

    # Search with filter - should only find fruit-related document
    query = "apple"
    results = await elasticsearch_collection.find_similar_documents(
        {"name": {"$eq": "Red Apple"}}, query, k=5
    )

    assert len(results) == 1
    assert results[0].document["id"] == ObjectId("1")


# ============================================================================
# PERSISTENCE & COLLECTION LOADING TESTS
# ============================================================================


@mark.asyncio
async def test_loading_collections(
    context: _TestContext,
    doc_version: Version.String,
    test_prefix: str,
) -> None:
    """Test that collections persist across database instances."""
    # Create and populate collection in first database instance
    async with create_database(context, test_prefix) as first_db:
        created_collection = await first_db.get_or_create_collection(
            "test_collection",
            _TestDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=_identity_loader,
        )

        document = _TestDocument(
            id=ObjectId("1"),
            version=doc_version,
            content="test content",
            name="test name",
            checksum=md5_checksum("test content"),
        )

        await created_collection.insert_one(document)

    # Load collection in second database instance
    async with create_database(context, test_prefix) as second_db:
        fetched_collection = await second_db.get_collection(
            "test_collection",
            _TestDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=_identity_loader,
        )

        result = await fetched_collection.find({"id": {"$eq": "1"}})

        assert len(result) == 1
        assert result[0] == document


# ============================================================================
# BULK OPERATIONS TESTS
# ============================================================================


@mark.asyncio
async def test_bulk_insert_many(
    elasticsearch_collection: ElasticsearchVectorCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    """Test bulk insertion of multiple documents."""
    documents = [
        _TestDocument(
            id=ObjectId(str(i)),
            version=doc_version,
            content=f"test content {i}",
            name=f"Document {i}",
            checksum=md5_checksum(f"test content {i}"),
        )
        for i in range(1, 101)  # 100 documents
    ]

    # Bulk insert
    result = await elasticsearch_collection.insert_many(documents, chunk_size=50)
    assert result.acknowledged

    # Verify all documents were inserted
    all_docs = await elasticsearch_collection.find({})
    assert len(all_docs) == 100


# ============================================================================
# PAGINATION TESTS
# ============================================================================


@mark.asyncio
async def test_find_paginated(
    elasticsearch_collection: ElasticsearchVectorCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    """Test paginated document retrieval."""
    # Insert 50 documents
    for i in range(1, 51):
        await elasticsearch_collection.insert_one(
            _TestDocument(
                id=ObjectId(str(i)),
                version=doc_version,
                content=f"content {i}",
                name=f"Doc {i}",
                checksum=md5_checksum(f"content {i}"),
            )
        )

    # First page
    page1, cursor1 = await elasticsearch_collection.find_paginated(
        filters={}, page_size=20, search_after=None
    )
    assert len(page1) == 20
    assert cursor1 is not None

    # Second page
    page2, cursor2 = await elasticsearch_collection.find_paginated(
        filters={}, page_size=20, search_after=cursor1
    )
    assert len(page2) == 20
    assert cursor2 is not None

    # Third page
    page3, cursor3 = await elasticsearch_collection.find_paginated(
        filters={}, page_size=20, search_after=cursor2
    )
    assert len(page3) == 10  # Remaining documents
    assert cursor3 is not None

    # Verify no duplicate IDs across pages
    all_ids = (
        {doc["id"] for doc in page1} | {doc["id"] for doc in page2} | {doc["id"] for doc in page3}
    )
    assert len(all_ids) == 50


# ============================================================================
# HYBRID SEARCH TESTS
# ============================================================================


@mark.asyncio
async def test_hybrid_search(
    elasticsearch_collection: ElasticsearchVectorCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    """Test hybrid search combining text and vector similarity."""
    # Insert documents with varying relevance
    await elasticsearch_collection.insert_one(
        _TestDocument(
            id=ObjectId("1"),
            version=doc_version,
            content="machine learning algorithms and neural networks",
            name="ML Doc",
            checksum=md5_checksum("machine learning algorithms and neural networks"),
        )
    )
    await elasticsearch_collection.insert_one(
        _TestDocument(
            id=ObjectId("2"),
            version=doc_version,
            content="deep learning with tensorflow",
            name="DL Doc",
            checksum=md5_checksum("deep learning with tensorflow"),
        )
    )
    await elasticsearch_collection.insert_one(
        _TestDocument(
            id=ObjectId("3"),
            version=doc_version,
            content="cooking recipes",
            name="Recipe Doc",
            checksum=md5_checksum("cooking recipes"),
        )
    )

    query = "machine learning neural networks"
    results = await elasticsearch_collection.hybrid_search(
        query=query, text_weight=0.4, vector_weight=0.6, k=2
    )

    assert len(results) <= 2
    # Most relevant document should be first
    assert results[0].document["id"] == ObjectId("1")


# ============================================================================
# GLOSSARY STORE INTEGRATION TESTS
# ============================================================================


@mark.asyncio
async def test_that_glossary_elasticsearch_store_correctly_finds_relevant_terms_from_large_query_input(
    container: Container,
    agent_id: AgentId,
    context: _TestContext,
    test_prefix: str,
) -> None:
    """Test glossary store with large query input (matches ChromaDB test)."""

    async def embedder_type_provider() -> type[Embedder]:
        return type(await container[NLPService].get_embedder())

    async with create_database(context, test_prefix) as es_db:
        async with GlossaryVectorStore(
            id_generator=container[IdGenerator],
            vector_db=es_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=EmbedderFactory(container),
            embedder_type_provider=embedder_type_provider,
        ) as glossary_store:
            bazoo = await glossary_store.create_term(
                name="Bazoo",
                description="a type of cow",
            )

            shazoo = await glossary_store.create_term(
                name="Shazoo",
                description="a type of zebra",
            )

            kazoo = await glossary_store.create_term(
                name="Kazoo",
                description="a type of horse",
            )

            # Very large query with embedded terms
            terms = await glossary_store.find_relevant_terms(
                query=("walla " * 5000)
                + "Kazoo"
                + ("balla " * 5000)
                + "Shazoo"
                + ("kalla " * 5000)
                + "Bazoo",
                available_terms=[bazoo, shazoo, kazoo],
                max_terms=3,
            )

            assert len(terms) == 3
            assert any(t.id == kazoo.id for t in terms)
            assert any(t.id == shazoo.id for t in terms)
            assert any(t.id == bazoo.id for t in terms)


# ============================================================================
# MIGRATION & VERSIONING TESTS
# ============================================================================


class _TestDocumentV2(BaseDocument):
    new_name: str


@mark.asyncio
async def test_that_when_persistence_and_store_version_match_allows_store_to_open_when_migrate_is_disabled(
    context: _TestContext,
    test_prefix: str,
) -> None:
    """Test that matching versions allow store to open without migration."""
    async with create_database(context, test_prefix) as es_db:
        async with GlossaryVectorStore(
            id_generator=IdGenerator(),
            vector_db=es_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=EmbedderFactory(context.container),
            embedder_type_provider=_null_embedder_type_provider,
            allow_migration=False,
        ):
            metadata = await es_db.read_metadata()

            assert metadata
            assert metadata["version"] == GlossaryVectorStore.VERSION.to_string()


@mark.asyncio
async def test_that_document_loader_updates_documents_in_current_elasticsearch_collection(
    context: _TestContext,
    test_prefix: str,
) -> None:
    """Test that document loader properly migrates documents."""

    async def _document_loader(doc: BaseDocument) -> _TestDocumentV2:
        if doc["version"] == Version.String("1.0.0"):
            doc_1 = cast(_TestDocument, doc)

            return _TestDocumentV2(
                id=doc_1["id"],
                version=Version.String("2.0.0"),
                content=doc_1["content"],
                checksum=md5_checksum(doc_1["content"] + doc_1["name"]),
                new_name=doc_1["name"],
            )

        if doc["version"] == Version.String("2.0.0"):
            return cast(_TestDocumentV2, doc)

        raise ValueError(f"Version {doc['version']} not supported")

    # Create initial collection with v1.0.0 documents
    async with create_database(context, test_prefix) as es_database:
        collection = await es_database.get_or_create_collection(
            "test_collection",
            _TestDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=_identity_loader,
        )

        documents = [
            _TestDocument(
                id=ObjectId("1"),
                version=Version.String("1.0.0"),
                content="strawberry",
                name="Document 1",
                checksum=md5_checksum("strawberry"),
            ),
            _TestDocument(
                id=ObjectId("2"),
                version=Version.String("1.0.0"),
                content="apple",
                name="Document 2",
                checksum=md5_checksum("apple"),
            ),
            _TestDocument(
                id=ObjectId("3"),
                version=Version.String("1.0.0"),
                content="cherry",
                name="Document 3",
                checksum=md5_checksum("cherry"),
            ),
        ]

        for doc in documents:
            await collection.insert_one(doc)

    # Reopen with document loader to migrate to v2.0.0
    async with create_database(context, test_prefix) as es_database:
        new_collection = await es_database.get_or_create_collection(
            "test_collection",
            _TestDocumentV2,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=_document_loader,
        )

        new_documents = await new_collection.find({})
        assert len(new_documents) == 3

        # Verify migration occurred correctly
        doc1 = next(d for d in new_documents if d["id"] == ObjectId("1"))
        assert doc1["content"] == "strawberry"
        assert doc1["new_name"] == "Document 1"
        assert doc1["version"] == Version.String("2.0.0")
        assert doc1["checksum"] == md5_checksum("strawberryDocument 1")


@mark.asyncio
async def test_that_failed_migrations_are_stored_in_failed_migrations_collection(
    context: _TestContext,
    test_prefix: str,
) -> None:
    """Test that documents that fail migration are stored separately."""
    # Create initial collection with documents
    async with create_database(context, test_prefix) as es_database:
        collection = await es_database.get_or_create_collection(
            "test_collection",
            _TestDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=_identity_loader,
        )

        documents = [
            _TestDocument(
                id=ObjectId("1"),
                version=Version.String("1.0.0"),
                content="valid content",
                name="Valid Document",
                checksum=md5_checksum("valid content"),
            ),
            _TestDocument(
                id=ObjectId("2"),
                version=Version.String("1.0.0"),
                content="invalid",
                name="Invalid Document",
                checksum=md5_checksum("invalid"),
            ),
            _TestDocument(
                id=ObjectId("3"),
                version=Version.String("1.0.0"),
                content="another valid content",
                name="Another Valid Document",
                checksum=md5_checksum("another valid content"),
            ),
        ]

        for doc in documents:
            await collection.insert_one(doc)

    # Reopen with loader that fails for specific documents
    async with create_database(context, test_prefix) as es_database:

        async def _document_loader(doc: BaseDocument) -> Optional[_TestDocumentV2]:
            doc_1 = cast(_TestDocument, doc)
            if doc_1["content"] == "invalid":
                return None  # Simulate migration failure
            return _TestDocumentV2(
                id=doc_1["id"],
                version=Version.String("2.0.0"),
                content=doc_1["content"],
                new_name=doc_1["name"],
                checksum=md5_checksum(doc_1["content"] + doc_1["name"]),
            )

        collection_with_loader = await es_database.get_or_create_collection(
            "test_collection",
            _TestDocumentV2,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=_document_loader,
        )

        # Verify only valid documents remain
        valid_documents = await collection_with_loader.find({})
        assert len(valid_documents) == 2

        valid_contents = {doc["content"] for doc in valid_documents}
        assert "valid content" in valid_contents
        assert "another valid content" in valid_contents
        assert "invalid" not in valid_contents

        valid_names = {doc["new_name"] for doc in valid_documents}
        assert "Valid Document" in valid_names
        assert "Another Valid Document" in valid_names

        # Check failed migrations collection
        failed_migrations_collection = await es_database.get_or_create_collection(
            "failed_migrations",
            BaseDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=_identity_loader,
        )

        failed_migrations = await failed_migrations_collection.find({})
        assert len(failed_migrations) == 1

        failed_doc = cast(_TestDocument, failed_migrations[0])
        assert failed_doc["id"] == ObjectId("2")
        assert failed_doc["content"] == "invalid"
        assert failed_doc["name"] == "Invalid Document"


@mark.asyncio
async def test_that_migration_error_raised_when_version_mismatch_and_migration_disabled(
    context: _TestContext,
    test_prefix: str,
) -> None:
    """Test that version mismatch raises error when migration is disabled."""
    # Set old version in metadata using the correct key that VectorDocumentStoreMigrationHelper checks
    async with create_database(context, test_prefix) as es_db:
        await es_db.upsert_metadata("GlossaryVectorStore_version", "0.0.1")

    # Attempt to open with migration disabled
    async with create_database(context, test_prefix) as es_db:
        with raises(MigrationRequired) as exc_info:
            async with GlossaryVectorStore(
                IdGenerator(),
                vector_db=es_db,
                document_db=TransientDocumentDatabase(),
                embedder_factory=EmbedderFactory(context.container),
                embedder_type_provider=_null_embedder_type_provider,
                allow_migration=False,
            ):
                pass

        assert "Migration required for GlossaryVectorStore." in str(exc_info.value)


@mark.asyncio
async def test_that_new_store_creates_metadata_with_correct_version(
    context: _TestContext,
    test_prefix: str,
) -> None:
    """Test that new store creates metadata with correct version."""
    async with create_database(context, test_prefix) as es_db:
        async with GlossaryVectorStore(
            IdGenerator(),
            vector_db=es_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=EmbedderFactory(context.container),
            embedder_type_provider=_openai_embedder_type_provider,
            allow_migration=False,
        ):
            metadata = await es_db.read_metadata()

            assert metadata
            assert metadata["version"] == GlossaryVectorStore.VERSION.to_string()


@mark.asyncio
async def test_that_documents_are_indexed_when_changing_embedder_type(
    context: _TestContext,
    agent_id: AgentId,
    test_prefix: str,
) -> None:
    """Test that documents are re-indexed when changing embedder type."""
    # Create store with OpenAI embedder
    async with create_database(context, test_prefix) as es_db:
        async with GlossaryVectorStore(
            IdGenerator(),
            vector_db=es_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=EmbedderFactory(context.container),
            embedder_type_provider=_openai_embedder_type_provider,
            allow_migration=True,
        ) as store:
            term = await store.create_term(
                name="Bazoo",
                description="a type of cow",
            )

            await store.upsert_tag(
                term_id=term.id,
                tag_id=Tag.for_agent_id(agent_id),
            )

    # Reopen with NullEmbedder
    async with create_database(context, test_prefix) as es_db:
        async with GlossaryVectorStore(
            id_generator=IdGenerator(),
            vector_db=es_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=EmbedderFactory(context.container),
            embedder_type_provider=_null_embedder_type_provider,
            allow_migration=True,
        ) as store:
            # Get embedded index name
            embedded_index_name = es_db._get_embedded_index_name("glossary", NullEmbedder)

            # Query Elasticsearch directly to verify embeddings
            response = await es_db.elasticsearch_client.search(
                index=embedded_index_name,
                query={"match_all": {}},
                size=10,
            )

            assert response["hits"]["total"]["value"] >= 1
            docs = [hit["_source"] for hit in response["hits"]["hits"]]

            # Verify embeddings are zero vectors (NullEmbedder characteristic)
            if docs and "content_vector" in docs[0]:
                embeddings = np.array(docs[0]["content_vector"])
                assert np.all(embeddings == 0)

            # Check that the term was indexed - handle both flat and nested metadata
            term_found = False
            for doc in docs:
                doc_id = doc.get("id")
                if not doc_id and "metadata" in doc:
                    doc_id = doc["metadata"].get("id")
                if doc_id == term.id:
                    term_found = True
                    break

            assert term_found, f"Term {term.id} not found in indexed documents"


@mark.asyncio
async def test_that_documents_are_migrated_and_reindexed_for_new_embedder_type(
    context: _TestContext,
    test_prefix: str,
) -> None:
    """Test migration and reindexing when switching embedder types."""

    async def _document_loader(doc: BaseDocument) -> _TestDocumentV2:
        doc_1 = cast(_TestDocument, doc)

        return _TestDocumentV2(
            id=doc_1["id"],
            version=Version.String("2.0.0"),
            content=doc_1["content"],
            new_name=doc_1["name"],
            checksum=md5_checksum(doc_1["content"] + doc_1["name"]),
        )

    # Create collection with OpenAI embedder
    async with create_database(context, test_prefix) as es_database:
        collection = await es_database.get_or_create_collection(
            "test_collection",
            _TestDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=_identity_loader,
        )

        documents = [
            _TestDocument(
                id=ObjectId("1"),
                version=Version.String("1.0.0"),
                content="test content 1",
                name="Document 1",
                checksum=md5_checksum("test content 1"),
            ),
            _TestDocument(
                id=ObjectId("2"),
                version=Version.String("1.0.0"),
                content="test content 2",
                name="Document 2",
                checksum=md5_checksum("test content 2"),
            ),
        ]
        for doc in documents:
            await collection.insert_one(doc)

    # Reopen with NullEmbedder and document loader
    async with create_database(context, test_prefix) as es_database:
        new_collection = await es_database.get_or_create_collection(
            "test_collection",
            _TestDocumentV2,
            embedder_type=NullEmbedder,
            document_loader=_document_loader,
        )

        migrated_docs = await new_collection.find({})
        assert len(migrated_docs) == 2

        # Verify migration and schema change
        assert any(
            d["id"] == ObjectId("1") and d["new_name"] == "Document 1" for d in migrated_docs
        )
        assert any(
            d["id"] == ObjectId("2") and d["new_name"] == "Document 2" for d in migrated_docs
        )
        assert all(d["version"] == Version.String("2.0.0") for d in migrated_docs)


# ============================================================================
# FILTER OPERATION TESTS
# ============================================================================


@mark.asyncio
async def test_that_in_filter_works_with_list_of_strings(
    context: _TestContext,
    test_prefix: str,
) -> None:
    """Test $in filter with list of tag IDs."""
    async with create_database(context, test_prefix) as es_db:
        async with GlossaryVectorStore(
            IdGenerator(),
            vector_db=es_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=EmbedderFactory(context.container),
            embedder_type_provider=_null_embedder_type_provider,
            allow_migration=True,
        ) as store:
            first_term = await store.create_term(
                name="Bazoo",
                description="a type of cow",
            )
            second_term = await store.create_term(
                name="Shazoo",
                description="a type of cow",
            )
            third_term = await store.create_term(
                name="Fazoo",
                description="a type of cow",
            )

            # Setup tags
            await store.upsert_tag(term_id=first_term.id, tag_id=TagId("a"))
            await store.upsert_tag(term_id=first_term.id, tag_id=TagId("b"))
            await store.upsert_tag(term_id=second_term.id, tag_id=TagId("b"))
            await store.upsert_tag(term_id=third_term.id, tag_id=TagId("c"))
            await store.upsert_tag(term_id=third_term.id, tag_id=TagId("d"))

            # Test various tag combinations
            terms = await store.list_terms(tags=[TagId("a"), TagId("b")])
            assert len(terms) == 2
            assert terms[0].id == first_term.id
            assert terms[1].id == second_term.id

            terms = await store.list_terms(tags=[TagId("a"), TagId("b"), TagId("c")])
            assert len(terms) == 3
            assert terms[0].id == first_term.id
            assert terms[1].id == second_term.id
            assert terms[2].id == third_term.id

            terms = await store.list_terms(tags=[TagId("a"), TagId("b"), TagId("c"), TagId("d")])
            assert len(terms) == 3


@mark.asyncio
async def test_that_in_filter_works_with_single_tag(
    context: _TestContext,
    test_prefix: str,
) -> None:
    """Test $in filter with a single tag."""
    async with create_database(context, test_prefix) as es_db:
        async with GlossaryVectorStore(
            id_generator=IdGenerator(),
            vector_db=es_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=EmbedderFactory(context.container),
            embedder_type_provider=_null_embedder_type_provider,
            allow_migration=True,
        ) as store:
            first_term = await store.create_term(
                name="Bazoo",
                description="a type of cow",
            )
            await store.upsert_tag(
                term_id=first_term.id,
                tag_id=TagId("unique_tag"),
            )

            # Test with a single tag that matches one term
            terms = await store.list_terms(tags=[TagId("unique_tag")])
            assert len(terms) == 1
            assert terms[0].id == first_term.id
            assert terms[0].name == "Bazoo"


# ============================================================================
# HEALTH CHECK & OPERATIONAL TESTS
# ============================================================================


@mark.asyncio
async def test_elasticsearch_health_check(
    elasticsearch_database: ElasticsearchVectorDatabase,
) -> None:
    """Test Elasticsearch cluster health check."""
    health = await elasticsearch_database.health_check()

    assert "status" in health
    assert health["status"] in ["green", "yellow", "red"]

    if health["status"] != "red":
        assert "cluster_name" in health
        assert "elasticsearch_version" in health
        assert "indices_count" in health


@mark.asyncio
async def test_dual_index_architecture(
    elasticsearch_database: ElasticsearchVectorDatabase,
    doc_version: Version.String,
) -> None:
    """Test that dual-index architecture (embedded + unembedded) works correctly."""
    collection = await elasticsearch_database.get_or_create_collection(
        "dual_index_test",
        _TestDocument,
        embedder_type=OpenAITextEmbedding3Large,
        document_loader=_identity_loader,
    )

    document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    await collection.insert_one(document)

    # Verify document exists in embedded index
    embedded_index_name = elasticsearch_database._get_embedded_index_name(
        "dual_index_test", OpenAITextEmbedding3Large
    )
    embedded_response = await elasticsearch_database.elasticsearch_client.get(
        index=embedded_index_name, id="1"
    )
    assert embedded_response["_source"]["content_vector"]  # Has vector

    # Verify document exists in unembedded index
    unembedded_index_name = elasticsearch_database._get_unembedded_index_name("dual_index_test")
    unembedded_response = await elasticsearch_database.elasticsearch_client.get(
        index=unembedded_index_name, id="1"
    )
    assert "content_vector" not in unembedded_response["_source"]  # No vector

    # Cleanup
    await elasticsearch_database.delete_collection("dual_index_test")
