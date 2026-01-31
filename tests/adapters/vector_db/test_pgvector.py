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

from __future__ import annotations

import json
from typing import Any, TypedDict, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from typing_extensions import Required

from parlant.adapters.vector_db.pgvector import (
    PostgresVectorCollection,
    PostgresVectorDatabase,
    _WhereTranslator,
    _compute_checksum,
)
from parlant.core.common import Version
from parlant.core.nlp.embedding import Embedder, EmbedderFactory, EmbeddingCacheProvider
from parlant.core.nlp.tokenization import ZeroEstimatingTokenizer
from parlant.core.persistence.common import ObjectId, Where
from tests.test_utilities import _TestLogger


class _TestDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    content: str
    checksum: Required[str]
    name: str


class _MockEmbedder(Embedder):
    """Mock embedder for testing."""

    @property
    def id(self) -> str:
        return "mock-embedder"

    @property
    def max_tokens(self) -> int:
        return 8192

    @property
    def tokenizer(self) -> ZeroEstimatingTokenizer:
        return ZeroEstimatingTokenizer()

    @property
    def dimensions(self) -> int:
        return 3

    async def embed(self, texts: list[str], hints: Any = None) -> Any:
        """Return deterministic vectors based on text content."""

        class MockResult:
            def __init__(self, vectors: list[list[float]]) -> None:
                self.vectors = vectors

        vectors = []
        for text in texts:
            # Create a simple deterministic vector based on text hash
            h = hash(text) % 1000
            vectors.append([h / 1000.0, (h + 100) / 1000.0, (h + 200) / 1000.0])
        return MockResult(vectors)


def _make_embedder_factory() -> EmbedderFactory:
    """Create a mock embedder factory."""
    factory = MagicMock(spec=EmbedderFactory)
    factory.create_embedder.return_value = _MockEmbedder()
    return factory


def _make_embedding_cache_provider() -> EmbeddingCacheProvider:
    """Create a mock embedding cache provider that always misses."""
    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    provider = MagicMock(return_value=cache)
    return provider


def _make_database() -> PostgresVectorDatabase:
    """Create a PostgresVectorDatabase for testing."""
    return PostgresVectorDatabase(
        connection_string="postgresql://localhost/test",
        logger=_TestLogger(),
        tracer=MagicMock(),
        embedder_factory=_make_embedder_factory(),
        embedding_cache_provider=_make_embedding_cache_provider(),
    )


def _test_document(
    *,
    doc_id: str = "doc-1",
    content: str = "test content",
    name: str = "Test Document",
) -> _TestDocument:
    return _TestDocument(
        id=ObjectId(doc_id),
        version=Version.String("0.1.0"),
        content=content,
        checksum=_compute_checksum(content),
        name=name,
    )


# _WhereTranslator Tests


def test_where_translator_renders_eq_operator() -> None:
    filters: Where = cast(Where, {"name": {"$eq": "test"}})
    translator = _WhereTranslator()
    clause = translator.render(filters)

    assert "data->>'name' = $1" in clause
    assert translator.params[0] == "test"


def test_where_translator_renders_comparison_operators() -> None:
    filters: Where = cast(
        Where,
        {
            "$and": [
                {"value": {"$gt": 10}},
                {"value": {"$lt": 100}},
                {"value": {"$gte": 5}},
                {"value": {"$lte": 200}},
                {"value": {"$ne": 50}},
            ]
        },
    )
    translator = _WhereTranslator()
    clause = translator.render(filters)

    assert "data->>'value' > $1" in clause
    assert "data->>'value' < $2" in clause
    assert "data->>'value' >= $3" in clause
    assert "data->>'value' <= $4" in clause
    assert "data->>'value' != $5" in clause
    # Values are converted to strings for JSONB text extraction compatibility
    assert translator.params == ["10", "100", "5", "200", "50"]


def test_where_translator_renders_in_operator() -> None:
    filters: Where = cast(Where, {"status": {"$in": ["active", "pending", "completed"]}})
    translator = _WhereTranslator()
    clause = translator.render(filters)

    assert "data->>'status' IN ($1, $2, $3)" in clause
    assert translator.params == ["active", "pending", "completed"]


def test_where_translator_renders_nin_operator() -> None:
    filters: Where = cast(Where, {"status": {"$nin": ["deleted", "archived"]}})
    translator = _WhereTranslator()
    clause = translator.render(filters)

    assert "data->>'status' NOT IN ($1, $2)" in clause
    assert translator.params == ["deleted", "archived"]


def test_where_translator_handles_empty_in() -> None:
    filters: Where = cast(Where, {"status": {"$in": []}})
    translator = _WhereTranslator()
    clause = translator.render(filters)

    assert "FALSE" in clause


def test_where_translator_handles_empty_nin() -> None:
    filters: Where = cast(Where, {"status": {"$nin": []}})
    translator = _WhereTranslator()
    clause = translator.render(filters)

    assert "TRUE" in clause


def test_where_translator_renders_and_operator() -> None:
    filters: Where = cast(
        Where,
        {
            "$and": [
                {"name": {"$eq": "test"}},
                {"status": {"$eq": "active"}},
            ]
        },
    )
    translator = _WhereTranslator()
    clause = translator.render(filters)

    assert "AND" in clause
    assert "data->>'name' = $1" in clause
    assert "data->>'status' = $2" in clause
    assert translator.params == ["test", "active"]


def test_where_translator_renders_or_operator() -> None:
    filters: Where = cast(
        Where,
        {
            "$or": [
                {"name": {"$eq": "test1"}},
                {"name": {"$eq": "test2"}},
            ]
        },
    )
    translator = _WhereTranslator()
    clause = translator.render(filters)

    assert "OR" in clause
    assert "data->>'name' = $1" in clause
    assert "data->>'name' = $2" in clause


def test_where_translator_renders_nested_and_or() -> None:
    filters: Where = cast(
        Where,
        {
            "$or": [
                {"agent_id": {"$eq": "agent-1"}},
                {
                    "$and": [
                        {"customer_id": {"$eq": "cust-9"}},
                        {"tag_id": {"$in": ["alpha", "beta"]}},
                    ]
                },
            ]
        },
    )
    translator = _WhereTranslator()
    clause = translator.render(filters)

    assert "OR" in clause
    assert "AND" in clause
    assert "data->>'agent_id'" in clause
    assert "data->>'customer_id'" in clause
    assert "data->>'tag_id'" in clause


def test_where_translator_with_param_offset() -> None:
    filters: Where = cast(Where, {"name": {"$eq": "test"}})
    translator = _WhereTranslator(param_offset=2)
    clause = translator.render(filters)

    assert "data->>'name' = $3" in clause
    assert translator.params[0] == "test"


def test_where_translator_renders_empty_filters() -> None:
    translator = _WhereTranslator()
    clause = translator.render({})

    assert clause == "TRUE"


# PostgresVectorCollection Tests


@pytest.mark.asyncio
async def test_insert_one_stores_document(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresVectorCollection(
        database=db,
        logger=_TestLogger(),
        tracer=MagicMock(),
        name="test",
        schema=_TestDocument,
        unembedded_table="test_unembedded",
        embedded_table="test_MockEmbedder",
        embedder=_MockEmbedder(),
        embedding_cache_provider=_make_embedding_cache_provider(),
    )

    execute_calls: list[tuple[str, Any]] = []

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        execute_calls.append((sql, params))
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)

    document = _test_document(doc_id="doc-1", content="hello world")
    await collection.insert_one(document)

    assert len(execute_calls) == 2

    # Check unembedded table insert
    unembedded_sql, unembedded_params = execute_calls[0]
    assert "INSERT INTO" in unembedded_sql
    assert "test_unembedded" in unembedded_sql
    assert unembedded_params[0] == "doc-1"
    assert "hello world" in unembedded_params[2]

    # Check embedded table insert
    embedded_sql, embedded_params = execute_calls[1]
    assert "INSERT INTO" in embedded_sql
    assert "test_MockEmbedder" in embedded_sql
    assert isinstance(embedded_params[5], list)  # embedding vector


@pytest.mark.asyncio
async def test_find_returns_matching_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresVectorCollection(
        database=db,
        logger=_TestLogger(),
        tracer=MagicMock(),
        name="test",
        schema=_TestDocument,
        unembedded_table="test_unembedded",
        embedded_table="test_MockEmbedder",
        embedder=_MockEmbedder(),
        embedding_cache_provider=_make_embedding_cache_provider(),
    )

    doc1 = _test_document(doc_id="doc-1", name="Alice")
    doc2 = _test_document(doc_id="doc-2", name="Bob")

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        if fetch == "all":
            return [{"data": doc1}, {"data": doc2}]
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)

    results = await collection.find({})

    assert len(results) == 2
    assert results[0]["id"] == ObjectId("doc-1")
    assert results[1]["id"] == ObjectId("doc-2")


@pytest.mark.asyncio
async def test_find_with_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresVectorCollection(
        database=db,
        logger=_TestLogger(),
        tracer=MagicMock(),
        name="test",
        schema=_TestDocument,
        unembedded_table="test_unembedded",
        embedded_table="test_MockEmbedder",
        embedder=_MockEmbedder(),
        embedding_cache_provider=_make_embedding_cache_provider(),
    )

    captured_sql: list[str] = []

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        captured_sql.append(sql)
        return []

    monkeypatch.setattr(db, "_execute", fake_execute)

    await collection.find({"name": {"$eq": "Alice"}})

    assert len(captured_sql) == 1
    assert "WHERE" in captured_sql[0]
    assert "data->>'name'" in captured_sql[0]


@pytest.mark.asyncio
async def test_find_one_returns_single_document(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresVectorCollection(
        database=db,
        logger=_TestLogger(),
        tracer=MagicMock(),
        name="test",
        schema=_TestDocument,
        unembedded_table="test_unembedded",
        embedded_table="test_MockEmbedder",
        embedder=_MockEmbedder(),
        embedding_cache_provider=_make_embedding_cache_provider(),
    )

    doc = _test_document(doc_id="doc-1")

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        if fetch == "one":
            return {"data": doc}
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)

    result = await collection.find_one({"id": {"$eq": "doc-1"}})

    assert result is not None
    assert result["id"] == ObjectId("doc-1")


@pytest.mark.asyncio
async def test_find_one_returns_none_when_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_database()
    collection = PostgresVectorCollection(
        database=db,
        logger=_TestLogger(),
        tracer=MagicMock(),
        name="test",
        schema=_TestDocument,
        unembedded_table="test_unembedded",
        embedded_table="test_MockEmbedder",
        embedder=_MockEmbedder(),
        embedding_cache_provider=_make_embedding_cache_provider(),
    )

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)

    result = await collection.find_one({"id": {"$eq": "nonexistent"}})

    assert result is None


@pytest.mark.asyncio
async def test_update_one_updates_existing_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_database()
    collection = PostgresVectorCollection(
        database=db,
        logger=_TestLogger(),
        tracer=MagicMock(),
        name="test",
        schema=_TestDocument,
        unembedded_table="test_unembedded",
        embedded_table="test_MockEmbedder",
        embedder=_MockEmbedder(),
        embedding_cache_provider=_make_embedding_cache_provider(),
    )

    existing_doc = _test_document(doc_id="doc-1", name="Old Name")
    execute_calls: list[tuple[str, Any]] = []

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        execute_calls.append((sql, params))
        if fetch == "one" and "SELECT" in sql:
            return {"data": existing_doc}
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)

    result = await collection.update_one(
        {"id": {"$eq": "doc-1"}},
        cast(_TestDocument, {"name": "New Name"}),
    )

    assert result.matched_count == 1
    assert result.modified_count == 1
    assert result.updated_document is not None
    assert result.updated_document["name"] == "New Name"


@pytest.mark.asyncio
async def test_update_one_with_upsert_inserts_new_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_database()
    collection = PostgresVectorCollection(
        database=db,
        logger=_TestLogger(),
        tracer=MagicMock(),
        name="test",
        schema=_TestDocument,
        unembedded_table="test_unembedded",
        embedded_table="test_MockEmbedder",
        embedder=_MockEmbedder(),
        embedding_cache_provider=_make_embedding_cache_provider(),
    )

    execute_calls: list[tuple[str, Any]] = []

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        execute_calls.append((sql, params))
        if fetch == "one" and "SELECT" in sql:
            return None  # No existing document
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)

    new_doc = _test_document(doc_id="new-doc", content="new content", name="New Doc")
    result = await collection.update_one(
        {"id": {"$eq": "new-doc"}},
        new_doc,
        upsert=True,
    )

    assert result.matched_count == 0
    assert result.updated_document is not None
    assert result.updated_document["id"] == ObjectId("new-doc")

    # Verify INSERT was called
    insert_calls = [call for call in execute_calls if "INSERT" in call[0]]
    assert len(insert_calls) == 2  # unembedded and embedded


@pytest.mark.asyncio
async def test_delete_one_removes_document(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresVectorCollection(
        database=db,
        logger=_TestLogger(),
        tracer=MagicMock(),
        name="test",
        schema=_TestDocument,
        unembedded_table="test_unembedded",
        embedded_table="test_MockEmbedder",
        embedder=_MockEmbedder(),
        embedding_cache_provider=_make_embedding_cache_provider(),
    )

    doc = _test_document(doc_id="doc-to-delete")
    delete_calls: list[str] = []

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        if "DELETE" in sql:
            delete_calls.append(sql)
        if fetch == "one" and "SELECT" in sql:
            return {"data": doc}
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)

    result = await collection.delete_one({"id": {"$eq": "doc-to-delete"}})

    assert result.deleted_count == 1
    assert result.deleted_document is not None
    assert result.deleted_document["id"] == ObjectId("doc-to-delete")
    assert len(delete_calls) == 2  # unembedded and embedded


@pytest.mark.asyncio
async def test_delete_one_returns_zero_when_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_database()
    collection = PostgresVectorCollection(
        database=db,
        logger=_TestLogger(),
        tracer=MagicMock(),
        name="test",
        schema=_TestDocument,
        unembedded_table="test_unembedded",
        embedded_table="test_MockEmbedder",
        embedder=_MockEmbedder(),
        embedding_cache_provider=_make_embedding_cache_provider(),
    )

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)

    result = await collection.delete_one({"id": {"$eq": "nonexistent"}})

    assert result.deleted_count == 0
    assert result.deleted_document is None


@pytest.mark.asyncio
async def test_do_find_similar_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresVectorCollection(
        database=db,
        logger=_TestLogger(),
        tracer=MagicMock(),
        name="test",
        schema=_TestDocument,
        unembedded_table="test_unembedded",
        embedded_table="test_MockEmbedder",
        embedder=_MockEmbedder(),
        embedding_cache_provider=_make_embedding_cache_provider(),
    )

    doc1 = _test_document(doc_id="doc-1", content="apple", name="Apple")
    doc2 = _test_document(doc_id="doc-2", content="banana", name="Banana")

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        if fetch == "all" and "embedding <=>" in sql:
            return [
                {"data": doc1, "distance": 0.1},
                {"data": doc2, "distance": 0.2},
            ]
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)

    results = await collection.do_find_similar_documents(
        filters={},
        query="fruit",
        k=2,
    )

    assert len(results) == 2
    assert results[0].document["id"] == ObjectId("doc-1")
    assert results[0].distance == 0.1
    assert results[1].document["id"] == ObjectId("doc-2")
    assert results[1].distance == 0.2


@pytest.mark.asyncio
async def test_do_find_similar_documents_with_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_database()
    collection = PostgresVectorCollection(
        database=db,
        logger=_TestLogger(),
        tracer=MagicMock(),
        name="test",
        schema=_TestDocument,
        unembedded_table="test_unembedded",
        embedded_table="test_MockEmbedder",
        embedder=_MockEmbedder(),
        embedding_cache_provider=_make_embedding_cache_provider(),
    )

    captured_sql: list[str] = []
    captured_params: list[Any] = []

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        captured_sql.append(sql)
        captured_params.append(params)
        return []

    monkeypatch.setattr(db, "_execute", fake_execute)

    await collection.do_find_similar_documents(
        filters={"name": {"$eq": "Apple"}},
        query="fruit",
        k=2,
    )

    assert len(captured_sql) == 1
    assert "embedding <=>" in captured_sql[0]
    assert "data->>'name'" in captured_sql[0]
    # Params should be: [query_embedding, k, "Apple"]
    assert len(captured_params[0]) == 3
    assert captured_params[0][1] == 2  # k
    assert captured_params[0][2] == "Apple"  # filter value


# PostgresVectorDatabase Tests


@pytest.mark.asyncio
async def test_create_collection_creates_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()

    execute_calls: list[str] = []

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        execute_calls.append(sql)
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)
    db._pool = MagicMock()  # Fake pool

    collection = await db.create_collection(
        "test_collection",
        _TestDocument,
        _MockEmbedder,
    )

    assert collection is not None

    # Verify table creation
    create_table_calls = [sql for sql in execute_calls if "CREATE TABLE" in sql]
    assert len(create_table_calls) >= 2  # unembedded and embedded

    # Verify index creation
    create_index_calls = [sql for sql in execute_calls if "CREATE INDEX" in sql]
    assert len(create_index_calls) >= 2  # HNSW and GIN indexes


@pytest.mark.asyncio
async def test_get_collection_raises_when_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_database()

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        if fetch == "one" and "information_schema" in sql:
            return {"exists": False}
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)
    db._pool = MagicMock()

    with pytest.raises(ValueError) as exc_info:
        await db.get_collection(
            "nonexistent",
            _TestDocument,
            _MockEmbedder,
            AsyncMock(return_value=None),
        )

    assert "not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_delete_collection_drops_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()

    # Pre-populate the collections dict
    mock_collection = MagicMock()
    mock_collection._unembedded_table = "test_unembedded"
    mock_collection._embedded_table = "test_MockEmbedder"
    db._collections["test"] = mock_collection

    execute_calls: list[str] = []

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        execute_calls.append(sql)
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)

    await db.delete_collection("test")

    drop_calls = [sql for sql in execute_calls if "DROP TABLE" in sql]
    assert len(drop_calls) == 3  # embedded, unembedded, and failed_migrations


@pytest.mark.asyncio
async def test_delete_collection_raises_when_not_found() -> None:
    db = _make_database()

    with pytest.raises(ValueError) as exc_info:
        await db.delete_collection("nonexistent")

    assert "not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_upsert_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()

    captured_sql: list[str] = []
    captured_params: list[Any] = []

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        captured_sql.append(sql)
        captured_params.append(params)
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)

    await db.upsert_metadata("test_key", {"value": 123})

    assert len(captured_sql) == 1
    assert "INSERT INTO" in captured_sql[0]
    assert "ON CONFLICT" in captured_sql[0]
    assert captured_params[0][0] == "test_key"


@pytest.mark.asyncio
async def test_read_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        if fetch == "all":
            # With JSON codec registered, asyncpg returns JSONB as decoded Python objects
            return [
                {"key": "key1", "value": "value1"},
                {"key": "key2", "value": {"nested": True}},
            ]
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)

    metadata = await db.read_metadata()

    assert metadata["key1"] == "value1"
    assert metadata["key2"] == {"nested": True}


@pytest.mark.asyncio
async def test_remove_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()

    captured_sql: list[str] = []

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        captured_sql.append(sql)
        if fetch == "one" and "DELETE" in sql:
            return {"key": "deleted_key"}
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)

    await db.remove_metadata("deleted_key")

    assert len(captured_sql) == 1
    assert "DELETE" in captured_sql[0]


@pytest.mark.asyncio
async def test_remove_metadata_raises_when_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_database()

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)

    with pytest.raises(ValueError) as exc_info:
        await db.remove_metadata("nonexistent")

    assert "not found" in str(exc_info.value)


# Checksum Tests


def test_compute_checksum_deterministic() -> None:
    content1 = "hello world"
    content2 = "hello world"

    checksum1 = _compute_checksum(content1)
    checksum2 = _compute_checksum(content2)

    assert checksum1 == checksum2


def test_compute_checksum_unique_for_different_content() -> None:
    checksum1 = _compute_checksum("hello")
    checksum2 = _compute_checksum("world")

    assert checksum1 != checksum2


# ==============================================================================
# Integration Tests - Require actual PostgreSQL with pgvector
# ==============================================================================
# These tests verify the actual asyncpg behavior that unit tests with mocks miss.
# Run with: TEST_POSTGRES_DSN=postgresql://user:pass@localhost/test pytest -k integration
# ==============================================================================

import os


def _get_test_dsn() -> str | None:
    """Get PostgreSQL DSN from environment, or None if not configured."""
    return os.environ.get("TEST_POSTGRES_DSN")


@pytest.fixture
async def integration_db() -> Any:
    """Create a real PostgreSQL database connection for integration tests."""
    dsn = _get_test_dsn()
    if dsn is None:
        pytest.skip("TEST_POSTGRES_DSN not set")

    db = PostgresVectorDatabase(
        connection_string=dsn,
        logger=_TestLogger(),
        tracer=MagicMock(),
        table_prefix="test_vec_",
        embedder_factory=_make_embedder_factory(),
        embedding_cache_provider=_make_embedding_cache_provider(),
    )

    async with db:
        yield db

        # Cleanup: drop test tables
        for name in list(db._collections.keys()):
            try:
                await db.delete_collection(name)
            except Exception:
                pass


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_insert_and_read_document_roundtrip(integration_db: Any) -> None:
    """Test that documents survive a round-trip through PostgreSQL JSONB."""
    collection = await integration_db.create_collection(
        "roundtrip_test",
        _TestDocument,
        _MockEmbedder,
    )

    doc = _test_document(doc_id="rt-1", content="hello world", name="Test Doc")
    await collection.insert_one(doc)

    result = await collection.find_one({"id": {"$eq": "rt-1"}})

    assert result is not None
    assert result["id"] == ObjectId("rt-1")
    assert result["name"] == "Test Doc"
    assert result["content"] == "hello world"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_document_with_integer_version(integration_db: Any) -> None:
    """Test that integer version fields are properly converted to strings."""
    collection = await integration_db.create_collection(
        "int_version_test",
        _TestDocument,
        _MockEmbedder,
    )

    # Create document with integer version (common case)
    doc: _TestDocument = {
        "id": ObjectId("iv-1"),
        "version": cast(Version.String, 1),  # Integer version
        "content": "test content",
        "checksum": _compute_checksum("test content"),
        "name": "Int Version Doc",
    }

    # This should not raise DataError
    await collection.insert_one(doc)

    result = await collection.find_one({"id": {"$eq": "iv-1"}})
    assert result is not None
    assert result["id"] == ObjectId("iv-1")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_data_persists_after_reconnect(integration_db: Any) -> None:
    """Test that data persists after closing and reopening connection."""
    dsn = _get_test_dsn()
    assert dsn is not None

    # Insert document
    collection = await integration_db.create_collection(
        "persist_test",
        _TestDocument,
        _MockEmbedder,
    )
    doc = _test_document(doc_id="persist-1", content="persistent data")
    await collection.insert_one(doc)

    # Create new database connection
    db2 = PostgresVectorDatabase(
        connection_string=dsn,
        logger=_TestLogger(),
        tracer=MagicMock(),
        table_prefix="test_vec_",
        embedder_factory=_make_embedder_factory(),
        embedding_cache_provider=_make_embedding_cache_provider(),
    )

    async with db2:
        # Load existing collection
        collection2 = await db2.get_or_create_collection(
            "persist_test",
            _TestDocument,
            _MockEmbedder,
            AsyncMock(side_effect=lambda d: d),  # Identity loader
        )

        result = await collection2.find_one({"id": {"$eq": "persist-1"}})
        assert result is not None
        assert result["content"] == "persistent data"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_similar_document_search(integration_db: Any) -> None:
    """Test vector similarity search works correctly."""
    collection = await integration_db.create_collection(
        "search_test",
        _TestDocument,
        _MockEmbedder,
    )

    # Insert multiple documents
    for i in range(3):
        doc = _test_document(
            doc_id=f"search-{i}",
            content=f"document number {i}",
            name=f"Doc {i}",
        )
        await collection.insert_one(doc)

    # Search for similar documents
    results = await collection.find_similar_documents(
        filters={},
        query="document number 1",
        k=2,
    )

    assert len(results) == 2
    # Results should have distance scores
    assert all(hasattr(r, "distance") for r in results)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_update_document(integration_db: Any) -> None:
    """Test that document updates work correctly."""
    collection = await integration_db.create_collection(
        "update_test",
        _TestDocument,
        _MockEmbedder,
    )

    doc = _test_document(doc_id="upd-1", content="original", name="Original Name")
    await collection.insert_one(doc)

    # Update the document
    result = await collection.update_one(
        {"id": {"$eq": "upd-1"}},
        cast(_TestDocument, {"name": "Updated Name", "content": "updated content"}),
    )

    assert result.matched_count == 1
    assert result.modified_count == 1

    # Verify update
    updated = await collection.find_one({"id": {"$eq": "upd-1"}})
    assert updated is not None
    assert updated["name"] == "Updated Name"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_delete_document(integration_db: Any) -> None:
    """Test that document deletion works correctly."""
    collection = await integration_db.create_collection(
        "delete_test",
        _TestDocument,
        _MockEmbedder,
    )

    doc = _test_document(doc_id="del-1", content="to be deleted")
    await collection.insert_one(doc)

    # Verify it exists
    found = await collection.find_one({"id": {"$eq": "del-1"}})
    assert found is not None

    # Delete it
    result = await collection.delete_one({"id": {"$eq": "del-1"}})
    assert result.deleted_count == 1

    # Verify it's gone
    gone = await collection.find_one({"id": {"$eq": "del-1"}})
    assert gone is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_metadata_roundtrip(integration_db: Any) -> None:
    """Test that metadata survives a round-trip."""
    await integration_db.upsert_metadata("test_key", {"nested": {"value": 123}})

    metadata = await integration_db.read_metadata()

    assert "test_key" in metadata
    assert metadata["test_key"] == {"nested": {"value": 123}}

    # Cleanup
    await integration_db.remove_metadata("test_key")
