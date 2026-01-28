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
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from parlant.adapters.db.postgres_db import (
    PostgresDocumentCollection,
    PostgresDocumentDatabase,
    _WhereTranslator,
)
from parlant.core.agents import AgentId
from parlant.core.common import Version
from parlant.core.customers import CustomerId
from parlant.core.persistence.common import Cursor, ObjectId, SortDirection, Where
from parlant.core.persistence.document_database import FindResult, InsertResult
from parlant.core.sessions import _SessionDocument
from tests.test_utilities import _TestLogger


def _make_database() -> PostgresDocumentDatabase:
    return PostgresDocumentDatabase(
        connection_string="postgresql://localhost/test",
        logger=_TestLogger(),
    )


def _session_document(
    *,
    doc_id: str = "session-1",
    customer_id: str = "customer-1",
    agent_id: str = "agent-1",
) -> _SessionDocument:
    return {
        "id": ObjectId(doc_id),
        "version": Version.String("0.7.0"),
        "creation_utc": "2025-01-01T00:00:00Z",
        "customer_id": CustomerId(customer_id),
        "agent_id": AgentId(agent_id),
        "title": None,
        "mode": "auto",
        "consumption_offsets": {"client": 0},
        "agent_states": [],
        "metadata": {},
    }


def test_where_clause_supports_nested_or_and_in() -> None:
    filters: Where = cast(
        Where,
        {
            "$or": [
                {"agent_id": {"$eq": "agent-1"}},
                {
                    "$and": [
                        {"customer_id": {"$eq": "cust-9"}},
                        {"tag_id": {"$in": ["alpha", "beta"]}},
                        {"offset": {"$gte": 3}},
                    ]
                },
            ]
        },
    )

    translator = _WhereTranslator({"agent_id", "customer_id", "offset"})
    clause = translator.render(filters)
    params = translator.params

    assert "agent_id" in clause
    assert "data->>'tag_id'" in clause
    assert "offset >=" in clause
    assert params[0] == "agent-1"
    assert params[1] == "cust-9"
    assert params[2] == "alpha"
    assert params[3] == "beta"
    # Integer values are converted to strings for TEXT column compatibility
    assert params[4] == "3"


def test_where_clause_handles_comparisons() -> None:
    filters: Where = cast(
        Where,
        {
            "creation_utc": {"$lt": "2025-01-01"},
            "offset": {"$ne": 4},
            "$and": [
                {"offset": {"$lte": 10}},
                {"offset": {"$gt": 2}},
            ],
        },
    )

    translator = _WhereTranslator({"offset"})
    clause = translator.render(filters)
    params = translator.params

    assert "offset !=" in clause
    assert "offset <=" in clause
    assert "offset >" in clause
    assert "data->>'creation_utc' <" in clause
    assert params[0] == "2025-01-01"
    # Integer values are converted to strings for TEXT column compatibility
    assert params[1] == "4"
    assert params[2] == "10"
    assert params[3] == "2"


def test_where_clause_handles_nin() -> None:
    filters: Where = cast(
        Where,
        {"status": {"$nin": ["deleted", "archived"]}},
    )

    translator = _WhereTranslator({"status"})
    clause = translator.render(filters)
    params = translator.params

    assert "status NOT IN" in clause
    assert params[0] == "deleted"
    assert params[1] == "archived"


def test_where_clause_handles_empty_in() -> None:
    filters: Where = cast(
        Where,
        {"status": {"$in": []}},
    )

    translator = _WhereTranslator({"status"})
    clause = translator.render(filters)

    assert "FALSE" in clause


def test_where_clause_handles_empty_nin() -> None:
    filters: Where = cast(
        Where,
        {"status": {"$nin": []}},
    )

    translator = _WhereTranslator({"status"})
    clause = translator.render(filters)

    assert "TRUE" in clause


@pytest.mark.asyncio
async def test_insert_one_serializes_document_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    execute_mock = AsyncMock()
    monkeypatch.setattr(db, "_execute", execute_mock)

    document = _session_document()

    await collection.insert_one(document)

    sql, params = execute_mock.call_args[0][0], execute_mock.call_args[0][1]
    assert "INSERT INTO" in sql
    assert json.loads(params[3]) == document
    assert params[0] == "session-1"


@pytest.mark.asyncio
async def test_find_uses_sql_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresDocumentCollection(db, "events", _SessionDocument, _TestLogger())

    execute_mock = AsyncMock(return_value=[{"data": {"id": "1"}}])
    monkeypatch.setattr(db, "_execute", execute_mock)

    result = await collection.find({"session_id": {"$eq": "abc"}})

    assert isinstance(result, FindResult)
    assert result.items[0]["id"] == "1"
    sql = execute_mock.call_args[0][0]
    params = execute_mock.call_args[0][1]
    assert "WHERE data->>'session_id' =" in sql
    assert "ORDER BY creation_utc ASC, id ASC" in sql
    assert params[0] == "abc"


@pytest.mark.asyncio
async def test_find_paginates_and_sets_next_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresDocumentCollection(db, "events", _SessionDocument, _TestLogger())

    rows = [
        {"data": {"id": "1", "creation_utc": "2025-01-01"}},
        {"data": {"id": "2", "creation_utc": "2025-01-02"}},
    ]
    execute_mock = AsyncMock(return_value=rows)
    monkeypatch.setattr(db, "_execute", execute_mock)

    result = await collection.find({}, limit=1)

    assert len(result.items) == 1
    assert result.has_more is True
    assert result.next_cursor == Cursor(creation_utc="2025-01-01", id=ObjectId("1"))
    assert result.total_count == 2
    sql = execute_mock.call_args[0][0]
    assert "LIMIT 2" in sql


@pytest.mark.asyncio
async def test_find_adds_cursor_clause(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresDocumentCollection(db, "events", _SessionDocument, _TestLogger())

    execute_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(db, "_execute", execute_mock)

    cursor = Cursor(creation_utc="2025-01-03", id=ObjectId("abc"))
    await collection.find({}, cursor=cursor, sort_direction=SortDirection.DESC)

    sql = execute_mock.call_args[0][0]
    params = execute_mock.call_args[0][1]
    assert "ORDER BY creation_utc DESC, id DESC" in sql
    assert "creation_utc <" in sql
    assert params[0] == "2025-01-03"
    assert params[1] == "abc"


@pytest.mark.asyncio
async def test_update_one_upserts_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    monkeypatch.setattr(collection, "find_one", AsyncMock(return_value=None))
    insert_mock = AsyncMock(return_value=InsertResult(True))
    monkeypatch.setattr(collection, "insert_one", insert_mock)

    payload = _session_document(doc_id="session-9", customer_id="customer-9", agent_id="agent-9")

    result = await collection.update_one({"id": {"$eq": "session-9"}}, payload, upsert=True)

    insert_mock.assert_awaited_once()
    assert result.updated_document == payload


@pytest.mark.asyncio
async def test_update_one_updates_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    existing = _session_document(doc_id="session-1")
    monkeypatch.setattr(collection, "find_one", AsyncMock(return_value=existing))
    replace_mock = AsyncMock()
    monkeypatch.setattr(collection, "_replace_document", replace_mock)

    update_params = cast(_SessionDocument, {"title": "Updated Title"})
    result = await collection.update_one({"id": {"$eq": "session-1"}}, update_params)

    replace_mock.assert_awaited_once()
    assert result.matched_count == 1
    assert result.modified_count == 1
    assert result.updated_document is not None
    assert result.updated_document.get("title") == "Updated Title"


@pytest.mark.asyncio
async def test_load_existing_documents_migrates(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    monkeypatch.setattr(
        db, "_execute", AsyncMock(return_value=[{"data": {"id": "abc", "version": "0.1"}}])
    )
    replace_mock = AsyncMock()
    monkeypatch.setattr(collection, "_replace_document", replace_mock)
    monkeypatch.setattr(collection, "_persist_failed_documents", AsyncMock())
    monkeypatch.setattr(collection, "_delete_documents", AsyncMock())

    async def loader(doc: Any) -> _SessionDocument:
        return _session_document(doc_id=str(doc["id"]))

    await db._load_documents_with_loader(collection, loader)

    replace_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_existing_documents_persists_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    calls: list[tuple[str, Any, str]] = []

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        calls.append((sql, params, fetch))
        if sql.startswith("SELECT data"):
            return [{"data": {"id": "bad", "version": "0.7.0"}}]
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)
    delete_mock = AsyncMock()
    monkeypatch.setattr(collection, "_delete_documents", delete_mock)

    async def loader(_: Any) -> _SessionDocument | None:
        return None

    await db._load_documents_with_loader(collection, loader)

    assert any("INSERT INTO" in sql and "failed_migrations" in sql for sql, _, _ in calls)
    delete_mock.assert_awaited_once_with(["bad"])


@pytest.mark.asyncio
async def test_delete_one_removes_document(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    doc = _session_document(doc_id="to-delete")
    monkeypatch.setattr(collection, "find_one", AsyncMock(return_value=doc))
    delete_mock = AsyncMock()
    monkeypatch.setattr(collection, "_delete_documents", delete_mock)

    result = await collection.delete_one({"id": {"$eq": "to-delete"}})

    delete_mock.assert_awaited_once_with([ObjectId("to-delete")])
    assert result.deleted_count == 1
    assert result.deleted_document == doc


@pytest.mark.asyncio
async def test_delete_one_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    monkeypatch.setattr(collection, "find_one", AsyncMock(return_value=None))
    delete_mock = AsyncMock()
    monkeypatch.setattr(collection, "_delete_documents", delete_mock)

    result = await collection.delete_one({"id": {"$eq": "missing"}})

    delete_mock.assert_not_called()
    assert result.deleted_count == 0
    assert result.deleted_document is None


@pytest.mark.asyncio
async def test_get_collection_initializes_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()

    collection = AsyncMock()
    collection._table = '"parlant_sessions"'
    collection._table_name = "parlant_sessions"
    collection._failed_table = '"parlant_sessions_failed_migrations"'

    db._collections["sessions"] = collection  # type: ignore[assignment]
    loader = AsyncMock(return_value=None)

    execute_mock = AsyncMock()
    monkeypatch.setattr(db, "_execute", execute_mock)

    load_mock = AsyncMock()
    monkeypatch.setattr(db, "_load_documents_with_loader", load_mock)

    await db.get_collection("sessions", _SessionDocument, loader)
    await db.get_collection("sessions", _SessionDocument, loader)

    # initialization is performed once (tables + indexes created once + loader run once)
    # 1 CREATE TABLE + 3 CREATE INDEX + 1 CREATE TABLE (failed) = 5 statements
    assert execute_mock.await_count == 5
    load_mock.assert_awaited_once_with(collection, loader)


@pytest.mark.asyncio
async def test_delete_collection_drops_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()

    execute_mock = AsyncMock()
    monkeypatch.setattr(db, "_execute", execute_mock)

    await db.delete_collection("sessions")

    drop_statements = [args.args[0] for args in execute_mock.await_args_list]
    assert any('DROP TABLE IF EXISTS "parlant_sessions"' in stmt for stmt in drop_statements)
    assert any(
        'DROP TABLE IF EXISTS "parlant_sessions_failed_migrations"' in stmt
        for stmt in drop_statements
    )


@pytest.mark.asyncio
async def test_get_collection_creates_base_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()

    execute_calls: list[str] = []

    async def fake_execute(sql: str, *_args: Any, **_kwargs: Any) -> None:
        execute_calls.append(sql)
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)
    monkeypatch.setattr(db, "_load_documents_with_loader", AsyncMock())

    await db.get_collection("sessions", _SessionDocument, AsyncMock(return_value=None))

    assert any(
        "CREATE TABLE IF NOT EXISTS" in sql and "id TEXT NOT NULL PRIMARY KEY" in sql
        for sql in execute_calls
    )
    assert any(
        "CREATE TABLE IF NOT EXISTS" in sql and "data JSONB NOT NULL" in sql
        for sql in execute_calls
    )


@pytest.mark.asyncio
async def test_get_collection_creates_indexes(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()

    execute_calls: list[str] = []

    async def fake_execute(sql: str, *_args: Any, **_kwargs: Any) -> None:
        execute_calls.append(sql)
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)
    monkeypatch.setattr(db, "_load_documents_with_loader", AsyncMock())

    await db.get_collection("sessions", _SessionDocument, AsyncMock(return_value=None))

    assert any(
        "CREATE INDEX IF NOT EXISTS" in sql and "creation_utc" in sql for sql in execute_calls
    )
    assert any("CREATE INDEX IF NOT EXISTS" in sql and "USING GIN" in sql for sql in execute_calls)


@pytest.mark.asyncio
async def test_find_one_returns_none_when_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    execute_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(db, "_execute", execute_mock)

    result = await collection.find_one({"id": {"$eq": "nonexistent"}})

    assert result is None


@pytest.mark.asyncio
async def test_find_one_returns_document_when_found(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PostgresDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    doc = _session_document(doc_id="found-1")
    execute_mock = AsyncMock(return_value={"data": doc})
    monkeypatch.setattr(db, "_execute", execute_mock)

    result = await collection.find_one({"id": {"$eq": "found-1"}})

    assert result is not None
    assert result["id"] == ObjectId("found-1")


# ==============================================================================
# Integration Tests - Require actual PostgreSQL
# ==============================================================================
# These tests verify actual asyncpg behavior that unit tests with mocks miss.
# Run with: TEST_POSTGRES_DSN=postgresql://user:pass@localhost/test pytest -k integration
# ==============================================================================

import os
from typing import Any


def _get_test_dsn() -> str | None:
    """Get PostgreSQL DSN from environment, or None if not configured."""
    return os.environ.get("TEST_POSTGRES_DSN")


@pytest.fixture
async def integration_db() -> Any:
    """Create a real PostgreSQL database connection for integration tests."""
    dsn = _get_test_dsn()
    if dsn is None:
        pytest.skip("TEST_POSTGRES_DSN not set")

    db = PostgresDocumentDatabase(
        connection_string=dsn,
        logger=_TestLogger(),
        table_prefix="test_doc_",
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
    collection = await integration_db.create_collection("roundtrip_test", _SessionDocument)

    doc = _session_document(doc_id="rt-1", customer_id="cust-1", agent_id="agent-1")
    await collection.insert_one(doc)

    result = await collection.find_one({"id": {"$eq": "rt-1"}})

    assert result is not None
    assert result["id"] == ObjectId("rt-1")
    assert result["customer_id"] == CustomerId("cust-1")
    assert result["agent_id"] == AgentId("agent-1")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_document_with_integer_version(integration_db: Any) -> None:
    """Test that integer version fields are properly converted to strings."""
    collection = await integration_db.create_collection("int_version_test", _SessionDocument)

    # Create document with integer version
    doc: _SessionDocument = {
        "id": ObjectId("iv-1"),
        "version": cast(Version.String, 1),  # Integer version - should be converted
        "creation_utc": "2025-01-01T00:00:00Z",
        "customer_id": CustomerId("cust-1"),
        "agent_id": AgentId("agent-1"),
        "title": None,
        "mode": "auto",
        "consumption_offsets": {"client": 0},
        "agent_states": [],
        "metadata": {},
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
    collection = await integration_db.create_collection("persist_test", _SessionDocument)
    doc = _session_document(doc_id="persist-1")
    await collection.insert_one(doc)

    # Create new database connection
    db2 = PostgresDocumentDatabase(
        connection_string=dsn,
        logger=_TestLogger(),
        table_prefix="test_doc_",
    )

    async with db2:
        # Load existing collection
        collection2 = await db2.get_collection(
            "persist_test",
            _SessionDocument,
            AsyncMock(side_effect=lambda d: d),  # Identity loader
        )

        result = await collection2.find_one({"id": {"$eq": "persist-1"}})
        assert result is not None
        assert result["id"] == ObjectId("persist-1")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_find_with_filters(integration_db: Any) -> None:
    """Test that find with filters works correctly."""
    collection = await integration_db.create_collection("filter_test", _SessionDocument)

    # Insert multiple documents
    for i in range(3):
        doc = _session_document(
            doc_id=f"filter-{i}",
            customer_id=f"cust-{i}",
            agent_id="agent-1",
        )
        await collection.insert_one(doc)

    # Filter by customer_id
    result = await collection.find({"customer_id": {"$eq": "cust-1"}})
    assert len(result.items) == 1
    assert result.items[0]["customer_id"] == CustomerId("cust-1")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_update_document(integration_db: Any) -> None:
    """Test that document updates work correctly."""
    collection = await integration_db.create_collection("update_test", _SessionDocument)

    doc = _session_document(doc_id="upd-1")
    await collection.insert_one(doc)

    # Update the document
    result = await collection.update_one(
        {"id": {"$eq": "upd-1"}},
        cast(_SessionDocument, {"title": "Updated Title"}),
    )

    assert result.matched_count == 1
    assert result.modified_count == 1

    # Verify update
    updated = await collection.find_one({"id": {"$eq": "upd-1"}})
    assert updated is not None
    assert updated["title"] == "Updated Title"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_delete_document(integration_db: Any) -> None:
    """Test that document deletion works correctly."""
    collection = await integration_db.create_collection("delete_test", _SessionDocument)

    doc = _session_document(doc_id="del-1")
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
async def test_integration_filter_with_integer_value(integration_db: Any) -> None:
    """Test that filtering with integer values works (converted to strings)."""
    collection = await integration_db.create_collection("int_filter_test", _SessionDocument)

    doc = _session_document(doc_id="if-1")
    await collection.insert_one(doc)

    # Filter using integer value - should be converted to string internally
    result = await collection.find({"consumption_offsets": {"$eq": 0}})
    # This tests that integer filter values don't cause DataError


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_pagination(integration_db: Any) -> None:
    """Test that pagination works correctly."""
    collection = await integration_db.create_collection("pagination_test", _SessionDocument)

    # Insert multiple documents with different creation times
    for i in range(5):
        doc = _session_document(doc_id=f"page-{i}")
        doc["creation_utc"] = f"2025-01-0{i + 1}T00:00:00Z"
        await collection.insert_one(doc)

    # Get first page
    result1 = await collection.find({}, limit=2, sort_direction=SortDirection.ASC)
    assert len(result1.items) == 2
    assert result1.has_more is True
    assert result1.next_cursor is not None

    # Get second page
    result2 = await collection.find(
        {}, limit=2, cursor=result1.next_cursor, sort_direction=SortDirection.ASC
    )
    assert len(result2.items) == 2

    # Verify no overlap
    page1_ids = {item["id"] for item in result1.items}
    page2_ids = {item["id"] for item in result2.items}
    assert page1_ids.isdisjoint(page2_ids)
