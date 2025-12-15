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

import json
import os
from typing import Any, Mapping
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import fixture

from parlant.adapters.db.supabase_db import (
    SupabaseDocumentCollection,
    SupabaseDocumentDatabase,
    SupabaseAdapterError,
)
from parlant.core.agents import AgentId
from parlant.core.common import Version
from parlant.core.customers import CustomerId
from parlant.core.persistence.common import Cursor, ObjectId, SortDirection
from parlant.core.persistence.document_database import FindResult, InsertResult
from parlant.core.sessions import _SessionDocument
from tests.test_utilities import _TestLogger


_SUPABASE_PARAMS: Mapping[str, Any] = {
    "url": "https://test.supabase.co",
    "key": "test-key",
    "schema": "public",
}


@fixture(scope="module", autouse=True)
def require_supabase_test_flag() -> None:
    if not os.environ.get("TEST_SUPABASE_SERVER"):
        print("could not find `TEST_SUPABASE_SERVER` in environment, skipping supabase tests...")
        pytest.skip("Supabase tests require TEST_SUPABASE_SERVER env variable")


def _make_database() -> SupabaseDocumentDatabase:
    return SupabaseDocumentDatabase(
        logger=_TestLogger(),
        connection_params=_SUPABASE_PARAMS,
        client_factory=lambda *_: MagicMock(),
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


@pytest.mark.asyncio
async def test_insert_one_serializes_document_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SupabaseDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    # Mock the Supabase client
    mock_table = MagicMock()
    mock_insert = MagicMock()
    mock_table.insert.return_value = mock_insert
    mock_insert.execute.return_value = MagicMock(data=[])

    db._client = MagicMock()
    db._client.table.return_value = mock_table

    document = _session_document()

    await collection.insert_one(document)

    # Verify insert was called
    mock_table.insert.assert_called_once()
    insert_params = mock_table.insert.call_args[0][0]
    assert insert_params["id"] == "session-1"
    assert json.loads(json.dumps(insert_params["data"])) == document


@pytest.mark.asyncio
async def test_find_uses_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SupabaseDocumentCollection(db, "events", _SessionDocument, _TestLogger())

    # Mock response
    mock_response = MagicMock()
    mock_response.data = [{"data": {"id": "1", "creation_utc": "2025-01-01T00:00:00Z"}}]

    mock_table = MagicMock()
    mock_query = MagicMock()
    mock_table.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.execute.return_value = mock_response

    db._client = MagicMock()
    db._client.table.return_value = mock_table

    result = await collection.find({"session_id": {"$eq": "abc"}})

    assert isinstance(result, FindResult)
    assert len(result.items) == 1
    assert result.items[0]["id"] == "1"


@pytest.mark.asyncio
async def test_find_paginates_and_sets_next_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SupabaseDocumentCollection(db, "events", _SessionDocument, _TestLogger())

    rows = [
        {"data": {"id": "1", "creation_utc": "2025-01-01T00:00:00Z"}},
        {"data": {"id": "2", "creation_utc": "2025-01-02T00:00:00Z"}},
    ]

    mock_response = MagicMock()
    mock_response.data = rows

    mock_table = MagicMock()
    mock_query = MagicMock()
    mock_table.select.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.execute.return_value = mock_response

    db._client = MagicMock()
    db._client.table.return_value = mock_table

    result = await collection.find({}, limit=1)

    assert len(result.items) == 1
    assert result.has_more is True
    assert result.next_cursor == Cursor(creation_utc="2025-01-01T00:00:00Z", id=ObjectId("1"))
    assert result.total_count == 2


@pytest.mark.asyncio
async def test_find_adds_cursor_clause(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SupabaseDocumentCollection(db, "events", _SessionDocument, _TestLogger())

    mock_response = MagicMock()
    mock_response.data = []

    mock_table = MagicMock()
    mock_query = MagicMock()
    mock_table.select.return_value = mock_query
    mock_query.or_.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.execute.return_value = mock_response

    db._client = MagicMock()
    db._client.table.return_value = mock_table

    cursor = Cursor(creation_utc="2025-01-03", id=ObjectId("abc"))
    await collection.find({}, cursor=cursor, sort_direction=SortDirection.DESC)

    # Verify or_ was called for cursor pagination
    mock_query.or_.assert_called()


@pytest.mark.asyncio
async def test_update_one_upserts_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SupabaseDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    monkeypatch.setattr(collection, "find_one", AsyncMock(return_value=None))
    insert_mock = AsyncMock(return_value=InsertResult(True))
    monkeypatch.setattr(collection, "insert_one", insert_mock)

    payload = _session_document(doc_id="session-9", customer_id="customer-9", agent_id="agent-9")

    result = await collection.update_one({"id": {"$eq": "session-9"}}, payload, upsert=True)

    insert_mock.assert_awaited_once()
    assert result.updated_document == payload


@pytest.mark.asyncio
async def test_load_existing_documents_migrates(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SupabaseDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    mock_response = MagicMock()
    mock_response.data = [{"data": {"id": "abc", "version": "0.1"}}]

    mock_table = MagicMock()
    mock_query = MagicMock()
    mock_table.select.return_value = mock_query
    mock_query.execute.return_value = mock_response

    db._client = MagicMock()
    db._client.table.return_value = mock_table

    replace_mock = AsyncMock()
    monkeypatch.setattr(collection, "_replace_document", replace_mock)
    monkeypatch.setattr(collection, "_persist_failed_documents", AsyncMock())
    monkeypatch.setattr(collection, "_delete_documents", AsyncMock())

    async def loader(doc: Any) -> _SessionDocument:
        return _session_document(doc_id=str(doc["id"]))

    await collection.load_existing_documents(loader)

    replace_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_existing_documents_persists_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SupabaseDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    mock_response = MagicMock()
    mock_response.data = [{"data": {"id": "bad", "version": "0.7.0"}}]

    mock_table = MagicMock()
    mock_query = MagicMock()
    mock_table.select.return_value = mock_query
    mock_query.execute.return_value = mock_response

    db._client = MagicMock()
    db._client.table.return_value = mock_table

    delete_mock = AsyncMock()
    monkeypatch.setattr(collection, "_delete_documents", delete_mock)

    async def loader(_: Any) -> _SessionDocument | None:
        return None

    await collection.load_existing_documents(loader)

    delete_mock.assert_awaited_once_with(["bad"])


@pytest.mark.asyncio
async def test_delete_one_removes_document(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SupabaseDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

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
    collection = SupabaseDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    monkeypatch.setattr(collection, "find_one", AsyncMock(return_value=None))
    delete_mock = AsyncMock()
    monkeypatch.setattr(collection, "_delete_documents", delete_mock)

    result = await collection.delete_one({"id": {"$eq": "missing"}})

    delete_mock.assert_not_called()
    assert result.deleted_count == 0
    assert result.deleted_document is None


# Note: ensure_table and _execute_sql methods are not part of the current implementation
# Tables must be created manually in Supabase. These tests are removed.


@pytest.mark.asyncio
async def test_connection_params_from_env() -> None:
    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_KEY": "test-key",
        },
    ):
        from parlant.adapters.db.supabase_db import _load_connection_params_from_env

        params = _load_connection_params_from_env()
        assert params["url"] == "https://test.supabase.co"
        assert params["key"] == "test-key"
        assert params["schema"] == "public"


@pytest.mark.asyncio
async def test_connection_params_missing_raises_error() -> None:
    with patch.dict(os.environ, {}, clear=True):
        from parlant.adapters.db.supabase_db import _load_connection_params_from_env

        with pytest.raises(SupabaseAdapterError) as exc_info:
            _load_connection_params_from_env()

        assert "Missing Supabase configuration" in str(exc_info.value)


@pytest.mark.asyncio
async def test_find_one_returns_none_when_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SupabaseDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    mock_response = MagicMock()
    mock_response.data = []

    mock_table = MagicMock()
    mock_query = MagicMock()
    mock_table.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.execute.return_value = mock_response

    db._client = MagicMock()
    db._client.table.return_value = mock_table

    result = await collection.find_one({"id": {"$eq": "missing"}})

    assert result is None


@pytest.mark.asyncio
async def test_retry_on_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that retry logic handles connection errors."""
    from parlant.adapters.db.supabase_db import _retry_on_connection_error
    import httpx
    
    call_count = 0
    
    def failing_func() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.ReadTimeout("The read operation timed out")
        return "success"
    
    result = _retry_on_connection_error(failing_func, max_retries=5)
    
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_jsonb_field_filtering_in_or_condition(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that JSONB fields are properly filtered in OR conditions."""
    db = _make_database()
    collection = SupabaseDocumentCollection(db, "variable_tag_associations", _SessionDocument, _TestLogger())
    
    mock_response = MagicMock()
    mock_response.data = [{"data": {"tag_id": "tag-1", "variable_id": "var-1"}}]
    
    mock_table = MagicMock()
    mock_query = MagicMock()
    mock_table.select.return_value = mock_query
    mock_query.or_.return_value = mock_query
    mock_query.execute.return_value = mock_response
    
    db._client = MagicMock()
    db._client.table.return_value = mock_table
    
    # Test OR condition with JSONB field (tag_id is in data JSONB column)
    result = await collection.find({"$or": [{"tag_id": {"$eq": "tag-1"}}]})
    
    # Verify or_ was called (indicating JSONB field filtering was attempted)
    mock_query.or_.assert_called()
    assert len(result.items) == 1
