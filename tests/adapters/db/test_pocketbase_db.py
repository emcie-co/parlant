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
from typing import Any, Mapping, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pytest import fixture

from parlant.adapters.db.pocketbase_db import (
    PocketBaseDocumentCollection,
    PocketBaseDocumentDatabase,
    _build_pocketbase_filter,
)
from parlant.core.agents import AgentId
from parlant.core.common import Version
from parlant.core.customers import CustomerId
from parlant.core.persistence.common import Cursor, ObjectId, SortDirection, Where
from parlant.core.persistence.document_database import FindResult, InsertResult
from parlant.core.sessions import _SessionDocument
from tests.test_utilities import _TestLogger


_POCKETBASE_PARAMS: Mapping[str, Any] = {
    "url": "http://localhost:8090",
    "admin_token": "test_token",
}


@fixture(scope="module", autouse=True)
def require_pocketbase_test_flag() -> None:
    if not os.environ.get("TEST_POCKETBASE_SERVER"):
        print("could not find `TEST_POCKETBASE_SERVER` in environment, skipping pocketbase tests...")
        pytest.skip("PocketBase tests require TEST_POCKETBASE_SERVER env variable")


def _make_database() -> PocketBaseDocumentDatabase:
    return PocketBaseDocumentDatabase(
        logger=_TestLogger(),
        connection_params=_POCKETBASE_PARAMS,
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


def test_pocketbase_filter_supports_nested_or_and_in() -> None:
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

    filter_str = _build_pocketbase_filter(filters, {"agent_id", "customer_id", "offset"})

    assert "agent_id" in filter_str
    assert 'data."tag_id"' in filter_str
    assert "offset >=" in filter_str
    assert "||" in filter_str or "&&" in filter_str


def test_pocketbase_filter_handles_comparisons() -> None:
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

    filter_str = _build_pocketbase_filter(filters, {"offset"})

    assert "offset !=" in filter_str
    assert "offset <=" in filter_str
    assert "offset >" in filter_str
    assert 'data."creation_utc" <' in filter_str


@pytest.mark.asyncio
async def test_insert_one_serializes_document_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PocketBaseDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())
    collection._collection_ready = True  # type: ignore[attr-defined]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"id": "session-1"}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    db._client = mock_client

    document = _session_document()

    await collection.insert_one(document)

    call_args = mock_client.post.call_args
    assert call_args is not None
    url = call_args[0][0]
    payload = call_args[1]["json"]

    assert "api/collections/parlant_sessions/records" in url
    assert payload["id"] == "session-1"
    assert json.loads(json.dumps(payload["data"])) == document


@pytest.mark.asyncio
async def test_find_uses_pocketbase_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PocketBaseDocumentCollection(db, "events", _SessionDocument, _TestLogger())
    collection._collection_ready = True  # type: ignore[attr-defined]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "items": [{"id": "1", "data": {"id": "1"}}],
        "totalItems": 1,
    }

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    db._client = mock_client

    result = await collection.find({"session_id": {"$eq": "abc"}})

    assert isinstance(result, FindResult)
    assert result.items[0]["id"] == "1"
    call_args = mock_client.get.call_args
    assert call_args is not None
    url = call_args[0][0]
    params = call_args[1]["params"]

    assert "api/collections/parlant_events/records" in url
    assert "filter" in params
    assert "session_id" in params["filter"]


@pytest.mark.asyncio
async def test_find_paginates_and_sets_next_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PocketBaseDocumentCollection(db, "events", _SessionDocument, _TestLogger())
    collection._collection_ready = True  # type: ignore[attr-defined]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "items": [
            {"id": "1", "creation_utc": "2025-01-01", "data": {"id": "1", "creation_utc": "2025-01-01"}},
            {"id": "2", "creation_utc": "2025-01-02", "data": {"id": "2", "creation_utc": "2025-01-02"}},
        ],
        "totalItems": 2,
    }

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    db._client = mock_client

    result = await collection.find({}, limit=1)

    assert len(result.items) == 1
    assert result.has_more is True
    assert result.next_cursor == Cursor(creation_utc="2025-01-01", id=ObjectId("1"))
    assert result.total_count == 2
    call_args = mock_client.get.call_args
    assert call_args is not None
    params = call_args[1]["params"]
    assert params["perPage"] == 2


@pytest.mark.asyncio
async def test_find_adds_cursor_clause(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PocketBaseDocumentCollection(db, "events", _SessionDocument, _TestLogger())
    collection._collection_ready = True  # type: ignore[attr-defined]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"items": [], "totalItems": 0}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    db._client = mock_client

    cursor = Cursor(creation_utc="2025-01-03", id=ObjectId("abc"))
    await collection.find({}, cursor=cursor, sort_direction=SortDirection.DESC)

    call_args = mock_client.get.call_args
    assert call_args is not None
    params = call_args[1]["params"]
    assert params["sort"] == "-creation_utc"
    assert "filter" in params
    assert "creation_utc <" in params["filter"]


@pytest.mark.asyncio
async def test_update_one_upserts_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PocketBaseDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())
    collection._collection_ready = True  # type: ignore[attr-defined]

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
    collection = PocketBaseDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())
    collection._collection_ready = True  # type: ignore[attr-defined]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "items": [{"id": "abc", "version": "0.1", "data": {"id": "abc", "version": "0.1"}}],
        "totalItems": 1,
    }

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    db._client = mock_client

    replace_mock = AsyncMock()
    monkeypatch.setattr(collection, "_replace_document", replace_mock)

    async def loader(doc: Any) -> _SessionDocument:
        return _session_document(doc_id=str(doc["id"]))

    await collection.load_existing_documents(loader)

    replace_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_existing_documents_persists_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PocketBaseDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())
    collection._collection_ready = True  # type: ignore[attr-defined]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "items": [{"id": "bad", "version": "0.7.0", "data": {"id": "bad", "version": "0.7.0"}}],
        "totalItems": 1,
    }

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.delete = AsyncMock()
    db._client = mock_client

    delete_mock = AsyncMock()
    monkeypatch.setattr(collection, "_delete_record", delete_mock)

    async def loader(_: Any) -> _SessionDocument | None:
        return None

    await collection.load_existing_documents(loader)

    delete_mock.assert_awaited_once_with("bad")


@pytest.mark.asyncio
async def test_delete_one_removes_document(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PocketBaseDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    doc = _session_document(doc_id="to-delete")
    monkeypatch.setattr(collection, "find_one", AsyncMock(return_value=doc))
    delete_mock = AsyncMock()
    monkeypatch.setattr(collection, "_delete_record", delete_mock)

    result = await collection.delete_one({"id": {"$eq": "to-delete"}})

    delete_mock.assert_awaited_once_with(ObjectId("to-delete"))
    assert result.deleted_count == 1
    assert result.deleted_document == doc


@pytest.mark.asyncio
async def test_delete_one_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PocketBaseDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    monkeypatch.setattr(collection, "find_one", AsyncMock(return_value=None))
    delete_mock = AsyncMock()
    monkeypatch.setattr(collection, "_delete_record", delete_mock)

    result = await collection.delete_one({"id": {"$eq": "missing"}})

    delete_mock.assert_not_called()
    assert result.deleted_count == 0
    assert result.deleted_document is None


@pytest.mark.asyncio
async def test_ensure_collection_creates_if_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PocketBaseDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    # First call returns 404 (not found), second call succeeds (created)
    not_found_response = MagicMock()
    not_found_response.status_code = 404
    mock_request = MagicMock()
    not_found_error = httpx.HTTPStatusError("404", request=mock_request, response=not_found_response)

    created_response = MagicMock()
    created_response.status_code = 200
    created_response.raise_for_status = MagicMock()
    created_response.json.return_value = {"id": "parlant_sessions"}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=[not_found_error])
    mock_client.post = AsyncMock(return_value=created_response)
    db._client = mock_client

    await collection.ensure_collection()

    assert collection._collection_ready  # type: ignore[attr-defined]
    mock_client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_collection_uses_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = PocketBaseDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    existing_response = MagicMock()
    existing_response.status_code = 200
    existing_response.raise_for_status = MagicMock()
    existing_response.json.return_value = {"id": "parlant_sessions"}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=existing_response)
    db._client = mock_client

    await collection.ensure_collection()

    assert collection._collection_ready  # type: ignore[attr-defined]
    mock_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_delete_collection_removes_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()

    mock_response = MagicMock()
    mock_response.status_code = 204
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.delete = AsyncMock(return_value=mock_response)
    db._client = mock_client

    await db.delete_collection("sessions")

    call_args = mock_client.delete.call_args
    assert call_args is not None
    url = call_args[0][0]
    assert "api/collections/parlant_sessions" in url


@pytest.mark.asyncio
async def test_authenticate_with_email_password(monkeypatch: pytest.MonkeyPatch) -> None:
    db = PocketBaseDocumentDatabase(
        logger=_TestLogger(),
        connection_params={
            "url": "http://localhost:8090",
            "admin_email": "admin@example.com",
            "admin_password": "password123",
        },
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"token": "new_token"}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    db._client = mock_client

    await db._authenticate()

    assert db._admin_token == "new_token"
    call_args = mock_client.post.call_args
    assert call_args is not None
    url = call_args[0][0]
    assert "api/admins/auth-with-password" in url


@pytest.mark.asyncio
async def test_custom_endpoint_connection() -> None:
    """Test that custom PocketBase endpoints work correctly."""
    custom_url = "https://custom-pocketbase.example.com"
    db = PocketBaseDocumentDatabase(
        logger=_TestLogger(),
        connection_params={
            "url": custom_url,
            "admin_token": "custom_token",
        },
    )

    assert db._base_url == custom_url
    assert db._admin_token == "custom_token"

    # Verify headers include custom token
    headers = db._get_headers()
    assert headers["Authorization"] == "Bearer custom_token"

