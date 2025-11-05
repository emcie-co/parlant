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
from typing import Any, Mapping
from unittest.mock import AsyncMock

import pytest

from parlant.adapters.db.snowflake_db import (
    SnowflakeDocumentCollection,
    SnowflakeDocumentDatabase,
    _build_where_clause,
)
from parlant.core.persistence.document_database import InsertResult
from parlant.core.sessions import _SessionDocument
from tests.test_utilities import _TestLogger


_SNOWFLAKE_PARAMS: Mapping[str, Any] = {
    "account": "acct",
    "user": "user",
    "password": "pwd",
    "warehouse": "warehouse",
    "database": "PARLANT",
    "schema": "PUBLIC",
}


def _make_database() -> SnowflakeDocumentDatabase:
    return SnowflakeDocumentDatabase(
        logger=_TestLogger(),
        connection_params=_SNOWFLAKE_PARAMS,
        connection_factory=lambda *_: object(),
    )


def test_where_clause_supports_nested_or_and_in() -> None:
    filters = {
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
    }

    clause, params = _build_where_clause(filters, {"agent_id", "customer_id", "offset"})

    assert '"AGENT_ID"' in clause
    assert 'DATA:"tag_id"' in clause
    assert "TO_VARIANT" in clause
    assert '"OFFSET" >=' in clause
    assert params["param_0"] == "agent-1"
    assert params["param_1"] == "cust-9"
    assert params["param_2"] == "alpha"
    assert params["param_3"] == "beta"
    assert params["param_4"] == 3


def test_where_clause_handles_comparisons() -> None:
    filters = {
        "creation_utc": {"$lt": "2025-01-01"},
        "offset": {"$ne": 4},
        "$and": [
            {"offset": {"$lte": 10}},
            {"offset": {"$gt": 2}},
        ],
    }

    clause, params = _build_where_clause(filters, {"offset"})

    assert '"OFFSET" !=' in clause
    assert '"OFFSET" <=' in clause
    assert '"OFFSET" >' in clause
    assert 'DATA:"creation_utc" <' in clause
    assert params["param_0"] == "2025-01-01"
    assert params["param_1"] == 4
    assert params["param_2"] == 10
    assert params["param_3"] == 2


@pytest.mark.asyncio
async def test_insert_one_serializes_document_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SnowflakeDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())
    collection._table_ready = True  # type: ignore[attr-defined]

    execute_mock = AsyncMock()
    monkeypatch.setattr(db, "_execute", execute_mock)

    document: _SessionDocument = {
        "id": "session-1",
        "version": "0.7.0",
        "creation_utc": "2025-01-01T00:00:00Z",
        "customer_id": "customer-1",
        "agent_id": "agent-1",
        "title": None,
        "mode": "auto",
        "consumption_offsets": {"client": 0},
        "agent_states": [],
        "metadata": {},
    }

    await collection.insert_one(document)

    sql, params = execute_mock.call_args[0][0], execute_mock.call_args[0][1]
    assert "INSERT INTO" in sql
    assert json.loads(params["data"]) == document
    assert params["id"] == "session-1"


@pytest.mark.asyncio
async def test_find_uses_sql_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SnowflakeDocumentCollection(db, "events", _SessionDocument, _TestLogger())
    collection._table_ready = True  # type: ignore[attr-defined]

    monkeypatch.setattr(db, "_execute", AsyncMock(return_value=[{"DATA": {"id": "1"}}]))

    documents = await collection.find({"session_id": {"$eq": "abc"}})

    assert documents[0]["id"] == "1"
    sql, params = db._execute.call_args[0][0], db._execute.call_args[0][1]  # type: ignore[attr-defined]
    assert 'WHERE "SESSION_ID" =' in sql
    assert params["param_0"] == "abc"


@pytest.mark.asyncio
async def test_update_one_upserts_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SnowflakeDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())
    collection._table_ready = True  # type: ignore[attr-defined]

    monkeypatch.setattr(collection, "find_one", AsyncMock(return_value=None))
    insert_mock = AsyncMock(return_value=InsertResult(True))
    monkeypatch.setattr(collection, "insert_one", insert_mock)

    payload: _SessionDocument = {
        "id": "session-9",
        "version": "0.7.0",
        "creation_utc": "2025-01-01T00:00:00Z",
        "customer_id": "customer-9",
        "agent_id": "agent-9",
        "title": None,
        "mode": "auto",
        "consumption_offsets": {"client": 0},
        "agent_states": [],
        "metadata": {},
    }

    result = await collection.update_one({"id": {"$eq": "session-9"}}, payload, upsert=True)

    insert_mock.assert_awaited_once()
    assert result.updated_document == payload


@pytest.mark.asyncio
async def test_load_existing_documents_migrates(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SnowflakeDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())
    collection._table_ready = True  # type: ignore[attr-defined]

    monkeypatch.setattr(
        db, "_execute", AsyncMock(return_value=[{"DATA": {"id": "abc", "version": "0.1"}}])
    )
    replace_mock = AsyncMock()
    monkeypatch.setattr(collection, "_replace_document", replace_mock)
    monkeypatch.setattr(collection, "_persist_failed_documents", AsyncMock())
    monkeypatch.setattr(collection, "_delete_documents", AsyncMock())

    async def loader(doc: Any) -> _SessionDocument:
        migrated = dict(doc)
        migrated["version"] = "0.7.0"
        return migrated  # return a new object so the adapter writes it back

    await collection.load_existing_documents(loader)

    replace_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_existing_documents_persists_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SnowflakeDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())
    collection._table_ready = True  # type: ignore[attr-defined]

    calls: list[tuple[str, Any, str]] = []

    async def fake_execute(sql: str, params: Any = None, fetch: str = "none") -> Any:
        calls.append((sql, params, fetch))
        if sql.startswith("SELECT DATA"):
            return [{"DATA": {"id": "bad", "version": "0.7.0"}}]
        return None

    monkeypatch.setattr(db, "_execute", fake_execute)
    delete_mock = AsyncMock()
    monkeypatch.setattr(collection, "_delete_documents", delete_mock)

    async def loader(_: Any) -> _SessionDocument | None:
        return None

    await collection.load_existing_documents(loader)

    assert any("INSERT INTO" in sql and "FAILED_MIGRATIONS" in sql for sql, _, _ in calls)
    delete_mock.assert_awaited_once_with(["bad"])


@pytest.mark.asyncio
async def test_delete_one_removes_document(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SnowflakeDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    doc = {"id": "to-delete"}
    monkeypatch.setattr(collection, "find_one", AsyncMock(return_value=doc))
    delete_mock = AsyncMock()
    monkeypatch.setattr(collection, "_delete_documents", delete_mock)

    result = await collection.delete_one({"id": {"$eq": "to-delete"}})

    delete_mock.assert_awaited_once_with(["to-delete"])
    assert result.deleted_count == 1
    assert result.deleted_document == doc


@pytest.mark.asyncio
async def test_delete_one_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SnowflakeDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    monkeypatch.setattr(collection, "find_one", AsyncMock(return_value=None))
    delete_mock = AsyncMock()
    monkeypatch.setattr(collection, "_delete_documents", delete_mock)

    result = await collection.delete_one({"id": {"$eq": "missing"}})

    delete_mock.assert_not_called()
    assert result.deleted_count == 0
    assert result.deleted_document is None


@pytest.mark.asyncio
async def test_ensure_table_runs_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()
    collection = SnowflakeDocumentCollection(db, "sessions", _SessionDocument, _TestLogger())

    execute_mock = AsyncMock()
    monkeypatch.setattr(db, "_execute", execute_mock)

    await collection.ensure_table()
    await collection.ensure_table()

    assert execute_mock.await_count == 2  # main + failed table


@pytest.mark.asyncio
async def test_delete_collection_drops_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_database()

    execute_mock = AsyncMock()
    monkeypatch.setattr(db, "_execute", execute_mock)

    await db.delete_collection("sessions")

    drop_statements = [call.args[0] for call in execute_mock.await_args_list]
    assert 'DROP TABLE IF EXISTS "PARLANT_SESSIONS"' in drop_statements[0]
    assert 'DROP TABLE IF EXISTS "PARLANT_SESSIONS_FAILED_MIGRATIONS"' in drop_statements[1]
