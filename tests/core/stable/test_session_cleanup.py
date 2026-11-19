"""Final cleanup handoff through real stores, metadata commits and persistence faults."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence, cast

import pytest
from typing_extensions import override

from parlant.adapters.db.json_file import JSONFileDocumentDatabase
from parlant.adapters.db.transient import TransientDocumentCollection, TransientDocumentDatabase
from parlant.core import metadata as metadata_utils
from parlant.core.agents import AgentId
from parlant.core.common import ItemNotFoundError, JSONSerializable
from parlant.core.customers import CustomerId
from parlant.core.engines.session_metadata_workspace import SessionMetadataWorkspace
from parlant.core.loggers import Logger
from parlant.core.persistence.common import Cursor, Where
from parlant.core.persistence.document_database import (
    BaseDocument,
    DeleteResult,
    InsertResult,
    TDocument,
    UpdateResult,
)
from parlant.core.sessions import (
    CompositeSessionStore,
    EventKind,
    EventSource,
    Session,
    SessionDocumentStore,
    SessionId,
    SessionStore,
)

NOW = datetime(2026, 11, 18, 12, tzinfo=timezone.utc)
OCCURRED = NOW - timedelta(minutes=31)
MODEL = "gemini:gemini-3.1-flash-lite"


class _CleanupRecord(Protocol):
    session_id: SessionId
    agent_id: AgentId
    session_creation_utc: datetime
    deleted_at: datetime
    is_simulation: bool
    customer_represented_by_an_agent: bool
    usage_records: Mapping[str, JSONSerializable]


class _CleanupPage(Protocol):
    records: Sequence[_CleanupRecord]
    has_more: bool
    next_cursor: Cursor | None


class _CleanupFeed(Protocol):
    async def list_cleanup_records(
        self, *, limit: int = 100, cursor: Cursor | None = None
    ) -> _CleanupPage: ...

    async def acknowledge_cleanup_records(self, session_ids: Sequence[SessionId]) -> int: ...


def _feed(store: SessionStore) -> _CleanupFeed:
    # Resolve the optional new capability inside tests, so its current absence
    # produces a useful RED result rather than a collection/import failure.
    return cast(_CleanupFeed, store)


async def _session(store: SessionStore) -> Session:
    return await store.create_session(CustomerId("guest"), AgentId("agent"), creation_utc=OCCURRED)


async def _commit_usage(
    store: SessionStore,
    session_id: SessionId,
    *,
    trace_id: str = "initialization-trace",
    amount: str = "0.000123",
    occurred: datetime = OCCURRED,
) -> None:
    session = await store.read_session(session_id)
    existing = metadata_utils.internal_value(session.metadata, ("usage", "records", trace_id), None)
    if isinstance(existing, Mapping):
        revised = dict(existing)
        revised["costs"] = {"currency": "USD", "total": amount, "per_model": {MODEL: amount}}
        await store.commit_session_metadata(
            session_id,
            session.metadata_revision,
            metadata_utils.set_internal_value(
                session.metadata, ("usage", "records", trace_id), revised
            ),
        )
        return
    workspace = SessionMetadataWorkspace(session, store)
    workspace.stage_usage_record(
        trace_id=trace_id,
        timestamp_utc=occurred.isoformat(),
        models={
            MODEL: {
                "requests": 1,
                "input_tokens": 25,
                "output_tokens": 3,
                "cached_input_tokens": 5,
                "extra": {},
            }
        },
        costs={"currency": "USD", "total": amount, "per_model": {MODEL: amount}},
        pricing_error=None,
    )
    await workspace.commit()


async def _usage(store: SessionStore, session_id: SessionId) -> Mapping[str, JSONSerializable]:
    session = await store.read_session(session_id)
    records = metadata_utils.internal_value(session.metadata, ("usage", "records"), {})
    assert isinstance(records, Mapping)
    return records


def _mapping(value: JSONSerializable) -> Mapping[str, JSONSerializable]:
    assert isinstance(value, Mapping)
    return value


async def _remove(store: SessionStore, session_id: SessionId) -> None:
    assert await store.delete_session_if_empty(
        session_id, created_before=NOW - timedelta(minutes=30), include_greetings=True
    )
    with pytest.raises(ItemNotFoundError):
        await store.read_session(session_id)


async def _all_cleanup_records(store: SessionStore, *, limit: int = 2) -> list[_CleanupRecord]:
    records: list[_CleanupRecord] = []
    cursor: Cursor | None = None
    seen: set[SessionId] = set()
    for _ in range(20):
        page = await _feed(store).list_cleanup_records(limit=limit, cursor=cursor)
        assert len(page.records) <= limit
        for record in page.records:
            assert record.session_id not in seen
            seen.add(record.session_id)
            records.append(record)
        if not page.has_more:
            return records
        assert page.next_cursor is not None and page.next_cursor != cursor
        cursor = page.next_cursor
    pytest.fail("Pending cleanup did not finish a bounded walk")


class _FaultCollection(TransientDocumentCollection[TDocument]):
    """Real in-memory adapter with deterministic persistence failures."""

    def __init__(self, name: str, schema: type[TDocument], owner: _FaultDatabase) -> None:
        super().__init__(name, schema)
        self._owner = owner

    def _before_write(self) -> None:
        if self._owner.fail_handoff and self._name.endswith(
            ("session_observability", "session_cleanup")
        ):
            raise OSError("Cleanup handoff storage is unavailable")

    @override
    async def insert_one(self, document: TDocument) -> InsertResult:
        self._before_write()
        return await super().insert_one(document)

    @override
    async def update_one(
        self, filters: Where, params: TDocument, upsert: bool = False
    ) -> UpdateResult[TDocument]:
        self._before_write()
        return await super().update_one(filters, params, upsert)

    @override
    async def delete_one(self, filters: Where) -> DeleteResult[TDocument]:
        if self._owner.fail_deletion and self._name.endswith("sessions"):
            raise OSError("Session storage is unavailable")
        return await super().delete_one(filters)


class _FaultDatabase(TransientDocumentDatabase):
    fail_handoff = False
    fail_deletion = False

    @override
    async def create_collection(
        self, name: str, schema: type[TDocument]
    ) -> TransientDocumentCollection[TDocument]:
        collection = _FaultCollection(name, schema, self)
        self._collections[name] = cast(TransientDocumentCollection[BaseDocument], collection)
        return collection


async def test_that_live_usage_updates_do_not_create_cleanup_records() -> None:
    database = _FaultDatabase()
    async with SessionDocumentStore(database) as store:
        session = await _session(store)
        # A cleanup-only storage outage must not break ordinary usage commits.
        database.fail_handoff = True
        await _commit_usage(store, session.id)
        await _commit_usage(store, session.id, amount="0.000456")
        usage = await _usage(store, session.id)
        assert _mapping(_mapping(usage["initialization-trace"])["costs"])["total"] == "0.000456"
        assert not await _all_cleanup_records(store)
        for name, collection in database._collections.items():
            if name.endswith(("session_observability", "session_cleanup")):
                assert not await collection.find({})


async def test_that_cleanup_exports_one_final_snapshot_with_original_usage_dates() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await _session(store)
        await _commit_usage(store, session.id, trace_id="warmup")
        second_time = OCCURRED + timedelta(seconds=42)
        await _commit_usage(store, session.id, trace_id="greeting", occurred=second_time)
        await _commit_usage(store, session.id, trace_id="warmup", amount="0.000456")
        expected = await _usage(store, session.id)
        await _remove(store, session.id)
        [record] = await _all_cleanup_records(store)
        assert record.session_id == session.id and record.agent_id == session.agent_id
        assert record.session_creation_utc == session.creation_utc
        assert record.deleted_at.tzinfo is not None
        assert record.usage_records == expected
        assert _mapping(record.usage_records["warmup"])["timestamp_utc"] == OCCURRED.isoformat()
        assert (
            _mapping(record.usage_records["greeting"])["timestamp_utc"] == second_time.isoformat()
        )
        assert _mapping(_mapping(record.usage_records["warmup"])["costs"])["total"] == "0.000456"
        assert await _all_cleanup_records(store) == [record], "Delivery must be immutable"


async def test_that_a_cleanup_snapshot_survives_restart_and_is_removed_after_acknowledgment(
    tmp_path: Path, logger: Logger
) -> None:
    path = tmp_path / "runtime.json"
    async with JSONFileDocumentDatabase(logger, path) as database:
        async with SessionDocumentStore(database) as store:
            session = await _session(store)
            await _commit_usage(store, session.id)
            expected = await _usage(store, session.id)
            await _remove(store, session.id)
    async with JSONFileDocumentDatabase(logger, path) as database:
        async with SessionDocumentStore(database) as store:
            with pytest.raises(ItemNotFoundError):
                await store.read_session(session.id)
            [record] = await _all_cleanup_records(store)
            assert record.usage_records == expected
            assert await _all_cleanup_records(store) == [record]
            assert await _feed(store).acknowledge_cleanup_records([session.id]) == 1
            assert await _feed(store).acknowledge_cleanup_records([session.id]) == 0
    # An ACK must reclaim storage, not merely hide a delivered record.
    raw = json.loads(path.read_text())
    assert raw["session_cleanup"] == []
    async with JSONFileDocumentDatabase(logger, path) as database:
        async with SessionDocumentStore(database) as store:
            assert not await _all_cleanup_records(store)


async def test_that_failed_snapshot_storage_keeps_the_session_and_its_usage() -> None:
    database = _FaultDatabase()
    async with SessionDocumentStore(database) as store:
        session = await _session(store)
        await _commit_usage(store, session.id)
        expected = await _usage(store, session.id)
        database.fail_handoff = True
        assert not await store.delete_session_if_empty(
            session.id, created_before=NOW, include_greetings=True
        )
        assert await _usage(store, session.id) == expected
        database.fail_handoff = False
        await _remove(store, session.id)
        [record] = await _all_cleanup_records(store)
        assert record.usage_records == expected


async def test_that_failed_deletion_does_not_publish_or_acknowledge_a_prepared_snapshot() -> None:
    database = _FaultDatabase()
    async with SessionDocumentStore(database) as store:
        session = await _session(store)
        await store.create_event(
            session.id,
            EventSource.CUSTOMER_UI,
            EventKind.CUSTOM,
            "page-view",
            {"type": "page_view"},
        )
        await _commit_usage(store, session.id)
        expected = await _usage(store, session.id)
        database.fail_deletion = True
        with pytest.raises(OSError, match="Session storage"):
            await _remove(store, session.id)
        assert await _usage(store, session.id) == expected
        assert not await store.list_events(session.id), "Exercise partial event deletion"
        assert not await _all_cleanup_records(store)
        assert await _feed(store).acknowledge_cleanup_records([session.id]) == 0
        database.fail_deletion = False
        await _remove(store, session.id)
        [record] = await _all_cleanup_records(store)
        assert record.usage_records == expected


async def test_that_retrying_a_prepared_cleanup_captures_the_latest_usage_before_removal() -> None:
    database = _FaultDatabase()
    async with SessionDocumentStore(database) as store:
        session = await _session(store)
        await _commit_usage(store, session.id)
        database.fail_deletion = True
        with pytest.raises(OSError, match="Session storage"):
            await _remove(store, session.id)
        await _commit_usage(store, session.id, amount="0.000789")
        await _commit_usage(store, session.id, trace_id="late-finalization")
        expected = await _usage(store, session.id)
        database.fail_deletion = False
        await _remove(store, session.id)
        [record] = await _all_cleanup_records(store)
        assert record.usage_records == expected
        assert await _all_cleanup_records(store) == [record]


async def test_that_cleanup_pages_remain_bounded_and_skip_prepared_deletions() -> None:
    database = _FaultDatabase()
    async with SessionDocumentStore(database) as store:
        blocked = await _session(store)
        database.fail_deletion = True
        with pytest.raises(OSError, match="Session storage"):
            await _remove(store, blocked.id)
        database.fail_deletion = False
        expected: set[SessionId] = set()
        for index in range(5):
            session = await _session(store)
            await _commit_usage(store, session.id, trace_id=f"warmup-{index}")
            await _remove(store, session.id)
            expected.add(session.id)
        first = await _feed(store).list_cleanup_records(limit=1)
        assert not first.records and first.has_more and first.next_cursor is not None
        actual = await _all_cleanup_records(store, limit=1)
        assert {record.session_id for record in actual} == expected
        assert (await store.read_session(blocked.id)).id == blocked.id


async def test_that_cleanup_paging_continues_after_the_previous_page_is_acknowledged() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        expected: set[SessionId] = set()
        for _ in range(5):
            session = await _session(store)
            await _remove(store, session.id)
            expected.add(session.id)
        seen: set[SessionId] = set()
        cursor: Cursor | None = None
        for _ in range(5):
            page = await _feed(store).list_cleanup_records(limit=2, cursor=cursor)
            ids = [record.session_id for record in page.records]
            assert ids and not (seen & set(ids))
            seen.update(ids)
            assert await _feed(store).acknowledge_cleanup_records(ids) == len(ids)
            if not page.has_more:
                break
            assert page.next_cursor is not None and page.next_cursor != cursor
            cursor = page.next_cursor
        assert seen == expected
        assert not await _all_cleanup_records(store)


@pytest.mark.parametrize("simulation_marker", [None, False, True, "true"])
async def test_that_cleanup_snapshots_preserve_filter_dimensions_without_customer_data(
    simulation_marker: bool | str | None,
) -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        metadata: dict[str, JSONSerializable] = {
            "private_email": "sensitive@example.invalid",
            "kind": "persona",
            "__parlant_cloud__": {"createdById": "legacy-simulation-marker"},
        }
        if simulation_marker is not None:
            metadata["__dashboard__"] = {"isSimulation": simulation_marker}
        represented = simulation_marker is not False
        session = await store.create_session(
            CustomerId("customer-profile-do-not-export"),
            AgentId("agent"),
            creation_utc=OCCURRED,
            metadata=metadata,
            customer_represented_by_an_agent=represented,
        )
        await _commit_usage(store, session.id)
        await _remove(store, session.id)
        [record] = await _all_cleanup_records(store)
        assert record.session_creation_utc == OCCURRED
        assert record.is_simulation is (simulation_marker is True)
        assert record.customer_represented_by_an_agent is represented
        assert is_dataclass(record) and not isinstance(record, type)
        exported = json.dumps(asdict(record), default=str)
        assert "sensitive@example.invalid" not in exported
        assert "customer-profile-do-not-export" not in exported
        assert "legacy-simulation-marker" not in exported


async def test_that_cleanup_handoffs_remain_scoped_to_the_writable_store() -> None:
    database = TransientDocumentDatabase()
    async with SessionDocumentStore(database, collections_prefix="runtime") as runtime:
        async with SessionDocumentStore(database, collections_prefix="declarations") as readonly:
            actual_session, other_session = await _session(runtime), await _session(readonly)
            await _remove(runtime, actual_session.id)
            await _remove(readonly, other_session.id)
            composite = CompositeSessionStore(runtime, [readonly])
            [actual] = await _all_cleanup_records(composite)
            assert actual.session_id == actual_session.id
            assert await _feed(composite).acknowledge_cleanup_records([other_session.id]) == 0
            [other] = await _all_cleanup_records(readonly)
            assert other.session_id == other_session.id
            assert await _feed(composite).acknowledge_cleanup_records([actual_session.id]) == 1
            assert len(await _all_cleanup_records(readonly)) == 1


async def test_that_manual_deletion_discards_pending_cleanup_accounting() -> None:
    database = _FaultDatabase()
    async with SessionDocumentStore(database) as store:
        established = await _session(store)
        await store.create_event(
            established.id,
            EventSource.CUSTOMER,
            EventKind.MESSAGE,
            "customer-trace",
            {"message": "Private question", "participant": {"display_name": "Guest"}},
        )
        await _commit_usage(store, established.id)
        await store.delete_session(established.id)
        assert not await _all_cleanup_records(store)
        prepared = await _session(store)
        await _commit_usage(store, prepared.id)
        database.fail_deletion = True
        with pytest.raises(OSError, match="Session storage"):
            await _remove(store, prepared.id)
        database.fail_deletion = False
        await store.delete_session(prepared.id)
        assert not await _all_cleanup_records(store), "Privacy deletion discards preparation"
        removed = await _session(store)
        await _remove(store, removed.id)
        assert len(await _all_cleanup_records(store)) == 1
        await store.delete_session(removed.id)
        assert not await _all_cleanup_records(store), "Explicit removal also purges pending data"


async def test_that_legacy_cleanup_is_discovered_in_bounded_batches_without_startup_rewrites(
    tmp_path: Path, logger: Logger
) -> None:
    path = tmp_path / "legacy.json"
    async with JSONFileDocumentDatabase(logger, path) as database:
        async with SessionDocumentStore(database) as store:
            sessions = [await _session(store) for _ in range(4)]
    raw = json.loads(path.read_text())
    for document in raw["sessions"]:
        document.pop("cleanup_established", None)
    path.write_text(json.dumps(raw))
    async with JSONFileDocumentDatabase(logger, path) as database:
        async with SessionDocumentStore(database) as reopened:
            # A loader default would rewrite all old rows during Mongo startup.
            loaded = await reopened._session_collection.find({})
            assert all("cleanup_established" not in document for document in loaded)
            page = await reopened.list_empty_session_candidates(created_before=NOW, limit=2)
            assert len(page.items) == 2 and page.has_more
            for session in page.items:
                assert await reopened.delete_session_if_empty(
                    session.id, created_before=NOW, include_greetings=False
                )
            assert (await reopened.list_sessions()).total_count == len(sessions) - 2
