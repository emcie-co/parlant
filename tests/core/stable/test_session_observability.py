"""Durable empty-session accounting through real stores and metadata commits."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence, cast

import pytest
from typing_extensions import override

from parlant.adapters.db.json_file import JSONFileDocumentDatabase
from parlant.adapters.db.transient import TransientDocumentCollection, TransientDocumentDatabase
from parlant.core.agents import AgentId
from parlant.core.common import ItemNotFoundError, JSONSerializable
from parlant.core import metadata as metadata_utils
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


class _Record(Protocol):
    id: str
    revision: int
    kind: str
    session_id: SessionId
    trace_id: str | None
    occurred_at: datetime
    payload: Mapping[str, JSONSerializable]
    session_creation_utc: datetime
    is_simulation: bool
    customer_represented_by_an_agent: bool


class _Page(Protocol):
    records: Sequence[_Record]
    has_more: bool
    next_cursor: Cursor | None


class _Feed(Protocol):
    async def list_observability_records(
        self, *, limit: int = 100, cursor: Cursor | None = None
    ) -> _Page: ...

    async def acknowledge_observability_records(
        self, acknowledgements: Sequence[tuple[str, int]]
    ) -> int: ...


def _feed(store: SessionStore) -> _Feed:
    return cast(_Feed, store)


async def _session(store: SessionStore) -> Session:
    return await store.create_session(CustomerId("guest"), AgentId("agent"), creation_utc=OCCURRED)


async def _commit_usage(
    store: SessionStore,
    session_id: SessionId,
    *,
    trace_id: str = "initialization-trace",
    amount: str = "0.000123",
) -> None:
    session = await store.read_session(session_id)
    existing = metadata_utils.internal_value(session.metadata, ("usage", "records", trace_id), None)
    if isinstance(existing, Mapping):
        # Workspace usage operations deliberately ignore a duplicate trace.
        # A revised authoritative snapshot therefore uses the real CAS store
        # interface, as a late accounting producer would, rather than expecting
        # an idempotent workspace replay to change its already committed input.
        revised = dict(existing)
        revised["costs"] = {"currency": "USD", "total": amount, "per_model": {MODEL: amount}}
        updated = await store.commit_session_metadata(
            session_id,
            session.metadata_revision,
            metadata_utils.set_internal_value(
                session.metadata, ("usage", "records", trace_id), revised
            ),
        )
        assert (
            metadata_utils.internal_value(
                updated.metadata, ("usage", "records", trace_id, "costs", "total"), None
            )
            == amount
        )
        return
    workspace = SessionMetadataWorkspace(session, store)
    workspace.stage_usage_record(
        trace_id=trace_id,
        timestamp_utc=OCCURRED.isoformat(),
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


async def _all_records(store: SessionStore, *, limit: int = 2) -> list[_Record]:
    result: list[_Record] = []
    cursor: Cursor | None = None
    seen: set[str] = set()
    for _ in range(20):
        page = await _feed(store).list_observability_records(limit=limit, cursor=cursor)
        assert len(page.records) <= limit
        for record in page.records:
            assert record.id not in seen, "One bounded walk must not repeat a record"
            seen.add(record.id)
            result.append(record)
        if not page.has_more:
            return result
        assert page.next_cursor is not None and page.next_cursor != cursor
        cursor = page.next_cursor
    pytest.fail("Pending feed did not finish a bounded walk")


async def test_that_usage_survives_cleanup_and_a_store_restart_with_original_timestamps(
    tmp_path: Path, logger: Logger
) -> None:
    path = tmp_path / "runtime.json"
    async with JSONFileDocumentDatabase(logger, path) as database:
        async with SessionDocumentStore(database) as store:
            session = await _session(store)
            await _commit_usage(store, session.id)
            [original] = await _all_records(store)
            assert original.kind == "usage"
            assert original.occurred_at == OCCURRED
            assert original.trace_id == "initialization-trace"
            assert await store.delete_session_if_empty(
                session.id, created_before=NOW - timedelta(minutes=30), include_greetings=False
            )

    async with JSONFileDocumentDatabase(logger, path) as database:
        async with SessionDocumentStore(database) as reopened:
            with pytest.raises(ItemNotFoundError):
                await reopened.read_session(session.id)
            records = await _all_records(reopened, limit=1)
            [usage] = [record for record in records if record.kind == "usage"]
            assert (usage.id, usage.revision) == (original.id, original.revision)
            assert usage.payload == original.payload
            assert usage.payload["timestamp_utc"] == OCCURRED.isoformat()
            assert usage.payload["costs"] == {
                "currency": "USD",
                "total": "0.000123",
                "per_model": {MODEL: "0.000123"},
            }
            [deletion] = [record for record in records if record.kind == "session_deleted"]
            assert deletion.session_id == session.id
            assert (
                await _feed(reopened).acknowledge_observability_records(
                    [(record.id, record.revision) for record in records]
                )
                == 2
            )
            assert not (await _feed(reopened).list_observability_records()).records


async def test_that_repeated_delivery_and_promotion_keep_the_same_usage_identity() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await _session(store)
        await _commit_usage(store, session.id)
        [before] = await _all_records(store)
        [redelivered] = await _all_records(store)
        assert (redelivered.id, redelivered.revision) == (before.id, before.revision)
        await store.create_event(
            session.id,
            EventSource.CUSTOMER,
            EventKind.MESSAGE,
            "customer-trace",
            {"message": "Can you help?", "participant": {"display_name": "Guest"}},
        )
        # A later metadata commit still carries the earlier warm-up ledger.
        await store.set_metadata(session.id, "customer_preference", "natural colours")
        [after] = [record for record in await _all_records(store) if record.kind == "usage"]
        assert after.id == before.id
        assert after.payload == before.payload
        assert after.occurred_at == OCCURRED
        assert not await store.delete_session_if_empty(
            session.id, created_before=NOW, include_greetings=True
        )


async def test_that_an_old_acknowledgement_cannot_clear_a_new_usage_revision() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await _session(store)
        await _commit_usage(store, session.id)
        [old] = await _all_records(store)
        await _commit_usage(store, session.id, amount="0.000456")
        [new] = await _all_records(store)
        assert new.id == old.id and new.revision > old.revision
        assert await _feed(store).acknowledge_observability_records([(old.id, old.revision)]) == 0
        [pending] = await _all_records(store)
        assert pending.revision == new.revision
        assert pending.payload["costs"] == {
            "currency": "USD",
            "total": "0.000456",
            "per_model": {MODEL: "0.000456"},
        }
        assert await _feed(store).acknowledge_observability_records([(new.id, new.revision)]) == 1
        assert await _feed(store).acknowledge_observability_records([(new.id, new.revision)]) == 0
        assert not await _all_records(store)
        # Deferred usage after acknowledgement makes the same identity pending again.
        await _commit_usage(store, session.id, amount="0.000789")
        [later] = await _all_records(store)
        assert later.id == old.id and later.revision > new.revision


class _FaultCollection(TransientDocumentCollection[TDocument]):
    """Real in-memory persistence with deterministic transport/storage faults."""

    def __init__(self, name: str, schema: type[TDocument], owner: _FaultDatabase) -> None:
        super().__init__(name, schema)
        self._owner = owner

    def _before_write(self) -> None:
        if self._owner.fail_journal and self._name.endswith("session_observability"):
            raise OSError("Observability storage is unavailable")

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
    fail_journal = False
    fail_deletion = False

    @override
    async def create_collection(
        self, name: str, schema: type[TDocument]
    ) -> TransientDocumentCollection[TDocument]:
        collection = _FaultCollection(name, schema, self)
        self._collections[name] = cast(TransientDocumentCollection[BaseDocument], collection)
        return collection


async def test_that_failed_accounting_capture_preserves_the_session_for_retry() -> None:
    database = _FaultDatabase()
    async with SessionDocumentStore(database) as store:
        session = await _session(store)
        database.fail_journal = True
        with pytest.raises(OSError, match="Observability storage"):
            await _commit_usage(store, session.id)
        assert not await store.delete_session_if_empty(
            session.id, created_before=NOW, include_greetings=True
        )
        assert (await store.read_session(session.id)).id == session.id
        database.fail_journal = False
        await _commit_usage(store, session.id)
        assert await store.delete_session_if_empty(
            session.id, created_before=NOW, include_greetings=True
        )
        assert any(record.kind == "usage" for record in await _all_records(store))


async def test_that_a_deletion_is_delivered_only_after_the_session_was_removed() -> None:
    database = _FaultDatabase()
    async with SessionDocumentStore(database) as store:
        session = await _session(store)
        database.fail_deletion = True
        with pytest.raises(OSError, match="Session storage"):
            await store.delete_session_if_empty(
                session.id, created_before=NOW, include_greetings=False
            )
        assert (await store.read_session(session.id)).id == session.id
        assert not any(record.kind == "session_deleted" for record in await _all_records(store))
        database.fail_deletion = False
        assert await store.delete_session_if_empty(
            session.id, created_before=NOW, include_greetings=False
        )
        [notification] = await _all_records(store)
        assert notification.kind == "session_deleted" and notification.session_id == session.id


async def test_that_pending_records_page_without_starving_after_an_unconfirmed_deletion() -> None:
    database = _FaultDatabase()
    async with SessionDocumentStore(database) as store:
        blocked = await _session(store)
        database.fail_deletion = True
        with pytest.raises(OSError):
            await store.delete_session_if_empty(
                blocked.id, created_before=NOW, include_greetings=False
            )
        database.fail_deletion = False
        ids = set()
        for index in range(5):
            session = await _session(store)
            ids.add(session.id)
            await _commit_usage(store, session.id, trace_id=f"trace-{index}")
        records = await _all_records(store, limit=1)
        assert {record.session_id for record in records} == ids
        assert all(record.kind == "usage" for record in records)


async def test_that_runtime_feeds_are_isolated_from_other_prefixes_and_read_only_stores() -> None:
    database = TransientDocumentDatabase()
    async with SessionDocumentStore(database, collections_prefix="runtime") as runtime:
        async with SessionDocumentStore(database, collections_prefix="declarations") as readonly:
            writable_session, declared_session = await _session(runtime), await _session(readonly)
            await _commit_usage(runtime, writable_session.id)
            await _commit_usage(readonly, declared_session.id)
            composite = CompositeSessionStore(runtime, [readonly])
            [actual] = await _all_records(composite)
            [other] = await _all_records(readonly)
            assert actual.session_id == writable_session.id
            assert other.session_id == declared_session.id and actual.id != other.id
            assert (
                await _feed(composite).acknowledge_observability_records(
                    [(other.id, other.revision)]
                )
                == 0
            )
            assert len(await _all_records(readonly)) == 1


async def test_that_manual_deletion_of_an_established_session_keeps_existing_privacy_semantics() -> (
    None
):
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await _session(store)
        await store.create_event(
            session.id,
            EventSource.CUSTOMER,
            EventKind.MESSAGE,
            "customer-trace",
            {"message": "Private question", "participant": {"display_name": "Guest"}},
        )
        await _commit_usage(store, session.id)
        assert await _all_records(store)
        await store.delete_session(session.id)
        assert not await _all_records(store), (
            "A manual deletion must not revive accounting delivery"
        )


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


@pytest.mark.parametrize("simulation_marker", [None, False, True, "true"])
async def test_that_cleanup_records_preserve_session_dimensions_without_customer_data(
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
        assert await store.delete_session_if_empty(
            session.id, created_before=NOW, include_greetings=True
        )
        records = await _all_records(store)
        assert {record.kind for record in records} == {"usage", "session_deleted"}
        for record in records:
            assert record.session_creation_utc == OCCURRED
            assert record.is_simulation is (simulation_marker is True)
            assert record.customer_represented_by_an_agent is represented
            assert is_dataclass(record) and not isinstance(record, type)
            serialized = json.dumps(asdict(record), default=str)
            assert "sensitive@example.invalid" not in serialized
            assert "customer-profile-do-not-export" not in serialized
