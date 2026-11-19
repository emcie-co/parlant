"""Cleanup-only handoff through the real dispatcher and runtime store."""

from typing import Any, cast

import pytest

from parlant.core import metadata as metadata_utils
from parlant.core.application import Application
from parlant.core.common import JSONSerializable
from parlant.core.container import Container
from parlant.core.health import HealthReporter
from parlant.core.loggers import Logger
from parlant.core.sessions import EventKind, EventSource, SessionStore
from parlant.core.tunnels import TunnelRequest, TunnelRequestDispatcher
from tests.core.stable.test_session_cleanup import (
    OCCURRED,
    _all_cleanup_records,
    _commit_usage,
    _remove,
    _session,
)


@pytest.fixture
def cleanup_dispatcher(container: Container) -> TunnelRequestDispatcher:
    return TunnelRequestDispatcher(
        application=container[Application],
        health_reporter=container[HealthReporter],
        logger=container[Logger],
    )


async def _call(
    dispatcher: TunnelRequestDispatcher, method: str, params: dict[str, Any]
) -> dict[str, Any]:
    result = await dispatcher.dispatch(TunnelRequest("cleanup-test", method, params))
    assert result.error is None, result.error
    assert isinstance(result.result, dict)
    return cast(dict[str, Any], result.result)


async def test_that_the_tunnel_delivers_cleanup_snapshots_with_existing_session_polls(
    container: Container, cleanup_dispatcher: TunnelRequestDispatcher
) -> None:
    store = container[SessionStore]
    live = await _session(store)
    await _commit_usage(store, live.id)
    expected_ids = set()
    for index in range(2):
        removed = await _session(store)
        await _commit_usage(store, removed.id, trace_id=f"warmup-{index}")
        await _remove(store, removed.id)
        expected_ids.add(removed.id)
    ordinary = await _call(cleanup_dispatcher, "sessions.list", {"limit": 1})
    assert "cleanup" not in ordinary
    result = await _call(
        cleanup_dispatcher,
        "sessions.list",
        {"limit": 1, "include_cleanup": True, "cleanup_limit": 1},
    )
    [listed] = result["sessions"]
    assert listed["session_id"] == live.id
    metadata = cast(dict[str, JSONSerializable], listed["metadata"])
    assert (
        metadata_utils.internal_value(
            metadata, ("usage", "records", "initialization-trace", "timestamp_utc"), None
        )
        == OCCURRED.isoformat()
    ), "Live costs use existing session metadata"
    first = result["cleanup"]
    assert len(first["records"]) == 1 and first["has_more"]
    second = await _call(
        cleanup_dispatcher, "sessions.cleanup.list", {"limit": 1, "cursor": first["next_cursor"]}
    )
    assert len(second["records"]) == 1 and not second["has_more"]
    records = first["records"] + second["records"]
    assert {record["session_id"] for record in records} == expected_ids
    for record in records:
        assert record["session_creation_utc"] == OCCURRED.isoformat()
        assert len(record["usage_records"]) == 1
        [usage] = record["usage_records"].values()
        assert usage["timestamp_utc"] == OCCURRED.isoformat()
        assert "revision" not in record and "fingerprint" not in record
    ack = {"session_ids": list(expected_ids)}
    assert (await _call(cleanup_dispatcher, "sessions.cleanup.acknowledge", ack))[
        "acknowledged"
    ] == 2
    assert (await _call(cleanup_dispatcher, "sessions.cleanup.acknowledge", ack))[
        "acknowledged"
    ] == 0
    assert not await _all_cleanup_records(store)


@pytest.mark.parametrize(
    "method, params, field",
    [
        ("sessions.cleanup.list", {"limit": 0}, "limit"),
        ("sessions.cleanup.list", {"limit": 101}, "limit"),
        ("sessions.cleanup.list", {"limit": True}, "limit"),
        ("sessions.cleanup.list", {"limit": "2"}, "limit"),
        ("sessions.cleanup.list", {"cursor": "not-a-cursor"}, "cursor"),
        ("sessions.cleanup.acknowledge", {"session_ids": "all"}, "session_ids"),
        ("sessions.cleanup.acknowledge", {"session_ids": [""]}, "session_ids"),
        ("sessions.cleanup.acknowledge", {"session_ids": [7]}, "session_ids"),
        ("sessions.cleanup.acknowledge", {"session_ids": [True]}, "session_ids"),
        ("sessions.cleanup.acknowledge", {"session_ids": ["x"] * 101}, "session_ids"),
        ("sessions.list", {"include_cleanup": True, "cleanup_limit": 0}, "limit"),
    ],
)
async def test_that_cleanup_tunnel_paging_and_acknowledgment_are_validated(
    container: Container,
    cleanup_dispatcher: TunnelRequestDispatcher,
    method: str,
    params: dict[str, Any],
    field: str,
) -> None:
    store = container[SessionStore]
    removed = await _session(store)
    await _remove(store, removed.id)
    response = await cleanup_dispatcher.dispatch(TunnelRequest("invalid-cleanup", method, params))
    assert response.error is not None
    assert field in str(response.error).lower(), "Reject the field, not an unknown method"
    assert len(await _all_cleanup_records(store)) == 1


async def test_that_session_lifecycle_facts_cannot_be_forged_by_public_metadata(
    container: Container, cleanup_dispatcher: TunnelRequestDispatcher
) -> None:
    store = container[SessionStore]
    session = await _session(store)
    await _call(
        cleanup_dispatcher,
        "sessions.update",
        {
            "session_id": session.id,
            "metadata": {"lifecycle": {"version": 1, "state": "established"}},
        },
    )
    provisional = await _call(cleanup_dispatcher, "sessions.read", {"session_id": session.id})
    assert provisional["lifecycle"] == {"version": 1, "state": "provisional"}
    await store.create_event(
        session.id,
        EventSource.CUSTOMER,
        EventKind.MESSAGE,
        "customer-trace",
        {"message": "Hello", "participant": {"display_name": "Guest"}},
    )
    await _call(
        cleanup_dispatcher,
        "sessions.update",
        {
            "session_id": session.id,
            "metadata": {"lifecycle": {"version": 1, "state": "provisional"}},
        },
    )
    established = await _call(cleanup_dispatcher, "sessions.read", {"session_id": session.id})
    assert established["lifecycle"] == {"version": 1, "state": "established"}
