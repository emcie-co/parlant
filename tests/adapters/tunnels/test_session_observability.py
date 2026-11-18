"""Accounting handoff through the real dispatcher and runtime session store."""

from typing import Any, cast

import pytest

from parlant.core.application import Application
from parlant.core.container import Container
from parlant.core.health import HealthReporter
from parlant.core.loggers import Logger
from parlant.core.sessions import EventKind, EventSource, SessionStore
from parlant.core.tunnels import TunnelRequest, TunnelRequestDispatcher
from tests.core.stable.test_session_observability import (
    OCCURRED,
    _all_records,
    _commit_usage,
    _session,
)


@pytest.fixture
def observability_dispatcher(container: Container) -> TunnelRequestDispatcher:
    return TunnelRequestDispatcher(
        application=container[Application],
        health_reporter=container[HealthReporter],
        logger=container[Logger],
    )


async def _call(
    dispatcher: TunnelRequestDispatcher, method: str, params: dict[str, Any]
) -> dict[str, Any]:
    result = await dispatcher.dispatch(TunnelRequest("observability-test", method, params))
    assert result.error is None, result.error
    assert isinstance(result.result, dict)
    return cast(dict[str, Any], result.result)


async def test_that_the_observability_tunnel_round_trips_bounded_records_and_exact_acks(
    container: Container, observability_dispatcher: TunnelRequestDispatcher
) -> None:
    store = container[SessionStore]
    sessions = [await _session(store) for _ in range(3)]
    for index, session in enumerate(sessions):
        await _commit_usage(store, session.id, trace_id=f"trace-{index}")
    first = await _call(observability_dispatcher, "observability.list", {"limit": 2})
    assert len(first["records"]) == 2 and first["has_more"]
    assert isinstance(first["next_cursor"], str)
    second = await _call(
        observability_dispatcher,
        "observability.list",
        {"limit": 2, "cursor": first["next_cursor"]},
    )
    assert len(second["records"]) == 1 and not second["has_more"]
    records = first["records"] + second["records"]
    assert len({record["id"] for record in records}) == 3
    assert {record["session_id"] for record in records} == {session.id for session in sessions}
    assert all(record["occurred_at"] == OCCURRED.isoformat() for record in records)
    assert all(record["payload"]["timestamp_utc"] == OCCURRED.isoformat() for record in records)
    ack = [{"id": record["id"], "revision": record["revision"]} for record in records]
    assert (await _call(observability_dispatcher, "observability.acknowledge", {"records": ack}))[
        "acknowledged"
    ] == 3
    assert (await _call(observability_dispatcher, "observability.acknowledge", {"records": ack}))[
        "acknowledged"
    ] == 0
    assert (await _call(observability_dispatcher, "observability.list", {}))["records"] == []


@pytest.mark.parametrize(
    "method, params, field",
    [
        ("observability.list", {"limit": 0}, "limit"),
        ("observability.list", {"limit": 101}, "limit"),
        ("observability.list", {"limit": True}, "limit"),
        ("observability.list", {"limit": "2"}, "limit"),
        ("observability.list", {"cursor": "not-a-cursor"}, "cursor"),
        ("observability.acknowledge", {"records": "all"}, "records"),
        ("observability.acknowledge", {"records": [{"id": "x", "revision": -1}]}, "revision"),
        ("observability.acknowledge", {"records": [{"id": "x", "revision": True}]}, "revision"),
        ("observability.acknowledge", {"records": [{"id": "", "revision": 1}]}, "id"),
    ],
)
async def test_that_invalid_observability_limits_cursors_and_acknowledgements_are_rejected(
    container: Container,
    observability_dispatcher: TunnelRequestDispatcher,
    method: str,
    params: dict[str, Any],
    field: str,
) -> None:
    store = container[SessionStore]
    session = await _session(store)
    await _commit_usage(store, session.id)
    result = await observability_dispatcher.dispatch(TunnelRequest("invalid", method, params))
    assert result.error is not None
    assert field in str(result.error).lower(), (
        "Report the invalid field, not an unknown-method error"
    )
    assert len(await _all_records(store)) == 1


async def test_that_session_lifecycle_facts_cannot_be_forged_by_public_metadata(
    container: Container, observability_dispatcher: TunnelRequestDispatcher
) -> None:
    store = container[SessionStore]
    session = await _session(store)
    await _call(
        observability_dispatcher,
        "sessions.update",
        {
            "session_id": session.id,
            "metadata": {"lifecycle": {"version": 1, "state": "established"}},
        },
    )
    provisional = await _call(observability_dispatcher, "sessions.read", {"session_id": session.id})
    assert provisional["lifecycle"] == {"version": 1, "state": "provisional"}
    await store.create_event(
        session.id,
        EventSource.CUSTOMER,
        EventKind.MESSAGE,
        "customer-trace",
        {"message": "Hello", "participant": {"display_name": "Guest"}},
    )
    await _call(
        observability_dispatcher,
        "sessions.update",
        {
            "session_id": session.id,
            "metadata": {"lifecycle": {"version": 1, "state": "provisional"}},
        },
    )
    established = await _call(observability_dispatcher, "sessions.read", {"session_id": session.id})
    assert established["lifecycle"] == {"version": 1, "state": "established"}


async def test_that_session_listing_can_include_a_bounded_pending_page_without_an_extra_rpc(
    container: Container, observability_dispatcher: TunnelRequestDispatcher
) -> None:
    store = container[SessionStore]
    for index in range(2):
        session = await _session(store)
        await _commit_usage(store, session.id, trace_id=f"trace-{index}")
    result = await _call(
        observability_dispatcher,
        "sessions.list",
        {
            "limit": 1,
            "include_observability": True,
            "observability_limit": 1,
        },
    )
    assert len(result["sessions"]) == 1
    attached = result["observability"]
    assert len(attached["records"]) == 1 and attached["has_more"]
    assert attached["next_cursor"]
