from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from parlant.core.tunnels import TunnelRequestDispatcher
from parlant.core.tunnels import TunnelRequest


async def test_that_dispatcher_routes_sessions_create_event_to_session_module() -> None:
    session_module = AsyncMock()
    session_module.create_customer_message = AsyncMock(
        return_value=MagicMock(
            id="evt-1",
            offset=0,
            source="customer",
            kind="message",
            creation_utc="2026-01-01T00:00:00Z",
        )
    )

    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    request = TunnelRequest(
        request_id="req-1",
        method="sessions.create_event",
        params={
            "session_id": "sess-1",
            "kind": "message",
            "source": "customer",
            "message": "Hello",
        },
    )

    response = await dispatcher.dispatch(request)

    assert response.request_id == "req-1"
    assert response.error is None
    session_module.create_customer_message.assert_awaited_once()


async def test_that_dispatcher_returns_error_for_unknown_method() -> None:
    session_module = AsyncMock()
    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    request = TunnelRequest(
        request_id="req-2",
        method="unknown.method",
        params={},
    )

    response = await dispatcher.dispatch(request)

    assert response.request_id == "req-2"
    assert response.error is not None
    assert "unknown" in response.error.lower()


async def test_that_dispatcher_routes_sessions_create() -> None:
    session_module = AsyncMock()
    session_module.create = AsyncMock(
        return_value=MagicMock(
            id="sess-1",
            creation_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            customer_id="cust-1",
            agent_id="agent-1",
            mode="auto",
            title=None,
            consumption_offsets={},
            agent_states=[],
            metadata={},
            labels=set(),
        )
    )

    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    request = TunnelRequest(
        request_id="req-create",
        method="sessions.create",
        params={
            "customer_id": "cust-1",
            "agent_id": "agent-1",
        },
    )

    response = await dispatcher.dispatch(request)

    assert response.request_id == "req-create"
    assert response.error is None
    assert isinstance(response.result, dict)
    assert response.result["session_id"] == "sess-1"
    session_module.create.assert_awaited_once()


async def test_that_dispatcher_routes_sessions_read() -> None:
    session_module = AsyncMock()
    session_module.read = AsyncMock(
        return_value=MagicMock(
            id="sess-1",
            creation_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            customer_id="cust-1",
            agent_id="agent-1",
            mode="auto",
            title="Test Session",
            consumption_offsets={},
            agent_states=[],
            metadata={},
            labels=set(),
        )
    )

    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    request = TunnelRequest(
        request_id="req-read",
        method="sessions.read",
        params={"session_id": "sess-1"},
    )

    response = await dispatcher.dispatch(request)

    assert response.error is None
    assert isinstance(response.result, dict)
    assert response.result["session_id"] == "sess-1"
    assert response.result["agent_id"] == "agent-1"
    session_module.read.assert_awaited_once()


async def test_that_dispatcher_routes_sessions_list() -> None:
    session_module = AsyncMock()
    session_module.find = AsyncMock(
        return_value=MagicMock(
            items=[
                MagicMock(
                    id="sess-1",
                    creation_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    customer_id="cust-1",
                    agent_id="agent-1",
                    mode="auto",
                    title=None,
                    metadata={},
                    labels=set(),
                ),
            ],
            total_count=1,
            has_more=False,
            next_cursor=None,
        )
    )

    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    request = TunnelRequest(
        request_id="req-list",
        method="sessions.list",
        params={"agent_id": "agent-1"},
    )

    response = await dispatcher.dispatch(request)

    assert response.error is None
    assert isinstance(response.result, dict)
    assert len(response.result["sessions"]) == 1
    assert response.result["total_count"] == 1
    session_module.find.assert_awaited_once()


async def test_that_dispatcher_routes_sessions_update() -> None:
    session_module = AsyncMock()
    session_module.update = AsyncMock(
        return_value=MagicMock(
            id="sess-1",
            creation_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            customer_id="cust-1",
            agent_id="agent-1",
            mode="auto",
            title="Updated Title",
            consumption_offsets={},
            agent_states=[],
            metadata={},
            labels=set(),
        )
    )

    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    request = TunnelRequest(
        request_id="req-update",
        method="sessions.update",
        params={
            "session_id": "sess-1",
            "title": "Updated Title",
        },
    )

    response = await dispatcher.dispatch(request)

    assert response.error is None
    assert isinstance(response.result, dict)
    assert response.result["title"] == "Updated Title"
    session_module.update.assert_awaited_once()


async def test_that_dispatcher_routes_sessions_delete() -> None:
    session_module = AsyncMock()
    session_module.delete = AsyncMock(return_value=None)

    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    request = TunnelRequest(
        request_id="req-delete",
        method="sessions.delete",
        params={"session_id": "sess-1"},
    )

    response = await dispatcher.dispatch(request)

    assert response.error is None
    session_module.delete.assert_awaited_once()


async def test_that_dispatcher_routes_sessions_read_event() -> None:
    session_module = AsyncMock()
    session_module.read_event = AsyncMock(
        return_value=MagicMock(
            id="evt-1",
            offset=0,
            source="customer",
            kind="message",
            creation_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            data={"message": "Hello"},
            metadata={},
        )
    )

    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    request = TunnelRequest(
        request_id="req-read-evt",
        method="sessions.read_event",
        params={"session_id": "sess-1", "event_id": "evt-1"},
    )

    response = await dispatcher.dispatch(request)

    assert response.error is None
    assert isinstance(response.result, dict)
    assert response.result["id"] == "evt-1"
    session_module.read_event.assert_awaited_once()


async def test_that_dispatcher_routes_sessions_update_event() -> None:
    session_module = AsyncMock()
    session_module.update_event = AsyncMock(
        return_value=MagicMock(
            id="evt-1",
            offset=0,
            source="customer",
            kind="message",
            creation_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            data={"message": "Hello"},
            metadata={"key": "value"},
        )
    )

    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    request = TunnelRequest(
        request_id="req-update-evt",
        method="sessions.update_event",
        params={
            "session_id": "sess-1",
            "event_id": "evt-1",
            "metadata": {"set": {"key": "value"}},
        },
    )

    response = await dispatcher.dispatch(request)

    assert response.error is None
    assert isinstance(response.result, dict)
    assert response.result["id"] == "evt-1"
    session_module.update_event.assert_awaited_once()


async def test_that_dispatcher_routes_sessions_delete_events() -> None:
    session_module = AsyncMock()
    session_module.delete_events = AsyncMock(return_value=None)

    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    request = TunnelRequest(
        request_id="req-del-evts",
        method="sessions.delete_events",
        params={"session_id": "sess-1", "min_offset": 5},
    )

    response = await dispatcher.dispatch(request)

    assert response.error is None
    session_module.delete_events.assert_awaited_once()
