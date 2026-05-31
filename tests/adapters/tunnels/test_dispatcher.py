from unittest.mock import AsyncMock, MagicMock
from parlant.adapters.tunnels.dispatcher import TunnelRequestDispatcher
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
