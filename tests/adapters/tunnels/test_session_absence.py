"""Source absence is established by a parent read, never exception wording."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.agents import AgentId
from parlant.core.app_modules.sessions import SessionModule
from parlant.core.common import ItemNotFoundError, UniqueId
from parlant.core.customers import CustomerId
from parlant.core.sessions import SessionDocumentStore
from parlant.core.tunnel_handlers.sessions import SessionTunnelHandlers
from parlant.core.tunnels import TunnelRequest
from tests.adapters.tunnels.test_dispatcher import make_dispatcher


def session_module(store: Any) -> SessionModule:
    provider = MagicMock()
    provider.get_store.return_value = store
    return SessionModule(
        request_context=MagicMock(),
        logger=MagicMock(),
        meter=MagicMock(),
        tracer=MagicMock(),
        session_listener=AsyncMock(),
        model_router=MagicMock(),
        engine_registry=MagicMock(),
        event_emitter_factory=MagicMock(),
        background_task_service=MagicMock(),
        health_reporter=MagicMock(),
        store_provider=provider,
        application_context=MagicMock(),
    )


@pytest.mark.parametrize("method", ["sessions.read", "sessions.list_events", "sessions.delete"])
async def test_missing_session_handler_uses_item_not_found(method: str) -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        handler = SessionTunnelHandlers(session_module(store)).handler_set().unary[method]
        with pytest.raises(ItemNotFoundError) as caught:
            await handler({"session_id": "absent", "wait_for_data": 0})
        assert caught.value.item_id == "absent"


@pytest.mark.parametrize("method", ["sessions.read", "sessions.list_events", "sessions.delete"])
async def test_missing_parent_session_has_typed_wire_error(method: str) -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        dispatcher = make_dispatcher(session_module=session_module(store))
        response = await dispatcher.dispatch(
            TunnelRequest(
                "req-1",
                method,
                {"session_id": "absent", "wait_for_data": 0},
            )
        )
        assert json.loads(json.dumps(response.to_dict())) == {
            "request_id": "req-1",
            "error": {
                "code": "session_not_found",
                "message": "Session not found",
                "session_id": "absent",
            },
        }


@pytest.mark.parametrize("wait_for_data", [0, 5])
async def test_empty_events_verify_parent_session(wait_for_data: int) -> None:
    module = AsyncMock()
    module.read.side_effect = ItemNotFoundError(UniqueId("absent"))
    module.find_events.return_value = []
    module.wait_for_more_events.return_value = False
    response = await make_dispatcher(session_module=module).dispatch(
        TunnelRequest(
            "req-1",
            "sessions.list_events",
            {"session_id": "absent", "wait_for_data": wait_for_data},
        )
    )
    assert response.error == {
        "code": "session_not_found",
        "message": "Session not found",
        "session_id": "absent",
    }
    module.wait_for_more_events.assert_not_awaited()


async def test_session_removed_while_waiting_is_confirmed_by_parent_read() -> None:
    module = AsyncMock()
    module.read.side_effect = [MagicMock(), ItemNotFoundError(UniqueId("absent"))]
    module.wait_for_more_events.return_value = False
    response = await make_dispatcher(session_module=module).dispatch(
        TunnelRequest(
            "req-1",
            "sessions.list_events",
            {"session_id": "absent", "wait_for_data": 5},
        )
    )
    assert response.error == {
        "code": "session_not_found",
        "message": "Session not found",
        "session_id": "absent",
    }


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Session not found"),
        TimeoutError("Session not found"),
        PermissionError("Session not found"),
        ItemNotFoundError(UniqueId("other"), "Session not found"),
        ItemNotFoundError(UniqueId("absent"), item_type="event"),
    ],
)
@pytest.mark.parametrize("method", ["sessions.read", "sessions.list_events", "sessions.delete"])
async def test_other_parent_read_failures_remain_untyped(method: str, error: Exception) -> None:
    module = AsyncMock()
    module.read.side_effect = error
    module.delete.side_effect = error
    response = await make_dispatcher(session_module=module).dispatch(
        TunnelRequest(
            "req-1",
            method,
            {"session_id": "absent", "wait_for_data": 0},
        )
    )
    assert isinstance(response.error, str)


@pytest.mark.parametrize(
    "method, operation",
    [
        ("sessions.read_event", "read_event"),
        ("sessions.delete", "delete"),
        ("sessions.list_events", "find_events"),
    ],
)
@pytest.mark.parametrize("item_type", [None, "event"])
async def test_missing_related_item_cannot_delete_existing_parent(
    method: str, operation: str, item_type: str | None
) -> None:
    module = AsyncMock()
    # Even an event with the same ID cannot be interpreted as the parent session.
    getattr(module, operation).side_effect = ItemNotFoundError(
        UniqueId("session-1"), item_type=item_type
    )
    response = await make_dispatcher(session_module=module).dispatch(
        TunnelRequest(
            "req-1",
            method,
            {"session_id": "session-1", "event_id": "session-1", "wait_for_data": 0},
        )
    )
    assert isinstance(response.error, str)


async def test_existing_session_with_no_events_is_not_missing() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await store.create_session(CustomerId("customer"), AgentId("agent"))
        response = await make_dispatcher(session_module=session_module(store)).dispatch(
            TunnelRequest(
                "req-1",
                "sessions.list_events",
                {"session_id": session.id, "wait_for_data": 0},
            )
        )
        assert response.to_dict() == {"request_id": "req-1", "result": {"events": []}}


async def test_session_removed_during_event_lookup_is_confirmed_by_parent_read() -> None:
    module = AsyncMock()
    module.read.side_effect = [MagicMock(), ItemNotFoundError(UniqueId("absent"))]
    module.find_events.side_effect = ItemNotFoundError(UniqueId("absent"))
    response = await make_dispatcher(session_module=module).dispatch(
        TunnelRequest(
            "req-1",
            "sessions.list_events",
            {"session_id": "absent", "wait_for_data": 0},
        )
    )
    assert response.error == {
        "code": "session_not_found",
        "message": "Session not found",
        "session_id": "absent",
    }


async def test_source_socket_sends_confirmed_absence_without_stringifying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from parlant.adapters.modules.parlant_dashboard import ParlantDashboardTunnelService
    from parlant.core.tracer import LocalTracer

    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        dispatcher = make_dispatcher(session_module=session_module(store))
        socket = MagicMock()
        socket.__aiter__.return_value = [
            json.dumps(
                {
                    "request_id": "req-wire",
                    "method": "sessions.read",
                    "params": {"session_id": "absent"},
                }
            )
        ]
        socket.send = AsyncMock()
        connection = MagicMock()
        connection.__aenter__.return_value = socket
        monkeypatch.setattr(
            "parlant.adapters.modules.parlant_dashboard.tunnel.websockets.connect",
            MagicMock(return_value=connection),
        )
        tunnel = ParlantDashboardTunnelService(
            "ws://unused",
            "test-token",
            dispatcher,
            tracer=LocalTracer(),
            logger=MagicMock(),
        )
        tunnel._running = True
        await tunnel._connect_and_listen()
        socket.send.assert_awaited_once()
        assert json.loads(socket.send.await_args.args[0]) == {
            "request_id": "req-wire",
            "error": {
                "code": "session_not_found",
                "message": "Session not found",
                "session_id": "absent",
            },
        }


async def test_bulk_delete_deduplicates_sessions_from_multiple_stores() -> None:
    module = AsyncMock()
    module.find.return_value = MagicMock(
        items=[MagicMock(id="session-1"), MagicMock(id="session-1")]
    )
    module.delete.side_effect = [None, RuntimeError("second delete failed")]
    response = await make_dispatcher(session_module=module).dispatch(
        TunnelRequest("req-1", "sessions.delete_many", {})
    )
    assert response.error is None
    assert response.result == {"deleted_session_ids": ["session-1"], "failed_session_ids": []}
    module.delete.assert_awaited_once_with(session_id="session-1")


@pytest.mark.parametrize("invalid_id", [None, "", 404])
async def test_bulk_delete_rejects_invalid_source_ids_before_any_deletion(invalid_id: Any) -> None:
    module = AsyncMock()
    module.find.return_value = MagicMock(
        items=[MagicMock(id="session-1"), MagicMock(id=invalid_id)]
    )
    response = await make_dispatcher(session_module=module).dispatch(
        TunnelRequest("req-1", "sessions.delete_many", {})
    )
    assert isinstance(response.error, str)
    module.delete.assert_not_awaited()


async def test_bulk_delete_counts_session_removed_after_listing_only_after_confirmation() -> None:
    module = AsyncMock()
    module.find.return_value = MagicMock(
        items=[MagicMock(id="session-1"), MagicMock(id="session-2")]
    )
    module.delete.side_effect = [
        ItemNotFoundError(UniqueId("session-1")),
        TimeoutError("Session not found"),
    ]
    module.read.side_effect = ItemNotFoundError(UniqueId("session-1"))
    response = await make_dispatcher(session_module=module).dispatch(
        TunnelRequest("req-1", "sessions.delete_many", {})
    )
    assert response.error is None
    assert response.result == {
        "deleted_session_ids": ["session-1"],
        "failed_session_ids": ["session-2"],
    }
    module.read.assert_awaited_once_with(session_id="session-1")


async def test_bulk_delete_missing_related_item_is_a_failure() -> None:
    module = AsyncMock()
    module.find.return_value = MagicMock(items=[MagicMock(id="session-1")])
    module.delete.side_effect = ItemNotFoundError(UniqueId("session-1"), item_type="event")
    response = await make_dispatcher(session_module=module).dispatch(
        TunnelRequest("req-1", "sessions.delete_many", {})
    )
    assert response.error is None
    assert response.result == {
        "deleted_session_ids": [],
        "failed_session_ids": ["session-1"],
    }
