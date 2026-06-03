from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from parlant.core.agents import CompositionMode, MessageOutputMode
from parlant.core.persistence.common import SortDirection
from parlant.core.tunnels import TunnelRequestDispatcher
from parlant.core.tunnels import TunnelRequest
from parlant.core.tunnels import _parse_sort_direction


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


async def test_that_dispatcher_routes_sessions_delete_many() -> None:
    session_module = AsyncMock()
    session_module.find = AsyncMock(
        return_value=MagicMock(
            items=[
                MagicMock(id="sess-1"),
                MagicMock(id="sess-2"),
            ],
            total_count=2,
            has_more=False,
            next_cursor=None,
        )
    )
    session_module.delete = AsyncMock(return_value=None)

    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    request = TunnelRequest(
        request_id="req-delete-many",
        method="sessions.delete_many",
        params={"agent_id": "agent-1"},
    )

    response = await dispatcher.dispatch(request)

    assert response.error is None
    assert isinstance(response.result, dict)
    assert response.result["deleted_session_ids"] == ["sess-1", "sess-2"]
    session_module.find.assert_awaited_once()
    assert session_module.delete.await_count == 2


async def test_that_dispatcher_reports_sessions_delete_many_partial_failure() -> None:
    session_module = AsyncMock()
    session_module.find = AsyncMock(
        return_value=MagicMock(
            items=[
                MagicMock(id="sess-1"),
                MagicMock(id="sess-2"),
            ],
            total_count=2,
            has_more=False,
            next_cursor=None,
        )
    )
    session_module.delete = AsyncMock(side_effect=[None, RuntimeError("boom")])

    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    request = TunnelRequest(
        request_id="req-delete-many-failure",
        method="sessions.delete_many",
        params={"agent_id": "agent-1"},
    )

    response = await dispatcher.dispatch(request)

    assert response.error is not None
    assert "sess-2" in response.error
    assert session_module.delete.await_count == 2


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


async def test_that_dispatcher_routes_agents_list() -> None:
    session_module = AsyncMock()
    agent_module = AsyncMock()
    agent_module.find = AsyncMock(
        return_value=[
            MagicMock(
                id="agent-1",
                name="Agent One",
                description=None,
                max_engine_iterations=3,
                composition_mode=CompositionMode.FLUID,
                message_output_mode=MessageOutputMode.BLOCK,
                tags=[],
            )
        ]
    )
    dispatcher = TunnelRequestDispatcher(
        session_module=session_module,
        agent_module=agent_module,
    )

    response = await dispatcher.dispatch(
        TunnelRequest(request_id="req-agents", method="agents.list", params={})
    )

    assert response.error is None
    assert isinstance(response.result, dict)
    assert response.result["agents"][0]["id"] == "agent-1"
    assert response.result["agents"][0]["composition_mode"] == "fluid"
    assert response.result["agents"][0]["message_output_mode"] == "block"
    agent_module.find.assert_awaited_once()


async def test_that_dispatcher_routes_customers_retrieve() -> None:
    session_module = AsyncMock()
    customer_module = AsyncMock()
    customer_module.read = AsyncMock(
        return_value=MagicMock(
            id="customer-1",
            name="Customer One",
            extra={"tier": "gold"},
            tags=[],
            creation_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    )
    dispatcher = TunnelRequestDispatcher(
        session_module=session_module,
        customer_module=customer_module,
    )

    response = await dispatcher.dispatch(
        TunnelRequest(
            request_id="req-customer",
            method="customers.retrieve",
            params={"customer_id": "customer-1"},
        )
    )

    assert response.error is None
    assert isinstance(response.result, dict)
    assert response.result["customer"]["id"] == "customer-1"
    assert response.result["customer"]["metadata"] == {"tier": "gold"}
    customer_module.read.assert_awaited_once()


async def test_that_dispatcher_routes_tags_list() -> None:
    session_module = AsyncMock()
    tag_module = AsyncMock()
    tag_module.find = AsyncMock(
        return_value=[
            SimpleNamespace(
                id="tag-1",
                name="Tag One",
            )
        ]
    )
    dispatcher = TunnelRequestDispatcher(
        session_module=session_module,
        tag_module=tag_module,
    )

    response = await dispatcher.dispatch(
        TunnelRequest(request_id="req-tags", method="tags.list", params={})
    )

    assert response.error is None
    assert isinstance(response.result, dict)
    assert response.result["tags"][0]["id"] == "tag-1"
    assert response.result["tags"][0]["name"] == "Tag One"
    tag_module.find.assert_awaited_once_with(name=None)


async def test_that_dispatcher_routes_tags_retrieve() -> None:
    session_module = AsyncMock()
    tag_module = AsyncMock()
    tag_module.read = AsyncMock(
        return_value=SimpleNamespace(
            id="tag-1",
            name="Tag One",
        )
    )
    dispatcher = TunnelRequestDispatcher(
        session_module=session_module,
        tag_module=tag_module,
    )

    response = await dispatcher.dispatch(
        TunnelRequest(
            request_id="req-tag",
            method="tags.retrieve",
            params={"tag_id": "tag-1"},
        )
    )

    assert response.error is None
    assert isinstance(response.result, dict)
    assert response.result["tag"]["id"] == "tag-1"
    tag_module.read.assert_awaited_once()


def test_that_parse_sort_direction_returns_none_for_none() -> None:
    assert _parse_sort_direction(None) is None


def test_that_parse_sort_direction_maps_asc_and_desc() -> None:
    assert _parse_sort_direction("asc") is SortDirection.ASC
    assert _parse_sort_direction("desc") is SortDirection.DESC


def test_that_parse_sort_direction_raises_for_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported sort direction"):
        _parse_sort_direction("sideways")


async def test_that_dispatcher_serializes_session_labels_in_sessions_list() -> None:
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
                    labels={"premium", "support"},
                ),
            ],
            total_count=1,
            has_more=False,
            next_cursor=None,
        )
    )

    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    response = await dispatcher.dispatch(
        TunnelRequest(request_id="req-labels", method="sessions.list", params={}),
    )

    assert response.error is None
    assert isinstance(response.result, dict)
    serialized = response.result["sessions"][0]
    assert set(serialized["labels"]) == {"premium", "support"}
