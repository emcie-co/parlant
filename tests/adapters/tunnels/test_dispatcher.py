from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from parlant.core.agents import CompositionMode, MessageOutputMode
from parlant.core.app_modules.common import decode_cursor, encode_cursor
from parlant.core.persistence.common import Cursor, ObjectId, SortDirection
from parlant.core.app_modules.sessions import Moderation
from parlant.core.sessions import EventKind, EventSource
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
    create_call = session_module.create_customer_message.await_args.kwargs
    assert create_call["moderation"] == Moderation.NONE


async def test_that_dispatcher_honors_sessions_create_event_moderation() -> None:
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
            "moderation": "paranoid",
        },
    )

    response = await dispatcher.dispatch(request)

    assert response.request_id == "req-1"
    assert response.error is None
    create_call = session_module.create_customer_message.await_args.kwargs
    assert create_call["moderation"] == Moderation.PARANOID


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


async def test_that_dispatcher_routes_sessions_create_without_customer_id() -> None:
    session_module = AsyncMock()
    session_module.create = AsyncMock(
        return_value=MagicMock(
            id="sess-1",
            creation_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            customer_id="guest",
            agent_id="agent-1",
            mode="auto",
            title="Test Session",
            consumption_offsets={},
            agent_states=[],
            metadata={"source": "simulation"},
            labels=set(),
        )
    )

    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    request = TunnelRequest(
        request_id="req-create",
        method="sessions.create",
        params={
            "agent_id": "agent-1",
            "title": "Test Session",
            "metadata": {"source": "simulation"},
        },
    )

    response = await dispatcher.dispatch(request)

    assert response.request_id == "req-create"
    assert response.error is None
    assert isinstance(response.result, dict)
    assert response.result["session_id"] == "sess-1"
    session_module.create.assert_awaited_once_with(
        customer_id="guest",
        agent_id="agent-1",
        title="Test Session",
        allow_greeting=False,
        metadata={"source": "simulation"},
    )


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
            source=EventSource.CUSTOMER,
            kind=EventKind.MESSAGE,
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
    assert response.result["source"] == "customer"
    assert response.result["kind"] == "message"
    session_module.read_event.assert_awaited_once()


async def test_that_dispatcher_routes_sessions_list_events_with_serialized_enums() -> None:
    session_module = AsyncMock()
    session_module.wait_for_more_events = AsyncMock(return_value=True)
    session_module.find_events = AsyncMock(
        return_value=[
            MagicMock(
                id="evt-1",
                offset=0,
                source=EventSource.CUSTOMER,
                kind=EventKind.MESSAGE,
                creation_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
                data={"message": "Hello"},
                metadata={},
            )
        ]
    )

    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    request = TunnelRequest(
        request_id="req-list-evts",
        method="sessions.list_events",
        params={"session_id": "sess-1", "min_offset": 0},
    )

    response = await dispatcher.dispatch(request)

    assert response.error is None
    assert isinstance(response.result, dict)
    assert response.result["events"][0]["source"] == "customer"
    assert response.result["events"][0]["kind"] == "message"
    session_module.wait_for_more_events.assert_awaited_once()
    wait_call = session_module.wait_for_more_events.await_args.kwargs
    assert wait_call["session_id"] == "sess-1"
    assert wait_call["min_offset"] == 0
    assert 0 < wait_call["timeout"].remaining() <= 60
    session_module.find_events.assert_awaited_once()


async def test_that_dispatcher_waits_when_listing_session_events_with_timeout() -> None:
    session_module = AsyncMock()
    session_module.wait_for_more_events = AsyncMock(return_value=True)
    session_module.find_events = AsyncMock(return_value=[])

    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    request = TunnelRequest(
        request_id="req-list-evts-wait",
        method="sessions.list_events",
        params={
            "session_id": "sess-1",
            "min_offset": 2,
            "source": "ai_agent",
            "kinds": "message,status",
            "trace_id": "trace-1",
            "wait_for_data": 5,
        },
    )

    response = await dispatcher.dispatch(request)

    assert response.error is None
    assert response.result == {"events": []}
    session_module.wait_for_more_events.assert_awaited_once()
    wait_call = session_module.wait_for_more_events.await_args.kwargs
    assert wait_call["session_id"] == "sess-1"
    assert wait_call["min_offset"] == 2
    assert wait_call["source"] == EventSource.AI_AGENT
    assert wait_call["kinds"] == [EventKind.MESSAGE, EventKind.STATUS]
    assert wait_call["trace_id"] == "trace-1"
    assert 0 < wait_call["timeout"].remaining() <= 5
    session_module.find_events.assert_awaited_once()


async def test_that_dispatcher_returns_empty_events_when_list_events_wait_times_out() -> None:
    session_module = AsyncMock()
    session_module.wait_for_more_events = AsyncMock(return_value=False)
    session_module.find_events = AsyncMock()

    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    request = TunnelRequest(
        request_id="req-list-evts-timeout",
        method="sessions.list_events",
        params={"session_id": "sess-1", "min_offset": 2, "wait_for_data": 5},
    )

    response = await dispatcher.dispatch(request)

    assert response.error is None
    assert response.result == {"events": []}
    session_module.wait_for_more_events.assert_awaited_once()
    session_module.find_events.assert_not_awaited()


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


async def test_that_dispatcher_forwards_cursor_to_sessions_list() -> None:
    session_module = AsyncMock()
    session_module.find = AsyncMock(
        return_value=MagicMock(items=[], total_count=0, has_more=False, next_cursor=None),
    )

    encoded = encode_cursor(Cursor(creation_utc="2026-01-01T00:00:00+00:00", id=ObjectId("sess-1")))

    dispatcher = TunnelRequestDispatcher(session_module=session_module)
    response = await dispatcher.dispatch(
        TunnelRequest(
            request_id="req-cursor",
            method="sessions.list",
            params={"cursor": encoded},
        ),
    )

    assert response.error is None
    session_module.find.assert_awaited_once()
    kwargs = session_module.find.await_args.kwargs
    assert kwargs["cursor"] == Cursor(
        creation_utc="2026-01-01T00:00:00+00:00", id=ObjectId("sess-1")
    )


async def test_that_dispatcher_treats_invalid_cursor_as_no_cursor_in_sessions_list() -> None:
    session_module = AsyncMock()
    session_module.find = AsyncMock(
        return_value=MagicMock(items=[], total_count=0, has_more=False, next_cursor=None),
    )

    dispatcher = TunnelRequestDispatcher(session_module=session_module)
    response = await dispatcher.dispatch(
        TunnelRequest(
            request_id="req-bad-cursor",
            method="sessions.list",
            params={"cursor": "not-a-real-cursor"},
        ),
    )

    assert response.error is None
    session_module.find.assert_awaited_once()
    assert session_module.find.await_args.kwargs["cursor"] is None


async def test_that_dispatcher_forwards_sort_direction_to_sessions_list() -> None:
    session_module = AsyncMock()
    session_module.find = AsyncMock(
        return_value=MagicMock(items=[], total_count=0, has_more=False, next_cursor=None),
    )

    dispatcher = TunnelRequestDispatcher(session_module=session_module)
    response = await dispatcher.dispatch(
        TunnelRequest(
            request_id="req-sort",
            method="sessions.list",
            params={"sort_direction": "desc"},
        ),
    )

    assert response.error is None
    session_module.find.assert_awaited_once()
    assert session_module.find.await_args.kwargs["sort_direction"] is SortDirection.DESC


async def test_that_dispatcher_rejects_unsupported_sort_direction_in_sessions_list() -> None:
    session_module = AsyncMock()
    session_module.find = AsyncMock()

    dispatcher = TunnelRequestDispatcher(session_module=session_module)
    response = await dispatcher.dispatch(
        TunnelRequest(
            request_id="req-bad-sort",
            method="sessions.list",
            params={"sort_direction": "up"},
        ),
    )

    assert response.error is not None
    assert "Unsupported sort direction" in response.error
    session_module.find.assert_not_awaited()


async def test_that_dispatcher_forwards_labels_filter_to_sessions_list() -> None:
    session_module = AsyncMock()
    session_module.find = AsyncMock(
        return_value=MagicMock(items=[], total_count=0, has_more=False, next_cursor=None),
    )

    dispatcher = TunnelRequestDispatcher(session_module=session_module)
    response = await dispatcher.dispatch(
        TunnelRequest(
            request_id="req-labels",
            method="sessions.list",
            params={"labels": ["premium", "support"]},
        ),
    )

    assert response.error is None
    session_module.find.assert_awaited_once()
    assert session_module.find.await_args.kwargs["labels"] == {"premium", "support"}


async def test_that_dispatcher_forwards_min_modified_utc_to_sessions_list() -> None:
    session_module = AsyncMock()
    session_module.find = AsyncMock(
        return_value=MagicMock(items=[], total_count=0, has_more=False, next_cursor=None),
    )

    dispatcher = TunnelRequestDispatcher(session_module=session_module)
    response = await dispatcher.dispatch(
        TunnelRequest(
            request_id="req-min-modified",
            method="sessions.list",
            params={"min_modified_utc": "2026-01-02T03:04:05+00:00"},
        ),
    )

    assert response.error is None
    session_module.find.assert_awaited_once()
    assert session_module.find.await_args.kwargs["min_modified_utc"] == datetime(
        2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc
    )


async def test_that_dispatcher_returns_encoded_next_cursor_in_sessions_list() -> None:
    real_cursor = Cursor(creation_utc="2026-01-02T00:00:00+00:00", id=ObjectId("sess-42"))
    session_module = AsyncMock()
    session_module.find = AsyncMock(
        return_value=MagicMock(items=[], total_count=0, has_more=True, next_cursor=real_cursor),
    )

    dispatcher = TunnelRequestDispatcher(session_module=session_module)
    response = await dispatcher.dispatch(
        TunnelRequest(request_id="req-next", method="sessions.list", params={}),
    )

    assert response.error is None
    assert isinstance(response.result, dict)
    encoded = response.result["next_cursor"]
    assert isinstance(encoded, str) and encoded
    assert decode_cursor(encoded) == real_cursor


async def test_that_dispatcher_returns_null_next_cursor_when_no_more_sessions() -> None:
    session_module = AsyncMock()
    session_module.find = AsyncMock(
        return_value=MagicMock(items=[], total_count=0, has_more=False, next_cursor=None),
    )

    dispatcher = TunnelRequestDispatcher(session_module=session_module)
    response = await dispatcher.dispatch(
        TunnelRequest(request_id="req-end", method="sessions.list", params={}),
    )

    assert response.error is None
    assert isinstance(response.result, dict)
    assert response.result["next_cursor"] is None


async def test_that_dispatcher_routes_customers_list() -> None:
    session_module = AsyncMock()
    customer_module = AsyncMock()
    customer_module.find = AsyncMock(
        return_value=MagicMock(
            items=[
                MagicMock(
                    id="customer-1",
                    name="Customer One",
                    extra={"tier": "gold"},
                    tags=[],
                    creation_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
                ),
            ],
            total_count=1,
            has_more=False,
            next_cursor=None,
        )
    )

    dispatcher = TunnelRequestDispatcher(
        session_module=session_module,
        customer_module=customer_module,
    )

    response = await dispatcher.dispatch(
        TunnelRequest(request_id="req-customers", method="customers.list", params={}),
    )

    assert response.error is None
    assert isinstance(response.result, dict)
    assert len(response.result["customers"]) == 1
    assert response.result["customers"][0]["id"] == "customer-1"
    assert response.result["total_count"] == 1
    assert response.result["has_more"] is False
    assert response.result["next_cursor"] is None
    customer_module.find.assert_awaited_once()


async def test_that_dispatcher_forwards_pagination_to_customers_list() -> None:
    session_module = AsyncMock()
    customer_module = AsyncMock()
    customer_module.find = AsyncMock(
        return_value=MagicMock(items=[], total_count=0, has_more=False, next_cursor=None),
    )

    encoded = encode_cursor(Cursor(creation_utc="2026-01-01T00:00:00+00:00", id=ObjectId("cust-1")))

    dispatcher = TunnelRequestDispatcher(
        session_module=session_module,
        customer_module=customer_module,
    )
    response = await dispatcher.dispatch(
        TunnelRequest(
            request_id="req-customers-page",
            method="customers.list",
            params={"limit": 5, "cursor": encoded, "sort_direction": "asc"},
        ),
    )

    assert response.error is None
    customer_module.find.assert_awaited_once()
    kwargs = customer_module.find.await_args.kwargs
    assert kwargs["limit"] == 5
    assert kwargs["cursor"] == Cursor(
        creation_utc="2026-01-01T00:00:00+00:00", id=ObjectId("cust-1")
    )
    assert kwargs["sort_direction"] is SortDirection.ASC


async def test_that_dispatcher_returns_encoded_next_cursor_in_customers_list() -> None:
    real_cursor = Cursor(creation_utc="2026-01-02T00:00:00+00:00", id=ObjectId("cust-42"))
    session_module = AsyncMock()
    customer_module = AsyncMock()
    customer_module.find = AsyncMock(
        return_value=MagicMock(items=[], total_count=0, has_more=True, next_cursor=real_cursor),
    )

    dispatcher = TunnelRequestDispatcher(
        session_module=session_module,
        customer_module=customer_module,
    )
    response = await dispatcher.dispatch(
        TunnelRequest(request_id="req-customers-next", method="customers.list", params={}),
    )

    assert response.error is None
    assert isinstance(response.result, dict)
    encoded = response.result["next_cursor"]
    assert isinstance(encoded, str) and encoded
    assert decode_cursor(encoded) == real_cursor


async def test_that_dispatcher_returns_error_when_customer_module_missing_for_list() -> None:
    session_module = AsyncMock()
    dispatcher = TunnelRequestDispatcher(session_module=session_module)

    response = await dispatcher.dispatch(
        TunnelRequest(request_id="req-no-cust-mod", method="customers.list", params={}),
    )

    assert response.error is not None
    assert "Customer module" in response.error
