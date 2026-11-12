# Copyright 2026 Emcie Co Ltd.
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

import json
import logging
from typing import Any, AsyncIterator, Mapping

import httpx
from fastapi import Request, status
from pytest import LogCaptureFixture, MonkeyPatch, fixture

from parlant.api.app import create_api_app
from parlant.api.authorization import (
    AuthorizationPolicy,
    DevelopmentAuthorizationPolicy,
    Operation,
)
from parlant.api.sessions import REDACTED_REASONING_LABEL
from parlant.api.sse import (
    SSE_HEARTBEAT_FRAME,
    encode_event_cursor,
    format_sse_frame,
    format_sse_retry_frame,
    parse_event_cursor,
    sse_heartbeat_seconds,
    sse_retry_milliseconds,
)
from parlant.core import metadata as metadata_utils
from parlant.core.common import generate_id
from parlant.core.container import Container
from parlant.core.sessions import (
    EventId,
    EventKind,
    EventSource,
    SessionId,
    SessionStore,
)


async def create_event(
    container: Container,
    session_id: SessionId,
    data: dict[str, Any] | None = None,
    *,
    kind: EventKind = EventKind.CUSTOM,
    source: EventSource = EventSource.CUSTOMER,
    metadata: Mapping[str, Any] | None = None,
) -> EventId:
    event = await container[SessionStore].create_event(
        session_id=session_id,
        source=source,
        kind=kind,
        trace_id=generate_id(),
        data=data or {},
        metadata=dict(metadata or {}),
    )

    return event.id


def frames_of(stream_text: str) -> list[dict[str, Any]]:
    """Split a raw SSE response body into frames, keeping every field we assert on."""

    frames: list[dict[str, Any]] = []

    for block in stream_text.split("\n\n"):
        if not block.strip():
            continue

        frame: dict[str, Any] = {"id": None, "data": [], "comments": [], "retry": None}

        for line in block.splitlines():
            if line.startswith(":"):
                frame["comments"].append(line)
            elif line.startswith("id: "):
                frame["id"] = line[len("id: ") :]
            elif line.startswith("retry: "):
                frame["retry"] = line[len("retry: ") :]
            elif line.startswith("data: "):
                frame["data"].append(line[len("data: ") :])

        frames.append(frame)

    return frames


def data_frames(stream_text: str) -> list[dict[str, Any]]:
    """The frames that carried an event, i.e. everything but retry and keep-alive."""

    return [f for f in frames_of(stream_text) if f["data"]]


def payload_of(frame: Mapping[str, Any]) -> dict[str, Any]:
    data = frame["data"]

    return json.loads("\n".join(data))  # type: ignore[no-any-return]


async def collect_stream(
    async_client: httpx.AsyncClient,
    url: str,
    params: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
) -> str:
    async with async_client.stream("GET", url, params=params, headers=headers) as response:
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        return "".join([chunk async for chunk in response.aiter_text()])


def list_events_params(wait_for_data: int = 0) -> dict[str, Any]:
    return {"sse": "true", "wait_for_data": wait_for_data}


def read_event_params(wait_for_data: int = 5) -> dict[str, Any]:
    return {"sse": "true", "wait_for_data": wait_for_data}


def test_that_a_cursor_is_read_only_from_a_well_formed_header() -> None:
    assert parse_event_cursor(None) is None
    assert parse_event_cursor("") is None
    assert parse_event_cursor("nope") is None
    assert parse_event_cursor("1:2:3") is None
    assert parse_event_cursor("-1") is None
    assert parse_event_cursor("1:-2") is None

    cursor = parse_event_cursor("12")
    assert cursor is not None
    assert (cursor.offset, cursor.chunk_count) == (12, 0)

    cursor = parse_event_cursor(" 12:3 ")
    assert cursor is not None
    assert (cursor.offset, cursor.chunk_count) == (12, 3)


def test_that_a_frame_without_a_resume_id_is_byte_identical_to_the_previous_format() -> None:
    assert format_sse_frame('{"offset":0}') == 'data: {"offset":0}\n\n'


def test_that_a_frame_carries_its_resume_id_before_the_payload() -> None:
    frame = format_sse_frame('{"offset":0}', event_id=encode_event_cursor(0))

    assert frame == 'id: 0\ndata: {"offset":0}\n\n'
    assert encode_event_cursor(0, 3) == "0:3"


def test_that_a_multi_line_payload_becomes_several_data_lines() -> None:
    assert format_sse_frame("a\nb") == "data: a\ndata: b\n\n"


def test_that_the_retry_frame_announces_the_delay_it_is_given() -> None:
    assert format_sse_retry_frame(1500) == "retry: 1500\n\n"
    assert format_sse_retry_frame(0) == ""
    assert SSE_HEARTBEAT_FRAME.startswith(":")


def test_that_the_timings_default_when_the_environment_says_nothing(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("PARLANT_SSE_HEARTBEAT_SECONDS", raising=False)
    monkeypatch.delenv("PARLANT_SSE_RETRY_MS", raising=False)

    assert sse_heartbeat_seconds() == 15.0
    assert sse_retry_milliseconds() == 3000


def test_that_the_timings_are_read_from_the_environment(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("PARLANT_SSE_HEARTBEAT_SECONDS", "0.5")
    monkeypatch.setenv("PARLANT_SSE_RETRY_MS", "1500")

    assert sse_heartbeat_seconds() == 0.5
    assert sse_retry_milliseconds() == 1500


def test_that_an_invalid_timing_is_reported_and_replaced_by_the_default(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    monkeypatch.setenv("PARLANT_SSE_RETRY_MS", "soon")
    monkeypatch.setenv("PARLANT_SSE_HEARTBEAT_SECONDS", "-1")

    with caplog.at_level(logging.WARNING, logger="parlant"):
        assert sse_retry_milliseconds() == 3000
        assert sse_heartbeat_seconds() == 15.0

    assert "Invalid PARLANT_SSE_RETRY_MS='soon'" in caplog.text
    assert "Invalid PARLANT_SSE_HEARTBEAT_SECONDS='-1'" in caplog.text


async def test_that_streamed_events_carry_their_offset_as_the_frame_id(
    async_client: httpx.AsyncClient,
    container: Container,
    session_id: SessionId,
) -> None:
    for _ in range(3):
        await create_event(container, session_id)

    frames = data_frames(
        await collect_stream(async_client, f"/sessions/{session_id}/events", list_events_params())
    )

    assert [f["id"] for f in frames] == ["0", "1", "2"]
    assert [payload_of(f)["offset"] for f in frames] == [0, 1, 2]


async def test_that_a_stream_resumes_after_the_event_named_by_last_event_id(
    async_client: httpx.AsyncClient,
    container: Container,
    session_id: SessionId,
) -> None:
    for _ in range(3):
        await create_event(container, session_id)

    frames = data_frames(
        await collect_stream(
            async_client,
            f"/sessions/{session_id}/events",
            list_events_params(),
            headers={"Last-Event-ID": "1"},
        )
    )

    assert [payload_of(f)["offset"] for f in frames] == [2]


async def test_that_the_later_of_min_offset_and_last_event_id_wins(
    async_client: httpx.AsyncClient,
    container: Container,
    session_id: SessionId,
) -> None:
    for _ in range(4):
        await create_event(container, session_id)

    # min_offset is behind the header, so the reconnect still moves forward.
    frames = data_frames(
        await collect_stream(
            async_client,
            f"/sessions/{session_id}/events",
            {**list_events_params(), "min_offset": 1},
            headers={"Last-Event-ID": "2"},
        )
    )

    assert [payload_of(f)["offset"] for f in frames] == [3]

    # min_offset is ahead of the header, so the explicit request wins.
    frames = data_frames(
        await collect_stream(
            async_client,
            f"/sessions/{session_id}/events",
            {**list_events_params(), "min_offset": 3},
            headers={"Last-Event-ID": "1"},
        )
    )

    assert [payload_of(f)["offset"] for f in frames] == [3]


async def test_that_an_unreadable_last_event_id_replays_from_the_first_event(
    async_client: httpx.AsyncClient,
    container: Container,
    session_id: SessionId,
) -> None:
    for _ in range(2):
        await create_event(container, session_id)

    frames = data_frames(
        await collect_stream(
            async_client,
            f"/sessions/{session_id}/events",
            list_events_params(),
            headers={"Last-Event-ID": "not-a-cursor"},
        )
    )

    assert [payload_of(f)["offset"] for f in frames] == [0, 1]


async def test_that_the_stream_announces_the_configured_retry_delay(
    async_client: httpx.AsyncClient,
    container: Container,
    session_id: SessionId,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARLANT_SSE_RETRY_MS", "1500")

    await create_event(container, session_id)

    frames = frames_of(
        await collect_stream(
            async_client,
            f"/sessions/{session_id}/events",
            list_events_params(),
        )
    )

    assert frames[0]["retry"] == "1500"


async def test_that_a_retry_delay_of_zero_omits_the_retry_line(
    async_client: httpx.AsyncClient,
    container: Container,
    session_id: SessionId,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARLANT_SSE_RETRY_MS", "0")

    await create_event(container, session_id)

    frames = frames_of(
        await collect_stream(
            async_client,
            f"/sessions/{session_id}/events",
            list_events_params(),
        )
    )

    assert all(f["retry"] is None for f in frames)


async def test_that_an_idle_stream_is_kept_alive(
    async_client: httpx.AsyncClient,
    container: Container,
    session_id: SessionId,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARLANT_SSE_HEARTBEAT_SECONDS", "0.05")

    await create_event(container, session_id)

    frames = frames_of(
        await collect_stream(
            async_client,
            f"/sessions/{session_id}/events",
            list_events_params(wait_for_data=1),
        )
    )

    assert any(f["comments"] == [SSE_HEARTBEAT_FRAME.strip()] for f in frames)
    # The stream still delivers the event itself, and still closes on its own.
    assert [payload_of(f)["offset"] for f in frames if f["data"]] == [0]


async def test_that_a_heartbeat_of_zero_disables_the_keep_alive(
    async_client: httpx.AsyncClient,
    container: Container,
    session_id: SessionId,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARLANT_SSE_HEARTBEAT_SECONDS", "0")

    await create_event(container, session_id)

    frames = frames_of(
        await collect_stream(
            async_client,
            f"/sessions/{session_id}/events",
            list_events_params(wait_for_data=1),
        )
    )

    assert not any(f["comments"] for f in frames)


async def test_that_a_single_event_stream_carries_its_offset_and_chunk_count(
    async_client: httpx.AsyncClient,
    container: Container,
    session_id: SessionId,
) -> None:
    event_id = await create_event(container, session_id, data={"chunks": ["hello", None]})

    frames = data_frames(
        await collect_stream(
            async_client,
            f"/sessions/{session_id}/events/{event_id}",
            read_event_params(),
        )
    )

    assert [f["id"] for f in frames] == ["0:2"]
    assert payload_of(frames[0])["data"]["chunks"] == ["hello", None]


async def test_that_a_resumed_chunk_cursor_does_not_replay_chunks_it_already_has(
    async_client: httpx.AsyncClient,
    container: Container,
    session_id: SessionId,
) -> None:
    event_id = await create_event(container, session_id, data={"chunks": ["a", "b"]})

    await container[SessionStore].update_event(
        session_id,
        event_id,
        {"data": {"chunks": ["a", "b", "c", None]}},
    )

    frames = data_frames(
        await collect_stream(
            async_client,
            f"/sessions/{session_id}/events/{event_id}",
            read_event_params(),
            headers={"Last-Event-ID": "0:2"},
        )
    )

    # The client already had two chunks, so the first thing it gets is the event
    # as it stands now - never the version it resumed from.
    assert [f["id"] for f in frames] == ["0:4"]
    assert payload_of(frames[0])["data"]["chunks"] == ["a", "b", "c", None]


async def test_that_a_non_streaming_event_is_sent_once_and_then_closes(
    async_client: httpx.AsyncClient,
    container: Container,
    session_id: SessionId,
) -> None:
    event_id = await create_event(container, session_id, data={"message": "hi"})

    frames = data_frames(
        await collect_stream(
            async_client,
            f"/sessions/{session_id}/events/{event_id}",
            read_event_params(),
        )
    )

    assert len(frames) == 1
    assert frames[0]["id"] == "0"
    assert payload_of(frames[0])["data"] == {"message": "hi"}


async def test_that_streaming_still_ends_when_no_event_arrives_in_time(
    async_client: httpx.AsyncClient,
    container: Container,
    session_id: SessionId,
) -> None:
    frames = data_frames(
        await collect_stream(
            async_client,
            f"/sessions/{session_id}/events",
            list_events_params(wait_for_data=1),
        )
    )

    assert frames == []


class _NoReasoningPolicy(DevelopmentAuthorizationPolicy):
    async def check_permission(self, request: Request, operation: Operation) -> bool:
        if operation is Operation.READ_EVENT_REASONING:
            return False
        return await super().check_permission(request, operation)


@fixture
async def no_reasoning_client(container: Container) -> AsyncIterator[httpx.AsyncClient]:
    restricted = container.clone()
    restricted[AuthorizationPolicy] = _NoReasoningPolicy()

    app = await create_api_app(restricted)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


async def test_that_a_streamed_frame_hides_reasoning_from_a_caller_without_the_permission(
    no_reasoning_client: httpx.AsyncClient,
    container: Container,
    session_id: SessionId,
) -> None:
    await create_event(
        container,
        session_id,
        data={
            "status": "processing",
            "message": "The customer is asking about order 123; I should check the warehouse.",
            "chunks": ["The customer is asking", " about order 123;", None],
        },
        kind=EventKind.STATUS,
        source=EventSource.AI_AGENT,
        metadata=metadata_utils.set_internal_value({}, ("internal_reasoning",), True),
    )

    frames = data_frames(
        await collect_stream(
            no_reasoning_client,
            f"/sessions/{session_id}/events",
            list_events_params(),
        )
    )

    assert len(frames) == 1

    data = payload_of(frames[0])["data"]
    assert data["message"] == REDACTED_REASONING_LABEL
    assert "chunks" not in data
    assert "warehouse" not in json.dumps(data)
