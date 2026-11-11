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

import asyncio
import json
from typing import Any, Mapping

import httpx
from fastapi import status
from pytest import MonkeyPatch

from parlant.api.sse import (
    ChunkDelta,
    SSE_HEARTBEAT_FRAME,
    encode_event_cursor,
    format_sse_frame,
    format_sse_retry_frame,
    parse_event_cursor,
)
from parlant.core.common import generate_id
from parlant.core.container import Container
from parlant.core.sessions import (
    EventId,
    EventKind,
    EventSource,
    SessionId,
    SessionStore,
)

import parlant.api.sessions as sessions_api


async def create_event(
    container: Container,
    session_id: SessionId,
    data: dict[str, Any] | None = None,
) -> EventId:
    event = await container[SessionStore].create_event(
        session_id=session_id,
        source=EventSource.CUSTOMER,
        kind=EventKind.CUSTOM,
        trace_id=generate_id(),
        data=data or {},
        metadata={},
    )

    return event.id


def frames_of(stream_text: str) -> list[dict[str, Any]]:
    """Split a raw SSE response body into frames, keeping every field we assert on."""

    frames: list[dict[str, Any]] = []

    for block in stream_text.split("\n\n"):
        if not block.strip():
            continue

        frame: dict[str, Any] = {"id": None, "data": [], "comments": []}

        for line in block.splitlines():
            if line.startswith(":"):
                frame["comments"].append(line)
            elif line.startswith("id: "):
                frame["id"] = line[len("id: ") :]
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


def read_event_params(wait_for_data: int = 5, delta: bool = False) -> dict[str, Any]:
    return {"sse": "true", "wait_for_data": wait_for_data, "delta": "true" if delta else "false"}


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


def test_that_the_retry_frame_announces_the_reconnect_delay() -> None:
    assert format_sse_retry_frame().startswith("retry: ")
    assert SSE_HEARTBEAT_FRAME.startswith(":")


def test_that_a_delta_describes_only_the_chunks_it_carries() -> None:
    delta = json.loads(
        ChunkDelta(
            event_id="evt_1",
            offset=4,
            chunk_index=2,
            chunks=["new"],
            is_complete=True,
        ).to_json()
    )

    assert delta == {
        "id": "evt_1",
        "offset": 4,
        "chunk_index": 2,
        "chunks": ["new"],
        "is_complete": True,
    }


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


async def test_that_an_explicit_min_offset_beats_last_event_id(
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
            {**list_events_params(), "min_offset": 2},
            headers={"Last-Event-ID": "0"},
        )
    )

    assert [payload_of(f)["offset"] for f in frames] == [2]


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


async def test_that_an_idle_stream_is_kept_alive(
    async_client: httpx.AsyncClient,
    container: Container,
    session_id: SessionId,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(sessions_api, "SSE_HEARTBEAT_SECONDS", 0.05)

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


async def test_that_delta_frames_carry_only_the_chunks_added_since_the_previous_frame(
    async_client: httpx.AsyncClient,
    container: Container,
    session_id: SessionId,
) -> None:
    store = container[SessionStore]
    event_id = await create_event(container, session_id, data={"chunks": ["hello"]})

    async def complete_the_event() -> None:
        # The test client hands the body over only once it is complete, so the
        # second chunk has to be written while the stream is still open.
        await asyncio.sleep(0.1)

        await store.update_event(
            session_id,
            event_id,
            {"data": {"chunks": ["hello", " world", None]}},
        )

    completing = asyncio.create_task(complete_the_event())

    frames = data_frames(
        await collect_stream(
            async_client,
            f"/sessions/{session_id}/events/{event_id}",
            read_event_params(delta=True),
        )
    )

    await completing

    # The first frame is always the whole event, so a client has a base to
    # append the deltas to.
    assert [f["id"] for f in frames] == ["0:1", "0:3"]
    assert payload_of(frames[0])["data"]["chunks"] == ["hello"]
    assert payload_of(frames[1]) == {
        "id": event_id,
        "offset": 0,
        "chunk_index": 1,
        "chunks": [" world"],
        "is_complete": True,
    }


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
