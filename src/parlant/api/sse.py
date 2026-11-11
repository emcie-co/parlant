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

"""Framing helpers for the session event streams.

The streams send bare `data:` frames, so a client that loses its connection has
no way to say where it stopped and has to replay from the beginning. Everything
here is pure text handling: a frame that carries a resume id, and a cursor that
turns that id back into a position.
"""

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

SSE_MEDIA_TYPE = "text/event-stream"

# Told to the client once per stream, so a dropped connection comes back within
# seconds instead of after the browser's default retry delay.
SSE_RETRY_MILLISECONDS = 3000

# An idle stream looks dead to proxies, which tend to cut connections around the
# same delay `wait_for_data` defaults to (60 seconds). A comment frame is valid
# SSE that EventSource ignores, so it keeps the connection warm without waking
# the client up with an event it would have to filter out.
SSE_HEARTBEAT_SECONDS = 15.0
SSE_HEARTBEAT_FRAME = ": keep-alive\n\n"

SSE_HEADERS: Mapping[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Without this, nginx buffers the response and streams arrive in bursts.
    "X-Accel-Buffering": "no",
}


@dataclass(frozen=True)
class EventCursor:
    """A position a client can resume from.

    `offset` is a position in the session's event list. `chunk_count` is a
    position inside one streaming event's `chunks`, and only means anything
    together with that event's offset - which is why the id carries both.
    """

    offset: int
    chunk_count: int = 0


def parse_event_cursor(value: str | None) -> EventCursor | None:
    """Parse a `Last-Event-ID` header into a cursor.

    Accepts `"<offset>"` and `"<offset>:<chunk_count>"`. Anything else - empty,
    negative, non-numeric, more than two segments - yields None, and the caller
    starts from its default position: an id we cannot read is a client from
    before this header existed, not a reason to fail the request.
    """

    if value is None:
        return None

    segments = value.strip().split(":")
    if len(segments) > 2:
        return None

    try:
        offset = int(segments[0])
        chunk_count = int(segments[1]) if len(segments) == 2 else 0
    except ValueError:
        return None

    if offset < 0 or chunk_count < 0:
        return None

    return EventCursor(offset=offset, chunk_count=chunk_count)


def encode_event_cursor(offset: int, chunk_count: int = 0) -> str:
    """Render a cursor as the value of a frame's `id` field."""

    if chunk_count <= 0:
        return f"{offset}"

    return f"{offset}:{chunk_count}"


def format_sse_frame(payload: str, *, event_id: str | None = None) -> str:
    """Render one `data:` frame, optionally carrying a resume id.

    The id line precedes the data line, so a client that processes a frame left
    to right always knows where the payload it just received sits.
    """

    frame = "" if event_id is None else f"id: {event_id}\n"
    for line in payload.splitlines() or [""]:
        frame += f"data: {line}\n"

    return frame + "\n"


def format_sse_retry_frame() -> str:
    """Render the `retry:` frame a stream sends before any event."""

    return f"retry: {SSE_RETRY_MILLISECONDS}\n\n"


@dataclass(frozen=True)
class ChunkDelta:
    """The part of a streaming event that a delta frame carries.

    Only sent when the client asked for `delta=true`, and only from the second
    frame onwards: the first frame is always the whole event, so a client that
    reconnects mid-stream still gets a base to append the deltas to.
    """

    event_id: str
    offset: int
    chunk_index: int
    chunks: Sequence[str]
    is_complete: bool

    def to_json(self) -> str:
        """Render the delta as the JSON body of a `data:` frame."""

        return json.dumps(
            {
                "id": self.event_id,
                "offset": self.offset,
                "chunk_index": self.chunk_index,
                "chunks": list(self.chunks),
                "is_complete": self.is_complete,
            },
            ensure_ascii=False,
        )
