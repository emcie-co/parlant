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

"""In-process push delivery of session activity.

Session events are persisted to the store and consumed by clients through
pull (long-poll / SSE). Real-time consumers — voice delivery in particular —
can't afford that roundtrip, so the tap offers a push lane: the event
publisher tees everything it persists here, and the voice loop additionally
publishes speech deltas that are never persisted at all.

The tap is cross-event-loop safe by construction: a subscription's queue
lives on the event loop that created it, and publishers (which may run on a
different loop or thread — e.g. the engine's main loop publishing to a voice
job's loop) deliver via ``call_soon_threadsafe``. Publishing with no
subscribers is a no-op, so the tap can be bound and teed unconditionally.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count

from parlant.core.common import JSONSerializable
from parlant.core.sessions import EventId, EventKind, SessionId


class TapEventKind(Enum):
    EVENT_PERSISTED = auto()
    """An event was created in the session store."""

    EVENT_UPDATED = auto()
    """A persisted event's data was updated (e.g. a growing message)."""

    SPEECH_DELTA = auto()
    """A speakable text segment from the voice loop; never persisted.
    ``data`` is ``{"text": <segment>}``."""


@dataclass(frozen=True)
class TapItem:
    session_id: SessionId
    trace_id: str
    kind: TapEventKind
    event_id: EventId | None
    """The persisted event's id, or None for SPEECH_DELTA items."""
    event_kind: EventKind | None
    """MESSAGE/STATUS/TOOL/CUSTOM for persisted items; None for SPEECH_DELTA."""
    data: JSONSerializable


@dataclass
class _Subscriber:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[TapItem]
    session_id: SessionId
    key: int
    closed: bool = field(default=False)


class TapSubscription:
    def __init__(self, tap: SessionEventTap, subscriber: _Subscriber) -> None:
        self._tap = tap
        self._subscriber = subscriber

    async def get(self) -> TapItem:
        return await self._subscriber.queue.get()

    def close(self) -> None:
        self._tap._unsubscribe(self._subscriber)

    def __aiter__(self) -> TapSubscription:
        return self

    async def __anext__(self) -> TapItem:
        return await self.get()


class SessionEventTap:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers_by_session: dict[SessionId, list[_Subscriber]] = {}
        self._keys = count()

    def subscribe(self, session_id: SessionId) -> TapSubscription:
        """Subscribe to a session's tap items.

        Must be called from a running event loop — the subscriber's queue is
        bound to it, and publishers deliver into it thread-safely."""
        subscriber = _Subscriber(
            loop=asyncio.get_running_loop(),
            queue=asyncio.Queue(),
            session_id=session_id,
            key=next(self._keys),
        )

        with self._lock:
            self._subscribers_by_session.setdefault(session_id, []).append(subscriber)

        return TapSubscription(self, subscriber)

    def publish(self, item: TapItem) -> None:
        """Deliver an item to the session's subscribers.

        Synchronous, thread-safe, and callable from any loop; a no-op when the
        session has no subscribers."""
        with self._lock:
            subscribers = list(self._subscribers_by_session.get(item.session_id, ()))

        for subscriber in subscribers:
            if subscriber.closed:
                continue

            try:
                subscriber.loop.call_soon_threadsafe(subscriber.queue.put_nowait, item)
            except RuntimeError:
                # The subscriber's loop is gone; drop it.
                self._unsubscribe(subscriber)

    def _unsubscribe(self, subscriber: _Subscriber) -> None:
        subscriber.closed = True

        with self._lock:
            subscribers = self._subscribers_by_session.get(subscriber.session_id)
            if not subscribers:
                return

            self._subscribers_by_session[subscriber.session_id] = [
                s for s in subscribers if s.key != subscriber.key
            ]

            if not self._subscribers_by_session[subscriber.session_id]:
                del self._subscribers_by_session[subscriber.session_id]
