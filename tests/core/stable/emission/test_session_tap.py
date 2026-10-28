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
import threading
from datetime import datetime, timezone

import pytest
from parlant.core.container import Container

from parlant.core.agents import Agent, AgentId
from parlant.core.customers import CustomerId
from parlant.core.emission.event_publisher import EventPublisher
from parlant.core.emission.session_tap import (
    SessionEventTap,
    TapEventKind,
    TapItem,
)
from parlant.core.sessions import (
    EventKind,
    MessageEventData,
    SessionId,
    SessionStore,
    StatusEventData,
)


def _item(
    session_id: str,
    *,
    trace_id: str = "trace-1",
    kind: TapEventKind = TapEventKind.SPEECH_DELTA,
) -> TapItem:
    return TapItem(
        session_id=SessionId(session_id),
        trace_id=trace_id,
        kind=kind,
        event_id=None,
        event_kind=None,
        data={"text": "hello"},
    )


def test_that_publishing_with_no_subscribers_is_a_noop() -> None:
    tap = SessionEventTap()

    tap.publish(_item("session-1"))


@pytest.mark.asyncio
async def test_that_a_subscriber_receives_items_for_its_session_only() -> None:
    tap = SessionEventTap()
    subscription = tap.subscribe(SessionId("session-1"))

    tap.publish(_item("session-1"))
    tap.publish(_item("session-2"))

    item = await asyncio.wait_for(subscription.get(), timeout=1.0)
    assert item.session_id == SessionId("session-1")

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(subscription.get(), timeout=0.05)


@pytest.mark.asyncio
async def test_that_a_closed_subscription_stops_receiving() -> None:
    tap = SessionEventTap()
    subscription = tap.subscribe(SessionId("session-1"))

    subscription.close()
    tap.publish(_item("session-1"))

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(subscription.get(), timeout=0.05)


@pytest.mark.asyncio
async def test_that_items_published_from_another_thread_and_loop_are_delivered() -> None:
    # Voice jobs run on their own threads/event loops; the publisher (engine, main
    # loop) must deliver into the subscriber's loop safely.
    tap = SessionEventTap()
    subscription = tap.subscribe(SessionId("session-1"))

    def publish_from_thread() -> None:
        tap.publish(_item("session-1", trace_id="cross-thread"))

    thread = threading.Thread(target=publish_from_thread)
    thread.start()
    thread.join()

    item = await asyncio.wait_for(subscription.get(), timeout=1.0)
    assert item.trace_id == "cross-thread"


@pytest.mark.asyncio
async def test_that_the_event_publisher_tees_persisted_events_to_the_tap(
    container: Container,
) -> None:
    session_store = container[SessionStore]
    now = datetime.now(timezone.utc)
    agent = Agent(
        engine="compass",
        id=AgentId("test-agent-id"),
        name="test-agent",
        instructions=None,
        creation_utc=now,
        modified_utc=now,
        max_steps=1,
        groups=[],
    )
    session = await session_store.create_session(
        customer_id=CustomerId("customer-1"),
        agent_id=agent.id,
    )

    tap = SessionEventTap()
    subscription = tap.subscribe(session.id)

    publisher = EventPublisher(agent, session_store, session.id, tap=tap)

    handle = await publisher.emit_message_event(
        trace_id="trace-1",
        data=MessageEventData(
            message="Hello there.",
            participant={"id": agent.id, "display_name": agent.name},
        ),
    )

    created = await asyncio.wait_for(subscription.get(), timeout=1.0)
    assert created.kind == TapEventKind.EVENT_PERSISTED
    assert created.event_kind == EventKind.MESSAGE
    assert created.trace_id == "trace-1"
    assert created.event_id is not None

    await handle.update(
        MessageEventData(
            message="Hello there. More text.",
            participant={"id": agent.id, "display_name": agent.name},
        )
    )

    updated = await asyncio.wait_for(subscription.get(), timeout=1.0)
    assert updated.kind == TapEventKind.EVENT_UPDATED
    assert updated.event_id == created.event_id

    await publisher.emit_status_event(
        trace_id="trace-1",
        data=StatusEventData(status="ready"),
    )

    status = await asyncio.wait_for(subscription.get(), timeout=1.0)
    assert status.kind == TapEventKind.EVENT_PERSISTED
    assert status.event_kind == EventKind.STATUS
