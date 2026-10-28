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

import pytest

from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.agents import AgentId
from parlant.core.common import ItemNotFoundError
from parlant.core.customers import CustomerId
from parlant.core.sessions import EventKind, EventSource, SessionDocumentStore


@pytest.mark.asyncio
async def test_that_reading_an_event_through_a_foreign_session_raises_item_not_found() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session_a = await store.create_session(CustomerId("customer-a"), AgentId("agent"))
        session_b = await store.create_session(CustomerId("customer-b"), AgentId("agent"))

        event_in_b = await store.create_event(
            session_id=session_b.id,
            source=EventSource.CUSTOMER,
            kind=EventKind.MESSAGE,
            trace_id="trace-b",
            data={"message": "b's private message"},
        )

        # Session A's owner must not be able to read session B's event by id,
        # even though they legitimately hold session A.
        with pytest.raises(ItemNotFoundError):
            await store.read_event(session_id=session_a.id, event_id=event_in_b.id)


@pytest.mark.asyncio
async def test_that_reading_an_event_through_its_own_session_succeeds() -> None:
    async with SessionDocumentStore(TransientDocumentDatabase()) as store:
        session = await store.create_session(CustomerId("customer"), AgentId("agent"))

        event = await store.create_event(
            session_id=session.id,
            source=EventSource.CUSTOMER,
            kind=EventKind.MESSAGE,
            trace_id="trace",
            data={"message": "hello"},
        )

        read = await store.read_event(session_id=session.id, event_id=event.id)

        assert read.id == event.id
