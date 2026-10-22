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

from contextlib import AsyncExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator, Sequence

import pytest

from parlant.adapters.db.json_file import JSONFileDocumentDatabase
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.agents import AgentId
from parlant.core.customers import CustomerId
from parlant.core.loggers import CompositeLogger
from parlant.core.persistence.common import Cursor, ObjectId, SortDirection
from parlant.core.persistence.document_database import DocumentDatabase
from parlant.core.sessions import (
    Session,
    SessionDocumentStore,
    SessionId,
    SessionListing,
    SessionStore,
)


_CREATION_UTC = datetime(2026, 1, 1, tzinfo=timezone.utc)
_AGENT_ID = AgentId("pagination-agent")
_OTHER_AGENT_ID = AgentId("other-agent")
_CUSTOMER_ID = CustomerId("customer-a")
_OTHER_CUSTOMER_ID = CustomerId("customer-b")


@pytest.fixture(params=["transient", "json_file"])
async def session_store(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> AsyncIterator[SessionStore]:
    async with AsyncExitStack() as stack:
        database: DocumentDatabase
        if request.param == "json_file":
            database = await stack.enter_async_context(
                JSONFileDocumentDatabase(CompositeLogger([]), tmp_path / "sessions.json")
            )
        else:
            database = TransientDocumentDatabase()

        yield await stack.enter_async_context(SessionDocumentStore(database))


async def _create_sessions(
    store: SessionStore,
    same_timestamp: bool = False,
) -> list[Session]:
    specifications = [
        (_AGENT_ID, _CUSTOMER_ID, {"basic"}, 0),
        (_AGENT_ID, _CUSTOMER_ID, {"premium", "support"}, 1),
        (_AGENT_ID, _OTHER_CUSTOMER_ID, {"premium", "support"}, 2),
        (_AGENT_ID, _CUSTOMER_ID, {"premium", "support"}, 2),
        (_AGENT_ID, _CUSTOMER_ID, {"basic"}, 3),
        (_AGENT_ID, _CUSTOMER_ID, {"premium"}, 4),
        (_OTHER_AGENT_ID, _CUSTOMER_ID, {"premium", "support"}, 5),
    ]
    sessions = []
    for index, (agent_id, customer_id, labels, seconds) in enumerate(specifications):
        sessions.append(
            await store.create_session(
                agent_id=agent_id,
                customer_id=customer_id,
                creation_utc=_CREATION_UTC + timedelta(seconds=0 if same_timestamp else seconds),
                title=f"Session {index}",
                metadata={"sequence": index},
                labels=labels,
            )
        )
    return sessions


def _ordered(sessions: Sequence[Session], direction: SortDirection) -> list[Session]:
    # Independent ordering of known fixture members, including generated ID ties.
    return sorted(
        sessions,
        key=lambda session: (session.creation_utc.isoformat(), str(session.id)),
        reverse=direction == SortDirection.DESC,
    )


def _cursor(session: Session) -> Cursor:
    return Cursor(creation_utc=session.creation_utc.isoformat(), id=ObjectId(session.id))


def _assert_page(
    page: SessionListing,
    expected: Sequence[Session],
    total_count: int,
    has_more: bool,
) -> None:
    assert [session.id for session in page.items] == [session.id for session in expected]
    # Compare all fields, including labels, customer, agent, timestamp and metadata.
    assert list(page.items) == list(expected)
    assert page.total_count == total_count
    assert page.has_more is has_more
    assert page.next_cursor == (_cursor(expected[-1]) if has_more else None)


async def test_that_labelled_sessions_beyond_the_first_raw_page_are_returned(
    session_store: SessionStore,
) -> None:
    sessions = await _create_sessions(session_store)

    page = await session_store.list_sessions(agent_id=_AGENT_ID, labels={"premium"}, limit=1)

    _assert_page(page, [sessions[1]], total_count=4, has_more=True)


@pytest.mark.parametrize("direction", [SortDirection.ASC, SortDirection.DESC])
@pytest.mark.parametrize("same_timestamp", [False, True], ids=["mixed-times", "id-ties"])
async def test_that_labelled_session_pages_report_a_cursor_and_complete_iteration(
    session_store: SessionStore,
    direction: SortDirection,
    same_timestamp: bool,
) -> None:
    sessions = await _create_sessions(session_store, same_timestamp=same_timestamp)
    expected = _ordered([sessions[1], sessions[2], sessions[3], sessions[5]], direction)
    cursor: Cursor | None = None
    seen: list[SessionId] = []

    for page_index, expected_session in enumerate(expected):
        page = await session_store.list_sessions(
            agent_id=_AGENT_ID,
            labels={"premium"},
            limit=1,
            cursor=cursor,
            sort_direction=direction,
        )
        # Real backends count matching documents remaining AFTER the cursor.
        _assert_page(
            page,
            [expected_session],
            total_count=4 - page_index,
            has_more=page_index < 3,
        )
        seen.extend(session.id for session in page.items)
        cursor = page.next_cursor

    assert seen == [session.id for session in expected]
    assert cursor is None


@pytest.mark.parametrize("direction", [SortDirection.ASC, SortDirection.DESC])
async def test_that_multiple_labels_customer_filter_and_descending_order_compose(
    session_store: SessionStore,
    direction: SortDirection,
) -> None:
    sessions = await _create_sessions(session_store)
    expected = _ordered([sessions[1], sessions[3]], direction)

    first = await session_store.list_sessions(
        agent_id=_AGENT_ID,
        customer_id=_CUSTOMER_ID,
        labels={"premium", "support"},
        limit=1,
        sort_direction=direction,
    )
    _assert_page(first, expected[:1], total_count=2, has_more=True)

    final = await session_store.list_sessions(
        agent_id=_AGENT_ID,
        customer_id=_CUSTOMER_ID,
        labels={"premium", "support"},
        limit=1,
        cursor=first.next_cursor,
        sort_direction=direction,
    )
    _assert_page(final, expected[1:], total_count=1, has_more=False)


@pytest.mark.parametrize(
    "control",
    [
        "unlabelled",
        "empty_labels",
        "unlabelled_unlimited",
        "empty_labels_unlimited",
        "labelled_unlimited",
        "multi_label_unlimited",
        "absent_label",
        "after_last_match",
    ],
)
async def test_that_unlabelled_and_unlimited_listing_preserve_existing_behavior(
    session_store: SessionStore,
    control: str,
) -> None:
    sessions = await _create_sessions(session_store)
    labels: set[str] | None = None
    limit: int | None = 2
    cursor: Cursor | None = None
    has_more = False

    if control in {"unlabelled", "empty_labels"}:
        labels = set() if control == "empty_labels" else None
        cursor = _cursor(sessions[0])
        expected = _ordered(sessions[1:6], SortDirection.ASC)[:2]
        total_count = 5
        has_more = True
    elif control in {"unlabelled_unlimited", "empty_labels_unlimited"}:
        labels = set() if control == "empty_labels_unlimited" else None
        limit = None
        expected = _ordered(sessions[:6], SortDirection.ASC)
        total_count = 6
    elif control == "labelled_unlimited":
        labels = {"premium"}
        limit = None
        cursor = _cursor(sessions[1])
        expected = _ordered([sessions[2], sessions[3], sessions[5]], SortDirection.ASC)
        total_count = 3
    elif control == "multi_label_unlimited":
        labels = {"premium", "support"}
        limit = None
        cursor = _cursor(sessions[1])
        expected = _ordered([sessions[2], sessions[3]], SortDirection.ASC)
        total_count = 2
    elif control == "absent_label":
        labels = {"absent"}
        expected = []
        total_count = 0
    else:
        assert control == "after_last_match"
        labels = {"premium"}
        cursor = _cursor(sessions[5])
        expected = []
        total_count = 0

    page = await session_store.list_sessions(
        agent_id=_AGENT_ID,
        labels=labels,
        limit=limit,
        cursor=cursor,
    )

    _assert_page(page, expected, total_count=total_count, has_more=has_more)
