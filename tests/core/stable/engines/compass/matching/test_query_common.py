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

from datetime import datetime, timezone
from typing import Mapping, cast

from parlant.core.common import JSONSerializable, generate_id
from parlant.core.engines.compass.matching.common import render_message_query
from parlant.core.sessions import (
    Event,
    EventId,
    EventKind,
    EventSource,
    MessageEventData,
    censored_message_notice,
)


def _message_event(
    *,
    source: EventSource,
    data: MessageEventData,
    metadata: Mapping[str, JSONSerializable] = {},
) -> Event:
    now = datetime.now(timezone.utc)
    return Event(
        id=EventId(generate_id()),
        source=source,
        kind=EventKind.MESSAGE,
        creation_utc=now,
        modified_utc=now,
        offset=0,
        trace_id="t",
        data=cast(Mapping[str, JSONSerializable], dict(data)),
        metadata=metadata,
        deleted=False,
    )


def test_that_a_plain_customer_message_renders_as_a_source_tagged_line() -> None:
    event = _message_event(
        source=EventSource.CUSTOMER,
        data={"message": "hello there", "participant": {"display_name": "C"}},
    )

    assert render_message_query(event) == "user: hello there"


def test_that_a_flagged_message_renders_its_censor_notice_not_the_raw_text() -> None:
    data: MessageEventData = {
        "message": "the raw forbidden text",
        "participant": {"display_name": "C"},
        "flagged": True,
        "groups": ["hate"],
    }
    event = _message_event(source=EventSource.CUSTOMER, data=data)

    rendered = render_message_query(event)

    # The flagged text must never surface in a recall query — only the notice does.
    assert "the raw forbidden text" not in rendered
    assert rendered == f"user: {censored_message_notice(data)}"


def test_that_voice_spoken_text_overrides_the_message_text() -> None:
    event = _message_event(
        source=EventSource.AI_AGENT,
        data={"message": "the full intended reply", "participant": {"display_name": "A"}},
        metadata={"__parlant__": {"schema_version": 1, "voice": {"spoken_text": "the part heard"}}},
    )

    # Voice delivery: only what was actually heard drives recall.
    assert render_message_query(event) == "ai_agent: the part heard"
