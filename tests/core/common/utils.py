# Copyright 2024 Emcie Co Ltd.
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

from dataclasses import dataclass
from datetime import datetime, timezone
from lagom import Container
from typing import Optional, cast

from parlant.core.common import (
    EventId,
    EventSource,
    JSONSerializable,
    generate_id,
)
from parlant.core.customers import Customer
from parlant.core.engines.alpha.guideline_proposition import GuidelineProposition
from parlant.core.engines.types import UtteranceRequest
from parlant.core.guidelines import Guideline
from parlant.core.sessions import Event, MessageEventData
from parlant.core.tools import Tool

from tests.test_utilities import SyncAwaiter


@dataclass
class ContextOfTest:
    sync_await: SyncAwaiter
    container: Container
    events: list[Event]
    guidelines: dict[str, Guideline]
    guideline_propositions: dict[str, GuidelineProposition]
    tools: dict[str, Tool]
    actions: list[UtteranceRequest]


def create_event_message(
    offset: int,
    source: EventSource,
    message: str,
    customer: Optional[Customer] = None,
) -> Event:
    message_data: MessageEventData = {
        "message": message,
        "participant": {
            "display_name": customer.name if customer else source,
        },
    }

    event = Event(
        id=EventId(generate_id()),
        source=source,
        kind="message",
        offset=offset,
        correlation_id="test_correlation_id",
        data=cast(JSONSerializable, message_data),
        creation_utc=datetime.now(timezone.utc),
        deleted=False,
    )

    return event
