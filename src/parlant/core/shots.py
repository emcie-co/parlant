# Copyright 2025 Emcie Co Ltd.
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
from typing import TypeVar, Generic, Sequence, cast

from parlant.core.common import generate_id, JSONSerializable
from parlant.core.sessions import (
    Event,
    EventId,
    EventKind,
    EventSource,
    MessageEventData,
    ToolEventData,
)


@dataclass
class Shot:
    description: str
    """An explanation of what makes this shot interesting"""

    @staticmethod
    def message_event(source: EventSource, data: MessageEventData) -> Event:
        return Event(
            id=EventId(generate_id()),
            source=source,
            kind=EventKind.MESSAGE,
            creation_utc=datetime.now(timezone.utc),  # unused in shots
            offset=0,  # unused in shots
            correlation_id="<unused>",  # unused in shots
            data=cast(JSONSerializable, data),
            deleted=False,
        )

    @staticmethod
    def tool_event(data: ToolEventData) -> Event:  # noqa: F821
        return Event(
            id=EventId(generate_id()),
            source=EventSource.SYSTEM,
            kind=EventKind.TOOL,
            creation_utc=datetime.now(timezone.utc),  # unused in shots
            offset=0,  # unused in shots
            correlation_id="<unused>",  # unused in shots
            data=cast(JSONSerializable, data),
            deleted=False,
        )


TShot = TypeVar("TShot", bound=Shot)


class ShotCollection(Generic[TShot]):
    def __init__(self, initial_shots: Sequence[TShot]) -> None:
        self._shots: list[TShot] = list(initial_shots)

    async def append(
        self,
        shot: TShot,
    ) -> None:
        self._shots.append(shot)

    async def insert(
        self,
        shot: TShot,
        index: int = 0,
    ) -> None:
        self._shots.insert(index, shot)

    async def list(self) -> Sequence[TShot]:
        return self._shots

    async def remove(
        self,
        shot: TShot,
    ) -> None:
        self._shots.remove(shot)

    async def clear(self) -> None:
        self._shots.clear()
