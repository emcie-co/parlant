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

"""Typed accounting handoff and cleanup configuration, independent of transcripts."""

from dataclasses import dataclass
from datetime import datetime
import os
from typing import Literal, Mapping, Sequence

from parlant.core.common import JSONSerializable
from parlant.core.persistence.common import Cursor
from parlant.core.persistence.document_database import BaseDocument


SessionLifecycle = Literal["provisional", "established"]
ObservabilityKind = Literal["usage", "session_deleted"]


def _boolean_setting(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false; restart the server after changing it")
    return normalized == "true"


@dataclass(frozen=True)
class SessionCleanupSettings:
    enabled: bool
    include_greetings: bool

    @classmethod
    def from_environment(cls) -> "SessionCleanupSettings":
        return cls(
            enabled=_boolean_setting("PARLANT_EMPTY_SESSION_CLEANUP_ENABLED", True),
            include_greetings=_boolean_setting(
                "PARLANT_EMPTY_SESSION_CLEANUP_INCLUDE_GREETINGS", False
            ),
        )


@dataclass(frozen=True)
class SessionObservabilityRecord:
    id: str
    revision: int
    kind: ObservabilityKind
    session_id: str
    agent_id: str
    trace_id: str | None
    occurred_at: datetime
    session_creation_utc: datetime
    is_simulation: bool
    customer_represented_by_an_agent: bool
    payload: Mapping[str, JSONSerializable]


@dataclass(frozen=True)
class SessionObservabilityPage:
    records: Sequence[SessionObservabilityRecord]
    has_more: bool = False
    next_cursor: Cursor | None = None


class SessionObservabilityDocument(BaseDocument, total=False):
    revision: int
    kind: ObservabilityKind
    session_id: str
    agent_id: str
    trace_id: str | None
    occurred_at: str
    session_creation_utc: str
    is_simulation: bool
    customer_represented_by_an_agent: bool
    payload: Mapping[str, JSONSerializable]
    fingerprint: str
    pending: bool


def observability_limit(value: object = 100) -> int:
    if type(value) is not int or not 1 <= value <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    return value
