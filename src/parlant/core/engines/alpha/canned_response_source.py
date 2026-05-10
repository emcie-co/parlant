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

"""Source attribution for canned responses surfaced to the engine.

Each candidate canned response can be traced back to one or more
"triggers" — the agent itself, an agent tag, the global pool, a journey,
a journey node, a specific guideline, or a tool that produced a
transient response. Carrying this information alongside the candidate
list lets the engine record *why* a particular canned response was
available and, ultimately, why it was selected.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, NamedTuple, Sequence

from parlant.core.canned_responses import CannedResponse, CannedResponseId


class CannedResponseSourceKind(str, Enum):
    AGENT = "agent"
    AGENT_TAG = "agent_tag"
    GLOBAL = "global"
    JOURNEY = "journey"
    JOURNEY_NODE = "journey_node"
    GUIDELINE = "guideline"
    TOOL = "tool"


class CannedResponseSource(NamedTuple):
    """A trigger that surfaced a canned response into the engine's pool."""

    kind: CannedResponseSourceKind
    id: str


GLOBAL_CANNED_RESPONSE_SOURCE = CannedResponseSource(
    kind=CannedResponseSourceKind.GLOBAL,
    id="global",
)


@dataclass(frozen=True)
class CannedResponseLookup:
    """Result of resolving canned responses for a request, with attribution.

    A canned response can be surfaced by more than one trigger
    (e.g. tagged for the agent AND for a specific guideline); ``sources``
    holds every trigger that contributed it.
    """

    canned_responses: Sequence[CannedResponse]
    sources: Mapping[CannedResponseId, Sequence[CannedResponseSource]] = field(default_factory=dict)
