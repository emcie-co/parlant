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

from typing import Literal, Optional, Sequence, TypedDict, Union

from parlant.core.common import (
    AgentId,
    CoherenceCheckKind,
    ConnectionPropositionKind,
    GuidelineId,
)
from parlant.core.persistence.common import ObjectId


class _GuidelineContentDocument_v1(TypedDict):
    condition: str
    action: str


class _GuidelinePayloadDocument_v1(TypedDict):
    content: _GuidelineContentDocument_v1
    action: Literal["add", "update"]
    updated_id: Optional[GuidelineId]
    coherence_check: bool
    connection_proposition: bool


_PayloadDocument_v1 = Union[_GuidelinePayloadDocument_v1]


class _CoherenceCheckDocument_v1(TypedDict):
    kind: CoherenceCheckKind
    first: _GuidelineContentDocument_v1
    second: _GuidelineContentDocument_v1
    issue: str
    severity: int


class _ConnectionPropositionDocument_v1(TypedDict):
    check_kind: ConnectionPropositionKind
    source: _GuidelineContentDocument_v1
    target: _GuidelineContentDocument_v1


class _InvoiceGuidelineDataDocument_v1(TypedDict):
    coherence_checks: Sequence[_CoherenceCheckDocument_v1]
    connection_propositions: Optional[Sequence[_ConnectionPropositionDocument_v1]]


_InvoiceDataDocument_v1 = Union[_InvoiceGuidelineDataDocument_v1]


class _InvoiceDocument_v1(TypedDict, total=False):
    kind: str
    payload: _PayloadDocument_v1
    checksum: str
    state_version: str
    approved: bool
    data: Optional[_InvoiceDataDocument_v1]
    error: Optional[str]


class _EvaluationDocument_v1(TypedDict, total=False):
    id: ObjectId
    agent_id: AgentId
    creation_utc: str
    status: str
    error: Optional[str]
    invoices: Sequence[_InvoiceDocument_v1]
    progress: float
