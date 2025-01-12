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

from typing import Mapping, Optional, Sequence
from typing_extensions import TypedDict

from parlant.core.common import (
    AgentId,
    ConsumerId,
    ContextVariable,
    CustomerId,
    EventKind,
    EventSource,
    GuidelineProposition,
    JSONSerializable,
    SessionId,
    SessionMode,
    Term,
)

from parlant.core.persistence.common import ObjectId
from parlant.core.common import ToolCall


class _SessionDocument_v1(TypedDict, total=False):
    id: ObjectId
    creation_utc: str
    customer_id: CustomerId
    agent_id: AgentId
    mode: SessionMode
    title: Optional[str]
    consumption_offsets: Mapping[ConsumerId, int]


class _EventDocument_v1(TypedDict, total=False):
    id: ObjectId
    creation_utc: str
    session_id: SessionId
    source: EventSource
    kind: EventKind
    offset: int
    correlation_id: str
    data: JSONSerializable
    deleted: bool


class _UsageInfoDocument_v1(TypedDict):
    input_tokens: int
    output_tokens: int
    extra: Optional[Mapping[str, int]]


class _GenerationInfoDocument_v1(TypedDict):
    schema_name: str
    model: str
    duration: float
    usage: _UsageInfoDocument_v1


class _GuidelinePropositionInspectionDocument_v1(TypedDict):
    total_duration: float
    batches: Sequence[_GenerationInfoDocument_v1]


class _PreparationIterationGenerationsDocument_v1(TypedDict):
    guideline_proposition: _GuidelinePropositionInspectionDocument_v1
    tool_calls: Sequence[_GenerationInfoDocument_v1]


class _MessageGenerationInspectionDocument_v1(TypedDict):
    generation: _GenerationInfoDocument_v1
    messages: Sequence[Optional[str]]


class _PreparationIterationDocument_v1(TypedDict):
    guideline_propositions: Sequence[GuidelineProposition]
    tool_calls: Sequence[ToolCall]
    terms: Sequence[Term]
    context_variables: Sequence[ContextVariable]
    generations: _PreparationIterationGenerationsDocument_v1


class _InspectionDocument_v1(TypedDict, total=False):
    id: ObjectId
    session_id: SessionId
    correlation_id: str
    message_generations: Sequence[_MessageGenerationInspectionDocument_v1]
    preparation_iterations: Sequence[_PreparationIterationDocument_v1]
