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

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional, cast

from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingBatch,
    ResponseAnalysisBatch,
)
from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.journeys import JourneyEdgeId, JourneyNodeId
from parlant.core.meter import DurationHistogram, Meter


@dataclass
class GuidelineInternalRepresentation:
    condition: str
    action: Optional[str]


def internal_representation(g: Guideline) -> GuidelineInternalRepresentation:
    action, condition = g.content.action, g.content.condition

    if agent_intention_condition := g.metadata.get("agent_intention_condition"):
        condition = cast(str, agent_intention_condition) or condition

    if internal_action := g.metadata.get("internal_action"):
        action = cast(str, internal_action) or action

    return GuidelineInternalRepresentation(condition, action)


def format_journey_node_guideline_id(
    node_id: JourneyNodeId,
    edge_id: Optional[JourneyEdgeId] = None,
) -> GuidelineId:
    if edge_id:
        return GuidelineId(f"journey_node:{node_id}:{edge_id}")

    return GuidelineId(f"journey_node:{node_id}")


_MATCHING_BATCH_DURATION_HISTOGRAM: DurationHistogram | None = None
_ANALYSIS_BATCH_DURATION_HISTOGRAM: DurationHistogram | None = None


@asynccontextmanager
async def measure_guideline_matching_batch(
    meter: Meter,
    batch: GuidelineMatchingBatch,
) -> AsyncIterator[None]:
    global _MATCHING_BATCH_DURATION_HISTOGRAM
    if _MATCHING_BATCH_DURATION_HISTOGRAM is None:
        _MATCHING_BATCH_DURATION_HISTOGRAM = meter.create_duration_histogram(
            name="gm.batch",
            description="Duration of guideline matching batch",
        )

    async with _MATCHING_BATCH_DURATION_HISTOGRAM.measure(
        attributes={
            "batch.name": batch.__class__.__name__,
            "batch.size": str(batch.size),
        }
    ):
        yield


@asynccontextmanager
async def measure_response_analysis_batch(
    meter: Meter,
    batch: ResponseAnalysisBatch,
) -> AsyncIterator[None]:
    global _ANALYSIS_BATCH_DURATION_HISTOGRAM
    if _ANALYSIS_BATCH_DURATION_HISTOGRAM is None:
        _ANALYSIS_BATCH_DURATION_HISTOGRAM = meter.create_duration_histogram(
            name="rn.batch",
            description="Duration of guideline matching batch",
        )

    async with _ANALYSIS_BATCH_DURATION_HISTOGRAM.measure(
        attributes={
            "batch.name": batch.__class__.__name__,
            "batch.size": str(batch.size),
        }
    ):
        yield
