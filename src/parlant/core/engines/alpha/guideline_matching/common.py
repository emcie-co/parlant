from contextlib import asynccontextmanager
from typing import AsyncIterator
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingBatch,
    ResponseAnalysisBatch,
)
from parlant.core.meter import DurationHistogram, Meter


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
            name="ra.batch",
            description="Duration of guideline matching batch",
        )

    async with _ANALYSIS_BATCH_DURATION_HISTOGRAM.measure(
        attributes={
            "batch.name": batch.__class__.__name__,
            "batch.size": str(batch.size),
        }
    ):
        yield
