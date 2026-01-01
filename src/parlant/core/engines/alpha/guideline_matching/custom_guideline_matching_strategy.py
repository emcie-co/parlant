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

import asyncio
from typing import Awaitable, Callable, Sequence
from typing_extensions import override

from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingBatch,
    GuidelineMatchingBatchResult,
    GuidelineMatchingStrategy,
    ResponseAnalysisBatch,
    ResponseAnalysisContext,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matching_context import (
    GuidelineMatchingContext,
)
from parlant.core.guidelines import Guideline
from parlant.core.loggers import Logger
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo


class CustomGuidelineMatchingBatch(GuidelineMatchingBatch):
    def __init__(
        self,
        guideline: Guideline,
        context: GuidelineMatchingContext,
        matcher: Callable[[GuidelineMatchingContext, Guideline], Awaitable[GuidelineMatch]],
        logger: Logger,
    ) -> None:
        self._guideline = guideline
        self._context = context
        self._matcher = matcher
        self._logger = logger

    @override
    async def process(self) -> GuidelineMatchingBatchResult:
        t_start = asyncio.get_event_loop().time()

        match: GuidelineMatch | None = None

        try:
            match = await self._matcher(self._context, self._guideline)
        except Exception as e:
            self._logger.error(f"Error in custom matcher: {e}")

        t_end = asyncio.get_event_loop().time()

        return GuidelineMatchingBatchResult(
            matches=[match] if match and match.score == 10 else [],
            generation_info=GenerationInfo(
                schema_name="custom_matcher",
                model="python",
                duration=t_end - t_start,
                usage=UsageInfo(
                    input_tokens=0,
                    output_tokens=0,
                    extra={},
                ),
            ),
        )

    @property
    @override
    def size(self) -> int:
        return 1


class CustomGuidelineMatchingStrategy(GuidelineMatchingStrategy):
    """A guideline matching strategy that uses a custom matcher function."""

    def __init__(
        self,
        guideline: Guideline,
        matcher: Callable[[GuidelineMatchingContext, Guideline], Awaitable[GuidelineMatch]],
        logger: Logger,
    ) -> None:
        self._guideline = guideline
        self._matcher = matcher
        self._logger = logger

    @override
    async def create_matching_batches(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> Sequence[GuidelineMatchingBatch]:
        # Only create a batch if our specific guideline is in the list (check by ID)
        guideline_ids = {g.id for g in guidelines}

        if self._guideline.id in guideline_ids:
            return [
                CustomGuidelineMatchingBatch(
                    guideline=self._guideline,
                    context=context,
                    matcher=self._matcher,
                    logger=self._logger,
                )
            ]
        return []

    @override
    async def create_response_analysis_batches(
        self,
        guideline_matches: Sequence[GuidelineMatch],
        context: ResponseAnalysisContext,
    ) -> Sequence[ResponseAnalysisBatch]:
        # Custom matchers don't need response analysis
        return []

    @override
    async def transform_matches(
        self,
        matches: Sequence[GuidelineMatch],
    ) -> Sequence[GuidelineMatch]:
        # Pass through without transformation
        return matches
