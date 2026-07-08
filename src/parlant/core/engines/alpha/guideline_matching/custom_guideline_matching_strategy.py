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
import json
from typing import Awaitable, Callable, Sequence
from typing_extensions import override

from parlant.core.engines.entity_context import EntityContext
from parlant.core.engines.rule_match import RuleMatch
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
from parlant.core.engines.engine_context import EngineContext
from parlant.core.rules import Rule as Guideline
from parlant.core.loggers import Logger
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo


DEFAULT_SKIPPED_RATIONALE = "Skipped by custom matcher"

# A code matcher receives the engine-agnostic EngineContext (its concrete state
# is hidden — EngineContext[Any]) so the same matcher runs under any engine.
CodeMatcher = Callable[[EngineContext, Guideline], Awaitable[GuidelineMatch | RuleMatch | None]]


class CustomGuidelineMatchingBatch(GuidelineMatchingBatch):
    def __init__(
        self,
        guideline: Guideline,
        matcher: CodeMatcher,
        logger: Logger,
    ) -> None:
        self._guideline = guideline
        self._matcher = matcher
        self._logger = logger

    @override
    async def process(self) -> GuidelineMatchingBatchResult:
        t_start = asyncio.get_event_loop().time()

        match: GuidelineMatch | None = None

        # The matcher takes the active EngineContext (set in the engine's
        # _load_context before matching runs), rather than the engine-specific
        # GuidelineMatchingContext, so it's engine-agnostic.
        engine_context = EntityContext.get()

        if engine_context is None:
            self._logger.error("Custom matcher invoked without an active engine context; skipping")
        else:
            try:
                candidate = await self._matcher(engine_context, self._guideline)
                if isinstance(candidate, RuleMatch):
                    match = GuidelineMatch(
                        guideline=candidate.rule,
                        rationale=candidate.rationale,
                        metadata=candidate.metadata,
                    )
                else:
                    match = candidate
            except Exception as e:
                self._logger.error(f"Error in custom matcher: {e}")

        t_end = asyncio.get_event_loop().time()

        data = json.dumps(
            {
                "guideline_id": self._guideline.id,
                "condition": self._guideline.content.condition,
                "action": self._guideline.content.action,
            },
            indent=2,
        )

        is_matched = match is not None

        if is_matched:
            self._logger.debug(f"Matched:\n{data}")
            assert match is not None
            matched_guidelines = [match]
            skipped_guidelines = []
        else:
            self._logger.debug(f"Not matched:\n{data}")
            matched_guidelines = []
            skipped_guidelines = [
                GuidelineMatch(
                    guideline=self._guideline, rationale=DEFAULT_SKIPPED_RATIONALE, metadata={}
                )
            ]

        return GuidelineMatchingBatchResult(
            matched_guidelines=matched_guidelines,
            skipped_guidelines=skipped_guidelines,
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
        matcher: CodeMatcher,
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
