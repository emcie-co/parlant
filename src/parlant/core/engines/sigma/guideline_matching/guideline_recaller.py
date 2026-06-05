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

from collections.abc import Sequence
from dataclasses import dataclass

from parlant.core.engines.sigma.response_state import EngineContext
from parlant.core.guidelines import Guideline, GuidelineStore
from parlant.core.tracer import Tracer


@dataclass(frozen=True)
class RecalledGuideline:
    guideline: Guideline
    is_relevant: bool
    score: float


@dataclass(frozen=True)
class GuidelineRecallResult:
    recalled_guidelines: Sequence[RecalledGuideline]


class GuidelineRecaller:
    DEFAULT_MAX_COUNT = 10

    def __init__(
        self,
        guideline_store: GuidelineStore,
        tracer: Tracer,
    ) -> None:
        self._guideline_store = guideline_store
        self._tracer = tracer

    async def recall(
        self,
        context: EngineContext,
        guidelines: Sequence[Guideline],
        max_count: int = DEFAULT_MAX_COUNT,
    ) -> GuidelineRecallResult:
        with self._tracer.span("guideline.recall"):
            return await self._do_recall(context, guidelines, max_count)

    async def _do_recall(
        self,
        context: EngineContext,
        guidelines: Sequence[Guideline],
        max_count: int = DEFAULT_MAX_COUNT,
    ) -> GuidelineRecallResult:
        if not guidelines:
            return GuidelineRecallResult([])

        query = self._build_query(context)

        if not query:
            return GuidelineRecallResult([])

        results = await self._guideline_store.find_relevant_guidelines(
            query=query,
            available_guidelines=guidelines,
            max_count=max_count,
        )

        return GuidelineRecallResult(
            [
                RecalledGuideline(guideline=result.guideline, is_relevant=True, score=result.score)
                for result in results
            ]
        )

    def _build_query(self, context: EngineContext) -> str:
        # Mirror the alpha engine's retrieval query: build it from the whole
        # interaction rather than just the last message, so the embedding search
        # reflects the full conversational context.
        if not context.interaction.events:
            return ""

        return str([f"{m.source}: {m.content}\n\n" for m in context.interaction.messages])
