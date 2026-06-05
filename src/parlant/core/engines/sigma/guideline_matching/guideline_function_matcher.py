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

from parlant.core.async_utils import safe_gather
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.guideline_matcher_registry import (
    GuidelineCodeMatcher,
    GuidelineMatcherRegistry,
)
from parlant.core.engines.sigma.response_state import EngineContext
from parlant.core.guidelines import Guideline
from parlant.core.loggers import Logger
from parlant.core.tracer import Tracer


class GuidelineFunctionMatcher:
    """Runs the code (Python) matchers of guidelines that have one registered.

    This is the sigma counterpart to the alpha engine's
    CustomGuidelineMatchingStrategy: it reads the same engine-agnostic
    GuidelineMatcherRegistry and runs each matcher with the current EngineContext.

    Every guideline passed in is expected to be function-attached — the caller
    (the engine) selects them from the registry. A guideline without a registered
    matcher is a programming error and raises.
    """

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        matcher_registry: GuidelineMatcherRegistry,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._matcher_registry = matcher_registry

    async def match(
        self,
        context: EngineContext,
        guidelines: Sequence[Guideline],
    ) -> Sequence[GuidelineMatch]:
        if not guidelines:
            return []

        with self._tracer.span("guideline.function_match"):
            # All guidelines must be function-attached; resolve each matcher up
            # front so a non-attached guideline fails loudly rather than silently.
            matchers = [(g, self._resolve_matcher(g)) for g in guidelines]

            # Code matchers are independent; run them concurrently.
            results = await safe_gather(
                *(self._run_matcher(context, g, matcher) for g, matcher in matchers)
            )

            return [match for match in results if match is not None]

    def _resolve_matcher(self, guideline: Guideline) -> GuidelineCodeMatcher:
        matcher = self._matcher_registry.get(guideline.id)

        if matcher is None:
            raise ValueError(
                f"Guideline '{guideline.id}' has no registered code matcher; "
                "only function-attached guidelines may be passed to the function matcher"
            )

        return matcher

    async def _run_matcher(
        self,
        context: EngineContext,
        guideline: Guideline,
        matcher: GuidelineCodeMatcher,
    ) -> GuidelineMatch | None:
        try:
            return await matcher(context, guideline)
        except Exception as e:
            self._logger.error(f"Error in code matcher for guideline {guideline.id}: {e}")
            return None
