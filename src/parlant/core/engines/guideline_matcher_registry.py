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

from collections.abc import Awaitable, Callable, Mapping

from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.engine_context import EngineContext
from parlant.core.guidelines import Guideline, GuidelineId

# A code (Python) matcher for a single guideline. It receives the
# engine-agnostic EngineContext (its concrete state is hidden — EngineContext[Any])
# and the guideline, and returns a GuidelineMatch if it matches, or None.
GuidelineCodeMatcher = Callable[[EngineContext, Guideline], Awaitable[GuidelineMatch | None]]


class GuidelineMatcherRegistry:
    """Engine-agnostic registry of per-guideline code matchers.

    The single source of truth for matchers registered via the SDK
    (``create_guideline(matcher=...)``), so the SDK doesn't depend on any
    engine's matching internals. Each engine consumes the same registry: the
    alpha engine wraps a hit in a ``CustomGuidelineMatchingStrategy``; the compass
    engine runs the matcher directly.
    """

    def __init__(self) -> None:
        self._matchers: dict[GuidelineId, GuidelineCodeMatcher] = {}

    def register(self, guideline_id: GuidelineId, matcher: GuidelineCodeMatcher) -> None:
        self._matchers[guideline_id] = matcher

    def get(self, guideline_id: GuidelineId) -> GuidelineCodeMatcher | None:
        return self._matchers.get(guideline_id)

    @property
    def matchers(self) -> Mapping[GuidelineId, GuidelineCodeMatcher]:
        return self._matchers
