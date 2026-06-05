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

from parlant.core.engines.engine_context import EngineContext
from parlant.core.engines.guideline_matcher_registry import GuidelineMatcherRegistry
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.guidelines import Guideline, GuidelineId


async def _noop_matcher(context: EngineContext, guideline: Guideline) -> GuidelineMatch | None:
    return None


def test_that_a_registered_matcher_can_be_retrieved_by_guideline_id() -> None:
    registry = GuidelineMatcherRegistry()
    guideline_id = GuidelineId("g-1")

    assert registry.get(guideline_id) is None

    registry.register(guideline_id, _noop_matcher)

    assert registry.get(guideline_id) is _noop_matcher
    assert dict(registry.matchers) == {guideline_id: _noop_matcher}


def test_that_a_guideline_without_a_registered_matcher_returns_none() -> None:
    registry = GuidelineMatcherRegistry()
    registry.register(GuidelineId("g-1"), _noop_matcher)

    assert registry.get(GuidelineId("g-2")) is None
