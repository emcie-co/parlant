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

from pytest import raises

from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.engine_context import EngineContext
from parlant.core.engines.guideline_matcher_registry import GuidelineMatcherRegistry
from parlant.core.engines.compass.guideline_matching.guideline_function_matcher import (
    GuidelineFunctionMatcher,
)
from parlant.core.engines.compass.response_state import ResponseState
from parlant.core.guidelines import Guideline
from parlant.core.loggers import StdoutLogger
from parlant.core.sessions import EventSource
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.guideline_matching.utils import (
    create_engine_context,
    create_guideline,
)


def _make_matcher(registry: GuidelineMatcherRegistry) -> GuidelineFunctionMatcher:
    tracer = LocalTracer()
    return GuidelineFunctionMatcher(StdoutLogger(tracer), tracer, registry)


def _context() -> EngineContext:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    return context


async def test_that_a_guideline_with_a_matching_code_matcher_is_returned_as_matched() -> None:
    registry = GuidelineMatcherRegistry()
    guideline = create_guideline(condition="customer says hi", action="greet back")

    async def matcher(context: EngineContext, g: Guideline) -> GuidelineMatch | None:
        return GuidelineMatch(guideline=g, rationale="matched by code")

    registry.register(guideline.id, matcher)

    matches = await _make_matcher(registry).match(_context(), [guideline])

    assert [m.guideline.id for m in matches] == [guideline.id]


async def test_that_a_guideline_whose_code_matcher_returns_none_is_not_matched() -> None:
    registry = GuidelineMatcherRegistry()
    guideline = create_guideline(condition="customer says hi", action="greet back")

    async def matcher(context: EngineContext, g: Guideline) -> GuidelineMatch | None:
        return None

    registry.register(guideline.id, matcher)

    matches = await _make_matcher(registry).match(_context(), [guideline])

    assert list(matches) == []


async def test_that_an_error_in_a_code_matcher_is_handled_and_yields_no_match() -> None:
    registry = GuidelineMatcherRegistry()
    guideline = create_guideline(condition="customer says hi", action="greet back")

    async def matcher(context: EngineContext, g: Guideline) -> GuidelineMatch | None:
        raise RuntimeError("boom")

    registry.register(guideline.id, matcher)

    matches = await _make_matcher(registry).match(_context(), [guideline])

    assert list(matches) == []


async def test_that_passing_a_guideline_without_a_code_matcher_raises() -> None:
    registry = GuidelineMatcherRegistry()
    guideline = create_guideline(condition="customer says hi", action="greet back")

    # Not registered — the caller (engine) is responsible for passing only
    # function-attached guidelines, so this is a programming error.
    with raises(ValueError):
        await _make_matcher(registry).match(_context(), [guideline])
