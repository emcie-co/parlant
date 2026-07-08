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

from parlant.core.engines.rule_match import RuleMatch
from parlant.core.engines.engine_context import EngineContext
from parlant.core.rules import Rule, RuleId

# A code (Python) matcher for a single rule. It receives the
# engine-agnostic EngineContext (its concrete state is hidden — EngineContext[Any])
# and the rule, and returns a RuleMatch if it matches, or None.
RuleCodeMatcher = Callable[[EngineContext, Rule], Awaitable[RuleMatch | None]]


class RuleMatcherRegistry:
    """Engine-agnostic registry of per-rule code matchers.

    The single source of truth for matchers registered via the SDK
    (``create_rule(matcher=...)``), so the SDK doesn't depend on any
    engine's matching internals. Each engine consumes the same registry: the
    alpha engine wraps a hit in a ``CustomRuleMatchingStrategy``; the compass
    engine runs the matcher directly.
    """

    def __init__(self) -> None:
        self._matchers: dict[RuleId, RuleCodeMatcher] = {}

    def register(self, rule_id: RuleId, matcher: RuleCodeMatcher) -> None:
        self._matchers[rule_id] = matcher

    def get(self, rule_id: RuleId) -> RuleCodeMatcher | None:
        return self._matchers.get(rule_id)

    @property
    def matchers(self) -> Mapping[RuleId, RuleCodeMatcher]:
        return self._matchers
