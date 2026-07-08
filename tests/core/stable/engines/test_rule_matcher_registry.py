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
from parlant.core.engines.rule_matcher_registry import RuleMatcherRegistry
from parlant.core.engines.rule_match import RuleMatch
from parlant.core.rules import Rule, RuleId


async def _noop_matcher(context: EngineContext, rule: Rule) -> RuleMatch | None:
    return None


def test_that_a_registered_matcher_can_be_retrieved_by_rule_id() -> None:
    registry = RuleMatcherRegistry()
    rule_id = RuleId("g-1")

    assert registry.get(rule_id) is None

    registry.register(rule_id, _noop_matcher)

    assert registry.get(rule_id) is _noop_matcher
    assert dict(registry.matchers) == {rule_id: _noop_matcher}


def test_that_a_rule_without_a_registered_matcher_returns_none() -> None:
    registry = RuleMatcherRegistry()
    registry.register(RuleId("g-1"), _noop_matcher)

    assert registry.get(RuleId("g-2")) is None
