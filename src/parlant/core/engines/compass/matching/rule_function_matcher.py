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
from parlant.core.engines.rule_match import RuleMatch
from parlant.core.engines.rule_matcher_registry import (
    RuleCodeMatcher,
    RuleMatcherRegistry,
)
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.engines.compass.tracing import CompassTracer
from parlant.core.rules import Rule
from parlant.core.loggers import Logger
from parlant.core.tracer import Tracer


class RuleFunctionMatcher:
    """Runs the code (Python) matchers of rules that have one registered.

    This is the compass counterpart to the alpha engine's
    CustomRuleMatchingStrategy: it reads the same engine-agnostic
    RuleMatcherRegistry and runs each matcher with the current EngineContext.

    Every rule passed in is expected to be function-attached — the caller
    (the engine) selects them from the registry. A rule without a registered
    matcher is a programming error and raises.
    """

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        matcher_registry: RuleMatcherRegistry,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._matcher_registry = matcher_registry

    async def match(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
    ) -> Sequence[RuleMatch]:
        if not rules:
            return []

        with self._tracer.span("match.rule.function"):
            # All rules must be function-attached; resolve each matcher up
            # front so a non-attached rule fails loudly rather than silently.
            matchers = [(g, self._resolve_matcher(g)) for g in rules]

            # Code matchers are independent; run them concurrently.
            results = await safe_gather(
                *(self._run_matcher(context, g, matcher) for g, matcher in matchers)
            )

            return [match for match in results if match is not None]

    def _resolve_matcher(self, rule: Rule) -> RuleCodeMatcher:
        matcher = self._matcher_registry.get(rule.id)

        if matcher is None:
            raise ValueError(
                f"Rule '{rule.id}' has no registered code matcher; "
                "only function-attached rules may be passed to the function matcher"
            )

        return matcher

    async def _run_matcher(
        self,
        context: EngineContext,
        rule: Rule,
        matcher: RuleCodeMatcher,
    ) -> RuleMatch | None:
        try:
            match = await matcher(context, rule)
            CompassTracer(context.tracer).rule_function_matched(rule, bool(match), None)
            return match
        except Exception as e:
            self._logger.error(f"Error in code matcher for rule {rule.id}: {e}")
            CompassTracer(context.tracer).rule_function_matched(rule, False, type(e).__name__)
            return None
