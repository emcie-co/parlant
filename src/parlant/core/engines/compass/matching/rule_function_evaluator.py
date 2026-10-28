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
from parlant.core.engines.compass.tracing import CompassTracer
from parlant.core.engines.compass.context import EngineContext
from parlant.core.engines.rule_evaluation import RuleEvaluation
from parlant.core.engines.rule_evaluator_registry import (
    RuleCodeEvaluator,
    RuleEvaluatorRegistry,
)
from parlant.core.loggers import Logger
from parlant.core.pricing import PricingConfigurationError
from parlant.core.rules import Rule
from parlant.core.tracer import Tracer


class RuleFunctionEvaluator:
    """Runs code evaluators registered for rules."""

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        evaluator_registry: RuleEvaluatorRegistry,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._evaluator_registry = evaluator_registry

    async def evaluate(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
    ) -> Sequence[RuleEvaluation]:
        if not rules:
            return []

        with (
            self._logger.scope("RuleFunctionEvaluator"),
            self._tracer.span("compass.rule_function_evaluator.evaluate"),
        ):
            evaluators = [(rule, self._resolve_evaluator(rule)) for rule in rules]
            results = await safe_gather(
                *(self._run_evaluator(context, rule, evaluator) for rule, evaluator in evaluators)
            )

            compass_tracer = CompassTracer(self._tracer)
            for (rule, _), evaluation in zip(evaluators, results, strict=True):
                compass_tracer.rule_function_matched(
                    rule,
                    matched=evaluation.is_relevant if evaluation is not None else False,
                    error_type=None if evaluation is not None else "evaluator_error",
                )

            return [evaluation for evaluation in results if evaluation is not None]

    def _resolve_evaluator(self, rule: Rule) -> RuleCodeEvaluator:
        evaluator = self._evaluator_registry.get(rule.id)

        if evaluator is None:
            raise ValueError(
                f"Rule '{rule.id}' has no registered code evaluator; "
                "only function-attached rules may be passed to the function evaluator"
            )

        return evaluator

    async def _run_evaluator(
        self,
        context: EngineContext,
        rule: Rule,
        evaluator: RuleCodeEvaluator,
    ) -> RuleEvaluation | None:
        try:
            return await evaluator(context, rule)
        except PricingConfigurationError:
            raise
        except Exception as exc:
            self._logger.error(f"Error in code evaluator for rule {rule.id}: {exc}")
            return None
