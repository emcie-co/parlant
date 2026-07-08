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

import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from typing_extensions import override

from parlant.core.common import Weight
from lagom import Container
from parlant.core.engines.alpha.optimization_policy import BasicOptimizationPolicy
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import StdoutLogger
from parlant.core.nlp.generation import SchematicGenerator, SchematicGenerationResult
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.rules import Rule, RuleContent, RuleId
from parlant.core.services.indexing.journey_reachable_nodes_evaluation import (
    ChildEvaluation,
    JourneyReachableNodesEvaluator,
    PathCondition,
    ReachableNodesEvaluation,
    ReachableNodesEvaluationSchema,
)
from parlant.core.store_provider import BasicStoreProvider
from parlant.core.tracer import LocalTracer


class _ForwardAllReachableNodesGenerator(SchematicGenerator[ReachableNodesEvaluationSchema]):
    # A deterministic stand-in for the LLM: it reads which children and onward-path ids
    # the prompt exposes for the current node and forwards all of them, mimicking an
    # ideal compliant model. This lets us exercise the graph-walk/path logic without
    # real generation. Only `generate` is used.

    _CHILD_RE = re.compile(r"Child id:\s*(\S+)")
    _PATH_RE = re.compile(r"-\s*Condition\s*\((\d+)\)\s*:")

    @override
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[ReachableNodesEvaluationSchema]:
        text = prompt.build() if isinstance(prompt, PromptBuilder) else prompt
        # Drop the few-shot examples, which render the same child/condition lines.
        real_data = text.split("Example section is over")[-1]

        children: dict[str, list[str]] = {}
        current_child: str | None = None
        for line in real_data.splitlines():
            if child_match := self._CHILD_RE.search(line):
                current_child = child_match.group(1)
                children[current_child] = []
            elif (path_match := self._PATH_RE.search(line)) and current_child is not None:
                children[current_child].append(path_match.group(1))

        children_conditions = [
            ChildEvaluation(
                child_id=child_id,
                child_action=f"action of {child_id}",
                condition_to_child=f"condition to {child_id}",
                condition_to_child_and_stop=f"reached {child_id} and stopped",
                conditions_to_child_and_forward=[
                    PathCondition(
                        id=path_id,
                        path_condition=f"path {path_id} from {child_id}",
                        condition_to_child_then_to_path=f"through {child_id} via {path_id}",
                    )
                    for path_id in path_ids
                ]
                or None,
            )
            for child_id, path_ids in children.items()
        ]

        return SchematicGenerationResult(
            content=ReachableNodesEvaluationSchema(
                step_action="step action",
                step_action_completed="step action completed",
                children_conditions=children_conditions or None,
            ),
            info=GenerationInfo(
                schema_name=ReachableNodesEvaluationSchema.__name__,
                model="fake",
                duration=0.0,
                usage=UsageInfo(input_tokens=0, output_tokens=0),
            ),
        )

    @property
    @override
    def id(self) -> str:
        raise NotImplementedError

    @property
    @override
    def max_tokens(self) -> int:
        raise NotImplementedError

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        raise NotImplementedError


def _node_guideline(
    index: str,
    action: str,
    follow_ups: Sequence[str],
) -> Rule:
    return Rule(
        id=RuleId(index),
        creation_utc=datetime.now(timezone.utc),
        modified_utc=datetime.now(timezone.utc),
        content=RuleContent(condition="", action=action),
        enabled=True,
        groups=[],
        metadata={
            "journey_node": {
                "index": index,
                "follow_ups": [RuleId(f) for f in follow_ups],
                "kind": "chat",
            },
            "customer_dependent_action_data": {
                "is_customer_dependent": True,
                "customer_action": f"the customer completed step {index}",
                "agent_action": "",
            },
        },
        weight=Weight.MEDIUM,
    )


def _fan_in_journey(root_follow_ups: Sequence[str]) -> Sequence[Rule]:
    # Diamond: root (1) branches to two parents B (2) and C (3) that reconverge on the
    # fan-in node D (4), followed by a chain D -> E (5) -> G (6) -> H (7). The depth of
    # the chain makes the look-ahead depth filter relevant at D's parents.
    return [
        _node_guideline("1", "greet the customer", root_follow_ups),
        _node_guideline("2", "ask the customer for B", ["4"]),
        _node_guideline("3", "ask the customer for C", ["4"]),
        _node_guideline("4", "ask the customer for D", ["5"]),
        _node_guideline("5", "ask the customer for E", ["6"]),
        _node_guideline("6", "ask the customer for G", ["7"]),
        _node_guideline("7", "ask the customer for H", []),
    ]


def _paths(result: ReachableNodesEvaluation, node_index: str) -> set[tuple[str, ...]]:
    return {tuple(path) for _, path in result.node_to_reachable_follow_ups[node_index]}


async def _evaluate(node_rules: Sequence[Rule]) -> ReachableNodesEvaluation:
    evaluator = JourneyReachableNodesEvaluator(
        logger=StdoutLogger(LocalTracer()),
        optimization_policy=BasicOptimizationPolicy(),
        schematic_generator=_ForwardAllReachableNodesGenerator(),
        store_provider=BasicStoreProvider(lambda: Container()),
    )
    return await evaluator.evaluate_reachable_follow_ups(node_rules=node_rules)


async def test_that_fan_in_node_parents_have_identical_reachable_follow_ups() -> None:
    result = await _evaluate(_fan_in_journey(["2", "3"]))

    assert _paths(result, "2") == _paths(result, "3")


async def test_that_reachable_follow_ups_are_independent_of_node_visit_order() -> None:
    result_b_first = await _evaluate(_fan_in_journey(["2", "3"]))
    result_c_first = await _evaluate(_fan_in_journey(["3", "2"]))

    for node_index in result_b_first.node_to_reachable_follow_ups:
        assert _paths(result_b_first, node_index) == _paths(result_c_first, node_index)


async def test_that_reachable_follow_up_paths_have_no_repeated_consecutive_node() -> None:
    result = await _evaluate(_fan_in_journey(["2", "3"]))

    for node_index, follow_ups in result.node_to_reachable_follow_ups.items():
        for _, path in follow_ups:
            for current, following in zip(path, path[1:]):
                assert current != following, (
                    f"node {node_index} has a repeated consecutive id in path {path}"
                )
