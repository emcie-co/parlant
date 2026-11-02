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

"""Diagnostic matcher telemetry: the per-rule match.* events emitted live by
the recaller, ranker, and function evaluator (unlike the authoritative
rule.selected/ruled_out decisions, these are not persistence-coupled)."""

from typing import Any, Iterator

import pytest
from pytest import fixture

from parlant.core.container import Container
from parlant.core.engines.compass.matching.rule_function_evaluator import RuleFunctionEvaluator
from parlant.core.engines.rule_evaluator_registry import RuleEvaluatorRegistry
from parlant.core.engines.rule_evaluation import RuleEvaluation
from parlant.core.loggers import StdoutLogger
from parlant.core.tracer import LocalTracer, Tracer

from tests.core.stable.engines.compass.matching.harness import make_rule, matching_harness
from tests.core.stable.engines.compass.matching.utils import create_engine_context


class _EventRecordingTracer(LocalTracer):
    def __init__(self) -> None:
        super().__init__()
        self.recorded_events: list[tuple[str, dict[str, Any]]] = []

    def add_event(self, name: str, attributes: Any = {}) -> None:
        self.recorded_events.append((name, dict(attributes)))


@fixture
def tracer(request: pytest.FixtureRequest) -> Iterator[Tracer]:
    # Overrides the root fixture so every component the container wires
    # (recaller, ranker, matcher) reports into the recording tracer.
    recording = _EventRecordingTracer()

    with recording.attributes({"scope": request.node.name}):
        yield recording


async def test_that_ranking_emits_per_rule_rank_events(
    container: Container,
    tracer: Tracer,
) -> None:
    chosen = make_rule("CHOSEN")
    rejected = make_rule("REJECTED")
    harness = await matching_harness(
        container,
        [chosen, rejected],
        rank_relevant={"CHOSEN"},
    )

    await harness.match()

    recording = tracer
    assert isinstance(recording, _EventRecordingTracer)

    rank_yes = [a for n, a in recording.recorded_events if n == "match.rank.yes"]
    rank_no = [a for n, a in recording.recorded_events if n == "match.rank.no"]

    assert any(a["rule_id"] == str(chosen.id) for a in rank_yes)
    assert any(a["rule_id"] == str(rejected.id) for a in rank_no)
    assert all("reasoning" in a for a in rank_yes + rank_no)


async def test_that_recall_emits_per_rule_recall_events(
    container: Container,
    tracer: Tracer,
) -> None:
    rule = make_rule("PINNED")
    harness = await matching_harness(
        container,
        [rule],
        rank_relevant={"PINNED"},
    )

    await harness.match()

    recording = tracer
    assert isinstance(recording, _EventRecordingTracer)

    recall_events = [
        a for n, a in recording.recorded_events if n in ("match.recall.yes", "match.recall.no")
    ]
    assert any(a["rule_id"] == str(rule.id) for a in recall_events)


async def test_that_the_function_evaluator_emits_per_rule_function_events() -> None:
    tracer = _EventRecordingTracer()

    matched_rule = make_rule("FUNCTION_YES")
    unmatched_rule = make_rule("FUNCTION_NO")

    registry = RuleEvaluatorRegistry()

    async def match_verdict(context: Any, rule: Any) -> RuleEvaluation:
        return RuleEvaluation(rule=rule, reasoning="matched by code", is_relevant=True)

    async def no_match_verdict(context: Any, rule: Any) -> RuleEvaluation:
        return RuleEvaluation(rule=rule, reasoning="rejected by code", is_relevant=False)

    registry.register(matched_rule.id, match_verdict)
    registry.register(unmatched_rule.id, no_match_verdict)

    evaluator = RuleFunctionEvaluator(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        evaluator_registry=registry,
    )

    context = create_engine_context(conversation=[])
    evaluations = await evaluator.evaluate(context, [matched_rule, unmatched_rule])
    assert len(evaluations) == 2

    yes = [a for n, a in tracer.recorded_events if n == "match.function.yes"]
    no = [a for n, a in tracer.recorded_events if n == "match.function.no"]

    assert [a["rule_id"] for a in yes] == [str(matched_rule.id)]
    assert [a["rule_id"] for a in no] == [str(unmatched_rule.id)]


async def test_that_preload_reports_loaded_rules_and_variables(
    container: Container,
    tracer: Tracer,
) -> None:
    from parlant.core.rules import RuleStore

    stored_rule = await container[RuleStore].create_rule(
        condition="Customer asks about pricing",
    )

    harness = await matching_harness(
        container,
        [make_rule("SEEDED")],
        rank_relevant=set(),
    )

    recording = tracer
    assert isinstance(recording, _EventRecordingTracer)
    # Component-load reporting rides the turn's context tracer.
    harness.context.tracer = recording

    await harness.matcher.preload(harness.context)

    loaded_rules = [a for n, a in recording.recorded_events if n == "loaded.rule"]
    assert str(stored_rule.id) in [a["rule_id"] for a in loaded_rules]
    assert all(a["last_modified"] for a in loaded_rules)

    rule_refs = [
        a
        for n, a in recording.recorded_events
        if n == "loaded.entity" and a["entity_type"] == "rule"
    ]
    assert str(stored_rule.id) in [a["parlant_id"] for a in rule_refs]
