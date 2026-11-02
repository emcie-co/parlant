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

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, cast

from parlant.core.agents import Effort
from parlant.core.container import Container
import pytest

from parlant.core.engines.engine_context import EngineContext
from parlant.core.engines.compass.matching.evaluation_scheduler import EvaluationSchedulingPolicy
from parlant.core.engines.rule_evaluation import RuleEvaluation
from parlant.core.groups import GroupStore
from parlant.core.loggers import Logger
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipKind,
)
from parlant.core.rules import Rule
from parlant.core.sessions import EventSource

from tests.core.stable.engines.compass.matching.harness import (
    make_rule as _rule,
    matching_harness as _matching_harness,
)


class _FakeLogger:
    """Records matcher log lines (local copy of test_matcher's _FakeLogger)."""

    def __init__(self) -> None:
        self.debug_messages: list[str] = []
        self.warning_messages: list[str] = []

    def debug(self, message: str) -> None:
        self.debug_messages.append(message)

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)


@pytest.mark.parametrize(
    ("recall_dependency", "rank_dependency", "dependent_is_evaluated", "dependent_is_admitted"),
    [
        # Recall admission alone never satisfies a dependency (only the
        # ever-matched ledger or a same-turn match does), so the dependent is
        # evaluated in the deferred round but stays out of the session set.
        (True, False, True, False),
        (False, True, True, True),
        (False, False, False, False),
    ],
)
async def test_that_real_recall_and_ranking_evidence_jointly_control_dependency_release(
    container: Container,
    recall_dependency: bool,
    rank_dependency: bool,
    dependent_is_evaluated: bool,
    dependent_is_admitted: bool,
) -> None:
    dependency = _rule("DEPENDENCY")
    dependent = _rule("DEPENDENT")
    fillers = [_rule(f"FILLER_{index}") for index in range(6)]
    rules = [dependent, dependency, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={
            "DEPENDENT",
            *(["DEPENDENCY"] if rank_dependency else []),
        },
        conversation=(
            "RECALL_DEPENDENCY" if recall_dependency else "No discovery marker is present."
        ),
    )
    await harness.depend(dependent, dependency)

    await harness.match()

    assert ("DEPENDENT" in harness.rank_generator.calls) is dependent_is_evaluated
    expected_session_ids = set()
    if recall_dependency or rank_dependency:
        expected_session_ids.add(str(dependency.id))
    if dependent_is_admitted:
        expected_session_ids.add(str(dependent.id))
    assert await harness.staged_discovered_rule_ids() == expected_session_ids


async def test_that_a_real_code_evaluator_can_unlock_a_model_evaluated_dependent(
    container: Container,
) -> None:
    dependency = _rule("CODE_DEPENDENCY")
    dependent = _rule("MODEL_DEPENDENT")
    fillers = [_rule(f"FILLER_{index}") for index in range(6)]
    rules = [dependent, dependency, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={"MODEL_DEPENDENT"},
    )
    await harness.depend(dependent, dependency)
    code_calls: list[str] = []

    async def evaluate_dependency(
        context: EngineContext[Any, Any],
        rule: Rule,
    ) -> RuleEvaluation:
        del context
        code_calls.append(rule.condition or "")
        return RuleEvaluation(rule=rule, reasoning="code matched", is_relevant=True)

    harness.evaluator_registry.register(dependency.id, evaluate_dependency)

    await harness.match()

    assert code_calls == [dependency.condition]
    assert harness.rank_generator.calls[-1] == "MODEL_DEPENDENT"
    assert await harness.staged_discovered_rule_ids() == {
        str(dependency.id),
        str(dependent.id),
    }


async def test_that_recalled_rules_are_pending_during_a_deferred_evaluation(
    container: Container,
) -> None:
    dependency = _rule("DEPENDENCY")
    dependent = _rule("CODE_DEPENDENT")
    fillers = [_rule(f"FILLER_{index}") for index in range(6)]
    rules = [dependent, dependency, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant=set(),
        conversation="RECALL_DEPENDENCY",
    )
    await harness.depend(dependent, dependency)
    pending_during_evaluation: set[Rule] = set()

    async def evaluate_dependent(
        context: EngineContext[Any, Any],
        rule: Rule,
    ) -> RuleEvaluation:
        pending_during_evaluation.update(context.state.matcher.pending_discovered_rules)
        return RuleEvaluation(rule=rule, reasoning="code matched", is_relevant=True)

    harness.evaluator_registry.register(dependent.id, evaluate_dependent)

    await harness.match()

    assert pending_during_evaluation == {dependency}
    assert harness.context.state.matcher.pending_discovered_rules == set()
    # The dependency was only recall-admitted, never matched, so it does not
    # satisfy the dependent's gate: the dependent is evaluated but stays out
    # of the discovered set (only the ever-matched ledger unlocks dependents).
    assert harness.context.state.discovered_rules == {dependency}


async def test_that_a_real_model_evaluator_can_unlock_a_code_evaluated_dependent(
    container: Container,
) -> None:
    dependency = _rule("MODEL_DEPENDENCY")
    dependent = _rule("CODE_DEPENDENT")
    fillers = [_rule(f"FILLER_{index}") for index in range(6)]
    rules = [dependent, dependency, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={"MODEL_DEPENDENCY"},
    )
    await harness.depend(dependent, dependency)
    code_calls: list[str] = []

    async def evaluate_dependent(
        context: EngineContext[Any, Any],
        rule: Rule,
    ) -> RuleEvaluation:
        del context
        code_calls.append(rule.condition or "")
        return RuleEvaluation(rule=rule, reasoning="code matched", is_relevant=True)

    harness.evaluator_registry.register(dependent.id, evaluate_dependent)

    await harness.match()

    assert code_calls == [dependent.condition]
    assert "CODE_DEPENDENT" not in harness.rank_generator.calls
    assert await harness.staged_discovered_rule_ids() == {
        str(dependency.id),
        str(dependent.id),
    }


async def test_that_an_irrelevant_real_code_evaluator_prevents_dependent_evaluation(
    container: Container,
) -> None:
    dependency = _rule("CODE_DEPENDENCY")
    dependent = _rule("MODEL_DEPENDENT")
    fillers = [_rule(f"FILLER_{index}") for index in range(6)]
    rules = [dependent, dependency, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={"MODEL_DEPENDENT"},
    )
    await harness.depend(dependent, dependency)

    async def reject_dependency(
        context: EngineContext[Any, Any],
        rule: Rule,
    ) -> RuleEvaluation:
        del context
        return RuleEvaluation(rule=rule, reasoning="code rejected", is_relevant=False)

    harness.evaluator_registry.register(dependency.id, reject_dependency)

    await harness.match()

    assert "MODEL_DEPENDENT" not in harness.rank_generator.calls
    assert await harness.staged_discovered_rule_ids() == set()


async def test_that_a_session_rule_unlocks_a_dependent_without_reopening_its_dependencies(
    container: Container,
) -> None:
    leaf = _rule("REJECTED_LEAF")
    session_dependency = _rule("SESSION_DEPENDENCY")
    dependent = _rule("DEPENDENT")
    fillers = [_rule(f"FILLER_{index}") for index in range(5)]
    rules = [dependent, session_dependency, leaf, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={"DEPENDENT"},
        discovered_rules={session_dependency},
        # The session rule has genuinely matched earlier in the session:
        # dependencies are satisfied by the ever-matched ledger, not by
        # discovery admission.
        rules_ever_matched={session_dependency},
    )
    await harness.depend(dependent, session_dependency)
    await harness.depend(session_dependency, leaf)

    await harness.match()

    assert "DEPENDENT" in harness.rank_generator.calls
    assert "SESSION_DEPENDENCY" not in harness.rank_generator.calls
    assert await harness.staged_discovered_rule_ids() == {
        str(session_dependency.id),
        str(dependent.id),
    }


async def test_that_a_three_level_chain_uses_dependency_order_across_real_evaluation_rounds(
    container: Container,
) -> None:
    root = _rule("ROOT")
    middle = _rule("MIDDLE")
    leaf = _rule("LEAF")
    fillers = [_rule(f"FILLER_{index}") for index in range(5)]
    rules = [root, middle, leaf, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={"ROOT", "MIDDLE", "LEAF"},
    )
    await harness.depend(root, middle)
    await harness.depend(middle, leaf)
    matcher_logger = _FakeLogger()
    harness.matcher._logger = cast(Logger, matcher_logger)

    await harness.match()

    assert set(harness.rank_generator.calls[:6]) == {
        "LEAF",
        *(f"FILLER_{index}" for index in range(5)),
    }
    assert set(harness.rank_generator.calls[6:]) == {"ROOT", "MIDDLE"}
    assert await harness.staged_discovered_rule_ids() == {
        str(root.id),
        str(middle.id),
        str(leaf.id),
    }
    root_prompt = harness.rank_generator.prompt_texts[harness.rank_generator.calls.index("ROOT")]
    assert "RANK_MIDDLE" in root_prompt
    assert "CONTEXT THIS RULE MAY OPERATE WITHIN" in root_prompt
    assert any(
        str(root.id) in message and str(middle.id) in message
        for message in matcher_logger.warning_messages
    )


async def test_that_real_dependency_any_relationships_release_on_one_relevant_alternative(
    container: Container,
) -> None:
    first = _rule("FIRST")
    second = _rule("SECOND")
    dependent = _rule("DEPENDENT")
    fillers = [_rule(f"FILLER_{index}") for index in range(5)]
    rules = [dependent, first, second, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={"SECOND", "DEPENDENT"},
    )
    await harness.depend(
        dependent,
        first,
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id="alternatives",
    )
    await harness.depend(
        dependent,
        second,
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id="alternatives",
    )

    await harness.match()

    assert "DEPENDENT" in harness.rank_generator.calls
    dependent_prompt = harness.rank_generator.prompt_texts[
        harness.rank_generator.calls.index("DEPENDENT")
    ]
    assert "RANK_SECOND" in dependent_prompt
    assert "RANK_FIRST" not in dependent_prompt
    assert harness.context.state.dependency_prompt_targets_by_rule[dependent.id] == [second]
    assert await harness.staged_discovered_rule_ids() == {
        str(second.id),
        str(dependent.id),
    }


async def test_that_a_satisfied_and_an_unknown_alternative_render_all_fallback_context(
    container: Container,
) -> None:
    satisfied = _rule("SATISFIED")
    unknown = _rule("UNKNOWN")
    dependent = _rule("DEPENDENT")
    fillers = [_rule(f"FILLER_{index}") for index in range(5)]
    rules = [dependent, satisfied, unknown, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={"SATISFIED", "DEPENDENT"},
        discovered_rules={satisfied},
    )
    await harness.depend(
        dependent,
        satisfied,
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id="alternatives",
    )
    await harness.depend(
        dependent,
        unknown,
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id="alternatives",
    )
    matcher_logger = _FakeLogger()
    harness.matcher._logger = cast(Logger, matcher_logger)

    await harness.match()

    dependent_prompt = harness.rank_generator.prompt_texts[
        harness.rank_generator.calls.index("DEPENDENT")
    ]
    assert "RANK_SATISFIED" in dependent_prompt
    assert "RANK_UNKNOWN" in dependent_prompt
    assert "CONTEXT THIS RULE MAY OPERATE WITHIN" in dependent_prompt
    assert any(
        str(dependent.id) in message and str(unknown.id) in message
        for message in matcher_logger.warning_messages
    )


async def test_that_one_pass_dependency_evaluation_warns_and_renders_all_unknown_targets(
    container: Container,
) -> None:
    target = _rule("TARGET")
    dependent = _rule("DEPENDENT")
    fillers = [_rule(f"FILLER_{index}") for index in range(4)]
    rules = [dependent, target, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={"TARGET", "DEPENDENT"},
        policy=EvaluationSchedulingPolicy(threshold=8, max_rounds=2),
    )
    await harness.depend(dependent, target)
    matcher_logger = _FakeLogger()
    harness.matcher._logger = cast(Logger, matcher_logger)

    await harness.match()

    dependent_prompt = harness.rank_generator.prompt_texts[
        harness.rank_generator.calls.index("DEPENDENT")
    ]
    assert "RANK_TARGET" in dependent_prompt
    assert "CONTEXT THIS RULE MAY OPERATE WITHIN" in dependent_prompt
    assert any(
        str(dependent.id) in message and str(target.id) in message
        for message in matcher_logger.warning_messages
    )


@pytest.mark.parametrize(
    ("target_kind", "dependent_is_evaluated"),
    [
        (RelationshipEntityKind.GROUP_ALL, False),
        (RelationshipEntityKind.GROUP_ANY, True),
    ],
)
async def test_that_real_group_target_semantics_control_dependent_evaluation(
    container: Container,
    target_kind: RelationshipEntityKind,
    dependent_is_evaluated: bool,
) -> None:
    group = await container[GroupStore].create_group(name=f"target-{target_kind.value}")
    first = replace(_rule("FIRST"), groups=[group.id])
    second = replace(_rule("SECOND"), groups=[group.id])
    dependent = _rule("DEPENDENT")
    fillers = [_rule(f"FILLER_{index}") for index in range(5)]
    rules = [dependent, first, second, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={"FIRST", "DEPENDENT"},
    )
    await harness.depend_on_group(
        dependent,
        group.id,
        target_kind=target_kind,
    )

    await harness.match()

    assert ("DEPENDENT" in harness.rank_generator.calls) is dependent_is_evaluated
    expected_session_ids = {str(first.id)}
    if dependent_is_evaluated:
        expected_session_ids.add(str(dependent.id))
        dependent_prompt = harness.rank_generator.prompt_texts[
            harness.rank_generator.calls.index("DEPENDENT")
        ]
        assert "RANK_FIRST" in dependent_prompt
        assert "RANK_SECOND" not in dependent_prompt
    assert await harness.staged_discovered_rule_ids() == expected_session_ids


async def test_that_an_early_satisfied_target_remains_in_the_dependent_prompt(
    container: Container,
) -> None:
    target = _rule("EARLY_TARGET")
    dependent = _rule("DEPENDENT")
    fillers = [_rule(f"FILLER_{index}") for index in range(6)]
    rules = [dependent, target, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={"DEPENDENT"},
    )
    await harness.depend(dependent, target)
    harness.context.state.early_evaluated_rule_ids = {target.id}
    harness.context.state.early_rule_evaluations = [
        RuleEvaluation(rule=target, reasoning="early match", is_relevant=True)
    ]
    harness.context.state.discovered_rules = {target}

    await harness.match()

    dependent_prompt = harness.rank_generator.prompt_texts[
        harness.rank_generator.calls.index("DEPENDENT")
    ]
    assert "RANK_EARLY_TARGET" in dependent_prompt
    assert "CONTEXT THIS RULE OPERATES WITHIN" in dependent_prompt


async def test_that_real_discovery_and_round_one_ranking_start_concurrently(
    container: Container,
) -> None:
    dependency = _rule("DEPENDENCY")
    dependent = _rule("DEPENDENT")
    fillers = [_rule(f"FILLER_{index}") for index in range(6)]
    rules = [dependent, dependency, *fillers]
    rank_started = asyncio.Event()
    embedding_started = asyncio.Event()
    release = asyncio.Event()
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={"DEPENDENCY", "DEPENDENT"},
        rank_started=rank_started,
        rank_release=release,
        embedding_started=embedding_started,
        embedding_release=release,
    )
    await harness.depend(dependent, dependency)

    matching = asyncio.create_task(harness.match())
    await asyncio.wait_for(rank_started.wait(), timeout=1.0)
    await asyncio.wait_for(embedding_started.wait(), timeout=1.0)
    release.set()
    await matching

    assert await harness.staged_discovered_rule_ids() == {
        str(dependency.id),
        str(dependent.id),
    }


async def test_that_a_released_second_round_rule_receives_real_glossary_context(
    container: Container,
) -> None:
    dependency = _rule("DEPENDENCY")
    dependent = _rule("DEPENDENT", extra_condition="SPECIAL_TERM")
    fillers = [_rule(f"FILLER_{index}") for index in range(6)]
    rules = [dependent, dependency, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={"DEPENDENCY", "DEPENDENT"},
    )
    await harness.depend(dependent, dependency)
    await harness.glossary_store.create_term(
        name="SPECIAL_TERM",
        description="ZZQ_GLOSSARY_DEFINITION",
    )

    await harness.match()

    dependent_prompts = [
        prompt
        for name, prompt in zip(
            harness.rank_generator.calls,
            harness.rank_generator.prompt_texts,
            strict=True,
        )
        if name == "DEPENDENT"
    ]
    assert len(dependent_prompts) == 1
    assert "ZZQ_GLOSSARY_DEFINITION" in dependent_prompts[0]


async def test_that_a_skipped_rule_never_reaches_real_glossary_lookup_or_ranking(
    container: Container,
) -> None:
    dependency = _rule("DEPENDENCY")
    dependent = _rule("DEPENDENT", extra_condition="SPECIAL_TERM")
    fillers = [_rule(f"FILLER_{index}") for index in range(6)]
    rules = [dependent, dependency, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={"DEPENDENT"},
    )
    await harness.depend(dependent, dependency)
    await harness.glossary_store.create_term(
        name="SPECIAL_TERM",
        description="ZZQ_GLOSSARY_DEFINITION",
    )

    await harness.match()

    assert "DEPENDENT" not in harness.rank_generator.calls
    glossary_queries = [
        text for batch in harness.glossary_embedder.calls for text in batch if "RANK_" in text
    ]
    assert all("SPECIAL_TERM" not in query for query in glossary_queries)


async def test_that_a_bare_cross_referential_rule_is_never_evaluated_end_to_end(
    container: Container,
) -> None:
    # Classical end-to-end: real matcher, real scheduler, real recall — only
    # the model boundaries are scripted. A bare cross-referential rule (no
    # attachments) must be statically included and never appear anywhere in
    # the evaluation pipeline: not ranked, not recalled, not persisted.
    # See matching/common.py: "Static inclusion".
    bare = replace(
        _rule("BARE_NET"),
        metadata={"references_other_policies": True},
    )
    plain = _rule("PLAIN")
    fillers = [_rule(f"FILLER_{index}") for index in range(6)]
    rules = [bare, plain, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={"PLAIN"},
        conversation="RECALL_PLAIN",
    )

    await harness.matcher._load_static_inclusion(harness.context)
    await harness.match()

    assert [rule.id for rule in harness.context.state.statically_included_rules] == [bare.id]
    # Never ranked...
    assert "BARE_NET" not in harness.rank_generator.calls
    # ...and never admitted to the discovered set.
    assert str(bare.id) not in await harness.staged_discovered_rule_ids()
    # The plain rule's flow is unaffected.
    assert str(plain.id) in await harness.staged_discovered_rule_ids()


async def test_that_an_attached_cross_referential_rule_is_ranked_end_to_end(
    container: Container,
) -> None:
    # The attached counterpart: a cross-referential rule carrying a review
    # request must flow through real ranking (escalated regardless of the
    # strategy matrix) and land as a turn match when judged relevant.
    attached = replace(
        _rule("ATTACHED_NET"),
        metadata={"references_other_policies": True},
        message_review=True,
    )
    fillers = [_rule(f"FILLER_{index}") for index in range(6)]
    rules = [attached, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={"ATTACHED_NET"},
        conversation="RECALL_ATTACHED_NET",
    )

    await harness.matcher._load_static_inclusion(harness.context)
    await harness.match()

    assert harness.context.state.statically_included_rules == []
    assert "ATTACHED_NET" in harness.rank_generator.calls
    assert any(
        evaluation.rule.id == attached.id
        for evaluation in harness.context.state.ordinary_rule_matches
    )


async def test_that_mid_turn_update_reevaluates_a_gated_cross_referential_rule(
    container: Container,
) -> None:
    # The update() path: after a tool runs, rules gated on it (REEVALUATION
    # relationships) are re-evaluated. An attached cross-referential rule
    # behind such a gate must flow through real ranking mid-turn and land as
    # a turn match — with the same routing guarantees as the initial match.
    from parlant.core.emissions import EmittedEvent
    from parlant.core.relationships import (
        RelationshipEntity,
        RelationshipEntityKind,
        RelationshipKind,
    )
    from parlant.core.sessions import EventKind
    from parlant.core.tools import ToolId

    gated = replace(
        _rule("GATED_NET"),
        metadata={"references_other_policies": True},
        message_review=True,
    )
    fillers = [_rule(f"FILLER_{index}") for index in range(6)]
    harness = await _matching_harness(
        container,
        [gated, *fillers],
        rank_relevant={"GATED_NET"},
        conversation="RECALL_GATED_NET",
    )

    tool_id = ToolId(service_name="svc", tool_name="check_coverage")
    await harness.relationship_store.create_relationship(
        source=RelationshipEntity(id=gated.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=tool_id, kind=RelationshipEntityKind.TOOL),
        kind=RelationshipKind.REEVALUATION,
    )

    await harness.matcher._load_static_inclusion(harness.context)

    harness.context.state.tool_ids_by_name = {"check_coverage": tool_id}
    harness.context.state.tool_events = [
        EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.TOOL,
            trace_id="",
            data={
                "tool_calls": [
                    {
                        "tool_id": "check_coverage",
                        "arguments": {},
                        "result": {"data": {}, "metadata": {}, "control": {}},
                    }
                ]
            },
            metadata=None,
        )
    ]

    await harness.matcher._reevaluate(harness.context)

    assert "GATED_NET" in harness.rank_generator.calls
    assert any(
        evaluation.rule.id == gated.id for evaluation in harness.context.state.ordinary_rule_matches
    )


async def test_that_mid_turn_reevaluation_keeps_out_of_batch_dependency_context(
    container: Container,
) -> None:
    from parlant.core.emissions import EmittedEvent
    from parlant.core.relationships import RelationshipEntity
    from parlant.core.sessions import EventKind
    from parlant.core.tools import ToolId

    dependency = _rule("DEPENDENCY")
    dependent = replace(_rule("GATED_DEPENDENT"), message_review=True)
    fillers = [_rule(f"FILLER_{index}") for index in range(5)]
    harness = await _matching_harness(
        container,
        [dependent, dependency, *fillers],
        rank_relevant={"GATED_DEPENDENT"},
        discovered_rules={dependency},
        effort=Effort.MEDIUM,
    )
    await harness.depend(dependent, dependency)

    tool_id = ToolId(service_name="svc", tool_name="refresh")
    await harness.relationship_store.create_relationship(
        source=RelationshipEntity(id=dependent.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=tool_id, kind=RelationshipEntityKind.TOOL),
        kind=RelationshipKind.REEVALUATION,
    )
    harness.context.state.tool_ids_by_name = {"refresh": tool_id}
    harness.context.state.tool_events = [
        EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.TOOL,
            trace_id="",
            data={
                "tool_calls": [
                    {
                        "tool_id": "refresh",
                        "arguments": {},
                        "result": {"data": {}, "metadata": {}, "control": {}},
                    }
                ]
            },
            metadata=None,
        )
    ]

    await harness.matcher._reevaluate(harness.context)

    dependent_prompt = harness.rank_generator.prompt_texts[
        harness.rank_generator.calls.index("GATED_DEPENDENT")
    ]
    assert "RANK_DEPENDENCY" in dependent_prompt
    assert "CONTEXT THIS RULE OPERATES WITHIN" in dependent_prompt


async def test_that_a_round_one_rank_match_missed_by_recall_joins_a_deferred_rules_policy_context(
    container: Container,
) -> None:
    # Large landscape: the flagged rule is deferred to the final round so its
    # policy context can absorb this turn's evidence. A rule that recall
    # misses but round-1 ranking judges relevant IS such evidence — the
    # embedding blind spot is the very case this feature serves.
    flagged = replace(
        _rule("FLAGGED_NET"),
        metadata={"references_other_policies": True},
        message_review=True,
    )
    caught = _rule("CAUGHT")
    fillers = [_rule(f"FILLER_{index}") for index in range(20)]
    rules = [flagged, caught, *fillers]
    harness = await _matching_harness(
        container,
        rules,
        rank_relevant={"FLAGGED_NET", "CAUGHT"},
        # No recall markers in the conversation: recall misses CAUGHT; only
        # its round-1 rank verdict knows it applies.
    )

    await harness.matcher._load_static_inclusion(harness.context)
    await harness.match()

    flagged_prompts = [
        prompt
        for name, prompt in zip(
            harness.rank_generator.calls,
            harness.rank_generator.prompt_texts,
            strict=True,
        )
        if name == "FLAGGED_NET"
    ]
    assert len(flagged_prompts) == 1
    assert "RANK_CAUGHT" in flagged_prompts[0]
