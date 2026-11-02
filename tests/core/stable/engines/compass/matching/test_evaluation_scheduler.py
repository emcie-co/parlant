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

# mypy: ignore-errors

from collections.abc import Generator
from datetime import datetime, timezone
from contextlib import contextmanager
from dataclasses import replace as _replace

import pytest

from parlant.core.engines.compass.matching.evaluation_scheduler import (
    CrossReferenceEvaluationPreconditionProvider,
    DependencyEvaluationPreconditionProvider,
    EvaluationEvidence,
    EvaluationPrecondition,
    EvaluationPreconditionProvider,
    EvaluationPreconditionRegistry,
    EvaluationPreconditionState,
    EvaluationScheduler,
    EvaluationSchedulingPolicy,
)
from parlant.core.rules import Rule, RuleId
from parlant.core.groups import GroupId
from parlant.core.relationships import (
    Relationship,
    RelationshipEntity,
    RelationshipEntityKind,
    RelationshipId,
    RelationshipKind,
)

from parlant.core.engines.compass.matching.common import MAX_FULL_POLICY_CONTEXT_RULES
from tests.core.stable.engines.compass.matching.utils import create_engine_context, create_rule


class _Logger:
    def __init__(self) -> None:
        self.warning_messages: list[str] = []
        self.debug_messages: list[str] = []

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)

    def debug(self, message: str) -> None:
        self.debug_messages.append(message)

    @contextmanager
    def scope(self, scope_id: str) -> Generator[None, None, None]:
        yield


class _RelationshipStore:
    def __init__(self, relationships=()):
        self.relationships = list(relationships)

    async def list_relationships(self, kind=None, indirect=False, source_id=None, target_id=None):
        return [
            relationship
            for relationship in self.relationships
            if (kind is None or relationship.kind == kind)
            and (source_id is None or relationship.source.id == source_id)
            and (target_id is None or relationship.target.id == target_id)
        ]


class _FailingProvider(EvaluationPreconditionProvider):
    async def provide(self, context, rules):
        raise RuntimeError("provider failed")


class _RequiresRule(EvaluationPrecondition):
    def __init__(self, rule_id: RuleId) -> None:
        self._rule_id = rule_id

    def evaluate(self, evidence: EvaluationEvidence) -> EvaluationPreconditionState:
        return evidence.status(self._rule_id)


class _CountingRequiresRule(_RequiresRule):
    def __init__(self, rule_id: RuleId) -> None:
        super().__init__(rule_id)
        self.calls = 0

    def evaluate(self, evidence: EvaluationEvidence) -> EvaluationPreconditionState:
        self.calls += 1
        return super().evaluate(evidence)


class _StaticProvider(EvaluationPreconditionProvider):
    def __init__(self, preconditions):
        self._preconditions = preconditions

    async def provide(self, context, rules):
        return self._preconditions


def _relationship(
    source,
    target,
    *,
    kind=RelationshipKind.DEPENDENCY,
    group_id=None,
) -> Relationship:
    now = datetime.now(timezone.utc)
    return Relationship(
        id=RelationshipId(f"rel-{source.id}-{target.id}-{group_id or kind.value}"),
        creation_utc=now,
        modified_utc=now,
        source=RelationshipEntity(source.id, RelationshipEntityKind.RULE),
        target=RelationshipEntity(target.id, RelationshipEntityKind.RULE),
        kind=kind,
        group_id=group_id,
    )


def _scheduler(relationships=(), *, threshold=8, max_rounds=2) -> EvaluationScheduler:
    provider = DependencyEvaluationPreconditionProvider(_RelationshipStore(relationships))
    return EvaluationScheduler(
        logger=_Logger(),
        precondition_registry=EvaluationPreconditionRegistry([provider]),
        policy=EvaluationSchedulingPolicy(threshold=threshold, max_rounds=max_rounds),
    )


async def _plan(
    scheduler,
    rules,
    *,
    candidates=None,
    pending=None,
    satisfied=frozenset(),
    relevant=frozenset(),
):
    return await scheduler.create_plan(
        create_engine_context(conversation=[]),
        candidates=rules if candidates is None else candidates,
        rules=rules,
        pending_evidence_sources=({rule.id: 1 for rule in rules} if pending is None else pending),
        initially_satisfied_rule_ids=satisfied,
        initially_relevant_rule_ids=relevant,
    )


def test_that_the_scheduling_policy_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="threshold must be positive"):
        EvaluationSchedulingPolicy(threshold=0)

    with pytest.raises(ValueError, match="max_rounds must be positive"):
        EvaluationSchedulingPolicy(max_rounds=0)


@pytest.mark.asyncio
async def test_that_fewer_than_the_threshold_uses_one_round() -> None:
    dependency = create_rule("dependency")
    dependent = create_rule("dependent")
    plan = await _plan(
        _scheduler([_relationship(dependent, dependency)]),
        [dependent, dependency],
    )

    assert not plan.enabled
    assert set(plan.next_round()) == {dependent, dependency}


@pytest.mark.asyncio
async def test_that_an_inventory_without_preconditions_uses_one_round() -> None:
    rules = [create_rule(f"rule {index}") for index in range(8)]
    plan = await _plan(_scheduler(), rules)

    assert not plan.enabled
    assert set(plan.next_round()) == set(rules)


@pytest.mark.asyncio
async def test_that_effective_state_snapshots_are_reused_until_new_evidence_arrives() -> None:
    dependency = create_rule("dependency")
    dependent = create_rule("dependent")
    fillers = [create_rule(f"filler {index}") for index in range(6)]
    rules = [dependent, dependency, *fillers]
    condition = _CountingRequiresRule(dependency.id)
    scheduler = EvaluationScheduler(
        logger=_Logger(),
        precondition_registry=EvaluationPreconditionRegistry(
            [_StaticProvider({dependent.id: [condition]})]
        ),
        policy=EvaluationSchedulingPolicy(),
    )
    plan = await _plan(scheduler, rules)

    plan.next_round()
    calls_after_scheduling = condition.calls
    plan.effective_rule_states()

    assert condition.calls == calls_after_scheduling

    plan.observe(checked_rule_ids={dependency.id}, relevant_rule_ids={dependency.id})
    plan.effective_rule_states()

    assert condition.calls > calls_after_scheduling


@pytest.mark.asyncio
async def test_that_only_actual_evaluation_candidates_count_toward_the_threshold() -> None:
    dependency = create_rule("dependency")
    dependent = create_rule("dependent")
    non_candidates = [create_rule(f"non-candidate {index}") for index in range(6)]
    rules = [dependent, dependency, *non_candidates]
    plan = await _plan(
        _scheduler([_relationship(dependent, dependency)]),
        rules,
        candidates=[dependent, dependency],
    )

    assert not plan.enabled
    assert set(plan.next_round()) == {dependent, dependency}


@pytest.mark.asyncio
async def test_that_a_relevant_dependency_releases_its_dependent_in_round_two() -> None:
    dependency = create_rule("dependency")
    dependent = create_rule("dependent")
    fillers = [create_rule(f"filler {index}") for index in range(6)]
    rules = [dependent, dependency, *fillers]
    plan = await _plan(_scheduler([_relationship(dependent, dependency)]), rules)

    first_round = plan.next_round()
    assert plan.enabled
    assert set(first_round) == {dependency, *fillers}

    plan.observe(
        checked_rule_ids={rule.id for rule in first_round},
        relevant_rule_ids={dependency.id},
    )

    assert plan.next_round() == [dependent]


@pytest.mark.asyncio
async def test_that_an_irrelevant_dependency_skips_its_dependent() -> None:
    dependency = create_rule("dependency")
    dependent = create_rule("dependent")
    fillers = [create_rule(f"filler {index}") for index in range(6)]
    rules = [dependent, dependency, *fillers]
    plan = await _plan(_scheduler([_relationship(dependent, dependency)]), rules)

    first_round = plan.next_round()
    plan.observe(
        checked_rule_ids={rule.id for rule in first_round},
        relevant_rule_ids=set(),
    )

    assert plan.next_round() == []
    assert plan.skipped_rules() == [dependent]


@pytest.mark.asyncio
async def test_that_two_negative_sources_are_required_before_a_dependency_is_rejected() -> None:
    dependency = create_rule("dependency")
    dependent = create_rule("dependent")
    fillers = [create_rule(f"filler {index}") for index in range(6)]
    rules = [dependent, dependency, *fillers]
    plan = await _plan(
        _scheduler([_relationship(dependent, dependency)], max_rounds=3),
        rules,
        pending={dependency.id: 2, **{rule.id: 1 for rule in fillers}},
    )

    plan.next_round()
    plan.observe(
        checked_rule_ids={dependency.id, *(rule.id for rule in fillers)},
        relevant_rule_ids=set(),
    )

    # One source still has an outstanding opinion about the dependency, so it
    # remains unknown rather than being rejected.
    assert plan.next_round() == []

    plan.observe(checked_rule_ids={dependency.id}, relevant_rule_ids=set())

    assert plan.next_round() == []
    assert plan.skipped_rules() == [dependent]


@pytest.mark.asyncio
async def test_that_positive_evidence_wins_over_a_later_negative_source() -> None:
    dependency = create_rule("dependency")
    dependent = create_rule("dependent")
    fillers = [create_rule(f"filler {index}") for index in range(6)]
    rules = [dependent, dependency, *fillers]
    plan = await _plan(
        _scheduler([_relationship(dependent, dependency)], max_rounds=3),
        rules,
        pending={dependency.id: 2, **{rule.id: 1 for rule in fillers}},
    )

    plan.next_round()
    plan.observe(
        checked_rule_ids={dependency.id, *(rule.id for rule in fillers)},
        relevant_rule_ids={dependency.id},
    )
    plan.observe(checked_rule_ids={dependency.id}, relevant_rule_ids=set())

    assert plan.next_round() == [dependent]


@pytest.mark.asyncio
async def test_that_the_final_round_broadly_evaluates_an_unresolved_deep_chain() -> None:
    leaf = create_rule("leaf")
    middle = create_rule("middle")
    root = create_rule("root")
    fillers = [create_rule(f"filler {index}") for index in range(5)]
    rules = [root, middle, leaf, *fillers]
    plan = await _plan(
        _scheduler(
            [
                _relationship(root, middle),
                _relationship(middle, leaf),
            ]
        ),
        rules,
    )

    first_round = plan.next_round()
    assert set(first_round) == {leaf, *fillers}
    plan.observe(
        checked_rule_ids={rule.id for rule in first_round},
        relevant_rule_ids={leaf.id},
    )

    assert set(plan.next_round()) == {middle, root}


@pytest.mark.asyncio
async def test_that_three_rounds_walk_a_three_level_chain_in_dependency_order() -> None:
    leaf = create_rule("leaf")
    middle = create_rule("middle")
    root = create_rule("root")
    fillers = [create_rule(f"filler {index}") for index in range(5)]
    rules = [root, middle, leaf, *fillers]
    plan = await _plan(
        _scheduler(
            [
                _relationship(root, middle),
                _relationship(middle, leaf),
            ],
            max_rounds=3,
        ),
        rules,
    )

    first_round = plan.next_round()
    assert set(first_round) == {leaf, *fillers}
    plan.observe(
        checked_rule_ids={rule.id for rule in first_round},
        relevant_rule_ids={leaf.id},
    )

    assert plan.next_round() == [middle]
    plan.observe(checked_rule_ids={middle.id}, relevant_rule_ids={middle.id})

    assert plan.next_round() == [root]


@pytest.mark.asyncio
async def test_that_a_dependency_cycle_is_evaluated_in_the_broad_final_round() -> None:
    first = create_rule("first")
    second = create_rule("second")
    fillers = [create_rule(f"filler {index}") for index in range(6)]
    rules = [first, second, *fillers]
    plan = await _plan(
        _scheduler(
            [
                _relationship(first, second),
                _relationship(second, first),
            ]
        ),
        rules,
    )

    assert set(plan.next_round()) == set(fillers)
    plan.observe(
        checked_rule_ids={rule.id for rule in fillers},
        relevant_rule_ids=set(),
    )

    assert set(plan.next_round()) == {first, second}
    assert not plan.can_start_round
    assert plan.next_round() == []


@pytest.mark.asyncio
async def test_that_a_session_dependency_unlocks_its_dependent_in_round_one() -> None:
    dependency = create_rule("dependency")
    dependent = create_rule("dependent")
    fillers = [create_rule(f"filler {index}") for index in range(6)]
    rules = [dependent, dependency, *fillers]
    plan = await _plan(
        _scheduler([_relationship(dependent, dependency)]),
        rules,
        satisfied={dependency.id},
    )

    assert set(plan.next_round()) == set(rules)


@pytest.mark.asyncio
async def test_that_a_session_dependency_is_not_rechecked_before_unlocking_a_dependent() -> None:
    leaf = create_rule("leaf")
    session_dependency = create_rule("session dependency")
    dependent = create_rule("dependent")
    fillers = [create_rule(f"filler {index}") for index in range(5)]
    rules = [dependent, session_dependency, leaf, *fillers]
    plan = await _plan(
        _scheduler(
            [
                _relationship(dependent, session_dependency),
                _relationship(session_dependency, leaf),
            ]
        ),
        rules,
        satisfied={session_dependency.id},
    )

    assert dependent in plan.next_round()


@pytest.mark.asyncio
async def test_that_an_initially_relevant_rule_must_still_satisfy_its_own_dependency() -> None:
    leaf = create_rule("leaf")
    relevant_dependency = create_rule("relevant dependency")
    dependent = create_rule("dependent")
    fillers = [create_rule(f"filler {index}") for index in range(5)]
    rules = [dependent, relevant_dependency, leaf, *fillers]
    plan = await _plan(
        _scheduler(
            [
                _relationship(dependent, relevant_dependency),
                _relationship(relevant_dependency, leaf),
            ]
        ),
        rules,
        relevant={relevant_dependency.id},
    )

    first_round = plan.next_round()
    assert dependent not in first_round
    plan.observe(
        checked_rule_ids={rule.id for rule in first_round},
        relevant_rule_ids=set(),
    )

    assert plan.next_round() == []
    assert set(plan.skipped_rules()) == {dependent, relevant_dependency}


@pytest.mark.asyncio
async def test_that_an_initially_relevant_rule_becomes_effective_after_its_dependency_matches() -> (
    None
):
    leaf = create_rule("leaf")
    relevant_dependency = create_rule("relevant dependency")
    dependent = create_rule("dependent")
    fillers = [create_rule(f"filler {index}") for index in range(5)]
    rules = [dependent, relevant_dependency, leaf, *fillers]
    plan = await _plan(
        _scheduler(
            [
                _relationship(dependent, relevant_dependency),
                _relationship(relevant_dependency, leaf),
            ]
        ),
        rules,
        relevant={relevant_dependency.id},
    )

    first_round = plan.next_round()
    plan.observe(
        checked_rule_ids={rule.id for rule in first_round},
        relevant_rule_ids={leaf.id},
    )

    assert set(plan.next_round()) == {relevant_dependency, dependent}


@pytest.mark.asyncio
async def test_that_an_initially_satisfied_rule_overrides_later_negative_evidence() -> None:
    dependency = create_rule("dependency")
    dependent = create_rule("dependent")
    fillers = [create_rule(f"filler {index}") for index in range(6)]
    rules = [dependent, dependency, *fillers]
    plan = await _plan(
        _scheduler([_relationship(dependent, dependency)]),
        rules,
        satisfied={dependency.id},
    )

    plan.observe(checked_rule_ids={dependency.id}, relevant_rule_ids=set())

    assert dependent in plan.next_round()


@pytest.mark.asyncio
async def test_that_dependency_any_releases_a_rule_when_one_alternative_is_relevant() -> None:
    first = create_rule("first alternative")
    second = create_rule("second alternative")
    dependent = create_rule("dependent")
    fillers = [create_rule(f"filler {index}") for index in range(5)]
    rules = [dependent, first, second, *fillers]
    relationships = [
        _relationship(
            dependent,
            first,
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id="alternatives",
        ),
        _relationship(
            dependent,
            second,
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id="alternatives",
        ),
    ]
    plan = await _plan(_scheduler(relationships), rules)

    first_round = plan.next_round()
    plan.observe(
        checked_rule_ids={rule.id for rule in first_round},
        relevant_rule_ids={second.id},
    )

    assert plan.next_round() == [dependent]


@pytest.mark.asyncio
async def test_that_separate_dependency_any_groups_must_each_be_satisfied() -> None:
    first_a = create_rule("first a")
    first_b = create_rule("first b")
    second_a = create_rule("second a")
    second_b = create_rule("second b")
    dependent = create_rule("dependent")
    fillers = [create_rule(f"filler {index}") for index in range(3)]
    rules = [dependent, first_a, first_b, second_a, second_b, *fillers]
    relationships = [
        _relationship(
            dependent,
            first_a,
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id="first",
        ),
        _relationship(
            dependent,
            first_b,
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id="first",
        ),
        _relationship(
            dependent,
            second_a,
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id="second",
        ),
        _relationship(
            dependent,
            second_b,
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id="second",
        ),
    ]
    plan = await _plan(_scheduler(relationships), rules)

    first_round = plan.next_round()
    plan.observe(
        checked_rule_ids={rule.id for rule in first_round},
        relevant_rule_ids={first_a.id},
    )

    assert plan.next_round() == []
    assert plan.skipped_rules() == [dependent]


@pytest.mark.asyncio
async def test_that_multiple_ordinary_dependencies_are_all_required() -> None:
    first = create_rule("first")
    second = create_rule("second")
    dependent = create_rule("dependent")
    fillers = [create_rule(f"filler {index}") for index in range(5)]
    rules = [dependent, first, second, *fillers]
    plan = await _plan(
        _scheduler(
            [
                _relationship(dependent, first),
                _relationship(dependent, second),
            ]
        ),
        rules,
    )

    first_round = plan.next_round()
    plan.observe(
        checked_rule_ids={rule.id for rule in first_round},
        relevant_rule_ids={first.id},
    )

    assert plan.next_round() == []
    assert plan.skipped_rules() == [dependent]


@pytest.mark.asyncio
async def test_that_group_all_and_group_any_targets_preserve_their_semantics() -> None:
    group_id = GroupId("prerequisites")
    first = create_rule("first", groups=[group_id])
    second = create_rule("second", groups=[group_id])
    all_dependent = create_rule("all dependent")
    any_dependent = create_rule("any dependent")
    fillers = [create_rule(f"filler {index}") for index in range(4)]
    rules = [all_dependent, any_dependent, first, second, *fillers]
    now = datetime.now(timezone.utc)
    relationships = [
        Relationship(
            id=RelationshipId("all"),
            creation_utc=now,
            modified_utc=now,
            source=RelationshipEntity(all_dependent.id, RelationshipEntityKind.RULE),
            target=RelationshipEntity(group_id, RelationshipEntityKind.GROUP_ALL),
            kind=RelationshipKind.DEPENDENCY,
        ),
        Relationship(
            id=RelationshipId("any"),
            creation_utc=now,
            modified_utc=now,
            source=RelationshipEntity(any_dependent.id, RelationshipEntityKind.RULE),
            target=RelationshipEntity(group_id, RelationshipEntityKind.GROUP_ANY),
            kind=RelationshipKind.DEPENDENCY,
        ),
    ]
    plan = await _plan(_scheduler(relationships), rules)

    first_round = plan.next_round()
    plan.observe(
        checked_rule_ids={rule.id for rule in first_round},
        relevant_rule_ids={first.id},
    )

    assert plan.next_round() == [any_dependent]
    assert plan.skipped_rules() == [all_dependent]


@pytest.mark.asyncio
async def test_that_dependencies_attached_to_a_source_group_apply_to_its_members() -> None:
    group_id = GroupId("dependent-group")
    dependency = create_rule("dependency")
    first = create_rule("first dependent", groups=[group_id])
    second = create_rule("second dependent", groups=[group_id])
    fillers = [create_rule(f"filler {index}") for index in range(5)]
    rules = [first, second, dependency, *fillers]
    now = datetime.now(timezone.utc)
    relationship = Relationship(
        id=RelationshipId("source-group"),
        creation_utc=now,
        modified_utc=now,
        source=RelationshipEntity(group_id, RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(dependency.id, RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )
    plan = await _plan(_scheduler([relationship]), rules)

    first_round = plan.next_round()
    assert set(first_round) == {dependency, *fillers}
    plan.observe(
        checked_rule_ids={rule.id for rule in first_round},
        relevant_rule_ids={dependency.id},
    )

    assert set(plan.next_round()) == {first, second}


@pytest.mark.asyncio
async def test_that_an_empty_target_group_is_unsatisfied() -> None:
    group_id = GroupId("empty")
    dependent = create_rule("dependent")
    fillers = [create_rule(f"filler {index}") for index in range(7)]
    rules = [dependent, *fillers]
    now = datetime.now(timezone.utc)
    relationship = Relationship(
        id=RelationshipId("empty-target"),
        creation_utc=now,
        modified_utc=now,
        source=RelationshipEntity(dependent.id, RelationshipEntityKind.RULE),
        target=RelationshipEntity(group_id, RelationshipEntityKind.GROUP_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )
    plan = await _plan(_scheduler([relationship]), rules)

    assert set(plan.next_round()) == set(fillers)
    plan.observe(
        checked_rule_ids={rule.id for rule in fillers},
        relevant_rule_ids=set(),
    )

    assert plan.next_round() == []
    assert plan.skipped_rules() == [dependent]


@pytest.mark.asyncio
async def test_that_a_custom_precondition_provider_can_schedule_evaluation_phases() -> None:
    prerequisite = create_rule("prerequisite")
    dependent = create_rule("dependent")
    fillers = [create_rule(f"filler {index}") for index in range(6)]
    rules = [dependent, prerequisite, *fillers]
    scheduler = EvaluationScheduler(
        logger=_Logger(),
        precondition_registry=EvaluationPreconditionRegistry(
            [_StaticProvider({dependent.id: [_RequiresRule(prerequisite.id)]})]
        ),
        policy=EvaluationSchedulingPolicy(),
    )
    plan = await _plan(scheduler, rules)

    first_round = plan.next_round()
    assert set(first_round) == {prerequisite, *fillers}
    plan.observe(
        checked_rule_ids={rule.id for rule in first_round},
        relevant_rule_ids={prerequisite.id},
    )

    assert plan.next_round() == [dependent]


@pytest.mark.asyncio
async def test_that_provider_failure_falls_back_to_one_round() -> None:
    logger = _Logger()
    scheduler = EvaluationScheduler(
        logger=logger,
        precondition_registry=EvaluationPreconditionRegistry([_FailingProvider()]),
        policy=EvaluationSchedulingPolicy(),
    )
    rules = [create_rule(f"rule {index}") for index in range(8)]

    plan = await _plan(scheduler, rules)

    assert not plan.enabled
    assert set(plan.next_round()) == set(rules)
    assert logger.warning_messages == [
        "Evaluation precondition preparation failed; using one pass: provider failed"
    ]


# --- Cross-referential candidates (policy-context deferral, landscape-gated) ----
#
# A rule flagged as referencing the agent's other policies is deferred to the
# final round ONLY in a large policy landscape (inventory above
# MAX_FULL_POLICY_CONTEXT_RULES): there its evaluation prompt uses the bounded
# discovered-set context, which the deferral keeps fresh — forcing phasing if
# the candidate count alone would not have enabled it (correctness, not
# optimization). In a small landscape the ranker stuffs the full inventory
# into the prompt, waiting buys nothing, and the provider emits no
# precondition at all. See matching/common.py for the full motivation.


def _cross_referential_rule(condition: str) -> Rule:
    return _replace(
        create_rule(condition=condition, action="act accordingly"),
        metadata={"references_other_policies": True},
    )


def _cross_reference_scheduler(
    *, threshold: int = 8, max_rounds: int = 2, logger: _Logger | None = None
) -> EvaluationScheduler:
    return EvaluationScheduler(
        logger=logger or _Logger(),
        precondition_registry=EvaluationPreconditionRegistry(
            [CrossReferenceEvaluationPreconditionProvider()]
        ),
        policy=EvaluationSchedulingPolicy(threshold=threshold, max_rounds=max_rounds),
    )


def _large_inventory_filler() -> list[Rule]:
    return [
        create_rule(condition=f"filler condition {i}", action=f"filler action {i}")
        for i in range(MAX_FULL_POLICY_CONTEXT_RULES)
    ]


@pytest.mark.asyncio
async def test_that_a_small_landscape_emits_no_deferral_and_stays_one_pass() -> None:
    flagged = _cross_referential_rule("the request is not covered by any of our policies")
    plain = create_rule(condition="customer asks about pricing", action="share prices")

    plan = await _plan(
        _cross_reference_scheduler(threshold=1),
        [flagged, plain],
    )

    assert not plan.enabled
    assert set(plan.next_round()) == {flagged, plain}


@pytest.mark.asyncio
async def test_that_a_large_landscape_defers_the_flagged_candidate_to_the_final_round() -> None:
    flagged = _cross_referential_rule("the request is not covered by any of our policies")
    plain = create_rule(condition="customer asks about pricing", action="share prices")
    inventory = [flagged, plain, *_large_inventory_filler()]

    plan = await _plan(
        _cross_reference_scheduler(threshold=2),
        inventory,
        candidates=[flagged, plain],
        pending={rule.id: 1 for rule in inventory},
    )

    assert plan.enabled

    first_round = plan.next_round()
    assert flagged not in first_round
    assert plain in first_round

    plan.observe({plain.id}, {plain.id})
    final_round = plan.next_round()
    assert flagged in final_round


@pytest.mark.asyncio
async def test_that_a_large_landscape_forces_phasing_below_the_threshold_with_a_debug_log() -> None:
    flagged = _cross_referential_rule("the request is not covered by any of our policies")
    plain = create_rule(condition="customer asks about pricing", action="share prices")
    inventory = [flagged, plain, *_large_inventory_filler()]

    forcing_logger = _Logger()
    plan = await _plan(
        _cross_reference_scheduler(logger=forcing_logger),  # threshold=8, 2 candidates
        inventory,
        candidates=[flagged, plain],
        pending={rule.id: 1 for rule in inventory},
    )

    assert plan.enabled
    assert any("forc" in message.lower() for message in forcing_logger.debug_messages)

    # Above the threshold the same setup phases anyway: no forcing, no log.
    quiet_logger = _Logger()
    plan = await _plan(
        _cross_reference_scheduler(threshold=2, logger=quiet_logger),
        inventory,
        candidates=[flagged, plain],
        pending={rule.id: 1 for rule in inventory},
    )
    assert plan.enabled
    assert not any("forc" in message.lower() for message in quiet_logger.debug_messages)


@pytest.mark.asyncio
async def test_that_a_flagged_non_candidate_does_not_force_phasing() -> None:
    # Forcing keys off CANDIDATES: a cross-referential rule that merely exists
    # in the inventory (e.g. already discovered, not being evaluated) must not
    # engage phasing for everyone else.
    flagged = _cross_referential_rule("the request is not covered by any of our policies")
    plain = create_rule(condition="customer asks about pricing", action="share prices")
    inventory = [flagged, plain, *_large_inventory_filler()]

    plan = await _plan(
        _cross_reference_scheduler(),
        inventory,
        candidates=[plain],
        pending={plain.id: 1},
    )

    assert not plan.enabled
    assert set(plan.next_round()) == {plain}


@pytest.mark.asyncio
async def test_that_forced_phasing_clamps_a_single_round_policy_to_two_rounds() -> None:
    # Forcing is correctness, not optimization: with max_rounds=1 the "final"
    # round would be round 1, evaluating the flagged rule alongside recall
    # with no deferral — the staleness the precondition exists to prevent.
    flagged = _cross_referential_rule("the request is not covered by any of our policies")
    plain = create_rule(condition="customer asks about pricing", action="share prices")
    inventory = [flagged, plain, *_large_inventory_filler()]

    plan = await _plan(
        _cross_reference_scheduler(threshold=2, max_rounds=1),
        inventory,
        candidates=[flagged, plain],
        pending={rule.id: 1 for rule in inventory},
    )

    first_round = plan.next_round()
    assert flagged not in first_round
    assert plain in first_round

    plan.observe({plain.id}, {plain.id})
    assert flagged in plan.next_round()
