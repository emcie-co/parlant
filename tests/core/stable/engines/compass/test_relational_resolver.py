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

from datetime import datetime, timezone

from parlant.core.container import Container

from parlant.core.common import Weight
from parlant.core.engines.rule_evaluation import RuleEvaluation
from parlant.core.engines.compass.relational_resolver import (
    RelationalResolver,
    RelationalResolverResult,
    Resolution,
    ResolutionKind,
    ResolvedEntity,
)
from parlant.core.rules import Rule, RuleId, RuleStore
from parlant.core.loggers import Logger
from parlant.core.relationships import (
    RelationshipEntity,
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.store_provider import StoreProvider
from parlant.core.groups import Group, GroupId, GroupStore
from parlant.core.tracer import Tracer


def make_resolver(container: Container) -> RelationalResolver:
    return RelationalResolver(
        store_provider=container[StoreProvider],
        logger=container[Logger],
        tracer=container[Tracer],
    )


def find_resolutions(
    result: RelationalResolverResult,
    target: Rule | Group | ResolvedEntity | RuleId | GroupId,
) -> list[Resolution]:
    if isinstance(target, ResolvedEntity):
        return result.resolutions.get(target, [])
    if isinstance(target, Rule):
        return result.resolutions.get(ResolvedEntity.rule(target), [])
    if isinstance(target, Group):
        return result.resolutions.get(ResolvedEntity.group(target), [])

    for entity, resolutions in result.resolutions.items():
        if entity.entity.id == target:
            return resolutions

    return []


def assert_resolutions(
    result: RelationalResolverResult,
    entity: Rule | Group | ResolvedEntity | RuleId | GroupId,
    expected_kinds: list[ResolutionKind],
) -> None:
    actual_kinds = [resolution.kind for resolution in find_resolutions(result, entity)]
    assert sorted(actual_kinds, key=lambda kind: kind.name) == sorted(
        expected_kinds,
        key=lambda kind: kind.name,
    )


def get_resolutions_by_kind(
    result: RelationalResolverResult,
    entity: Rule | Group | ResolvedEntity | RuleId | GroupId,
    kind: ResolutionKind,
) -> list[Resolution]:
    return [
        resolution for resolution in find_resolutions(result, entity) if resolution.kind == kind
    ]


async def test_that_compass_relational_resolver_filters_unmet_rule_dependency(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1")
    g2 = await rule_store.create_rule(condition="b", content="g2")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2],
        [RuleEvaluation(rule=g1, reasoning="")],
    )

    assert [match.rule.id for match in result.evaluations] == []
    assert_resolutions(result, g1, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_compass_relational_resolver_keeps_met_rule_dependency(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1")
    g2 = await rule_store.create_rule(condition="b", content="g2")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
        ],
    )

    assert {match.rule.id for match in result.evaluations} == {g1.id, g2.id}
    assert_resolutions(result, g1, [ResolutionKind.NONE])
    assert_resolutions(result, g2, [ResolutionKind.NONE])


async def test_that_compass_relational_resolver_supports_tag_all_dependency(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    group = await group_store.create_group(name="group-all")
    g1 = await rule_store.create_rule(condition="a", content="g1")
    g2 = await rule_store.create_rule(condition="b", content="g2", groups=[group.id])
    g3 = await rule_store.create_rule(condition="c", content="g3", groups=[group.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=group.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    missing_member = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
        ],
    )

    assert {match.rule.id for match in missing_member.evaluations} == {g2.id}
    assert_resolutions(missing_member, g1, [ResolutionKind.UNMET_DEPENDENCY_ALL])

    all_members = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            RuleEvaluation(rule=g3, reasoning=""),
        ],
    )

    assert {match.rule.id for match in all_members.evaluations} == {g1.id, g2.id, g3.id}
    assert_resolutions(all_members, g1, [ResolutionKind.NONE])


async def test_that_compass_relational_resolver_supports_tag_any_dependency(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    group = await group_store.create_group(name="group-any")
    g1 = await rule_store.create_rule(condition="a", content="g1")
    g2 = await rule_store.create_rule(condition="b", content="g2", groups=[group.id])
    g3 = await rule_store.create_rule(condition="c", content="g3", groups=[group.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=group.id, kind=RelationshipEntityKind.GROUP_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    one_member = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
        ],
    )

    assert {match.rule.id for match in one_member.evaluations} == {g1.id, g2.id}
    assert_resolutions(one_member, g1, [ResolutionKind.NONE])

    no_members = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [RuleEvaluation(rule=g1, reasoning="")],
    )

    assert [match.rule.id for match in no_members.evaluations] == []
    assert_resolutions(no_members, g1, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_compass_relational_resolver_supports_dependency_any_groups(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1")
    g2 = await rule_store.create_rule(condition="b", content="g2")
    g3 = await rule_store.create_rule(condition="c", content="g3")

    for target in [g2, g3]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.RULE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id="any-group",
        )

    one_target = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
        ],
    )

    assert {match.rule.id for match in one_target.evaluations} == {g1.id, g2.id}
    assert_resolutions(one_target, g1, [ResolutionKind.NONE])

    no_targets = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [RuleEvaluation(rule=g1, reasoning="")],
    )

    assert [match.rule.id for match in no_targets.evaluations] == []
    assert_resolutions(no_targets, g1, [ResolutionKind.UNMET_DEPENDENCY_ANY])


async def test_that_compass_relational_resolver_applies_dependencies_from_source_groups(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    group = await group_store.create_group(name="source-group")
    grouped = await rule_store.create_rule(
        condition="a",
        content="grouped",
        groups=[group.id],
    )
    dependency = await rule_store.create_rule(condition="b", content="dependency")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=group.id, kind=RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(id=dependency.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    unmet = await resolver.filter_unmet_dependencies(
        [grouped, dependency],
        [RuleEvaluation(rule=grouped, reasoning="")],
    )

    assert [match.rule.id for match in unmet.evaluations] == []
    assert_resolutions(unmet, grouped, [ResolutionKind.UNMET_DEPENDENCY_ALL])

    met = await resolver.filter_unmet_dependencies(
        [grouped, dependency],
        [
            RuleEvaluation(rule=grouped, reasoning=""),
            RuleEvaluation(rule=dependency, reasoning=""),
        ],
    )

    assert {match.rule.id for match in met.evaluations} == {grouped.id, dependency.id}
    assert_resolutions(met, grouped, [ResolutionKind.NONE])


async def test_that_compass_relational_resolver_cascades_dependency_chains(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    root_dependent = await rule_store.create_rule(condition="a", content="root")
    middle = await rule_store.create_rule(condition="b", content="middle")
    missing_leaf = await rule_store.create_rule(condition="c", content="leaf")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=root_dependent.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=middle.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=middle.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=missing_leaf.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [root_dependent, middle, missing_leaf],
        [
            RuleEvaluation(rule=root_dependent, reasoning=""),
            RuleEvaluation(rule=middle, reasoning=""),
        ],
    )

    assert [match.rule.id for match in result.evaluations] == []
    assert_resolutions(result, root_dependent, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, middle, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_compass_relational_resolver_applies_numeric_priority(
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    low = await rule_store.create_rule(condition="a", content="low", priority=0)
    high = await rule_store.create_rule(condition="b", content="high", priority=10)

    result = resolver.keep_highest_priority(
        [
            RuleEvaluation(rule=low, reasoning=""),
            RuleEvaluation(rule=high, reasoning=""),
        ],
    )

    assert [match.rule.id for match in result.evaluations] == [high.id]
    assert_resolutions(result, low, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, high, [ResolutionKind.NONE])

    [resolution] = get_resolutions_by_kind(result, low, ResolutionKind.DEPRIORITIZED)
    assert resolution.details.counterparts == (ResolvedEntity.rule(high),)
    assert resolution.details.relationship is None


async def test_that_priority_filtering_is_decoupled_from_dependency_filtering(
    container: Container,
) -> None:
    # The matcher gates dependencies over ALL matches first, then applies
    # priority to the turn-matched subset only — and a deprioritized rule still
    # satisfies its dependents (it remains in the session working set). The old
    # coupled resolve() cascaded priority drops into dependency failures; the
    # split API deliberately does not.
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    dependent = await rule_store.create_rule(
        condition="a",
        content="dependent",
        priority=100,
    )
    dependency = await rule_store.create_rule(
        condition="b",
        content="dependency",
        priority=0,
    )
    winner = await rule_store.create_rule(
        condition="c",
        content="winner",
        priority=100,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=dependent.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=dependency.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    matches = [
        RuleEvaluation(rule=dependent, reasoning=""),
        RuleEvaluation(rule=dependency, reasoning=""),
        RuleEvaluation(rule=winner, reasoning=""),
    ]

    dependency_result = await resolver.filter_unmet_dependencies(
        [dependent, dependency, winner],
        matches,
    )
    assert {match.rule.id for match in dependency_result.evaluations} == {
        dependent.id,
        dependency.id,
        winner.id,
    }

    priority_result = resolver.keep_highest_priority(list(dependency_result.evaluations))
    assert {match.rule.id for match in priority_result.evaluations} == {dependent.id, winner.id}
    assert_resolutions(priority_result, dependency, [ResolutionKind.DEPRIORITIZED])


async def test_that_satisfied_rule_ids_meet_dependencies_without_a_fresh_match(
    container: Container,
) -> None:
    # The "met at least once" contract: a dependency satisfied by a session
    # working-set member unlocks its dependents even when the dependency has no
    # match this turn.
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1")
    g2 = await rule_store.create_rule(condition="b", content="g2")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    unmet = await resolver.filter_unmet_dependencies(
        [g1, g2],
        [RuleEvaluation(rule=g1, reasoning="")],
    )
    assert [match.rule.id for match in unmet.evaluations] == []

    met = await resolver.filter_unmet_dependencies(
        [g1, g2],
        [RuleEvaluation(rule=g1, reasoning="")],
        satisfied_rule_ids={g2.id},
    )
    assert [match.rule.id for match in met.evaluations] == [g1.id]
    assert_resolutions(met, g1, [ResolutionKind.NONE])


async def test_that_satisfied_rule_ids_count_toward_group_dependencies(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    group = await group_store.create_group(name="group-all")
    g1 = await rule_store.create_rule(condition="a", content="g1")
    g2 = await rule_store.create_rule(condition="b", content="g2", groups=[group.id])
    g3 = await rule_store.create_rule(condition="c", content="g3", groups=[group.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=group.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    # One member matched this turn, the other already in the session set:
    # the ALL-group dependency is met.
    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
        ],
        satisfied_rule_ids={g3.id},
    )

    assert {match.rule.id for match in result.evaluations} == {g1.id, g2.id}
    assert_resolutions(result, g1, [ResolutionKind.NONE])


# ---------------------------------------------------------------------------
# Ported from the alpha engine's relational resolver suite (dependencies).
# Alpha's combined resolve() maps onto filter_unmet_dependencies here;
# journey-specific scenarios were dropped with the journeys feature.
# ---------------------------------------------------------------------------


async def test_that_any_semantics_filters_when_all_tag_members_cascade_fail(
    container: Container,
) -> None:
    """
    G1 depends on T1 (ANY). T1 has G2 and G3.
    G2 depends on G4, G3 depends on G5. Neither G4 nor G5 matched.
    → G2 filtered, G3 filtered → T1 has no active members → G1 filtered.
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t1.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t1.id])
    g4 = await rule_store.create_rule(condition="d", content="g4 action")
    g5 = await rule_store.create_rule(condition="e", content="g5 action")

    # G1 depends on T1 (ANY)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G2 depends on G4
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G3 depends on G5
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g5.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3, g4, g5],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            RuleEvaluation(rule=g3, reasoning=""),
            # G4 NOT matched
            # G5 NOT matched
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == set()

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g3.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_any_semantics_survives_cascading_failure_within_tag_member(
    container: Container,
) -> None:
    """
    G1 depends on T1 (ANY). T1 has G2 and G3.
    G2 depends on G4, G4 NOT matched → G2 is filtered.
    But G3 is still matched → G1 survives (ANY: G3 is active under T1).
    Result: {G1, G3}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t1.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t1.id])
    g4 = await rule_store.create_rule(condition="d", content="g4 action")

    # G1 depends on T1 (ANY)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G2 depends on G4
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3, g4],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            RuleEvaluation(rule=g3, reasoning=""),
            # G4 NOT matched → G2 will be filtered
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g3.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_dependency_any_group_filters_when_no_target_is_matched(
    container: Container,
) -> None:
    """
    G1 depends_on_any(G2, G3). Neither matched.
    OR group: none met → G1 filtered.
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")

    group_id = "test-group-1"
    group_rel_ids = []
    for target in [g2, g3]:
        rel = await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.RULE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )
        group_rel_ids.append(rel.id)

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            # G2 NOT matched
            # G3 NOT matched
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == set()

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ANY])

    # The resolution should list both OR-group targets as counterparts and
    # reference one of the dependency relationships that contributed to the
    # failure.
    g1_res = get_resolutions_by_kind(result, g1.id, ResolutionKind.UNMET_DEPENDENCY_ANY)
    assert len(g1_res) == 1
    assert set(g1_res[0].details.counterparts) == {
        ResolvedEntity.rule(g2),
        ResolvedEntity.rule(g3),
    }
    assert g1_res[0].details.relationship is not None
    assert g1_res[0].details.relationship.id in group_rel_ids


async def test_that_dependency_any_group_survives_when_one_target_cascading_fails_but_another_survives(
    container: Container,
) -> None:
    """
    G1 depend_on_any(G2, G3). G2 depends on G4. G4 NOT matched.
    G2 filtered by its own dep. G3 matched → OR group passes via G3.
    Result: {G1, G3}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")
    g4 = await rule_store.create_rule(condition="d", content="g4 action")

    # G1 depend_on_any(G2, G3)
    group_id = "test-cascade-group"
    for target in [g2, g3]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.RULE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )

    # G2 depends on G4 (AND)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3, g4],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            RuleEvaluation(rule=g3, reasoning=""),
            # G4 NOT matched → G2 will be filtered
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_dependency_any_group_survives_when_one_target_is_matched(
    container: Container,
) -> None:
    """
    G1 depends_on_any(G2, G3). G2 matched, G3 NOT matched.
    OR group: any one met → G1 survives.
    Result: {G1, G2}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")

    group_id = "test-group-1"
    for target in [g2, g3]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.RULE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            # G3 NOT matched
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g2.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_dependency_any_group_with_tag_all_target_falls_back_to_rule_target(
    container: Container,
) -> None:
    """
    G1 depend_on_any(AllOf(group=T1), G4).
    T1 has G2 and G3. Only G2 matched → AllOf(T1) fails.
    G4 matched → OR group passes via G4.
    Result: {G1, G2, G4}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t1.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t1.id])
    g4 = await rule_store.create_rule(condition="d", content="g4 action")

    group_id = "test-group-group"

    # AllOf(T1) target
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id=group_id,
    )

    # G4 target
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id=group_id,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3, g4],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            # G3 NOT matched → AllOf(T1) fails
            RuleEvaluation(rule=g4, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g2.id, g4.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g4.id, [ResolutionKind.NONE])


async def test_that_dependency_any_group_with_tag_all_target_succeeds_when_all_members_matched(
    container: Container,
) -> None:
    """
    G1 depend_on_any(AllOf(group=T1), G4).
    T1 has G2 and G3, both matched → AllOf(T1) succeeds.
    OR group met via the group target (G4 not needed).
    Result: {G1, G2, G3}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t1.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t1.id])
    g4 = await rule_store.create_rule(condition="d", content="g4 action")

    group_id = "group-all-success-group"

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id=group_id,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id=group_id,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3, g4],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            RuleEvaluation(rule=g3, reasoning=""),
            # G4 NOT matched — but AllOf(T1) succeeds so OR group is met
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g2.id, g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


# ── Resolution attribution edge-case tests ────────────────────────────────


async def test_that_diamond_dependency_filters_all_dependents_when_root_is_not_matched(
    container: Container,
) -> None:
    """
    Diamond: G2 depends on G4, G3 depends on G4, G1 depends on both G2 and G3.
    G4 NOT matched → G2 filtered, G3 filtered → G1 filtered (both deps unmet).
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")
    g4 = await rule_store.create_rule(condition="d", content="g4 action")

    # G2 depends on G4
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G3 depends on G4
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G1 depends on G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G1 depends on G3
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3, g4],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            RuleEvaluation(rule=g3, reasoning=""),
            # G4 NOT matched
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == set()

    # Resolution assertions
    # G1 has two direct deps (G2 and G3) — both unmet
    assert_resolutions(
        result,
        g1.id,
        [
            ResolutionKind.UNMET_DEPENDENCY_ALL,
            ResolutionKind.UNMET_DEPENDENCY_ALL,
        ],
    )
    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g3.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_diamond_dependency_keeps_all_when_root_is_matched(
    container: Container,
) -> None:
    """
    Diamond: G2 depends on G4, G3 depends on G4, G1 depends on both G2 and G3.
    All matched → all survive.
    Result: {G1, G2, G3, G4}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")
    g4 = await rule_store.create_rule(condition="d", content="g4 action")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3, g4],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            RuleEvaluation(rule=g3, reasoning=""),
            RuleEvaluation(rule=g4, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g2.id, g3.id, g4.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])
    assert_resolutions(result, g4.id, [ResolutionKind.NONE])


async def test_that_empty_tag_dependency_filters_dependent_rule(
    container: Container,
) -> None:
    """
    G1 depends on T1. T1 has no grouped members at all.
    any([]) is False → G1 should be filtered.
    Result: {g_extra}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="empty-group")

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g_extra = await rule_store.create_rule(condition="b", content="extra action")

    # G1 depends on T1 (which has no members)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g_extra],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g_extra, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g_extra.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g_extra.id, [ResolutionKind.NONE])


async def test_that_rule_depending_on_tag_is_filtered_when_no_tagged_rule_is_matched(
    container: Container,
) -> None:
    """
    g1 depends on group t1. t1 has g2 and g3, neither matched.
    0 of 2 matched → g1 filtered.
    Result: {g_extra}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t1.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t1.id])
    g_extra = await rule_store.create_rule(condition="d", content="extra action")

    # g1 depends on t1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3, g_extra],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            # g2 NOT matched
            # g3 NOT matched
            RuleEvaluation(rule=g_extra, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g_extra.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g_extra.id, [ResolutionKind.NONE])


async def test_that_rule_depending_on_tag_survives_when_at_least_one_tagged_rule_is_matched(
    container: Container,
) -> None:
    """
    g1 depends on group t1. t1 has g2 and g3, only g2 matched.
    1 of 2 matched → g1 survives (ANY semantics).
    Result: {g1, g2}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t1.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t1.id])

    # g1 depends on t1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            # g3 NOT matched
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g2.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_hierarchical_rule_dependency_cascades_when_middle_is_not_matched(
    container: Container,
) -> None:
    """
    G2 depends on G1, G3 depends on G2.
    G1 matched, G2 NOT matched → G3 filtered, G1 survives.
    Result: {G1}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")

    # G2 depends on G1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G3 depends on G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            # G2 NOT matched
            RuleEvaluation(rule=g3, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_hierarchical_rule_dependency_cascades_when_root_is_not_matched(
    container: Container,
) -> None:
    """
    G2 depends on G1, G3 depends on G2.
    G1 NOT matched → G2 filtered, G3 filtered (cascade).
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")

    # G2 depends on G1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G3 depends on G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            # G1 NOT matched
            RuleEvaluation(rule=g2, reasoning=""),
            RuleEvaluation(rule=g3, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == set()

    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(
        result,
        g3.id,
        [ResolutionKind.UNMET_DEPENDENCY_ALL],
    )


async def test_that_hierarchical_tag_dependency_cascades_when_middle_tag_member_is_not_matched(
    container: Container,
) -> None:
    """
    T2 depends on T1, T3 depends on T2.
    T1's rule matched, T2's rule NOT matched → T3's rule filtered, T1's survives.
    Result: {g1}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")
    t2 = await group_store.create_group(name="t2")
    t3 = await group_store.create_group(name="t3")

    g1 = await rule_store.create_rule(condition="a", content="g1 action", groups=[t1.id])
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t2.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t3.id])

    # T2 depends on T1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    # T3 depends on T2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t3.id, kind=RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            # g2 NOT matched (T2 unmet)
            RuleEvaluation(rule=g3, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_hierarchical_tag_dependency_cascades_when_root_tag_member_is_not_matched(
    container: Container,
) -> None:
    """
    T2 depends on T1, T3 depends on T2.
    T1's rule NOT matched → T2's rule filtered, T3's rule filtered.
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")
    t2 = await group_store.create_group(name="t2")
    t3 = await group_store.create_group(name="t3")

    g1 = await rule_store.create_rule(condition="a", content="g1 action", groups=[t1.id])
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t2.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t3.id])

    # T2 depends on T1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    # T3 depends on T2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t3.id, kind=RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            # g1 NOT matched (T1 unmet)
            RuleEvaluation(rule=g2, reasoning=""),
            RuleEvaluation(rule=g3, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == set()

    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(
        result,
        g3.id,
        [ResolutionKind.UNMET_DEPENDENCY_ALL],
    )


async def test_that_mixed_dependency_all_and_dependency_any_filters_when_and_dep_unmet(
    container: Container,
) -> None:
    """
    G1 depend_on(G2) AND depend_on_any(G3, G4).
    G2 NOT matched. G4 matched.
    AND dep unmet → G1 filtered even though OR group is met.
    Result: {G4}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")
    g4 = await rule_store.create_rule(condition="d", content="g4 action")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    group_id = "test-group-1"
    for target in [g3, g4]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.RULE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3, g4],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            # G2 NOT matched
            # G3 NOT matched
            RuleEvaluation(rule=g4, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g4.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g4.id, [ResolutionKind.NONE])


async def test_that_mixed_dependency_all_and_dependency_any_requires_both(
    container: Container,
) -> None:
    """
    G1 depend_on(G2) AND depend_on_any(G3, G4).
    G2 matched, G3 NOT matched, G4 matched.
    AND: G2 met. OR group: G4 met. Both met → G1 survives.
    Result: {G1, G2, G4}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")
    g4 = await rule_store.create_rule(condition="d", content="g4 action")

    # G1 depends on G2 (AND)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G1 depends_on_any(G3, G4) (OR)
    group_id = "test-group-1"
    for target in [g3, g4]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.RULE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3, g4],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            # G3 NOT matched
            RuleEvaluation(rule=g4, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g2.id, g4.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g4.id, [ResolutionKind.NONE])


async def test_that_mixed_tag_all_and_tag_any_dependencies_both_evaluated(
    container: Container,
) -> None:
    """
    G1 depends on GROUP_ALL(t1) AND GROUP_ANY(t2).
    t1 has G2, G3 (both matched). t2 has G4, G5 (only G4 matched).
    GROUP_ALL(t1) met, GROUP_ANY(t2) met → G1 survives.
    Result: {G1, G2, G3, G4}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")
    t2 = await group_store.create_group(name="t2")

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t1.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t1.id])
    g4 = await rule_store.create_rule(condition="d", content="g4 action", groups=[t2.id])
    g5 = await rule_store.create_rule(condition="e", content="g5 action", groups=[t2.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.GROUP_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3, g4, g5],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            RuleEvaluation(rule=g3, reasoning=""),
            RuleEvaluation(rule=g4, reasoning=""),
            # G5 NOT matched
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g2.id, g3.id, g4.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])
    assert_resolutions(result, g4.id, [ResolutionKind.NONE])


async def test_that_mixed_tag_all_and_tag_any_filters_when_tag_all_not_fully_met(
    container: Container,
) -> None:
    """
    G1 depends on GROUP_ALL(t1) AND GROUP_ANY(t2).
    t1 has G2, G3 (only G2 matched). t2 has G4, G5 (G4 matched).
    GROUP_ALL(t1) NOT met (G3 missing) → G1 filtered, even though GROUP_ANY(t2) is met.
    Result: {G2, G4}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")
    t2 = await group_store.create_group(name="t2")

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t1.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t1.id])
    g4 = await rule_store.create_rule(condition="d", content="g4 action", groups=[t2.id])
    g5 = await rule_store.create_rule(condition="e", content="g5 action", groups=[t2.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.GROUP_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3, g4, g5],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            # G3 NOT matched
            RuleEvaluation(rule=g4, reasoning=""),
            # G5 NOT matched
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g2.id, g4.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g4.id, [ResolutionKind.NONE])


# ── DEPENDENCY_ANY (OR group) tests ────────────────────────────────────────


async def test_that_multi_iteration_convergence_filters_dependent_when_transitive_dependency_fails(
    container: Container,
) -> None:
    """
    G4 depends on G2 and G3 (individually). G3 depends on G1.
    G2, G3, G4 all matched. G1 NOT matched.

    Iteration 1: G3 is filtered (depends on G1). G4 survives because G3 is
    still in the initial matched_rule_ids snapshot.
    Iteration 2: G4 is filtered (G3 no longer in matches).

    Result: {G2}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")
    g4 = await rule_store.create_rule(condition="d", content="g4 action")

    # G4 depends on G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G4 depends on G3
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G3 depends on G1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3, g4],
        [
            # G1 NOT matched
            RuleEvaluation(rule=g2, reasoning=""),
            RuleEvaluation(rule=g3, reasoning=""),
            RuleEvaluation(rule=g4, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g2.id}

    # Resolution assertions
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    # G4 depends on G2 (met) and G3 (filtered), plus transitive dep on G1
    assert_resolutions(
        result,
        g4.id,
        [ResolutionKind.UNMET_DEPENDENCY_ALL],
    )


# ── Numerical priority + dependency interaction tests ──────────────────────


async def test_that_multiple_independent_dependencies_must_all_be_met(
    container: Container,
) -> None:
    """
    G1 depends on G2 (direct) AND on T1 (group).
    G2 matched, T1 member NOT matched → G1 filtered (both must be met).
    Result: {G2}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t1.id])

    # G1 depends on G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G1 depends on T1
    rel_t1 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            # g3 NOT matched (T1 unmet)
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g2.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])

    # The single resolution should specifically identify T1 as the unmet target
    # and reference the dependency relationship that caused the failure.
    g1_res = get_resolutions_by_kind(result, g1.id, ResolutionKind.UNMET_DEPENDENCY_ALL)
    assert len(g1_res) == 1
    assert ResolvedEntity.group(t1) in g1_res[0].details.counterparts
    assert g1_res[0].details.relationship is not None
    assert g1_res[0].details.relationship.id == rel_t1.id


async def test_that_multiple_independent_dependencies_survive_when_all_met(
    container: Container,
) -> None:
    """
    G1 depends on G2 (direct) AND on T1 (group).
    G2 matched, T1 member matched → G1 survives (both met).
    Result: {G1, G2, G3}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t1.id])

    # G1 depends on G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G1 depends on T1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            RuleEvaluation(rule=g3, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g2.id, g3.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_relational_resolver_filters_out_rules_with_unmet_dependencies(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    source_rule = await rule_store.create_rule(
        condition="Customer has not specified if it's a repeat transaction or a new one",
        content="Ask them which it is",
    )
    target_rule = await rule_store.create_rule(
        condition="Customer wants to make a transaction", content="Help them"
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=source_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=target_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [source_rule, target_rule],
        [
            RuleEvaluation(rule=source_rule, reasoning=""),
        ],
    )

    assert result.evaluations == []

    assert_resolutions(result, source_rule.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_relational_resolver_filters_tagged_rule_when_custom_group_dependency_is_unmet(
    container: Container,
) -> None:
    """
    Tests that a custom group used as source in a DEPENDENCY relationship
    deactivates the grouped rule when the dependency target is not matched.

    - Group t1 is attached to g1
    - t1 DEPENDENCY → g2  (g1, via t1, depends on g2)
    - g1 is matched but g2 is NOT matched
    - Expected: g1 is filtered out (unmet dependency via group)
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="b")
    g2 = await rule_store.create_rule(condition="c", content="d")

    t1 = await group_store.create_group(name="t1")
    await rule_store.upsert_group(g1.id, t1.id)
    g1 = await rule_store.read_rule(g1.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            # g2 is NOT matched — dependency unmet
        ],
    )

    assert result.evaluations == []

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_relational_resolver_handles_indirect_rules_from_tags(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="x", content="y")
    g2 = await rule_store.create_rule(condition="y", content="z")
    g3 = await rule_store.create_rule(condition="z", content="t")

    t1 = await group_store.create_group(name="t1")

    await rule_store.upsert_group(g2.id, t1.id)

    # Re-read after tagging so rule.groups is up-to-date
    g2 = await rule_store.read_rule(g2.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g1.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.GROUP_ALL,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.GROUP_ALL,
        ),
        target=RelationshipEntity(
            id=g3.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve_priority_relationships(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g3, reasoning=""),
        ],
    )

    assert len(result.evaluations) == 1
    assert result.evaluations[0].rule.id == g1.id

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.DEPRIORITIZED])


async def test_that_relational_resolver_keeps_rule_depending_on_tag_when_at_least_one_tagged_member_is_matched(
    container: Container,
) -> None:
    """
    Group dependency uses ANY semantics: a rule depending on a group survives
    as long as at least one grouped member is matched.

    - source_rule depends on target_group
    - target_group has tagged_rule_1 and tagged_rule_2
    - Only tagged_rule_1 is matched
    - Expected: source_rule survives (ANY member matched)
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    source_rule = await rule_store.create_rule(condition="a", content="b")

    tagged_rule_1 = await rule_store.create_rule(condition="c", content="d")
    tagged_rule_2 = await rule_store.create_rule(condition="e", content="f")

    target_group = await group_store.create_group(name="t1")

    await rule_store.upsert_group(tagged_rule_1.id, target_group.id)
    await rule_store.upsert_group(tagged_rule_2.id, target_group.id)

    # Re-read after tagging so usable_rules has up-to-date group lists
    tagged_rule_1 = await rule_store.read_rule(tagged_rule_1.id)
    tagged_rule_2 = await rule_store.read_rule(tagged_rule_2.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=source_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=target_group.id,
            kind=RelationshipEntityKind.GROUP_ANY,
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [source_rule, tagged_rule_1, tagged_rule_2],
        [
            RuleEvaluation(rule=source_rule, reasoning=""),
            RuleEvaluation(rule=tagged_rule_1, reasoning=""),
            # Missing match for tagged_rule_2
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {source_rule.id, tagged_rule_1.id}

    assert_resolutions(result, source_rule.id, [ResolutionKind.NONE])
    assert_resolutions(result, tagged_rule_1.id, [ResolutionKind.NONE])


async def test_that_resolver_does_not_crash_on_transient_tool_returned_rule(
    container: Container,
) -> None:
    """Tool calls can inject rules that don't exist in the store.

    Such rules arrive only via ``matches``, never via
    ``usable_rules``. The resolver must include them in its internal
    rule-by-id map so downstream lookups don't KeyError.
    """
    resolver = make_resolver(container)

    transient_rule = Rule(
        id=RuleId("<tool-rule-1>"),
        creation_utc=datetime.now(timezone.utc),
        modified_utc=datetime.now(timezone.utc),
        condition="x",
        content="y",
        enabled=True,
        groups=[],
        metadata={},
        weight=Weight.MEDIUM,
    )

    # usable_rules is empty: the transient rule isn't stored.
    result = await resolver.filter_unmet_dependencies(
        usable_rules=[],
        evaluations=[RuleEvaluation(rule=transient_rule, reasoning="")],
    )

    assert result.evaluations == [RuleEvaluation(rule=transient_rule, reasoning="")]
    assert_resolutions(result, transient_rule.id, [ResolutionKind.NONE])


async def test_that_shared_target_in_dependency_all_and_dependency_any_filters_when_shared_target_not_matched(
    container: Container,
) -> None:
    """
    G1 depend_on(G2) AND depend_on_any(G2, G3).
    G2 NOT matched, G3 matched.
    AND: G2 unmet → G1 filtered, even though OR group passes via G3.
    Result: {G3}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")

    # G1 depends on G2 (AND)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G1 depend_on_any(G2, G3)
    group_id = "shared-group"
    for target in [g2, g3]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.RULE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            # G2 NOT matched
            RuleEvaluation(rule=g3, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_shared_target_in_dependency_all_and_dependency_any_survives_when_shared_target_matched(
    container: Container,
) -> None:
    """
    G1 depend_on(G2) AND depend_on_any(G2, G3).
    G2 matched, G3 NOT matched.
    AND: G2 met. OR group: G2 met. Both satisfied → G1 survives.
    Result: {G1, G2}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")

    # G1 depends on G2 (AND)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G1 depend_on_any(G2, G3)
    group_id = "shared-group"
    for target in [g2, g3]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.RULE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            # G3 NOT matched
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g2.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_single_target_dependency_any_group_filters_when_target_not_matched(
    container: Container,
) -> None:
    """
    G1 depend_on_any(G2). G2 NOT matched.
    Degenerate OR group with one member → G1 filtered (same as depend_on).
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id="single-group",
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            # G2 NOT matched
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == set()

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ANY])


async def test_that_tag_all_dependency_cascades_when_one_member_own_dependency_fails(
    container: Container,
) -> None:
    """
    G1 depends on GROUP_ALL(t1). t1 has G2 and G3.
    G2 depends on G4. G4 NOT matched → G2 filtered.
    GROUP_ALL requires all members — G2 gone → G1 filtered.
    Result: {G3}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t1.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t1.id])
    g4 = await rule_store.create_rule(condition="d", content="g4 action")

    # G1 depends on GROUP_ALL(t1)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G2 depends on G4
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3, g4],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            RuleEvaluation(rule=g3, reasoning=""),
            # G4 NOT matched → G2 will be filtered
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_tag_all_dependency_filters_when_not_all_members_matched(
    container: Container,
) -> None:
    """
    G1 depends on GROUP_ALL(t1). t1 has G2 and G3. Only G2 matched.
    GROUP_ALL requires all members → G1 filtered.
    Result: {G2}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t1.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t1.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            # G3 NOT matched
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g2.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_tag_all_dependency_survives_when_all_members_matched(
    container: Container,
) -> None:
    """
    G1 depends on GROUP_ALL(t1). t1 has G2 and G3. Both matched.
    GROUP_ALL: all members active → G1 survives.
    Result: {G1, G2, G3}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t1.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t1.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            RuleEvaluation(rule=g3, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g2.id, g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_tag_any_dependency_filters_when_no_members_matched(
    container: Container,
) -> None:
    """
    G1 depends on GROUP_ANY(t1). t1 has G2 and G3. Neither matched.
    GROUP_ANY: no members → G1 filtered.
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t1.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t1.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            # G2 NOT matched
            # G3 NOT matched
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == set()

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_tag_any_dependency_survives_when_one_of_two_members_matched(
    container: Container,
) -> None:
    """
    G1 depends on GROUP_ANY(t1). t1 has G2 and G3. Only G2 matched.
    GROUP_ANY: at least one member → G1 survives.
    Result: {G1, G2}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t1.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t1.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            # G3 NOT matched
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g2.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_tag_dependency_allows_tagged_rules_when_dependency_is_met(
    container: Container,
) -> None:
    """
    Happy-path for group-level dependency: when the dependency IS met,
    grouped rules should survive normally.

    - t1 depends on t2
    - g1 grouped t1, g2 grouped t2 — both matched
    - After resolution: both survive
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")
    t2 = await group_store.create_group(name="t2")

    g1 = await rule_store.create_rule(condition="a", content="g1 action", groups=[t1.id])
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t2.id])

    # t1 depends on t2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g2.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_tag_dependency_deactivates_tagged_rules_when_target_group_not_met(
    container: Container,
) -> None:
    """
    t1 depends on t2. g1 grouped t1, g2 grouped t2, g3 untagged.
    g2 is NOT matched → t2 dependency unmet → g1 deactivated.
    Result: {g3}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")
    t2 = await group_store.create_group(name="t2")

    g1 = await rule_store.create_rule(condition="a", content="g1 action", groups=[t1.id])
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t2.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action")

    # t1 depends on t2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            # g2 NOT matched
            RuleEvaluation(rule=g3, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_tag_dependency_deactivates_tagged_rules_when_target_rule_not_met(
    container: Container,
) -> None:
    """
    t1 depends on g2. g1 grouped t1, g2 untagged, g3 untagged.
    g2 NOT matched → t1 dependency unmet → g1 deactivated.
    Result: {g3}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")

    g1 = await rule_store.create_rule(condition="a", content="g1 action", groups=[t1.id])
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")

    # t1 depends on g2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            # g2 NOT matched
            RuleEvaluation(rule=g3, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_tag_to_tag_dependency_survives_when_at_least_one_target_group_member_is_matched(
    container: Container,
) -> None:
    """
    t1 depends on t2. g1 grouped t1, g2 and g3 grouped t2.
    Only g2 matched → t1 dependency met (ANY). g1 survives.
    Result: {g1, g2}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")
    t2 = await group_store.create_group(name="t2")

    g1 = await rule_store.create_rule(condition="a", content="g1 action", groups=[t1.id])
    g2 = await rule_store.create_rule(condition="b", content="g2 action", groups=[t2.id])
    g3 = await rule_store.create_rule(condition="c", content="g3 action", groups=[t2.id])

    # t1 depends on t2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.GROUP_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            # g3 NOT matched
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g2.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_two_dependency_any_groups_are_anded_together(
    container: Container,
) -> None:
    """
    G1 depend_on_any(G2, G3) AND depend_on_any(G4, G5).
    G2 matched (group A met). G4 NOT, G5 NOT (group B unmet).
    Group A met but group B unmet → G1 filtered.
    Result: {G2}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")
    g4 = await rule_store.create_rule(condition="d", content="g4 action")
    g5 = await rule_store.create_rule(condition="e", content="g5 action")

    # Group A: G2 or G3
    for target in [g2, g3]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.RULE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id="group-a",
        )

    # Group B: G4 or G5
    for target in [g4, g5]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.RULE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id="group-b",
        )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3, g4, g5],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            # G3, G4, G5 NOT matched
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g2.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ANY])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


# ── DEPENDENCY_ANY edge case tests ─────────────────────────────────────────


# ---------------------------------------------------------------------------
# Ported from the alpha engine's relational resolver suite (relational
# priority). resolve_priority_relationships prunes the turn-matched set:
# a match is deprioritized when a PRIORITY relationship (direct only —
# the reinstatement principle) targets it or one of its groups and the
# source is another rule in the input set (or a group with such a member).
# Sources judged against the full input set, so dropped rules still
# suppress their own targets (chains collapse in one pass).
# ---------------------------------------------------------------------------


async def test_that_priority_chain_attributes_to_direct_deprioritizer(
    container: Container,
) -> None:
    """
    A prioritizes over B, B prioritizes over C. All matched.
    B should be attributed to A (direct deprioritizer).
    C should be attributed to B (direct deprioritizer), NOT A (transitive).
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g_a = await rule_store.create_rule(condition="a", content="action A")
    g_b = await rule_store.create_rule(condition="b", content="action B")
    g_c = await rule_store.create_rule(condition="c", content="action C")

    # A prio over B
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g_a.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g_b.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.PRIORITY,
    )

    # B prio over C
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g_b.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g_c.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve_priority_relationships(
        [g_a, g_b, g_c],
        [
            RuleEvaluation(rule=g_a, reasoning=""),
            RuleEvaluation(rule=g_b, reasoning=""),
            RuleEvaluation(rule=g_c, reasoning=""),
        ],
    )

    assert result.evaluations == [RuleEvaluation(rule=g_a, reasoning="")]

    assert_resolutions(result, g_a.id, [ResolutionKind.NONE])
    assert_resolutions(result, g_b.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g_c.id, [ResolutionKind.DEPRIORITIZED])

    # B deprioritized by A (direct)
    b_res = get_resolutions_by_kind(result, g_b.id, ResolutionKind.DEPRIORITIZED)
    assert len(b_res) == 1
    assert ResolvedEntity.rule(g_a) in b_res[0].details.counterparts

    # C deprioritized by B (direct), NOT by A (transitive)
    c_res = get_resolutions_by_kind(result, g_c.id, ResolutionKind.DEPRIORITIZED)
    assert len(c_res) == 1
    assert ResolvedEntity.rule(g_b) in c_res[0].details.counterparts


async def test_that_priority_chain_with_gaps_does_not_transitively_deprioritize(
    container: Container,
) -> None:
    """
    G1 → G2 → G3 → G4 (priority chain).
    Only G2 and G4 are matched.
    G2's direct deprioritizer (G1) is not matched → G2 survives.
    G4's direct deprioritizer (G3) is not matched → what happens to G4?
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="action 1")
    g2 = await rule_store.create_rule(condition="b", content="action 2")
    g3 = await rule_store.create_rule(condition="c", content="action 3")
    g4 = await rule_store.create_rule(condition="d", content="action 4")

    # G1 prio over G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.PRIORITY,
    )

    # G2 prio over G3
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.PRIORITY,
    )

    # G3 prio over G4
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve_priority_relationships(
        [g1, g2, g3, g4],
        [
            # Only G2 and G4 matched
            RuleEvaluation(rule=g2, reasoning=""),
            RuleEvaluation(rule=g4, reasoning=""),
        ],
    )

    # Both survive: G2's deprioritizer (G1) is not matched, and G4's
    # deprioritizer (G3) is not matched.  Priority does NOT propagate
    # through inactive intermediaries (reinstatement principle).
    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g2.id, g4.id}

    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g4.id, [ResolutionKind.NONE])


async def test_that_relational_resolver_deprioritizes_target_rule_when_source_is_custom_group(
    container: Container,
) -> None:
    """
    Tests that a custom group used as source in a PRIORITY relationship
    deprioritizes the target rule when a rule grouped with that
    group is matched.

    - Group t1 is attached to g1
    - t1 PRIORITY → g2
    - Both g1 and g2 are matched
    - Expected: g2 is deprioritized, only g1 remains
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="b")
    g2 = await rule_store.create_rule(condition="c", content="d")

    t1 = await group_store.create_group(name="t1")
    await rule_store.upsert_group(g1.id, t1.id)
    g1 = await rule_store.read_rule(g1.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve_priority_relationships(
        [g1, g2],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
        ],
    )

    assert len(result.evaluations) == 1
    assert result.evaluations[0].rule.id == g1.id

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.DEPRIORITIZED])


async def test_that_relational_resolver_does_not_ignore_a_deprioritized_rule_when_its_prioritized_counterpart_is_not_active(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    prioritized_rule = await rule_store.create_rule(condition="x", content="y")
    deprioritized_rule = await rule_store.create_rule(condition="y", content="z")

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=prioritized_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=deprioritized_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    matches: list[RuleEvaluation] = [
        RuleEvaluation(rule=deprioritized_rule, reasoning=""),
    ]

    result = await resolver.resolve_priority_relationships(
        [prioritized_rule, deprioritized_rule], matches
    )

    assert result.evaluations == [RuleEvaluation(rule=deprioritized_rule, reasoning="")]

    assert_resolutions(result, deprioritized_rule.id, [ResolutionKind.NONE])


async def test_that_relational_resolver_does_not_ignore_a_deprioritized_tag_when_its_prioritized_counterpart_is_not_active(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    prioritized_rule = await rule_store.create_rule(condition="x", content="y")
    deprioritized_rule = await rule_store.create_rule(condition="y", content="z")

    deprioritized_tag = await group_store.create_group(name="t1")

    await rule_store.upsert_group(deprioritized_rule.id, deprioritized_tag.id)

    # Re-read after tagging so rule.groups is up-to-date
    deprioritized_rule = await rule_store.read_rule(deprioritized_rule.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=prioritized_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=deprioritized_tag.id,
            kind=RelationshipEntityKind.GROUP_ALL,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=deprioritized_tag.id,
            kind=RelationshipEntityKind.GROUP_ALL,
        ),
        target=RelationshipEntity(
            id=deprioritized_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve_priority_relationships(
        [prioritized_rule, deprioritized_rule],
        [
            RuleEvaluation(rule=deprioritized_rule, reasoning=""),
        ],
    )

    assert len(result.evaluations) == 1
    assert result.evaluations[0].rule.id == deprioritized_rule.id

    assert_resolutions(result, deprioritized_rule.id, [ResolutionKind.NONE])


async def test_that_relational_resolver_prioritizes_rules(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    prioritized_rule = await rule_store.create_rule(condition="x", content="y")
    deprioritized_rule = await rule_store.create_rule(condition="y", content="z")

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=prioritized_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=deprioritized_rule.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    matches: list[RuleEvaluation] = [
        RuleEvaluation(rule=prioritized_rule, reasoning=""),
        RuleEvaluation(rule=deprioritized_rule, reasoning=""),
    ]

    result = await resolver.resolve_priority_relationships(
        [prioritized_rule, deprioritized_rule], matches
    )

    assert result.evaluations == [RuleEvaluation(rule=prioritized_rule, reasoning="")]

    assert_resolutions(result, prioritized_rule.id, [ResolutionKind.NONE])
    assert_resolutions(result, deprioritized_rule.id, [ResolutionKind.DEPRIORITIZED])


async def test_that_relational_resolver_prioritizes_rules_from_tags(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="x", content="y")
    g2 = await rule_store.create_rule(condition="y", content="z")

    t1 = await group_store.create_group(name="t1")

    await rule_store.upsert_group(g2.id, t1.id)

    # Re-read after tagging so rule.groups is up-to-date
    g2 = await rule_store.read_rule(g2.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g1.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.GROUP_ALL,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.GROUP_ALL,
        ),
        target=RelationshipEntity(
            id=g2.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve_priority_relationships(
        [g1, g2],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
        ],
    )

    assert len(result.evaluations) == 1
    assert result.evaluations[0].rule.id == g1.id

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.DEPRIORITIZED])


async def test_that_relational_resolver_prioritizes_indirectly_between_rules(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="x", content="y")
    g2 = await rule_store.create_rule(condition="y", content="z")
    g3 = await rule_store.create_rule(condition="z", content="t")

    rel_g1_g2 = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g1.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=g2.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    rel_g2_g3 = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g2.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=g3.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve_priority_relationships(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            RuleEvaluation(rule=g3, reasoning=""),
        ],
    )

    assert result.evaluations == [RuleEvaluation(rule=g1, reasoning="")]

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g3.id, [ResolutionKind.DEPRIORITIZED])

    # Each DEPRIORITIZED resolution should identify the prioritizing
    # rule as a counterpart and reference the PRIORITY relationship
    # that caused the deprioritization.
    g2_res = get_resolutions_by_kind(result, g2.id, ResolutionKind.DEPRIORITIZED)
    assert len(g2_res) == 1
    assert ResolvedEntity.rule(g1) in g2_res[0].details.counterparts
    assert g2_res[0].details.relationship is not None
    assert g2_res[0].details.relationship.id == rel_g1_g2.id

    g3_res = get_resolutions_by_kind(result, g3.id, ResolutionKind.DEPRIORITIZED)
    assert len(g3_res) == 1
    assert ResolvedEntity.rule(g2) in g3_res[0].details.counterparts
    assert g3_res[0].details.relationship is not None
    assert g3_res[0].details.relationship.id == rel_g2_g3.id


async def test_that_tag_priority_deprioritizes_all_rules_of_target_group(
    container: Container,
) -> None:
    """
    Tests that group-level prioritization filters out all rules of the target group:
    - t1 prioritizes over t2
    - g1_1, g1_2 grouped with t1; g2_1, g2_2 grouped with t2
    - After resolution only g1_1 and g1_2 remain (t2 rules are deprioritized).
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")
    t2 = await group_store.create_group(name="t2")

    g1_1 = await rule_store.create_rule(condition="a", content="g1_1 action")
    g1_2 = await rule_store.create_rule(condition="b", content="g1_2 action")
    g2_1 = await rule_store.create_rule(condition="c", content="g2_1 action")
    g2_2 = await rule_store.create_rule(condition="d", content="g2_2 action")

    await rule_store.upsert_group(g1_1.id, t1.id)
    await rule_store.upsert_group(g1_2.id, t1.id)
    await rule_store.upsert_group(g2_1.id, t2.id)
    await rule_store.upsert_group(g2_2.id, t2.id)

    g1_1 = await rule_store.read_rule(g1_1.id)
    g1_2 = await rule_store.read_rule(g1_2.id)
    g2_1 = await rule_store.read_rule(g2_1.id)
    g2_2 = await rule_store.read_rule(g2_2.id)

    # t1 prioritizes over t2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve_priority_relationships(
        [g1_1, g1_2, g2_1, g2_2],
        [
            RuleEvaluation(rule=g1_1, reasoning=""),
            RuleEvaluation(rule=g1_2, reasoning=""),
            RuleEvaluation(rule=g2_1, reasoning=""),
            RuleEvaluation(rule=g2_2, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1_1.id, g1_2.id}

    assert_resolutions(result, g1_1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g1_2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2_1.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g2_2.id, [ResolutionKind.DEPRIORITIZED])


async def test_that_tag_priority_does_not_deprioritize_when_no_source_group_member_is_matched(
    container: Container,
) -> None:
    """
    Group-level priority t1→t2 should not fire if no t1 member is matched.
    t2 members should survive.

    - t1 prioritizes over t2
    - g1_1 grouped t1 (NOT matched), g2_1 grouped t2 (matched)
    - After resolution: g2_1 survives (no t1 member is active to trigger deprioritization)
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")
    t2 = await group_store.create_group(name="t2")

    g1_1 = await rule_store.create_rule(condition="a", content="g1_1 action", groups=[t1.id])
    g2_1 = await rule_store.create_rule(condition="b", content="g2_1 action", groups=[t2.id])

    # t1 prioritizes over t2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve_priority_relationships(
        [g1_1, g2_1],
        [
            # g1_1 NOT matched
            RuleEvaluation(rule=g2_1, reasoning=""),
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g2_1.id}

    # Resolution assertions
    assert_resolutions(result, g2_1.id, [ResolutionKind.NONE])


async def test_that_tag_priority_excludes_all_target_members_regardless_of_individual_priority(
    container: Container,
) -> None:
    """
    Tests that group-level prioritization is absolute — individual-level priority
    relationships do NOT grant immunity from group-level deprioritization:
    - t1 prioritizes over t2 (group-level: all of t1 beats all of t2)
    - g2_1 prioritizes over g1_1 (rule-level)
    - After resolution: g1_1 deprioritized by g2_1 (rule-level),
      g2_1 and g2_2 deprioritized by t1 (group-level). Only g1_2 survives.
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    t1 = await group_store.create_group(name="t1")
    t2 = await group_store.create_group(name="t2")

    g1_1 = await rule_store.create_rule(condition="a", content="g1_1 action")
    g1_2 = await rule_store.create_rule(condition="b", content="g1_2 action")
    g2_1 = await rule_store.create_rule(condition="c", content="g2_1 action")
    g2_2 = await rule_store.create_rule(condition="d", content="g2_2 action")

    await rule_store.upsert_group(g1_1.id, t1.id)
    await rule_store.upsert_group(g1_2.id, t1.id)
    await rule_store.upsert_group(g2_1.id, t2.id)
    await rule_store.upsert_group(g2_2.id, t2.id)

    g1_1 = await rule_store.read_rule(g1_1.id)
    g1_2 = await rule_store.read_rule(g1_2.id)
    g2_1 = await rule_store.read_rule(g2_1.id)
    g2_2 = await rule_store.read_rule(g2_2.id)

    # t1 prioritizes over t2 (group-level)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.PRIORITY,
    )

    # g2_1 prioritizes over g1_1 (rule-level — does NOT grant immunity from group-level)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2_1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g1_1.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve_priority_relationships(
        [g1_1, g1_2, g2_1, g2_2],
        [
            RuleEvaluation(rule=g1_1, reasoning=""),
            RuleEvaluation(rule=g1_2, reasoning=""),
            RuleEvaluation(rule=g2_1, reasoning=""),
            RuleEvaluation(rule=g2_2, reasoning=""),
        ],
    )

    # g1_1 deprioritized by g2_1, then g2_1 and g2_2 deprioritized by t1→t2.
    # Only g1_2 survives.
    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1_2.id}

    assert_resolutions(result, g1_1.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g1_2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2_1.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g2_2.id, [ResolutionKind.DEPRIORITIZED])


# ---------------------------------------------------------------------------
# Ported from the alpha engine's relational resolver suite (entailment).
# expand_entailment activates entailment targets of the input evaluations
# (indirect closure, traversing group targets); entailed additions carry an
# ENTAILED resolution per entailing source.
# ---------------------------------------------------------------------------


async def test_that_already_matched_entailment_target_gets_none_not_entailed(
    container: Container,
) -> None:
    """
    G1 entails G2. Both G1 and G2 are already matched.
    G2 should get NONE because it was already active — entailment
    was not needed to bring it in.
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")

    # G1 entails G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.ENTAILMENT,
    )

    result = await resolver.expand_entailment(
        [g1, g2],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),  # Already matched
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g2.id}

    # Both get NONE — G2 was already active, entailment was unnecessary
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_chained_entailment_activates_all_linked_rules(
    container: Container,
) -> None:
    """
    G1 entails G2, G2 entails G3. Only G1 matched.
    Entailment should activate G2 (from G1), then G3 (from G2).
    Result: {G1, G2, G3}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")

    # G1 entails G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.ENTAILMENT,
    )

    # G2 entails G3
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.ENTAILMENT,
    )

    result = await resolver.expand_entailment(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            # G2 and G3 NOT initially matched
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g2.id, g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.ENTAILED])
    assert_resolutions(result, g3.id, [ResolutionKind.ENTAILED])


async def test_that_rule_entailed_by_two_sources_records_both_entailment_resolutions(
    container: Container,
) -> None:
    """
    G1 entails G3, G2 entails G3. G1 and G2 are matched, G3 is not.
    G3 should have TWO ENTAILED resolutions — one for each source.
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")

    # G1 entails G3
    r1 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.ENTAILMENT,
    )

    # G2 entails G3
    r2 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.ENTAILMENT,
    )

    result = await resolver.expand_entailment(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
            # G3 NOT matched — activated via entailment from both G1 and G2
        ],
    )

    result_ids = {m.rule.id for m in result.evaluations}
    assert result_ids == {g1.id, g2.id, g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.ENTAILED, ResolutionKind.ENTAILED])

    # Both entailment relationships should be referenced
    g3_res = get_resolutions_by_kind(result, g3.id, ResolutionKind.ENTAILED)
    assert len(g3_res) == 2
    for r in g3_res:
        assert r.details.relationship is not None
    rel_ids = {r.details.relationship.id for r in g3_res if r.details.relationship is not None}
    assert rel_ids == {r1.id, r2.id}

    # Each ENTAILED resolution should also identify the entailing source
    # rule as a counterpart, paired with the matching relationship.
    rel_to_counterparts = {
        r.details.relationship.id: r.details.counterparts
        for r in g3_res
        if r.details.relationship is not None
    }
    assert rel_to_counterparts[r1.id] == (ResolvedEntity.rule(g1),)
    assert rel_to_counterparts[r2.id] == (ResolvedEntity.rule(g2),)


async def test_that_relational_resolver_infers_rules_from_tags(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="x", content="y")
    g2 = await rule_store.create_rule(condition="y", content="z")
    g3 = await rule_store.create_rule(condition="z", content="t")
    g4 = await rule_store.create_rule(condition="t", content="u")

    t1 = await group_store.create_group(name="t1")

    await rule_store.upsert_group(rule_id=g2.id, group_id=t1.id)
    await rule_store.upsert_group(rule_id=g3.id, group_id=t1.id)

    # Re-read after tagging so usable_rules has up-to-date group lists
    g2 = await rule_store.read_rule(g2.id)
    g3 = await rule_store.read_rule(g3.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g1.id,
            kind=RelationshipEntityKind.RULE,
        ),
        target=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.GROUP_ALL,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.GROUP_ALL,
        ),
        target=RelationshipEntity(
            id=g4.id,
            kind=RelationshipEntityKind.RULE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )

    result = await resolver.expand_entailment(
        [g1, g2, g3, g4],
        [
            RuleEvaluation(rule=g1, reasoning=""),
        ],
    )

    assert len(result.evaluations) == 4
    assert any(m.rule.id == g1.id for m in result.evaluations)
    assert any(m.rule.id == g2.id for m in result.evaluations)
    assert any(m.rule.id == g3.id for m in result.evaluations)
    assert any(m.rule.id == g4.id for m in result.evaluations)

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.ENTAILED])
    assert_resolutions(result, g3.id, [ResolutionKind.ENTAILED])
    assert_resolutions(result, g4.id, [ResolutionKind.ENTAILED])


async def test_that_entailed_rule_is_filtered_when_its_dependency_is_unmet(
    container: Container,
) -> None:
    """
    G1 entails G2. G2 depends on G3. G1 matched, G3 NOT matched.
    G2 is entailed, but the subsequent dependency gate drops it.
    Result: {G1}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.ENTAILMENT,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    expanded = await resolver.expand_entailment(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            # G2 not initially matched (entailed by G1)
            # G3 NOT matched
        ],
    )

    assert {m.rule.id for m in expanded.evaluations} == {g1.id, g2.id}
    assert_resolutions(expanded, g2.id, [ResolutionKind.ENTAILED])

    filtered = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        list(expanded.evaluations),
    )

    assert {m.rule.id for m in filtered.evaluations} == {g1.id}
    assert_resolutions(filtered, g1.id, [ResolutionKind.NONE])
    assert_resolutions(filtered, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_entailed_rule_satisfies_dependency_of_matched_rule(
    container: Container,
) -> None:
    """
    G3 depends on G2. G1 entails G2.
    G1 and G3 are matched, G2 is not.
    Entailment expansion runs before the dependency gate, so by the time
    dependencies are checked, G2 is present and G3's dependency is met.
    Result: {G1, G2, G3}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")

    # G1 entails G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.ENTAILMENT,
    )

    # G3 depends on G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    expanded = await resolver.expand_entailment(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            # G2 NOT matched — will be entailed by G1
            RuleEvaluation(rule=g3, reasoning=""),
        ],
    )

    assert_resolutions(expanded, g2.id, [ResolutionKind.ENTAILED])

    filtered = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        list(expanded.evaluations),
    )

    assert {m.rule.id for m in filtered.evaluations} == {g1.id, g2.id, g3.id}
    assert_resolutions(filtered, g1.id, [ResolutionKind.NONE])
    assert_resolutions(filtered, g3.id, [ResolutionKind.NONE])


async def test_that_entailed_rule_survives_when_its_dependency_any_group_is_met(
    container: Container,
) -> None:
    """
    G1 entails G2. G2 depend_on_any(G3, G4). G1 and G3 matched.
    G2 entailed, OR group met via G3 → G2 survives.
    Result: {G1, G2, G3}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action")
    g2 = await rule_store.create_rule(condition="b", content="g2 action")
    g3 = await rule_store.create_rule(condition="c", content="g3 action")
    g4 = await rule_store.create_rule(condition="d", content="g4 action")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.ENTAILMENT,
    )

    group_id = "entailed-or-group"
    for target in [g3, g4]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.RULE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )

    expanded = await resolver.expand_entailment(
        [g1, g2, g3, g4],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            # G2 not initially matched (entailed)
            RuleEvaluation(rule=g3, reasoning=""),
            # G4 NOT matched
        ],
    )

    assert_resolutions(expanded, g2.id, [ResolutionKind.ENTAILED])

    filtered = await resolver.filter_unmet_dependencies(
        [g1, g2, g3, g4],
        list(expanded.evaluations),
    )

    assert {m.rule.id for m in filtered.evaluations} == {g1.id, g2.id, g3.id}
    assert_resolutions(filtered, g1.id, [ResolutionKind.NONE])
    assert_resolutions(filtered, g3.id, [ResolutionKind.NONE])


async def test_that_numerical_priority_does_not_filter_entailer_when_entailed_has_higher_priority(
    container: Container,
) -> None:
    """
    G1 (priority 0) entails G2 (priority 100). G1 matched.
    An entailed rule owes its activation to its entailer, so it must never
    numerically suppress it: entailed evaluations don't raise the priority
    bar. Both survive.
    Result: {G1, G2}
    """
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1 action", priority=0)
    g2 = await rule_store.create_rule(condition="b", content="g2 action", priority=100)

    # G1 entails G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.ENTAILMENT,
    )

    expanded = await resolver.expand_entailment(
        [g1, g2],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            # G2 NOT initially matched — added via entailment
        ],
    )

    assert_resolutions(expanded, g2.id, [ResolutionKind.ENTAILED])

    entailed_ids = {
        RuleId(entity.entity.id)
        for entity, resolutions in expanded.resolutions.items()
        if any(resolution.kind == ResolutionKind.ENTAILED for resolution in resolutions)
    }

    result = resolver.keep_highest_priority(
        list(expanded.evaluations),
        entailed_rule_ids=entailed_ids,
    )

    assert {m.rule.id for m in result.evaluations} == {g1.id, g2.id}
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])


async def test_that_only_designated_same_turn_satisfiers_meet_dependencies(
    container: Container,
) -> None:
    # The matcher distinguishes turn MATCHES from discovery ADMISSIONS when it
    # gates a batch: only matches (and the session's matched-rule ledger, via
    # satisfied_rule_ids) satisfy dependencies. Admission-usage evaluations are
    # still gated themselves, but satisfy nobody.
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    dependent = await rule_store.create_rule(condition="a", content="dependent")
    admitted_target = await rule_store.create_rule(condition="b", content="admitted target")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=dependent.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=admitted_target.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    evaluations = [
        RuleEvaluation(rule=dependent, reasoning=""),
        RuleEvaluation(rule=admitted_target, reasoning=""),
    ]

    # Default behavior: every evaluation in the batch satisfies (unchanged).
    unrestricted = await resolver.filter_unmet_dependencies(
        [dependent, admitted_target],
        evaluations,
    )
    assert {m.rule.id for m in unrestricted.evaluations} == {dependent.id, admitted_target.id}

    # Restricted: the target is a mere admission — the dependent stays locked,
    # while the admission itself survives.
    restricted = await resolver.filter_unmet_dependencies(
        [dependent, admitted_target],
        evaluations,
        same_turn_satisfier_ids=frozenset(),
    )
    assert {m.rule.id for m in restricted.evaluations} == {admitted_target.id}
    assert_resolutions(restricted, dependent, [ResolutionKind.UNMET_DEPENDENCY_ALL])

    # The ledger unlocks it even when nothing satisfies same-turn.
    via_ledger = await resolver.filter_unmet_dependencies(
        [dependent, admitted_target],
        evaluations,
        satisfied_rule_ids={admitted_target.id},
        same_turn_satisfier_ids=frozenset(),
    )
    assert {m.rule.id for m in via_ledger.evaluations} == {dependent.id, admitted_target.id}


async def test_that_a_dropped_same_turn_satisfier_stops_satisfying(
    container: Container,
) -> None:
    # A designated satisfier that itself fails its dependency must not unlock
    # its dependents (same-turn chains stay consistent under restriction).
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", content="g1")
    g2 = await rule_store.create_rule(condition="b", content="g2")
    g3 = await rule_store.create_rule(condition="c", content="g3")

    # g1 depends on g2, g2 depends on g3 (never matched).
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.filter_unmet_dependencies(
        [g1, g2, g3],
        [
            RuleEvaluation(rule=g1, reasoning=""),
            RuleEvaluation(rule=g2, reasoning=""),
        ],
        same_turn_satisfier_ids={g1.id, g2.id},
    )

    assert [m.rule.id for m in result.evaluations] == []
    assert_resolutions(result, g1, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g2, [ResolutionKind.UNMET_DEPENDENCY_ALL])
