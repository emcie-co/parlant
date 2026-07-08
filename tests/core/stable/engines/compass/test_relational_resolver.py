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

from lagom import Container

from parlant.core.engines.rule_match import RuleMatch
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

    g1 = await rule_store.create_rule(condition="a", action="g1")
    g2 = await rule_store.create_rule(condition="b", action="g2")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2],
        [RuleMatch(rule=g1, rationale="")],
    )

    assert [match.rule.id for match in result.matches] == []
    assert_resolutions(result, g1, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_compass_relational_resolver_keeps_met_rule_dependency(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", action="g1")
    g2 = await rule_store.create_rule(condition="b", action="g2")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2],
        [
            RuleMatch(rule=g1, rationale=""),
            RuleMatch(rule=g2, rationale=""),
        ],
    )

    assert {match.rule.id for match in result.matches} == {g1.id, g2.id}
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
    g1 = await rule_store.create_rule(condition="a", action="g1")
    g2 = await rule_store.create_rule(condition="b", action="g2", groups=[group.id])
    g3 = await rule_store.create_rule(condition="c", action="g3", groups=[group.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=group.id, kind=RelationshipEntityKind.GROUP_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    missing_member = await resolver.resolve(
        [g1, g2, g3],
        [
            RuleMatch(rule=g1, rationale=""),
            RuleMatch(rule=g2, rationale=""),
        ],
    )

    assert {match.rule.id for match in missing_member.matches} == {g2.id}
    assert_resolutions(missing_member, g1, [ResolutionKind.UNMET_DEPENDENCY_ALL])

    all_members = await resolver.resolve(
        [g1, g2, g3],
        [
            RuleMatch(rule=g1, rationale=""),
            RuleMatch(rule=g2, rationale=""),
            RuleMatch(rule=g3, rationale=""),
        ],
    )

    assert {match.rule.id for match in all_members.matches} == {g1.id, g2.id, g3.id}
    assert_resolutions(all_members, g1, [ResolutionKind.NONE])


async def test_that_compass_relational_resolver_supports_tag_any_dependency(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    group = await group_store.create_group(name="group-any")
    g1 = await rule_store.create_rule(condition="a", action="g1")
    g2 = await rule_store.create_rule(condition="b", action="g2", groups=[group.id])
    g3 = await rule_store.create_rule(condition="c", action="g3", groups=[group.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=group.id, kind=RelationshipEntityKind.GROUP_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    one_member = await resolver.resolve(
        [g1, g2, g3],
        [
            RuleMatch(rule=g1, rationale=""),
            RuleMatch(rule=g2, rationale=""),
        ],
    )

    assert {match.rule.id for match in one_member.matches} == {g1.id, g2.id}
    assert_resolutions(one_member, g1, [ResolutionKind.NONE])

    no_members = await resolver.resolve(
        [g1, g2, g3],
        [RuleMatch(rule=g1, rationale="")],
    )

    assert [match.rule.id for match in no_members.matches] == []
    assert_resolutions(no_members, g1, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_compass_relational_resolver_supports_dependency_any_groups(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    g1 = await rule_store.create_rule(condition="a", action="g1")
    g2 = await rule_store.create_rule(condition="b", action="g2")
    g3 = await rule_store.create_rule(condition="c", action="g3")

    for target in [g2, g3]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.RULE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.RULE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id="any-group",
        )

    one_target = await resolver.resolve(
        [g1, g2, g3],
        [
            RuleMatch(rule=g1, rationale=""),
            RuleMatch(rule=g2, rationale=""),
        ],
    )

    assert {match.rule.id for match in one_target.matches} == {g1.id, g2.id}
    assert_resolutions(one_target, g1, [ResolutionKind.NONE])

    no_targets = await resolver.resolve(
        [g1, g2, g3],
        [RuleMatch(rule=g1, rationale="")],
    )

    assert [match.rule.id for match in no_targets.matches] == []
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
        action="grouped",
        groups=[group.id],
    )
    dependency = await rule_store.create_rule(condition="b", action="dependency")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=group.id, kind=RelationshipEntityKind.GROUP_ALL),
        target=RelationshipEntity(id=dependency.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    unmet = await resolver.resolve(
        [grouped, dependency],
        [RuleMatch(rule=grouped, rationale="")],
    )

    assert [match.rule.id for match in unmet.matches] == []
    assert_resolutions(unmet, grouped, [ResolutionKind.UNMET_DEPENDENCY_ALL])

    met = await resolver.resolve(
        [grouped, dependency],
        [
            RuleMatch(rule=grouped, rationale=""),
            RuleMatch(rule=dependency, rationale=""),
        ],
    )

    assert {match.rule.id for match in met.matches} == {grouped.id, dependency.id}
    assert_resolutions(met, grouped, [ResolutionKind.NONE])


async def test_that_compass_relational_resolver_cascades_dependency_chains(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    root_dependent = await rule_store.create_rule(condition="a", action="root")
    middle = await rule_store.create_rule(condition="b", action="middle")
    missing_leaf = await rule_store.create_rule(condition="c", action="leaf")

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

    result = await resolver.resolve(
        [root_dependent, middle, missing_leaf],
        [
            RuleMatch(rule=root_dependent, rationale=""),
            RuleMatch(rule=middle, rationale=""),
        ],
    )

    assert [match.rule.id for match in result.matches] == []
    assert_resolutions(result, root_dependent, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, middle, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_compass_relational_resolver_applies_numeric_priority(
    container: Container,
) -> None:
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    low = await rule_store.create_rule(condition="a", action="low", priority=0)
    high = await rule_store.create_rule(condition="b", action="high", priority=10)

    result = await resolver.resolve(
        [low, high],
        [
            RuleMatch(rule=low, rationale=""),
            RuleMatch(rule=high, rationale=""),
        ],
    )

    assert [match.rule.id for match in result.matches] == [high.id]
    assert_resolutions(result, low, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, high, [ResolutionKind.NONE])

    [resolution] = get_resolutions_by_kind(result, low, ResolutionKind.DEPRIORITIZED)
    assert resolution.details.counterparts == (ResolvedEntity.rule(high),)
    assert resolution.details.relationship is None


async def test_that_compass_relational_resolver_cascades_priority_through_dependencies(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    resolver = make_resolver(container)

    dependent = await rule_store.create_rule(
        condition="a",
        action="dependent",
        priority=100,
    )
    dependency = await rule_store.create_rule(
        condition="b",
        action="dependency",
        priority=0,
    )
    winner = await rule_store.create_rule(
        condition="c",
        action="winner",
        priority=100,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=dependent.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=dependency.id, kind=RelationshipEntityKind.RULE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [dependent, dependency, winner],
        [
            RuleMatch(rule=dependent, rationale=""),
            RuleMatch(rule=dependency, rationale=""),
            RuleMatch(rule=winner, rationale=""),
        ],
    )

    assert {match.rule.id for match in result.matches} == {winner.id}
    assert_resolutions(result, dependent, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, dependency, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, winner, [ResolutionKind.NONE])


async def test_that_compass_relational_resolver_cascades_tag_dependency_after_priority(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    rule_store = container[RuleStore]
    group_store = container[GroupStore]
    resolver = make_resolver(container)

    group = await group_store.create_group(name="priority-group")
    dependent = await rule_store.create_rule(
        condition="a",
        action="dependent",
        priority=100,
    )
    low_member = await rule_store.create_rule(
        condition="b",
        action="low member",
        priority=0,
        groups=[group.id],
    )
    high_member = await rule_store.create_rule(
        condition="c",
        action="high member",
        priority=100,
        groups=[group.id],
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=dependent.id, kind=RelationshipEntityKind.RULE),
        target=RelationshipEntity(id=group.id, kind=RelationshipEntityKind.GROUP_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [dependent, low_member, high_member],
        [
            RuleMatch(rule=dependent, rationale=""),
            RuleMatch(rule=low_member, rationale=""),
            RuleMatch(rule=high_member, rationale=""),
        ],
    )

    assert {match.rule.id for match in result.matches} == {dependent.id, high_member.id}
    assert_resolutions(result, dependent, [ResolutionKind.NONE])
    assert_resolutions(result, low_member, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, high_member, [ResolutionKind.NONE])
