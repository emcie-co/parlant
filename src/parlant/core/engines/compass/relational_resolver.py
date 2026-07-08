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

"""Relationship resolution for Compass rule matches.

Compass currently supports dependency relationships and numeric priority only.
Journeys, relational priority, and entailment are intentionally out of scope.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal, Sequence, TypeAlias, cast

from parlant.core.engines.rule_match import RuleMatch
from parlant.core.rules import Rule, RuleId
from parlant.core.loggers import Logger
from parlant.core.relationships import (
    Relationship,
    RelationshipEntityKind,
    RelationshipId,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.store_provider import StoreProvider, StoreProviderHints
from parlant.core.groups import Group, GroupId, GroupStore
from parlant.core.tools import ToolId
from parlant.core.tracer import Tracer

_CacheKey = tuple[RelationshipKind, bool, str, RuleId | GroupId | ToolId]
_RelationshipCache = dict[_CacheKey, list[Relationship]]

ResolvedEntityId: TypeAlias = RuleId


@dataclass(frozen=True)
class ResolvedEntity:
    entity_type: Literal["rule", "group"]
    entity: Rule | Group

    def __hash__(self) -> int:
        return hash((self.entity_type, self.entity.id))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ResolvedEntity)
            and self.entity_type == other.entity_type
            and self.entity.id == other.entity.id
        )

    @classmethod
    def rule(cls, rule: Rule) -> ResolvedEntity:
        return cls(entity_type="rule", entity=rule)

    @classmethod
    def group(cls, group: Group) -> ResolvedEntity:
        return cls(entity_type="group", entity=group)


class ResolutionKind(str, Enum):
    NONE = "none"
    UNMET_DEPENDENCY_ALL = "unmet_dependency_all"
    UNMET_DEPENDENCY_ANY = "unmet_dependency_any"
    DEPRIORITIZED = "deprioritized"


@dataclass(frozen=True)
class ResolutionDetails:
    description: str
    relationship: Relationship | None = None
    counterparts: tuple[ResolvedEntity, ...] = ()


@dataclass(frozen=True)
class Resolution:
    kind: ResolutionKind
    details: ResolutionDetails


@dataclass
class RelationalResolverResult:
    matches: Sequence[RuleMatch]
    resolutions: dict[ResolvedEntity, list[Resolution]] = field(default_factory=dict)


class _DependencyTargetKind(Enum):
    MATCHED_RULE = auto()
    ANY_MATCHED_GROUP_MEMBER = auto()
    UNMET = auto()


@dataclass
class _DependencyTarget:
    kind: _DependencyTargetKind
    rule_ids: set[RuleId] = field(default_factory=set)
    relationship: Relationship | None = None
    target_id: ResolvedEntityId | GroupId | None = None


class RelationalResolver:
    """Resolves Compass rule dependencies and numeric priority."""

    MAX_ITERATIONS = 3

    def __init__(
        self,
        store_provider: StoreProvider,
        logger: Logger,
        tracer: Tracer,
    ) -> None:
        self._relationship_store = store_provider.get_store(
            RelationshipStore, StoreProviderHints(call_site="engine")
        )
        self._group_store = store_provider.get_store(
            GroupStore, StoreProviderHints(call_site="engine")
        )
        self._logger = logger
        self._tracer = tracer

    async def resolve(
        self,
        usable_rules: Sequence[Rule],
        matches: Sequence[RuleMatch],
    ) -> RelationalResolverResult:
        with self._logger.scope("CompassRelationalResolver"):
            with self._tracer.span("compass.relational_resolver.resolve"):
                cache: _RelationshipCache = {}
                resolutions: dict[ResolvedEntity, list[Resolution]] = {}

                rules_by_id: dict[RuleId, Rule] = {rule.id: rule for rule in usable_rules}
                for match in matches:
                    rules_by_id.setdefault(match.rule.id, match.rule)

                tags_by_id: dict[GroupId, Group] = {
                    group.id: group for group in await self._group_store.list_groups()
                }
                rules_by_tag: dict[GroupId, list[Rule]] = defaultdict(list)
                for rule in usable_rules:
                    for group_id in rule.groups:
                        rules_by_tag[group_id].append(rule)

                current_matches = list(matches)

                for iteration in range(self.MAX_ITERATIONS):
                    self._logger.trace(f"CompassRelationalResolver iteration {iteration + 1}")

                    filtered_by_deps = await self._apply_dependencies(
                        current_matches,
                        cache,
                        rules_by_tag,
                        rules_by_id,
                        tags_by_id,
                        resolutions,
                    )

                    new_matches = self._filter_highest_priority_matches(
                        filtered_by_deps,
                        resolutions,
                    )

                    if self._matches_equal(new_matches, current_matches):
                        self._logger.trace(
                            f"CompassRelationalResolver converged after {iteration + 1} iteration(s)"
                        )
                        current_matches = new_matches
                        break

                    current_matches = new_matches
                else:
                    self._logger.trace(
                        f"CompassRelationalResolver reached max iterations ({self.MAX_ITERATIONS})"
                    )

                all_entities = {
                    ResolvedEntity.rule(match.rule) for match in [*matches, *current_matches]
                }
                for entity in all_entities:
                    if entity not in resolutions:
                        resolutions[entity] = [
                            Resolution(
                                kind=ResolutionKind.NONE,
                                details=ResolutionDetails(description="No relational changes"),
                            )
                        ]

                return RelationalResolverResult(matches=current_matches, resolutions=resolutions)

    def _filter_highest_priority_matches(
        self,
        matches: Sequence[RuleMatch],
        resolutions: dict[ResolvedEntity, list[Resolution]],
    ) -> list[RuleMatch]:
        if not matches:
            return []

        max_priority = max(match.rule.priority for match in matches)
        winners = tuple(
            ResolvedEntity.rule(match.rule)
            for match in matches
            if match.rule.priority >= max_priority
        )

        filtered: list[RuleMatch] = []
        for match in matches:
            if match.rule.priority >= max_priority:
                filtered.append(match)
                continue

            self._logger.debug(
                f"Dropped (lower priority): Rule {match.rule.id} "
                f"({match.rule.content.action}) - {match.rule.priority} < {max_priority}"
            )
            resolutions.setdefault(ResolvedEntity.rule(match.rule), []).append(
                Resolution(
                    kind=ResolutionKind.DEPRIORITIZED,
                    details=ResolutionDetails(
                        description=(
                            "Filtered due to lower priority "
                            f"({match.rule.priority} < {max_priority})"
                        ),
                        counterparts=winners,
                    ),
                )
            )

        return filtered

    async def _apply_dependencies(
        self,
        matches: Sequence[RuleMatch],
        cache: _RelationshipCache,
        rules_by_tag: dict[GroupId, list[Rule]],
        rules_by_id: dict[RuleId, Rule],
        tags_by_id: dict[GroupId, Group],
        resolutions: dict[ResolvedEntity, list[Resolution]],
    ) -> Sequence[RuleMatch]:
        matched_ids = {match.rule.id for match in matches}

        tag_to_matched: dict[GroupId, set[RuleId]] = defaultdict(set)
        for match in matches:
            for group_id in match.rule.groups:
                tag_to_matched[group_id].add(match.rule.id)

        and_deps: dict[RuleId, list[_DependencyTarget]] = {}
        or_groups: dict[RuleId, dict[str, list[_DependencyTarget]]] = {}
        topo_edges: dict[RuleId, set[RuleId]] = {match.rule.id: set() for match in matches}

        for match in matches:
            gid: RuleId = match.rule.id
            source_ids: list[RuleId | GroupId] = [gid, *match.rule.groups]
            relationships = await self._gather_dependency_relationships(source_ids, cache)

            gid_and: list[_DependencyTarget] = []
            gid_or: dict[str, list[_DependencyTarget]] = {}

            for relationship in relationships:
                target = self._resolve_dependency_target(
                    relationship,
                    gid,
                    matched_ids,
                    tag_to_matched,
                    rules_by_tag,
                    topo_edges,
                )
                if relationship.kind == RelationshipKind.DEPENDENCY_ANY and relationship.group_id:
                    gid_or.setdefault(relationship.group_id, []).append(target)
                else:
                    gid_and.append(target)

            and_deps[gid] = gid_and
            if gid_or:
                or_groups[gid] = gid_or

        topo_order = self._topological_sort(topo_edges)
        surviving = set(matched_ids)

        for gid in topo_order:
            if gid not in surviving:
                continue

            failed = False

            for dep in and_deps.get(gid, []):
                if self._is_dep_target_met(dep, surviving):
                    continue

                failed = True
                counterparts: tuple[ResolvedEntity, ...] = ()
                if dep.target_id is not None:
                    wrapped = self._resolve_counterpart(dep.target_id, rules_by_id, tags_by_id)
                    if wrapped is not None:
                        counterparts = (wrapped,)
                resolutions.setdefault(ResolvedEntity.rule(rules_by_id[gid]), []).append(
                    Resolution(
                        kind=ResolutionKind.UNMET_DEPENDENCY_ALL,
                        details=ResolutionDetails(
                            description=f"AND dependency target {dep.target_id} not met",
                            relationship=dep.relationship,
                            counterparts=counterparts,
                        ),
                    )
                )

            for dependency_group_id, targets in or_groups.get(gid, {}).items():
                if any(self._is_dep_target_met(dep, surviving) for dep in targets):
                    continue

                failed = True
                group_counterparts = tuple(
                    wrapped
                    for dep in targets
                    if dep.target_id is not None
                    and (
                        wrapped := self._resolve_counterpart(dep.target_id, rules_by_id, tags_by_id)
                    )
                    is not None
                )
                group_relationship = next(
                    (dep.relationship for dep in targets if dep.relationship),
                    None,
                )
                resolutions.setdefault(ResolvedEntity.rule(rules_by_id[gid]), []).append(
                    Resolution(
                        kind=ResolutionKind.UNMET_DEPENDENCY_ANY,
                        details=ResolutionDetails(
                            description=(
                                f"OR dependency group '{dependency_group_id}' not met - "
                                f"none of {[c.entity.id for c in group_counterparts]} active"
                            ),
                            relationship=group_relationship,
                            counterparts=group_counterparts,
                        ),
                    )
                )

            if failed:
                surviving.discard(gid)
                self._logger.debug(f"Dropped (unmet dependency): Rule {gid}")

        return [match for match in matches if match.rule.id in surviving]

    async def _gather_dependency_relationships(
        self,
        source_ids: Sequence[RuleId | GroupId],
        cache: _RelationshipCache,
    ) -> list[Relationship]:
        result: list[Relationship] = []
        seen: set[RelationshipId] = set()

        for source_id in source_ids:
            for kind in (RelationshipKind.DEPENDENCY, RelationshipKind.DEPENDENCY_ANY):
                for relationship in await self._get_relationships(
                    cache, kind, False, source_id=source_id
                ):
                    if relationship.id in seen:
                        continue

                    result.append(relationship)
                    seen.add(relationship.id)

        return result

    def _resolve_dependency_target(
        self,
        relationship: Relationship,
        gid: RuleId,
        matched_ids: set[RuleId],
        tag_to_matched: dict[GroupId, set[RuleId]],
        rules_by_tag: dict[GroupId, list[Rule]],
        topo_edges: dict[RuleId, set[RuleId]],
    ) -> _DependencyTarget:
        if relationship.target.kind == RelationshipEntityKind.RULE:
            dep_target_id = cast(RuleId, relationship.target.id)
            if dep_target_id not in matched_ids:
                return _DependencyTarget(
                    kind=_DependencyTargetKind.UNMET,
                    relationship=relationship,
                    target_id=dep_target_id,
                )

            if dep_target_id != gid:
                topo_edges[gid].add(dep_target_id)

            return _DependencyTarget(
                kind=_DependencyTargetKind.MATCHED_RULE,
                rule_ids={dep_target_id},
                relationship=relationship,
                target_id=dep_target_id,
            )

        if relationship.target.kind.is_group:
            group_id = cast(GroupId, relationship.target.id)
            all_member_ids = {rule.id for rule in rules_by_tag.get(group_id, [])}
            all_member_ids.update(tag_to_matched.get(group_id, set()))
            matched_members = all_member_ids & matched_ids

            for member_id in matched_members:
                if member_id != gid:
                    topo_edges[gid].add(member_id)

            if relationship.target.kind == RelationshipEntityKind.GROUP_ANY:
                if not matched_members:
                    return _DependencyTarget(
                        kind=_DependencyTargetKind.UNMET,
                        relationship=relationship,
                        target_id=group_id,
                    )

                return _DependencyTarget(
                    kind=_DependencyTargetKind.ANY_MATCHED_GROUP_MEMBER,
                    rule_ids=matched_members,
                    relationship=relationship,
                    target_id=group_id,
                )

            if not all_member_ids or all_member_ids - matched_ids:
                return _DependencyTarget(
                    kind=_DependencyTargetKind.UNMET,
                    relationship=relationship,
                    target_id=group_id,
                )

            return _DependencyTarget(
                kind=_DependencyTargetKind.MATCHED_RULE,
                rule_ids=matched_members,
                relationship=relationship,
                target_id=group_id,
            )

        return _DependencyTarget(kind=_DependencyTargetKind.UNMET)

    @staticmethod
    def _topological_sort(edges: dict[RuleId, set[RuleId]]) -> list[RuleId]:
        in_degree: dict[RuleId, int] = {gid: 0 for gid in edges}
        reverse: dict[RuleId, set[RuleId]] = defaultdict(set)

        for gid, targets in edges.items():
            for dep_id in targets:
                if dep_id not in in_degree:
                    continue

                in_degree[gid] += 1
                reverse[dep_id].add(gid)

        queue: deque[RuleId] = deque(gid for gid, degree in in_degree.items() if degree == 0)
        order: list[RuleId] = []

        while queue:
            gid = queue.popleft()
            order.append(gid)

            for dependent in reverse.get(gid, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return order

    @staticmethod
    def _is_dep_target_met(dep: _DependencyTarget, surviving: set[RuleId]) -> bool:
        if dep.kind == _DependencyTargetKind.UNMET:
            return False
        if dep.kind == _DependencyTargetKind.MATCHED_RULE:
            return dep.rule_ids <= surviving
        if dep.kind == _DependencyTargetKind.ANY_MATCHED_GROUP_MEMBER:
            return bool(dep.rule_ids & surviving)
        return False

    @staticmethod
    def _resolve_counterpart(
        raw_id: RuleId | GroupId,
        rules_by_id: dict[RuleId, Rule],
        tags_by_id: dict[GroupId, Group],
    ) -> ResolvedEntity | None:
        raw = cast(str, raw_id)
        if raw in rules_by_id:
            return ResolvedEntity.rule(rules_by_id[cast(RuleId, raw)])
        if raw in tags_by_id:
            return ResolvedEntity.group(tags_by_id[cast(GroupId, raw)])
        return None

    async def _get_relationships(
        self,
        cache: _RelationshipCache,
        kind: RelationshipKind,
        indirect: bool,
        source_id: RuleId | GroupId | ToolId | None = None,
        target_id: RuleId | GroupId | ToolId | None = None,
    ) -> list[Relationship]:
        entity_id = source_id if source_id else target_id
        assert entity_id is not None

        direction = "source" if source_id else "target"
        key: _CacheKey = (kind, indirect, direction, entity_id)

        if key in cache:
            return list(cache[key])

        if source_id:
            cache[key] = list(
                await self._relationship_store.list_relationships(
                    kind=kind,
                    indirect=indirect,
                    source_id=source_id,
                )
            )
        else:
            cache[key] = list(
                await self._relationship_store.list_relationships(
                    kind=kind,
                    indirect=indirect,
                    target_id=target_id,
                )
            )

        return list(cache[key])

    @staticmethod
    def _matches_equal(a: Sequence[RuleMatch], b: Sequence[RuleMatch]) -> bool:
        if len(a) != len(b):
            return False
        return {match.rule.id for match in a} == {match.rule.id for match in b}
