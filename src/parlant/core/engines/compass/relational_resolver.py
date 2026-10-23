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

"""Relationship resolution for Compass rule evaluations.

Compass supports dependency, priority (relational and numeric), and entailment
relationships. Journeys are out of scope.

The resolutions are deliberately separate entry points, because the matcher
applies them to different sets: dependency filtering and entailment expansion
gate/extend everything (session admission and turn matches alike), while
priority filtering — relational and numeric — applies only to the turn-matched
set. A deprioritized rule still enters the session working set, and still
satisfies its dependents' dependencies; it just cannot appear in the turn
matches alongside the rule that outprioritized it.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from collections.abc import Set as AbstractSet
from typing import Literal, Sequence, TypeAlias, cast

from parlant.core.engines.rule_evaluation import RuleEvaluation
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
    ENTAILED = "entailed"


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
    evaluations: Sequence[RuleEvaluation]
    resolutions: dict[ResolvedEntity, list[Resolution]] = field(default_factory=dict)


class _DependencyTargetKind(Enum):
    RELEVANT_RULE = auto()
    ANY_RELEVANT_GROUP_MEMBER = auto()
    UNMET = auto()


@dataclass
class _DependencyTarget:
    kind: _DependencyTargetKind
    rule_ids: set[RuleId] = field(default_factory=set)
    relationship: Relationship | None = None
    target_id: ResolvedEntityId | GroupId | None = None


class RelationalResolver:
    """Resolves Compass rule dependencies and numeric priority."""

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

    async def filter_unmet_dependencies(
        self,
        usable_rules: Sequence[Rule],
        evaluations: Sequence[RuleEvaluation],
        satisfied_rule_ids: AbstractSet[RuleId] = frozenset(),
        same_turn_satisfier_ids: AbstractSet[RuleId] | None = None,
    ) -> RelationalResolverResult:
        """Drop evaluations whose dependencies are unmet.

        A dependency is met by another satisfying evaluation in the batch
        (same-turn chains resolve in dependency order) or by any rule in
        ``satisfied_rule_ids`` — the caller's record of rules whose relevance
        is already established (in Compass: the session's matched-rule
        ledger).

        ``same_turn_satisfier_ids`` restricts which of the batch's evaluations
        can satisfy dependencies (in Compass: turn matches — a discovery
        admission is gated like everything else but satisfies nobody). ``None``
        lets every evaluation satisfy."""
        with self._logger.scope("RelationalResolver"):
            with self._tracer.span("compass.relational_resolver.dependencies"):
                cache: _RelationshipCache = {}
                resolutions: dict[ResolvedEntity, list[Resolution]] = {}

                rules_by_id: dict[RuleId, Rule] = {rule.id: rule for rule in usable_rules}
                for evaluation in evaluations:
                    rules_by_id.setdefault(evaluation.rule.id, evaluation.rule)

                tags_by_id: dict[GroupId, Group] = {
                    group.id: group for group in await self._group_store.list_groups()
                }
                rules_by_tag: dict[GroupId, list[Rule]] = defaultdict(list)
                for rule in usable_rules:
                    for group_id in rule.groups:
                        rules_by_tag[group_id].append(rule)

                filtered = await self._apply_dependencies(
                    evaluations,
                    cache,
                    rules_by_tag,
                    rules_by_id,
                    tags_by_id,
                    resolutions,
                    satisfied_rule_ids,
                    same_turn_satisfier_ids,
                )

                self._backfill_none_resolutions([*evaluations, *filtered], resolutions)

                return RelationalResolverResult(evaluations=list(filtered), resolutions=resolutions)

    def keep_highest_priority(
        self,
        evaluations: Sequence[RuleEvaluation],
        entailed_rule_ids: AbstractSet[RuleId] = frozenset(),
    ) -> RelationalResolverResult:
        """Keep only the evaluations sharing the set's top numeric priority.

        Entailed evaluations owe their activation to their entailer, so they
        neither raise the priority bar nor get filtered by it."""
        with self._logger.scope("RelationalResolver"):
            with self._tracer.span("compass.relational_resolver.priority"):
                resolutions: dict[ResolvedEntity, list[Resolution]] = {}

                filtered = self._filter_highest_priority_evaluations(
                    evaluations, resolutions, entailed_rule_ids
                )

                self._backfill_none_resolutions([*evaluations, *filtered], resolutions)

                return RelationalResolverResult(evaluations=filtered, resolutions=resolutions)

    async def resolve_priority_relationships(
        self,
        usable_rules: Sequence[Rule],
        evaluations: Sequence[RuleEvaluation],
    ) -> RelationalResolverResult:
        """Drop evaluations outprioritized by a PRIORITY relationship.

        An evaluation is deprioritized when a direct PRIORITY relationship
        targets its rule (or one of the rule's groups) and the relationship's
        source is another rule in the input set, or a group with such a member.
        Only direct relationships apply — priority does not propagate through
        inactive intermediaries (the reinstatement principle) — but sources are
        judged against the full input set, so a dropped rule still suppresses
        its own targets and chains collapse in a single pass. Group-mediated
        chains (e.g. G1 → T1 → G2) work by traversing unmatched members of
        source groups and querying their own priority relationships."""
        with self._logger.scope("RelationalResolver"):
            with self._tracer.span("compass.relational_resolver.relational_priority"):
                cache: _RelationshipCache = {}
                resolutions: dict[ResolvedEntity, list[Resolution]] = {}

                rules_by_id: dict[RuleId, Rule] = {rule.id: rule for rule in usable_rules}
                for evaluation in evaluations:
                    rules_by_id.setdefault(evaluation.rule.id, evaluation.rule)

                rules_by_tag: dict[GroupId, list[Rule]] = defaultdict(list)
                for rule in usable_rules:
                    for group_id in rule.groups:
                        rules_by_tag[group_id].append(rule)

                match_ids = {evaluation.rule.id for evaluation in evaluations}

                tag_to_matched: dict[GroupId, set[RuleId]] = defaultdict(set)
                for evaluation in evaluations:
                    for group_id in evaluation.rule.groups:
                        tag_to_matched[group_id].add(evaluation.rule.id)

                surviving: list[RuleEvaluation] = []
                for evaluation in evaluations:
                    deprioritizer = await self._find_deprioritizer(
                        evaluation,
                        match_ids,
                        tag_to_matched,
                        rules_by_tag,
                        cache,
                    )

                    if deprioritizer is None:
                        surviving.append(evaluation)
                        continue

                    prioritizer_id, relationship = deprioritizer
                    self._logger.debug(
                        f"Dropped (deprioritized): Rule {evaluation.rule.id} "
                        f"({evaluation.rule.content or evaluation.rule.condition}) - "
                        f"outprioritized by {prioritizer_id}"
                    )

                    prioritizer = rules_by_id.get(prioritizer_id)
                    counterparts = (
                        (ResolvedEntity.rule(prioritizer),) if prioritizer is not None else ()
                    )
                    resolutions.setdefault(ResolvedEntity.rule(evaluation.rule), []).append(
                        Resolution(
                            kind=ResolutionKind.DEPRIORITIZED,
                            details=ResolutionDetails(
                                description=f"Deprioritized by rule {prioritizer_id}",
                                relationship=relationship,
                                counterparts=counterparts,
                            ),
                        )
                    )

                self._backfill_none_resolutions([*evaluations, *surviving], resolutions)

                return RelationalResolverResult(evaluations=surviving, resolutions=resolutions)

    async def _find_deprioritizer(
        self,
        evaluation: RuleEvaluation,
        match_ids: set[RuleId],
        tag_to_matched: dict[GroupId, set[RuleId]],
        rules_by_tag: dict[GroupId, list[Rule]],
        cache: _RelationshipCache,
    ) -> tuple[RuleId, Relationship] | None:
        """Find a matched rule that outprioritizes this evaluation, if any.

        Returns the (direct deprioritizer's rule id, causing relationship)."""
        priority_rels = await self._get_priority_relationships_for_rule(evaluation.rule, cache)

        # Guards the group-member traversal against group cycles.
        traversed: set[RuleId] = set()

        while priority_rels:
            relationship = priority_rels.pop()
            source = relationship.source

            if source.kind == RelationshipEntityKind.RULE:
                if source.id in match_ids and source.id != evaluation.rule.id:
                    return cast(RuleId, source.id), relationship

            elif source.kind.is_group:
                source_group_id = cast(GroupId, source.id)
                group_rules = rules_by_tag.get(source_group_id, [])

                # Any matched member of the source group deprioritizes the
                # target — persisted members first, then transient (matched
                # but unstored) ones.
                prioritizer_id = next(
                    (
                        rule.id
                        for rule in group_rules
                        if rule.id in match_ids and rule.id != evaluation.rule.id
                    ),
                    None,
                ) or next(
                    (
                        rule_id
                        for rule_id in tag_to_matched.get(source_group_id, set())
                        if rule_id != evaluation.rule.id
                    ),
                    None,
                )
                if prioritizer_id is not None:
                    return prioritizer_id, relationship

                # No member matched: traverse into unmatched members so that
                # group-mediated chains (G1 → T1 → G2) are still honored via
                # the members' own priority relationships.
                for rule in group_rules:
                    if rule.id in traversed or rule.id in match_ids:
                        continue

                    priority_rels.extend(
                        await self._get_relationships(
                            cache, RelationshipKind.PRIORITY, False, target_id=rule.id
                        )
                    )
                    for member_group_id in rule.groups:
                        priority_rels.extend(
                            await self._get_relationships(
                                cache,
                                RelationshipKind.PRIORITY,
                                False,
                                target_id=member_group_id,
                            )
                        )

                traversed.update(rule.id for rule in group_rules if rule.id not in match_ids)

        return None

    async def _get_priority_relationships_for_rule(
        self,
        rule: Rule,
        cache: _RelationshipCache,
    ) -> list[Relationship]:
        """Gather all PRIORITY relationships directly targeting this rule.

        Uses ``indirect=False`` so that priority does not propagate through
        inactive intermediaries (the reinstatement principle). The rule's
        groups are queried as additional targets so group-level priority
        applies to every member."""
        relationships = await self._get_relationships(
            cache, RelationshipKind.PRIORITY, False, target_id=rule.id
        )

        for group_id in rule.groups:
            relationships.extend(
                await self._get_relationships(
                    cache, RelationshipKind.PRIORITY, False, target_id=group_id
                )
            )

        return relationships

    async def expand_entailment(
        self,
        usable_rules: Sequence[Rule],
        evaluations: Sequence[RuleEvaluation],
    ) -> RelationalResolverResult:
        """Activate additional rules implied by entailment relationships.

        Follows ENTAILMENT relationships (indirect closure) from each
        evaluation's rule, traversing group targets into their members.
        The result contains the input evaluations plus one evaluation per
        newly activated rule, each carrying an ENTAILED resolution per
        entailing source."""
        with self._logger.scope("RelationalResolver"):
            with self._tracer.span("compass.relational_resolver.entailment"):
                cache: _RelationshipCache = {}
                resolutions: dict[ResolvedEntity, list[Resolution]] = {}

                rules_by_id: dict[RuleId, Rule] = {rule.id: rule for rule in usable_rules}
                for evaluation in evaluations:
                    rules_by_id.setdefault(evaluation.rule.id, evaluation.rule)

                rules_by_tag: dict[GroupId, list[Rule]] = defaultdict(list)
                for rule in usable_rules:
                    for group_id in rule.groups:
                        rules_by_tag[group_id].append(rule)

                match_ids = {evaluation.rule.id for evaluation in evaluations}

                # Map: entailed rule id -> (source rule, relationship) pairs.
                entailed_sources: dict[RuleId, list[tuple[Rule, Relationship]]] = defaultdict(list)

                for evaluation in evaluations:
                    relationships = await self._get_relationships(
                        cache,
                        RelationshipKind.ENTAILMENT,
                        True,
                        source_id=evaluation.rule.id,
                    )

                    while relationships:
                        relationship = relationships.pop()

                        if relationship.target.kind == RelationshipEntityKind.RULE:
                            target_id = cast(RuleId, relationship.target.id)
                            if target_id in match_ids:
                                continue

                            if target_id in rules_by_id:
                                entailed_sources[target_id].append((evaluation.rule, relationship))

                        elif relationship.target.kind.is_group:
                            group_id = cast(GroupId, relationship.target.id)
                            for member in rules_by_tag.get(group_id, []):
                                if member.id not in match_ids:
                                    entailed_sources[member.id].append(
                                        (evaluation.rule, relationship)
                                    )
                                relationships.extend(
                                    await self._get_relationships(
                                        cache,
                                        RelationshipKind.ENTAILMENT,
                                        True,
                                        source_id=member.id,
                                    )
                                )

                entailed_evaluations: list[RuleEvaluation] = []
                for entailed_id, sources in entailed_sources.items():
                    entailed_rule = rules_by_id[entailed_id]
                    entailed_evaluations.append(
                        RuleEvaluation(
                            rule=entailed_rule,
                            reasoning=(
                                "[Activated via entailment] Automatically inferred from context"
                            ),
                            metadata={"entailed": True},
                        )
                    )

                    for source_rule, relationship in sources:
                        resolutions.setdefault(ResolvedEntity.rule(entailed_rule), []).append(
                            Resolution(
                                kind=ResolutionKind.ENTAILED,
                                details=ResolutionDetails(
                                    description=f"Entailed by rule {source_rule.id}",
                                    relationship=relationship,
                                    counterparts=(ResolvedEntity.rule(source_rule),),
                                ),
                            )
                        )

                result = [*evaluations, *entailed_evaluations]
                self._backfill_none_resolutions(result, resolutions)

                return RelationalResolverResult(evaluations=result, resolutions=resolutions)

    def _backfill_none_resolutions(
        self,
        evaluations: Sequence[RuleEvaluation],
        resolutions: dict[ResolvedEntity, list[Resolution]],
    ) -> None:
        for entity in {ResolvedEntity.rule(evaluation.rule) for evaluation in evaluations}:
            if entity not in resolutions:
                resolutions[entity] = [
                    Resolution(
                        kind=ResolutionKind.NONE,
                        details=ResolutionDetails(description="No relational changes"),
                    )
                ]

    def _filter_highest_priority_evaluations(
        self,
        evaluations: Sequence[RuleEvaluation],
        resolutions: dict[ResolvedEntity, list[Resolution]],
        entailed_rule_ids: AbstractSet[RuleId] = frozenset(),
    ) -> list[RuleEvaluation]:
        if not evaluations:
            return []

        competing = [
            evaluation for evaluation in evaluations if evaluation.rule.id not in entailed_rule_ids
        ]
        if not competing:
            return list(evaluations)

        max_priority = max(evaluation.rule.priority for evaluation in competing)
        winners = tuple(
            ResolvedEntity.rule(evaluation.rule)
            for evaluation in competing
            if evaluation.rule.priority >= max_priority
        )

        filtered: list[RuleEvaluation] = []
        for evaluation in evaluations:
            if evaluation.rule.id in entailed_rule_ids or evaluation.rule.priority >= max_priority:
                filtered.append(evaluation)
                continue

            self._logger.debug(
                f"Dropped (lower priority): Rule {evaluation.rule.id} "
                f"({evaluation.rule.content or evaluation.rule.condition}) - "
                f"{evaluation.rule.priority} < {max_priority}"
            )
            resolutions.setdefault(ResolvedEntity.rule(evaluation.rule), []).append(
                Resolution(
                    kind=ResolutionKind.DEPRIORITIZED,
                    details=ResolutionDetails(
                        description=(
                            "Filtered due to lower priority "
                            f"({evaluation.rule.priority} < {max_priority})"
                        ),
                        counterparts=winners,
                    ),
                )
            )

        return filtered

    async def _apply_dependencies(
        self,
        evaluations: Sequence[RuleEvaluation],
        cache: _RelationshipCache,
        rules_by_tag: dict[GroupId, list[Rule]],
        rules_by_id: dict[RuleId, Rule],
        tags_by_id: dict[GroupId, Group],
        resolutions: dict[ResolvedEntity, list[Resolution]],
        satisfied_rule_ids: AbstractSet[RuleId] = frozenset(),
        same_turn_satisfier_ids: AbstractSet[RuleId] | None = None,
    ) -> Sequence[RuleEvaluation]:
        # Satisfied rules count as met dependency targets throughout, but are
        # not themselves re-judged. When the caller restricts same-turn
        # satisfiers, only that subset of the batch (plus the satisfied set)
        # can meet targets — every evaluation is still gated either way.
        evaluation_ids = {evaluation.rule.id for evaluation in evaluations}
        if same_turn_satisfier_ids is None:
            relevant_ids = evaluation_ids | set(satisfied_rule_ids)
        else:
            relevant_ids = (evaluation_ids & set(same_turn_satisfier_ids)) | set(satisfied_rule_ids)

        group_to_relevant: dict[GroupId, set[RuleId]] = defaultdict(set)
        for evaluation in evaluations:
            if evaluation.rule.id not in relevant_ids:
                continue
            for group_id in evaluation.rule.groups:
                group_to_relevant[group_id].add(evaluation.rule.id)

        and_deps: dict[RuleId, list[_DependencyTarget]] = {}
        or_groups: dict[RuleId, dict[str, list[_DependencyTarget]]] = {}
        topo_edges: dict[RuleId, set[RuleId]] = {
            evaluation.rule.id: set() for evaluation in evaluations
        }

        for evaluation in evaluations:
            gid: RuleId = evaluation.rule.id
            source_ids: list[RuleId | GroupId] = [gid, *evaluation.rule.groups]
            relationships = await self._gather_dependency_relationships(source_ids, cache)

            gid_and: list[_DependencyTarget] = []
            gid_or: dict[str, list[_DependencyTarget]] = {}

            for relationship in relationships:
                target = self._resolve_dependency_target(
                    relationship,
                    gid,
                    relevant_ids,
                    group_to_relevant,
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
        # ``surviving`` is the satisfaction pool; ``dropped`` tracks gated-out
        # evaluations (a non-satisfier evaluation is gated without ever being
        # in the pool).
        surviving = set(relevant_ids)
        dropped: set[RuleId] = set()

        for gid in topo_order:
            if gid in dropped:
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
                dropped.add(gid)
                surviving.discard(gid)
                self._logger.debug(f"Dropped (unmet dependency): Rule {gid}")

        return [evaluation for evaluation in evaluations if evaluation.rule.id not in dropped]

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
        relevant_ids: set[RuleId],
        group_to_relevant: dict[GroupId, set[RuleId]],
        rules_by_tag: dict[GroupId, list[Rule]],
        topo_edges: dict[RuleId, set[RuleId]],
    ) -> _DependencyTarget:
        if relationship.target.kind == RelationshipEntityKind.RULE:
            dep_target_id = cast(RuleId, relationship.target.id)
            if dep_target_id not in relevant_ids:
                return _DependencyTarget(
                    kind=_DependencyTargetKind.UNMET,
                    relationship=relationship,
                    target_id=dep_target_id,
                )

            if dep_target_id != gid:
                topo_edges[gid].add(dep_target_id)

            return _DependencyTarget(
                kind=_DependencyTargetKind.RELEVANT_RULE,
                rule_ids={dep_target_id},
                relationship=relationship,
                target_id=dep_target_id,
            )

        if relationship.target.kind.is_group:
            group_id = cast(GroupId, relationship.target.id)
            all_member_ids = {rule.id for rule in rules_by_tag.get(group_id, [])}
            all_member_ids.update(group_to_relevant.get(group_id, set()))
            relevant_members = all_member_ids & relevant_ids

            for member_id in relevant_members:
                if member_id != gid:
                    topo_edges[gid].add(member_id)

            if relationship.target.kind == RelationshipEntityKind.GROUP_ANY:
                if not relevant_members:
                    return _DependencyTarget(
                        kind=_DependencyTargetKind.UNMET,
                        relationship=relationship,
                        target_id=group_id,
                    )

                return _DependencyTarget(
                    kind=_DependencyTargetKind.ANY_RELEVANT_GROUP_MEMBER,
                    rule_ids=relevant_members,
                    relationship=relationship,
                    target_id=group_id,
                )

            if not all_member_ids or all_member_ids - relevant_ids:
                return _DependencyTarget(
                    kind=_DependencyTargetKind.UNMET,
                    relationship=relationship,
                    target_id=group_id,
                )

            return _DependencyTarget(
                kind=_DependencyTargetKind.RELEVANT_RULE,
                rule_ids=relevant_members,
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
        if dep.kind == _DependencyTargetKind.RELEVANT_RULE:
            return dep.rule_ids <= surviving
        if dep.kind == _DependencyTargetKind.ANY_RELEVANT_GROUP_MEMBER:
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
