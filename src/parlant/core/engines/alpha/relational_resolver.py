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

"""Relational resolver for guideline matching.

Resolves relationships between matched guidelines, including dependencies,
priorities, and entailment. The resolver iterates until stable, applying
each step in order:

  1. **Dependencies** — Filter guidelines whose dependency targets are not met.
     Uses topological sorting for single-pass resolution within each iteration.
  2. **Relational prioritization** — Filter guidelines that are deprioritized by
     higher-priority guidelines, tags, or journeys. Includes transitive filtering
     of guidelines that depend on deprioritized entities.
  3. **Numerical priority** — Keep only entities at the highest priority level.
     Runs before entailment so that entailed guidelines cannot cause their
     entailer to be filtered by having a higher priority.
  4. **Entailment** — Activate additional guidelines implied by matched ones.

The iteration loop (steps 1–4) runs until the set of matches stabilizes or
MAX_ITERATIONS is reached. This handles cross-step interactions, e.g. when
priority filtering removes a guideline that was a dependency target.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal, Optional, Sequence, TypeAlias, cast

from parlant.core.common import JSONSerializable
from parlant.core.journeys import Journey, JourneyId
from parlant.core.loggers import Logger
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.relationships import (
    Relationship,
    RelationshipEntityKind,
    RelationshipId,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.tags import Tag, TagId, TagStore
from parlant.core.tools import ToolId
from parlant.core.tracer import Tracer


# ---------------------------------------------------------------------------
# Type aliases for the relationship cache used throughout the resolver.
# ---------------------------------------------------------------------------

_CacheKey = tuple[RelationshipKind, bool, str, GuidelineId | TagId | ToolId]
_RelationshipCache = dict[_CacheKey, list[Relationship]]


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

#: Union of entity IDs that can appear as resolver map keys or targets.
ResolvedEntityId: TypeAlias = GuidelineId | JourneyId


@dataclass(frozen=True)
class ResolvedEntity:
    """A typed wrapper identifying an entity participating in resolution.

    The ``entity_type`` field discriminates the union and supports cheap
    equality / hashing without comparing full entity payloads. Two
    ``ResolvedEntity`` values are equal iff they share both the type tag
    and the underlying entity's id.
    """

    entity_type: Literal["guideline", "journey", "tag"]
    entity: Guideline | Journey | Tag

    def __hash__(self) -> int:
        return hash((self.entity_type, self.entity.id))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ResolvedEntity)
            and self.entity_type == other.entity_type
            and self.entity.id == other.entity.id
        )

    @classmethod
    def guideline(cls, g: Guideline) -> ResolvedEntity:
        return cls(entity_type="guideline", entity=g)

    @classmethod
    def journey(cls, j: Journey) -> ResolvedEntity:
        return cls(entity_type="journey", entity=j)

    @classmethod
    def tag(cls, t: Tag) -> ResolvedEntity:
        return cls(entity_type="tag", entity=t)


class ResolutionKind(str, Enum):
    """The type of decision the resolver made about an entity."""

    NONE = "none"
    """No relational changes — the entity passed through as-is."""

    UNMET_DEPENDENCY_ALL = "unmet_dependency_all"
    """Removed: a ``depend_on()`` target was not active (AND semantics)."""

    UNMET_DEPENDENCY_ANY = "unmet_dependency_any"
    """Removed: all targets in a ``depend_on_any()`` OR group were inactive."""

    DEPRIORITIZED = "deprioritized"
    """Removed: a higher-priority entity took precedence (relational,
    numerical, or transitive)."""

    ENTAILED = "entailed"
    """Added: activated via an entailment relationship from a matched guideline."""


@dataclass(frozen=True)
class ResolutionDetails:
    """Structured information about why a resolution decision was made."""

    description: str
    """Human-readable explanation."""

    relationship: Relationship | None = None
    """The relationship that caused this resolution, if applicable."""

    counterparts: tuple[ResolvedEntity, ...] = ()
    """The other entities involved in the relationship that caused this
    resolution — e.g. the unmet dependency targets, the prioritizing
    entity, or the entailing guideline."""


@dataclass(frozen=True)
class Resolution:
    """A single decision the resolver made about an entity."""

    kind: ResolutionKind
    details: ResolutionDetails


@dataclass
class RelationalResolverResult:
    """Output of the relational resolver."""

    matches: Sequence[GuidelineMatch]
    """Guidelines that survived resolution — includes both ordinary and
    journey node guidelines, as well as any entailed guidelines. These are
    the guidelines the agent should follow when generating its response."""

    journeys: Sequence[Journey]
    """Journeys that survived priority filtering. When competing journeys
    match the same context, lower-priority journeys are removed here.
    A deprioritized journey's node guidelines are also removed from
    ``matches``. The engine uses this list to track active journey paths
    and determine which node guidelines are eligible in subsequent
    preparation iterations."""

    resolutions: dict[ResolvedEntity, list[Resolution]] = field(default_factory=dict)
    """Map of entities to all resolution decisions made about them.
    Includes removed guidelines/journeys (UNMET_*, DEPRIORITIZED),
    added guidelines (ENTAILED), and unchanged entities (NONE).
    Every entity that entered the resolver gets an entry."""


# ---------------------------------------------------------------------------
# Internal types for dependency resolution
# ---------------------------------------------------------------------------


class _DependencyTargetKind(Enum):
    """How a single resolved dependency target should be evaluated."""

    MATCHED_GUIDELINE = auto()
    """The target is one or more specific guidelines. All of them must remain
    in the surviving set for the dependency to be satisfied."""

    ANY_MATCHED_TAG_MEMBER = auto()
    """The target is a tag with ANY semantics (TAG_ANY). The dependency is
    satisfied if at least one of the tag's matched member guidelines remains
    in the surviving set."""

    MET = auto()
    """The dependency is unconditionally satisfied (e.g. an active journey).
    No guideline IDs to track."""

    UNMET = auto()
    """The dependency is unconditionally failed (e.g. a guideline that was
    never matched, an inactive journey, or an empty/unmatched tag)."""


@dataclass
class _DependencyTarget:
    """A single resolved dependency of a matched guideline.

    Created during phase 1 of ``_apply_dependencies`` and consumed during
    phase 3 (the topological walk). The ``guideline_ids`` field is only
    meaningful for ``MATCHED_GUIDELINE`` and ``ANY_MATCHED_TAG_MEMBER``.
    """

    kind: _DependencyTargetKind
    guideline_ids: set[GuidelineId] = field(default_factory=set)
    relationship: Relationship | None = None
    target_id: ResolvedEntityId | TagId | None = None


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class RelationalResolver:
    """Resolves relational constraints between matched guidelines.

    See module docstring for the overall algorithm.
    """

    MAX_ITERATIONS = 3
    """Maximum number of resolution loop iterations.

    Each iteration runs dependencies → prioritization → numerical priority →
    entailment. Multiple iterations are needed because these steps interact:
    for example, priority filtering may remove a guideline that was a
    dependency target, requiring another dependency pass to cascade the
    removal.

    The dependency step itself is single-pass (topological sort), so
    iterations are only needed for cross-step interactions. In practice,
    2 iterations suffice for most cases; 3 provides a safety margin for
    deeper interaction chains (e.g. priority removes a target, which
    breaks a dependency, which causes entailment to drop a guideline).
    If the resolver doesn't converge within this limit, it logs a warning
    and returns the current state.
    """

    def __init__(
        self,
        relationship_store: RelationshipStore,
        tag_store: TagStore,
        logger: Logger,
        tracer: Tracer,
    ) -> None:
        self._relationship_store = relationship_store
        self._tag_store = tag_store
        self._logger = logger
        self._tracer = tracer

    # -- Public API ---------------------------------------------------------

    async def resolve(
        self,
        usable_guidelines: Sequence[Guideline],
        matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
    ) -> RelationalResolverResult:
        """Run the full resolution loop and return the final matches and journeys."""
        with self._logger.scope("GuidelineMatcher"):
            with self._logger.scope("RelationalResolver"):
                with self._tracer.span("relational_resolver.resolve"):
                    cache: _RelationshipCache = {}
                resolutions: dict[ResolvedEntity, list[Resolution]] = {}

                # Pre-compute per-call lookup maps so counterparts can be
                # resolved to full entity objects without hitting the store
                # in inner loops. Built fresh on every call because the
                # entity sets may have grown since __init__.
                guidelines_by_id: dict[GuidelineId, Guideline] = {
                    g.id: g for g in usable_guidelines
                }
                journeys_by_id: dict[JourneyId, Journey] = {j.id: j for j in journeys}
                tags_by_id: dict[TagId, Tag] = {t.id: t for t in await self._tag_store.list_tags()}

                # Build a tag → guidelines index from usable_guidelines so that
                # all tag lookups are in-memory instead of hitting the store.
                guidelines_by_tag: dict[TagId, list[Guideline]] = defaultdict(list)
                for g in usable_guidelines:
                    for tid in g.tags:
                        guidelines_by_tag[tid].append(g)

                initial_match_ids = {m.guideline.id for m in matches}
                initial_journey_ids = {j.id for j in journeys}
                # Keep the full original match list so that dep-failed
                # guidelines can be re-evaluated when entailment expands
                # the match set in a later iteration.
                all_candidate_matches: dict[GuidelineId, GuidelineMatch] = {
                    m.guideline.id: m for m in matches
                }
                current_matches = list(matches)
                current_journeys = list(journeys)
                entailed_ids: set[GuidelineId] = set()
                # Track guidelines removed by priority (relational or
                # numerical) — these must NOT be re-seeded because their
                # removal is authoritative, not contingent on match-set
                # expansion.
                priority_removed: set[GuidelineId] = set()

                for iteration in range(self.MAX_ITERATIONS):
                    self._logger.trace(f"RelationalResolver iteration {iteration + 1}")

                    # Re-seed: start from all candidates (original matches
                    # + entailed) minus those removed by priority.  This
                    # lets dep-failed guidelines be re-evaluated when the
                    # match set expanded (e.g. entailment added a dep
                    # target that was missing in the previous iteration).
                    # Clear stale dep-failure resolutions so they aren't
                    # duplicated if the guideline is re-evaluated.
                    candidate_ids = set(all_candidate_matches.keys()) - priority_removed
                    reseeded = [all_candidate_matches[gid] for gid in candidate_ids]

                    # Clear dep-failure resolutions for re-seeded
                    # guidelines so they can get fresh evaluations.
                    for gid in candidate_ids:
                        entity = ResolvedEntity.guideline(guidelines_by_id[gid])
                        if entity in resolutions:
                            resolutions[entity] = [
                                r
                                for r in resolutions[entity]
                                if r.kind
                                not in (
                                    ResolutionKind.UNMET_DEPENDENCY_ALL,
                                    ResolutionKind.UNMET_DEPENDENCY_ANY,
                                )
                            ]
                            if not resolutions[entity]:
                                del resolutions[entity]

                    # Step 1: Dependencies
                    filtered_by_deps = await self._apply_dependencies(
                        reseeded,
                        current_journeys,
                        cache,
                        guidelines_by_tag,
                        guidelines_by_id,
                        journeys_by_id,
                        tags_by_id,
                        resolutions,
                    )

                    # Step 2: Relational prioritization (includes transitive dep filtering)
                    prio_result = await self._apply_prioritization(
                        filtered_by_deps,
                        current_journeys,
                        cache,
                        guidelines_by_tag,
                        guidelines_by_id,
                        journeys_by_id,
                        tags_by_id,
                        resolutions,
                    )

                    # Track what priority removed this iteration
                    post_prio_ids = {m.guideline.id for m in prio_result.matches}
                    for m in filtered_by_deps:
                        if m.guideline.id not in post_prio_ids:
                            priority_removed.add(m.guideline.id)

                    # Step 3: Numerical priority filtering.
                    # Entailed guidelines are excluded — they were activated by
                    # implication and should not drive the priority ceiling or be
                    # filtered by it (otherwise they could cause their entailer
                    # to be removed for having a lower priority).
                    non_entailed = [
                        m for m in prio_result.matches if m.guideline.id not in entailed_ids
                    ]
                    entailed_matches = [
                        m for m in prio_result.matches if m.guideline.id in entailed_ids
                    ]
                    filtered_non_entailed, new_journeys = self.find_highest_priority_entities(
                        non_entailed, list(prio_result.journeys), resolutions
                    )
                    new_matches = filtered_non_entailed + entailed_matches

                    # Track numerical-priority removals
                    post_num_ids = {m.guideline.id for m in new_matches}
                    for m in list(prio_result.matches):
                        if m.guideline.id not in post_num_ids:
                            priority_removed.add(m.guideline.id)

                    # Step 4: Entailment.
                    # Exclude guidelines that were removed by priority — they
                    # must not be re-added.  Dep-failed guidelines are NOT
                    # excluded here: they may be re-seeded in the next
                    # iteration once the entailed guideline is available.
                    entailed_with_rels = await self._apply_entailment(
                        usable_guidelines, new_matches, cache, guidelines_by_tag
                    )
                    entailed: list[GuidelineMatch] = []
                    for m, sources in entailed_with_rels:
                        if (
                            m.guideline.id not in priority_removed
                            and m.guideline.id not in entailed_ids
                        ):
                            entailed.append(m)
                            entailed_ids.add(m.guideline.id)
                            all_candidate_matches[m.guideline.id] = m
                            for source_rel, source_gid in sources:
                                source_g = guidelines_by_id.get(source_gid)
                                counterparts = (
                                    (ResolvedEntity.guideline(source_g),)
                                    if source_g is not None
                                    else ()
                                )
                                resolutions.setdefault(
                                    ResolvedEntity.guideline(m.guideline), []
                                ).append(
                                    Resolution(
                                        kind=ResolutionKind.ENTAILED,
                                        details=ResolutionDetails(
                                            description=("Activated via entailment"),
                                            relationship=source_rel,
                                            counterparts=counterparts,
                                        ),
                                    )
                                )
                    new_matches = list(new_matches) + entailed

                    if self._matches_equal(new_matches, current_matches) and self._journeys_equal(
                        new_journeys, current_journeys
                    ):
                        self._logger.trace(
                            f"RelationalResolver converged after {iteration + 1} iteration(s)"
                        )
                        break

                    current_matches = new_matches
                    current_journeys = new_journeys
                else:
                    self._logger.trace(
                        f"RelationalResolver reached max iterations ({self.MAX_ITERATIONS})"
                    )

                # Add NONE resolutions for entities that passed through unchanged
                all_entities: set[ResolvedEntity] = set()
                all_entities.update(ResolvedEntity.guideline(m.guideline) for m in matches)
                all_entities.update(ResolvedEntity.journey(j) for j in journeys)
                all_entities.update(ResolvedEntity.guideline(m.guideline) for m in current_matches)
                all_entities.update(ResolvedEntity.journey(j) for j in current_journeys)
                for entity in all_entities:
                    if entity not in resolutions:
                        resolutions[entity] = [
                            Resolution(
                                kind=ResolutionKind.NONE,
                                details=ResolutionDetails(
                                    description="No relational changes",
                                ),
                            )
                        ]

                self._emit_tracer_events(
                    initial_match_ids, current_matches, matches, resolutions, guidelines_by_id
                )

                return RelationalResolverResult(
                    matches=current_matches,
                    journeys=current_journeys,
                    resolutions=resolutions,
                )

    def find_highest_priority_entities(
        self,
        matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
        resolutions: dict[ResolvedEntity, list[Resolution]],
    ) -> tuple[list[GuidelineMatch], list[Journey]]:
        """Keep only entities sharing the highest numerical priority value.

        For standalone guidelines, the effective priority is the guideline's own.
        For journey-associated guidelines, the effective priority is the journey's.
        """
        if not matches and not journeys:
            return [], []

        journey_priority_by_id = {j.id: j.priority for j in journeys}

        match_priorities: list[tuple[GuidelineMatch, int]] = []
        for match in matches:
            jid = self._extract_journey_id_from_guideline(match.guideline)
            if jid and cast(JourneyId, jid) in journey_priority_by_id:
                effective = journey_priority_by_id[cast(JourneyId, jid)]
            else:
                effective = match.guideline.priority
            match_priorities.append((match, effective))

        all_priorities = [p for _, p in match_priorities] + [j.priority for j in journeys]
        if not all_priorities:
            return list(matches), list(journeys)

        max_priority = max(all_priorities)

        # Identify the entities that set the priority ceiling — the
        # "winners" are the counterparts that caused everything below to
        # be deprioritized.
        winners: tuple[ResolvedEntity, ...] = tuple(
            ResolvedEntity.guideline(m.guideline) for m, p in match_priorities if p >= max_priority
        ) + tuple(ResolvedEntity.journey(j) for j in journeys if j.priority >= max_priority)

        filtered_matches = []
        for match, priority in match_priorities:
            if priority >= max_priority:
                filtered_matches.append(match)
            else:
                self._logger.debug(
                    f"Dropped (lower priority): Guideline {match.guideline.id} "
                    f"({match.guideline.content.action}) — {priority} < {max_priority}"
                )
                resolutions.setdefault(ResolvedEntity.guideline(match.guideline), []).append(
                    Resolution(
                        kind=ResolutionKind.DEPRIORITIZED,
                        details=ResolutionDetails(
                            description=(
                                f"Filtered due to lower priority ({priority} < {max_priority})"
                            ),
                            counterparts=winners,
                        ),
                    )
                )

        filtered_journeys = []
        for j in journeys:
            if j.priority >= max_priority:
                filtered_journeys.append(j)
            else:
                resolutions.setdefault(ResolvedEntity.journey(j), []).append(
                    Resolution(
                        kind=ResolutionKind.DEPRIORITIZED,
                        details=ResolutionDetails(
                            description=(
                                f"Filtered due to lower priority ({j.priority} < {max_priority})"
                            ),
                            counterparts=winners,
                        ),
                    )
                )

        return filtered_matches, filtered_journeys

    # -- Dependency resolution ----------------------------------------------

    async def _apply_dependencies(
        self,
        matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
        cache: _RelationshipCache,
        guidelines_by_tag: dict[TagId, list[Guideline]],
        guidelines_by_id: dict[GuidelineId, Guideline],
        journeys_by_id: dict[JourneyId, Journey],
        tags_by_id: dict[TagId, Tag],
        resolutions: dict[ResolvedEntity, list[Resolution]],
    ) -> Sequence[GuidelineMatch]:
        """Filter guidelines with unmet dependencies using topological ordering.

        The algorithm has three phases:

        **Phase 1 — Resolve**: For each matched guideline, gather its DEPENDENCY
        and DEPENDENCY_ANY relationships and resolve each target into a
        ``_DependencyTarget``. Build a topological-ordering graph at the same time.

        **Phase 2 — Topological sort**: Order guidelines so that dependencies are
        always processed before their dependents (Kahn's algorithm).

        **Phase 3 — Evaluate**: Walk the topological order. For each guideline,
        check its AND dependencies (all must be met) and OR groups (at least one
        target per group must be met). Remove guidelines whose dependencies are
        not satisfied.
        """
        matched_ids = {m.guideline.id for m in matches}

        # Map tag → set of matched guideline IDs (includes non-persisted guidelines)
        tag_to_matched: dict[TagId, set[GuidelineId]] = defaultdict(set)
        for m in matches:
            for tid in m.guideline.tags:
                tag_to_matched[tid].add(m.guideline.id)
            # Also register journey node guidelines under their journey tag
            # so that journey-tag dependency targets can participate in the
            # topological sort and cascade correctly.
            if jid := self._extract_journey_id_from_guideline(m.guideline):
                tag_to_matched[Tag.for_journey_id(cast(JourneyId, jid)).id].add(m.guideline.id)

        # ── Phase 1: Resolve dependency targets ──────────────────────────

        # AND deps: every target must be individually met.
        and_deps: dict[GuidelineId, list[_DependencyTarget]] = {}
        # OR groups: targets sharing a group_id are OR'd; groups are AND'd.
        or_groups: dict[GuidelineId, dict[str, list[_DependencyTarget]]] = {}
        # Topological edges: gid → set of guideline IDs it must wait for.
        topo_edges: dict[GuidelineId, set[GuidelineId]] = {m.guideline.id: set() for m in matches}

        for match in matches:
            gid = match.guideline.id
            source_ids = self._dependency_source_ids(match.guideline)

            relationships = await self._gather_dependency_relationships(source_ids, cache)

            gid_and: list[_DependencyTarget] = []
            gid_or: dict[str, list[_DependencyTarget]] = {}

            for rel in relationships:
                target = await self._resolve_dependency_target(
                    rel,
                    gid,
                    matched_ids,
                    tag_to_matched,
                    journeys,
                    guidelines_by_tag,
                    topo_edges,
                )
                if rel.kind == RelationshipKind.DEPENDENCY_ANY and rel.group_id:
                    gid_or.setdefault(rel.group_id, []).append(target)
                else:
                    gid_and.append(target)

            and_deps[gid] = gid_and
            if gid_or:
                or_groups[gid] = gid_or

        # ── Phase 2: Topological sort (Kahn's algorithm) ─────────────────

        topo_order = self._topological_sort(topo_edges)

        # ── Phase 3: Evaluate in topological order ───────────────────────

        surviving: set[GuidelineId] = set(matched_ids)

        for gid in topo_order:
            if gid not in surviving:
                continue

            failed = False

            # Check AND dependencies — collect ALL failures
            for dep in and_deps.get(gid, []):
                if not self._is_dep_target_met(dep, surviving):
                    failed = True
                    counterparts: tuple[ResolvedEntity, ...] = ()
                    if dep.target_id is not None:
                        wrapped = self._resolve_counterpart(
                            dep.target_id, guidelines_by_id, journeys_by_id, tags_by_id
                        )
                        if wrapped is not None:
                            counterparts = (wrapped,)
                    resolutions.setdefault(
                        ResolvedEntity.guideline(guidelines_by_id[gid]), []
                    ).append(
                        Resolution(
                            kind=ResolutionKind.UNMET_DEPENDENCY_ALL,
                            details=ResolutionDetails(
                                description=(f"AND dependency target {dep.target_id} not met"),
                                relationship=dep.relationship,
                                counterparts=counterparts,
                            ),
                        )
                    )

            # Check OR groups — collect ALL failing groups
            for group_id, targets in or_groups.get(gid, {}).items():
                if not any(self._is_dep_target_met(dep, surviving) for dep in targets):
                    failed = True
                    group_counterparts: tuple[ResolvedEntity, ...] = tuple(
                        wrapped
                        for dep in targets
                        if dep.target_id is not None
                        and (
                            wrapped := self._resolve_counterpart(
                                dep.target_id, guidelines_by_id, journeys_by_id, tags_by_id
                            )
                        )
                        is not None
                    )
                    # Use the relationship from the first target in the group
                    group_relationship = next(
                        (dep.relationship for dep in targets if dep.relationship),
                        None,
                    )
                    resolutions.setdefault(
                        ResolvedEntity.guideline(guidelines_by_id[gid]), []
                    ).append(
                        Resolution(
                            kind=ResolutionKind.UNMET_DEPENDENCY_ANY,
                            details=ResolutionDetails(
                                description=(
                                    f"OR dependency group '{group_id}' not met — "
                                    f"none of {[c.entity.id for c in group_counterparts]} active"
                                ),
                                relationship=group_relationship,
                                counterparts=group_counterparts,
                            ),
                        )
                    )

            if failed:
                surviving.discard(gid)
                self._logger.debug(f"Dropped (unmet dependency): Guideline {gid}")

        return [m for m in matches if m.guideline.id in surviving]

    # -- Dependency helpers -------------------------------------------------

    def _dependency_source_ids(self, guideline: Guideline) -> list[GuidelineId | TagId]:
        """Return all entity IDs from which dependency relationships should be
        queried for a given guideline (itself, its journey tag, its custom tags)."""
        source_ids: list[GuidelineId | TagId] = [guideline.id]

        if journey_id := self._extract_journey_id_from_guideline(guideline):
            source_ids.append(Tag.for_journey_id(journey_id).id)

        for tag_id in guideline.tags:
            source_ids.append(tag_id)

        return source_ids

    async def _gather_dependency_relationships(
        self,
        source_ids: Sequence[GuidelineId | TagId],
        cache: _RelationshipCache,
    ) -> list[Relationship]:
        """Fetch all DEPENDENCY and DEPENDENCY_ANY relationships for a set of
        source IDs, deduplicating by relationship ID.

        Both kinds use ``indirect=False`` — only direct relationships are
        fetched. Cascading (transitive) failures are handled by the
        topological-sort evaluation loop in Phase 3: when a dependency
        target is removed from the surviving set, its dependents fail
        naturally on the next evaluation step.
        """
        result: list[Relationship] = []
        seen: set[RelationshipId] = set()

        for source_id in source_ids:
            for dep_kind in [
                RelationshipKind.DEPENDENCY,
                RelationshipKind.DEPENDENCY_ANY,
            ]:
                for rel in await self._get_relationships(
                    cache, dep_kind, False, source_id=source_id
                ):
                    if rel.id not in seen:
                        result.append(rel)
                        seen.add(rel.id)

        return result

    async def _resolve_dependency_target(
        self,
        rel: Relationship,
        gid: GuidelineId,
        matched_ids: set[GuidelineId],
        tag_to_matched: dict[TagId, set[GuidelineId]],
        journeys: Sequence[Journey],
        guidelines_by_tag: dict[TagId, list[Guideline]],
        topo_edges: dict[GuidelineId, set[GuidelineId]],
    ) -> _DependencyTarget:
        """Resolve a single dependency relationship into a ``_DependencyTarget``.

        Also registers topological edges so that dependents are processed after
        their dependencies.
        """
        # --- Target is a specific guideline ---
        if rel.target.kind == RelationshipEntityKind.GUIDELINE:
            dep_target_id = cast(GuidelineId, rel.target.id)
            if dep_target_id not in matched_ids:
                return _DependencyTarget(
                    kind=_DependencyTargetKind.UNMET,
                    relationship=rel,
                    target_id=dep_target_id,
                )
            if dep_target_id != gid:
                topo_edges[gid].add(dep_target_id)
            return _DependencyTarget(
                kind=_DependencyTargetKind.MATCHED_GUIDELINE,
                guideline_ids={dep_target_id},
                relationship=rel,
                target_id=dep_target_id,
            )

        # --- Target is a tag (journey tag, TAG_ANY, or TAG_ALL) ---
        if rel.target.kind.is_tag:
            tag_id = cast(TagId, rel.target.id)

            # Journey tag: check journey activity and link to its node
            # guidelines so the topological sort can cascade correctly
            # (e.g. when a node guideline is removed, dependents of the
            # journey tag are re-evaluated).
            if journey_id := Tag.extract_journey_id(tag_id):
                if not any(j.id == journey_id for j in journeys):
                    return _DependencyTarget(
                        kind=_DependencyTargetKind.UNMET,
                        relationship=rel,
                        target_id=tag_id,
                    )
                # Find matched node guidelines belonging to this journey.
                journey_node_ids = tag_to_matched.get(tag_id, set()) & matched_ids
                if not journey_node_ids:
                    # Journey is active but has no matched node guidelines
                    # in the current set — treat as MET (journey-level dep).
                    return _DependencyTarget(
                        kind=_DependencyTargetKind.MET,
                        relationship=rel,
                        target_id=tag_id,
                    )
                # Register topo edges so dependents are processed after
                # the journey's node guidelines.
                for mid in journey_node_ids:
                    if mid != gid:
                        topo_edges[gid].add(mid)
                return _DependencyTarget(
                    kind=_DependencyTargetKind.ANY_MATCHED_TAG_MEMBER,
                    guideline_ids=journey_node_ids,
                    relationship=rel,
                    target_id=tag_id,
                )

            # Custom tag: collect members and check match status.
            all_member_ids = {g.id for g in guidelines_by_tag.get(tag_id, [])}
            all_member_ids.update(tag_to_matched.get(tag_id, set()))
            matched_members = all_member_ids & matched_ids

            # Register topo edges to all matched members.
            for mid in matched_members:
                if mid != gid:
                    topo_edges[gid].add(mid)

            if rel.target.kind == RelationshipEntityKind.TAG_ANY:
                if not matched_members:
                    return _DependencyTarget(
                        kind=_DependencyTargetKind.UNMET,
                        relationship=rel,
                        target_id=tag_id,
                    )
                return _DependencyTarget(
                    kind=_DependencyTargetKind.ANY_MATCHED_TAG_MEMBER,
                    guideline_ids=matched_members,
                    relationship=rel,
                    target_id=tag_id,
                )
            else:
                # TAG_ALL: every member must be matched.
                if not all_member_ids or (all_member_ids - matched_ids):
                    return _DependencyTarget(
                        kind=_DependencyTargetKind.UNMET,
                        relationship=rel,
                        target_id=tag_id,
                    )
                return _DependencyTarget(
                    kind=_DependencyTargetKind.MATCHED_GUIDELINE,
                    guideline_ids=matched_members,
                    relationship=rel,
                    target_id=tag_id,
                )

        # Unknown target kind — treat as unmet.
        return _DependencyTarget(kind=_DependencyTargetKind.UNMET)

    @staticmethod
    def _topological_sort(
        edges: dict[GuidelineId, set[GuidelineId]],
    ) -> list[GuidelineId]:
        """Kahn's algorithm: return nodes in dependency-first order."""
        in_degree: dict[GuidelineId, int] = {gid: 0 for gid in edges}
        reverse: dict[GuidelineId, set[GuidelineId]] = defaultdict(set)

        for gid, targets in edges.items():
            for dep_id in targets:
                if dep_id in in_degree:
                    in_degree[gid] += 1
                    reverse[dep_id].add(gid)

        queue: deque[GuidelineId] = deque(gid for gid, deg in in_degree.items() if deg == 0)
        order: list[GuidelineId] = []

        while queue:
            gid = queue.popleft()
            order.append(gid)
            for dependent in reverse.get(gid, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return order

    @staticmethod
    def _is_dep_target_met(dep: _DependencyTarget, surviving: set[GuidelineId]) -> bool:
        """Check whether a single dependency target is satisfied."""
        if dep.kind == _DependencyTargetKind.MET:
            return True
        if dep.kind == _DependencyTargetKind.UNMET:
            return False
        if dep.kind == _DependencyTargetKind.MATCHED_GUIDELINE:
            return dep.guideline_ids <= surviving
        if dep.kind == _DependencyTargetKind.ANY_MATCHED_TAG_MEMBER:
            return bool(dep.guideline_ids & surviving)
        return False

    @classmethod
    def _check_and_deps(
        cls,
        deps: list[_DependencyTarget],
        surviving: set[GuidelineId],
    ) -> bool:
        """Return True if ALL AND-dependency targets are satisfied."""
        return all(cls._is_dep_target_met(dep, surviving) for dep in deps)

    @classmethod
    def _check_or_groups(
        cls,
        groups: dict[str, list[_DependencyTarget]],
        surviving: set[GuidelineId],
    ) -> bool:
        """Return True if every OR-group has at least one satisfied target."""
        for targets in groups.values():
            if not any(cls._is_dep_target_met(dep, surviving) for dep in targets):
                return False
        return True

    # -- Prioritization -----------------------------------------------------

    async def _apply_prioritization(
        self,
        matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
        cache: _RelationshipCache,
        guidelines_by_tag: dict[TagId, list[Guideline]],
        guidelines_by_id: dict[GuidelineId, Guideline],
        journeys_by_id: dict[JourneyId, Journey],
        tags_by_id: dict[TagId, Tag],
        resolutions: dict[ResolvedEntity, list[Resolution]],
    ) -> RelationalResolverResult:
        """Apply priority relationships and filter both matches and journeys.

        Also performs transitive filtering: guidelines whose dependencies were
        deprioritized are removed.
        """
        match_ids = {m.guideline.id for m in matches}

        # Map tag → matched guideline IDs for non-persisted guidelines
        tag_to_matched: dict[TagId, set[GuidelineId]] = defaultdict(set)
        for m in matches:
            for tid in m.guideline.tags:
                tag_to_matched[tid].add(m.guideline.id)

        iterated: set[GuidelineId] = set()
        deprioritized_guidelines: set[GuidelineId] = set()
        deprioritized_journeys: set[JourneyId] = set()

        # Pre-populate deprioritized journeys from journey-to-journey priority.
        active_journey_ids = {j.id for j in journeys}
        for journey in journeys:
            journey_tag = Tag.for_journey_id(journey.id).id
            rels = await self._get_relationships(
                cache, RelationshipKind.PRIORITY, False, target_id=journey_tag
            )
            for rel in rels:
                if rel.source.kind.is_tag:
                    if src_jid := Tag.extract_journey_id(cast(TagId, rel.source.id)):
                        if src_jid in active_journey_ids:
                            deprioritized_journeys.add(journey.id)
                            break

        # -- Per-match priority check --
        result = []
        for match in matches:
            priority_rels = await self._get_priority_relationships_for_match(match, cache)

            if not priority_rels:
                result.append(match)
                iterated.add(match.guideline.id)
                continue

            deprioritized = False
            prioritized_guideline_id: GuidelineId | None = None
            prioritized_journey_id: JourneyId | None = None
            prioritizing_relationship: Relationship | None = None

            while priority_rels:
                relationship = priority_rels.pop()
                source = relationship.source

                if source.kind == RelationshipEntityKind.GUIDELINE and source.id in match_ids:
                    deprioritized = True
                    prioritized_guideline_id = cast(GuidelineId, source.id)
                    prioritizing_relationship = relationship
                    break

                elif source.kind.is_tag:
                    tag_guidelines = guidelines_by_tag.get(cast(TagId, source.id), [])

                    # Check persisted guidelines
                    if pid := next(
                        (
                            g.id
                            for g in tag_guidelines
                            if g.id in match_ids and g.id != match.guideline.id
                        ),
                        None,
                    ):
                        deprioritized = True
                        prioritized_guideline_id = pid
                        prioritizing_relationship = relationship
                        break

                    # Check non-persisted (projected) guidelines
                    if not deprioritized:
                        if pid := next(
                            (
                                gid
                                for gid in tag_to_matched.get(cast(TagId, source.id), set())
                                if gid != match.guideline.id
                            ),
                            None,
                        ):
                            deprioritized = True
                            prioritized_guideline_id = pid
                            prioritizing_relationship = relationship
                            break

                    # Traverse into tag members for further priority checks.
                    # For each unmatched member, check both direct guidelines
                    # and their custom tags (so G1→T1 is found via member G2's
                    # tag T1 when T1 is the source of the priority relationship).
                    for g in tag_guidelines:
                        if g.id not in iterated and g.id not in match_ids:
                            priority_rels.extend(
                                await self._get_relationships(
                                    cache, RelationshipKind.PRIORITY, False, target_id=g.id
                                )
                            )
                            for g_tid in g.tags:
                                if not Tag.extract_journey_id(g_tid):
                                    priority_rels.extend(
                                        await self._get_relationships(
                                            cache,
                                            RelationshipKind.PRIORITY,
                                            False,
                                            target_id=g_tid,
                                        )
                                    )
                    iterated.update(g.id for g in tag_guidelines if g.id not in match_ids)

                    if jid := Tag.extract_journey_id(cast(TagId, source.id)):
                        if any(j.id == jid for j in journeys):
                            deprioritized = True
                            prioritized_journey_id = cast(JourneyId, jid)
                            prioritizing_relationship = relationship
                            break

            iterated.add(match.guideline.id)

            if not deprioritized:
                result.append(match)
            else:
                deprioritized_guidelines.add(match.guideline.id)
                if self._is_journey_node_guideline(match.guideline):
                    if jid := self._extract_journey_id_from_guideline(match.guideline):
                        deprioritized_journeys.add(cast(JourneyId, jid))

                # Record resolution
                if prioritized_guideline_id:
                    prioritizer_g = guidelines_by_id.get(prioritized_guideline_id)
                    counterparts = (
                        (ResolvedEntity.guideline(prioritizer_g),)
                        if prioritizer_g is not None
                        else ()
                    )
                    resolutions.setdefault(ResolvedEntity.guideline(match.guideline), []).append(
                        Resolution(
                            kind=ResolutionKind.DEPRIORITIZED,
                            details=ResolutionDetails(
                                description=(
                                    f"Deprioritized by guideline {prioritized_guideline_id}"
                                ),
                                relationship=prioritizing_relationship,
                                counterparts=counterparts,
                            ),
                        )
                    )
                elif prioritized_journey_id:
                    prioritizer_j = journeys_by_id.get(prioritized_journey_id)
                    counterparts = (
                        (ResolvedEntity.journey(prioritizer_j),)
                        if prioritizer_j is not None
                        else ()
                    )
                    resolutions.setdefault(ResolvedEntity.guideline(match.guideline), []).append(
                        Resolution(
                            kind=ResolutionKind.DEPRIORITIZED,
                            details=ResolutionDetails(
                                description=(f"Deprioritized by journey {prioritized_journey_id}"),
                                relationship=prioritizing_relationship,
                                counterparts=counterparts,
                            ),
                        )
                    )

                self._log_deprioritization(
                    match,
                    matches,
                    prioritized_guideline_id,
                    prioritized_journey_id,
                    deprioritized_journeys,
                )

        # -- Guideline → journey deprioritization --
        result_ids = {m.guideline.id for m in result}
        for journey in journeys:
            journey_tag = Tag.for_journey_id(journey.id).id
            rels = await self._get_relationships(
                cache, RelationshipKind.PRIORITY, False, target_id=journey_tag
            )
            for rel in rels:
                if (
                    rel.source.kind == RelationshipEntityKind.GUIDELINE
                    and rel.source.id in result_ids
                ):
                    deprioritized_journeys.add(journey.id)
                    break

        # -- Transitive filtering of dependencies on deprioritized entities --
        final_result = await self._filter_deprioritized_dependents(
            result,
            cache,
            guidelines_by_tag,
            guidelines_by_id,
            journeys_by_id,
            tags_by_id,
            deprioritized_guidelines,
            deprioritized_journeys,
            resolutions,
        )

        filtered_journeys = [j for j in journeys if j.id not in deprioritized_journeys]
        return RelationalResolverResult(matches=final_result, journeys=filtered_journeys)

    async def _get_priority_relationships_for_match(
        self,
        match: GuidelineMatch,
        cache: _RelationshipCache,
    ) -> list[Relationship]:
        """Gather all PRIORITY relationships directly targeting this match.

        Uses ``indirect=False`` so that priority does not propagate through
        inactive intermediaries (reinstatement principle). Tag-mediated
        chains (e.g. G1 → T1 → G2) still work because we explicitly query
        each of the guideline's tags as additional targets.
        """
        rels = await self._get_relationships(
            cache, RelationshipKind.PRIORITY, False, target_id=match.guideline.id
        )

        # Journey node guidelines also inherit journey-level priority
        if self._is_journey_node_guideline(match.guideline):
            if jid := self._extract_journey_id_from_guideline(match.guideline):
                rels.extend(
                    await self._get_relationships(
                        cache,
                        RelationshipKind.PRIORITY,
                        False,
                        target_id=Tag.for_journey_id(jid).id,
                    )
                )

        # Custom tag priority (skip journey tags — handled above for nodes only)
        for tid in match.guideline.tags:
            if Tag.extract_journey_id(tid):
                continue
            rels.extend(
                await self._get_relationships(
                    cache, RelationshipKind.PRIORITY, False, target_id=tid
                )
            )

        return rels

    def _is_dep_target_deprioritized(
        self,
        dep: Relationship,
        deprioritized_guidelines: set[GuidelineId],
        deprioritized_journeys: set[JourneyId],
        tagged_cache: dict[TagId, Sequence[Guideline]],
    ) -> bool:
        """Check if a single dependency target is deprioritized."""
        if dep.target.kind == RelationshipEntityKind.GUIDELINE:
            return dep.target.id in deprioritized_guidelines
        elif dep.target.kind.is_tag:
            if jid := Tag.extract_journey_id(cast(TagId, dep.target.id)):
                return jid in deprioritized_journeys
            else:
                tagged = tagged_cache.get(cast(TagId, dep.target.id), [])
                return bool(tagged) and all(g.id in deprioritized_guidelines for g in tagged)
        return False

    async def _filter_deprioritized_dependents(
        self,
        matches: list[GuidelineMatch],
        cache: _RelationshipCache,
        guidelines_by_tag: dict[TagId, list[Guideline]],
        guidelines_by_id: dict[GuidelineId, Guideline],
        journeys_by_id: dict[JourneyId, Journey],
        tags_by_id: dict[TagId, Tag],
        deprioritized_guidelines: set[GuidelineId],
        deprioritized_journeys: set[JourneyId],
        resolutions: dict[ResolvedEntity, list[Resolution]],
    ) -> list[GuidelineMatch]:
        """Remove guidelines that depend on deprioritized entities.

        AND dependencies (DEPENDENCY): if any target is deprioritized, the
        guideline is removed.
        OR dependencies (DEPENDENCY_ANY): grouped by group_id. A group fails
        only if ALL targets in the group are deprioritized.
        """
        result = []
        for match in matches:
            and_deps: list[Relationship] = []
            or_groups: dict[str, list[Relationship]] = {}

            for dep_kind in (RelationshipKind.DEPENDENCY, RelationshipKind.DEPENDENCY_ANY):
                rels = await self._get_relationships(
                    cache, dep_kind, False, source_id=match.guideline.id
                )
                for tid in match.guideline.tags:
                    rels.extend(
                        await self._get_relationships(cache, dep_kind, False, source_id=tid)
                    )
                for rel in rels:
                    if rel.kind == RelationshipKind.DEPENDENCY_ANY and rel.group_id:
                        or_groups.setdefault(rel.group_id, []).append(rel)
                    else:
                        and_deps.append(rel)

            # Pre-fetch tagged guidelines for tag targets
            tagged_cache: dict[TagId, Sequence[Guideline]] = {}
            all_deps = and_deps + [r for rels in or_groups.values() for r in rels]
            for dep in all_deps:
                if dep.target.kind.is_tag and not Tag.extract_journey_id(
                    cast(TagId, dep.target.id)
                ):
                    tid = cast(TagId, dep.target.id)
                    if tid not in tagged_cache:
                        tagged_cache[tid] = guidelines_by_tag.get(tid, [])

            # Check AND deps: any one deprioritized → fail
            depends_on_deprioritized = any(
                self._is_dep_target_deprioritized(
                    dep, deprioritized_guidelines, deprioritized_journeys, tagged_cache
                )
                for dep in and_deps
            )

            # Check OR groups: a group fails only if ALL targets are deprioritized
            if not depends_on_deprioritized:
                for group_rels in or_groups.values():
                    all_in_group_deprioritized = all(
                        self._is_dep_target_deprioritized(
                            dep, deprioritized_guidelines, deprioritized_journeys, tagged_cache
                        )
                        for dep in group_rels
                    )
                    if all_in_group_deprioritized:
                        depends_on_deprioritized = True
                        break

            if depends_on_deprioritized:
                self._logger.debug(
                    f"Dropped (dependency on dropped entity): Guideline {match.guideline.id} "
                    f"({match.guideline.content.action})"
                )
                # Find the specific deprioritized dependencies for the
                # resolution, pairing each target id with the relationship
                # that established the dependency.
                deprioritized_deps: list[tuple[ResolvedEntityId | TagId, Relationship]] = []
                for dep in and_deps:
                    if self._is_dep_target_deprioritized(
                        dep, deprioritized_guidelines, deprioritized_journeys, tagged_cache
                    ):
                        deprioritized_deps.append(
                            (cast(ResolvedEntityId | TagId, dep.target.id), dep)
                        )
                for group_rels in or_groups.values():
                    if all(
                        self._is_dep_target_deprioritized(
                            dep,
                            deprioritized_guidelines,
                            deprioritized_journeys,
                            tagged_cache,
                        )
                        for dep in group_rels
                    ):
                        for dep in group_rels:
                            deprioritized_deps.append(
                                (cast(ResolvedEntityId | TagId, dep.target.id), dep)
                            )

                for dep_id, dep_rel in deprioritized_deps:
                    wrapped = self._resolve_counterpart(
                        dep_id, guidelines_by_id, journeys_by_id, tags_by_id
                    )
                    counterparts = (wrapped,) if wrapped is not None else ()
                    resolutions.setdefault(ResolvedEntity.guideline(match.guideline), []).append(
                        Resolution(
                            kind=ResolutionKind.DEPRIORITIZED,
                            details=ResolutionDetails(
                                description=(f"Dependency {dep_id} was deprioritized"),
                                relationship=dep_rel,
                                counterparts=counterparts,
                            ),
                        )
                    )
            else:
                result.append(match)

        return result

    def _log_deprioritization(
        self,
        match: GuidelineMatch,
        all_matches: Sequence[GuidelineMatch],
        prioritized_guideline_id: GuidelineId | None,
        prioritized_journey_id: JourneyId | None,
        deprioritized_journeys: set[JourneyId],
    ) -> None:
        """Log the reason a match was deprioritized."""
        if prioritized_guideline_id:
            prioritized = next(
                m.guideline for m in all_matches if m.guideline.id == prioritized_guideline_id
            )
            self._logger.debug(
                f"Dropped (deprioritized by guideline): Guideline {match.guideline.id} "
                f"({match.guideline.content.action}) — by "
                f"{prioritized_guideline_id} ({prioritized.content.action})"
            )
        elif prioritized_journey_id:
            deprioritized_journeys.add(prioritized_journey_id)
            self._logger.debug(
                f"Dropped (deprioritized by journey): Guideline {match.guideline.id} "
                f"({match.guideline.content.action}) — by journey "
                f"{prioritized_journey_id}"
            )

    # -- Entailment ---------------------------------------------------------

    async def _apply_entailment(
        self,
        usable_guidelines: Sequence[Guideline],
        matches: Sequence[GuidelineMatch],
        cache: _RelationshipCache,
        guidelines_by_tag: dict[TagId, list[Guideline]],
    ) -> list[tuple[GuidelineMatch, list[tuple[Relationship, GuidelineId]]]]:
        """Activate additional guidelines implied by entailment relationships.

        Returns a list of ``(entailed_match, sources)`` tuples. Each
        ``sources`` entry is a ``(relationship, source_guideline_id)``
        pair identifying one entailing source for the activated guideline.
        The caller can create one ENTAILED resolution per source.
        """
        # Map: guideline -> (entailing match, target guideline, relationship)
        related_by_match: dict[
            GuidelineId, list[tuple[GuidelineMatch, Guideline, Relationship]]
        ] = defaultdict(list)
        match_ids = {m.guideline.id for m in matches}

        for match in matches:
            relationships = await self._get_relationships(
                cache, RelationshipKind.ENTAILMENT, True, source_id=match.guideline.id
            )

            while relationships:
                rel = relationships.pop()

                if rel.target.kind == RelationshipEntityKind.GUIDELINE:
                    if any(rel.target.id == m.guideline.id for m in matches):
                        continue
                    target_guideline = next(g for g in usable_guidelines if g.id == rel.target.id)
                    related_by_match[target_guideline.id].append((match, target_guideline, rel))

                elif rel.target.kind.is_tag:
                    tagged = guidelines_by_tag.get(cast(TagId, rel.target.id), [])
                    for g in tagged:
                        if g.id not in match_ids:
                            related_by_match[g.id].append((match, g, rel))
                        relationships.extend(
                            await self._get_relationships(
                                cache, RelationshipKind.ENTAILMENT, True, source_id=g.id
                            )
                        )

        # For each entailed guideline: use the highest-scoring match for the
        # GuidelineMatch, but collect (relationship, source_guideline_id)
        # pairs for every entailing source.
        result: list[tuple[GuidelineMatch, list[tuple[Relationship, GuidelineId]]]] = []
        seen_guidelines: set[GuidelineId] = set()
        for gid, entries in related_by_match.items():
            if gid in seen_guidelines:
                continue
            seen_guidelines.add(gid)

            best: tuple[GuidelineMatch, Guideline, Relationship] | None = None
            for entry in entries:
                if best is None or entry[0].score > best[0].score:
                    best = entry
            if best is not None:
                sources = [(entry[2], entry[0].guideline.id) for entry in entries]
                result.append(
                    (
                        GuidelineMatch(
                            guideline=best[1],
                            score=best[0].score,
                            rationale="[Activated via entailment] Automatically inferred from context",
                        ),
                        sources,
                    )
                )

        return result

    # -- Counterpart resolution --------------------------------------------

    @staticmethod
    def _resolve_counterpart(
        raw_id: GuidelineId | JourneyId | TagId,
        guidelines_by_id: dict[GuidelineId, Guideline],
        journeys_by_id: dict[JourneyId, Journey],
        tags_by_id: dict[TagId, Tag],
    ) -> ResolvedEntity | None:
        """Wrap a raw counterpart id as a ``ResolvedEntity``.

        Journey-tag ids resolve to the underlying ``Journey`` (the
        semantically meaningful entity) when the journey is known.
        Returns ``None`` if the id cannot be resolved against any of the
        provided lookup maps — callers may skip in that case.
        """
        raw = cast(str, raw_id)
        if raw in guidelines_by_id:
            return ResolvedEntity.guideline(guidelines_by_id[cast(GuidelineId, raw)])
        if raw in journeys_by_id:
            return ResolvedEntity.journey(journeys_by_id[cast(JourneyId, raw)])
        # A journey tag should resolve to its journey, not the tag wrapper.
        if jid := Tag.extract_journey_id(cast(TagId, raw)):
            if jid in journeys_by_id:
                return ResolvedEntity.journey(journeys_by_id[cast(JourneyId, jid)])
        if raw in tags_by_id:
            return ResolvedEntity.tag(tags_by_id[cast(TagId, raw)])
        return None

    # -- Shared helpers -----------------------------------------------------

    async def _get_relationships(
        self,
        cache: _RelationshipCache,
        kind: RelationshipKind,
        indirect: bool,
        source_id: GuidelineId | TagId | ToolId | None = None,
        target_id: GuidelineId | TagId | ToolId | None = None,
    ) -> list[Relationship]:
        """Fetch relationships with per-query caching."""
        entity_id = source_id if source_id else target_id
        assert entity_id is not None, "Either source_id or target_id must be provided"

        direction = "source" if source_id else "target"
        key: _CacheKey = (kind, indirect, direction, entity_id)

        if key not in cache:
            if source_id:
                cache[key] = list(
                    await self._relationship_store.list_relationships(
                        kind=kind, indirect=indirect, source_id=source_id
                    )
                )
            else:
                cache[key] = list(
                    await self._relationship_store.list_relationships(
                        kind=kind, indirect=indirect, target_id=target_id
                    )
                )

        return list(cache[key])

    def _extract_journey_id_from_guideline(self, guideline: Guideline) -> Optional[str]:
        """Extract the journey ID associated with a guideline, if any."""
        if "journey_node" in guideline.metadata:
            return cast(
                JourneyId,
                cast(dict[str, JSONSerializable], guideline.metadata["journey_node"])["journey_id"],
            )
        if any(Tag.extract_journey_id(tid) for tid in guideline.tags):
            return next(
                (
                    Tag.extract_journey_id(tid)
                    for tid in guideline.tags
                    if Tag.extract_journey_id(tid)
                ),
                None,
            )
        return None

    def _is_journey_node_guideline(self, guideline: Guideline) -> bool:
        """Check if a guideline is a journey node (projected from a journey graph).

        Journey node guidelines carry ``journey_node`` metadata and represent
        the journey's behavior. This is distinct from journey CONDITION guidelines,
        which are plain observations tagged with the journey tag. Condition
        guidelines should not be subject to journey-level deprioritization.
        """
        return "journey_node" in guideline.metadata

    @staticmethod
    def _matches_equal(a: Sequence[GuidelineMatch], b: Sequence[GuidelineMatch]) -> bool:
        if len(a) != len(b):
            return False
        return all(x.guideline.id == y.guideline.id and x.score == y.score for x, y in zip(a, b))

    @staticmethod
    def _journeys_equal(a: Sequence[Journey], b: Sequence[Journey]) -> bool:
        if len(a) != len(b):
            return False
        return {j.id for j in a} == {j.id for j in b}

    def _emit_tracer_events(
        self,
        initial_ids: set[GuidelineId],
        final_matches: list[GuidelineMatch],
        original_matches: Sequence[GuidelineMatch],
        resolutions: dict[ResolvedEntity, list[Resolution]],
        guidelines_by_id: dict[GuidelineId, Guideline],
    ) -> None:
        """Emit tracer events for activated (entailed) and deactivated guidelines."""
        final_ids = {m.guideline.id for m in final_matches}
        all_matches = {m.guideline.id: m for m in list(original_matches) + final_matches}

        for match in final_matches:
            if match.guideline.id not in initial_ids:
                self._tracer.add_event(
                    "gm.activate",
                    attributes={
                        "guideline_id": match.guideline.id,
                        "condition": match.guideline.content.condition,
                        "action": match.guideline.content.action or "",
                        "rationale": "Activated via entailment",
                    },
                )

        for gid in initial_ids - final_ids:
            m = all_matches[gid]
            entity = ResolvedEntity.guideline(guidelines_by_id[gid])
            res_list = resolutions.get(entity, [])
            rationale = (
                "; ".join(r.details.description for r in res_list) if res_list else "Unknown reason"
            )
            self._tracer.add_event(
                "gm.deactivate",
                attributes={
                    "guideline_id": gid,
                    "condition": m.guideline.content.condition,
                    "action": m.guideline.content.action or "",
                    "rationale": rationale,
                },
            )
