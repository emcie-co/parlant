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

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence, cast

from parlant.core.common import JSONSerializable
from parlant.core.journeys import Journey, JourneyId
from parlant.core.loggers import Logger
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.relationships import (
    Relationship,
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.guidelines import Guideline, GuidelineId, GuidelineStore
from parlant.core.tags import TagId, Tag
from parlant.core.tracer import Tracer


@dataclass
class RelationalResolverResult:
    matches: Sequence[GuidelineMatch]
    journeys: Sequence[Journey]


class RelationalResolver:
    MAX_ITERATIONS = 3

    def __init__(
        self,
        relationship_store: RelationshipStore,
        guideline_store: GuidelineStore,
        logger: Logger,
        tracer: Tracer,
    ) -> None:
        self._relationship_store = relationship_store
        self._guideline_store = guideline_store
        self._logger = logger
        self._tracer = tracer

    def _extract_journey_id_from_guideline(self, guideline: Guideline) -> Optional[str]:
        if "journey_node" in guideline.metadata:
            return cast(
                JourneyId,
                cast(dict[str, JSONSerializable], guideline.metadata["journey_node"])["journey_id"],
            )

        if any(Tag.extract_journey_id(tag_id) for tag_id in guideline.tags):
            return next(
                (
                    Tag.extract_journey_id(tag_id)
                    for tag_id in guideline.tags
                    if Tag.extract_journey_id(tag_id)
                ),
                None,
            )

        return None

    def _matches_equal(
        self, matches1: Sequence[GuidelineMatch], matches2: Sequence[GuidelineMatch]
    ) -> bool:
        """Check if two match sequences are equal (same guidelines, same order)."""
        if len(matches1) != len(matches2):
            return False
        return all(
            m1.guideline.id == m2.guideline.id and m1.score == m2.score
            for m1, m2 in zip(matches1, matches2)
        )

    def _journeys_equal(self, journeys1: Sequence[Journey], journeys2: Sequence[Journey]) -> bool:
        """Check if two journey sequences are equal (same IDs)."""
        if len(journeys1) != len(journeys2):
            return False
        ids1 = {j.id for j in journeys1}
        ids2 = {j.id for j in journeys2}
        return ids1 == ids2

    async def resolve(
        self,
        usable_guidelines: Sequence[Guideline],
        matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
    ) -> RelationalResolverResult:
        # Use the guideline matcher scope to associate logs with it
        with self._logger.scope("GuidelineMatcher"):
            with self._logger.scope("RelationalResolver"):
                # Cache for relationship queries to avoid redundant calls
                relationship_cache: dict[
                    tuple[RelationshipKind, bool, str, str], list[Relationship]
                ] = {}

                # Track deactivation reasons for guidelines
                deactivation_reasons: dict[GuidelineId, str] = {}

                initial_match_ids = {m.guideline.id for m in matches}
                current_matches = list(matches)
                current_journeys = list(journeys)

                for iteration in range(self.MAX_ITERATIONS):
                    self._logger.debug(f"RelationalResolver iteration {iteration + 1}")

                    # Step 1: Apply prioritization (filter based on priority relationships and filter journeys)
                    prioritization_result = await self._apply_prioritization(
                        current_matches, current_journeys, relationship_cache, deactivation_reasons
                    )

                    # Step 2: Apply entailment (add new matches based on entailment relationships)
                    entailed_matches = await self._apply_entailment(
                        usable_guidelines, prioritization_result.matches, relationship_cache
                    )

                    # Combine prioritized and entailed matches
                    combined_matches = list(prioritization_result.matches) + entailed_matches

                    # Step 3: Apply dependencies (filter out matches with unmet dependencies)
                    # This must run AFTER prioritization because we need to filter based on deprioritized entities
                    filtered_matches = await self._apply_dependencies(
                        usable_guidelines,
                        combined_matches,
                        prioritization_result.journeys,
                        relationship_cache,
                        deactivation_reasons,
                    )

                    new_matches = filtered_matches
                    new_journeys = list(prioritization_result.journeys)

                    # Check if we've reached a stable state
                    if self._matches_equal(new_matches, current_matches) and self._journeys_equal(
                        new_journeys, current_journeys
                    ):
                        self._logger.debug(
                            f"RelationalResolver converged after {iteration + 1} iteration(s)"
                        )
                        break

                    current_matches = new_matches
                    current_journeys = new_journeys
                else:
                    self._logger.debug(
                        f"RelationalResolver reached max iterations ({self.MAX_ITERATIONS})"
                    )

                # Emit tracer events for final results
                final_match_ids = {m.guideline.id for m in current_matches}
                matches_by_id = {m.guideline.id: m for m in matches + current_matches}

                # Emit events for activated guidelines (entailed)
                for match in current_matches:
                    if match.guideline.id not in initial_match_ids:
                        self._tracer.add_event(
                            "gm.activate",
                            attributes={
                                "guideline_id": match.guideline.id,
                                "condition": match.guideline.content.condition,
                                "action": match.guideline.content.action or "",
                                "rationale": "Activated via entailment",
                            },
                        )

                # Emit events for deactivated guidelines
                for guideline_id in initial_match_ids - final_match_ids:
                    match = matches_by_id[guideline_id]
                    rationale = deactivation_reasons.get(guideline_id, "Unknown reason")
                    self._tracer.add_event(
                        "gm.deactivate",
                        attributes={
                            "guideline_id": guideline_id,
                            "condition": match.guideline.content.condition,
                            "action": match.guideline.content.action or "",
                            "rationale": rationale,
                        },
                    )

                return RelationalResolverResult(
                    matches=current_matches,
                    journeys=current_journeys,
                )

    async def _get_relationships(
        self,
        cache: dict[tuple[RelationshipKind, bool, str, str], list[Relationship]],
        kind: RelationshipKind,
        indirect: bool,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
    ) -> list[Relationship]:
        """Get relationships with caching."""
        entity_id = source_id if source_id else target_id
        assert entity_id is not None, "Either source_id or target_id must be provided"

        # Cache key must distinguish between source and target queries
        direction = "source" if source_id else "target"
        cache_key = (kind, indirect, direction, entity_id)
        if cache_key not in cache:
            if source_id:
                cache[cache_key] = list(
                    await self._relationship_store.list_relationships(
                        kind=kind,
                        indirect=indirect,
                        source_id=source_id,
                    )
                )
            else:
                cache[cache_key] = list(
                    await self._relationship_store.list_relationships(
                        kind=kind,
                        indirect=indirect,
                        target_id=target_id,
                    )
                )

        return cache[cache_key]

    async def _apply_dependencies(
        self,
        usable_guidelines: Sequence[Guideline],
        matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
        cache: dict[tuple[RelationshipKind, bool, str, str], list[Relationship]],
        deactivation_reasons: dict[GuidelineId, str],
    ) -> Sequence[GuidelineMatch]:
        """Filter out guidelines with unmet dependencies."""
        # This is the logic from filter_unmet_dependencies in the old implementation
        matched_guideline_ids = {m.guideline.id for m in matches}

        result: list[GuidelineMatch] = []

        for match in matches:
            dependencies = await self._get_relationships(
                cache, RelationshipKind.DEPENDENCY, True, source_id=match.guideline.id
            )

            if journey_id := self._extract_journey_id_from_guideline(match.guideline):
                dependencies.extend(
                    await self._get_relationships(
                        cache,
                        RelationshipKind.DEPENDENCY,
                        True,
                        source_id=Tag.for_journey_id(journey_id),
                    )
                )

            if not dependencies:
                result.append(match)
                continue

            iterated_guidelines: set[GuidelineId] = set()

            dependent_on_inactive_guidelines = False

            while dependencies:
                dependency = dependencies.pop()

                if (
                    dependency.target.kind == RelationshipEntityKind.GUIDELINE
                    and dependency.target.id not in matched_guideline_ids
                ):
                    dependent_on_inactive_guidelines = True
                    break

                if dependency.target.kind == RelationshipEntityKind.TAG:
                    if journey_id := Tag.extract_journey_id(cast(TagId, dependency.target.id)):
                        if any(journey.id == journey_id for journey in journeys):
                            # If the tag is a journey tag and the journey is active,
                            # then this dependency is met.
                            continue
                        else:
                            dependent_on_inactive_guidelines = True
                            break

                    guidelines_associated_to_tag = await self._guideline_store.list_guidelines(
                        tags=[cast(TagId, dependency.target.id)]
                    )

                    for g in guidelines_associated_to_tag:
                        if g.id not in matched_guideline_ids:
                            dependent_on_inactive_guidelines = True
                            break

                        if g.id not in iterated_guidelines:
                            dependencies.extend(
                                await self._get_relationships(
                                    cache, RelationshipKind.DEPENDENCY, True, source_id=g.id
                                )
                            )

                    iterated_guidelines.update(g.id for g in guidelines_associated_to_tag)

            if not dependent_on_inactive_guidelines:
                result.append(match)
            else:
                self._logger.debug(
                    f"Skipped: Guideline {match.guideline.id} deactivated due to unmet dependencies"
                )
                deactivation_reasons[match.guideline.id] = "Unmet dependencies"

        return result

    async def _apply_prioritization(
        self,
        matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
        cache: dict[tuple[RelationshipKind, bool, str, str], list[Relationship]],
        deactivation_reasons: dict[GuidelineId, str],
    ) -> RelationalResolverResult:
        """Apply priority relationships and filter both matches and journeys."""
        # This is the logic from replace_with_prioritized in the old implementation
        match_guideline_ids = {m.guideline.id for m in matches}

        iterated_guidelines: set[GuidelineId] = set()

        # Track deprioritized entities for transitive filtering
        deprioritized_guideline_ids: set[GuidelineId] = set()
        deprioritized_journey_ids: set[JourneyId] = set()

        result = []

        for match in matches:
            priority_relationships = await self._get_relationships(
                cache, RelationshipKind.PRIORITY, True, target_id=match.guideline.id
            )

            if journey_id := self._extract_journey_id_from_guideline(match.guideline):
                priority_relationships.extend(
                    await self._get_relationships(
                        cache,
                        RelationshipKind.PRIORITY,
                        True,
                        target_id=Tag.for_journey_id(journey_id),
                    )
                )

            if not priority_relationships:
                result.append(match)
                continue

            deprioritized = False
            prioritized_guideline_id: GuidelineId | None = None

            while priority_relationships:
                relationship = priority_relationships.pop()

                prioritized_entity = relationship.source

                if (
                    prioritized_entity.kind == RelationshipEntityKind.GUIDELINE
                    and prioritized_entity.id in match_guideline_ids
                ):
                    deprioritized = True
                    prioritized_guideline_id = cast(GuidelineId, prioritized_entity.id)
                    break

                elif prioritized_entity.kind == RelationshipEntityKind.TAG:
                    guideline_associated_with_prioritized_tag = (
                        await self._guideline_store.list_guidelines(
                            tags=[cast(TagId, prioritized_entity.id)]
                        )
                    )

                    if prioritized_guideline_id := next(
                        (
                            g.id
                            for g in guideline_associated_with_prioritized_tag
                            if g.id in match_guideline_ids and g.id != match.guideline.id
                        ),
                        None,
                    ):
                        deprioritized = True
                        break

                    for g in guideline_associated_with_prioritized_tag:
                        if g.id in iterated_guidelines or g.id in match_guideline_ids:
                            continue

                        priority_relationships.extend(
                            await self._get_relationships(
                                cache, RelationshipKind.PRIORITY, True, target_id=g.id
                            )
                        )

                    iterated_guidelines.update(
                        g.id
                        for g in guideline_associated_with_prioritized_tag
                        if g.id not in match_guideline_ids
                    )

                    if journey_id := Tag.extract_journey_id(cast(TagId, prioritized_entity.id)):
                        if any(journey.id == journey_id for journey in journeys):
                            deprioritized = True
                            prioritized_journey_id = journey_id
                            break

            iterated_guidelines.add(match.guideline.id)

            if not deprioritized:
                result.append(match)
            else:
                # Track deprioritized entities for transitive filtering
                deprioritized_guideline_ids.add(match.guideline.id)
                if journey_id := self._extract_journey_id_from_guideline(match.guideline):
                    deprioritized_journey_ids.add(cast(JourneyId, journey_id))

                if prioritized_guideline_id:
                    prioritized_guideline = next(
                        m.guideline for m in matches if m.guideline.id == prioritized_guideline_id
                    )

                    self._logger.debug(
                        f"Skipped: Guideline {match.guideline.id} ({match.guideline.content.action}) deactivated due to contextual prioritization by {prioritized_guideline_id} ({prioritized_guideline.content.action})"
                    )
                    deactivation_reasons[match.guideline.id] = (
                        f"Deprioritized by guideline {prioritized_guideline_id}"
                    )
                elif prioritized_journey_id:
                    deprioritized_journey_ids.add(cast(JourneyId, prioritized_journey_id))
                    self._logger.debug(
                        f"Skipped: Guideline {match.guideline.id} ({match.guideline.content.action}) deactivated due to contextual prioritization by journey {prioritized_journey_id}"
                    )
                    deactivation_reasons[match.guideline.id] = (
                        f"Deprioritized by journey {prioritized_journey_id}"
                    )

        # Check if any matched guidelines prioritize over active journeys
        result_guideline_ids = {m.guideline.id for m in result}
        for journey in journeys:
            journey_tag = Tag.for_journey_id(journey.id)
            priority_relationships = await self._get_relationships(
                cache, RelationshipKind.PRIORITY, True, target_id=journey_tag
            )

            for relationship in priority_relationships:
                if (
                    relationship.source.kind == RelationshipEntityKind.GUIDELINE
                    and relationship.source.id in result_guideline_ids
                ):
                    # A matched guideline prioritizes over this journey
                    deprioritized_journey_ids.add(journey.id)
                    break

        # Transitive filtering: Remove guidelines that depend on deprioritized entities
        final_result = []
        for match in result:
            dependencies = await self._get_relationships(
                cache, RelationshipKind.DEPENDENCY, True, source_id=match.guideline.id
            )

            depends_on_deprioritized = False

            for dependency in dependencies:
                # Check if depends on a deprioritized guideline
                if (
                    dependency.target.kind == RelationshipEntityKind.GUIDELINE
                    and dependency.target.id in deprioritized_guideline_ids
                ):
                    depends_on_deprioritized = True
                    break

                # Check if depends on a deprioritized journey
                if dependency.target.kind == RelationshipEntityKind.TAG:
                    if journey_id := Tag.extract_journey_id(cast(TagId, dependency.target.id)):
                        if journey_id in deprioritized_journey_ids:
                            depends_on_deprioritized = True
                            break

            if not depends_on_deprioritized:
                final_result.append(match)
            else:
                self._logger.debug(
                    f"Skipped: Guideline {match.guideline.id} ({match.guideline.content.action}) deactivated due to dependency on deprioritized entity"
                )
                deactivation_reasons[match.guideline.id] = "Depends on deprioritized entity"

        # Filter journeys to remove deprioritized ones
        filtered_journeys = [j for j in journeys if j.id not in deprioritized_journey_ids]

        return RelationalResolverResult(matches=final_result, journeys=filtered_journeys)

    async def _apply_entailment(
        self,
        usable_guidelines: Sequence[Guideline],
        matches: Sequence[GuidelineMatch],
        cache: dict[tuple[RelationshipKind, bool, str, str], list[Relationship]],
    ) -> Sequence[GuidelineMatch]:
        """Add guidelines based on entailment relationships."""
        # This is the logic from get_entailed in the old implementation
        related_guidelines_by_match = defaultdict[GuidelineMatch, set[Guideline]](set)

        match_guideline_ids = {m.guideline.id for m in matches}

        for match in matches:
            relationships = await self._get_relationships(
                cache, RelationshipKind.ENTAILMENT, True, source_id=match.guideline.id
            )

            while relationships:
                relationship = relationships.pop()

                if relationship.target.kind == RelationshipEntityKind.GUIDELINE:
                    if any(relationship.target.id == m.guideline.id for m in matches):
                        # no need to add this related guideline as it's already an assumed match
                        continue
                    related_guidelines_by_match[match].add(
                        next(g for g in usable_guidelines if g.id == relationship.target.id)
                    )

                elif relationship.target.kind == RelationshipEntityKind.TAG:
                    # In case target is a tag, we need to find all guidelines
                    # that are associated with this tag.
                    guidelines_associated_to_tag = await self._guideline_store.list_guidelines(
                        tags=[cast(TagId, relationship.target.id)]
                    )

                    related_guidelines_by_match[match].update(
                        g for g in guidelines_associated_to_tag if g.id not in match_guideline_ids
                    )

                    # Add all the relationships for the related guidelines to the stack
                    for g in guidelines_associated_to_tag:
                        relationships.extend(
                            await self._get_relationships(
                                cache, RelationshipKind.ENTAILMENT, True, source_id=g.id
                            )
                        )

        match_and_inferred_guideline_pairs: list[tuple[GuidelineMatch, Guideline]] = []

        for match, related_guidelines in related_guidelines_by_match.items():
            for related_guideline in related_guidelines:
                if existing_related_guidelines := [
                    (match, inferred_guideline)
                    for match, inferred_guideline in match_and_inferred_guideline_pairs
                    if inferred_guideline == related_guideline
                ]:
                    assert len(existing_related_guidelines) == 1
                    existing_related_guideline = existing_related_guidelines[0]

                    if existing_related_guideline[0].score >= match.score:
                        continue  # Stay with existing one
                    else:
                        # This match's score is higher, so it's better that
                        # we associate the related guideline with this one.
                        match_and_inferred_guideline_pairs.remove(
                            existing_related_guideline,
                        )

                match_and_inferred_guideline_pairs.append(
                    (match, related_guideline),
                )

        entailed_matches = [
            GuidelineMatch(
                guideline=inferred_guideline,
                score=match.score,
                rationale="Automatically inferred from context",
            )
            for match, inferred_guideline in match_and_inferred_guideline_pairs
        ]

        return entailed_matches
