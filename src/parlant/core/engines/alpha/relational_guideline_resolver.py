# Copyright 2025 Emcie Co Ltd.
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
from itertools import chain
from typing import Optional, Sequence, cast

from parlant.core.common import JSONSerializable
from parlant.core.journeys import Journey, JourneyId
from parlant.core.loggers import Logger
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.guidelines import Guideline, GuidelineId, GuidelineStore
from parlant.core.tags import TagId, Tag


class RelationalGuidelineResolver:
    def __init__(
        self,
        relationship_store: RelationshipStore,
        guideline_store: GuidelineStore,
        logger: Logger,
    ) -> None:
        self._relationship_store = relationship_store
        self._guideline_store = guideline_store
        self._logger = logger

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

    async def resolve(
        self,
        usable_guidelines: Sequence[Guideline],
        matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
    ) -> Sequence[GuidelineMatch]:
        # Use the guideline matcher scope to associate logs with it
        with self._logger.scope("GuidelineMatcher"):
            with self._logger.scope("RelationalGuidelineResolver"):
                result = await self.filter_unmet_dependencies(
                    usable_guidelines=usable_guidelines,
                    matches=matches,
                    journeys=journeys,
                )
                result = await self.replace_with_prioritized(
                    result,
                    journeys=journeys,
                )

                return list(
                    chain(
                        result,
                        await self.get_entailed(
                            usable_guidelines=usable_guidelines,
                            matches=result,
                        ),
                    )
                )

    async def replace_with_prioritized(
        self,
        matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
    ) -> Sequence[GuidelineMatch]:
        # Some guidelines have priority relationships that dictate activation.
        #
        # For example, if we matched guidelines "When X, Then Y" (S) and "When A, Then B" (T),
        # and S is prioritized, only "When X, Then Y" should be activated.
        # Such priority relationships are stored in RelationshipStore,
        # and those are the ones we are loading here.
        match_guideline_ids = {m.guideline.id for m in matches}

        iterated_guidelines: set[GuidelineId] = set()

        result = []

        for match in matches:
            priority_relationships = list(
                await self._relationship_store.list_relationships(
                    kind=RelationshipKind.PRIORITY,
                    indirect=True,
                    target_id=match.guideline.id,
                )
            )

            if journey_id := self._extract_journey_id_from_guideline(match.guideline):
                priority_relationships.extend(
                    await self._relationship_store.list_relationships(
                        kind=RelationshipKind.PRIORITY,
                        indirect=True,
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
                    # In case source is a tag, we need to find all guidelines
                    # that are associated with this tag.
                    # If the tag is a journey tag and the journey is active,
                    # than match deprioritized.
                    #
                    # We then need to check if any of those guidelines have a priority relationship
                    #
                    # If not, we need to iterate over all those guidelines and add their priority relationships
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
                        # In case we already iterated over this guideline,
                        # we don't need to iterate over it again.
                        if g.id in iterated_guidelines or g.id in match_guideline_ids:
                            continue

                        priority_relationships.extend(
                            await self._relationship_store.list_relationships(
                                kind=RelationshipKind.PRIORITY,
                                indirect=True,
                                target_id=g.id,
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
                if prioritized_guideline_id:
                    prioritized_guideline = next(
                        m.guideline for m in matches if m.guideline.id == prioritized_guideline_id
                    )

                    self._logger.info(
                        f"Skipped: Guideline {match.guideline.id} ({match.guideline.content.action}) deactivated due to contextual prioritization by {prioritized_guideline_id} ({prioritized_guideline.content.action})"
                    )
                elif prioritized_journey_id:
                    self._logger.info(
                        f"Skipped: Guideline {match.guideline.id} ({match.guideline.content.action}) deactivated due to contextual prioritization by journey {prioritized_journey_id}"
                    )

        return result

    async def get_entailed(
        self,
        usable_guidelines: Sequence[Guideline],
        matches: Sequence[GuidelineMatch],
    ) -> Sequence[GuidelineMatch]:
        # Some guidelines cannot be inferred simply by evaluating an interaction.
        #
        # For example, if we matched a guideline, "When X, Then Y",
        # we also need to load and account for "When Y, Then Z".
        # Such relationships are pre-indexed in a graph behind the scenes,
        # and those are the ones we are loading here.

        related_guidelines_by_match = defaultdict[GuidelineMatch, set[Guideline]](set)

        match_guideline_ids = {m.guideline.id for m in matches}

        for match in matches:
            relationships = list(
                await self._relationship_store.list_relationships(
                    kind=RelationshipKind.ENTAILMENT,
                    indirect=True,
                    source_id=match.guideline.id,
                )
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
                            await self._relationship_store.list_relationships(
                                kind=RelationshipKind.ENTAILMENT,
                                indirect=True,
                                source_id=g.id,
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

                    # We're basically saying, if this related guideline is already
                    # related to a match with a higher priority than the match
                    # at hand, then we want to keep the associated with the match
                    # that has the higher priority, because it will go down as the inferred
                    # priority of our related guideline's match...
                    #
                    # Now try to read that out loud in one go :)
                    if existing_related_guideline[0].score >= match.score:
                        continue  # Stay with existing one
                    else:
                        # This match's score is higher, so it's better that
                        # we associate the related guideline with this one.
                        # we'll add it soon, but meanwhile let's remove the old one.
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

        for m in entailed_matches:
            self._logger.info(f"Activated: Entailed guideline {m.guideline.id}")

        return entailed_matches

    async def filter_unmet_dependencies(
        self,
        usable_guidelines: Sequence[Guideline],
        matches: Sequence[GuidelineMatch],
        journeys: Sequence[Journey],
    ) -> Sequence[GuidelineMatch]:
        # Some guidelines have dependencies that dictate activation.
        #
        # For example, if we matched guidelines "When X, Then Y" (S) and "When Y, Then Z" (T),
        # and S is depends on T, then S should not be activated unless T is activated.
        matched_guideline_ids = {m.guideline.id for m in matches}

        result: list[GuidelineMatch] = []

        for match in matches:
            dependencies = list(
                await self._relationship_store.list_relationships(
                    kind=RelationshipKind.DEPENDENCY,
                    indirect=True,
                    source_id=match.guideline.id,
                )
            )

            if journey_id := self._extract_journey_id_from_guideline(match.guideline):
                dependencies.extend(
                    await self._relationship_store.list_relationships(
                        kind=RelationshipKind.DEPENDENCY,
                        indirect=True,
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
                                await self._relationship_store.list_relationships(
                                    kind=RelationshipKind.DEPENDENCY,
                                    indirect=True,
                                    source_id=g.id,
                                )
                            )

                    iterated_guidelines.update(g.id for g in guidelines_associated_to_tag)

            if not dependent_on_inactive_guidelines:
                result.append(match)
            else:
                self._logger.info(
                    f"Skipped: Guideline {match.guideline.id} deactivated due to unmet dependencies"
                )

        return result
