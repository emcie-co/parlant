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
from typing import Sequence, cast

from parlant.core.engines.alpha.guideline_match import GuidelineMatch
from parlant.core.relationships import (
    EntityType,
    GuidelineRelationshipKind,
    RelationshipStore,
)
from parlant.core.guidelines import Guideline, GuidelineId, GuidelineStore
from parlant.core.tags import TagId


class RelationalGuidelineResolver:
    def __init__(
        self,
        relationship_store: RelationshipStore,
        guideline_store: GuidelineStore,
    ) -> None:
        self._relationship_store = relationship_store
        self._guideline_store = guideline_store

    async def resolve(
        self,
        usable_guidelines: Sequence[Guideline],
        matches: Sequence[GuidelineMatch],
    ) -> Sequence[GuidelineMatch]:
        result = await self.filter_unmet_dependencies(
            usable_guidelines=usable_guidelines,
            matches=matches,
        )
        result = await self.replace_with_prioritized(result)

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
    ) -> Sequence[GuidelineMatch]:
        # Some guidelines have priority relationships that dictate activation.
        #
        # For example, if we matched guidelines "When X, Then Y" (S) and "When A, Then B" (T),
        # and S is prioritized, only "When X, Then Y" should be activated.
        # Such priority relationships are stored in RelationshipStore,
        # and those are the ones we are loading here.
        match_guideline_ids = {m.guideline.id for m in matches}

        itarated_guidelines: set[GuidelineId] = set()

        result = []
        for match in matches:
            relationships = list(
                await self._relationship_store.list_relationships(
                    kind=GuidelineRelationshipKind.PRIORITY,
                    indirect=True,
                    target=match.guideline.id,
                )
            )

            if not relationships:
                result.append(match)
                continue

            prioritized = True
            while relationships:
                relationship = relationships.pop()
                if (
                    relationship.target_type == EntityType.GUIDELINE
                    and relationship.target in match_guideline_ids
                ):
                    prioritized = False
                    break

                elif relationship.target_type == EntityType.TAG:
                    # In case target is a tag, we need to find all guidelines
                    # that are associated with this tag.
                    #
                    # We then need to check if any of those guidelines have a priority relationship
                    #
                    # If not, we need to iterate over all those guidelines and add their priority relationships
                    guideline_associated_to_tag = await self._guideline_store.list_guidelines(
                        tags=[cast(TagId, relationship.target)]
                    )

                    if any(
                        g.id in match_guideline_ids and g.id != match.guideline.id
                        for g in guideline_associated_to_tag
                    ):
                        prioritized = False
                        break

                    for g in guideline_associated_to_tag:
                        # In case we already iterated over this guideline,
                        # we don't need to iterate over it again.
                        if g.id in itarated_guidelines or g.id in match_guideline_ids:
                            continue

                        relationships.extend(
                            await self._relationship_store.list_relationships(
                                kind=GuidelineRelationshipKind.PRIORITY,
                                indirect=True,
                                target=g.id,
                            )
                        )

                    itarated_guidelines.update(
                        g.id for g in guideline_associated_to_tag if g.id not in match_guideline_ids
                    )

            itarated_guidelines.add(match.guideline.id)

            if prioritized:
                result.append(match)

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
                    kind=GuidelineRelationshipKind.ENTAILMENT,
                    indirect=True,
                    source=match.guideline.id,
                )
            )

            while relationships:
                relationship = relationships.pop()

                if relationship.target_type == EntityType.GUIDELINE:
                    if any(relationship.target == m.guideline.id for m in matches):
                        # no need to add this related guideline as it's already an assumed match
                        continue
                    related_guidelines_by_match[match].add(
                        next(g for g in usable_guidelines if g.id == relationship.target)
                    )

                elif relationship.target_type == EntityType.TAG:
                    # In case target is a tag, we need to find all guidelines
                    # that are associated with this tag.
                    guidelines_associated_to_tag = await self._guideline_store.list_guidelines(
                        tags=[cast(TagId, relationship.target)]
                    )

                    related_guidelines_by_match[match].update(
                        g for g in guidelines_associated_to_tag if g.id not in match_guideline_ids
                    )

                    # Add all the relationships for the related guidelines to the stack
                    for g in guidelines_associated_to_tag:
                        relationships.extend(
                            await self._relationship_store.list_relationships(
                                kind=GuidelineRelationshipKind.ENTAILMENT,
                                indirect=True,
                                source=g.id,
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

        return [
            GuidelineMatch(
                guideline=inferred_guideline,
                score=match.score,
                rationale="Automatically inferred from context",
            )
            for match, inferred_guideline in match_and_inferred_guideline_pairs
        ]

    async def filter_unmet_dependencies(
        self,
        usable_guidelines: Sequence[Guideline],
        matches: Sequence[GuidelineMatch],
    ) -> Sequence[GuidelineMatch]:
        # Some guidelines have dependencies that dictate activation.
        #
        # For example, if we matched guidelines "When X, Then Y" (S) and "When Y, Then Z" (T),
        # and S is dependes on T, then S should not be activated unless T is activated.
        match_guideline_ids = {m.guideline.id for m in matches}

        result: list[GuidelineMatch] = []

        for match in matches:
            dependencies = list(
                await self._relationship_store.list_relationships(
                    kind=GuidelineRelationshipKind.DEPENDENCY,
                    indirect=True,
                    source=match.guideline.id,
                )
            )

            if not dependencies:
                result.append(match)
                continue

            itarated_guidelines: set[GuidelineId] = set()

            dependent = False
            while dependencies:
                dependency = dependencies.pop()

                if (
                    dependency.target_type == EntityType.GUIDELINE
                    and dependency.target not in match_guideline_ids
                ):
                    dependent = True
                    break

                if dependency.target_type == EntityType.TAG:
                    guidelines_associated_to_tag = await self._guideline_store.list_guidelines(
                        tags=[cast(TagId, dependency.target)]
                    )

                    for g in guidelines_associated_to_tag:
                        if g.id in itarated_guidelines or g.id in match_guideline_ids:
                            continue

                        dependencies.extend(
                            await self._relationship_store.list_relationships(
                                kind=GuidelineRelationshipKind.DEPENDENCY,
                                indirect=True,
                                source=g.id,
                            )
                        )

                    itarated_guidelines.update(g.id for g in guidelines_associated_to_tag)

            if not dependent:
                result.append(match)

        return result
