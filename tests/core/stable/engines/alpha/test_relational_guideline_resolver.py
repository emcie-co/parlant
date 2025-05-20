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

from lagom import Container

from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.relational_guideline_resolver import RelationalGuidelineResolver
from parlant.core.journeys import JourneyStore
from parlant.core.relationships import (
    RelationshipEntityKind,
    GuidelineRelationshipKind,
    RelationshipEntity,
    RelationshipStore,
)
from parlant.core.guidelines import GuidelineStore
from parlant.core.tags import TagStore, Tag


async def test_that_relational_guideline_resolver_prioritizes_indirectly_between_guidelines(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalGuidelineResolver]

    g1 = await guideline_store.create_guideline(condition="x", action="y")
    g2 = await guideline_store.create_guideline(condition="y", action="z")
    g3 = await guideline_store.create_guideline(condition="z", action="t")

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g1.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=g2.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g2.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=g3.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=8, rationale=""),
            GuidelineMatch(guideline=g2, score=5, rationale=""),
            GuidelineMatch(guideline=g3, score=9, rationale=""),
        ],
        journeys=[],
    )

    assert result == [GuidelineMatch(guideline=g1, score=8, rationale="")]


async def test_that_relational_guideline_resolver_does_not_ignore_a_deprioritized_guideline_when_its_prioritized_counterpart_is_not_active(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalGuidelineResolver]

    prioritized_guideline = await guideline_store.create_guideline(condition="x", action="y")
    deprioritized_guideline = await guideline_store.create_guideline(condition="y", action="z")

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=prioritized_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=deprioritized_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    matches: list[GuidelineMatch] = [
        GuidelineMatch(guideline=deprioritized_guideline, score=5, rationale=""),
    ]

    result = await resolver.resolve([prioritized_guideline, deprioritized_guideline], matches, [])

    assert result == [GuidelineMatch(guideline=deprioritized_guideline, score=5, rationale="")]


async def test_that_relational_guideline_resolver_prioritizes_guidelines(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalGuidelineResolver]

    prioritized_guideline = await guideline_store.create_guideline(condition="x", action="y")
    deprioritized_guideline = await guideline_store.create_guideline(condition="y", action="z")

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=prioritized_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=deprioritized_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    matches: list[GuidelineMatch] = [
        GuidelineMatch(guideline=prioritized_guideline, score=8, rationale=""),
        GuidelineMatch(guideline=deprioritized_guideline, score=5, rationale=""),
    ]

    result = await resolver.resolve([prioritized_guideline, deprioritized_guideline], matches, [])

    assert result == [GuidelineMatch(guideline=prioritized_guideline, score=8, rationale="")]


async def test_that_relational_guideline_resolver_infers_guidelines_from_tags(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalGuidelineResolver]

    g1 = await guideline_store.create_guideline(condition="x", action="y")
    g2 = await guideline_store.create_guideline(condition="y", action="z")
    g3 = await guideline_store.create_guideline(condition="z", action="t")
    g4 = await guideline_store.create_guideline(condition="t", action="u")

    t1 = await tag_store.create_tag(name="t1")

    await guideline_store.upsert_tag(guideline_id=g2.id, tag_id=t1.id)
    await guideline_store.upsert_tag(guideline_id=g3.id, tag_id=t1.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g1.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.TAG,
        ),
        kind=GuidelineRelationshipKind.ENTAILMENT,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.TAG,
        ),
        target=RelationshipEntity(
            id=g4.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.ENTAILMENT,
    )

    result = await resolver.resolve(
        [g1, g2, g3, g4],
        [
            GuidelineMatch(guideline=g1, score=8, rationale=""),
        ],
        journeys=[],
    )

    assert len(result) == 4
    assert any(m.guideline.id == g1.id for m in result)
    assert any(m.guideline.id == g2.id for m in result)
    assert any(m.guideline.id == g3.id for m in result)
    assert any(m.guideline.id == g4.id for m in result)


async def test_that_relational_guideline_resolver_does_not_ignore_a_deprioritized_tag_when_its_prioritized_counterpart_is_not_active(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalGuidelineResolver]

    prioritized_guideline = await guideline_store.create_guideline(condition="x", action="y")
    deprioritized_guideline = await guideline_store.create_guideline(condition="y", action="z")

    deprioritized_tag = await tag_store.create_tag(name="t1")

    await guideline_store.upsert_tag(deprioritized_guideline.id, deprioritized_tag.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=prioritized_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=deprioritized_tag.id,
            kind=RelationshipEntityKind.TAG,
        ),
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=deprioritized_tag.id,
            kind=RelationshipEntityKind.TAG,
        ),
        target=RelationshipEntity(
            id=deprioritized_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [prioritized_guideline, deprioritized_guideline],
        [
            GuidelineMatch(guideline=deprioritized_guideline, score=5, rationale=""),
        ],
        journeys=[],
    )

    assert len(result) == 1
    assert result[0].guideline.id == deprioritized_guideline.id


async def test_that_relational_guideline_resolver_prioritizes_guidelines_from_tags(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalGuidelineResolver]

    g1 = await guideline_store.create_guideline(condition="x", action="y")
    g2 = await guideline_store.create_guideline(condition="y", action="z")

    t1 = await tag_store.create_tag(name="t1")

    await guideline_store.upsert_tag(g2.id, t1.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g1.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.TAG,
        ),
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.TAG,
        ),
        target=RelationshipEntity(
            id=g2.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g1, g2],
        [
            GuidelineMatch(guideline=g1, score=8, rationale=""),
            GuidelineMatch(guideline=g2, score=5, rationale=""),
        ],
        journeys=[],
    )

    assert len(result) == 1
    assert result[0].guideline.id == g1.id


async def test_that_relational_guideline_resolver_handles_indirect_guidelines_from_tags(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalGuidelineResolver]

    g1 = await guideline_store.create_guideline(condition="x", action="y")
    g2 = await guideline_store.create_guideline(condition="y", action="z")
    g3 = await guideline_store.create_guideline(condition="z", action="t")

    t1 = await tag_store.create_tag(name="t1")

    await guideline_store.upsert_tag(g2.id, t1.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g1.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.TAG,
        ),
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.TAG,
        ),
        target=RelationshipEntity(
            id=g3.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=8, rationale=""),
            GuidelineMatch(guideline=g3, score=9, rationale=""),
        ],
        journeys=[],
    )

    assert len(result) == 1
    assert result[0].guideline.id == g1.id


async def test_that_relational_guideline_resolver_filters_out_guidelines_with_unmet_dependencies(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalGuidelineResolver]

    source_guideline = await guideline_store.create_guideline(
        condition="Customer has not specified if it's a repeat transaction or a new one",
        action="Ask them which it is",
    )
    target_guideline = await guideline_store.create_guideline(
        condition="Customer wants to make a transaction", action="Help them"
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=source_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=target_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [source_guideline, target_guideline],
        [
            GuidelineMatch(guideline=source_guideline, score=8, rationale=""),
        ],
        journeys=[],
    )

    assert result == []


async def test_that_relational_guideline_resolver_filters_out_guidelines_with_unmet_dependencies_connected_through_tag(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalGuidelineResolver]

    source_guideline = await guideline_store.create_guideline(condition="a", action="b")

    tagged_guideline_1 = await guideline_store.create_guideline(condition="c", action="d")
    tagged_guideline_2 = await guideline_store.create_guideline(condition="e", action="f")

    target_tag = await tag_store.create_tag(name="t1")

    await guideline_store.upsert_tag(tagged_guideline_1.id, target_tag.id)
    await guideline_store.upsert_tag(tagged_guideline_2.id, target_tag.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=source_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=target_tag.id,
            kind=RelationshipEntityKind.TAG,
        ),
        kind=GuidelineRelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [source_guideline, tagged_guideline_1, tagged_guideline_2],
        [
            GuidelineMatch(guideline=source_guideline, score=8, rationale=""),
            GuidelineMatch(guideline=tagged_guideline_1, score=10, rationale=""),
            # Missing match for tagged_guideline_2
        ],
        journeys=[],
    )

    assert len(result) == 1
    assert result[0].guideline.id == tagged_guideline_1.id


async def test_that_relational_guideline_resolver_filters_dependent_guidelines_by_journey_tags_when_journeys_are_not_relatively_enabled(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalGuidelineResolver]

    enabled_journey = await journey_store.create_journey(
        title="First Journey",
        description="Description",
        conditions=[],
    )
    disabled_journey = await journey_store.create_journey(
        title="Second Journey",
        description="Description",
        conditions=[],
    )

    enabled_journey_tagged_guideline = await guideline_store.create_guideline(
        condition="a", action="b"
    )
    disabled_journey_tagged_guideline = await guideline_store.create_guideline(
        condition="c", action="d"
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=enabled_journey_tagged_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(enabled_journey.id),
            kind=RelationshipEntityKind.TAG,
        ),
        kind=GuidelineRelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=disabled_journey_tagged_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(disabled_journey.id),
            kind=RelationshipEntityKind.TAG,
        ),
        kind=GuidelineRelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [enabled_journey_tagged_guideline, disabled_journey_tagged_guideline],
        [
            GuidelineMatch(guideline=enabled_journey_tagged_guideline, score=8, rationale=""),
            GuidelineMatch(guideline=disabled_journey_tagged_guideline, score=10, rationale=""),
        ],
        journeys=[enabled_journey],
    )

    assert len(result) == 1
    assert result[0].guideline.id == enabled_journey_tagged_guideline.id
