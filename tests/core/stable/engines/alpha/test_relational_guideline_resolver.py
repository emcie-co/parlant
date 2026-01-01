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

from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.relational_guideline_resolver import RelationalGuidelineResolver
from parlant.core.journey_guideline_projection import JourneyGuidelineProjection
from parlant.core.journeys import JourneyStore
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipKind,
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
        kind=RelationshipKind.PRIORITY,
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
        kind=RelationshipKind.PRIORITY,
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


async def test_that_relational_guideline_resolver_prioritizes_between_journey_nodes(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]

    resolver = container[RelationalGuidelineResolver]

    j1_condition = await guideline_store.create_guideline(
        condition="Customer is interested in Journey 1"
    )
    j2_condition = await guideline_store.create_guideline(
        condition="Customer is interested in Journey 2"
    )

    j1 = await journey_store.create_journey(
        title="Journey 1",
        description="Description for Journey 1",
        conditions=[j1_condition.id],
    )

    j2 = await journey_store.create_journey(
        title="Journey 2",
        description="Description for Journey 2",
        conditions=[j2_condition.id],
    )

    j1_guidelines = await container[JourneyGuidelineProjection].project_journey_to_guidelines(j1.id)
    j2_guidelines = await container[JourneyGuidelineProjection].project_journey_to_guidelines(j2.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(j1.id),
            kind=RelationshipEntityKind.TAG,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j2.id),
            kind=RelationshipEntityKind.TAG,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    assert len(j1_guidelines) == 1
    assert len(j2_guidelines) == 1

    result = await resolver.resolve(
        [j1_guidelines[0], j2_guidelines[0]],
        [
            GuidelineMatch(guideline=j1_guidelines[0], score=8, rationale=""),
            GuidelineMatch(guideline=j2_guidelines[0], score=5, rationale=""),
        ],
        journeys=[j1, j2],
    )

    assert result == [GuidelineMatch(guideline=j1_guidelines[0], score=8, rationale="")]


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
        kind=RelationshipKind.PRIORITY,
    )

    matches: list[GuidelineMatch] = [
        GuidelineMatch(guideline=deprioritized_guideline, score=5, rationale=""),
    ]

    result = await resolver.resolve([prioritized_guideline, deprioritized_guideline], matches, [])

    assert result == [GuidelineMatch(guideline=deprioritized_guideline, score=5, rationale="")]


async def test_that_relational_guideline_resolver_does_not_ignore_deprioritized_journey_node_when_prioritized_journey_is_not_active(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    projection = container[JourneyGuidelineProjection]
    resolver = container[RelationalGuidelineResolver]

    prioritized_condition = await guideline_store.create_guideline(
        condition="Customer is interested in Journey A"
    )
    deprioritized_condition = await guideline_store.create_guideline(
        condition="Customer is interested in Journey B"
    )

    prioritized_journey = await journey_store.create_journey(
        title="Journey A",
        description="High priority journey",
        conditions=[prioritized_condition.id],
    )
    deprioritized_journey = await journey_store.create_journey(
        title="Journey B",
        description="Lower priority journey",
        conditions=[deprioritized_condition.id],
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(prioritized_journey.id),
            kind=RelationshipEntityKind.TAG,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(deprioritized_journey.id),
            kind=RelationshipEntityKind.TAG,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    prioritized_guidelines = await projection.project_journey_to_guidelines(prioritized_journey.id)
    deprioritized_guidelines = await projection.project_journey_to_guidelines(
        deprioritized_journey.id
    )

    assert len(prioritized_guidelines) == 1
    assert len(deprioritized_guidelines) == 1

    deprioritized_guideline = deprioritized_guidelines[0]
    prioritized_guideline = prioritized_guidelines[0]

    result = await resolver.resolve(
        [prioritized_guideline, deprioritized_guideline],
        [
            GuidelineMatch(guideline=deprioritized_guideline, score=5, rationale=""),
        ],
        journeys=[],
    )

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
        kind=RelationshipKind.PRIORITY,
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
        kind=RelationshipKind.ENTAILMENT,
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
        kind=RelationshipKind.ENTAILMENT,
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
        kind=RelationshipKind.PRIORITY,
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
        kind=RelationshipKind.PRIORITY,
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
        kind=RelationshipKind.PRIORITY,
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
        kind=RelationshipKind.PRIORITY,
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
        kind=RelationshipKind.PRIORITY,
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
        kind=RelationshipKind.PRIORITY,
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
        kind=RelationshipKind.DEPENDENCY,
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
        kind=RelationshipKind.DEPENDENCY,
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


async def test_that_relational_guideline_resolver_filters_out_journey_nodes_with_unmet_journey_dependency_with_guideline(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    projection = container[JourneyGuidelineProjection]
    resolver = container[RelationalGuidelineResolver]

    source_condition = await guideline_store.create_guideline(
        condition="Customer has not specified if it's a repeat transaction or a new one",
        action="Ask them which it is",
    )

    source_journey = await journey_store.create_journey(
        title="Clarify Transaction Type",
        description="Journey for asking if it's repeat or new transaction",
        conditions=[source_condition.id],
    )

    guideline = await guideline_store.create_guideline(
        condition="Customer wants to make a transaction",
        action="Help them",
    )

    source_journey_guidelines = await projection.project_journey_to_guidelines(source_journey.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(source_journey.id),
            kind=RelationshipEntityKind.TAG,
        ),
        target=RelationshipEntity(
            id=guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    assert len(source_journey_guidelines) == 1

    result = await resolver.resolve(
        [source_journey_guidelines[0], guideline],
        [
            GuidelineMatch(guideline=source_journey_guidelines[0], score=8, rationale=""),
        ],
        journeys=[],
    )

    assert result == []


async def test_that_relational_guideline_resolver_filters_out_journey_nodes_with_unmet_journey_dependencies(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    projection = container[JourneyGuidelineProjection]
    resolver = container[RelationalGuidelineResolver]

    source_condition = await guideline_store.create_guideline(
        condition="Customer has not specified if it's a repeat transaction or a new one",
        action="Ask them which it is",
    )

    source_journey = await journey_store.create_journey(
        title="Clarify Transaction Type",
        description="Journey for asking if it's repeat or new transaction",
        conditions=[source_condition.id],
    )

    target_journey = await journey_store.create_journey(
        title="Validate Account",
        description="Journey for validating account",
        conditions=[],
    )

    source_journey_guidelines = await projection.project_journey_to_guidelines(source_journey.id)
    target_journey_guidelines = await projection.project_journey_to_guidelines(target_journey.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(source_journey.id),
            kind=RelationshipEntityKind.TAG,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(target_journey.id),
            kind=RelationshipEntityKind.TAG,
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    assert len(source_journey_guidelines) == 1
    assert len(target_journey_guidelines) == 1

    result = await resolver.resolve(
        [source_journey_guidelines[0], target_journey_guidelines[0]],
        [
            GuidelineMatch(guideline=source_journey_guidelines[0], score=8, rationale=""),
        ],
        journeys=[source_journey],
    )

    assert result == []


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
        kind=RelationshipKind.DEPENDENCY,
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
        kind=RelationshipKind.DEPENDENCY,
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
