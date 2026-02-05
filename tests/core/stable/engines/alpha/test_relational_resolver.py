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
from parlant.core.engines.alpha.relational_resolver import RelationalResolver
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


async def test_that_relational_resolver_prioritizes_indirectly_between_guidelines(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

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

    assert result.matches == [GuidelineMatch(guideline=g1, score=8, rationale="")]


async def test_that_relational_resolver_prioritizes_between_journey_nodes(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]

    resolver = container[RelationalResolver]

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

    assert result.matches == [GuidelineMatch(guideline=j1_guidelines[0], score=8, rationale="")]


async def test_that_relational_resolver_prioritizes_guideline_over_journey(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    projection = container[JourneyGuidelineProjection]
    resolver = container[RelationalResolver]

    # Create a standalone guideline
    standalone_guideline = await guideline_store.create_guideline(
        condition="Customer asks about drinks",
        action="Recommend Pepsi",
    )

    # Create a journey with a condition
    journey_condition = await guideline_store.create_guideline(
        condition="Customer asks about drinks"
    )

    journey = await journey_store.create_journey(
        title="Drink Recommendation Journey",
        description="Recommend Coca-Cola to the customer",
        conditions=[journey_condition.id],
    )

    # Add nodes to the journey to create a graph
    journey_node_1 = await journey_store.create_node(
        journey_id=journey.id,
        action="Ask what drink they want",
        tools=[],
    )

    journey_node_2 = await journey_store.create_node(
        journey_id=journey.id,
        action="Recommend Coca-Cola",
        tools=[],
    )

    # Add an edge between the nodes
    await journey_store.create_edge(
        journey_id=journey.id,
        source=journey_node_1.id,
        target=journey_node_2.id,
        condition=None,
    )

    # Project journey to get journey-guidelines
    journey_guidelines = await projection.project_journey_to_guidelines(journey.id)
    assert len(journey_guidelines) > 0

    # Create priority relationship: standalone guideline > journey
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=standalone_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(journey.id),
            kind=RelationshipEntityKind.TAG,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    # Both the standalone guideline and journey-guidelines match
    journey_matches = [
        GuidelineMatch(guideline=g, score=5 + i, rationale="")
        for i, g in enumerate(journey_guidelines)
    ]
    result = await resolver.resolve(
        [standalone_guideline] + list(journey_guidelines),
        [GuidelineMatch(guideline=standalone_guideline, score=8, rationale="")] + journey_matches,
        journeys=[journey],
    )

    # Only the standalone guideline should remain (all journey-guidelines are filtered out)
    assert result.matches == [GuidelineMatch(guideline=standalone_guideline, score=8, rationale="")]


async def test_that_relational_resolver_prioritizes_journey_over_guideline(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    projection = container[JourneyGuidelineProjection]
    resolver = container[RelationalResolver]

    # Create a journey with a condition
    journey_condition = await guideline_store.create_guideline(
        condition="Customer asks about drinks"
    )

    journey = await journey_store.create_journey(
        title="Drink Recommendation Journey",
        description="Recommend Pepsi to the customer",
        conditions=[journey_condition.id],
    )

    # Add nodes to the journey to create a graph
    journey_node_1 = await journey_store.create_node(
        journey_id=journey.id,
        action="Ask what drink they want",
        tools=[],
    )

    journey_node_2 = await journey_store.create_node(
        journey_id=journey.id,
        action="Recommend Pepsi",
        tools=[],
    )

    # Add an edge between the nodes
    await journey_store.create_edge(
        journey_id=journey.id,
        source=journey_node_1.id,
        target=journey_node_2.id,
        condition=None,
    )

    # Project journey to get journey-guidelines
    journey_guidelines = await projection.project_journey_to_guidelines(journey.id)
    assert len(journey_guidelines) > 0

    # Create a standalone guideline
    standalone_guideline = await guideline_store.create_guideline(
        condition="Customer asks about drinks",
        action="Recommend Coca-Cola",
    )

    # Create priority relationship: journey > standalone guideline
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(journey.id),
            kind=RelationshipEntityKind.TAG,
        ),
        target=RelationshipEntity(
            id=standalone_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    # Both journey-guidelines and standalone guideline match
    journey_matches = [
        GuidelineMatch(guideline=g, score=8 - i, rationale="")
        for i, g in enumerate(journey_guidelines)
    ]
    result = await resolver.resolve(
        list(journey_guidelines) + [standalone_guideline],
        journey_matches + [GuidelineMatch(guideline=standalone_guideline, score=10, rationale="")],
        journeys=[journey],
    )

    # The standalone guideline should be filtered out because journey prioritizes over it
    # Only the journey-guidelines remain
    assert result.matches == journey_matches


async def test_that_relational_resolver_filters_journey_dependent_guideline_when_journey_is_deprioritized(
    container: Container,
) -> None:
    """
    Tests the transitive effect of priority + dependency:
    - Guideline Y prioritizes over Journey J
    - Guideline X depends on Journey J
    - When Y, X, and J are all active, Y's priority over J should filter out X
      (because X depends on J, and J is deprioritized)
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    # Create a journey
    journey_condition = await guideline_store.create_guideline(
        condition="Customer asks about drinks"
    )

    journey = await journey_store.create_journey(
        title="Drink Recommendation Journey",
        description="Recommend Coca-Cola to the customer",
        conditions=[journey_condition.id],
    )

    # Create guideline X that depends on the journey
    guideline_x = await guideline_store.create_guideline(
        condition="Customer asks about drinks",
        action="Recommend Sprite",
    )

    # Create dependency: X depends on Journey
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=guideline_x.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(journey.id),
            kind=RelationshipEntityKind.TAG,
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    # Create guideline Y that prioritizes over the journey
    guideline_y = await guideline_store.create_guideline(
        condition="Customer asks about drinks",
        action="Recommend Pepsi",
    )

    # Create priority: Y prioritizes over Journey
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=guideline_y.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(journey.id),
            kind=RelationshipEntityKind.TAG,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    # Both Y and X are active
    result = await resolver.resolve(
        [guideline_y, guideline_x],
        [
            GuidelineMatch(guideline=guideline_y, score=8, rationale=""),
            GuidelineMatch(guideline=guideline_x, score=6, rationale=""),
        ],
        journeys=[journey],
    )

    # Only Y should remain:
    # - Y prioritizes over J, so J is effectively deprioritized
    # - X depends on J, so when J is deprioritized, X is also filtered out
    assert result.matches == [GuidelineMatch(guideline=guideline_y, score=8, rationale="")]


async def test_that_relational_resolver_does_not_ignore_a_deprioritized_guideline_when_its_prioritized_counterpart_is_not_active(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

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

    assert result.matches == [
        GuidelineMatch(guideline=deprioritized_guideline, score=5, rationale="")
    ]


async def test_that_relational_resolver_does_not_ignore_deprioritized_journey_node_when_prioritized_journey_is_not_active(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    projection = container[JourneyGuidelineProjection]
    resolver = container[RelationalResolver]

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

    assert result.matches == [
        GuidelineMatch(guideline=deprioritized_guideline, score=5, rationale="")
    ]


async def test_that_relational_resolver_prioritizes_guidelines(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

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

    assert result.matches == [
        GuidelineMatch(guideline=prioritized_guideline, score=8, rationale="")
    ]


async def test_that_relational_resolver_infers_guidelines_from_tags(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

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

    assert len(result.matches) == 4
    assert any(m.guideline.id == g1.id for m in result.matches)
    assert any(m.guideline.id == g2.id for m in result.matches)
    assert any(m.guideline.id == g3.id for m in result.matches)
    assert any(m.guideline.id == g4.id for m in result.matches)


async def test_that_relational_resolver_does_not_ignore_a_deprioritized_tag_when_its_prioritized_counterpart_is_not_active(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

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

    assert len(result.matches) == 1
    assert result.matches[0].guideline.id == deprioritized_guideline.id


async def test_that_relational_resolver_prioritizes_guidelines_from_tags(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

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

    assert len(result.matches) == 1
    assert result.matches[0].guideline.id == g1.id


async def test_that_relational_resolver_handles_indirect_guidelines_from_tags(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

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

    assert len(result.matches) == 1
    assert result.matches[0].guideline.id == g1.id


async def test_that_relational_resolver_filters_out_guidelines_with_unmet_dependencies(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

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

    assert result.matches == []


async def test_that_relational_resolver_filters_out_guidelines_with_unmet_dependencies_connected_through_tag(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

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

    assert len(result.matches) == 1
    assert result.matches[0].guideline.id == tagged_guideline_1.id


async def test_that_relational_resolver_filters_out_journey_nodes_with_unmet_journey_dependency_with_guideline(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    projection = container[JourneyGuidelineProjection]
    resolver = container[RelationalResolver]

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

    assert result.matches == []


async def test_that_relational_resolver_filters_out_journey_nodes_with_unmet_journey_dependencies(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    projection = container[JourneyGuidelineProjection]
    resolver = container[RelationalResolver]

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

    assert result.matches == []


async def test_that_relational_resolver_filters_dependent_guidelines_by_journey_tags_when_journeys_are_not_relatively_enabled(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

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

    assert len(result.matches) == 1
    assert result.matches[0].guideline.id == enabled_journey_tagged_guideline.id


async def test_that_relational_resolver_iterates_until_stable_with_cascading_priorities(
    container: Container,
) -> None:
    """
    Tests iterative resolution with cascading priorities:
    - Guideline A prioritizes over B
    - Guideline B prioritizes over C
    - Guideline C depends on D
    - All four start as matches
    - First iteration: A deprioritizes B
    - Second iteration: C loses dependency on B (B is gone)
    - Expected: A, D remain (B deprioritized, C filtered due to lost dependency on B)
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    # Create guidelines
    guideline_a = await guideline_store.create_guideline(
        condition="Customer asks about priority",
        action="Recommend option A",
    )
    guideline_b = await guideline_store.create_guideline(
        condition="Customer asks about priority",
        action="Recommend option B",
    )
    guideline_c = await guideline_store.create_guideline(
        condition="Customer asks about priority",
        action="Recommend option C",
    )
    guideline_d = await guideline_store.create_guideline(
        condition="Customer asks about priority",
        action="Recommend option D",
    )

    # A prioritizes over B
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=guideline_a.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=guideline_b.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    # B prioritizes over C
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=guideline_b.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=guideline_c.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    # C depends on B
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=guideline_c.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=guideline_b.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    # All four are matches
    result = await resolver.resolve(
        [guideline_a, guideline_b, guideline_c, guideline_d],
        [
            GuidelineMatch(guideline=guideline_a, score=8, rationale=""),
            GuidelineMatch(guideline=guideline_b, score=7, rationale=""),
            GuidelineMatch(guideline=guideline_c, score=6, rationale=""),
            GuidelineMatch(guideline=guideline_d, score=5, rationale=""),
        ],
        journeys=[],
    )

    # Only A and D should remain:
    # - First iteration: B is deprioritized by A
    # - Second iteration: C loses dependency on B (B is gone), so C is filtered
    # - D has no relationships, remains
    assert len(result.matches) == 2
    assert any(m.guideline.id == guideline_a.id for m in result.matches)
    assert any(m.guideline.id == guideline_d.id for m in result.matches)


async def test_that_relational_resolver_handles_priority_affecting_dependency_in_second_iteration(
    container: Container,
) -> None:
    """
    Tests that priority relationships discovered via entailment affect dependencies:
    - Guideline X depends on Y
    - Guideline A entails Z
    - Z prioritizes over Y
    - Initial matches: [A, X, Y]
    - First iteration: A entails Z (now matches: [A, X, Y, Z])
    - Second iteration: Z prioritizes over Y, X loses dependency
    - Expected: Only A and Z remain
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    # Create guidelines
    guideline_a = await guideline_store.create_guideline(
        condition="Customer needs help",
        action="Offer help",
    )
    guideline_x = await guideline_store.create_guideline(
        condition="Customer needs help",
        action="Provide option X",
    )
    guideline_y = await guideline_store.create_guideline(
        condition="Customer needs help",
        action="Provide option Y",
    )
    guideline_z = await guideline_store.create_guideline(
        condition="Customer needs help",
        action="Provide option Z (override)",
    )

    # X depends on Y
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=guideline_x.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=guideline_y.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    # A entails Z
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=guideline_a.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=guideline_z.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )

    # Z prioritizes over Y
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=guideline_z.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=guideline_y.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    # Initial matches: A, X, Y
    result = await resolver.resolve(
        [guideline_a, guideline_x, guideline_y, guideline_z],
        [
            GuidelineMatch(guideline=guideline_a, score=8, rationale=""),
            GuidelineMatch(guideline=guideline_x, score=7, rationale=""),
            GuidelineMatch(guideline=guideline_y, score=6, rationale=""),
        ],
        journeys=[],
    )

    # Only A and Z should remain:
    # - First iteration: A entails Z (Z added to matches)
    # - Second iteration: Z prioritizes over Y (Y deprioritized), X loses dependency
    assert len(result.matches) == 2
    assert any(m.guideline.id == guideline_a.id for m in result.matches)
    assert any(m.guideline.id == guideline_z.id for m in result.matches)
