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
from parlant.core.engines.alpha.relational_resolver import (
    RelationalResolver,
    RelationalResolverResult,
    Resolution,
    ResolutionKind,
    ResolvedEntity,
    ResolvedEntityId,
)
from parlant.core.journey_guideline_projection import JourneyGuidelineProjection
from parlant.core.journeys import JourneyStore
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipEntity,
    RelationshipStore,
)
from parlant.core.guidelines import Guideline, GuidelineId, GuidelineStore
from parlant.core.journeys import Journey, JourneyId
from parlant.core.tags import TagStore, Tag


def _find_resolutions(
    result: RelationalResolverResult,
    target: Guideline | Journey | ResolvedEntity | GuidelineId | JourneyId,
) -> list[Resolution]:
    """Look up resolutions for a guideline/journey, accepting full objects,
    ResolvedEntity wrappers, or bare ids (legacy)."""
    if isinstance(target, ResolvedEntity):
        return result.resolutions.get(target, [])
    if isinstance(target, Guideline):
        return result.resolutions.get(ResolvedEntity.guideline(target), [])
    if isinstance(target, Journey):
        return result.resolutions.get(ResolvedEntity.journey(target), [])
    # Bare id: scan the dict for a matching entity id.
    for entity, res in result.resolutions.items():
        if entity.entity.id == target:
            return res
    return []


def assert_resolutions(
    result: RelationalResolverResult,
    entity: Guideline | Journey | ResolvedEntity | GuidelineId | JourneyId,
    expected_kinds: list[ResolutionKind],
) -> None:
    """Assert that an entity has exactly the given resolution kinds."""
    resolutions = _find_resolutions(result, entity)
    actual_kinds = [r.kind for r in resolutions]
    label = (
        entity.entity.id if isinstance(entity, ResolvedEntity) else getattr(entity, "id", entity)
    )
    assert sorted(actual_kinds, key=lambda k: k.name) == sorted(
        expected_kinds, key=lambda k: k.name
    ), (
        f"Entity {label}: expected resolution kinds "
        f"{[k.name for k in expected_kinds]}, got {[k.name for k in actual_kinds]}"
    )


def get_resolutions_by_kind(
    result: RelationalResolverResult,
    entity: Guideline | Journey | ResolvedEntity | GuidelineId | JourneyId,
    kind: ResolutionKind,
) -> list[Resolution]:
    """Get all resolutions of a specific kind for an entity."""
    return [r for r in _find_resolutions(result, entity) if r.kind == kind]


async def test_that_relational_resolver_prioritizes_indirectly_between_guidelines(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="x", action="y")
    g2 = await guideline_store.create_guideline(condition="y", action="z")
    g3 = await guideline_store.create_guideline(condition="z", action="t")

    rel_g1_g2 = await relationship_store.create_relationship(
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

    rel_g2_g3 = await relationship_store.create_relationship(
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

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g3.id, [ResolutionKind.DEPRIORITIZED])

    # Each DEPRIORITIZED resolution should identify the prioritizing
    # guideline as a counterpart and reference the PRIORITY relationship
    # that caused the deprioritization.
    g2_res = get_resolutions_by_kind(result, g2.id, ResolutionKind.DEPRIORITIZED)
    assert len(g2_res) == 1
    assert ResolvedEntity.guideline(g1) in g2_res[0].details.counterparts
    assert g2_res[0].details.relationship is not None
    assert g2_res[0].details.relationship.id == rel_g1_g2.id

    g3_res = get_resolutions_by_kind(result, g3.id, ResolutionKind.DEPRIORITIZED)
    assert len(g3_res) == 1
    assert ResolvedEntity.guideline(g2) in g3_res[0].details.counterparts
    assert g3_res[0].details.relationship is not None
    assert g3_res[0].details.relationship.id == rel_g2_g3.id


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
        triggers=[j1_condition.id],
    )

    j2 = await journey_store.create_journey(
        title="Journey 2",
        description="Description for Journey 2",
        triggers=[j2_condition.id],
    )

    j1_guidelines = await container[JourneyGuidelineProjection].project_journey_to_guidelines(j1.id)
    j2_guidelines = await container[JourneyGuidelineProjection].project_journey_to_guidelines(j2.id)

    journey_priority_rel = await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j2.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
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

    assert_resolutions(result, j1_guidelines[0].id, [ResolutionKind.NONE])
    assert_resolutions(result, j2_guidelines[0].id, [ResolutionKind.DEPRIORITIZED])

    # The deprioritized journey-node guideline's resolution should identify
    # the prioritizing journey as a counterpart and reference the PRIORITY
    # relationship between the journey tags.
    j2g_res = get_resolutions_by_kind(result, j2_guidelines[0].id, ResolutionKind.DEPRIORITIZED)
    assert len(j2g_res) == 1
    assert ResolvedEntity.journey(j1) in j2g_res[0].details.counterparts
    assert j2g_res[0].details.relationship is not None
    assert j2g_res[0].details.relationship.id == journey_priority_rel.id


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
        triggers=[journey_condition.id],
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
            id=Tag.for_journey_id(journey.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
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

    assert_resolutions(result, standalone_guideline.id, [ResolutionKind.NONE])
    for jg in journey_guidelines:
        assert_resolutions(result, jg.id, [ResolutionKind.DEPRIORITIZED])


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
        triggers=[journey_condition.id],
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
            id=Tag.for_journey_id(journey.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
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

    for jg in journey_guidelines:
        assert_resolutions(result, jg.id, [ResolutionKind.NONE])
    assert_resolutions(result, standalone_guideline.id, [ResolutionKind.DEPRIORITIZED])


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
        triggers=[journey_condition.id],
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
            id=Tag.for_journey_id(journey.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
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
            id=Tag.for_journey_id(journey.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
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

    assert_resolutions(result, guideline_y.id, [ResolutionKind.NONE])
    assert_resolutions(result, guideline_x.id, [ResolutionKind.DEPRIORITIZED])


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

    assert_resolutions(result, deprioritized_guideline.id, [ResolutionKind.NONE])


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
        triggers=[prioritized_condition.id],
    )
    deprioritized_journey = await journey_store.create_journey(
        title="Journey B",
        description="Lower priority journey",
        triggers=[deprioritized_condition.id],
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(prioritized_journey.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(deprioritized_journey.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
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

    assert_resolutions(result, deprioritized_guideline.id, [ResolutionKind.NONE])


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

    assert_resolutions(result, prioritized_guideline.id, [ResolutionKind.NONE])
    assert_resolutions(result, deprioritized_guideline.id, [ResolutionKind.DEPRIORITIZED])


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

    # Re-read after tagging so usable_guidelines has up-to-date tag lists
    g2 = await guideline_store.read_guideline(g2.id)
    g3 = await guideline_store.read_guideline(g3.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g1.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.TAG_ALL,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.TAG_ALL,
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

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.ENTAILED])
    assert_resolutions(result, g3.id, [ResolutionKind.ENTAILED])
    assert_resolutions(result, g4.id, [ResolutionKind.ENTAILED])


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

    # Re-read after tagging so guideline.tags is up-to-date
    deprioritized_guideline = await guideline_store.read_guideline(deprioritized_guideline.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=prioritized_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=deprioritized_tag.id,
            kind=RelationshipEntityKind.TAG_ALL,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=deprioritized_tag.id,
            kind=RelationshipEntityKind.TAG_ALL,
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

    assert_resolutions(result, deprioritized_guideline.id, [ResolutionKind.NONE])


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

    # Re-read after tagging so guideline.tags is up-to-date
    g2 = await guideline_store.read_guideline(g2.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g1.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.TAG_ALL,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.TAG_ALL,
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

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.DEPRIORITIZED])


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

    # Re-read after tagging so guideline.tags is up-to-date
    g2 = await guideline_store.read_guideline(g2.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g1.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.TAG_ALL,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=t1.id,
            kind=RelationshipEntityKind.TAG_ALL,
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

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.DEPRIORITIZED])


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

    assert_resolutions(result, source_guideline.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_relational_resolver_keeps_guideline_depending_on_tag_when_at_least_one_tagged_member_is_matched(
    container: Container,
) -> None:
    """
    Tag dependency uses ANY semantics: a guideline depending on a tag survives
    as long as at least one tagged member is matched.

    - source_guideline depends on target_tag
    - target_tag has tagged_guideline_1 and tagged_guideline_2
    - Only tagged_guideline_1 is matched
    - Expected: source_guideline survives (ANY member matched)
    """
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

    # Re-read after tagging so usable_guidelines has up-to-date tag lists
    tagged_guideline_1 = await guideline_store.read_guideline(tagged_guideline_1.id)
    tagged_guideline_2 = await guideline_store.read_guideline(tagged_guideline_2.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=source_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=target_tag.id,
            kind=RelationshipEntityKind.TAG_ANY,
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

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {source_guideline.id, tagged_guideline_1.id}

    assert_resolutions(result, source_guideline.id, [ResolutionKind.NONE])
    assert_resolutions(result, tagged_guideline_1.id, [ResolutionKind.NONE])


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
        triggers=[source_condition.id],
    )

    guideline = await guideline_store.create_guideline(
        condition="Customer wants to make a transaction",
        action="Help them",
    )

    source_journey_guidelines = await projection.project_journey_to_guidelines(source_journey.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(source_journey.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
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

    assert_resolutions(
        result, source_journey_guidelines[0].id, [ResolutionKind.UNMET_DEPENDENCY_ALL]
    )


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
        triggers=[source_condition.id],
    )

    target_journey = await journey_store.create_journey(
        title="Validate Account",
        description="Journey for validating account",
        triggers=[],
    )

    source_journey_guidelines = await projection.project_journey_to_guidelines(source_journey.id)
    target_journey_guidelines = await projection.project_journey_to_guidelines(target_journey.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(source_journey.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(target_journey.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
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

    assert_resolutions(
        result, source_journey_guidelines[0].id, [ResolutionKind.UNMET_DEPENDENCY_ALL]
    )


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
        triggers=[],
    )
    disabled_journey = await journey_store.create_journey(
        title="Second Journey",
        description="Description",
        triggers=[],
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
            id=Tag.for_journey_id(enabled_journey.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=disabled_journey_tagged_guideline.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(disabled_journey.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
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

    assert_resolutions(result, enabled_journey_tagged_guideline.id, [ResolutionKind.NONE])
    assert_resolutions(
        result, disabled_journey_tagged_guideline.id, [ResolutionKind.UNMET_DEPENDENCY_ALL]
    )


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

    assert_resolutions(result, guideline_a.id, [ResolutionKind.NONE])
    assert_resolutions(result, guideline_b.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, guideline_c.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, guideline_d.id, [ResolutionKind.NONE])


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

    assert_resolutions(result, guideline_a.id, [ResolutionKind.NONE])
    assert_resolutions(result, guideline_z.id, [ResolutionKind.ENTAILED])
    assert_resolutions(result, guideline_y.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, guideline_x.id, [ResolutionKind.DEPRIORITIZED])


async def test_that_relational_resolver_filters_guidelines_by_priority_keeping_only_highest(
    container: Container,
) -> None:
    """
    Tests that after all relational resolution, only guidelines sharing the
    highest priority value survive.

    - Guideline A has priority=1
    - Guideline B has priority=0 (default)
    - Both are active matches with no relationships between them
    - Expected: Only A survives because it has the highest priority
    """
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    guideline_a = await guideline_store.create_guideline(
        condition="Customer asks about pricing",
        action="Provide premium pricing",
        priority=1,
    )
    guideline_b = await guideline_store.create_guideline(
        condition="Customer asks about pricing",
        action="Provide standard pricing",
        priority=0,
    )

    result = await resolver.resolve(
        [guideline_a, guideline_b],
        [
            GuidelineMatch(guideline=guideline_a, score=8, rationale=""),
            GuidelineMatch(guideline=guideline_b, score=9, rationale=""),
        ],
        journeys=[],
    )

    assert len(result.matches) == 1
    assert result.matches[0].guideline.id == guideline_a.id

    assert_resolutions(result, guideline_a.id, [ResolutionKind.NONE])
    assert_resolutions(result, guideline_b.id, [ResolutionKind.DEPRIORITIZED])


async def test_that_relational_resolver_filters_journeys_by_priority_keeping_only_highest(
    container: Container,
) -> None:
    """
    Tests that after all relational resolution, only journeys sharing the
    highest priority value (and their guidelines) survive.

    - Journey 1 has priority=2
    - Journey 2 has priority=0 (default)
    - Both journeys' guidelines are active matches
    - Expected: Only Journey 1's guidelines survive
    """
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    projection = container[JourneyGuidelineProjection]
    resolver = container[RelationalResolver]

    j1_condition = await guideline_store.create_guideline(
        condition="Customer is interested in Journey 1"
    )
    j2_condition = await guideline_store.create_guideline(
        condition="Customer is interested in Journey 2"
    )

    j1 = await journey_store.create_journey(
        title="Journey 1",
        description="High priority journey",
        triggers=[j1_condition.id],
        priority=2,
    )

    j2 = await journey_store.create_journey(
        title="Journey 2",
        description="Default priority journey",
        triggers=[j2_condition.id],
        priority=0,
    )

    j1_guidelines = await projection.project_journey_to_guidelines(j1.id)
    j2_guidelines = await projection.project_journey_to_guidelines(j2.id)

    assert len(j1_guidelines) == 1
    assert len(j2_guidelines) == 1

    result = await resolver.resolve(
        list(j1_guidelines) + list(j2_guidelines),
        [
            GuidelineMatch(guideline=j1_guidelines[0], score=8, rationale=""),
            GuidelineMatch(guideline=j2_guidelines[0], score=9, rationale=""),
        ],
        journeys=[j1, j2],
    )

    assert len(result.matches) == 1
    assert result.matches[0].guideline.id == j1_guidelines[0].id
    assert len(result.journeys) == 1
    assert result.journeys[0].id == j1.id

    assert_resolutions(result, j1_guidelines[0].id, [ResolutionKind.NONE])
    assert_resolutions(result, j2_guidelines[0].id, [ResolutionKind.DEPRIORITIZED])


async def test_that_relational_resolver_filters_mixed_entities_by_priority_with_prioritized_guideline_to_keep_only_the_guideline(
    container: Container,
) -> None:
    """
    Tests cross-entity priority comparison between standalone guidelines and journeys.

    - Standalone guideline has priority=1
    - Journey has priority=0 (default)
    - Both are active
    - Expected: Only the standalone guideline survives; the journey and its
      guidelines are filtered out because priority=0 < priority=1
    """
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    projection = container[JourneyGuidelineProjection]
    resolver = container[RelationalResolver]

    standalone_guideline = await guideline_store.create_guideline(
        condition="Customer asks about drinks",
        action="Recommend water",
        priority=1,
    )

    journey_condition = await guideline_store.create_guideline(
        condition="Customer asks about drinks"
    )

    journey = await journey_store.create_journey(
        title="Drink Recommendation Journey",
        description="Recommend soda",
        triggers=[journey_condition.id],
        priority=0,
    )

    journey_guidelines = await projection.project_journey_to_guidelines(journey.id)
    assert len(journey_guidelines) > 0

    journey_matches = [
        GuidelineMatch(guideline=g, score=7 + i, rationale="")
        for i, g in enumerate(journey_guidelines)
    ]

    result = await resolver.resolve(
        [standalone_guideline] + list(journey_guidelines),
        [GuidelineMatch(guideline=standalone_guideline, score=8, rationale="")] + journey_matches,
        journeys=[journey],
    )

    assert len(result.matches) == 1
    assert result.matches[0].guideline.id == standalone_guideline.id
    assert len(result.journeys) == 0

    assert_resolutions(result, standalone_guideline.id, [ResolutionKind.NONE])
    for jg in journey_guidelines:
        assert_resolutions(result, jg.id, [ResolutionKind.DEPRIORITIZED])


async def test_that_relational_resolver_filters_mixed_entities_by_priority_with_prioritized_journey_to_keep_only_the_journey(
    container: Container,
) -> None:
    """
    Tests cross-entity priority comparison where the journey has higher priority.

    - Standalone guideline has priority=0 (default)
    - Journey has priority=1
    - Both are active
    - Expected: Only the journey and its guidelines survive; the standalone
      guideline is filtered out because priority=0 < priority=1
    """
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    projection = container[JourneyGuidelineProjection]
    resolver = container[RelationalResolver]

    standalone_guideline = await guideline_store.create_guideline(
        condition="Customer asks about drinks",
        action="Recommend water",
        priority=0,
    )

    journey_condition = await guideline_store.create_guideline(
        condition="Customer asks about drinks"
    )

    journey = await journey_store.create_journey(
        title="Drink Recommendation Journey",
        description="Recommend soda",
        triggers=[journey_condition.id],
        priority=1,
    )

    journey_guidelines = await projection.project_journey_to_guidelines(journey.id)
    assert len(journey_guidelines) > 0

    journey_matches = [
        GuidelineMatch(guideline=g, score=7 + i, rationale="")
        for i, g in enumerate(journey_guidelines)
    ]

    result = await resolver.resolve(
        [standalone_guideline] + list(journey_guidelines),
        [GuidelineMatch(guideline=standalone_guideline, score=10, rationale="")] + journey_matches,
        journeys=[journey],
    )

    assert all(m.guideline.id != standalone_guideline.id for m in result.matches)
    assert len(result.matches) == len(journey_guidelines)
    assert len(result.journeys) == 1
    assert result.journeys[0].id == journey.id

    assert_resolutions(result, standalone_guideline.id, [ResolutionKind.DEPRIORITIZED])
    for jg in journey_guidelines:
        assert_resolutions(result, jg.id, [ResolutionKind.NONE])


async def test_that_relational_resolver_deprioritizes_target_guideline_when_source_is_custom_tag(
    container: Container,
) -> None:
    """
    Tests that a custom tag used as source in a PRIORITY relationship
    deprioritizes the target guideline when a guideline tagged with that
    tag is matched.

    - Tag t1 is attached to g1
    - t1 PRIORITY → g2
    - Both g1 and g2 are matched
    - Expected: g2 is deprioritized, only g1 remains
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="b")
    g2 = await guideline_store.create_guideline(condition="c", action="d")

    t1 = await tag_store.create_tag(name="t1")
    await guideline_store.upsert_tag(g1.id, t1.id)
    g1 = await guideline_store.read_guideline(g1.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
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

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.DEPRIORITIZED])


async def test_that_relational_resolver_filters_tagged_guideline_when_custom_tag_dependency_is_unmet(
    container: Container,
) -> None:
    """
    Tests that a custom tag used as source in a DEPENDENCY relationship
    deactivates the tagged guideline when the dependency target is not matched.

    - Tag t1 is attached to g1
    - t1 DEPENDENCY → g2  (g1, via t1, depends on g2)
    - g1 is matched but g2 is NOT matched
    - Expected: g1 is filtered out (unmet dependency via tag)
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="b")
    g2 = await guideline_store.create_guideline(condition="c", action="d")

    t1 = await tag_store.create_tag(name="t1")
    await guideline_store.upsert_tag(g1.id, t1.id)
    g1 = await guideline_store.read_guideline(g1.id)

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2],
        [
            GuidelineMatch(guideline=g1, score=8, rationale=""),
            # g2 is NOT matched — dependency unmet
        ],
        journeys=[],
    )

    assert result.matches == []

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_relational_resolver_transitively_filters_guideline_depending_on_custom_tag_with_deprioritized_member(
    container: Container,
) -> None:
    """
    Tests the transitive effect of priority + dependency via a custom tag:
    - g1 prioritizes over g2
    - g2 is tagged with t1
    - g3 depends on tag t1
    - When all three are matched, g2 is deprioritized by g1,
      then g3 is transitively filtered (t1 member g2 was deprioritized).
      Only g1 remains.
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="Recommend Pepsi")
    g2 = await guideline_store.create_guideline(condition="b", action="Recommend Coke")
    g3 = await guideline_store.create_guideline(condition="c", action="Recommend Sprite")

    t1 = await tag_store.create_tag(name="drink-group")
    await guideline_store.upsert_tag(g2.id, t1.id)
    g2 = await guideline_store.read_guideline(g2.id)

    # g1 prioritizes over g2 (g2 gets deprioritized)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.PRIORITY,
    )

    # g3 depends on tag t1 (i.e. at least one guideline tagged with t1 being active)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=9, rationale=""),
            GuidelineMatch(guideline=g2, score=7, rationale=""),
            GuidelineMatch(guideline=g3, score=6, rationale=""),
        ],
        journeys=[],
    )

    # Only g1 should remain:
    # - g2 is deprioritized by g1
    # - g3 depends on tag t1, whose member g2 was deprioritized, so g3 is filtered
    assert result.matches == [GuidelineMatch(guideline=g1, score=9, rationale="")]

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g3.id, [ResolutionKind.DEPRIORITIZED])


async def test_that_tag_priority_excludes_all_target_members_regardless_of_individual_priority(
    container: Container,
) -> None:
    """
    Tests that tag-level prioritization is absolute — individual-level priority
    relationships do NOT grant immunity from tag-level deprioritization:
    - t1 prioritizes over t2 (tag-level: all of t1 beats all of t2)
    - g2_1 prioritizes over g1_1 (guideline-level)
    - After resolution: g1_1 deprioritized by g2_1 (guideline-level),
      g2_1 and g2_2 deprioritized by t1 (tag-level). Only g1_2 survives.
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")

    g1_1 = await guideline_store.create_guideline(condition="a", action="g1_1 action")
    g1_2 = await guideline_store.create_guideline(condition="b", action="g1_2 action")
    g2_1 = await guideline_store.create_guideline(condition="c", action="g2_1 action")
    g2_2 = await guideline_store.create_guideline(condition="d", action="g2_2 action")

    await guideline_store.upsert_tag(g1_1.id, t1.id)
    await guideline_store.upsert_tag(g1_2.id, t1.id)
    await guideline_store.upsert_tag(g2_1.id, t2.id)
    await guideline_store.upsert_tag(g2_2.id, t2.id)

    g1_1 = await guideline_store.read_guideline(g1_1.id)
    g1_2 = await guideline_store.read_guideline(g1_2.id)
    g2_1 = await guideline_store.read_guideline(g2_1.id)
    g2_2 = await guideline_store.read_guideline(g2_2.id)

    # t1 prioritizes over t2 (tag-level)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.PRIORITY,
    )

    # g2_1 prioritizes over g1_1 (guideline-level — does NOT grant immunity from tag-level)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2_1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g1_1.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g1_1, g1_2, g2_1, g2_2],
        [
            GuidelineMatch(guideline=g1_1, score=10, rationale=""),
            GuidelineMatch(guideline=g1_2, score=10, rationale=""),
            GuidelineMatch(guideline=g2_1, score=10, rationale=""),
            GuidelineMatch(guideline=g2_2, score=10, rationale=""),
        ],
        journeys=[],
    )

    # g1_1 deprioritized by g2_1, then g2_1 and g2_2 deprioritized by t1→t2.
    # Only g1_2 survives.
    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1_2.id}

    assert_resolutions(result, g1_1.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g1_2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2_1.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g2_2.id, [ResolutionKind.DEPRIORITIZED])


async def test_that_tag_priority_deprioritizes_all_guidelines_of_target_tag(
    container: Container,
) -> None:
    """
    Tests that tag-level prioritization filters out all guidelines of the target tag:
    - t1 prioritizes over t2
    - g1_1, g1_2 tagged with t1; g2_1, g2_2 tagged with t2
    - After resolution only g1_1 and g1_2 remain (t2 guidelines are deprioritized).
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")

    g1_1 = await guideline_store.create_guideline(condition="a", action="g1_1 action")
    g1_2 = await guideline_store.create_guideline(condition="b", action="g1_2 action")
    g2_1 = await guideline_store.create_guideline(condition="c", action="g2_1 action")
    g2_2 = await guideline_store.create_guideline(condition="d", action="g2_2 action")

    await guideline_store.upsert_tag(g1_1.id, t1.id)
    await guideline_store.upsert_tag(g1_2.id, t1.id)
    await guideline_store.upsert_tag(g2_1.id, t2.id)
    await guideline_store.upsert_tag(g2_2.id, t2.id)

    g1_1 = await guideline_store.read_guideline(g1_1.id)
    g1_2 = await guideline_store.read_guideline(g1_2.id)
    g2_1 = await guideline_store.read_guideline(g2_1.id)
    g2_2 = await guideline_store.read_guideline(g2_2.id)

    # t1 prioritizes over t2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g1_1, g1_2, g2_1, g2_2],
        [
            GuidelineMatch(guideline=g1_1, score=10, rationale=""),
            GuidelineMatch(guideline=g1_2, score=10, rationale=""),
            GuidelineMatch(guideline=g2_1, score=10, rationale=""),
            GuidelineMatch(guideline=g2_2, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1_1.id, g1_2.id}

    assert_resolutions(result, g1_1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g1_2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2_1.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g2_2.id, [ResolutionKind.DEPRIORITIZED])


async def test_that_journey_tag_priority_deprioritizes_all_guidelines_of_target_tag(
    container: Container,
) -> None:
    """
    Tests that a journey prioritizing over a custom tag filters out all
    guidelines tagged with that tag:
    - Journey J (with j_cond) prioritizes over t1
    - g1, g2 tagged with t1
    - After resolution only j_cond remains (t1 guidelines are deprioritized).
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")

    await guideline_store.upsert_tag(g1.id, t1.id)
    await guideline_store.upsert_tag(g2.id, t1.id)
    g1 = await guideline_store.read_guideline(g1.id)
    g2 = await guideline_store.read_guideline(g2.id)

    j_cond = await guideline_store.create_guideline(condition="c", action="journey action")
    journey = await journey_store.create_journey(
        title="J",
        description="A journey",
        triggers=[j_cond.id],
    )

    # Tag condition guideline with its journey tag (as the real projection does)
    await guideline_store.upsert_tag(j_cond.id, Tag.for_journey_id(journey.id).id)
    j_cond = await guideline_store.read_guideline(j_cond.id)

    # Journey J prioritizes over t1
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(journey.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g1, g2, j_cond],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=j_cond, score=10, rationale=""),
        ],
        journeys=[journey],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {j_cond.id}

    assert_resolutions(result, j_cond.id, [ResolutionKind.NONE])
    assert_resolutions(result, g1.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g2.id, [ResolutionKind.DEPRIORITIZED])


async def test_that_journey_tag_priority_deprioritizes_target_journey_tag(
    container: Container,
) -> None:
    """
    Tests that a journey prioritizing over another journey filters out the
    target journey's node guidelines:
    - Journey J1 prioritizes over Journey J2
    - j1_g and j2_g are node guidelines (with journey_node metadata)
    - After resolution only j1_g remains (j2_g is deprioritized).
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])
    j2 = await journey_store.create_journey(title="J2", description="Journey 2", triggers=[])

    j1_g = await guideline_store.create_guideline(
        condition="a",
        action="j1 action",
        metadata={"journey_node": {"journey_id": j1.id}},
    )
    j2_g = await guideline_store.create_guideline(
        condition="b",
        action="j2 action",
        metadata={"journey_node": {"journey_id": j2.id}},
    )

    # J1 prioritizes over J2
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j2.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [j1_g, j2_g],
        [
            GuidelineMatch(guideline=j1_g, score=10, rationale=""),
            GuidelineMatch(guideline=j2_g, score=10, rationale=""),
        ],
        journeys=[j1, j2],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {j1_g.id}

    assert_resolutions(result, j1_g.id, [ResolutionKind.NONE])
    assert_resolutions(result, j2_g.id, [ResolutionKind.DEPRIORITIZED])


async def test_that_tag_priority_deprioritizes_target_journey(
    container: Container,
) -> None:
    """
    Tests that a custom tag prioritizing over a journey filters out the
    journey's node guidelines:
    - t1 (with g1, g2) prioritizes over Journey J (with j_g node guideline)
    - After resolution g1 and g2 remain, j_g is deprioritized.
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")

    await guideline_store.upsert_tag(g1.id, t1.id)
    await guideline_store.upsert_tag(g2.id, t1.id)
    g1 = await guideline_store.read_guideline(g1.id)
    g2 = await guideline_store.read_guideline(g2.id)

    journey = await journey_store.create_journey(
        title="J",
        description="A journey",
        triggers=[],
    )

    j_g = await guideline_store.create_guideline(
        condition="c",
        action="journey action",
        metadata={"journey_node": {"journey_id": journey.id}},
    )

    # t1 prioritizes over Journey J
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(
            id=Tag.for_journey_id(journey.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g1, g2, j_g],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=j_g, score=10, rationale=""),
        ],
        journeys=[journey],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, j_g.id, [ResolutionKind.DEPRIORITIZED])


# ── Tag-level dependency tests ──────────────────────────────────────────────


async def test_that_tag_dependency_deactivates_tagged_guidelines_when_target_guideline_not_met(
    container: Container,
) -> None:
    """
    t1 depends on g2. g1 tagged t1, g2 untagged, g3 untagged.
    g2 NOT matched → t1 dependency unmet → g1 deactivated.
    Result: {g3}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", tags=[t1.id])
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")

    # t1 depends on g2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # g2 NOT matched
            GuidelineMatch(guideline=g3, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_tag_dependency_deactivates_tagged_guidelines_when_target_tag_not_met(
    container: Container,
) -> None:
    """
    t1 depends on t2. g1 tagged t1, g2 tagged t2, g3 untagged.
    g2 is NOT matched → t2 dependency unmet → g1 deactivated.
    Result: {g3}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", tags=[t1.id])
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t2.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")

    # t1 depends on t2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # g2 NOT matched
            GuidelineMatch(guideline=g3, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_journey_tag_dependency_deactivates_node_guidelines_when_target_tag_not_met(
    container: Container,
) -> None:
    """
    Journey j1 depends on t1. j1_g is a j1 node guideline, g1 tagged t1, g_extra untagged.
    g1 NOT matched → t1 dependency unmet → j1_g deactivated.
    Result: {g_extra}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", tags=[t1.id])

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])
    j1_g = await guideline_store.create_guideline(
        condition="b",
        action="j1 action",
        metadata={"journey_node": {"journey_id": j1.id}},
    )

    g_extra = await guideline_store.create_guideline(condition="c", action="extra action")

    # j1 depends on t1
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, j1_g, g_extra],
        [
            # g1 NOT matched
            GuidelineMatch(guideline=j1_g, score=10, rationale=""),
            GuidelineMatch(guideline=g_extra, score=10, rationale=""),
        ],
        journeys=[j1],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g_extra.id}

    assert_resolutions(result, j1_g.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g_extra.id, [ResolutionKind.NONE])


async def test_that_tag_dependency_deactivates_tagged_guidelines_when_target_journey_not_active(
    container: Container,
) -> None:
    """
    t1 depends on journey j1. g1 tagged t1, g_extra untagged.
    j1 NOT in active journeys → t1 dependency unmet → g1 deactivated.
    Result: {g_extra}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", tags=[t1.id])
    g_extra = await guideline_store.create_guideline(condition="b", action="extra action")

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])

    # t1 depends on j1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g_extra],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g_extra, score=10, rationale=""),
        ],
        journeys=[],  # j1 NOT active
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g_extra.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g_extra.id, [ResolutionKind.NONE])


async def test_that_journey_tag_dependency_deactivates_node_guidelines_when_target_journey_tag_not_active(
    container: Container,
) -> None:
    """
    Journey j1 depends on journey j2. j1_g is a j1 node guideline, g_extra untagged.
    j2 NOT in active journeys → j1 dependency unmet → j1_g deactivated.
    Result: {g_extra}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])
    j2 = await journey_store.create_journey(title="J2", description="Journey 2", triggers=[])

    j1_g = await guideline_store.create_guideline(
        condition="a",
        action="j1 action",
        metadata={"journey_node": {"journey_id": j1.id}},
    )

    g_extra = await guideline_store.create_guideline(condition="b", action="extra action")

    # j1 depends on j2
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j2.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [j1_g, g_extra],
        [
            GuidelineMatch(guideline=j1_g, score=10, rationale=""),
            GuidelineMatch(guideline=g_extra, score=10, rationale=""),
        ],
        journeys=[j1],  # j2 NOT active
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g_extra.id}

    assert_resolutions(result, j1_g.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g_extra.id, [ResolutionKind.NONE])


# ── ANY-semantics tag dependency tests ─────────────────────────────────────


async def test_that_guideline_depending_on_tag_is_filtered_when_no_tagged_guideline_is_matched(
    container: Container,
) -> None:
    """
    g1 depends on tag t1. t1 has g2 and g3, neither matched.
    0 of 2 matched → g1 filtered.
    Result: {g_extra}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t1.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t1.id])
    g_extra = await guideline_store.create_guideline(condition="d", action="extra action")

    # g1 depends on t1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3, g_extra],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # g2 NOT matched
            # g3 NOT matched
            GuidelineMatch(guideline=g_extra, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g_extra.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g_extra.id, [ResolutionKind.NONE])


async def test_that_guideline_depending_on_tag_survives_when_at_least_one_tagged_guideline_is_matched(
    container: Container,
) -> None:
    """
    g1 depends on tag t1. t1 has g2 and g3, only g2 matched.
    1 of 2 matched → g1 survives (ANY semantics).
    Result: {g1, g2}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t1.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t1.id])

    # g1 depends on t1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            # g3 NOT matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_guideline_depending_on_tag_survives_when_at_least_one_tagged_journey_is_active(
    container: Container,
) -> None:
    """
    g1 depends on tag t1. t1 has journey j1 and journey j2 (via journey tags).
    Only j1 is active → g1 survives (ANY semantics).
    Result: {g1, j1_g}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])
    j1_g = await guideline_store.create_guideline(
        condition="b",
        action="j1 action",
        metadata={"journey_node": {"journey_id": j1.id}},
        tags=[t1.id],
    )

    j2 = await journey_store.create_journey(title="J2", description="Journey 2", triggers=[])
    j2_g = await guideline_store.create_guideline(
        condition="c",
        action="j2 action",
        metadata={"journey_node": {"journey_id": j2.id}},
        tags=[t1.id],
    )

    # g1 depends on t1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, j1_g, j2_g],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=j1_g, score=10, rationale=""),
            # j2_g NOT matched
        ],
        journeys=[j1],  # only j1 active, j2 NOT active
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, j1_g.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, j1_g.id, [ResolutionKind.NONE])


async def test_that_guideline_depending_on_tag_is_filtered_when_no_tagged_journey_is_active(
    container: Container,
) -> None:
    """
    g1 depends on tag t1. t1 has journey j1 and journey j2 (via journey tags).
    Neither j1 nor j2 is active → g1 filtered.
    Result: {g_extra}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g_extra = await guideline_store.create_guideline(condition="d", action="extra action")

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])
    j1_g = await guideline_store.create_guideline(
        condition="b",
        action="j1 action",
        metadata={"journey_node": {"journey_id": j1.id}},
        tags=[t1.id],
    )

    j2 = await journey_store.create_journey(title="J2", description="Journey 2", triggers=[])
    j2_g = await guideline_store.create_guideline(
        condition="c",
        action="j2 action",
        metadata={"journey_node": {"journey_id": j2.id}},
        tags=[t1.id],
    )

    # g1 depends on t1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, j1_g, j2_g, g_extra],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # j1_g NOT matched
            # j2_g NOT matched
            GuidelineMatch(guideline=g_extra, score=10, rationale=""),
        ],
        journeys=[],  # neither j1 nor j2 active
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g_extra.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g_extra.id, [ResolutionKind.NONE])


async def test_that_tag_to_tag_dependency_survives_when_at_least_one_target_tag_member_is_matched(
    container: Container,
) -> None:
    """
    t1 depends on t2. g1 tagged t1, g2 and g3 tagged t2.
    Only g2 matched → t1 dependency met (ANY). g1 survives.
    Result: {g1, g2}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", tags=[t1.id])
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t2.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t2.id])

    # t1 depends on t2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            # g3 NOT matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_journey_tag_dependency_survives_when_at_least_one_target_tag_member_is_matched(
    container: Container,
) -> None:
    """
    Journey j1 depends on t1. t1 has g1 and g2.
    Only g1 matched → j1 dependency met (ANY). j1_g survives.
    Result: {j1_g, g1}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", tags=[t1.id])
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t1.id])

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])
    j1_g = await guideline_store.create_guideline(
        condition="c",
        action="j1 action",
        metadata={"journey_node": {"journey_id": j1.id}},
    )

    # j1 depends on t1
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, j1_g],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # g2 NOT matched
            GuidelineMatch(guideline=j1_g, score=10, rationale=""),
        ],
        journeys=[j1],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {j1_g.id, g1.id}

    assert_resolutions(result, j1_g.id, [ResolutionKind.NONE])
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])


async def test_that_tag_dependency_survives_when_tagged_journey_is_active_but_tagged_guideline_is_not_matched(
    container: Container,
) -> None:
    """
    g1 depends on tag t1. t1 has both a guideline (g2) and a journey (j1 node).
    g2 is NOT matched but j1 is active → g1 survives (ANY semantics across entity types).
    Result: {g1, j1_g}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t1.id])

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])
    j1_g = await guideline_store.create_guideline(
        condition="c",
        action="j1 action",
        metadata={"journey_node": {"journey_id": j1.id}},
        tags=[t1.id],
    )

    # g1 depends on t1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, j1_g],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # g2 NOT matched
            GuidelineMatch(guideline=j1_g, score=10, rationale=""),
        ],
        journeys=[j1],  # j1 active
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, j1_g.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, j1_g.id, [ResolutionKind.NONE])


# ── Dependency hierarchy tests ─────────────────────────────────────────────
# Arrow notation: A -> B -> C means B depends on A, C depends on B.


# Case 1: G1 -> G2 -> G3 (guideline chain)


async def test_that_hierarchical_guideline_dependency_cascades_when_root_is_not_matched(
    container: Container,
) -> None:
    """
    G2 depends on G1, G3 depends on G2.
    G1 NOT matched → G2 filtered, G3 filtered (cascade).
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")

    # G2 depends on G1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G3 depends on G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            # G1 NOT matched
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == set()

    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(
        result,
        g3.id,
        [ResolutionKind.UNMET_DEPENDENCY_ALL],
    )


async def test_that_hierarchical_guideline_dependency_cascades_when_middle_is_not_matched(
    container: Container,
) -> None:
    """
    G2 depends on G1, G3 depends on G2.
    G1 matched, G2 NOT matched → G3 filtered, G1 survives.
    Result: {G1}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")

    # G2 depends on G1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G3 depends on G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # G2 NOT matched
            GuidelineMatch(guideline=g3, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


# Case 2: G1 -> J1 -> G2 (guideline → journey → guideline)


async def test_that_hierarchical_journey_dependency_cascades_when_root_guideline_is_not_matched(
    container: Container,
) -> None:
    """
    J1 depends on G1, G2 depends on J1.
    G1 NOT matched → J1 node filtered, G2 filtered (cascade).
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])
    j1_g = await guideline_store.create_guideline(
        condition="b",
        action="j1 action",
        metadata={"journey_node": {"journey_id": j1.id}},
    )

    g2 = await guideline_store.create_guideline(condition="c", action="g2 action")

    # J1 depends on G1
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        target=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G2 depends on J1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, j1_g, g2],
        [
            # G1 NOT matched
            GuidelineMatch(guideline=j1_g, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
        ],
        journeys=[j1],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == set()

    assert_resolutions(result, j1_g.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_hierarchical_journey_dependency_cascades_when_journey_is_not_active(
    container: Container,
) -> None:
    """
    J1 depends on G1, G2 depends on J1.
    G1 matched, J1 NOT active → G2 filtered, G1 survives.
    Result: {G1}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])
    j1_g = await guideline_store.create_guideline(
        condition="b",
        action="j1 action",
        metadata={"journey_node": {"journey_id": j1.id}},
    )

    g2 = await guideline_store.create_guideline(condition="c", action="g2 action")

    # J1 depends on G1
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        target=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G2 depends on J1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, j1_g, g2],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # j1_g NOT matched (journey not active)
            GuidelineMatch(guideline=g2, score=10, rationale=""),
        ],
        journeys=[],  # J1 NOT active
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id}


# Case 3: T1 -> T2 -> T3 (tag chain)


async def test_that_hierarchical_tag_dependency_cascades_when_root_tag_member_is_not_matched(
    container: Container,
) -> None:
    """
    T2 depends on T1, T3 depends on T2.
    T1's guideline NOT matched → T2's guideline filtered, T3's guideline filtered.
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")
    t3 = await tag_store.create_tag(name="t3")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", tags=[t1.id])
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t2.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t3.id])

    # T2 depends on T1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    # T3 depends on T2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t3.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            # g1 NOT matched (T1 unmet)
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == set()

    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(
        result,
        g3.id,
        [ResolutionKind.UNMET_DEPENDENCY_ALL],
    )


async def test_that_hierarchical_tag_dependency_cascades_when_middle_tag_member_is_not_matched(
    container: Container,
) -> None:
    """
    T2 depends on T1, T3 depends on T2.
    T1's guideline matched, T2's guideline NOT matched → T3's guideline filtered, T1's survives.
    Result: {g1}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")
    t3 = await tag_store.create_tag(name="t3")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", tags=[t1.id])
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t2.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t3.id])

    # T2 depends on T1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    # T3 depends on T2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t3.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # g2 NOT matched (T2 unmet)
            GuidelineMatch(guideline=g3, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


# Case 4: T1 -> J1 -> T2 (tag → journey → tag)


async def test_that_hierarchical_tag_journey_tag_dependency_cascades_when_root_tag_is_not_matched(
    container: Container,
) -> None:
    """
    J1 depends on T1, T2 depends on J1.
    T1's guideline NOT matched → J1 node filtered, T2's guideline filtered.
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", tags=[t1.id])

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])
    j1_g = await guideline_store.create_guideline(
        condition="b",
        action="j1 action",
        metadata={"journey_node": {"journey_id": j1.id}},
    )

    g2 = await guideline_store.create_guideline(condition="c", action="g2 action", tags=[t2.id])

    # J1 depends on T1
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    # T2 depends on J1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, j1_g, g2],
        [
            # g1 NOT matched (T1 unmet)
            GuidelineMatch(guideline=j1_g, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
        ],
        journeys=[j1],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == set()

    assert_resolutions(result, j1_g.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_hierarchical_tag_journey_tag_dependency_cascades_when_journey_is_not_active(
    container: Container,
) -> None:
    """
    J1 depends on T1, T2 depends on J1.
    T1's guideline matched, J1 NOT active → T2's guideline filtered, T1's survives.
    Result: {g1}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", tags=[t1.id])

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])
    j1_g = await guideline_store.create_guideline(
        condition="b",
        action="j1 action",
        metadata={"journey_node": {"journey_id": j1.id}},
    )

    g2 = await guideline_store.create_guideline(condition="c", action="g2 action", tags=[t2.id])

    # J1 depends on T1
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    # T2 depends on J1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, j1_g, g2],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # j1_g NOT matched (journey not active)
            GuidelineMatch(guideline=g2, score=10, rationale=""),
        ],
        journeys=[],  # J1 NOT active
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


# Case 5: G1 -> T1 -> J1 (guideline → tag → journey)


async def test_that_hierarchical_guideline_tag_journey_dependency_cascades_when_root_is_not_matched(
    container: Container,
) -> None:
    """
    T1 depends on G1, J1 depends on T1.
    G1 NOT matched → T1's guideline filtered, J1 node filtered.
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g_t1 = await guideline_store.create_guideline(condition="b", action="t1 action", tags=[t1.id])

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])
    j1_g = await guideline_store.create_guideline(
        condition="c",
        action="j1 action",
        metadata={"journey_node": {"journey_id": j1.id}},
    )

    # T1 depends on G1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # J1 depends on T1
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g_t1, j1_g],
        [
            # G1 NOT matched
            GuidelineMatch(guideline=g_t1, score=10, rationale=""),
            GuidelineMatch(guideline=j1_g, score=10, rationale=""),
        ],
        journeys=[j1],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == set()

    # Resolution assertions
    assert_resolutions(result, g_t1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(
        result,
        j1_g.id,
        [ResolutionKind.UNMET_DEPENDENCY_ALL],
    )


async def test_that_hierarchical_guideline_tag_journey_dependency_cascades_when_tag_member_is_not_matched(
    container: Container,
) -> None:
    """
    T1 depends on G1, J1 depends on T1.
    G1 matched, T1's guideline NOT matched → J1 node filtered, G1 survives.
    Result: {G1}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g_t1 = await guideline_store.create_guideline(condition="b", action="t1 action", tags=[t1.id])

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])
    j1_g = await guideline_store.create_guideline(
        condition="c",
        action="j1 action",
        metadata={"journey_node": {"journey_id": j1.id}},
    )

    # T1 depends on G1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # J1 depends on T1
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g_t1, j1_g],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # g_t1 NOT matched (T1 member unmet)
            GuidelineMatch(guideline=j1_g, score=10, rationale=""),
        ],
        journeys=[j1],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, j1_g.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


# ── Edge-case tests ─────────────────────────────────────────────────────────


async def test_that_condition_guideline_survives_when_its_journey_is_deprioritized(
    container: Container,
) -> None:
    """
    Condition guidelines (tagged with journey tag but no journey_node metadata)
    should NOT be deprioritized when the journey loses a priority fight.
    Only node guidelines (with journey_node metadata) are subject to journey-level
    deprioritization.

    - j1 has a condition guideline (j1_cond) and a node guideline (j1_node)
    - Standalone guideline g1 prioritizes over j1
    - After resolution: j1_node is deprioritized, but j1_cond survives
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])

    j1_cond = await guideline_store.create_guideline(
        condition="customer is interested",
        action="observe interest",
        tags=[Tag.for_journey_id(j1.id).id],
    )

    j1_node = await guideline_store.create_guideline(
        condition="customer is interested",
        action="recommend product",
        metadata={"journey_node": {"journey_id": j1.id}},
    )

    g1 = await guideline_store.create_guideline(
        condition="customer is interested",
        action="recommend alternative",
    )

    # g1 prioritizes over j1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [j1_cond, j1_node, g1],
        [
            GuidelineMatch(guideline=j1_cond, score=10, rationale=""),
            GuidelineMatch(guideline=j1_node, score=10, rationale=""),
            GuidelineMatch(guideline=g1, score=10, rationale=""),
        ],
        journeys=[j1],
    )

    result_ids = {m.guideline.id for m in result.matches}
    # j1_node deprioritized (journey node), j1_cond survives (condition guideline)
    assert result_ids == {g1.id, j1_cond.id}

    # Resolution assertions
    assert_resolutions(result, j1_cond.id, [ResolutionKind.NONE])
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, j1_node.id, [ResolutionKind.DEPRIORITIZED])
    deprioritized = get_resolutions_by_kind(result, j1_node.id, ResolutionKind.DEPRIORITIZED)
    assert any(ResolvedEntity.guideline(g1) in r.details.counterparts for r in deprioritized)


async def test_that_tag_priority_does_not_deprioritize_when_no_source_tag_member_is_matched(
    container: Container,
) -> None:
    """
    Tag-level priority t1→t2 should not fire if no t1 member is matched.
    t2 members should survive.

    - t1 prioritizes over t2
    - g1_1 tagged t1 (NOT matched), g2_1 tagged t2 (matched)
    - After resolution: g2_1 survives (no t1 member is active to trigger deprioritization)
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")

    g1_1 = await guideline_store.create_guideline(condition="a", action="g1_1 action", tags=[t1.id])
    g2_1 = await guideline_store.create_guideline(condition="b", action="g2_1 action", tags=[t2.id])

    # t1 prioritizes over t2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g1_1, g2_1],
        [
            # g1_1 NOT matched
            GuidelineMatch(guideline=g2_1, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g2_1.id}

    # Resolution assertions
    assert_resolutions(result, g2_1.id, [ResolutionKind.NONE])


async def test_that_tag_dependency_allows_tagged_guidelines_when_dependency_is_met(
    container: Container,
) -> None:
    """
    Happy-path for tag-level dependency: when the dependency IS met,
    tagged guidelines should survive normally.

    - t1 depends on t2
    - g1 tagged t1, g2 tagged t2 — both matched
    - After resolution: both survive
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", tags=[t1.id])
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t2.id])

    # t1 depends on t2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_tag_priority_transitively_filters_guideline_depending_on_deprioritized_tag(
    container: Container,
) -> None:
    """
    Tag→tag priority causes deprioritization, then a guideline depending on the
    deprioritized tag is transitively filtered.

    - t1 prioritizes over t2 (tag-level)
    - g3 depends on t2
    - g1_1 tagged t1, g2_1 tagged t2, g3 untagged — all matched
    - After resolution: g2_1 deprioritized by t1, g3 transitively filtered
      (depends on t2, whose member g2_1 was deprioritized). Only g1_1 survives.
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")

    g1_1 = await guideline_store.create_guideline(condition="a", action="g1_1 action", tags=[t1.id])
    g2_1 = await guideline_store.create_guideline(condition="b", action="g2_1 action", tags=[t2.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")

    # t1 prioritizes over t2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.PRIORITY,
    )

    # g3 depends on t2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1_1, g2_1, g3],
        [
            GuidelineMatch(guideline=g1_1, score=10, rationale=""),
            GuidelineMatch(guideline=g2_1, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1_1.id}

    # Resolution assertions
    assert_resolutions(result, g1_1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2_1.id, [ResolutionKind.DEPRIORITIZED])
    deprioritized = get_resolutions_by_kind(result, g2_1.id, ResolutionKind.DEPRIORITIZED)
    assert any(ResolvedEntity.guideline(g1_1) in r.details.counterparts for r in deprioritized)
    assert_resolutions(result, g3.id, [ResolutionKind.DEPRIORITIZED])


# ── Custom journey tag propagation tests ───────────────────────────────────


async def test_that_custom_tagged_journey_deprioritizes_guidelines_with_lower_priority_tag(
    container: Container,
) -> None:
    """
    Journey with custom tag t1, standalone guideline with t2, relationship t1 > t2.
    Node guideline (with journey_node metadata and tags=[t1]) and t2-tagged guideline
    both match → only node guideline survives.
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])

    # Node guideline carrying the journey's custom tag
    j1_node = await guideline_store.create_guideline(
        condition="a",
        action="j1 node action",
        metadata={"journey_node": {"journey_id": j1.id}},
        tags=[t1.id],
    )

    # Standalone guideline tagged t2
    g1 = await guideline_store.create_guideline(condition="b", action="g1 action", tags=[t2.id])

    # t1 prioritizes over t2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [j1_node, g1],
        [
            GuidelineMatch(guideline=j1_node, score=10, rationale=""),
            GuidelineMatch(guideline=g1, score=10, rationale=""),
        ],
        journeys=[j1],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {j1_node.id}

    # Resolution assertions
    assert_resolutions(result, j1_node.id, [ResolutionKind.NONE])
    assert_resolutions(result, g1.id, [ResolutionKind.DEPRIORITIZED])
    deprioritized = get_resolutions_by_kind(result, g1.id, ResolutionKind.DEPRIORITIZED)
    assert any(ResolvedEntity.guideline(j1_node) in r.details.counterparts for r in deprioritized)


async def test_that_higher_priority_tag_deprioritizes_journey_with_matching_custom_tag(
    container: Container,
) -> None:
    """
    Standalone guideline with tag t2, journey node guideline with custom tag t1,
    relationship t2 > t1. Both match → node guideline is deprioritized.
    Result: only t2-tagged guidelines survive.
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])

    # Node guideline carrying the journey's custom tag
    j1_node = await guideline_store.create_guideline(
        condition="a",
        action="j1 node action",
        metadata={"journey_node": {"journey_id": j1.id}},
        tags=[t1.id],
    )

    # Standalone guideline tagged t2
    g1 = await guideline_store.create_guideline(condition="b", action="g1 action", tags=[t2.id])

    # t2 prioritizes over t1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [j1_node, g1],
        [
            GuidelineMatch(guideline=j1_node, score=10, rationale=""),
            GuidelineMatch(guideline=g1, score=10, rationale=""),
        ],
        journeys=[j1],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, j1_node.id, [ResolutionKind.DEPRIORITIZED])
    deprioritized = get_resolutions_by_kind(result, j1_node.id, ResolutionKind.DEPRIORITIZED)
    assert any(ResolvedEntity.guideline(g1) in r.details.counterparts for r in deprioritized)


async def test_that_custom_tagged_journey_dependency_deactivates_node_guidelines_when_target_tag_not_met(
    container: Container,
) -> None:
    """
    Journey with custom tag t1, relationship t1 depends on t2.
    t2-tagged guideline NOT matched → node guideline deactivated.
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])

    # Node guideline carrying the journey's custom tag
    j1_node = await guideline_store.create_guideline(
        condition="a",
        action="j1 node action",
        metadata={"journey_node": {"journey_id": j1.id}},
        tags=[t1.id],
    )

    # Standalone guideline tagged t2 (will NOT be matched)
    g1 = await guideline_store.create_guideline(condition="b", action="g1 action", tags=[t2.id])

    g_extra = await guideline_store.create_guideline(condition="c", action="extra action")

    # t1 depends on t2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [j1_node, g1, g_extra],
        [
            GuidelineMatch(guideline=j1_node, score=10, rationale=""),
            # g1 NOT matched
            GuidelineMatch(guideline=g_extra, score=10, rationale=""),
        ],
        journeys=[j1],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g_extra.id}

    # Resolution assertions
    assert_resolutions(result, j1_node.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g_extra.id, [ResolutionKind.NONE])


async def test_that_tag_dependency_on_custom_tagged_journey_deactivates_when_journey_not_active(
    container: Container,
) -> None:
    """
    Standalone guideline with t2, relationship t2 depends on t1.
    Journey with custom tag t1 not active (no node guidelines matched).
    Result: t2-tagged guideline deactivated.
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])

    # Node guideline carrying the journey's custom tag (will NOT be matched)
    j1_node = await guideline_store.create_guideline(
        condition="a",
        action="j1 node action",
        metadata={"journey_node": {"journey_id": j1.id}},
        tags=[t1.id],
    )

    # Standalone guideline tagged t2
    g1 = await guideline_store.create_guideline(condition="b", action="g1 action", tags=[t2.id])

    g_extra = await guideline_store.create_guideline(condition="c", action="extra action")

    # t2 depends on t1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ALL),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [j1_node, g1, g_extra],
        [
            # j1_node NOT matched (journey not active)
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g_extra, score=10, rationale=""),
        ],
        journeys=[],  # j1 NOT active
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g_extra.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g_extra.id, [ResolutionKind.NONE])


async def test_that_relational_resolver_deprioritizes_journey_scoped_guideline_when_journey_is_deprioritized(
    container: Container,
) -> None:
    """When two journeys both have scoped guidelines (persisted, with dependency
    on journey tag but without journey_node metadata), and one journey has
    priority over the other, only the prioritized journey's scoped guideline
    should survive resolution."""
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    # Create two journeys with conditions
    j1_condition = await guideline_store.create_guideline(condition="Customer asks about drinks")
    j2_condition = await guideline_store.create_guideline(condition="Customer asks about drinks")

    j1 = await journey_store.create_journey(
        title="Journey 1",
        description="",
        triggers=[j1_condition.id],
    )

    j2 = await journey_store.create_journey(
        title="Journey 2",
        description="",
        triggers=[j2_condition.id],
    )

    # Create scoped guidelines for each journey (persisted, no journey_node metadata).
    # This mirrors what the SDK's journey.create_guideline() produces.
    g1 = await guideline_store.create_guideline(
        condition="always",
        action="Recommend Pepsi",
    )

    g2 = await guideline_store.create_guideline(
        condition="always",
        action="Recommend Coca-Cola",
    )

    # Create DEPENDENCY from each guideline to its journey's tag
    # (this is what journey.create_guideline() does in the SDK)
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g1.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g2.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j2.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    # Journey 1 has priority over Journey 2
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j2.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g1, g2],
        [
            GuidelineMatch(guideline=g1, score=8, rationale=""),
            GuidelineMatch(guideline=g2, score=5, rationale=""),
        ],
        journeys=[j1, j2],
    )

    # Only g1 (from the prioritized journey) should survive
    assert result.matches == [GuidelineMatch(guideline=g1, score=8, rationale="")]

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.DEPRIORITIZED])


async def test_that_relational_resolver_deprioritizes_journey_scoped_guideline_when_guideline_prioritizes_over_journey(
    container: Container,
) -> None:
    """When a standalone guideline has priority over a journey, the journey's
    scoped guidelines (persisted, with dependency on journey tag) should be
    filtered out."""
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    j1_condition = await guideline_store.create_guideline(condition="Customer asks about drinks")

    j1 = await journey_store.create_journey(
        title="Journey 1",
        description="",
        triggers=[j1_condition.id],
    )

    # Journey-scoped guideline (persisted, no journey_node metadata)
    g_scoped = await guideline_store.create_guideline(
        condition="always",
        action="Recommend Coca-Cola",
    )

    # Dependency from scoped guideline to the journey tag
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g_scoped.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    # Standalone guideline that prioritizes over the journey
    g_standalone = await guideline_store.create_guideline(
        condition="Customer asks about drinks",
        action="Recommend Pepsi",
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=g_standalone.id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id,
            kind=RelationshipEntityKind.TAG_ALL,
        ),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g_standalone, g_scoped],
        [
            GuidelineMatch(guideline=g_standalone, score=8, rationale=""),
            GuidelineMatch(guideline=g_scoped, score=5, rationale=""),
        ],
        journeys=[j1],
    )

    # Only the standalone guideline should survive
    assert result.matches == [GuidelineMatch(guideline=g_standalone, score=8, rationale="")]

    # Resolution assertions
    assert_resolutions(result, g_standalone.id, [ResolutionKind.NONE])
    assert_resolutions(result, g_scoped.id, [ResolutionKind.DEPRIORITIZED])


# ── Dependency edge-case tests ─────────────────────────────────────────────


async def test_that_diamond_dependency_filters_all_dependents_when_root_is_not_matched(
    container: Container,
) -> None:
    """
    Diamond: G2 depends on G4, G3 depends on G4, G1 depends on both G2 and G3.
    G4 NOT matched → G2 filtered, G3 filtered → G1 filtered (both deps unmet).
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action")

    # G2 depends on G4
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G3 depends on G4
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G1 depends on G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G1 depends on G3
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3, g4],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
            # G4 NOT matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == set()

    # Resolution assertions
    # G1 has two direct deps (G2 and G3) — both unmet
    assert_resolutions(
        result,
        g1.id,
        [
            ResolutionKind.UNMET_DEPENDENCY_ALL,
            ResolutionKind.UNMET_DEPENDENCY_ALL,
        ],
    )
    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g3.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_diamond_dependency_keeps_all_when_root_is_matched(
    container: Container,
) -> None:
    """
    Diamond: G2 depends on G4, G3 depends on G4, G1 depends on both G2 and G3.
    All matched → all survive.
    Result: {G1, G2, G3, G4}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3, g4],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
            GuidelineMatch(guideline=g4, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id, g3.id, g4.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])
    assert_resolutions(result, g4.id, [ResolutionKind.NONE])


async def test_that_any_semantics_survives_cascading_failure_within_tag_member(
    container: Container,
) -> None:
    """
    G1 depends on T1 (ANY). T1 has G2 and G3.
    G2 depends on G4, G4 NOT matched → G2 is filtered.
    But G3 is still matched → G1 survives (ANY: G3 is active under T1).
    Result: {G1, G3}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t1.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t1.id])
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action")

    # G1 depends on T1 (ANY)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G2 depends on G4
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3, g4],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
            # G4 NOT matched → G2 will be filtered
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g3.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_any_semantics_filters_when_all_tag_members_cascade_fail(
    container: Container,
) -> None:
    """
    G1 depends on T1 (ANY). T1 has G2 and G3.
    G2 depends on G4, G3 depends on G5. Neither G4 nor G5 matched.
    → G2 filtered, G3 filtered → T1 has no active members → G1 filtered.
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t1.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t1.id])
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action")
    g5 = await guideline_store.create_guideline(condition="e", action="g5 action")

    # G1 depends on T1 (ANY)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G2 depends on G4
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G3 depends on G5
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g5.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3, g4, g5],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
            # G4 NOT matched
            # G5 NOT matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == set()

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g3.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_empty_tag_dependency_filters_dependent_guideline(
    container: Container,
) -> None:
    """
    G1 depends on T1. T1 has no tagged members at all.
    any([]) is False → G1 should be filtered.
    Result: {g_extra}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="empty-tag")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g_extra = await guideline_store.create_guideline(condition="b", action="extra action")

    # G1 depends on T1 (which has no members)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g_extra],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g_extra, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g_extra.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g_extra.id, [ResolutionKind.NONE])


async def test_that_multiple_independent_dependencies_must_all_be_met(
    container: Container,
) -> None:
    """
    G1 depends on G2 (direct) AND on T1 (tag).
    G2 matched, T1 member NOT matched → G1 filtered (both must be met).
    Result: {G2}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t1.id])

    # G1 depends on G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G1 depends on T1
    rel_t1 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            # g3 NOT matched (T1 unmet)
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g2.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])

    # The single resolution should specifically identify T1 as the unmet target
    # and reference the dependency relationship that caused the failure.
    g1_res = get_resolutions_by_kind(result, g1.id, ResolutionKind.UNMET_DEPENDENCY_ALL)
    assert len(g1_res) == 1
    assert ResolvedEntity.tag(t1) in g1_res[0].details.counterparts
    assert g1_res[0].details.relationship is not None
    assert g1_res[0].details.relationship.id == rel_t1.id


async def test_that_multiple_independent_dependencies_survive_when_all_met(
    container: Container,
) -> None:
    """
    G1 depends on G2 (direct) AND on T1 (tag).
    G2 matched, T1 member matched → G1 survives (both met).
    Result: {G1, G2, G3}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t1.id])

    # G1 depends on G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G1 depends on T1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id, g3.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_multi_iteration_convergence_filters_dependent_when_transitive_dependency_fails(
    container: Container,
) -> None:
    """
    G4 depends on G2 and G3 (individually). G3 depends on G1.
    G2, G3, G4 all matched. G1 NOT matched.

    Iteration 1: G3 is filtered (depends on G1). G4 survives because G3 is
    still in the initial matched_guideline_ids snapshot.
    Iteration 2: G4 is filtered (G3 no longer in matches).

    Result: {G2}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action")

    # G4 depends on G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G4 depends on G3
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G3 depends on G1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3, g4],
        [
            # G1 NOT matched
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
            GuidelineMatch(guideline=g4, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g2.id}

    # Resolution assertions
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    # G4 depends on G2 (met) and G3 (filtered), plus transitive dep on G1
    assert_resolutions(
        result,
        g4.id,
        [ResolutionKind.UNMET_DEPENDENCY_ALL],
    )


# ── Numerical priority + dependency interaction tests ──────────────────────


async def test_that_numerical_priority_filtering_removes_dependent_when_dependency_is_lower_priority(
    container: Container,
) -> None:
    """
    G1 (priority 100) depends on G2 (priority 0). G3 (priority 100) is independent.
    All matched. Final priority filter keeps only priority 100 → G2 removed.
    G1's dependency on G2 is broken → G1 should also be removed.
    Result: {G3}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", priority=100)
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", priority=0)
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", priority=100)

    # G1 depends on G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g3.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g2.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])

    # G2's numerical-priority deprioritization should list the entities at
    # the max priority (G1 and G3) as counterparts. Numerical priority is
    # not a relationship, so relationship stays None.
    g2_res = get_resolutions_by_kind(result, g2.id, ResolutionKind.DEPRIORITIZED)
    assert len(g2_res) == 1
    assert set(g2_res[0].details.counterparts) == {
        ResolvedEntity.guideline(g1),
        ResolvedEntity.guideline(g3),
    }
    assert g2_res[0].details.relationship is None


async def test_that_numerical_priority_filtering_keeps_dependent_when_dependency_shares_highest_priority(
    container: Container,
) -> None:
    """
    G1 (priority 100) depends on G2 (priority 100). Both at max priority.
    Final priority filter keeps both → G1's dependency met.
    Result: {G1, G2}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", priority=100)
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", priority=100)

    # G1 depends on G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_numerical_priority_filtering_cascades_through_tag_dependency(
    container: Container,
) -> None:
    """
    G1 (priority 100) depends on T1 (ANY). T1 has G2 (priority 0).
    G3 (priority 100) is independent. All matched.
    Final priority filter removes G2 (priority 0) → T1 has no surviving members → G1 filtered.
    Result: {G3}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", priority=100)
    g2 = await guideline_store.create_guideline(
        condition="b", action="g2 action", priority=0, tags=[t1.id]
    )
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", priority=100)

    # G1 depends on T1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g3.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g2.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_numerical_priority_filtering_with_tag_any_keeps_dependent_when_one_member_survives(
    container: Container,
) -> None:
    """
    G1 (priority 100) depends on T1 (ANY). T1 has G2 (priority 0) and G3 (priority 100).
    Final priority filter removes G2, keeps G3 → ANY met → G1 survives.
    Result: {G1, G3}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", priority=100)
    g2 = await guideline_store.create_guideline(
        condition="b", action="g2 action", priority=0, tags=[t1.id]
    )
    g3 = await guideline_store.create_guideline(
        condition="c", action="g3 action", priority=100, tags=[t1.id]
    )

    # G1 depends on T1
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g3.id}

    # Resolution assertions
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


# ── TAG_ALL vs TAG_ANY explicit tests ──────────────────────────────────────


async def test_that_tag_all_dependency_filters_when_not_all_members_matched(
    container: Container,
) -> None:
    """
    G1 depends on TAG_ALL(t1). t1 has G2 and G3. Only G2 matched.
    TAG_ALL requires all members → G1 filtered.
    Result: {G2}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t1.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t1.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            # G3 NOT matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g2.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_tag_all_dependency_survives_when_all_members_matched(
    container: Container,
) -> None:
    """
    G1 depends on TAG_ALL(t1). t1 has G2 and G3. Both matched.
    TAG_ALL: all members active → G1 survives.
    Result: {G1, G2, G3}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t1.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t1.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id, g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_tag_any_dependency_survives_when_one_of_two_members_matched(
    container: Container,
) -> None:
    """
    G1 depends on TAG_ANY(t1). t1 has G2 and G3. Only G2 matched.
    TAG_ANY: at least one member → G1 survives.
    Result: {G1, G2}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t1.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t1.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            # G3 NOT matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_tag_any_dependency_filters_when_no_members_matched(
    container: Container,
) -> None:
    """
    G1 depends on TAG_ANY(t1). t1 has G2 and G3. Neither matched.
    TAG_ANY: no members → G1 filtered.
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t1.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t1.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # G2 NOT matched
            # G3 NOT matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == set()

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_mixed_tag_all_and_tag_any_dependencies_both_evaluated(
    container: Container,
) -> None:
    """
    G1 depends on TAG_ALL(t1) AND TAG_ANY(t2).
    t1 has G2, G3 (both matched). t2 has G4, G5 (only G4 matched).
    TAG_ALL(t1) met, TAG_ANY(t2) met → G1 survives.
    Result: {G1, G2, G3, G4}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t1.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t1.id])
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action", tags=[t2.id])
    g5 = await guideline_store.create_guideline(condition="e", action="g5 action", tags=[t2.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3, g4, g5],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
            GuidelineMatch(guideline=g4, score=10, rationale=""),
            # G5 NOT matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id, g3.id, g4.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])
    assert_resolutions(result, g4.id, [ResolutionKind.NONE])


async def test_that_mixed_tag_all_and_tag_any_filters_when_tag_all_not_fully_met(
    container: Container,
) -> None:
    """
    G1 depends on TAG_ALL(t1) AND TAG_ANY(t2).
    t1 has G2, G3 (only G2 matched). t2 has G4, G5 (G4 matched).
    TAG_ALL(t1) NOT met (G3 missing) → G1 filtered, even though TAG_ANY(t2) is met.
    Result: {G2, G4}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")
    t2 = await tag_store.create_tag(name="t2")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t1.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t1.id])
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action", tags=[t2.id])
    g5 = await guideline_store.create_guideline(condition="e", action="g5 action", tags=[t2.id])

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t2.id, kind=RelationshipEntityKind.TAG_ANY),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3, g4, g5],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            # G3 NOT matched
            GuidelineMatch(guideline=g4, score=10, rationale=""),
            # G5 NOT matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g2.id, g4.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g4.id, [ResolutionKind.NONE])


# ── DEPENDENCY_ANY (OR group) tests ────────────────────────────────────────


async def test_that_dependency_any_group_survives_when_one_target_is_matched(
    container: Container,
) -> None:
    """
    G1 depends_on_any(G2, G3). G2 matched, G3 NOT matched.
    OR group: any one met → G1 survives.
    Result: {G1, G2}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")

    group_id = "test-group-1"
    for target in [g2, g3]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GUIDELINE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            # G3 NOT matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_dependency_any_group_filters_when_no_target_is_matched(
    container: Container,
) -> None:
    """
    G1 depends_on_any(G2, G3). Neither matched.
    OR group: none met → G1 filtered.
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")

    group_id = "test-group-1"
    group_rel_ids = []
    for target in [g2, g3]:
        rel = await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GUIDELINE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )
        group_rel_ids.append(rel.id)

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # G2 NOT matched
            # G3 NOT matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == set()

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ANY])

    # The resolution should list both OR-group targets as counterparts and
    # reference one of the dependency relationships that contributed to the
    # failure.
    g1_res = get_resolutions_by_kind(result, g1.id, ResolutionKind.UNMET_DEPENDENCY_ANY)
    assert len(g1_res) == 1
    assert set(g1_res[0].details.counterparts) == {
        ResolvedEntity.guideline(g2),
        ResolvedEntity.guideline(g3),
    }
    assert g1_res[0].details.relationship is not None
    assert g1_res[0].details.relationship.id in group_rel_ids


async def test_that_mixed_dependency_all_and_dependency_any_requires_both(
    container: Container,
) -> None:
    """
    G1 depend_on(G2) AND depend_on_any(G3, G4).
    G2 matched, G3 NOT matched, G4 matched.
    AND: G2 met. OR group: G4 met. Both met → G1 survives.
    Result: {G1, G2, G4}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action")

    # G1 depends on G2 (AND)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G1 depends_on_any(G3, G4) (OR)
    group_id = "test-group-1"
    for target in [g3, g4]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GUIDELINE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )

    result = await resolver.resolve(
        [g1, g2, g3, g4],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            # G3 NOT matched
            GuidelineMatch(guideline=g4, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id, g4.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g4.id, [ResolutionKind.NONE])


async def test_that_mixed_dependency_all_and_dependency_any_filters_when_and_dep_unmet(
    container: Container,
) -> None:
    """
    G1 depend_on(G2) AND depend_on_any(G3, G4).
    G2 NOT matched. G4 matched.
    AND dep unmet → G1 filtered even though OR group is met.
    Result: {G4}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    group_id = "test-group-1"
    for target in [g3, g4]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GUIDELINE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )

    result = await resolver.resolve(
        [g1, g2, g3, g4],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # G2 NOT matched
            # G3 NOT matched
            GuidelineMatch(guideline=g4, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g4.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g4.id, [ResolutionKind.NONE])


async def test_that_two_dependency_any_groups_are_anded_together(
    container: Container,
) -> None:
    """
    G1 depend_on_any(G2, G3) AND depend_on_any(G4, G5).
    G2 matched (group A met). G4 NOT, G5 NOT (group B unmet).
    Group A met but group B unmet → G1 filtered.
    Result: {G2}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action")
    g5 = await guideline_store.create_guideline(condition="e", action="g5 action")

    # Group A: G2 or G3
    for target in [g2, g3]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GUIDELINE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id="group-a",
        )

    # Group B: G4 or G5
    for target in [g4, g5]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GUIDELINE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id="group-b",
        )

    result = await resolver.resolve(
        [g1, g2, g3, g4, g5],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            # G3, G4, G5 NOT matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g2.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ANY])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


# ── DEPENDENCY_ANY edge case tests ─────────────────────────────────────────


async def test_that_dependency_any_group_with_tag_all_target_falls_back_to_guideline_target(
    container: Container,
) -> None:
    """
    G1 depend_on_any(AllOf(tag=T1), G4).
    T1 has G2 and G3. Only G2 matched → AllOf(T1) fails.
    G4 matched → OR group passes via G4.
    Result: {G1, G2, G4}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t1.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t1.id])
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action")

    group_id = "test-group-tag"

    # AllOf(T1) target
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id=group_id,
    )

    # G4 target
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id=group_id,
    )

    result = await resolver.resolve(
        [g1, g2, g3, g4],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            # G3 NOT matched → AllOf(T1) fails
            GuidelineMatch(guideline=g4, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id, g4.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g4.id, [ResolutionKind.NONE])


async def test_that_dependency_any_group_with_journey_targets_survives_when_one_journey_is_active(
    container: Container,
) -> None:
    """
    G1 depend_on_any(J1, J2). J1 active, J2 NOT active.
    OR group: J1 met → G1 survives.
    Result: {G1, J1_g}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")

    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])
    j1_g = await guideline_store.create_guideline(
        condition="b",
        action="j1 action",
        metadata={"journey_node": {"journey_id": j1.id}},
    )

    j2 = await journey_store.create_journey(title="J2", description="Journey 2", triggers=[])
    j2_g = await guideline_store.create_guideline(
        condition="c",
        action="j2 action",
        metadata={"journey_node": {"journey_id": j2.id}},
    )

    group_id = "test-journey-group"

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id=group_id,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j2.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id=group_id,
    )

    result = await resolver.resolve(
        [g1, j1_g, j2_g],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=j1_g, score=10, rationale=""),
        ],
        journeys=[j1],  # Only J1 active
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, j1_g.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, j1_g.id, [ResolutionKind.NONE])


async def test_that_dependency_any_group_survives_when_one_target_cascading_fails_but_another_survives(
    container: Container,
) -> None:
    """
    G1 depend_on_any(G2, G3). G2 depends on G4. G4 NOT matched.
    G2 filtered by its own dep. G3 matched → OR group passes via G3.
    Result: {G1, G3}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action")

    # G1 depend_on_any(G2, G3)
    group_id = "test-cascade-group"
    for target in [g2, g3]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GUIDELINE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )

    # G2 depends on G4 (AND)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3, g4],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
            # G4 NOT matched → G2 will be filtered
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_dependency_any_group_survives_when_priority_removes_one_target_but_another_survives(
    container: Container,
) -> None:
    """
    G1 depend_on_any(G2, G3). G2 priority 0, G3 priority 100, G4 priority 100.
    Priority filter removes G2. OR group: G3 still active → G1 survives.
    Result: {G1, G3, G4}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", priority=100)
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", priority=0)
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", priority=100)
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action", priority=100)

    group_id = "test-priority-group"
    for target in [g2, g3]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GUIDELINE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )

    result = await resolver.resolve(
        [g1, g2, g3, g4],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
            GuidelineMatch(guideline=g4, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g3.id, g4.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])
    assert_resolutions(result, g4.id, [ResolutionKind.NONE])


async def test_that_tag_all_dependency_cascades_when_one_member_own_dependency_fails(
    container: Container,
) -> None:
    """
    G1 depends on TAG_ALL(t1). t1 has G2 and G3.
    G2 depends on G4. G4 NOT matched → G2 filtered.
    TAG_ALL requires all members — G2 gone → G1 filtered.
    Result: {G3}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t1.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t1.id])
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action")

    # G1 depends on TAG_ALL(t1)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G2 depends on G4
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3, g4],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
            # G4 NOT matched → G2 will be filtered
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g2.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_single_target_dependency_any_group_filters_when_target_not_matched(
    container: Container,
) -> None:
    """
    G1 depend_on_any(G2). G2 NOT matched.
    Degenerate OR group with one member → G1 filtered (same as depend_on).
    Result: {}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id="single-group",
    )

    result = await resolver.resolve(
        [g1, g2],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # G2 NOT matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == set()

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ANY])


async def test_that_shared_target_in_dependency_all_and_dependency_any_survives_when_shared_target_matched(
    container: Container,
) -> None:
    """
    G1 depend_on(G2) AND depend_on_any(G2, G3).
    G2 matched, G3 NOT matched.
    AND: G2 met. OR group: G2 met. Both satisfied → G1 survives.
    Result: {G1, G2}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")

    # G1 depends on G2 (AND)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G1 depend_on_any(G2, G3)
    group_id = "shared-group"
    for target in [g2, g3]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GUIDELINE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            # G3 NOT matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_shared_target_in_dependency_all_and_dependency_any_filters_when_shared_target_not_matched(
    container: Container,
) -> None:
    """
    G1 depend_on(G2) AND depend_on_any(G2, G3).
    G2 NOT matched, G3 matched.
    AND: G2 unmet → G1 filtered, even though OR group passes via G3.
    Result: {G3}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")

    # G1 depends on G2 (AND)
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G1 depend_on_any(G2, G3)
    group_id = "shared-group"
    for target in [g2, g3]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GUIDELINE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # G2 NOT matched
            GuidelineMatch(guideline=g3, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_numerical_priority_does_not_filter_entailer_when_entailed_has_higher_priority(
    container: Container,
) -> None:
    """
    G1 (priority 0) entails G2 (priority 100). G1 matched, G2 not initially matched.
    Numerical priority runs before entailment: only G1 present, nothing filtered.
    Then entailment adds G2. Both survive.
    Result: {G1, G2}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action", priority=0)
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", priority=100)

    # G1 entails G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.ENTAILMENT,
    )

    result = await resolver.resolve(
        [g1, g2],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # G2 NOT initially matched — should be added via entailment
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.ENTAILED])


async def test_that_chained_entailment_activates_all_linked_guidelines(
    container: Container,
) -> None:
    """
    G1 entails G2, G2 entails G3. Only G1 matched.
    Entailment should activate G2 (from G1), then G3 (from G2).
    Result: {G1, G2, G3}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")

    # G1 entails G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.ENTAILMENT,
    )

    # G2 entails G3
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.ENTAILMENT,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # G2 and G3 NOT initially matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id, g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.ENTAILED])
    assert_resolutions(result, g3.id, [ResolutionKind.ENTAILED])


async def test_that_entailed_guideline_is_filtered_when_its_dependency_is_unmet(
    container: Container,
) -> None:
    """
    G1 entails G2. G2 depends on G3. G1 matched, G3 NOT matched.
    G2 is entailed but its dependency on G3 is unmet → G2 filtered.
    Result: {G1}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.ENTAILMENT,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # G2 not initially matched (entailed by G1)
            # G3 NOT matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(
        result, g2.id, [ResolutionKind.ENTAILED, ResolutionKind.UNMET_DEPENDENCY_ALL]
    )


async def test_that_entailed_guideline_survives_when_its_dependency_any_group_is_met(
    container: Container,
) -> None:
    """
    G1 entails G2. G2 depend_on_any(G3, G4). G1 and G3 matched.
    G2 entailed, OR group met via G3 → G2 survives.
    Result: {G1, G2, G3}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.ENTAILMENT,
    )

    group_id = "entailed-or-group"
    for target in [g3, g4]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GUIDELINE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )

    result = await resolver.resolve(
        [g1, g2, g3, g4],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # G2 not initially matched (entailed)
            GuidelineMatch(guideline=g3, score=10, rationale=""),
            # G4 NOT matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id, g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.ENTAILED])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


async def test_that_relational_priority_on_dependency_any_target_does_not_break_group_when_sibling_survives(
    container: Container,
) -> None:
    """
    G1 depend_on_any(G2, G3). G4 prioritizes over G2 (relational).
    All matched. G2 deprioritized → OR group still met via G3.
    Result: {G1, G3, G4}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action")

    group_id = "prio-or-group"
    for target in [g2, g3]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
            target=RelationshipEntity(id=target.id, kind=RelationshipEntityKind.GUIDELINE),
            kind=RelationshipKind.DEPENDENCY_ANY,
            group_id=group_id,
        )

    # G4 prioritizes over G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g1, g2, g3, g4],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
            GuidelineMatch(guideline=g4, score=10, rationale=""),
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g3.id, g4.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])
    assert_resolutions(result, g4.id, [ResolutionKind.NONE])


async def test_that_dependency_any_group_with_tag_all_target_succeeds_when_all_members_matched(
    container: Container,
) -> None:
    """
    G1 depend_on_any(AllOf(tag=T1), G4).
    T1 has G2 and G3, both matched → AllOf(T1) succeeds.
    OR group met via the tag target (G4 not needed).
    Result: {G1, G2, G3}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    tag_store = container[TagStore]
    resolver = container[RelationalResolver]

    t1 = await tag_store.create_tag(name="t1")

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action", tags=[t1.id])
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action", tags=[t1.id])
    g4 = await guideline_store.create_guideline(condition="d", action="g4 action")

    group_id = "tag-all-success-group"

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=t1.id, kind=RelationshipEntityKind.TAG_ALL),
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id=group_id,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY_ANY,
        group_id=group_id,
    )

    result = await resolver.resolve(
        [g1, g2, g3, g4],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=10, rationale=""),
            GuidelineMatch(guideline=g3, score=10, rationale=""),
            # G4 NOT matched — but AllOf(T1) succeeds so OR group is met
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id, g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])


# ── Resolution attribution edge-case tests ────────────────────────────────


async def test_that_priority_chain_attributes_to_direct_deprioritizer(
    container: Container,
) -> None:
    """
    A prioritizes over B, B prioritizes over C. All matched.
    B should be attributed to A (direct deprioritizer).
    C should be attributed to B (direct deprioritizer), NOT A (transitive).
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g_a = await guideline_store.create_guideline(condition="a", action="action A")
    g_b = await guideline_store.create_guideline(condition="b", action="action B")
    g_c = await guideline_store.create_guideline(condition="c", action="action C")

    # A prio over B
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g_a.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g_b.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.PRIORITY,
    )

    # B prio over C
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g_b.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g_c.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g_a, g_b, g_c],
        [
            GuidelineMatch(guideline=g_a, score=10, rationale=""),
            GuidelineMatch(guideline=g_b, score=8, rationale=""),
            GuidelineMatch(guideline=g_c, score=6, rationale=""),
        ],
        journeys=[],
    )

    assert result.matches == [GuidelineMatch(guideline=g_a, score=10, rationale="")]

    assert_resolutions(result, g_a.id, [ResolutionKind.NONE])
    assert_resolutions(result, g_b.id, [ResolutionKind.DEPRIORITIZED])
    assert_resolutions(result, g_c.id, [ResolutionKind.DEPRIORITIZED])

    # B deprioritized by A (direct)
    b_res = get_resolutions_by_kind(result, g_b.id, ResolutionKind.DEPRIORITIZED)
    assert len(b_res) == 1
    assert ResolvedEntity.guideline(g_a) in b_res[0].details.counterparts

    # C deprioritized by B (direct), NOT by A (transitive)
    c_res = get_resolutions_by_kind(result, g_c.id, ResolutionKind.DEPRIORITIZED)
    assert len(c_res) == 1
    assert ResolvedEntity.guideline(g_b) in c_res[0].details.counterparts


async def test_that_transitive_deprioritized_dependency_records_only_direct_resolution(
    container: Container,
) -> None:
    """
    A depends on B, B depends on C. X deprioritizes C.
    Expected: C deprioritized by X. B gets DEPRIORITIZED (dep on deprioritized C).
    A gets UNMET_DEPENDENCY_ALL (dep on B, which was removed).
    Each entity has exactly ONE resolution — no transitive duplicates.
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g_a = await guideline_store.create_guideline(condition="a", action="action A")
    g_b = await guideline_store.create_guideline(condition="b", action="action B")
    g_c = await guideline_store.create_guideline(condition="c", action="action C")
    g_x = await guideline_store.create_guideline(condition="x", action="action X")

    # A depends on B
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g_a.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g_b.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # B depends on C
    rel_b_c = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g_b.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g_c.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # X prioritizes over C
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g_x.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g_c.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g_a, g_b, g_c, g_x],
        [
            GuidelineMatch(guideline=g_a, score=10, rationale=""),
            GuidelineMatch(guideline=g_b, score=8, rationale=""),
            GuidelineMatch(guideline=g_c, score=6, rationale=""),
            GuidelineMatch(guideline=g_x, score=9, rationale=""),
        ],
        journeys=[],
    )

    # Only X should survive
    assert len(result.matches) == 1
    assert result.matches[0].guideline.id == g_x.id

    assert_resolutions(result, g_x.id, [ResolutionKind.NONE])
    # C deprioritized by X (relational priority)
    assert_resolutions(result, g_c.id, [ResolutionKind.DEPRIORITIZED])
    # B removed because its dependency (C) was deprioritized — single resolution
    assert_resolutions(result, g_b.id, [ResolutionKind.DEPRIORITIZED])
    # B's transitive deprioritization should identify C as the counterpart and
    # reference the B→C DEPENDENCY relationship that caused the cascade.
    b_res = get_resolutions_by_kind(result, g_b.id, ResolutionKind.DEPRIORITIZED)
    assert len(b_res) == 1
    assert ResolvedEntity.guideline(g_c) in b_res[0].details.counterparts
    assert b_res[0].details.relationship is not None
    assert b_res[0].details.relationship.id == rel_b_c.id
    # A removed because its dependency (B) is gone — single resolution
    assert_resolutions(result, g_a.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_journey_tag_guideline_journey_tag_dependency_cascades(
    container: Container,
) -> None:
    """
    Cross-entity cascade: J1 node depends on G, G depends on J2 tag, J2 not active.
    Expected: G filtered (dep on J2 unmet), J1 node filtered (dep on G unmet).
    Each gets exactly one resolution.
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    journey_store = container[JourneyStore]
    resolver = container[RelationalResolver]

    # Create J1 (active) and its node guideline
    j1 = await journey_store.create_journey(title="J1", description="Journey 1", triggers=[])
    j1_g = await guideline_store.create_guideline(
        condition="a",
        action="j1 action",
        metadata={"journey_node": {"journey_id": j1.id}},
    )

    # Create standalone guideline G
    g = await guideline_store.create_guideline(condition="b", action="bridge action")

    # Create J2 (NOT active — won't be in journeys list)
    j2 = await journey_store.create_journey(title="J2", description="Journey 2", triggers=[])

    # J1 tag depends on G
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=Tag.for_journey_id(j1.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        target=RelationshipEntity(id=g.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    # G depends on J2 tag
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(
            id=Tag.for_journey_id(j2.id).id, kind=RelationshipEntityKind.TAG_ALL
        ),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [j1_g, g],
        [
            GuidelineMatch(guideline=j1_g, score=10, rationale=""),
            GuidelineMatch(guideline=g, score=8, rationale=""),
        ],
        journeys=[j1],  # J2 NOT active
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == set()

    # G: dep on J2 tag unmet (J2 not active) — single resolution
    assert_resolutions(result, g.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])
    # J1 node: dep on G unmet (G was filtered) — single resolution
    assert_resolutions(result, j1_g.id, [ResolutionKind.UNMET_DEPENDENCY_ALL])


async def test_that_priority_chain_with_gaps_does_not_transitively_deprioritize(
    container: Container,
) -> None:
    """
    G1 → G2 → G3 → G4 (priority chain).
    Only G2 and G4 are matched.
    G2's direct deprioritizer (G1) is not matched → G2 survives.
    G4's direct deprioritizer (G3) is not matched → what happens to G4?
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="action 1")
    g2 = await guideline_store.create_guideline(condition="b", action="action 2")
    g3 = await guideline_store.create_guideline(condition="c", action="action 3")
    g4 = await guideline_store.create_guideline(condition="d", action="action 4")

    # G1 prio over G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.PRIORITY,
    )

    # G2 prio over G3
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.PRIORITY,
    )

    # G3 prio over G4
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g4.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g1, g2, g3, g4],
        [
            # Only G2 and G4 matched
            GuidelineMatch(guideline=g2, score=8, rationale=""),
            GuidelineMatch(guideline=g4, score=6, rationale=""),
        ],
        journeys=[],
    )

    # Both survive: G2's deprioritizer (G1) is not matched, and G4's
    # deprioritizer (G3) is not matched.  Priority does NOT propagate
    # through inactive intermediaries (reinstatement principle).
    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g2.id, g4.id}

    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g4.id, [ResolutionKind.NONE])


async def test_that_already_matched_entailment_target_gets_none_not_entailed(
    container: Container,
) -> None:
    """
    G1 entails G2. Both G1 and G2 are already matched.
    G2 should get NONE because it was already active — entailment
    was not needed to bring it in.
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")

    # G1 entails G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.ENTAILMENT,
    )

    result = await resolver.resolve(
        [g1, g2],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=8, rationale=""),  # Already matched
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id}

    # Both get NONE — G2 was already active, entailment was unnecessary
    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])


async def test_that_guideline_entailed_by_two_sources_records_both_entailment_resolutions(
    container: Container,
) -> None:
    """
    G1 entails G3, G2 entails G3. G1 and G2 are matched, G3 is not.
    G3 should have TWO ENTAILED resolutions — one for each source.
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")

    # G1 entails G3
    r1 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.ENTAILMENT,
    )

    # G2 entails G3
    r2 = await relationship_store.create_relationship(
        source=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.ENTAILMENT,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            GuidelineMatch(guideline=g2, score=8, rationale=""),
            # G3 NOT matched — activated via entailment from both G1 and G2
        ],
        journeys=[],
    )

    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id, g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.NONE])
    assert_resolutions(result, g3.id, [ResolutionKind.ENTAILED, ResolutionKind.ENTAILED])

    # Both entailment relationships should be referenced
    g3_res = get_resolutions_by_kind(result, g3.id, ResolutionKind.ENTAILED)
    assert len(g3_res) == 2
    for r in g3_res:
        assert r.details.relationship is not None
    rel_ids = {r.details.relationship.id for r in g3_res if r.details.relationship is not None}
    assert rel_ids == {r1.id, r2.id}

    # Each ENTAILED resolution should also identify the entailing source
    # guideline as a counterpart, paired with the matching relationship.
    rel_to_counterparts = {
        r.details.relationship.id: r.details.counterparts
        for r in g3_res
        if r.details.relationship is not None
    }
    assert rel_to_counterparts[r1.id] == (ResolvedEntity.guideline(g1),)
    assert rel_to_counterparts[r2.id] == (ResolvedEntity.guideline(g2),)


async def test_that_entailed_guideline_satisfies_dependency_of_matched_guideline(
    container: Container,
) -> None:
    """
    G3 depends on G2. G1 entails G2.
    G1 and G3 are matched, G2 is not.
    Iteration 1: G3's dep on G2 fails (G2 not yet in matches).
                 Entailment adds G2.
    Iteration 2: G2 is now in matches, G3's dep on G2 is satisfied.
    Result: {G1, G2, G3}
    """
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalResolver]

    g1 = await guideline_store.create_guideline(condition="a", action="g1 action")
    g2 = await guideline_store.create_guideline(condition="b", action="g2 action")
    g3 = await guideline_store.create_guideline(condition="c", action="g3 action")

    # G1 entails G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g1.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.ENTAILMENT,
    )

    # G3 depends on G2
    await relationship_store.create_relationship(
        source=RelationshipEntity(id=g3.id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=g2.id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=10, rationale=""),
            # G2 NOT matched — will be entailed by G1
            GuidelineMatch(guideline=g3, score=8, rationale=""),
        ],
        journeys=[],
    )

    # All three should survive:
    # - Iteration 1: G3's dep on G2 fails (G2 not yet matched).
    #   Entailment adds G2. Match set changed → iterate.
    # - Iteration 2: G2 is now available. G3's dep on G2 is satisfied.
    result_ids = {m.guideline.id for m in result.matches}
    assert result_ids == {g1.id, g2.id, g3.id}

    assert_resolutions(result, g1.id, [ResolutionKind.NONE])
    assert_resolutions(result, g2.id, [ResolutionKind.ENTAILED])
    assert_resolutions(result, g3.id, [ResolutionKind.NONE])
