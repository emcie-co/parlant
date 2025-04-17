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

from parlant.core.engines.alpha.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.relational_guideline_resolver import RelationalGuidelineResolver
from parlant.core.relationships import EntityType, GuidelineRelationshipKind, RelationshipStore
from parlant.core.guidelines import GuidelineStore
from parlant.core.tags import TagStore


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
        source=g1.id,
        source_type=EntityType.GUIDELINE,
        target=g2.id,
        target_type=EntityType.GUIDELINE,
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    await relationship_store.create_relationship(
        source=g2.id,
        source_type=EntityType.GUIDELINE,
        target=g3.id,
        target_type=EntityType.GUIDELINE,
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=8, rationale=""),
            GuidelineMatch(guideline=g2, score=5, rationale=""),
            GuidelineMatch(guideline=g3, score=9, rationale=""),
        ],
    )

    assert result == [GuidelineMatch(guideline=g1, score=8, rationale="")]


async def test_that_relational_guideline_resolver_prioritizes_guidelines(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalGuidelineResolver]

    g1 = await guideline_store.create_guideline(condition="x", action="y")
    g2 = await guideline_store.create_guideline(condition="y", action="z")

    matches: list[GuidelineMatch] = [
        GuidelineMatch(guideline=g1, score=8, rationale=""),
        GuidelineMatch(guideline=g2, score=5, rationale=""),
    ]

    await relationship_store.create_relationship(
        source=g1.id,
        source_type=EntityType.GUIDELINE,
        target=g2.id,
        target_type=EntityType.GUIDELINE,
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    result = await resolver.resolve([g1, g2], matches)

    assert result == [GuidelineMatch(guideline=g1, score=8, rationale="")]


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

    await guideline_store.upsert_tag(g2.id, t1.id)
    await guideline_store.upsert_tag(g3.id, t1.id)

    await relationship_store.create_relationship(
        source=g1.id,
        source_type=EntityType.GUIDELINE,
        target=t1.id,
        target_type=EntityType.TAG,
        kind=GuidelineRelationshipKind.ENTAILMENT,
    )

    await relationship_store.create_relationship(
        source=t1.id,
        source_type=EntityType.TAG,
        target=g4.id,
        target_type=EntityType.GUIDELINE,
        kind=GuidelineRelationshipKind.ENTAILMENT,
    )

    result = await resolver.resolve(
        [g1, g2, g3, g4],
        [
            GuidelineMatch(guideline=g1, score=8, rationale=""),
        ],
    )

    assert len(result) == 4
    assert any(m.guideline.id == g1.id for m in result)
    assert any(m.guideline.id == g2.id for m in result)
    assert any(m.guideline.id == g3.id for m in result)
    assert any(m.guideline.id == g4.id for m in result)


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
        source=g1.id,
        source_type=EntityType.GUIDELINE,
        target=t1.id,
        target_type=EntityType.TAG,
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    await relationship_store.create_relationship(
        source=t1.id,
        source_type=EntityType.TAG,
        target=g2.id,
        target_type=EntityType.GUIDELINE,
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g1, g2],
        [
            GuidelineMatch(guideline=g1, score=8, rationale=""),
            GuidelineMatch(guideline=g2, score=5, rationale=""),
        ],
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
        source=g1.id,
        source_type=EntityType.GUIDELINE,
        target=t1.id,
        target_type=EntityType.TAG,
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    await relationship_store.create_relationship(
        source=t1.id,
        source_type=EntityType.TAG,
        target=g3.id,
        target_type=EntityType.GUIDELINE,
        kind=GuidelineRelationshipKind.PRIORITY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=8, rationale=""),
            GuidelineMatch(guideline=g3, score=9, rationale=""),
        ],
    )

    assert len(result) == 1
    assert result[0].guideline.id == g1.id


async def test_that_relational_guideline_resolver_filters_out_unmet_dependencies_between_matched_guidelines(
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
        source=source_guideline.id,
        source_type=EntityType.GUIDELINE,
        target=target_guideline.id,
        target_type=EntityType.GUIDELINE,
        kind=GuidelineRelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [source_guideline, target_guideline],
        [
            GuidelineMatch(guideline=source_guideline, score=8, rationale=""),
        ],
    )

    assert result == []


async def test_that_relational_guideline_resolver_filters_out_unmet_dependencies(
    container: Container,
) -> None:
    relationship_store = container[RelationshipStore]
    guideline_store = container[GuidelineStore]
    resolver = container[RelationalGuidelineResolver]

    g1 = await guideline_store.create_guideline(condition="x", action="y")
    g2 = await guideline_store.create_guideline(condition="y", action="z")

    await relationship_store.create_relationship(
        source=g1.id,
        source_type=EntityType.GUIDELINE,
        target=g2.id,
        target_type=EntityType.GUIDELINE,
        kind=GuidelineRelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2],
        [
            GuidelineMatch(guideline=g1, score=8, rationale=""),
        ],
    )

    assert result == []


async def test_that_relational_guideline_resolver_filters_out_unmet_dependencies_connected_through_tag(
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

    await guideline_store.upsert_tag(g1.id, t1.id)

    await relationship_store.create_relationship(
        source=g2.id,
        source_type=EntityType.GUIDELINE,
        target=t1.id,
        target_type=EntityType.TAG,
        kind=GuidelineRelationshipKind.DEPENDENCY,
    )

    await relationship_store.create_relationship(
        source=t1.id,
        source_type=EntityType.TAG,
        target=g3.id,
        target_type=EntityType.GUIDELINE,
        kind=GuidelineRelationshipKind.DEPENDENCY,
    )

    result = await resolver.resolve(
        [g1, g2, g3],
        [
            GuidelineMatch(guideline=g1, score=8, rationale=""),
            GuidelineMatch(guideline=g2, score=5, rationale=""),
            GuidelineMatch(guideline=g3, score=9, rationale=""),
        ],
    )

    assert len(result) == 3
    assert any(m.guideline.id == g1.id for m in result)
    assert any(m.guideline.id == g2.id for m in result)
    assert any(m.guideline.id == g3.id for m in result)
