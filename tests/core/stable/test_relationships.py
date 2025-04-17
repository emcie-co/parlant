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

from typing import AsyncIterator, Sequence
from pytest import fixture, raises

from parlant.core.relationships import (
    EntityType,
    GuidelineRelationshipKind,
    Relationship,
    RelationshipDocumentStore,
    RelationshipEntity,
    RelationshipStore,
)
from parlant.core.guidelines import GuidelineId
from parlant.core.persistence.document_database import DocumentDatabase
from parlant.adapters.db.transient import TransientDocumentDatabase


@fixture
def underlying_database() -> DocumentDatabase:
    return TransientDocumentDatabase()


@fixture
async def relationship_store(
    underlying_database: DocumentDatabase,
) -> AsyncIterator[RelationshipStore]:
    async with RelationshipDocumentStore(database=underlying_database) as store:
        yield store


def has_relationship(
    guidelines: Sequence[Relationship],
    relationship: tuple[str, str],
) -> bool:
    return any(
        g.source.id == relationship[0] and g.target.id == relationship[1] for g in guidelines
    )


async def test_that_direct_guideline_relationships_can_be_listed(
    relationship_store: RelationshipStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")
    d_id = GuidelineId("d")
    z_id = GuidelineId("z")

    for source, target in [
        (a_id, b_id),
        (a_id, c_id),
        (b_id, d_id),
        (z_id, b_id),
    ]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(
                id=source,
                type=EntityType.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=target,
                type=EntityType.GUIDELINE,
            ),
            kind=GuidelineRelationshipKind.ENTAILMENT,
        )

    a_relationships = await relationship_store.list_relationships(
        kind=GuidelineRelationshipKind.ENTAILMENT,
        indirect=False,
        source_id=a_id,
    )

    assert len(a_relationships) == 2
    assert has_relationship(a_relationships, (a_id, b_id))
    assert has_relationship(a_relationships, (a_id, c_id))


async def test_that_indirect_guideline_relationships_can_be_listed(
    relationship_store: RelationshipStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")
    d_id = GuidelineId("d")
    z_id = GuidelineId("z")

    for source, target in [(a_id, b_id), (a_id, c_id), (b_id, d_id), (z_id, b_id)]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(
                id=source,
                type=EntityType.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=target,
                type=EntityType.GUIDELINE,
            ),
            kind=GuidelineRelationshipKind.ENTAILMENT,
        )

    a_relationships = await relationship_store.list_relationships(
        kind=GuidelineRelationshipKind.ENTAILMENT,
        indirect=True,
        source_id=a_id,
    )

    assert len(a_relationships) == 3
    assert has_relationship(a_relationships, (a_id, b_id))
    assert has_relationship(a_relationships, (a_id, c_id))
    assert has_relationship(a_relationships, (b_id, d_id))


async def test_that_db_data_is_loaded_correctly(
    relationship_store: RelationshipStore,
    underlying_database: DocumentDatabase,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")
    d_id = GuidelineId("d")
    z_id = GuidelineId("z")

    for source, target in [(a_id, b_id), (a_id, c_id), (b_id, d_id), (z_id, b_id)]:
        await relationship_store.create_relationship(
            source=RelationshipEntity(
                id=source,
                type=EntityType.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=target,
                type=EntityType.GUIDELINE,
            ),
            kind=GuidelineRelationshipKind.ENTAILMENT,
        )

    async with RelationshipDocumentStore(underlying_database) as new_store_with_same_db:
        a_relationships = await new_store_with_same_db.list_relationships(
            kind=GuidelineRelationshipKind.ENTAILMENT,
            source_id=a_id,
            indirect=True,
        )

    assert len(a_relationships) == 3
    assert has_relationship(a_relationships, (a_id, b_id))
    assert has_relationship(a_relationships, (a_id, c_id))
    assert has_relationship(a_relationships, (b_id, d_id))


async def test_that_relationships_are_returned_for_source_without_indirect_relationships(
    relationship_store: RelationshipStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=a_id,
            type=EntityType.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=b_id,
            type=EntityType.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.ENTAILMENT,
    )
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=b_id,
            type=EntityType.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=c_id,
            type=EntityType.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.ENTAILMENT,
    )

    connections = await relationship_store.list_relationships(
        kind=GuidelineRelationshipKind.ENTAILMENT,
        indirect=False,
        source_id=a_id,
    )

    assert len(connections) == 1
    assert has_relationship(connections, (a_id, b_id))
    assert not has_relationship(connections, (b_id, c_id))


async def test_that_connections_are_returned_for_source_with_indirect_connections(
    relationship_store: RelationshipStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=a_id,
            type=EntityType.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=b_id,
            type=EntityType.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.ENTAILMENT,
    )
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=b_id,
            type=EntityType.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=c_id,
            type=EntityType.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.ENTAILMENT,
    )

    relationships = await relationship_store.list_relationships(
        kind=GuidelineRelationshipKind.ENTAILMENT,
        indirect=True,
        source_id=a_id,
    )

    assert len(relationships) == 2
    assert has_relationship(relationships, (a_id, b_id))
    assert has_relationship(relationships, (b_id, c_id))
    assert len(relationships) == len(set((c.source, c.target) for c in relationships))


async def test_that_relationships_are_returned_for_target_without_indirect_connections(
    relationship_store: RelationshipStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=a_id,
            type=EntityType.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=b_id,
            type=EntityType.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.ENTAILMENT,
    )
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=b_id,
            type=EntityType.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=c_id,
            type=EntityType.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.ENTAILMENT,
    )

    relationships = await relationship_store.list_relationships(
        kind=GuidelineRelationshipKind.ENTAILMENT,
        indirect=False,
        target_id=b_id,
    )

    assert len(relationships) == 1
    assert has_relationship(relationships, (a_id, b_id))
    assert not has_relationship(relationships, (b_id, c_id))


async def test_that_relationships_are_returned_for_target_with_indirect_connections(
    relationship_store: RelationshipStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")

    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=a_id,
            type=EntityType.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=b_id,
            type=EntityType.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.ENTAILMENT,
    )
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=b_id,
            type=EntityType.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=c_id,
            type=EntityType.GUIDELINE,
        ),
        kind=GuidelineRelationshipKind.ENTAILMENT,
    )

    relationships = await relationship_store.list_relationships(
        kind=GuidelineRelationshipKind.ENTAILMENT,
        indirect=True,
        target_id=c_id,
    )

    assert len(relationships) == 2
    assert has_relationship(relationships, (a_id, b_id))
    assert has_relationship(relationships, (b_id, c_id))
    assert len(relationships) == len(set((c.source, c.target) for c in relationships))


async def test_that_error_is_raised_when_neither_source_nor_target_is_provided(
    relationship_store: RelationshipStore,
) -> None:
    with raises(AssertionError):
        await relationship_store.list_relationships(
            kind=GuidelineRelationshipKind.ENTAILMENT,
            indirect=False,
            source_id=None,
            target_id=None,
        )


async def test_that_error_is_raised_when_both_source_and_target_are_provided(
    relationship_store: RelationshipStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")

    with raises(AssertionError):
        await relationship_store.list_relationships(
            kind=GuidelineRelationshipKind.ENTAILMENT,
            indirect=False,
            source_id=a_id,
            target_id=b_id,
        )
