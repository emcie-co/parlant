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

from typing import AsyncIterator, Sequence
from pytest import fixture

from parlant.core.common import IdGenerator
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipKind,
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
    async with RelationshipDocumentStore(IdGenerator(), database=underlying_database) as store:
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
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=target,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            kind=RelationshipKind.ENTAILMENT,
        )

    a_relationships = await relationship_store.list_relationships(
        kind=RelationshipKind.ENTAILMENT,
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
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=target,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            kind=RelationshipKind.ENTAILMENT,
        )

    a_relationships = await relationship_store.list_relationships(
        kind=RelationshipKind.ENTAILMENT,
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
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            target=RelationshipEntity(
                id=target,
                kind=RelationshipEntityKind.GUIDELINE,
            ),
            kind=RelationshipKind.ENTAILMENT,
        )

    async with RelationshipDocumentStore(
        IdGenerator(), underlying_database
    ) as new_store_with_same_db:
        a_relationships = await new_store_with_same_db.list_relationships(
            kind=RelationshipKind.ENTAILMENT,
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
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=b_id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=b_id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=c_id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )

    connections = await relationship_store.list_relationships(
        kind=RelationshipKind.ENTAILMENT,
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
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=b_id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=b_id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=c_id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )

    relationships = await relationship_store.list_relationships(
        kind=RelationshipKind.ENTAILMENT,
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
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=b_id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=b_id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=c_id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )

    relationships = await relationship_store.list_relationships(
        kind=RelationshipKind.ENTAILMENT,
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
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=b_id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )
    await relationship_store.create_relationship(
        source=RelationshipEntity(
            id=b_id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        target=RelationshipEntity(
            id=c_id,
            kind=RelationshipEntityKind.GUIDELINE,
        ),
        kind=RelationshipKind.ENTAILMENT,
    )

    relationships = await relationship_store.list_relationships(
        kind=RelationshipKind.ENTAILMENT,
        indirect=True,
        target_id=c_id,
    )

    assert len(relationships) == 2
    assert has_relationship(relationships, (a_id, b_id))
    assert has_relationship(relationships, (b_id, c_id))
    assert len(relationships) == len(set((c.source, c.target) for c in relationships))


async def test_that_all_relationships_can_be_listed(
    relationship_store: RelationshipStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")

    relationships_data = [
        (a_id, b_id, RelationshipKind.ENTAILMENT),
        (b_id, c_id, RelationshipKind.PRIORITY),
        (c_id, a_id, RelationshipKind.DEPENDENCY),
        (a_id, c_id, RelationshipKind.DISAMBIGUATION),
        (b_id, a_id, RelationshipKind.REEVALUATION),
    ]

    for source, target, kind in relationships_data:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=source, kind=RelationshipEntityKind.GUIDELINE),
            target=RelationshipEntity(id=target, kind=RelationshipEntityKind.GUIDELINE),
            kind=kind,
        )

    all_relationships = await relationship_store.list_relationships()

    assert len(all_relationships) == len(relationships_data)
    for source, target, _ in relationships_data:
        assert has_relationship(all_relationships, (source, target))


async def test_that_relationships_can_be_listed_by_kind_without_entity_filters(
    relationship_store: RelationshipStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=a_id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=b_id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.ENTAILMENT,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=b_id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=c_id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.PRIORITY,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=a_id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=c_id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.DISAMBIGUATION,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=a_id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=c_id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.REEVALUATION,
    )

    entailments = await relationship_store.list_relationships(
        kind=RelationshipKind.ENTAILMENT,
    )

    assert len(entailments) == 1
    assert has_relationship(entailments, (a_id, b_id))

    assert not has_relationship(entailments, (b_id, c_id))


async def test_that_relationships_can_be_listed_by_source_id_without_kind_filter(
    relationship_store: RelationshipStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=a_id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=b_id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.ENTAILMENT,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=a_id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=c_id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.PRIORITY,
    )

    relationships = await relationship_store.list_relationships(source_id=a_id, indirect=False)

    assert len(relationships) == 2
    assert has_relationship(relationships, (a_id, b_id))
    assert has_relationship(relationships, (a_id, c_id))


async def test_that_relationships_can_be_listed_by_target_id_without_kind_filter(
    relationship_store: RelationshipStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=a_id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=b_id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.ENTAILMENT,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=c_id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=b_id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.PRIORITY,
    )

    relationships = await relationship_store.list_relationships(target_id=b_id, indirect=False)

    assert len(relationships) == 2
    assert has_relationship(relationships, (a_id, b_id))
    assert has_relationship(relationships, (c_id, b_id))


async def test_that_relationships_can_be_listed_with_both_source_and_target_filters(
    relationship_store: RelationshipStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=a_id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=b_id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.ENTAILMENT,
    )

    await relationship_store.create_relationship(
        source=RelationshipEntity(id=c_id, kind=RelationshipEntityKind.GUIDELINE),
        target=RelationshipEntity(id=a_id, kind=RelationshipEntityKind.GUIDELINE),
        kind=RelationshipKind.PRIORITY,
    )

    relationships = await relationship_store.list_relationships(
        source_id=a_id,
        target_id=a_id,
        indirect=False,
    )

    unique_pairs = {(rel.source.id, rel.target.id) for rel in relationships}

    assert unique_pairs == {(a_id, b_id), (c_id, a_id)}
