# Copyright 2024 Emcie Co Ltd.
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

from typing import Sequence
from pytest import fixture, raises

from parlant.core.guideline_connections import (
    ConnectionKind,
    GuidelineConnection,
    GuidelineConnectionDocumentStore,
    GuidelineConnectionStore,
)
from parlant.core.guidelines import GuidelineId
from parlant.core.persistence.document_database import DocumentDatabase
from parlant.adapters.db.transient import TransientDocumentDatabase


@fixture
def underlying_database() -> DocumentDatabase:
    return TransientDocumentDatabase()


@fixture
def store(
    underlying_database: DocumentDatabase,
) -> GuidelineConnectionStore:
    return GuidelineConnectionDocumentStore(database=underlying_database)


def has_connection(
    guidelines: Sequence[GuidelineConnection],
    connection: tuple[str, str],
) -> bool:
    return any(g.source == connection[0] and g.target == connection[1] for g in guidelines)


async def test_that_direct_guideline_connections_can_be_listed(
    store: GuidelineConnectionStore,
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
        await store.create_connection(
            source=source,
            target=target,
            kind=ConnectionKind.ENTAILS,
        )

    a_connections = await store.list_connections(
        source=a_id,
        indirect=False,
    )

    assert len(a_connections) == 2
    assert has_connection(a_connections, (a_id, b_id))
    assert has_connection(a_connections, (a_id, c_id))


async def test_that_indirect_guideline_connections_can_be_listed(
    store: GuidelineConnectionStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")
    d_id = GuidelineId("d")
    z_id = GuidelineId("z")

    for source, target in [(a_id, b_id), (a_id, c_id), (b_id, d_id), (z_id, b_id)]:
        await store.create_connection(
            source=source,
            target=target,
            kind=ConnectionKind.ENTAILS,
        )

    a_connections = await store.list_connections(
        source=a_id,
        indirect=True,
    )

    assert len(a_connections) == 3
    assert has_connection(a_connections, (a_id, b_id))
    assert has_connection(a_connections, (a_id, c_id))
    assert has_connection(a_connections, (b_id, d_id))


async def test_that_db_data_is_loaded_correctly(
    store: GuidelineConnectionStore,
    underlying_database: DocumentDatabase,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")
    d_id = GuidelineId("d")
    z_id = GuidelineId("z")

    for source, target in [(a_id, b_id), (a_id, c_id), (b_id, d_id), (z_id, b_id)]:
        await store.create_connection(
            source=source,
            target=target,
            kind=ConnectionKind.ENTAILS,
        )

    new_store_with_same_db = GuidelineConnectionDocumentStore(underlying_database)

    a_connections = await new_store_with_same_db.list_connections(
        source=a_id,
        indirect=True,
    )

    assert len(a_connections) == 3
    assert has_connection(a_connections, (a_id, b_id))
    assert has_connection(a_connections, (a_id, c_id))
    assert has_connection(a_connections, (b_id, d_id))


async def test_that_connections_are_returned_for_source_without_indirect_connections(
    store: GuidelineConnectionStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")

    await store.create_connection(source=a_id, target=b_id, kind=ConnectionKind.ENTAILS)
    await store.create_connection(source=b_id, target=c_id, kind=ConnectionKind.ENTAILS)

    connections = await store.list_connections(
        source=a_id,
        indirect=False,
    )

    assert len(connections) == 1
    assert has_connection(connections, (a_id, b_id))
    assert not has_connection(connections, (b_id, c_id))


async def test_that_connections_are_returned_for_source_with_indirect_connections(
    store: GuidelineConnectionStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")

    await store.create_connection(source=a_id, target=b_id, kind=ConnectionKind.ENTAILS)
    await store.create_connection(source=b_id, target=c_id, kind=ConnectionKind.ENTAILS)

    connections = await store.list_connections(
        source=a_id,
        indirect=True,
    )

    assert len(connections) == 2
    assert has_connection(connections, (a_id, b_id))
    assert has_connection(connections, (b_id, c_id))
    assert len(connections) == len(set((c.source, c.target) for c in connections))


async def test_that_connections_are_returned_for_target_without_indirect_connections(
    store: GuidelineConnectionStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")

    await store.create_connection(source=a_id, target=b_id, kind=ConnectionKind.ENTAILS)
    await store.create_connection(source=b_id, target=c_id, kind=ConnectionKind.ENTAILS)

    connections = await store.list_connections(
        target=b_id,
        indirect=False,
    )

    assert len(connections) == 1
    assert has_connection(connections, (a_id, b_id))
    assert not has_connection(connections, (b_id, c_id))


async def test_that_connections_are_returned_for_target_with_indirect_connections(
    store: GuidelineConnectionStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")
    c_id = GuidelineId("c")

    await store.create_connection(source=a_id, target=b_id, kind=ConnectionKind.ENTAILS)
    await store.create_connection(source=b_id, target=c_id, kind=ConnectionKind.ENTAILS)

    connections = await store.list_connections(
        target=c_id,
        indirect=True,
    )

    assert len(connections) == 2
    assert has_connection(connections, (a_id, b_id))
    assert has_connection(connections, (b_id, c_id))
    assert len(connections) == len(set((c.source, c.target) for c in connections))


async def test_that_error_is_raised_when_neither_source_nor_target_is_provided(
    store: GuidelineConnectionStore,
) -> None:
    with raises(AssertionError):
        await store.list_connections(
            indirect=False,
            source=None,
            target=None,
        )


async def test_that_error_is_raised_when_both_source_and_target_are_provided(
    store: GuidelineConnectionStore,
) -> None:
    a_id = GuidelineId("a")
    b_id = GuidelineId("b")

    with raises(AssertionError):
        await store.list_connections(
            source=a_id,
            target=b_id,
            indirect=False,
        )
