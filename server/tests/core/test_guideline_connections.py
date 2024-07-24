from typing import Sequence, Tuple
from pytest import fixture

from emcie.server.core.guideline_connections import (
    GuidelineConnection,
    GuidelineConnectionDocumentStore,
    GuidelineConnectionStore,
)
from emcie.server.core.guidelines import GuidelineId
from emcie.server.core.persistence.document_database import DocumentDatabase
from emcie.server.core.persistence.transient_database import TransientDocumentDatabase


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
    connection: Tuple[str, str],
) -> bool:
    return any(g.source == connection[0] and g.target == connection[1] for g in guidelines)


async def test_that_direct_guideline_connections_can_be_listed(
    store: GuidelineConnectionStore,
) -> None:
    for source, target in [("a", "b"), ("a", "c"), ("b", "d"), ("z", "b")]:
        await store.update_connection(
            source=GuidelineId(source),
            target=GuidelineId(target),
            kind="entails",
        )

    a_connections = await store.list_connections(
        source=GuidelineId("a"),
        indirect=False,
    )

    assert len(a_connections) == 2
    assert has_connection(a_connections, ("a", "b"))
    assert has_connection(a_connections, ("a", "c"))


async def test_that_indirect_guideline_connections_can_be_listed(
    store: GuidelineConnectionStore,
) -> None:
    for source, target in [("a", "b"), ("a", "c"), ("b", "d"), ("z", "b")]:
        await store.update_connection(
            source=GuidelineId(source),
            target=GuidelineId(target),
            kind="entails",
        )

    a_connections = await store.list_connections(
        source=GuidelineId("a"),
        indirect=True,
    )

    assert len(a_connections) == 3
    assert has_connection(a_connections, ("a", "b"))
    assert has_connection(a_connections, ("a", "c"))
    assert has_connection(a_connections, ("b", "d"))


async def test_that_db_data_is_loaded_correctly(
    store: GuidelineConnectionStore,
    underlying_database: DocumentDatabase,
) -> None:
    for source, target in [("a", "b"), ("a", "c"), ("b", "d"), ("z", "b")]:
        await store.update_connection(
            source=GuidelineId(source),
            target=GuidelineId(target),
            kind="entails",
        )

    new_store_with_same_db = GuidelineConnectionDocumentStore(underlying_database)

    a_connections = await new_store_with_same_db.list_connections(
        source=GuidelineId("a"),
        indirect=True,
    )

    assert len(a_connections) == 3
    assert has_connection(a_connections, ("a", "b"))
    assert has_connection(a_connections, ("a", "c"))
    assert has_connection(a_connections, ("b", "d"))
