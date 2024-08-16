from contextlib import asynccontextmanager
import json
from pathlib import Path
import tempfile
from typing import AsyncIterator

from lagom import Container
from pytest import fixture

from emcie.server.core.agents import Agent, AgentStore
from emcie.server.core.guidelines import GuidelineStore
from emcie.server.core.guideline_connections import GuidelineConnectionStore
from emcie.server.indexer import GuidelineIndexer, Indexer
from tests.test_utilities import SyncAwaiter


@asynccontextmanager
async def new_file_path() -> AsyncIterator[Path]:
    with tempfile.NamedTemporaryFile(delete=True) as new_file:
        yield Path(new_file.name)


@fixture
def agent(
    container: Container,
    sync_await: SyncAwaiter,
) -> Agent:
    store = container[AgentStore]
    agent = sync_await(store.create_agent(name="test-agent"))
    return agent


async def test_that_guidelines_written_in_the_index_file(
    container: Container,
    agent: Agent,
) -> None:
    guideline_store = container[GuidelineStore]

    first_guideline = await guideline_store.create_guideline(
        guideline_set=agent.id,
        predicate="greeting the user",
        content="do your job when the user says hello",
    )

    second_guideline = await guideline_store.create_guideline(
        guideline_set=agent.id,
        predicate="the user asks what is your favourite food",
        content="tell him it is pizza",
    )

    async with new_file_path() as index_file:
        indexer = Indexer(index_json_file=index_file)
        await indexer.index(container)
        with open(index_file, "r") as f:
            indexes = json.load(f)

            indexed_guidelines = indexes["guidelines"][agent.id]

            assert GuidelineIndexer._guideline_checksum(first_guideline) in indexed_guidelines
            assert GuidelineIndexer._guideline_checksum(second_guideline) in indexed_guidelines

            assert (
                indexed_guidelines[GuidelineIndexer._guideline_checksum(first_guideline)]
                == first_guideline.id
            )
            assert (
                indexed_guidelines[GuidelineIndexer._guideline_checksum(second_guideline)]
                == second_guideline.id
            )


async def test_that_removed_guidelines_are_also_removed_from_the_index_file(
    container: Container,
    agent: Agent,
) -> None:
    guideline_store = container[GuidelineStore]

    await guideline_store.create_guideline(
        guideline_set=agent.id,
        predicate="greeting the user",
        content="do your job when the user says hello",
    )

    _guideline = await guideline_store.create_guideline(
        guideline_set=agent.id,
        predicate="the user asks what is your favourite food",
        content="tell him it is pizza",
    )

    async with new_file_path() as index_file:
        await Indexer(index_json_file=index_file).index(container)
        with open(index_file, "r") as f:
            indexes = json.load(f)

            assert len(indexes["guidelines"][agent.id]) == 2

        await guideline_store.delete_guideline(
            guideline_set=agent.id,
            guideline_id=_guideline.id,
        )

        await Indexer(index_json_file=index_file).index(container)

        with open(index_file, "r") as f:
            indexes = json.load(f)

            assert len(indexes["guidelines"][agent.id]) == 1


async def test_that_guideline_connections_are_created(
    container: Container,
    agent: Agent,
) -> None:
    guideline_store = container[GuidelineStore]
    connection_store = container[GuidelineConnectionStore]

    first_guideline = await guideline_store.create_guideline(
        guideline_set=agent.id,
        predicate="The user asks about the weather",
        content="provide the current weather update",
    )

    second_guideline = await guideline_store.create_guideline(
        guideline_set=agent.id,
        predicate="providing the weather update",
        content="mention the best time to go for a walk",
    )

    async with new_file_path() as index_file:
        indexer = Indexer(index_json_file=index_file)
        await indexer.index(container)

        connections = await connection_store.list_connections(
            indirect=False, source=first_guideline.id
        )

        assert len(connections) == 1
        assert connections[0].source == first_guideline.id
        assert connections[0].target == second_guideline.id


async def test_that_guideline_connections_are_removed_when_guideline_deleted(
    container: Container,
    agent: Agent,
) -> None:
    guideline_store = container[GuidelineStore]
    connection_store = container[GuidelineConnectionStore]

    first_guideline = await guideline_store.create_guideline(
        guideline_set=agent.id,
        predicate="The user asks about the weather",
        content="provide the current weather update",
    )

    await guideline_store.create_guideline(
        guideline_set=agent.id,
        predicate="providing the weather update",
        content="mention the best time to go for a walk",
    )

    async with new_file_path() as index_file:
        indexer = Indexer(index_json_file=index_file)
        await indexer.index(container)

        connections = await connection_store.list_connections(
            indirect=False, source=first_guideline.id
        )

        assert len(connections) == 1

        await guideline_store.delete_guideline(
            guideline_set=agent.id,
            guideline_id=first_guideline.id,
        )

        await indexer.index(container)

        connections = await connection_store.list_connections(
            indirect=False, source=first_guideline.id
        )

        assert len(connections) == 0