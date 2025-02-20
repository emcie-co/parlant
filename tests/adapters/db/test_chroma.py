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

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import AsyncIterator, Iterator, TypedDict
from lagom import Container
from pytest import fixture

from parlant.adapters.nlp.openai import OpenAITextEmbedding3Large
from parlant.adapters.vector_db.chroma import ChromaCollection, ChromaDatabase
from parlant.core.agents import AgentStore, AgentId
from parlant.core.common import Version
from parlant.core.glossary import GlossaryVectorStore
from parlant.core.nlp.embedding import EmbedderFactory
from parlant.core.logging import Logger
from parlant.core.nlp.service import NLPService
from parlant.core.persistence.common import ObjectId

from tests.test_utilities import SyncAwaiter


class _TestDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    content: str
    name: str


@dataclass(frozen=True)
class _TestContext:
    home_dir: Path
    container: Container


@fixture
def agent_id(
    container: Container,
    sync_await: SyncAwaiter,
) -> AgentId:
    store = container[AgentStore]
    agent = sync_await(store.create_agent(name="test-agent", max_engine_iterations=2))
    return agent.id


@fixture
def context(container: Container) -> Iterator[_TestContext]:
    with tempfile.TemporaryDirectory() as home_dir:
        home_dir_path = Path(home_dir)
        yield _TestContext(
            container=container,
            home_dir=home_dir_path,
        )


@fixture
def doc_version() -> Version.String:
    return Version.from_string("0.1.0").to_string()


@fixture
async def chroma_database(context: _TestContext) -> AsyncIterator[ChromaDatabase]:
    async with create_database(context) as chroma_database:
        yield chroma_database


def create_database(context: _TestContext) -> ChromaDatabase:
    return ChromaDatabase(
        logger=context.container[Logger],
        dir_path=context.home_dir,
        embedder_factory=EmbedderFactory(context.container),
    )


@fixture
async def chroma_collection(
    chroma_database: ChromaDatabase,
) -> AsyncIterator[ChromaCollection[_TestDocument]]:
    collection = await chroma_database.get_or_create_collection(
        "test_collection",
        _TestDocument,
        embedder_type=OpenAITextEmbedding3Large,
    )
    yield collection
    await chroma_database.delete_collection("test_collection")


async def test_that_a_document_can_be_found_based_on_a_metadata_field(
    chroma_collection: ChromaCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    doc = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
    )

    await chroma_collection.insert_one(doc)

    find_by_id_result = await chroma_collection.find({"id": {"$eq": "1"}})

    assert len(find_by_id_result) == 1

    assert find_by_id_result[0] == doc

    find_one_result = await chroma_collection.find_one({"id": {"$eq": "1"}})

    assert find_one_result == doc

    find_by_name_result = await chroma_collection.find({"name": {"$eq": "test name"}})

    assert len(find_by_name_result) == 1
    assert find_by_name_result[0] == doc

    find_by_not_existing_name_result = await chroma_collection.find(
        {"name": {"$eq": "not existing"}}
    )

    assert len(find_by_not_existing_name_result) == 0


async def test_that_update_one_without_upsert_updates_existing_document(
    chroma_collection: ChromaCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
    )

    await chroma_collection.insert_one(document)

    updated_document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="new name",
    )

    await chroma_collection.update_one(
        {"name": {"$eq": "test name"}},
        updated_document,
        upsert=False,
    )

    result = await chroma_collection.find({"name": {"$eq": "test name"}})
    assert len(result) == 0

    result = await chroma_collection.find({"name": {"$eq": "new name"}})
    assert len(result) == 1
    assert result[0] == updated_document


async def test_that_update_one_without_upsert_and_no_preexisting_document_with_same_id_does_not_insert(
    chroma_collection: ChromaCollection[_TestDocument],
) -> None:
    updated_document = _TestDocument(
        id=ObjectId("1"),
        content="test content",
        name="test name",
    )

    result = await chroma_collection.update_one(
        {"name": {"$eq": "new name"}},
        updated_document,
        upsert=False,
    )

    assert result.matched_count == 0
    assert 0 == len(await chroma_collection.find({}))


async def test_that_update_one_with_upsert_and_no_preexisting_document_with_same_id_does_insert_new_document(
    chroma_collection: ChromaCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    updated_document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
    )

    await chroma_collection.update_one(
        {"name": {"$eq": "test name"}},
        updated_document,
        upsert=True,
    )

    result = await chroma_collection.find({"name": {"$eq": "test name"}})

    assert len(result) == 1
    assert result[0] == updated_document


async def test_delete_one(
    chroma_collection: ChromaCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
    )

    await chroma_collection.insert_one(document)

    result = await chroma_collection.find({"id": {"$eq": "1"}})
    assert len(result) == 1

    deleted_result = await chroma_collection.delete_one({"id": {"$eq": "1"}})

    assert deleted_result.deleted_count == 1

    if deleted_result.deleted_document:
        assert deleted_result.deleted_document["id"] == ObjectId("1")

    result = await chroma_collection.find({"id": {"$eq": "1"}})
    assert len(result) == 0


async def test_find_similar_documents(
    chroma_collection: ChromaCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    apple_document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="apple",
        name="Apple",
    )

    banana_document = _TestDocument(
        id=ObjectId("2"),
        version=doc_version,
        content="banana",
        name="Banana",
    )

    cherry_document = _TestDocument(
        id=ObjectId("3"),
        version=doc_version,
        content="cherry",
        name="Cherry",
    )

    await chroma_collection.insert_one(apple_document)
    await chroma_collection.insert_one(banana_document)
    await chroma_collection.insert_one(cherry_document)
    await chroma_collection.insert_one(
        _TestDocument(
            id=ObjectId("4"),
            version=doc_version,
            content="date",
            name="Date",
        )
    )
    await chroma_collection.insert_one(
        _TestDocument(
            id=ObjectId("5"),
            version=doc_version,
            content="elderberry",
            name="Elderberry",
        )
    )

    query = "apple banana cherry"
    k = 3

    result = [s.document for s in await chroma_collection.find_similar_documents({}, query, k)]

    assert len(result) == 3
    assert apple_document in result
    assert banana_document in result
    assert cherry_document in result


async def test_loading_collections(
    context: _TestContext,
    doc_version: Version.String,
) -> None:
    async with create_database(context) as first_db:
        created_collection = await first_db.get_or_create_collection(
            "test_collection",
            _TestDocument,
            embedder_type=OpenAITextEmbedding3Large,
        )

        document = _TestDocument(
            id=ObjectId("1"),
            version=doc_version,
            content="test content",
            name="test name",
        )

        await created_collection.insert_one(document)

    async with create_database(context) as second_db:
        fetched_collection: ChromaCollection[_TestDocument] = await second_db.get_collection(
            "test_collection"
        )

        result = await fetched_collection.find({"id": {"$eq": "1"}})

        assert len(result) == 1
        assert result[0] == document


async def test_that_glossary_chroma_store_correctly_finds_relevant_terms_from_large_query_input(
    container: Container,
    agent_id: AgentId,
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        async with ChromaDatabase(
            container[Logger], Path(temp_dir), EmbedderFactory(container)
        ) as chroma_db:
            async with GlossaryVectorStore(
                chroma_db,
                embedder_factory=EmbedderFactory(container),
                embedder_type=type(await container[NLPService].get_embedder()),
            ) as glossary_chroma_store:
                bazoo = await glossary_chroma_store.create_term(
                    term_set=agent_id,
                    name="Bazoo",
                    description="a type of cow",
                )

                shazoo = await glossary_chroma_store.create_term(
                    term_set=agent_id,
                    name="Shazoo",
                    description="a type of zebra",
                )

                kazoo = await glossary_chroma_store.create_term(
                    term_set=agent_id,
                    name="Kazoo",
                    description="a type of horse",
                )

                terms = await glossary_chroma_store.find_relevant_terms(
                    agent_id,
                    ("walla " * 5000)
                    + "Kazoo"
                    + ("balla " * 5000)
                    + "Shazoo"
                    + ("kalla " * 5000)
                    + "Bazoo",
                    max_terms=3,
                )

                assert len(terms) == 3
                assert any(t == kazoo for t in terms)
                assert any(t == shazoo for t in terms)
                assert any(t == bazoo for t in terms)
