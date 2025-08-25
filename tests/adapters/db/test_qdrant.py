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
from typing import AsyncIterator, Iterator, TypedDict, cast
from typing_extensions import Required
from lagom import Container
from pytest import fixture

from parlant.adapters.nlp.openai_service import OpenAITextEmbedding3Large
from parlant.adapters.vector_db.qdrant import QdrantCollection, QdrantDatabase
from parlant.core.agents import AgentStore, AgentId
from parlant.core.common import Version, md5_checksum
from parlant.core.glossary import GlossaryVectorStore
from parlant.core.nlp.embedding import EmbedderFactory, NoOpEmbedder
from parlant.core.loggers import Logger
from parlant.core.nlp.service import NLPService
from parlant.core.persistence.common import ObjectId
from qdrant_client import QdrantClient

from parlant.core.persistence.vector_database import BaseDocument
from tests.test_utilities import SyncAwaiter


class _TestDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    content: str
    checksum: Required[str]
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
async def qdrant_database(context: _TestContext) -> AsyncIterator[QdrantDatabase]:
    async with create_database(context) as qdrant_database:
        yield qdrant_database


def create_database(context: _TestContext) -> QdrantDatabase:
    return QdrantDatabase(
        logger=context.container[Logger],
        client=QdrantClient(path=context.home_dir),
        embedder_factory=EmbedderFactory(context.container),
    )


@fixture
async def qdrant_collection(
    qdrant_database: QdrantDatabase,
) -> AsyncIterator[QdrantCollection[_TestDocument]]:
    collection = await qdrant_database.get_or_create_collection(
        "test_collection",
        _TestDocument,
        embedder_type=OpenAITextEmbedding3Large,
        document_loader=_identity_loader,
    )
    yield collection
    await qdrant_database.delete_collection("test_collection")


async def test_that_a_document_can_be_found_based_on_a_metadata_field(
    qdrant_collection: QdrantCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    doc = _TestDocument(
        id=ObjectId("3a996bf5-185d-4ca6-a470-130b3b0f7de6"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum="test content",
    )

    await qdrant_collection.insert_one(doc)

    find_by_id_result = await qdrant_collection.find(
        {"id": {"$eq": "3a996bf5-185d-4ca6-a470-130b3b0f7de6"}}
    )

    assert len(find_by_id_result) == 1

    assert find_by_id_result[0] == doc

    find_one_result = await qdrant_collection.find_one(
        {"id": {"$eq": "3a996bf5-185d-4ca6-a470-130b3b0f7de6"}}
    )

    assert find_one_result == doc

    find_by_name_result = await qdrant_collection.find({"name": {"$eq": "test name"}})

    assert len(find_by_name_result) == 1
    assert find_by_name_result[0] == doc

    find_by_not_existing_name_result = await qdrant_collection.find(
        {"name": {"$eq": "not existing"}}
    )

    assert len(find_by_not_existing_name_result) == 0


async def test_that_update_one_without_upsert_updates_existing_document(
    qdrant_collection: QdrantCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    document = _TestDocument(
        id=ObjectId("3a996bf5-185d-4ca6-a470-130b3b0f7de6"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    await qdrant_collection.insert_one(document)

    updated_document = _TestDocument(
        id=ObjectId("3a996bf5-185d-4ca6-a470-130b3b0f7de6"),
        version=doc_version,
        content="test content",
        name="new name",
        checksum=md5_checksum("test content"),
    )

    await qdrant_collection.update_one(
        {"name": {"$eq": "test name"}},
        updated_document,
        upsert=False,
    )

    result = await qdrant_collection.find({"name": {"$eq": "test name"}})
    assert len(result) == 0

    result = await qdrant_collection.find({"name": {"$eq": "new name"}})
    assert len(result) == 1
    assert result[0] == updated_document


async def test_that_update_one_without_upsert_and_no_preexisting_document_with_same_id_does_not_insert(
    qdrant_collection: QdrantCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    updated_document = _TestDocument(
        id=ObjectId("3a996bf5-185d-4ca6-a470-130b3b0f7de6"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    result = await qdrant_collection.update_one(
        {"name": {"$eq": "new name"}},
        updated_document,
        upsert=False,
    )

    assert result.matched_count == 0
    print("WHAT I FOUND ", await qdrant_collection.find({}))
    assert 0 == len(await qdrant_collection.find({}))


async def test_that_update_one_with_upsert_and_no_preexisting_document_with_same_id_does_insert_new_document(
    qdrant_collection: QdrantCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    updated_document = _TestDocument(
        id=ObjectId("3a996bf5-185d-4ca6-a470-130b3b0f7de6"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    await qdrant_collection.update_one(
        {"name": {"$eq": "test name"}},
        updated_document,
        upsert=True,
    )

    result = await qdrant_collection.find({"name": {"$eq": "test name"}})

    assert len(result) == 1
    assert result[0] == updated_document


async def test_delete_one(
    qdrant_collection: QdrantCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    document = _TestDocument(
        id=ObjectId("3a996bf5-185d-4ca6-a470-130b3b0f7de6"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    await qdrant_collection.insert_one(document)

    result = await qdrant_collection.find({"id": {"$eq": "3a996bf5-185d-4ca6-a470-130b3b0f7de6"}})
    assert len(result) == 1

    deleted_result = await qdrant_collection.delete_one(
        {"id": {"$eq": "3a996bf5-185d-4ca6-a470-130b3b0f7de6"}}
    )

    assert deleted_result.deleted_count == 1

    if deleted_result.deleted_document:
        assert deleted_result.deleted_document["id"] == ObjectId(
            "3a996bf5-185d-4ca6-a470-130b3b0f7de6"
        )

    result = await qdrant_collection.find({"id": {"$eq": "3a996bf5-185d-4ca6-a470-130b3b0f7de6"}})
    assert len(result) == 0


async def test_find_similar_documents(
    qdrant_collection: QdrantCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    apple_document = _TestDocument(
        id=ObjectId("3a996bf5-185d-4ca6-a470-130b3b0f7de6"),
        version=doc_version,
        content="apple",
        name="Apple",
        checksum=md5_checksum("apple"),
    )

    banana_document = _TestDocument(
        id=ObjectId("85bf7d2c-47e9-47c1-813b-40acfd7f0ff4"),
        version=doc_version,
        content="banana",
        name="Banana",
        checksum=md5_checksum("banana"),
    )

    cherry_document = _TestDocument(
        id=ObjectId("619aa8eb-4b3a-424d-8fde-69df99b5c675"),
        version=doc_version,
        content="cherry",
        name="Cherry",
        checksum=md5_checksum("cherry"),
    )

    await qdrant_collection.insert_one(apple_document)
    await qdrant_collection.insert_one(banana_document)
    await qdrant_collection.insert_one(cherry_document)
    await qdrant_collection.insert_one(
        _TestDocument(
            id=ObjectId("f6572cfa-a227-4eaf-95e0-813054dcd33e"),
            version=doc_version,
            content="date",
            name="Date",
            checksum=md5_checksum("date"),
        )
    )
    await qdrant_collection.insert_one(
        _TestDocument(
            id=ObjectId("01acae77-b06f-4581-a897-4b43e2aaf2dc"),
            version=doc_version,
            content="elderberry",
            name="Elderberry",
            checksum=md5_checksum("elderberry"),
        )
    )

    query = "apple banana cherry"
    k = 3

    result = [s.document for s in await qdrant_collection.find_similar_documents({}, query, k)]

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
            document_loader=_identity_loader,
        )

        document = _TestDocument(
            id=ObjectId("3a996bf5-185d-4ca6-a470-130b3b0f7de6"),
            version=doc_version,
            content="test content",
            name="test name",
            checksum=md5_checksum("test content"),
        )

        await created_collection.insert_one(document)

    async with create_database(context) as second_db:
        fetched_collection: QdrantCollection[_TestDocument] = await second_db.get_collection(
            "test_collection",
            _TestDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=_identity_loader,
        )

        result = await fetched_collection.find(
            {"id": {"$eq": "3a996bf5-185d-4ca6-a470-130b3b0f7de6"}}
        )

        assert len(result) == 1
        assert result[0] == document


async def test_that_glossary_qdrant_store_correctly_finds_relevant_terms_from_large_query_input(
    container: Container,
    agent_id: AgentId,
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        async with QdrantDatabase(
            container[Logger], QdrantClient(path=temp_dir), EmbedderFactory(container)
        ) as qdrant_db:
            async with GlossaryVectorStore(
                qdrant_db,
                embedder_factory=EmbedderFactory(container),
                embedder_type=type(await container[NLPService].get_embedder()),
            ) as glossary_qdrant_store:
                bazoo = await glossary_qdrant_store.create_term(
                    term_set=agent_id,
                    name="Bazoo",
                    description="a type of cow",
                )

                shazoo = await glossary_qdrant_store.create_term(
                    term_set=agent_id,
                    name="Shazoo",
                    description="a type of zebra",
                )

                kazoo = await glossary_qdrant_store.create_term(
                    term_set=agent_id,
                    name="Kazoo",
                    description="a type of horse",
                )

                terms = await glossary_qdrant_store.find_relevant_terms(
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


class _TestDocumentV2(BaseDocument):
    new_name: str


async def _identity_loader(doc: BaseDocument) -> _TestDocument:
    return cast(_TestDocument, doc)


async def test_that_when_persistence_and_store_version_match_allows_store_to_open_when_migrate_is_disabled(
    context: _TestContext,
) -> None:
    async with create_database(context) as qdrant_db:
        async with GlossaryVectorStore(
            vector_db=qdrant_db,
            embedder_factory=EmbedderFactory(context.container),
            embedder_type=NoOpEmbedder,
            allow_migration=False,
        ):
            metadata = await qdrant_db.read_metadata()

            assert metadata
            assert metadata["version"] == GlossaryVectorStore.VERSION.to_string()


async def test_that_new_store_creates_metadata_with_correct_version(
    context: _TestContext,
) -> None:
    async with create_database(context) as qdrant_db:
        async with GlossaryVectorStore(
            vector_db=qdrant_db,
            embedder_factory=EmbedderFactory(context.container),
            embedder_type=OpenAITextEmbedding3Large,
            allow_migration=False,
        ):
            metadata = await qdrant_db.read_metadata()

            assert metadata
            assert metadata["version"] == GlossaryVectorStore.VERSION.to_string()
