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

import os
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Optional, TypedDict, cast
from typing_extensions import Required
from lagom import Container
from pytest import fixture, raises, skip

from parlant.adapters.nlp.openai_service import OpenAITextEmbedding3Large
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.adapters.vector_db.pinecone import PineconeCollection, PineconeDatabase
from parlant.core.agents import AgentStore, AgentId
from parlant.core.common import IdGenerator, Version, md5_checksum
from parlant.core.glossary import GlossaryVectorStore
from parlant.core.nlp.embedding import Embedder, EmbedderFactory, NullEmbedder, NullEmbeddingCache
from parlant.core.loggers import Logger
from parlant.core.nlp.service import NLPService
from parlant.core.persistence.common import MigrationRequired, ObjectId
from parlant.core.persistence.vector_database import BaseDocument
from parlant.core.persistence.vector_database_helper import VectorDocumentStoreMigrationHelper
from parlant.core.tags import Tag, TagId
from tests.test_utilities import SyncAwaiter


async def _openai_embedder_type_provider() -> type[Embedder]:
    return OpenAITextEmbedding3Large


async def _null_embedder_type_provider() -> type[Embedder]:
    return NullEmbedder


class _TestDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    content: str
    checksum: Required[str]
    name: str


@dataclass(frozen=True)
class _TestContext:
    container: Container


def _check_pinecone_api_key() -> None:
    """Check if Pinecone API key is available, skip test if not."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        skip("PINECONE_API_KEY environment variable not set. Skipping Pinecone tests.")


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
    yield _TestContext(container=container)


@fixture
def doc_version() -> Version.String:
    return Version.from_string("0.1.0").to_string()


@fixture
async def pinecone_database(context: _TestContext) -> AsyncIterator[PineconeDatabase]:
    _check_pinecone_api_key()
    async with create_database(context) as pinecone_database:
        yield pinecone_database


def create_database(context: _TestContext) -> PineconeDatabase:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable must be set")
    return PineconeDatabase(
        logger=context.container[Logger],
        api_key=api_key,
        embedder_factory=EmbedderFactory(context.container),
        embedding_cache_provider=NullEmbeddingCache,
    )


@fixture
async def pinecone_collection(
    pinecone_database: PineconeDatabase,
) -> AsyncIterator[PineconeCollection[_TestDocument]]:
    collection = await pinecone_database.get_or_create_collection(
        "test_collection",
        _TestDocument,
        embedder_type=OpenAITextEmbedding3Large,
        document_loader=_identity_loader,
    )
    yield collection
    try:
        await pinecone_database.delete_collection("test_collection")
    except Exception:
        pass  # Collection might not exist or already deleted


async def test_that_a_document_can_be_found_based_on_a_metadata_field(
    pinecone_collection: PineconeCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    doc = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum="test content",
    )

    await pinecone_collection.insert_one(doc)

    find_by_id_result = await pinecone_collection.find({"id": {"$eq": "1"}})

    assert len(find_by_id_result) == 1

    assert find_by_id_result[0] == doc

    find_one_result = await pinecone_collection.find_one({"id": {"$eq": "1"}})

    assert find_one_result == doc

    find_by_name_result = await pinecone_collection.find({"name": {"$eq": "test name"}})

    assert len(find_by_name_result) == 1
    assert find_by_name_result[0] == doc

    find_by_not_existing_name_result = await pinecone_collection.find(
        {"name": {"$eq": "not existing"}}
    )

    assert len(find_by_not_existing_name_result) == 0


async def test_that_update_one_without_upsert_updates_existing_document(
    pinecone_collection: PineconeCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    await pinecone_collection.insert_one(document)

    updated_document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="new name",
        checksum=md5_checksum("test content"),
    )

    await pinecone_collection.update_one(
        {"name": {"$eq": "test name"}},
        updated_document,
        upsert=False,
    )

    result = await pinecone_collection.find({"name": {"$eq": "test name"}})
    assert len(result) == 0

    result = await pinecone_collection.find({"name": {"$eq": "new name"}})
    assert len(result) == 1
    assert result[0] == updated_document


async def test_that_update_one_without_upsert_and_no_preexisting_document_with_same_id_does_not_insert(
    pinecone_collection: PineconeCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    updated_document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    result = await pinecone_collection.update_one(
        {"name": {"$eq": "new name"}},
        updated_document,
        upsert=False,
    )

    assert result.matched_count == 0
    assert 0 == len(await pinecone_collection.find({}))


async def test_that_update_one_with_upsert_and_no_preexisting_document_with_same_id_does_insert_new_document(
    pinecone_collection: PineconeCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    updated_document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    await pinecone_collection.update_one(
        {"name": {"$eq": "test name"}},
        updated_document,
        upsert=True,
    )

    result = await pinecone_collection.find({"name": {"$eq": "test name"}})

    assert len(result) == 1
    assert result[0] == updated_document


async def test_delete_one(
    pinecone_collection: PineconeCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    await pinecone_collection.insert_one(document)

    result = await pinecone_collection.find({"id": {"$eq": "1"}})
    assert len(result) == 1

    deleted_result = await pinecone_collection.delete_one({"id": {"$eq": "1"}})

    assert deleted_result.deleted_count == 1

    if deleted_result.deleted_document:
        assert deleted_result.deleted_document["id"] == ObjectId("1")

    result = await pinecone_collection.find({"id": {"$eq": "1"}})
    assert len(result) == 0


async def test_find_similar_documents(
    pinecone_collection: PineconeCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    apple_document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="apple",
        name="Apple",
        checksum=md5_checksum("apple"),
    )

    banana_document = _TestDocument(
        id=ObjectId("2"),
        version=doc_version,
        content="banana",
        name="Banana",
        checksum=md5_checksum("banana"),
    )

    cherry_document = _TestDocument(
        id=ObjectId("3"),
        version=doc_version,
        content="cherry",
        name="Cherry",
        checksum=md5_checksum("cherry"),
    )

    await pinecone_collection.insert_one(apple_document)
    await pinecone_collection.insert_one(banana_document)
    await pinecone_collection.insert_one(cherry_document)
    await pinecone_collection.insert_one(
        _TestDocument(
            id=ObjectId("4"),
            version=doc_version,
            content="date",
            name="Date",
            checksum=md5_checksum("date"),
        )
    )
    await pinecone_collection.insert_one(
        _TestDocument(
            id=ObjectId("5"),
            version=doc_version,
            content="elderberry",
            name="Elderberry",
            checksum=md5_checksum("elderberry"),
        )
    )

    query = "apple banana cherry"
    k = 3

    result = [s.document for s in await pinecone_collection.find_similar_documents({}, query, k)]

    assert len(result) == 3
    assert apple_document in result
    assert banana_document in result
    assert cherry_document in result


async def test_loading_collections(
    context: _TestContext,
    doc_version: Version.String,
) -> None:
    _check_pinecone_api_key()
    async with create_database(context) as first_db:
        created_collection = await first_db.get_or_create_collection(
            "test_collection",
            _TestDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=_identity_loader,
        )

        document = _TestDocument(
            id=ObjectId("1"),
            version=doc_version,
            content="test content",
            name="test name",
            checksum=md5_checksum("test content"),
        )

        await created_collection.insert_one(document)

    async with create_database(context) as second_db:
        fetched_collection: PineconeCollection[_TestDocument] = await second_db.get_collection(
            "test_collection",
            _TestDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=_identity_loader,
        )

        result = await fetched_collection.find({"id": {"$eq": "1"}})

        assert len(result) == 1
        assert result[0] == document


async def _identity_loader(doc: BaseDocument) -> _TestDocument:
    return cast(_TestDocument, doc)


async def test_and_operator_with_multiple_conditions(
    pinecone_collection: PineconeCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    """Test that $and operator works with multiple conditions."""
    doc1 = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="apple",
        name="Apple",
        checksum=md5_checksum("apple"),
    )
    doc2 = _TestDocument(
        id=ObjectId("2"),
        version=doc_version,
        content="banana",
        name="Banana",
        checksum=md5_checksum("banana"),
    )
    doc3 = _TestDocument(
        id=ObjectId("3"),
        version=doc_version,
        content="cherry",
        name="Apple",  # Same name as doc1
        checksum=md5_checksum("cherry"),
    )

    await pinecone_collection.insert_one(doc1)
    await pinecone_collection.insert_one(doc2)
    await pinecone_collection.insert_one(doc3)

    # Find documents where name is "Apple" AND id is "1"
    results = await pinecone_collection.find(
        {
            "$and": [
                {"name": {"$eq": "Apple"}},
                {"id": {"$eq": "1"}},
            ]
        }
    )
    assert len(results) == 1
    assert results[0]["id"] == ObjectId("1")


async def test_or_operator_with_multiple_conditions(
    pinecone_collection: PineconeCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    """Test that $or operator works with multiple conditions."""
    doc1 = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="apple",
        name="Apple",
        checksum=md5_checksum("apple"),
    )
    doc2 = _TestDocument(
        id=ObjectId("2"),
        version=doc_version,
        content="banana",
        name="Banana",
        checksum=md5_checksum("banana"),
    )
    doc3 = _TestDocument(
        id=ObjectId("3"),
        version=doc_version,
        content="cherry",
        name="Cherry",
        checksum=md5_checksum("cherry"),
    )

    await pinecone_collection.insert_one(doc1)
    await pinecone_collection.insert_one(doc2)
    await pinecone_collection.insert_one(doc3)

    # Find documents where name is "Apple" OR name is "Banana"
    results = await pinecone_collection.find(
        {
            "$or": [
                {"name": {"$eq": "Apple"}},
                {"name": {"$eq": "Banana"}},
            ]
        }
    )
    assert len(results) == 2
    result_names = {r["name"] for r in results}
    assert "Apple" in result_names
    assert "Banana" in result_names

