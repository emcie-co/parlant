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

import os
import uuid
from typing import AsyncIterator, TypedDict, cast

from lagom import Container
from pytest import fixture, mark
from typing_extensions import Required

from parlant.adapters.nlp.openai_service import OpenAITextEmbedding3Large
from parlant.adapters.vector_db.mongo import MongoVectorCollection, MongoVectorDatabase
from parlant.core.common import Version, md5_checksum
from parlant.core.nlp.embedding import EmbedderFactory, NullEmbeddingCache
from parlant.core.loggers import Logger
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.vector_database import BaseDocument
from parlant.core.tracer import Tracer

from pymongo import AsyncMongoClient

pytestmark = mark.skipif(
    not os.environ.get("TEST_MONGO_ATLAS_URI"),
    reason="TEST_MONGO_ATLAS_URI not set",
)


class _TestDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    content: str
    checksum: Required[str]
    name: str


async def _identity_loader(doc: BaseDocument) -> _TestDocument:
    return cast(_TestDocument, doc)


@fixture
def doc_version() -> Version.String:
    return Version.from_string("0.1.0").to_string()


@fixture
async def mongo_client() -> AsyncIterator[AsyncMongoClient[dict[str, object]]]:
    uri = os.environ["TEST_MONGO_ATLAS_URI"]
    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(uri)
    yield client
    await client.close()


@fixture
async def mongo_database(
    container: Container,
    mongo_client: AsyncMongoClient[dict[str, object]],
) -> AsyncIterator[MongoVectorDatabase]:
    # Use a unique database name per test run to avoid collisions
    db_name = f"parlant_vector_test_{uuid.uuid4().hex[:8]}"

    async with MongoVectorDatabase(
        mongo_client=mongo_client,
        database_name=db_name,
        logger=container[Logger],
        tracer=container[Tracer],
        embedder_factory=EmbedderFactory(container),
        embedding_cache_provider=NullEmbeddingCache,
    ) as db:
        yield db

    # Cleanup: drop the test database
    await mongo_client.drop_database(db_name)


@fixture
async def mongo_collection(
    mongo_database: MongoVectorDatabase,
) -> AsyncIterator[MongoVectorCollection[_TestDocument]]:
    collection = await mongo_database.get_or_create_collection(
        "test_collection",
        _TestDocument,
        embedder_type=OpenAITextEmbedding3Large,
        document_loader=_identity_loader,
    )
    yield collection
    await mongo_database.delete_collection("test_collection")


async def test_that_a_document_can_be_found_based_on_a_metadata_field(
    mongo_collection: MongoVectorCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    doc = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum="test content",
    )

    await mongo_collection.insert_one(doc)

    find_by_id_result = await mongo_collection.find({"id": {"$eq": "1"}})
    assert len(find_by_id_result) == 1
    assert find_by_id_result[0] == doc

    find_one_result = await mongo_collection.find_one({"id": {"$eq": "1"}})
    assert find_one_result == doc

    find_by_name_result = await mongo_collection.find({"name": {"$eq": "test name"}})
    assert len(find_by_name_result) == 1
    assert find_by_name_result[0] == doc

    find_by_not_existing_name_result = await mongo_collection.find(
        {"name": {"$eq": "not existing"}}
    )
    assert len(find_by_not_existing_name_result) == 0


async def test_that_update_one_without_upsert_updates_existing_document(
    mongo_collection: MongoVectorCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    await mongo_collection.insert_one(document)

    updated_document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="new name",
        checksum=md5_checksum("test content"),
    )

    await mongo_collection.update_one(
        {"name": {"$eq": "test name"}},
        updated_document,
        upsert=False,
    )

    result = await mongo_collection.find({"name": {"$eq": "test name"}})
    assert len(result) == 0

    result = await mongo_collection.find({"name": {"$eq": "new name"}})
    assert len(result) == 1
    assert result[0] == updated_document


async def test_that_update_one_without_upsert_and_no_preexisting_document_does_not_insert(
    mongo_collection: MongoVectorCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    updated_document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    result = await mongo_collection.update_one(
        {"name": {"$eq": "new name"}},
        updated_document,
        upsert=False,
    )

    assert result.matched_count == 0
    assert 0 == len(await mongo_collection.find({}))


async def test_that_update_one_with_upsert_and_no_preexisting_document_inserts_new_document(
    mongo_collection: MongoVectorCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    updated_document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    await mongo_collection.update_one(
        {"name": {"$eq": "test name"}},
        updated_document,
        upsert=True,
    )

    result = await mongo_collection.find({"name": {"$eq": "test name"}})
    assert len(result) == 1
    assert result[0] == updated_document


async def test_that_delete_one_removes_document(
    mongo_collection: MongoVectorCollection[_TestDocument],
    doc_version: Version.String,
) -> None:
    document = _TestDocument(
        id=ObjectId("1"),
        version=doc_version,
        content="test content",
        name="test name",
        checksum=md5_checksum("test content"),
    )

    await mongo_collection.insert_one(document)

    result = await mongo_collection.find({"id": {"$eq": "1"}})
    assert len(result) == 1

    deleted_result = await mongo_collection.delete_one({"id": {"$eq": "1"}})
    assert deleted_result.deleted_count == 1

    if deleted_result.deleted_document:
        assert deleted_result.deleted_document["id"] == ObjectId("1")

    result = await mongo_collection.find({"id": {"$eq": "1"}})
    assert len(result) == 0


async def test_that_find_similar_documents_returns_ranked_results(
    mongo_collection: MongoVectorCollection[_TestDocument],
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

    await mongo_collection.insert_one(apple_document)
    await mongo_collection.insert_one(banana_document)
    await mongo_collection.insert_one(cherry_document)
    await mongo_collection.insert_one(
        _TestDocument(
            id=ObjectId("4"),
            version=doc_version,
            content="date",
            name="Date",
            checksum=md5_checksum("date"),
        )
    )
    await mongo_collection.insert_one(
        _TestDocument(
            id=ObjectId("5"),
            version=doc_version,
            content="elderberry",
            name="Elderberry",
            checksum=md5_checksum("elderberry"),
        )
    )

    # Atlas vector search indexes update asynchronously;
    # wait for all documents to become searchable.
    import asyncio

    query = "apple banana cherry"
    k = 3

    for _ in range(10):
        result = [s.document for s in await mongo_collection.find_similar_documents({}, query, k)]
        if len(result) >= 3:
            break
        await asyncio.sleep(2)

    assert len(result) == 3
    assert apple_document in result
    assert banana_document in result
    assert cherry_document in result


async def test_that_metadata_operations_work(
    mongo_database: MongoVectorDatabase,
) -> None:
    await mongo_database.upsert_metadata("key1", "value1")
    await mongo_database.upsert_metadata("key2", 42)

    metadata = await mongo_database.read_metadata()
    assert metadata["key1"] == "value1"
    assert metadata["key2"] == 42

    await mongo_database.upsert_metadata("key1", "updated_value")
    metadata = await mongo_database.read_metadata()
    assert metadata["key1"] == "updated_value"

    await mongo_database.remove_metadata("key1")
    metadata = await mongo_database.read_metadata()
    assert "key1" not in metadata
    assert metadata["key2"] == 42


async def test_that_get_or_create_collection_is_idempotent(
    mongo_database: MongoVectorDatabase,
) -> None:
    collection1 = await mongo_database.get_or_create_collection(
        "idempotent_test",
        _TestDocument,
        embedder_type=OpenAITextEmbedding3Large,
        document_loader=_identity_loader,
    )

    collection2 = await mongo_database.get_or_create_collection(
        "idempotent_test",
        _TestDocument,
        embedder_type=OpenAITextEmbedding3Large,
        document_loader=_identity_loader,
    )

    assert collection1 is collection2

    await mongo_database.delete_collection("idempotent_test")


async def test_that_delete_collection_removes_it(
    mongo_database: MongoVectorDatabase,
) -> None:
    await mongo_database.get_or_create_collection(
        "to_delete",
        _TestDocument,
        embedder_type=OpenAITextEmbedding3Large,
        document_loader=_identity_loader,
    )

    await mongo_database.delete_collection("to_delete")

    collection_names = await mongo_database._database.list_collection_names()
    assert "to_delete" not in collection_names


async def test_that_loading_collection_preserves_documents(
    mongo_database: MongoVectorDatabase,
    mongo_client: AsyncMongoClient[dict[str, object]],
    container: Container,
    doc_version: Version.String,
) -> None:
    collection = await mongo_database.get_or_create_collection(
        "persist_test",
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

    await collection.insert_one(document)

    # Create a new database instance pointing to the same database
    async with MongoVectorDatabase(
        mongo_client=mongo_client,
        database_name=mongo_database._database_name,
        logger=container[Logger],
        tracer=container[Tracer],
        embedder_factory=EmbedderFactory(container),
        embedding_cache_provider=NullEmbeddingCache,
    ) as second_db:
        fetched_collection: MongoVectorCollection[_TestDocument] = await second_db.get_collection(
            "persist_test",
            _TestDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=_identity_loader,
        )

        result = await fetched_collection.find({"id": {"$eq": "1"}})
        assert len(result) == 1
        assert result[0] == document

    await mongo_database.delete_collection("persist_test")
