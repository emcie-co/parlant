from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import AsyncIterator, Iterator
from lagom import Container
from pytest import fixture

from emcie.server.core.generation.embedding import EmbedderFactory, OpenAITextEmbedding3Large
from emcie.server.core.persistence.chroma_database import (
    ChromaCollection,
    ChromaDatabase,
    ChromaDocument,
)
from emcie.server.core.persistence.common import ObjectId
from emcie.server.logger import Logger


class _TestModel(ChromaDocument):
    id: ObjectId
    content: str
    name: str


@dataclass(frozen=True)
class _TestContext:
    home_dir: Path
    container: Container


@fixture
def context(container: Container) -> Iterator[_TestContext]:
    with tempfile.TemporaryDirectory() as home_dir:
        home_dir_path = Path(home_dir)
        yield _TestContext(
            container=container,
            home_dir=home_dir_path,
        )


@fixture
def chroma_database(context: _TestContext) -> ChromaDatabase:
    return create_database(context)


def create_database(context: _TestContext) -> ChromaDatabase:
    return ChromaDatabase(
        logger=context.container[Logger],
        dir_path=context.home_dir,
        embedder_factory=EmbedderFactory(context.container),
    )


@fixture
async def chroma_collection(
    chroma_database: ChromaDatabase,
) -> AsyncIterator[ChromaCollection[_TestModel]]:
    collection = chroma_database.get_or_create_collection(
        "test_collection",
        _TestModel,
        embedder_type=OpenAITextEmbedding3Large,
    )
    yield collection
    chroma_database.delete_collection("test_collection")


async def test_that_create_document_and_find_with_metadata_field(
    chroma_collection: ChromaCollection[_TestModel],
) -> None:
    doc = _TestModel(
        id=ObjectId("1"),
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


async def test_that_update_one_without_upsert_is_updating_existing_document(
    chroma_collection: ChromaCollection[_TestModel],
) -> None:
    document = _TestModel(
        id=ObjectId("1"),
        content="test content",
        name="test name",
    )

    await chroma_collection.insert_one(document)

    updated_document = _TestModel(
        id=ObjectId("1"),
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


async def test_that_update_one_without_upsert_and_no_existing_content_does_not_insert(
    chroma_collection: ChromaCollection[_TestModel],
) -> None:
    updated_document = _TestModel(
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


async def test_that_update_one_with_upsert_and_no_existing_content_inserts_new_document(
    chroma_collection: ChromaCollection[_TestModel],
) -> None:
    updated_document = _TestModel(
        id=ObjectId("1"),
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
    chroma_collection: ChromaCollection[_TestModel],
) -> None:
    document = _TestModel(
        id=ObjectId("1"),
        content="test content",
        name="test name",
    )

    await chroma_collection.insert_one(document)

    result = await chroma_collection.find({"id": {"$eq": "1"}})
    assert len(result) == 1

    deleted_result = await chroma_collection.delete_one({"id": {"$eq": "1"}})

    assert deleted_result.deleted_count == 1

    if deleted_result.deleted_document:
        assert deleted_result.deleted_document.id == ObjectId("1")

    result = await chroma_collection.find({"id": {"$eq": "1"}})
    assert len(result) == 0


async def test_find_similar_documents(
    chroma_collection: ChromaCollection[_TestModel],
) -> None:
    apple_document = _TestModel(
        id=ObjectId("1"),
        content="apple",
        name="Apple",
    )

    banana_document = _TestModel(
        id=ObjectId("2"),
        content="banana",
        name="Banana",
    )

    cherry_document = _TestModel(
        id=ObjectId("3"),
        content="cherry",
        name="Cherry",
    )

    await chroma_collection.insert_one(apple_document)
    await chroma_collection.insert_one(banana_document)
    await chroma_collection.insert_one(cherry_document)
    await chroma_collection.insert_one(
        _TestModel(
            id=ObjectId("4"),
            content="date",
            name="Date",
        )
    )
    await chroma_collection.insert_one(
        _TestModel(
            id=ObjectId("5"),
            content="elderberry",
            name="Elderberry",
        )
    )

    query = "apple banana cherry"
    k = 3

    result = await chroma_collection.find_similar_documents({}, query, k)

    assert len(result) == 3
    assert apple_document in result
    assert banana_document in result
    assert cherry_document in result


async def test_loading_collections_succeed(context: _TestContext, container: Container) -> None:
    chroma_database_1 = create_database(context)
    chroma_collection_1 = chroma_database_1.get_or_create_collection(
        "test_collection",
        _TestModel,
        embedder_type=OpenAITextEmbedding3Large,
    )

    document = _TestModel(
        id=ObjectId("1"),
        content="test content",
        name="test name",
    )

    await chroma_collection_1.insert_one(document)

    chroma_database_2 = create_database(context)
    chroma_collection_2: ChromaCollection[_TestModel] = chroma_database_2.get_collection(
        "test_collection"
    )

    result = await chroma_collection_2.find({"id": {"$eq": "1"}})

    assert len(result) == 1
    assert result[0] == document
