from typing import AsyncIterator, TypedDict, cast

from lagom import Container
from pytest import fixture
from typing_extensions import Required

from parlant.adapters.vector_db.transient import TransientVectorCollection, TransientVectorDatabase
from parlant.core.common import Version
from parlant.core.loggers import Logger
from parlant.core.nlp.embedding import EmbedderFactory, NullEmbeddingCache
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.vector_database import BaseDocument, VectorCollectionIndex
from parlant.core.tracer import Tracer


class _TestDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    content: str
    checksum: Required[str]
    category: str


@fixture
async def transient_database(
    container: Container,
) -> AsyncIterator[TransientVectorDatabase]:
    db = TransientVectorDatabase(
        logger=container[Logger],
        tracer=container[Tracer],
        embedder_factory=EmbedderFactory(container),
        embedding_cache_provider=NullEmbeddingCache,
    )
    yield db


@fixture
async def transient_collection(
    transient_database: TransientVectorDatabase,
) -> TransientVectorCollection[_TestDocument]:
    async def loader(doc: BaseDocument) -> _TestDocument:
        return cast(_TestDocument, doc)

    from parlant.core.nlp.embedding import NullEmbedder

    collection = await transient_database.get_or_create_collection(
        "test_collection",
        _TestDocument,
        embedder_type=NullEmbedder,
        document_loader=loader,
    )
    return collection


def test_that_negative_cosine_similarity_is_not_treated_as_close_distance() -> None:
    assert TransientVectorCollection._distance_from_similarity(1.0) == 0.0
    assert TransientVectorCollection._distance_from_similarity(0.0) == 1.0
    assert TransientVectorCollection._distance_from_similarity(-1.0) == 2.0


async def test_that_ensure_indexes_accepts_vector_collection_indexes(
    transient_collection: TransientVectorCollection[_TestDocument],
) -> None:
    await transient_collection.ensure_indexes([VectorCollectionIndex(field="category")])


async def test_that_ensure_indexes_is_idempotent(
    transient_collection: TransientVectorCollection[_TestDocument],
) -> None:
    indexes = [VectorCollectionIndex(field="category")]
    await transient_collection.ensure_indexes(indexes)
    await transient_collection.ensure_indexes(indexes)
