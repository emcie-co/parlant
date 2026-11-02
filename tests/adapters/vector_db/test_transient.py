from typing import Any, AsyncIterator, Mapping, TypedDict, cast
from unittest.mock import AsyncMock, Mock

import pytest

from parlant.core.container import Container
from pytest import fixture
from typing_extensions import Required

from parlant.adapters.vector_db.transient import TransientVectorCollection, TransientVectorDatabase
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.common import Version
from parlant.core.loggers import Logger
from parlant.core.meter import LocalMeter
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.embedding import (
    BaseEmbedder,
    BasicEmbeddingCache,
    EmbeddingResult,
    NullEmbedder,
    NullEmbeddingCache,
)
from parlant.core.nlp.common import UsageInfo
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.vector_database import (
    BaseDocument,
    VectorCollection,
    VectorCollectionIndex,
)
from parlant.core.tracer import Tracer
from tests.test_utilities import DeterministicEmbedder


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
        embedding_cache_provider=NullEmbeddingCache,
    )
    yield db


@fixture
async def transient_collection(
    transient_database: TransientVectorDatabase,
) -> TransientVectorCollection[_TestDocument]:
    async def loader(doc: BaseDocument) -> _TestDocument:
        return cast(_TestDocument, doc)

    collection = await transient_database.get_or_create_collection(
        "test_collection",
        _TestDocument,
        embedder=NullEmbedder(),
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


class _ReportingEmbedder(BaseEmbedder):
    """A deterministic embedder with the base class's usage reporting."""

    def __init__(self, logger: Logger, tracer: Tracer, usage_reporter: Any) -> None:
        super().__init__(
            logger=logger,
            tracer=tracer,
            meter=LocalMeter(logger),
            model_name="deterministic",
            usage_reporter=usage_reporter,
            provider_name="test",
        )
        self._inner = DeterministicEmbedder()

    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        return await self._inner.embed(texts, hints)

    @property
    def id(self) -> str:
        return self._inner.id

    @property
    def max_tokens(self) -> int:
        return self._inner.max_tokens

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return self._inner.tokenizer

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions


@fixture
def usage_reporter() -> Mock:
    return Mock()


@fixture
async def replacement_collection(
    logger: Logger,
    tracer: Tracer,
    usage_reporter: Mock,
) -> TransientVectorCollection[_TestDocument]:
    cache = BasicEmbeddingCache(TransientDocumentDatabase())
    db = TransientVectorDatabase(logger, tracer, lambda: cache)
    return await db.create_collection(
        "replacements",
        _TestDocument,
        _ReportingEmbedder(logger, tracer, usage_reporter),
    )


def _document(id: str, content: str, category: str = "replace") -> _TestDocument:
    return _TestDocument(
        id=ObjectId(id),
        content=content,
        category=category,
        version=Version.String("0.1.0"),
        checksum=content,
    )


@pytest.mark.parametrize("fallback", [False, True])
async def test_replacement_preserves_unrelated_documents_and_search_results(
    replacement_collection: TransientVectorCollection[_TestDocument],
    fallback: bool,
) -> None:
    collection = replacement_collection
    other = _document("other", "weather forecast", "keep")
    for document in [_document("old", "old content"), _document("reuse", "old signal"), other]:
        await collection.insert_one(document)
    replacements = [_document("reuse", "money back"), _document("new", "reimbursement")]

    if fallback:
        await VectorCollection.replace_documents(
            collection, {"category": {"$eq": "replace"}}, replacements
        )
    else:
        await collection.replace_documents({"category": {"$eq": "replace"}}, replacements)

    assert sorted(await collection.find({}), key=lambda d: d["id"]) == sorted(
        [other, *replacements], key=lambda d: d["id"]
    )
    hits = await collection.find_similar_documents({}, "reimbursement", 3)
    assert hits[0].document["id"] == "new"
    assert {h.document["id"] for h in hits} == {"reuse", "new", "other"}
    assert "__vector__" not in replacements[0]


@pytest.mark.parametrize("fallback", [False, True])
async def test_replacement_handles_empty_matches_and_empty_replacements(
    replacement_collection: TransientVectorCollection[_TestDocument],
    fallback: bool,
) -> None:
    collection = replacement_collection
    replace = (
        VectorCollection.replace_documents.__get__(collection)
        if fallback
        else collection.replace_documents
    )
    other = _document("other", "weather", "keep")
    await collection.insert_one(other)

    await replace({"category": {"$eq": "replace"}}, [_document("new", "refund")])
    assert len(await collection.find({})) == 2
    await replace({"category": {"$eq": "replace"}}, [])
    await replace({"category": {"$eq": "replace"}}, [])
    assert await collection.find({}) == [other]
    assert [h.document for h in await collection.find_similar_documents({}, "weather", 1)] == [
        other
    ]
    await replace({}, [])
    assert await collection.find({}) == []
    assert await collection.find_similar_documents({}, "weather", 1) == []


async def test_replacement_batches_matrix_mutations(
    replacement_collection: TransientVectorCollection[_TestDocument],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = replacement_collection
    for i in range(3):
        await collection.insert_one(_document(f"old-{i}", f"old {i}"))
    delete = Mock(wraps=collection._nano_db.delete)
    upsert = Mock(wraps=collection._nano_db.upsert)
    monkeypatch.setattr(collection._nano_db, "delete", delete)
    monkeypatch.setattr(collection._nano_db, "upsert", upsert)

    await collection.replace_documents({}, [_document(f"new-{i}", f"new {i}") for i in range(4)])

    delete.assert_called_once_with(["old-0", "old-1", "old-2"])
    upsert.assert_called_once()
    assert len(upsert.call_args.args[0]) == 4


@pytest.mark.parametrize(
    "failure", ["provider", "dimensions", "nonfinite", "duplicate", "collision"]
)
async def test_failed_replacement_keeps_existing_documents_and_vectors(
    replacement_collection: TransientVectorCollection[_TestDocument],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    collection = replacement_collection
    original = [_document("old", "refund"), _document("other", "weather", "keep")]
    for document in original:
        await collection.insert_one(document)
    replacements = [_document("new", "money back")]
    with monkeypatch.context() as patch:
        if failure == "provider":
            patch.setattr(
                collection._embedder, "embed", AsyncMock(side_effect=RuntimeError("offline"))
            )
        elif failure in ("dimensions", "nonfinite"):
            vector = (
                [1.0]
                if failure == "dimensions"
                else [float("nan")] * collection._embedder.dimensions
            )
            patch.setattr(
                collection._embedder,
                "embed",
                AsyncMock(return_value=EmbeddingResult(vectors=[vector])),
            )
        elif failure == "duplicate":
            replacements *= 2
        else:
            replacements = [_document("other", "money back")]

        with pytest.raises((RuntimeError, ValueError)):
            await collection.replace_documents({"category": {"$eq": "replace"}}, replacements)

    assert await collection.find({}) == original
    hits = await collection.find_similar_documents({}, "refund", 2)
    assert hits[0].document["id"] == "old"
    assert {h.document["id"] for h in hits} == {"old", "other"}


async def test_replacement_reuses_per_text_cache_and_reports_cached_usage(
    replacement_collection: TransientVectorCollection[_TestDocument],
    usage_reporter: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = replacement_collection
    await collection.insert_one(_document("old", "old content"))
    cache = collection._embedding_cache_provider()
    warm_vector = (await collection._embedder.embed(["warm text"])).vectors[0]
    await cache.set(collection._embedder, ["warm text"], [warm_vector], input_tokens=[7])
    cold_vector = (await collection._embedder.embed(["cold text"])).vectors[0]
    reporter = usage_reporter
    reporter.reset_mock()
    embed = AsyncMock(
        return_value=EmbeddingResult(
            vectors=[cold_vector],
            usage=UsageInfo(input_tokens=3, output_tokens=0, requests=1),
        )
    )
    monkeypatch.setattr(collection._embedder, "embed", embed)

    await collection.replace_documents(
        {}, [_document("warm", "warm text"), _document("cold", "cold text")]
    )

    embed.assert_awaited_once_with(["cold text"])
    reporter.report_usage.assert_called_once()
    assert reporter.report_usage.call_args.args[0] == "deterministic"
    assert reporter.report_usage.call_args.args[1].cached_input_tokens == 7
    assert reporter.report_usage.call_args.kwargs["provider_name"] == "test"
    cached = await cache.get(collection._embedder, ["cold text"])
    assert cached is not None
    assert cached.vectors == [cold_vector]
    assert cached.usage.cached_input_tokens == 3
