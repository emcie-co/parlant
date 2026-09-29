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
import pytest
from collections.abc import Mapping
from typing import Any, Sequence
from typing_extensions import override

from parlant.core.nlp.embedding import (
    Embedder,
    EmbedderFactory,
    EmbeddingCache,
    EmbeddingResult,
    NullEmbeddingCache,
)
from parlant.core.nlp.tokenization import EstimatingTokenizer, ZeroEstimatingTokenizer
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.vector_database import BaseDocument
from parlant.adapters.vector_db.valkey import ValkeyVectorDatabase


VALKEY_URL = os.environ.get("VALKEY_URL", "valkey://localhost:6379")

pytestmark = pytest.mark.skipif(
    not os.environ.get("VALKEY_URL"),
    reason="VALKEY_URL not set; Valkey instance required for integration tests",
)


class FakeEmbedder(Embedder):
    """A deterministic embedder that produces vectors based on content hash."""

    def __init__(self, dimensions: int = 128) -> None:
        self._dimensions = dimensions
        self._tokenizer = ZeroEstimatingTokenizer()

    @override
    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        vectors: list[Sequence[float]] = []
        for text in texts:
            # Create a vector based on character frequencies for semantic similarity
            vec = [0.0] * self._dimensions
            for i, ch in enumerate(text):
                vec[ord(ch) % self._dimensions] += 1.0
            # Normalize
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return EmbeddingResult(vectors=vectors)

    @property
    @override
    def id(self) -> str:
        return "fake_embedder"

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def dimensions(self) -> int:
        return self._dimensions


class FakeEmbedderFactory(EmbedderFactory):
    def __init__(self) -> None:
        pass  # type: ignore[override]

    @override
    def create_embedder(self, embedder_type: type[Embedder]) -> Embedder:
        return FakeEmbedder()


def _null_cache_provider() -> EmbeddingCache:
    return NullEmbeddingCache()


class FakeLogger:
    def trace(self, msg: str) -> None:
        pass

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass


class FakeTracer:
    def span(self, name: str) -> "FakeSpan":
        return FakeSpan()


class FakeSpan:
    def __enter__(self) -> "FakeSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


@pytest.fixture
async def valkey_db():
    db = ValkeyVectorDatabase(
        logger=FakeLogger(),  # type: ignore[arg-type]
        tracer=FakeTracer(),  # type: ignore[arg-type]
        url=VALKEY_URL,
        embedder_factory=FakeEmbedderFactory(),  # type: ignore[arg-type]
        embedding_cache_provider=_null_cache_provider,
    )
    async with db:
        yield db


@pytest.fixture
async def valkey_collection(valkey_db: ValkeyVectorDatabase):
    collection = await valkey_db.create_collection(
        name="test_collection",
        schema=BaseDocument,
        embedder_type=FakeEmbedder,
    )
    yield collection
    try:
        await valkey_db.delete_collection("test_collection")
    except Exception as e:
        print(f"[fixture teardown] Failed to delete test_collection: {e}")


@pytest.mark.asyncio
async def test_that_valkey_db_create_collection_creates_index(
    valkey_db: ValkeyVectorDatabase,
) -> None:
    collection = await valkey_db.create_collection(
        name="create_test",
        schema=BaseDocument,
        embedder_type=FakeEmbedder,
    )
    try:
        assert collection is not None
    finally:
        await valkey_db.delete_collection("create_test")


@pytest.mark.asyncio
async def test_that_valkey_db_get_collection_returns_existing_collection(
    valkey_db: ValkeyVectorDatabase,
) -> None:
    await valkey_db.create_collection(
        name="get_test",
        schema=BaseDocument,
        embedder_type=FakeEmbedder,
    )
    try:
        collection = await valkey_db.get_collection(
            name="get_test",
            schema=BaseDocument,
            embedder_type=FakeEmbedder,
            document_loader=lambda doc: doc,  # type: ignore[arg-type, return-value]
        )
        assert collection is not None
    finally:
        await valkey_db.delete_collection("get_test")


@pytest.mark.asyncio
async def test_that_valkey_db_get_collection_raises_on_missing(
    valkey_db: ValkeyVectorDatabase,
) -> None:
    with pytest.raises(ValueError, match="not found"):
        await valkey_db.get_collection(
            name="nonexistent",
            schema=BaseDocument,
            embedder_type=FakeEmbedder,
            document_loader=lambda doc: doc,  # type: ignore[arg-type, return-value]
        )


@pytest.mark.asyncio
async def test_that_valkey_db_get_or_create_collection_creates_when_missing(
    valkey_db: ValkeyVectorDatabase,
) -> None:
    collection = await valkey_db.get_or_create_collection(
        name="get_or_create_test",
        schema=BaseDocument,
        embedder_type=FakeEmbedder,
        document_loader=lambda doc: doc,  # type: ignore[arg-type, return-value]
    )
    try:
        assert collection is not None
    finally:
        await valkey_db.delete_collection("get_or_create_test")


@pytest.mark.asyncio
async def test_that_valkey_db_delete_collection_removes_index_and_data(
    valkey_db: ValkeyVectorDatabase,
) -> None:
    await valkey_db.create_collection(
        name="delete_test",
        schema=BaseDocument,
        embedder_type=FakeEmbedder,
    )
    await valkey_db.delete_collection("delete_test")
    with pytest.raises(ValueError, match="not found"):
        await valkey_db.get_collection(
            name="delete_test",
            schema=BaseDocument,
            embedder_type=FakeEmbedder,
            document_loader=lambda doc: doc,  # type: ignore[arg-type, return-value]
        )


@pytest.mark.asyncio
async def test_that_valkey_db_upsert_metadata_stores_value(
    valkey_db: ValkeyVectorDatabase,
) -> None:
    await valkey_db.upsert_metadata("test_key", "test_value")
    metadata = await valkey_db.read_metadata()
    assert metadata["test_key"] == "test_value"
    await valkey_db.remove_metadata("test_key")


@pytest.mark.asyncio
async def test_that_valkey_db_remove_metadata_deletes_key(
    valkey_db: ValkeyVectorDatabase,
) -> None:
    await valkey_db.upsert_metadata("to_remove", 42)
    await valkey_db.remove_metadata("to_remove")
    metadata = await valkey_db.read_metadata()
    assert "to_remove" not in metadata


@pytest.mark.asyncio
async def test_that_valkey_db_read_metadata_returns_all(
    valkey_db: ValkeyVectorDatabase,
) -> None:
    await valkey_db.upsert_metadata("key1", "val1")
    await valkey_db.upsert_metadata("key2", 123)
    metadata = await valkey_db.read_metadata()
    assert metadata["key1"] == "val1"
    assert metadata["key2"] == 123
    await valkey_db.remove_metadata("key1")
    await valkey_db.remove_metadata("key2")


@pytest.mark.asyncio
async def test_that_valkey_collection_insert_one_stores_document(
    valkey_collection,
) -> None:
    doc: BaseDocument = {
        "id": ObjectId("doc1"),
        "version": "1.0.0",
        "checksum": "abc",
        "content": "hello world",
    }
    result = await valkey_collection.insert_one(doc)
    assert result.acknowledged

    found = await valkey_collection.find_one({"id": {"$eq": "doc1"}})
    assert found is not None
    assert found["id"] == "doc1"
    assert found["content"] == "hello world"


@pytest.mark.asyncio
async def test_that_valkey_collection_find_one_with_eq_filter_returns_match(
    valkey_collection,
) -> None:
    doc: BaseDocument = {
        "id": ObjectId("find_eq"),
        "version": "1.0.0",
        "checksum": "xyz",
        "content": "findable content",
    }
    await valkey_collection.insert_one(doc)
    found = await valkey_collection.find_one({"checksum": {"$eq": "xyz"}})
    assert found is not None
    assert found["id"] == "find_eq"


@pytest.mark.asyncio
async def test_that_valkey_collection_find_with_and_filter_returns_matches(
    valkey_collection,
) -> None:
    doc: BaseDocument = {
        "id": ObjectId("and_test"),
        "version": "2.0.0",
        "checksum": "and_chk",
        "content": "and filter test",
    }
    await valkey_collection.insert_one(doc)
    results = await valkey_collection.find(
        {"$and": [{"version": {"$eq": "2.0.0"}}, {"checksum": {"$eq": "and_chk"}}]}
    )
    assert len(results) >= 1
    assert any(r["id"] == "and_test" for r in results)


@pytest.mark.asyncio
async def test_that_valkey_collection_find_with_or_filter_returns_matches(
    valkey_collection,
) -> None:
    doc1: BaseDocument = {
        "id": ObjectId("or_a"),
        "version": "1.0.0",
        "checksum": "or1",
        "content": "or test a",
    }
    doc2: BaseDocument = {
        "id": ObjectId("or_b"),
        "version": "1.0.0",
        "checksum": "or2",
        "content": "or test b",
    }
    await valkey_collection.insert_one(doc1)
    await valkey_collection.insert_one(doc2)
    results = await valkey_collection.find(
        {"$or": [{"checksum": {"$eq": "or1"}}, {"checksum": {"$eq": "or2"}}]}
    )
    ids = [r["id"] for r in results]
    assert "or_a" in ids
    assert "or_b" in ids


@pytest.mark.asyncio
async def test_that_valkey_collection_update_one_modifies_document(
    valkey_collection,
) -> None:
    doc: BaseDocument = {
        "id": ObjectId("update_me"),
        "version": "1.0.0",
        "checksum": "old",
        "content": "original content",
    }
    await valkey_collection.insert_one(doc)
    result = await valkey_collection.update_one(
        {"id": {"$eq": "update_me"}},
        BaseDocument(id=ObjectId("update_me"), version="2.0.0", checksum="new", content="updated"),
    )
    assert result.matched_count == 1
    found = await valkey_collection.find_one({"id": {"$eq": "update_me"}})
    assert found is not None
    assert found["content"] == "updated"
    assert found["checksum"] == "new"


@pytest.mark.asyncio
async def test_that_valkey_collection_delete_one_removes_document(
    valkey_collection,
) -> None:
    doc: BaseDocument = {
        "id": ObjectId("delete_me"),
        "version": "1.0.0",
        "checksum": "del",
        "content": "to be deleted",
    }
    await valkey_collection.insert_one(doc)
    result = await valkey_collection.delete_one({"id": {"$eq": "delete_me"}})
    assert result.deleted_count == 1
    found = await valkey_collection.find_one({"id": {"$eq": "delete_me"}})
    assert found is None


@pytest.mark.asyncio
async def test_that_valkey_collection_find_similar_documents_returns_nearest(
    valkey_collection,
) -> None:
    docs = [
        BaseDocument(
            id=ObjectId(f"sim_{i}"),
            version="1.0.0",
            checksum=f"chk_{i}",
            content=f"document about topic {i}",
        )
        for i in range(5)
    ]
    for doc in docs:
        await valkey_collection.insert_one(doc)

    results = await valkey_collection.find_similar_documents(
        filters={},
        query="document about topic 0",
        k=3,
    )
    assert len(results) > 0
    assert len(results) <= 3
    # Verify the matching doc appears in results
    result_ids = [r.document["id"] for r in results]
    assert "sim_0" in result_ids


@pytest.mark.asyncio
async def test_that_valkey_collection_find_similar_documents_with_filter_narrows_results(
    valkey_collection,
) -> None:
    doc_a: BaseDocument = {
        "id": ObjectId("filter_sim_a"),
        "version": "1.0.0",
        "checksum": "fa",
        "content": "alpha topic content",
    }
    doc_b: BaseDocument = {
        "id": ObjectId("filter_sim_b"),
        "version": "2.0.0",
        "checksum": "fb",
        "content": "alpha topic content similar",
    }
    await valkey_collection.insert_one(doc_a)
    await valkey_collection.insert_one(doc_b)

    results = await valkey_collection.find_similar_documents(
        filters={"version": {"$eq": "2.0.0"}},
        query="alpha topic content",
        k=5,
    )
    # Only doc_b should match the version filter
    ids = [r.document["id"] for r in results]
    assert "filter_sim_b" in ids
    assert "filter_sim_a" not in ids
