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

from collections import Counter
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from parlant.adapters.db.json_file import JSONFileDocumentDatabase
from parlant.adapters.vector_db.chroma import ChromaDatabase
from parlant.core.canned_responses import CannedResponseId, CannedResponseVectorStore
from parlant.core.common import IdGenerator
from parlant.core.loggers import CompositeLogger
from parlant.core.nlp.embedding import Embedder, NullEmbedder, NullEmbeddingCache
from parlant.core.tracer import LocalTracer


async def _embedder() -> Embedder:
    # Identity/persistence is under test; use a real local embedding adapter.
    return NullEmbedder()


@asynccontextmanager
async def _store(path: Path) -> AsyncIterator[CannedResponseVectorStore]:
    logger = CompositeLogger([])
    async with AsyncExitStack() as stack:
        documents = await stack.enter_async_context(
            JSONFileDocumentDatabase(logger, path / "documents.json")
        )
        vectors = await stack.enter_async_context(
            ChromaDatabase(logger, LocalTracer(), path / "vectors", NullEmbeddingCache)
        )
        yield await stack.enter_async_context(
            CannedResponseVectorStore(IdGenerator(), vectors, documents, _embedder)
        )


async def _contents(store: CannedResponseVectorStore, id: CannedResponseId) -> Counter[str]:
    documents = await store._canreps_vector_collection.find({"canned_response_id": {"$eq": id}})
    return Counter(document["content"] for document in documents)


async def test_that_canned_response_vector_documents_use_unique_ids_for_each_content(
    tmp_path: Path,
) -> None:
    async with _store(tmp_path) as store:
        response = await store.create_canned_response(
            value="Payment failed.",
            signals=["card declined", "card declined", "Payment failed."],
        )
        documents = await store._canreps_vector_collection.find(
            {"canned_response_id": {"$eq": response.id}}
        )
        assert len(documents) == 4
        assert len({document["id"] for document in documents}) == 4
        assert await _contents(store, response.id) == Counter(
            {"Payment failed.": 2, "card declined": 2}
        )
        assert (await store.read_canned_response(response.id)).id == response.id


async def test_that_updating_a_canned_response_replaces_its_vector_documents(
    tmp_path: Path,
) -> None:
    async with _store(tmp_path) as store:
        response = await store.create_canned_response(
            value="Payment failed.", signals=["card declined", "billing issue"]
        )
        await store.update_canned_response(
            response.id,
            {"value": "Transaction declined.", "signals": ["insufficient funds"]},
        )
        assert await _contents(store, response.id) == Counter(
            ["Transaction declined.", "insufficient funds"]
        )


async def test_that_reopening_a_canned_response_store_preserves_each_signal_vector(
    tmp_path: Path,
) -> None:
    async with _store(tmp_path) as store:
        response = await store.create_canned_response(
            value="Payment failed.", signals=["card declined", "billing issue"]
        )
    async with _store(tmp_path) as reopened:
        assert (await reopened.read_canned_response(response.id)).signals == response.signals
        assert await _contents(reopened, response.id) == Counter(
            ["Payment failed.", "card declined", "billing issue"]
        )


async def test_that_deleting_a_canned_response_removes_all_its_vectors(tmp_path: Path) -> None:
    async with _store(tmp_path) as store:
        response = await store.create_canned_response(
            value="Payment failed.", signals=["card declined", "billing issue"]
        )
        await store.delete_canned_response(response.id)
        assert await _contents(store, response.id) == Counter()
