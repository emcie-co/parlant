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

from collections.abc import Mapping
from typing import Any, cast

from lagom import Container
import pytest

from parlant.core.canned_responses import (
    CannedResponseStore,
    CannedResponseVectorStore,
)
from parlant.core.nlp.embedding import EmbeddingResult


def _stub_embedder(store: CannedResponseVectorStore) -> None:
    dimensions = store._canreps_vector_collection._embedder.dimensions  # type: ignore[attr-defined]

    async def embed(
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[[float((len(text) + i) % 13) for i in range(dimensions)] for text in texts]
        )

    store._canreps_vector_collection._embedder.embed = embed  # type: ignore[attr-defined, method-assign]


@pytest.mark.asyncio
async def test_that_canned_response_vector_documents_use_unique_ids_for_each_content(
    container: Container,
) -> None:
    store = cast(CannedResponseVectorStore, container[CannedResponseStore])
    _stub_embedder(store)

    canrep = await store.create_canned_response(
        value="Payment failed.",
        signals=["card declined", "card declined", "Payment failed."],
    )

    vector_docs = await store._canreps_vector_collection.find(  # type: ignore[attr-defined]
        filters={"canned_response_id": {"$eq": canrep.id}}
    )

    assert len(vector_docs) == 4
    assert len({doc["id"] for doc in vector_docs}) == 4


@pytest.mark.asyncio
async def test_that_updating_a_canned_response_replaces_its_vector_documents(
    container: Container,
) -> None:
    store = cast(CannedResponseVectorStore, container[CannedResponseStore])
    _stub_embedder(store)

    canrep = await store.create_canned_response(
        value="Payment failed.",
        signals=["card declined", "billing issue"],
    )

    original_vector_docs = await store._canreps_vector_collection.find(  # type: ignore[attr-defined]
        filters={"canned_response_id": {"$eq": canrep.id}}
    )
    assert {doc["content"] for doc in original_vector_docs} == {
        "Payment failed.",
        "card declined",
        "billing issue",
    }

    await store.update_canned_response(
        canrep.id,
        {"value": "Transaction declined.", "signals": ["insufficient funds"]},
    )

    updated_vector_docs = await store._canreps_vector_collection.find(  # type: ignore[attr-defined]
        filters={"canned_response_id": {"$eq": canrep.id}}
    )
    assert {doc["content"] for doc in updated_vector_docs} == {
        "Transaction declined.",
        "insufficient funds",
    }
