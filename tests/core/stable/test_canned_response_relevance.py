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

from contextlib import asynccontextmanager
from typing import AsyncIterator, cast

import pytest
from typing_extensions import override

from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.adapters.vector_db.transient import TransientVectorDatabase

from parlant.core.canned_responses import (
    CannedResponse,
    CannedResponseStore,
    CannedResponseVectorStore,
)
from parlant.core.common import IdGenerator
from parlant.core.container import Container
from parlant.core.loggers import CompositeLogger
from parlant.core.nlp.embedding import Embedder, NullEmbeddingCache
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.persistence.vector_database_helper import query_chunks
from parlant.core.tracer import LocalTracer
from tests.test_utilities import DeterministicEmbedder


class _WordCountingTokenizer(EstimatingTokenizer):
    @override
    async def estimate_token_count(self, prompt: str) -> int:
        return len(prompt.split())


class _ChunkedDeterministicEmbedder(DeterministicEmbedder):
    @property
    @override
    def max_tokens(self) -> int:
        # query_chunks uses one fifth of this budget, giving two-word chunks.
        return 10

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return _WordCountingTokenizer()


@asynccontextmanager
async def _local_relevance_store() -> AsyncIterator[CannedResponseVectorStore]:
    logger = CompositeLogger([])
    embedder = _ChunkedDeterministicEmbedder()

    async def embedder_provider() -> Embedder:
        return embedder

    async with CannedResponseVectorStore(
        IdGenerator(),
        TransientVectorDatabase(logger, LocalTracer(), NullEmbeddingCache),
        TransientDocumentDatabase(),
        embedder_provider,
    ) as store:
        yield store


async def _create_chunked_responses(
    store: CannedResponseVectorStore,
) -> list[CannedResponse]:
    return [
        await store.create_canned_response(value="shipping tracking extra"),
        await store.create_canned_response(value="shipping extra"),
        await store.create_canned_response(value="refunds policy"),
    ]


async def test_that_canned_response_relevance_selects_the_closest_responses_across_all_query_chunks() -> (
    None
):
    async with _local_relevance_store() as store:
        first, second, later = await _create_chunked_responses(store)
        query = "shipping tracking refunds policy"
        assert await query_chunks(query, store._embedder) == [
            "shipping tracking",
            "refunds policy",
        ]

        results = await store.filter_relevant_canned_responses(
            query, [first, second, later], max_count=2
        )

        assert [result.canned_response.id for result in results] == [later.id, first.id]
        assert results[0].score == pytest.approx(1.0)
        assert results[0].score > results[1].score


async def test_that_canned_response_relevance_keeps_each_responses_best_score_across_query_chunks() -> (
    None
):
    async with _local_relevance_store() as store:
        later_improves = await store.create_canned_response(
            value="shipping extra", signals=["refunds policy"]
        )
        first_best = await store.create_canned_response(value="shipping tracking extra")

        results = await store.filter_relevant_canned_responses(
            "shipping tracking refunds policy", [later_improves, first_best], max_count=2
        )

        assert [result.canned_response.id for result in results] == [
            later_improves.id,
            first_best.id,
        ]
        assert results[0].score == pytest.approx(1.0)
        assert len({result.canned_response.id for result in results}) == 2


async def test_that_multi_chunk_canned_response_relevance_excludes_unavailable_responses() -> None:
    async with _local_relevance_store() as store:
        first, second, unavailable = await _create_chunked_responses(store)

        results = await store.filter_relevant_canned_responses(
            "shipping tracking refunds policy", [first, second], max_count=2
        )

        assert [result.canned_response.id for result in results] == [first.id, second.id]
        assert unavailable.id not in {result.canned_response.id for result in results}


async def test_that_multi_chunk_canned_response_relevance_preserves_score_alignment_after_a_response_is_missing() -> (
    None
):
    async with _local_relevance_store() as store:
        responses = await _create_chunked_responses(store)
        query = "shipping tracking refunds policy"
        before = await store.filter_relevant_canned_responses(query, responses, max_count=3)
        assert len(before) == 3
        scores = {result.canned_response.id: result.score for result in before}
        assert len(set(scores.values())) == 3

        missing = before[0].canned_response
        await store._canreps_collection.delete_one({"id": {"$eq": missing.id}})
        after = await store.filter_relevant_canned_responses(query, responses, max_count=3)

        assert [result.canned_response.id for result in after] == [
            result.canned_response.id
            for result in before
            if result.canned_response.id != missing.id
        ]
        assert len(after) == 2
        for result in after:
            assert result.score == scores[result.canned_response.id]


async def test_that_a_relevance_hit_with_no_stored_response_keeps_the_other_scores_on_their_own_responses(
    container: Container,
) -> None:
    """A vector hit can outlive the response it points at — the vector index and
    the document collection are written separately, so a response deleted from
    one can still be matched through the other.

    Each surviving response must keep its own score. Pairing the responses that
    were found against the full list of hits, position by position, hands every
    response after the missing one a score that belongs to its neighbour.
    """
    store = cast(CannedResponseVectorStore, container[CannedResponseStore])

    responses = [
        await store.create_canned_response(value="Your order has shipped and is on its way."),
        await store.create_canned_response(value="Our opening hours are nine to five."),
        await store.create_canned_response(value="I can help you reset your password."),
    ]

    query = "where is my package"

    before = await store.filter_relevant_canned_responses(query, responses, max_count=3)

    scores_before = {result.canned_response.id: result.score for result in before}

    assert len(before) == 3
    # Distinct scores, or a score landing on the wrong response is invisible.
    assert len(set(scores_before.values())) == 3

    # Delete the closest hit's document while leaving its vector in place. The
    # caller still offers it as available, exactly as a stale index would.
    missing = max(before, key=lambda result: result.score).canned_response

    await store._canreps_collection.delete_one({"id": {"$eq": missing.id}})

    after = await store.filter_relevant_canned_responses(query, responses, max_count=3)

    assert [result.canned_response.id for result in after] == [
        result.canned_response.id for result in before if result.canned_response.id != missing.id
    ]

    for result in after:
        assert result.score == scores_before[result.canned_response.id]
