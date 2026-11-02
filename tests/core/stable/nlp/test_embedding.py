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

import asyncio

import pytest
from collections.abc import Mapping
from typing import Any

from typing_extensions import override

from parlant.core.loggers import LogLevel, StdoutLogger
from parlant.core.meter import LocalMeter
from parlant.core.nlp.embedding import (
    BaseEmbedder,
    EmbeddingResult,
    chunk_to_embedder_budget,
    _EMBEDDING_BUDGET_HEADROOM,
    _EMBEDDING_CACHE_MAX_SIZE,
)
from parlant.core.nlp.common import UsageInfo
from parlant.core.nlp.tokenization import (
    EstimatingTokenizer,
    ZeroEstimatingTokenizer,
    estimate_tokens,
)
from parlant.core.tracer import LocalTracer
from parlant.core.usage_reporter import UsageReporter


class FakeEmbedder(BaseEmbedder):
    """A minimal concrete BaseEmbedder for testing cache behavior.

    Its ``do_embed`` reports no provider token count (like Gemini embeddings),
    so the cache falls back to the gpt-5 estimator for cached-token accounting.
    """

    def __init__(self, usage_reporter: UsageReporter | None = None) -> None:
        tracer = LocalTracer()
        logger = StdoutLogger(tracer=tracer, log_level=LogLevel.INFO)
        meter = LocalMeter(logger)
        super().__init__(
            logger=logger,
            tracer=tracer,
            meter=meter,
            model_name="fake",
            usage_reporter=usage_reporter,
            provider_name="test",
        )
        self._tokenizer = ZeroEstimatingTokenizer()
        self.do_embed_call_count = 0

    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        self.do_embed_call_count += 1
        return EmbeddingResult(vectors=[[float(len(t))] for t in texts])

    @property
    @override
    def id(self) -> str:
        return "fake"

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
        return 1


def _make_unique_text(length: int, index: int) -> str:
    """Generate a unique text of the exact given length using an index suffix."""
    suffix = f"_{index:02d}"
    assert length > len(suffix), "length must be long enough to fit the suffix"
    return "a" * (length - len(suffix)) + suffix


_PROVIDER_TOKENS_PER_TEXT = 10


class _ProviderCountingEmbedder(FakeEmbedder):
    """do_embed reports a provider input-token total of 10 per text in the batch."""

    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        self.do_embed_call_count += 1
        return EmbeddingResult(
            vectors=[[float(len(t))] for t in texts],
            usage=UsageInfo(
                input_tokens=_PROVIDER_TOKENS_PER_TEXT * len(texts),
                output_tokens=0,
            ),
        )


@pytest.mark.asyncio
async def test_that_a_cache_served_embedding_still_counts_as_one_request() -> None:
    embedder = FakeEmbedder()

    uncached = await embedder.embed(["hello"])
    cached = await embedder.embed(["hello"])

    assert uncached.usage.requests == 1
    # A cache hit is served without a provider call, but it is still a request the
    # system made — so it counts as one (visible in the usage report).
    assert cached.usage.requests == 1


@pytest.mark.asyncio
async def test_that_a_full_cache_hit_reports_input_tokens_equal_to_cached_input_tokens() -> None:
    embedder = _ProviderCountingEmbedder()

    await embedder.embed(["hello"])
    cached = await embedder.embed(["hello"])

    # A fully cached response still carries the token count, marked entirely as cached.
    assert cached.usage.input_tokens == _PROVIDER_TOKENS_PER_TEXT
    assert cached.usage.cached_input_tokens == _PROVIDER_TOKENS_PER_TEXT
    # The cache-served call still counts as one request.
    assert cached.usage.requests == 1


@pytest.mark.asyncio
async def test_that_a_partial_cache_hit_counts_both_cached_and_uncached_input_tokens() -> None:
    embedder = _ProviderCountingEmbedder()

    await embedder.embed(["alpha"])  # caches "alpha"
    partial = await embedder.embed(["alpha", "beta"])  # "alpha" hit, "beta" miss

    # input_tokens covers BOTH the cached and the freshly-embedded slice;
    # cached_input_tokens is just the cached slice.
    assert partial.usage.input_tokens == 2 * _PROVIDER_TOKENS_PER_TEXT
    assert partial.usage.cached_input_tokens == _PROVIDER_TOKENS_PER_TEXT


@pytest.mark.asyncio
async def test_that_cached_tokens_fall_back_to_the_gpt5_estimate_when_the_provider_reports_none() -> (
    None
):
    # FakeEmbedder reports no provider tokens (like Gemini); the cache estimates
    # via gpt-5 so cached tokens are still counted rather than lost.
    embedder = FakeEmbedder()
    text = "hello there world"

    await embedder.embed([text])
    cached = await embedder.embed([text])

    expected = estimate_tokens(text)
    assert expected > 0
    assert cached.usage.cached_input_tokens == expected
    assert cached.usage.input_tokens == expected


class _GatedEmbedder(FakeEmbedder):
    """do_embed blocks on a release event, so a test can force two embed() calls
    to overlap in flight before either returns."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.started = asyncio.Event()

    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        self.do_embed_call_count += 1
        self.started.set()
        await self.release.wait()
        return EmbeddingResult(
            vectors=[[float(len(t))] for t in texts],
            usage=UsageInfo(input_tokens=_PROVIDER_TOKENS_PER_TEXT * len(texts), output_tokens=0),
        )


@pytest.mark.asyncio
async def test_that_concurrent_identical_embeds_trigger_a_single_provider_call() -> None:
    embedder = _GatedEmbedder()

    leader = asyncio.ensure_future(embedder.embed(["same text"]))
    await embedder.started.wait()  # leader is now in flight (registered)
    follower = asyncio.ensure_future(embedder.embed(["same text"]))
    await asyncio.sleep(0)  # let the follower classify and coalesce

    embedder.release.set()
    leader_result, follower_result = await asyncio.gather(leader, follower)

    # Only one provider call for the two concurrent identical embeds.
    assert embedder.do_embed_call_count == 1
    assert leader_result.vectors == follower_result.vectors
    # The coalesced follower is a cache hit — its tokens are cached, not billed again.
    assert follower_result.usage.input_tokens == _PROVIDER_TOKENS_PER_TEXT
    assert follower_result.usage.cached_input_tokens == _PROVIDER_TOKENS_PER_TEXT
    # The in-flight map is drained afterward.
    assert embedder._inflight == {}


@pytest.mark.asyncio
async def test_that_the_inflight_map_is_cleared_after_a_normal_embed() -> None:
    embedder = FakeEmbedder()

    await embedder.embed(["a", "b"])

    assert embedder._inflight == {}


@pytest.mark.asyncio
async def test_that_a_failed_embed_clears_inflight_and_wakes_awaiters() -> None:
    class _BoomEmbedder(_GatedEmbedder):
        @override
        async def do_embed(
            self,
            texts: list[str],
            hints: Mapping[str, Any] = {},
        ) -> EmbeddingResult:
            self.do_embed_call_count += 1
            self.started.set()
            await self.release.wait()
            raise RuntimeError("provider boom")

    embedder = _BoomEmbedder()

    leader = asyncio.ensure_future(embedder.embed(["same text"]))
    await embedder.started.wait()
    follower = asyncio.ensure_future(embedder.embed(["same text"]))
    await asyncio.sleep(0)

    embedder.release.set()

    # Both the leader and the coalesced follower see the failure (neither hangs),
    # and the in-flight map is drained so a later embed can retry cleanly.
    with pytest.raises(RuntimeError):
        await leader
    with pytest.raises(RuntimeError):
        await follower
    assert embedder._inflight == {}


class _ShortchangingEmbedder(FakeEmbedder):
    """A provider that returns one vector fewer than the texts it was sent."""

    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        result = await super().do_embed(texts, hints)
        return EmbeddingResult(vectors=result.vectors[:-1], usage=result.usage)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "texts",
    [
        ["a", "b"],
        # The repeated "b" waits on its first copy's result, which never arrives.
        ["a", "b", "b"],
    ],
)
async def test_that_an_embedder_returning_fewer_vectors_than_texts_raises_instead_of_hanging(
    texts: list[str],
) -> None:
    embedder = _ShortchangingEmbedder()

    with pytest.raises(Exception) as error:
        await asyncio.wait_for(embedder.embed(texts), timeout=2)

    assert not isinstance(error.value, TimeoutError), "embed() hung"
    assert "fake" in str(error.value)
    assert "returned 1 vector" in str(error.value)
    assert embedder._inflight == {}


@pytest.mark.asyncio
async def test_that_a_cache_hit_is_reported_to_the_usage_reporter() -> None:
    reporter = UsageReporter(LocalTracer())
    embedder = _ProviderCountingEmbedder(usage_reporter=reporter)

    await embedder.embed(["hello"])  # miss: input=10, cached=0
    await embedder.embed(["hello"])  # hit: input=10, cached=10

    reported = reporter.get_usage()["test:fake"]
    # The hit contributed its cached tokens through the reporter (the miss reports none).
    assert reported.cached_input_tokens == _PROVIDER_TOKENS_PER_TEXT


@pytest.mark.asyncio
async def test_that_cache_eviction_preserves_entries_with_the_same_text_length() -> None:
    embedder = FakeEmbedder()

    # Embed two texts that share the same length (10 chars).
    # These are embedded first, so they'll be the oldest in the LRU cache.
    text_a = _make_unique_text(10, 0)
    text_b = _make_unique_text(10, 1)

    await embedder.embed([text_a])
    await embedder.embed([text_b])

    # Fill the rest of the cache to capacity with unique-length filler texts.
    for i in range(_EMBEDDING_CACHE_MAX_SIZE - 2):
        filler = "x" * (20 + i)
        await embedder.embed([filler])

    # Trigger eviction of the oldest entry (text_a) by adding one more.
    await embedder.embed(["trigger_eviction!"])

    # Reset the call count so we can observe whether text_b hits the cache.
    embedder.do_embed_call_count = 0

    # Embed text_b again. Since it was NOT evicted, this should be a cache hit
    # and do_embed should not be called.
    await embedder.embed([text_b])

    assert embedder.do_embed_call_count == 0, (
        "Expected text_b to be served from cache, but do_embed was called. "
        "Evicting text_a (same text length) incorrectly invalidated text_b's cache entry."
    )


class _WordCountTokenizer(EstimatingTokenizer):
    @override
    async def estimate_token_count(self, prompt: str) -> int:
        return len(prompt.split())


class _BudgetedFakeEmbedder(FakeEmbedder):
    """A FakeEmbedder with a configurable token cap and a word-count tokenizer."""

    def __init__(self, max_tokens: int) -> None:
        super().__init__()
        self._budget_max_tokens = max_tokens
        self._word_tokenizer = _WordCountTokenizer()

    @property
    @override
    def max_tokens(self) -> int:
        return self._budget_max_tokens

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._word_tokenizer


@pytest.mark.asyncio
async def test_that_text_within_the_embedder_token_budget_is_returned_verbatim_as_a_single_chunk() -> (
    None
):
    embedder = _BudgetedFakeEmbedder(max_tokens=100)
    # Whitespace layout must survive untouched - chunk texts are embedding-cache
    # keys, so the in-budget case has to be byte-identical.
    text = "line one\nline two  with   spacing"

    assert await chunk_to_embedder_budget(text, embedder) == [text]


@pytest.mark.asyncio
async def test_that_text_exceeding_the_embedder_token_budget_is_split_into_chunks_each_within_budget() -> (
    None
):
    embedder = _BudgetedFakeEmbedder(max_tokens=16)
    budget = int(16 * _EMBEDDING_BUDGET_HEADROOM)
    words = [f"word{i}" for i in range(30)]

    chunks = await chunk_to_embedder_budget(" ".join(words), embedder)

    assert len(chunks) > 1
    assert all(len(chunk.split()) <= budget for chunk in chunks)
    # No content is lost across the chunk boundaries.
    assert " ".join(chunks).split() == words


@pytest.mark.asyncio
async def test_that_an_empty_text_yields_no_chunks() -> None:
    embedder = _BudgetedFakeEmbedder(max_tokens=16)

    assert await chunk_to_embedder_budget("", embedder) == []
    assert await chunk_to_embedder_budget("   \n", embedder) == []


@pytest.mark.asyncio
async def test_that_a_zero_estimating_tokenizer_never_triggers_chunking() -> None:
    # FakeEmbedder carries a ZeroEstimatingTokenizer: an estimate of zero must be
    # treated as in-budget, never as a division-by-zero or a chunking trigger.
    embedder = FakeEmbedder()
    text = "word " * 100_000

    assert await chunk_to_embedder_budget(text, embedder) == [text]
