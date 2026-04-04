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

import pytest
from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

from typing_extensions import override

from parlant.core.nlp.embedding import (
    BaseEmbedder,
    EmbeddingResult,
    _EMBEDDING_CACHE_MAX_SIZE,
)
from parlant.core.nlp.tokenization import EstimatingTokenizer, ZeroEstimatingTokenizer


class FakeEmbedder(BaseEmbedder):
    """A minimal concrete BaseEmbedder for testing cache behavior."""

    def __init__(self) -> None:
        logger = MagicMock()
        tracer = MagicMock()
        meter = MagicMock()
        meter.create_duration_histogram = MagicMock(return_value=MagicMock())
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name="fake")
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
