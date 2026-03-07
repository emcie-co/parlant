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
from typing import Any
from unittest.mock import MagicMock, patch

from typing_extensions import override

from parlant.core.nlp.embedding import BaseEmbedder, EmbeddingResult
from parlant.core.nlp.tokenization import ZeroEstimatingTokenizer, EstimatingTokenizer


class _StubEmbedder(BaseEmbedder):
    """Minimal embedder for testing cache behavior."""

    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        return EmbeddingResult(vectors=[[0.0] for _ in texts])

    @property
    @override
    def id(self) -> str:
        return "stub"

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return ZeroEstimatingTokenizer()

    @property
    @override
    def dimensions(self) -> int:
        return 1


def _make_embedder() -> _StubEmbedder:
    logger = MagicMock()
    tracer = MagicMock()
    meter = MagicMock()
    meter.create_duration_histogram.return_value = MagicMock()
    return _StubEmbedder(logger=logger, tracer=tracer, meter=meter, model_name="stub")


@patch("parlant.core.nlp.embedding._EMBEDDING_CACHE_MAX_SIZE", 3)
async def test_that_cache_eviction_does_not_raise_when_entries_share_text_length() -> (
    None
):
    """Regression test for https://github.com/emcie-co/parlant/issues/731.

    When two cached texts share the same character length, evicting the older
    entry must not remove the length-index key that the remaining entry still
    needs. Previously, the eviction path unconditionally deleted the key,
    causing a KeyError on the next eviction of a same-length entry.
    """
    embedder = _make_embedder()

    # Insert three entries; two share text length 3.
    embedder._cache_put("aaa", [1.0])  # length=3, oldest
    embedder._cache_put("bbb", [2.0])  # length=3
    embedder._cache_put("cc", [3.0])  # length=2

    # Cache is full (3 entries). Next insert evicts "aaa" (oldest).
    # With the bug, this would delete _cache_length_index[3] entirely,
    # orphaning "bbb".
    embedder._cache_put("dddd", [4.0])  # length=4, triggers eviction of "aaa"

    # "bbb" (length=3) must still be in the cache.
    # Check via the internal dict to avoid _cache_get moving it to the end.
    assert embedder._compute_checksum("bbb") in embedder._cache
    assert embedder._compute_checksum("aaa") not in embedder._cache

    # Now evict "bbb" (it is the oldest remaining entry).
    # With the bug, this line would raise KeyError because the length-index
    # key 3 was already deleted during the first eviction.
    embedder._cache_put("eeeee", [5.0])  # length=5, triggers eviction of "bbb"

    assert embedder._cache_get("bbb") is None
    assert embedder._cache_get("cc") == [3.0]


@patch("parlant.core.nlp.embedding._EMBEDDING_CACHE_MAX_SIZE", 3)
async def test_that_cache_eviction_cleans_up_length_index_for_unique_lengths() -> (
    None
):
    embedder = _make_embedder()

    # Insert entries each with a unique text length.
    embedder._cache_put("a", [1.0])  # length=1
    embedder._cache_put("bb", [2.0])  # length=2
    embedder._cache_put("ccc", [3.0])  # length=3

    # Evict "a" (length=1, the only entry with that length).
    embedder._cache_put("dddd", [4.0])

    # Length 1 should be fully removed from the index.
    assert 1 not in embedder._cache_length_index
    assert embedder._cache_get("a") is None
    assert embedder._cache_get("bb") == [2.0]
