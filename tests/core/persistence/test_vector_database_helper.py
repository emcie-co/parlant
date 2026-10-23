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

from typing import Any, Mapping

from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.persistence.vector_database_helper import query_chunks


class _WordTokenizer(EstimatingTokenizer):
    """A trivial tokenizer that counts whitespace-delimited words as tokens."""

    async def estimate_token_count(self, prompt: str) -> int:
        return len(prompt.split())


class _FakeEmbedder(Embedder):
    @property
    def id(self) -> str:
        return "fake"

    @property
    def max_tokens(self) -> int:
        return 100

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return _WordTokenizer()

    @property
    def dimensions(self) -> int:
        return 8

    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        return EmbeddingResult(vectors=[[0.0] * self.dimensions for _ in texts])


async def test_that_query_chunks_returns_empty_for_a_whitespace_only_query() -> None:
    # A query with no words must not divide by a zero word count (regression).
    assert await query_chunks("   \n\t ", _FakeEmbedder()) == []


async def test_that_query_chunks_returns_empty_for_an_empty_query() -> None:
    assert await query_chunks("", _FakeEmbedder()) == []


async def test_that_query_chunks_chunks_a_normal_query() -> None:
    # max_tokens // 5 = 20 token budget, 1 token/word here → 20 words/chunk.
    query = " ".join(f"w{i}" for i in range(45))
    chunks = await query_chunks(query, _FakeEmbedder())
    assert len(chunks) == 3
    assert chunks[0].split()[:2] == ["w0", "w1"]
