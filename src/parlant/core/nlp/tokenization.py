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

from abc import ABC, abstractmethod
from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _gpt5_encoding() -> tiktoken.Encoding:
    """The gpt-5 (o200k_base) encoding — a fast, provider-agnostic local estimator."""
    return tiktoken.encoding_for_model("gpt-5")


def estimate_tokens(text: str) -> int:
    """A fast, provider-agnostic local token estimate (gpt-5 / o200k_base).

    Used where a provider does not report a token count (e.g. Gemini embeddings)
    or where a per-item split of a batch total is needed.
    """
    return len(_gpt5_encoding().encode(text))


class EstimatingTokenizer(ABC):
    """An interface for estimating the token count of a prompt."""

    @abstractmethod
    async def estimate_token_count(self, prompt: str) -> int:
        """Estimate the number of tokens in the given prompt."""
        ...


class ZeroEstimatingTokenizer(EstimatingTokenizer):
    """A tokenizer that always returns zero for token count estimation."""

    async def estimate_token_count(self, prompt: str) -> int:
        return 0
