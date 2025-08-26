# Copyright 2025 Emcie Co Ltd.
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

import time
from pydantic import ValidationError
from together import AsyncTogether  # type: ignore
from together.error import (  # type: ignore
    RateLimitError,
    Timeout,
    APIConnectionError,
    APIError,
    ServiceUnavailableError,
)
from typing import Any, Mapping
from typing_extensions import override
import jsonfinder  # type: ignore
import os
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output
from parlant.adapters.nlp.hugging_face import HuggingFaceEstimatingTokenizer
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    SchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.loggers import Logger
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.tokenization import EstimatingTokenizer

RATE_LIMIT_ERROR_MESSAGE = (
    "Together API rate limit exceeded. Possible reasons:\n"
    "1. Your account may have insufficient API credits.\n"
    "2. You may be using a free-tier account with limited request capacity.\n"
    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your Together account balance and billing status.\n"
    "- Review your API usage limits in Together's dashboard.\n"
    "- For more details on rate limits and usage tiers, visit:\n"
    "  https://docs.together.ai/docs/rate-limits"
)


class LlamaEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self) -> None:
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens) + 36


class TogetherAISchematicGenerator(SchematicGenerator[T]):
    supported_hints = ["temperature"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
    ) -> None:
        self.model_name = model_name
        self._logger = logger
        self._client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

    @policy(
        [
            retry(
                exceptions=(
                    RateLimitError,
                    Timeout,
                    APIConnectionError,
                    APIError,
                )
            ),
            retry(ServiceUnavailableError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        together_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        t_start = time.time()
        try:
            response = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format={"type": "json_object"},
                **together_api_arguments,
            )
        except RateLimitError:
            self._logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise

        t_end = time.time()

        raw_content = response.choices[0].message.content or "{}"

        try:
            json_content = normalize_json_output(raw_content)
            json_object = jsonfinder.only_json(json_content)[2]
        except Exception:
            self._logger.error(
                f"Failed to extract JSON returned by {self.model_name}:\n{raw_content}"
            )
            raise

        try:
            model_content = self.schema.model_validate(json_object)

            return SchematicGenerationResult(
                content=model_content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        extra={},
                    ),
                ),
            )
        except ValidationError:
            self._logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class Llama3_1_8B(TogetherAISchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            logger=logger,
        )
        self._estimating_tokenizer = LlamaEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        return self.model_name

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024

    @property
    @override
    def tokenizer(self) -> LlamaEstimatingTokenizer:
        return self._estimating_tokenizer


class Llama3_1_70B(TogetherAISchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            logger=logger,
        )

        self._estimating_tokenizer = LlamaEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        return self.model_name

    @property
    @override
    def tokenizer(self) -> LlamaEstimatingTokenizer:
        return self._estimating_tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class Llama3_1_405B(TogetherAISchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            logger=logger,
        )

        self._estimating_tokenizer = LlamaEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        return self.model_name

    @property
    @override
    def tokenizer(self) -> LlamaEstimatingTokenizer:
        return self._estimating_tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class Llama3_3_70B(TogetherAISchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            logger=logger,
        )

        self._estimating_tokenizer = LlamaEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        return self.model_name

    @property
    @override
    def tokenizer(self) -> LlamaEstimatingTokenizer:
        return self._estimating_tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class TogetherAIEmbedder(Embedder):
    def __init__(self, model_name: str, logger: Logger) -> None:
        self.model_name = model_name

        self._logger = logger
        self._client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

    @policy(
        [
            retry(
                exceptions=(
                    RateLimitError,
                    Timeout,
                    APIConnectionError,
                    APIError,
                )
            ),
            retry(ServiceUnavailableError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        _ = hints

        try:
            response = await self._client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
        except RateLimitError:
            self._logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise

        vectors = [data_point.embedding for data_point in response.data]
        return EmbeddingResult(vectors=vectors)


class M2Bert32K(TogetherAIEmbedder):
    def __init__(self, logger: Logger) -> None:
        super().__init__(model_name="togethercomputer/m2-bert-80M-32k-retrieval", logger=logger)
        self._estimating_tokenizer = HuggingFaceEstimatingTokenizer(self.model_name)

    @property
    @override
    def id(self) -> str:
        return self.model_name

    @property
    @override
    def max_tokens(self) -> int:
        return 32 * 1024

    @property
    @override
    def tokenizer(self) -> HuggingFaceEstimatingTokenizer:
        return self._estimating_tokenizer

    @property
    @override
    def dimensions(self) -> int:
        return 768


class TogetherService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("TOGETHER_API_KEY"):
            return """\
You're using the OpenAI NLP service, but TOGETHER_API_KEY is not set.
Please set TOGETHER_API_KEY in your environment before running Parlant.
"""

        return None

    def __init__(
        self,
        logger: Logger,
    ) -> None:
        self._logger = logger
        self._logger.info("Initialized TogetherService")

    @override
    async def get_schematic_generator(self, t: type[T]) -> TogetherAISchematicGenerator[T]:
        return Llama3_3_70B[t](self._logger)  # type: ignore

    @override
    async def get_embedder(self) -> Embedder:
        return M2Bert32K(self._logger)

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
