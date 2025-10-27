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

from __future__ import annotations
import time
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AsyncClient,
    ConflictError,
    InternalServerError,
    RateLimitError,
)
from typing import Any, Mapping
from typing_extensions import override
import json
import jsonfinder  # type: ignore
import os

from pydantic import ValidationError
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    SchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.moderation import (
    ModerationService,
    NoModeration,
)

RATE_LIMIT_ERROR_MESSAGE = """\
OpenRouter API rate limit exceeded. Possible reasons:
1. Your account may have insufficient API credits.
2. You may be using a free-tier account with limited request capacity.
3. You might have exceeded the requests-per-minute limit for your account.

Recommended actions:
- Check your OpenRouter account balance and billing status.
- Review your API usage limits in OpenRouter's dashboard.
- For more details on rate limits and usage tiers, visit:
    https://openrouter.ai/docs/api-reference/limits
"""


class OpenRouterEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # Use gpt-4 encoding as default for token estimation
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class OpenRouterEmbedder(Embedder):
    supported_arguments = ["dimensions"]

    def __init__(self, model_name: str, logger: Logger) -> None:
        self.model_name = model_name
        self._logger = logger
        
        # Build extra headers from environment variables
        extra_headers = {}
        if os.environ.get("OPENROUTER_HTTP_REFERER"):
            extra_headers["HTTP-Referer"] = os.environ.get("OPENROUTER_HTTP_REFERER")
        if os.environ.get("OPENROUTER_SITE_NAME"):
            extra_headers["X-Title"] = os.environ.get("OPENROUTER_SITE_NAME")

        self._client = AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            default_headers=extra_headers if extra_headers else None,
        )
        self._tokenizer = OpenRouterEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"openrouter/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> OpenRouterEstimatingTokenizer:
        return self._tokenizer

    @policy(
        [
            retry(
                exceptions=(
                    APIConnectionError,
                    APITimeoutError,
                    ConflictError,
                    RateLimitError,
                    APIResponseValidationError,
                ),
            ),
            retry(InternalServerError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        filtered_hints = {k: v for k, v in hints.items() if k in self.supported_arguments}
        try:
            response = await self._client.embeddings.create(
                model=self.model_name,
                input=texts,
                **filtered_hints,
            )
        except RateLimitError:
            self._logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise

        vectors = [data_point.embedding for data_point in response.data]
        return EmbeddingResult(vectors=vectors)


class OpenRouterTextEmbedding(OpenRouterEmbedder):
    def __init__(self, model_name: str, logger: Logger) -> None:
        super().__init__(model_name=model_name, logger=logger)

    @property
    @override
    def max_tokens(self) -> int:
        return 8192


class OpenRouterSchematicGenerator(SchematicGenerator[T]):
    supported_openrouter_params = ["temperature", "max_tokens"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
    ) -> None:
        self.model_name = model_name
        self._logger = logger

        # Build extra headers from environment variables
        extra_headers = {}
        if os.environ.get("OPENROUTER_HTTP_REFERER"):
            extra_headers["HTTP-Referer"] = os.environ.get("OPENROUTER_HTTP_REFERER")
        if os.environ.get("OPENROUTER_SITE_NAME"):
            extra_headers["X-Title"] = os.environ.get("OPENROUTER_SITE_NAME")

        self._client = AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_headers=extra_headers if extra_headers else None,
        )

        self._tokenizer = OpenRouterEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"openrouter/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> OpenRouterEstimatingTokenizer:
        return self._tokenizer

    @policy(
        [
            retry(
                exceptions=(
                    APIConnectionError,
                    APITimeoutError,
                    ConflictError,
                    RateLimitError,
                    APIResponseValidationError,
                ),
            ),
            retry(InternalServerError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self._logger.operation(f"OpenRouter LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        openrouter_api_arguments = {
            k: v for k, v in hints.items() if k in self.supported_openrouter_params
        }

        t_start = time.time()
        try:
            response = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format={"type": "json_object"},
                **openrouter_api_arguments,
            )
        except RateLimitError:
            self._logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise
        
        t_end = time.time()

        if response.usage:
            self._logger.trace(response.usage.model_dump_json(indent=2))

        raw_content = response.choices[0].message.content or "{}"

        try:
            json_content = json.loads(normalize_json_output(raw_content))
        except json.JSONDecodeError:
            self._logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content})")
            json_content = jsonfinder.only_json(raw_content)[2]
            self._logger.warning("Found JSON content within model response; continuing...")

        try:
            content = self.schema.model_validate(json_content)

            assert response.usage

            return SchematicGenerationResult(
                content=content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        extra={
                            "cached_input_tokens": getattr(
                                response.usage,
                                "prompt_cache_hit_tokens",
                                0,
                            )
                        },
                    ),
                ),
            )
        except ValidationError:
            self._logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class OpenRouterGPT4O(OpenRouterSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(model_name="openai/gpt-4o", logger=logger)

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class OpenRouterGPT4OMini(OpenRouterSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(model_name="openai/gpt-4o-mini", logger=logger)

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class OpenRouterClaude35Sonnet(OpenRouterSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(model_name="anthropic/claude-3.5-sonnet", logger=logger)

    @property
    @override
    def max_tokens(self) -> int:
        return 8192


class OpenRouterLlama33_70B(OpenRouterSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(model_name="meta-llama/llama-3.3-70b-instruct", logger=logger)

    @property
    @override
    def max_tokens(self) -> int:
        return 8192


class OpenRouterService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("OPENROUTER_API_KEY"):
            return """\
You're using the OpenRouter NLP service, but OPENROUTER_API_KEY is not set.
Please set OPENROUTER_API_KEY in your environment before running Parlant.
"""

        return None

    def __init__(
        self,
        logger: Logger,
        model_name: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._logger = logger
        self._logger.info("Initialized OpenRouterService")
        # Use provided model_name or fall back to environment variable
        self.model_name = model_name or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")
        self._custom_max_tokens = max_tokens
        self._logger.info(f"OpenRouter model name: {self.model_name}")

    def _get_specialized_generator_class(
        self,
        model_name: str,
        t: type[T],
    ) -> type[OpenRouterSchematicGenerator[T]]:
        """
        Returns the specialized generator class for known models.
        For unknown models, creates a dynamic generator that works with any OpenRouter model.
        """
        model_mapping: dict[str, type[OpenRouterSchematicGenerator[T]]] = {
            "openai/gpt-4o": OpenRouterGPT4O[t],  # type: ignore
            "openai/gpt-4o-mini": OpenRouterGPT4OMini[t],  # type: ignore
            "anthropic/claude-3.5-sonnet": OpenRouterClaude35Sonnet[t],  # type: ignore
            "meta-llama/llama-3.3-70b-instruct": OpenRouterLlama33_70B[t],  # type: ignore
        }

        # Check if we have a predefined generator for this model
        if generator_class := model_mapping.get(model_name):
            return generator_class
        
        # Create a dynamic generator for any OpenRouter model
        # Get max_tokens from parameter, environment variable, or use sensible defaults based on model name
        if self._custom_max_tokens is not None:
            max_tokens = self._custom_max_tokens
        else:
            max_tokens_str = os.environ.get("OPENROUTER_MAX_TOKENS")
            if max_tokens_str:
                max_tokens = int(max_tokens_str)
            else:
                # Provide sensible defaults based on model family
                if "gpt-4" in model_name:
                    max_tokens = 128 * 1024
                elif "claude" in model_name:
                    max_tokens = 8192
                elif "llama" in model_name or "gemma" in model_name:
                    max_tokens = 8192
                else:
                    max_tokens = 8192  # Safe default for unknown models
        
        # Create dynamic generator class with the specific max_tokens
        final_max_tokens = max_tokens
        
        class DynamicOpenRouterGenerator(OpenRouterSchematicGenerator[T]):
            def __init__(self, logger: Logger):
                super().__init__(model_name=model_name, logger=logger)
            
            @property
            @override
            def max_tokens(self) -> int:
                return final_max_tokens
        
        return DynamicOpenRouterGenerator

    @override
    async def get_schematic_generator(self, t: type[T]) -> OpenRouterSchematicGenerator[T]:
        generator_class = self._get_specialized_generator_class(self.model_name, t)
        return generator_class(self._logger)  # type: ignore

    @override
    async def get_embedder(self) -> Embedder:
        embedding_model = os.environ.get("OPENROUTER_EMBEDDING_MODEL", "text-embedding-ada-002")
        self._logger.info(f"Using OpenRouter embedding model: {embedding_model}")
        return OpenRouterTextEmbedding(model_name=embedding_model, logger=self._logger)

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
