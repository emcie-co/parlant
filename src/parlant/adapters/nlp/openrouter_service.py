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
    BadRequestError,
    ConflictError,
    InternalServerError,
    RateLimitError,
)
from typing import Any, Callable, Mapping
from typing_extensions import override
import json
import jsonfinder  # type: ignore
import os

from pydantic import ValidationError
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.embedding import BaseEmbedder, Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
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


class OpenRouterSchematicGenerator(BaseSchematicGenerator[T]):
    supported_openrouter_params = ["temperature", "max_tokens"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        meter: Meter,
    ) -> None:
        super().__init__(logger=logger, meter=meter, model_name=model_name)
        self._logger = logger

        # Build extra headers from environment variables
        extra_headers = {}
        if "OPENROUTER_HTTP_REFERER" in os.environ:
            extra_headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
        if "OPENROUTER_SITE_NAME" in os.environ:
            extra_headers["X-Title"] = os.environ["OPENROUTER_SITE_NAME"]

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

    @property
    @override
    def max_tokens(self) -> int:
        # Default implementation - should be overridden by subclasses
        return 8192

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
    async def do_generate(
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
        
        # Try with JSON mode first, but catch errors gracefully
        response = None
        
        try:
            # Try with JSON mode
            response = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format={"type": "json_object"},
                **openrouter_api_arguments,
            )
        except BadRequestError as e:
            # Check if it's a JSON mode error
            error_str = str(e)
            if "JSON mode" in error_str or "json_object" in error_str.lower():
                self._logger.error(
                    f"\n⚠️  Model '{self.model_name}' does not support JSON mode.\n"
                    f"Please switch to a model that supports JSON mode (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet').\n"
                    f"Attempting to continue without JSON mode enforcement, but results may be less reliable.\n"
                )
                # Retry without JSON mode with a system message to instruct JSON output
                try:
                    # Add system message to instruct the model to output JSON
                    json_instruction = "IMPORTANT: You must respond with ONLY valid JSON. No explanatory text before or after the JSON. The response must be a valid JSON object."
                    response = await self._client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": json_instruction},
                            {"role": "user", "content": prompt}
                        ],
                        model=self.model_name,
                        **openrouter_api_arguments,
                    )
                except Exception as retry_error:
                    self._logger.error(
                        f"\n❌ Failed to use model '{self.model_name}' even without JSON mode.\n"
                        f"Error: {retry_error}\n"
                        f"Please change your model to one that supports JSON mode or use a different model entirely.\n"
                    )
                    raise
            else:
                # Some other BadRequest error - just log it once and raise
                self._logger.error(f"OpenRouter API BadRequest: {e}")
                raise
        except RateLimitError as e:
            self._logger.error(
                f"\n⏱️  Rate limit exceeded for model '{self.model_name}'.\n"
                f"{RATE_LIMIT_ERROR_MESSAGE}\n"
                f"Consider:\n"
                f"  - Using a different model\n"
                f"  - Waiting a moment before retrying\n"
                f"  - Adding your own API key for higher limits\n"
            )
            raise
        except Exception as e:
            self._logger.error(
                f"\n❌ OpenRouter API error with model '{self.model_name}': {type(e).__name__}\n"
                f"{e}\n"
                f"Consider switching to a more compatible model.\n"
            )
            raise
        
        t_end = time.time()

        if response.usage:
            self._logger.trace(response.usage.model_dump_json(indent=2))

        raw_content = response.choices[0].message.content or "{}"
        
        # Check if we got empty response
        if not raw_content.strip() or raw_content.strip() == "{}":
            self._logger.error(
                f"\n❌ Model '{self.model_name}' returned empty or invalid JSON.\n"
                f"Response: {raw_content}\n"
                f"This model may not be compatible with structured output requirements.\n"
                f"Please switch to a model that supports JSON mode (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet').\n"
            )
            # Set empty JSON as fallback
            json_content = {}
        else:
            try:
                json_content = json.loads(normalize_json_output(raw_content))
                # Check if parsed JSON is empty
                if not json_content or json_content == {}:
                    self._logger.warning("Model returned empty JSON object. Attempting to find JSON in response...")
                    # Try to find JSON in the response
                    try:
                        json_content = jsonfinder.only_json(raw_content)[2]
                        if json_content and json_content != {}:
                            self._logger.info("Found valid JSON content within response.")
                    except Exception:
                        self._logger.error(f"Could not extract valid JSON from response: {raw_content}")
            except json.JSONDecodeError as e:
                self._logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content}")
                try:
                    # Try to extract JSON using jsonfinder
                    json_content = jsonfinder.only_json(raw_content)[2]
                    self._logger.warning("Found JSON content within model response; continuing...")
                except Exception as finder_error:
                    self._logger.error(
                        f"\n❌ Could not parse JSON from model response.\n"
                        f"Raw response: {raw_content}\n"
                        f"Error: {finder_error}\n"
                        f"Model '{self.model_name}' may not be compatible.\n"
                        f"Consider switching to a model that supports structured output.\n"
                    )
                    json_content = {}

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
        except ValidationError as e:
            self._logger.error(
                f"\n❌ JSON content returned by '{self.model_name}' does not match expected schema.\n"
                f"Schema: {self.schema.__name__}\n"
                f"Raw response: {raw_content}\n"
                f"Parsed JSON: {json.dumps(json_content, indent=2) if json_content else 'Empty'}\n"
                f"Validation errors: {str(e)}\n"
                f"This model may not be producing valid structured output.\n"
                f"Consider switching to a model that supports JSON mode.\n"
            )
            raise


class OpenRouterGPT4O(OpenRouterSchematicGenerator[T]):
    def __init__(self, logger: Logger, meter: Meter) -> None:
        super().__init__(model_name="openai/gpt-4o", logger=logger, meter=meter)

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class OpenRouterGPT4OMini(OpenRouterSchematicGenerator[T]):
    def __init__(self, logger: Logger, meter: Meter) -> None:
        super().__init__(model_name="openai/gpt-4o-mini", logger=logger, meter=meter)

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class OpenRouterClaude35Sonnet(OpenRouterSchematicGenerator[T]):
    def __init__(self, logger: Logger, meter: Meter) -> None:
        super().__init__(model_name="anthropic/claude-3.5-sonnet", logger=logger, meter=meter)

    @property
    @override
    def max_tokens(self) -> int:
        return 8192


class OpenRouterLlama33_70B(OpenRouterSchematicGenerator[T]):
    def __init__(self, logger: Logger, meter: Meter) -> None:
        super().__init__(model_name="meta-llama/llama-3.3-70b-instruct", logger=logger, meter=meter)

    @property
    @override
    def max_tokens(self) -> int:
        return 8192


class OpenRouterEmbedder(BaseEmbedder):
    supported_arguments = ["dimensions"]

    def __init__(self, model_name: str, logger: Logger, meter: Meter) -> None:
        super().__init__(logger, meter, model_name)
        self.model_name = model_name

        # Build extra headers from environment variables
        extra_headers = {}
        if "OPENROUTER_HTTP_REFERER" in os.environ:
            extra_headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
        if "OPENROUTER_SITE_NAME" in os.environ:
            extra_headers["X-Title"] = os.environ["OPENROUTER_SITE_NAME"]

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

    @property
    @override
    def max_tokens(self) -> int:
        # Default max tokens for embedding models
        return 8192

    @property
    @override
    def dimensions(self) -> int:
        # Default dimensions for text-embedding-3-large
        # For other models, this should be overridden
        if "text-embedding-3-large" in self.model_name:
            return 3072
        elif "text-embedding-3-small" in self.model_name:
            return 1536
        else:
            # Default fallback - most embedding models use 1536 or 3072
            return 1536

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
    async def do_embed(
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
            self.logger.error(
                f"\n⏱️  Rate limit exceeded for embedder model '{self.model_name}'.\n"
                f"{RATE_LIMIT_ERROR_MESSAGE}\n"
                f"Consider:\n"
                f"  - Using a different embedder model\n"
                f"  - Waiting a moment before retrying\n"
                f"  - Adding your own API key for higher limits\n"
            )
            raise

        vectors = [data_point.embedding for data_point in response.data]
        return EmbeddingResult(vectors=vectors)


class OpenRouterTextEmbedding3Large(OpenRouterEmbedder):
    def __init__(self, logger: Logger, meter: Meter) -> None:
        super().__init__(model_name="openai/text-embedding-3-large", logger=logger, meter=meter)

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    @override
    def dimensions(self) -> int:
        return 3072


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
        meter: Meter,
        model_name: str | None = None,
        max_tokens: int | None = None,
        embedder_model_name: str | None = None,
    ) -> None:
        self._logger = logger
        self._meter = meter
        self._logger.info("Initialized OpenRouterService")
        # Use provided model_name or fall back to environment variable
        self.model_name = model_name or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")
        self._custom_max_tokens = max_tokens
        # Use provided embedder_model_name or fall back to environment variable or default
        self.embedder_model_name = (
            embedder_model_name
            or os.environ.get("OPENROUTER_EMBEDDER_MODEL", "openai/text-embedding-3-large")
        )
        self._logger.info(f"OpenRouter model name: {self.model_name}")
        self._logger.info(f"OpenRouter embedder model name: {self.embedder_model_name}")

    def _get_specialized_generator_class(
        self,
        model_name: str,
        t: type[T],
    ) -> Callable[[Logger, Meter], OpenRouterSchematicGenerator[T]]:
        """
        Returns the specialized generator class for known models.
        For unknown models, creates a dynamic generator that works with any OpenRouter model.
        """
        model_mapping: dict[str, Callable[[Logger, Meter], OpenRouterSchematicGenerator[T]]] = {
            "openai/gpt-4o": lambda logger, meter: OpenRouterGPT4O[t](logger, meter),  # type: ignore
            "openai/gpt-4o-mini": lambda logger, meter: OpenRouterGPT4OMini[t](logger, meter),  # type: ignore
            "anthropic/claude-3.5-sonnet": lambda logger, meter: OpenRouterClaude35Sonnet[t](logger, meter),  # type: ignore
            "meta-llama/llama-3.3-70b-instruct": lambda logger, meter: OpenRouterLlama33_70B[t](logger, meter),  # type: ignore
        }

        # Check if we have a predefined generator for this model
        if generator_factory := model_mapping.get(model_name):
            return generator_factory
        
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
            def __init__(self, logger: Logger, meter: Meter):
                super().__init__(model_name=model_name, logger=logger, meter=meter)
            
            @property
            @override
            def max_tokens(self) -> int:
                return final_max_tokens
        
        # Return a factory function that creates the properly typed instance
        def create_generator(logger: Logger, meter: Meter) -> OpenRouterSchematicGenerator[T]:
            return DynamicOpenRouterGenerator[t](logger, meter)  # type: ignore
        
        return create_generator

    @override
    async def get_schematic_generator(self, t: type[T]) -> OpenRouterSchematicGenerator[T]:
        generator_factory = self._get_specialized_generator_class(self.model_name, t)
        return generator_factory(self._logger, self._meter)

    @override
    async def get_embedder(self) -> Embedder:
        # Use OpenRouter embedder with the configured embedder model name
        # Default to text-embedding-3-large if not specified
        if self.embedder_model_name == "openai/text-embedding-3-large":
            return OpenRouterTextEmbedding3Large(logger=self._logger, meter=self._meter)
        else:
            return OpenRouterEmbedder(
                model_name=self.embedder_model_name,
                logger=self._logger,
                meter=self._meter,
            )

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
