# src/parlant/adapters/nlp/openrouter_service.py
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

"""
OpenRouter NLP Service Adapter for Parlant.

This module provides integration with OpenRouter's API gateway, offering a unified
interface to access language models from multiple providers (OpenAI, Anthropic, 
Google, Meta, and more) through a single API endpoint.

Key Benefits:
- Single API key for all LLM providers
- Centralized billing and usage tracking
- Mix models from different vendors for optimal performance
- No vendor lock-in - switch models without code changes

Example Mixed Model Configuration:
    Use different models for different tasks within the same application:
    - GPT-4 for tool execution (high accuracy)
    - Claude 3.5 for journey reasoning (complex logic)
    - Gemini Pro for content generation (creativity)
    - Llama 3 for simple selections (cost efficiency)

Configuration is managed entirely through environment variables.

Required Environment Variables:
    OPENROUTER_API_KEY: API key for OpenRouter authentication

Optional Environment Variables:
    OPENROUTER_BASE_URL: Base URL for OpenRouter API (default: https://openrouter.ai/api/v1)
    OPENROUTER_DEFAULT_MODEL: Default model to use (default: openai/gpt-4o)

Schema-Specific Model Configuration:
    OPENROUTER_SINGLE_TOOL_MODEL: Model for SingleToolBatchSchema (default: openai/gpt-4o)
    OPENROUTER_SINGLE_TOOL_MAX_TOKENS: Max tokens for SingleToolBatchSchema (default: 131072)
    OPENROUTER_JOURNEY_NODE_MODEL: Model for JourneyNodeSelectionSchema (default: anthropic/claude-3.5-sonnet)
    OPENROUTER_JOURNEY_NODE_MAX_TOKENS: Max tokens for JourneyNodeSelectionSchema (default: 204800)
    OPENROUTER_CANNED_RESPONSE_DRAFT_MODEL: Model for CannedResponseDraftSchema (default: anthropic/claude-3.5-sonnet)
    OPENROUTER_CANNED_RESPONSE_DRAFT_MAX_TOKENS: Max tokens for CannedResponseDraftSchema (default: 204800)
    OPENROUTER_CANNED_RESPONSE_SELECTION_MODEL: Model for CannedResponseSelectionSchema (default: anthropic/claude-3-haiku)
    OPENROUTER_CANNED_RESPONSE_SELECTION_MAX_TOKENS: Max tokens for CannedResponseSelectionSchema (default: 204800)

Embedding Configuration (Third-party provider required):
    Note: OpenRouter does not provide embedding services. Configure a third-party provider
    that follows the OpenAI embedding API format (e.g., OpenAI, Together AI).
    
    OPENROUTER_EMBEDDING_MODEL: Model name for the embedding provider (default: openai/text-embedding-3-small)
    OPENROUTER_EMBEDDING_DIMENSIONS: Embedding vector dimensions (default: 1536)
    OPENROUTER_EMBEDDING_BASE_URL: Base URL of your embedding provider (e.g., https://api.openai.com/v1)
    OPENROUTER_EMBEDDING_API_KEY: API key for the embedding provider (defaults to OPENROUTER_API_KEY)

Example Usage:
    # OpenRouter API for LLM
    export OPENROUTER_API_KEY="your-openrouter-api-key"
    export OPENROUTER_SINGLE_TOOL_MODEL="openai/gpt-4o-mini"
    export OPENROUTER_SINGLE_TOOL_MAX_TOKENS=65536
    
    # Third-party embedding provider (e.g., OpenAI)
    export OPENROUTER_EMBEDDING_BASE_URL="https://api.openai.com/v1"
    export OPENROUTER_EMBEDDING_API_KEY="your-openai-api-key"
    export OPENROUTER_EMBEDDING_MODEL="text-embedding-3-small"
"""

from __future__ import annotations
import time
from typing import Any, Mapping
from typing_extensions import override
import json
import jsonfinder  # type: ignore
import os
from dataclasses import dataclass

from pydantic import ValidationError
import tiktoken
import openai

from parlant.adapters.nlp.common import normalize_json_output
from parlant.core.engines.alpha.canned_response_generator import (
    CannedResponseDraftSchema,
    CannedResponseSelectionSchema,
)
from parlant.core.engines.alpha.guideline_matching.generic.journey_node_selection_batch import (
    JourneyNodeSelectionSchema,
)
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.engines.alpha.tool_calling.single_tool_batch import SingleToolBatchSchema
from parlant.core.loggers import Logger
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
from parlant.core.nlp.policies import policy, retry


# Environment variable configuration keys
ENV_API_KEY = "OPENROUTER_API_KEY"
ENV_BASE_URL = "OPENROUTER_BASE_URL"
ENV_DEFAULT_MODEL = "OPENROUTER_DEFAULT_MODEL"

# Schema-specific model configuration
ENV_SINGLE_TOOL_MODEL = "OPENROUTER_SINGLE_TOOL_MODEL"
ENV_SINGLE_TOOL_MAX_TOKENS = "OPENROUTER_SINGLE_TOOL_MAX_TOKENS"
ENV_JOURNEY_NODE_MODEL = "OPENROUTER_JOURNEY_NODE_MODEL"
ENV_JOURNEY_NODE_MAX_TOKENS = "OPENROUTER_JOURNEY_NODE_MAX_TOKENS"
ENV_CANNED_RESPONSE_DRAFT_MODEL = "OPENROUTER_CANNED_RESPONSE_DRAFT_MODEL"
ENV_CANNED_RESPONSE_DRAFT_MAX_TOKENS = "OPENROUTER_CANNED_RESPONSE_DRAFT_MAX_TOKENS"
ENV_CANNED_RESPONSE_SELECTION_MODEL = "OPENROUTER_CANNED_RESPONSE_SELECTION_MODEL"
ENV_CANNED_RESPONSE_SELECTION_MAX_TOKENS = "OPENROUTER_CANNED_RESPONSE_SELECTION_MAX_TOKENS"

# Embedding configuration
ENV_EMBEDDING_MODEL = "OPENROUTER_EMBEDDING_MODEL"
ENV_EMBEDDING_DIMENSIONS = "OPENROUTER_EMBEDDING_DIMENSIONS"
ENV_EMBEDDING_BASE_URL = "OPENROUTER_EMBEDDING_BASE_URL"
ENV_EMBEDDING_API_KEY = "OPENROUTER_EMBEDDING_API_KEY"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model: str
    max_tokens: int


def get_env_int(key: str, default: int) -> int:
    """Get integer value from environment variable with default."""
    value = os.environ.get(key)
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return default


def get_model_config_for_schema(schema_type: type) -> ModelConfig:
    """Get model configuration for a specific schema type from environment variables."""
    # Default configurations
    defaults = {
        SingleToolBatchSchema: ModelConfig("openai/gpt-4o", 128 * 1024),
        JourneyNodeSelectionSchema: ModelConfig("anthropic/claude-3.5-sonnet", 200 * 1024),
        CannedResponseDraftSchema: ModelConfig("anthropic/claude-3.5-sonnet", 200 * 1024),
        CannedResponseSelectionSchema: ModelConfig("anthropic/claude-3-haiku", 200 * 1024),
    }
    
    # Environment variable mappings
    env_mappings = {
        SingleToolBatchSchema: (ENV_SINGLE_TOOL_MODEL, ENV_SINGLE_TOOL_MAX_TOKENS),
        JourneyNodeSelectionSchema: (ENV_JOURNEY_NODE_MODEL, ENV_JOURNEY_NODE_MAX_TOKENS),
        CannedResponseDraftSchema: (ENV_CANNED_RESPONSE_DRAFT_MODEL, ENV_CANNED_RESPONSE_DRAFT_MAX_TOKENS),
        CannedResponseSelectionSchema: (ENV_CANNED_RESPONSE_SELECTION_MODEL, ENV_CANNED_RESPONSE_SELECTION_MAX_TOKENS),
    }
    
    default_config = defaults.get(schema_type, defaults[SingleToolBatchSchema])
    
    if schema_type in env_mappings:
        model_env, tokens_env = env_mappings[schema_type]
        model = os.environ.get(model_env, default_config.model)
        max_tokens = get_env_int(tokens_env, default_config.max_tokens)
        return ModelConfig(model, max_tokens)
    
    return default_config


RATE_LIMIT_ERROR_MESSAGE = (
    "OpenRouter API rate limit exceeded. Possible reasons:\n"
    "1. Your account may have insufficient API credits.\n"
    "2. You may be using a free-tier account with limited request capacity.\n"
    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your OpenRouter account balance and billing status.\n"
    "- Review your API usage limits in OpenRouter dashboard.\n"
    "- For more details on rate limits and usage tiers, visit:\n"
    "  https://openrouter.ai/docs"
)


class OpenRouterEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # Use GPT-4 tokenizer as fallback for most models
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class OpenRouterSchematicGenerator(SchematicGenerator[T]):
    supported_openrouter_params = ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    supported_hints = supported_openrouter_params + ["strict"]

    def __init__(
        self,
        model_name: str,
        max_tokens: int,
        logger: Logger,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._logger = logger
        self._api_key = api_key
        self._base_url = base_url
        
        # Initialize OpenAI client
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://parlant.ai",
                "X-Title": "Parlant",
            }
        )
        self._tokenizer = OpenRouterEstimatingTokenizer(model_name=self._model_name)

    @property
    @override
    def id(self) -> str:
        return f"{self._model_name}"

    @property
    @override
    def tokenizer(self) -> OpenRouterEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return self._max_tokens

    @policy(
        [
            retry(
                exceptions=(
                    openai.RateLimitError,
                    openai.APITimeoutError,
                    openai.APIError,
                ),
            ),
            retry(openai.APIError, max_exceptions=2, wait_times=(1.0, 5.0)),
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

        openrouter_api_arguments = {k: v for k, v in hints.items() if k in self.supported_openrouter_params}

        t_start = time.time()
        try:
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                **openrouter_api_arguments,
            )
        except openai.RateLimitError:
            self._logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise

        t_end = time.time()

        self._logger.trace(json.dumps(response.model_dump(), indent=2))

        raw_content = response.choices[0].message.content or "{}"

        try:
            json_content = json.loads(normalize_json_output(raw_content))
        except json.JSONDecodeError:
            self._logger.warning(
                f"Invalid JSON returned by openrouter/{self._model_name}:\n{raw_content}"
            )
            json_content = jsonfinder.only_json(raw_content)[2]
            self._logger.warning("Found JSON content within model response; continuing...")

        try:
            content = self.schema.model_validate(json_content)
            usage_data = response.usage

            return SchematicGenerationResult(
                content=content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=usage_data.prompt_tokens if usage_data else 0,
                        output_tokens=usage_data.completion_tokens if usage_data else 0,
                    ),
                ),
            )
        except ValidationError:
            self._logger.error(
                f"JSON content returned by openrouter/{self._model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class OpenRouterEmbedder(Embedder):
    """
    Embedder that connects to third-party embedding services.
    
    Note: OpenRouter does not provide embedding services. This embedder is designed
    to work with any third-party provider that follows the OpenAI embedding API format,
    such as OpenAI, Together AI, or other compatible services.
    """
    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        
        # Get embedding configuration from environment variables
        self._embedding_model = os.environ.get(ENV_EMBEDDING_MODEL, "openai/text-embedding-3-small")
        self._embedding_dimensions = get_env_int(ENV_EMBEDDING_DIMENSIONS, 1536)
        
        # Get base URL and API key from environment
        base_url = os.environ.get(ENV_BASE_URL, "https://openrouter.ai/api/v1")
        api_key = os.environ.get(ENV_API_KEY, "")
        
        self._embedding_base_url = os.environ.get(ENV_EMBEDDING_BASE_URL, base_url)
        # Use specific embedding API key if provided, otherwise use main API key
        self._embedding_api_key = os.environ.get(ENV_EMBEDDING_API_KEY, api_key)
        
        if not self._embedding_api_key:
            raise ValueError(f"Either {ENV_API_KEY} or {ENV_EMBEDDING_API_KEY} must be set")
        
        # Initialize OpenAI client
        self._client = openai.AsyncOpenAI(
            api_key=self._embedding_api_key,
            base_url=self._embedding_base_url,
        )
        self._logger.info(f"Initialized OpenRouterEmbedder with model: {self._embedding_model}")

    @property
    @override
    def id(self) -> str:
        return f"{self._embedding_model}"

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return OpenRouterEstimatingTokenizer(self._embedding_model)

    @property
    @override
    def dimensions(self) -> int:
        return self._embedding_dimensions

    @policy(
        [
            retry(
                exceptions=(
                    openai.RateLimitError,
                    openai.APITimeoutError,
                    openai.APIError,
                ),
            ),
            retry(openai.APIError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        _ = hints  # Not used for OpenRouter
        
        response = await self._client.embeddings.create(
            model=self._embedding_model,
            input=texts,
        )

        vectors = [data_point.embedding for data_point in response.data]
        return EmbeddingResult(vectors=vectors)


# Specific generator classes for different models/use cases
class OpenRouter_Default(OpenRouterSchematicGenerator[T]):
    """Default OpenRouter generator using configuration from environment."""
    def __init__(self, logger: Logger, api_key: str, base_url: str, model_config: ModelConfig) -> None:
        super().__init__(
            model_name=model_config.model,
            max_tokens=model_config.max_tokens,
            logger=logger,
            api_key=api_key,
            base_url=base_url,
        )


class OpenRouterService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""
        if not os.environ.get(ENV_API_KEY):
            return f"""\
You're using the OpenRouter NLP service, but {ENV_API_KEY} is not set.
Please set {ENV_API_KEY} in your environment before running Parlant.
"""
        return None

    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._api_key = os.environ[ENV_API_KEY]
        self._base_url = os.environ.get(ENV_BASE_URL, "https://openrouter.ai/api/v1")
        self._default_model = os.environ.get(ENV_DEFAULT_MODEL, "openai/gpt-4o")
        self._logger.info("Initialized OpenRouterService")

    @override
    async def get_schematic_generator(self, t: type[T]) -> SchematicGenerator[T]:
        # Get the specific model config for the requested schema type
        model_config = get_model_config_for_schema(t)
        
        # Create generator instance with the appropriate config
        return OpenRouter_Default[t](  # type: ignore
            logger=self._logger,
            api_key=self._api_key,
            base_url=self._base_url,
            model_config=model_config,
        )

    @override
    async def get_embedder(self) -> Embedder:
        # Create embedder which loads configuration from environment
        return OpenRouterEmbedder(logger=self._logger)

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()