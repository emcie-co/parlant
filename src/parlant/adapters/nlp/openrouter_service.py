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

from __future__ import annotations
import time
from typing import Any, Mapping, Dict, Optional
from typing_extensions import override
import json
import jsonfinder  # type: ignore
import os
from pathlib import Path
from dataclasses import dataclass
import dataclasses

from pydantic import ValidationError
import tiktoken
import openai
import toml

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


# Configuration loading function
def load_openrouter_config() -> dict[str, Any]:
    """Load OpenRouter configuration from TOML file and environment variables."""
    config = {}
    
    # Load from TOML file
    config_path = Path("parlant_openrouter.toml")
    if config_path.exists():
        try:
            toml_config = toml.load(config_path)
            config = toml_config.get("parlant", {}).get("openrouter", {})
        except Exception as e:
            print(f"Warning: Failed to load TOML config: {e}")
    
    # Override with environment variables
    for env_var in ["MODEL", "OPENROUTER_BASE_URL"]:
        if value := os.environ.get(env_var):
            config[env_var] = value
    
    # Embedding configuration override
    embedding_config = config.setdefault("embedding", {})
    for env_var in ["OPENROUTER_EMBEDDING_MODEL", "OPENROUTER_EMBEDDING_DIMENSIONS", 
                   "OPENROUTER_EMBEDDING_BASE_URL", "OPENROUTER_EMBEDDING_API_KEY"]:
        if value := os.environ.get(env_var):
            if env_var == "OPENROUTER_EMBEDDING_DIMENSIONS":
                embedding_config[env_var] = int(value)
            else:
                embedding_config[env_var] = value
    
    return config


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model: str
    max_tokens: int

@dataclass
class SchemaConfig:
    """Configuration for different schema types."""
    single_tool_batch: ModelConfig
    journey_node_selection: ModelConfig
    canned_response_draft: ModelConfig
    canned_response_selection: ModelConfig
    
    @classmethod
    def get_default(cls) -> "SchemaConfig":
        """Get default configuration."""
        return cls(
            single_tool_batch=ModelConfig("openai/gpt-4o", 128 * 1024),
            journey_node_selection=ModelConfig("anthropic/claude-3.5-sonnet", 200 * 1024),
            canned_response_draft=ModelConfig("anthropic/claude-3.5-sonnet", 200 * 1024),
            canned_response_selection=ModelConfig("anthropic/claude-3-haiku", 200 * 1024)
        )
    
    def get_config_for_schema(self, schema_type: type) -> ModelConfig:
        """Get configuration for a specific schema type."""
        schema_mapping = {
            SingleToolBatchSchema: self.single_tool_batch,
            JourneyNodeSelectionSchema: self.journey_node_selection,
            CannedResponseDraftSchema: self.canned_response_draft,
            CannedResponseSelectionSchema: self.canned_response_selection,
        }
        return schema_mapping.get(schema_type, self.single_tool_batch)


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
        logger: Logger,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self.model_name = model_name
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
        self._tokenizer = OpenRouterEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"{self.model_name}"

    @property
    @override
    def tokenizer(self) -> OpenRouterEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024

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
                model=self.model_name,
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
                f"Invalid JSON returned by openrouter/{self.model_name}:\n{raw_content}"
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
                f"JSON content returned by openrouter/{self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class OpenRouterEmbedder(Embedder):
    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        # Load full OpenRouter configuration
        config = load_openrouter_config()
        embedding_config = config.get("embedding", {})
        # Extract embedding settings with defaults
        self._embedding_model = embedding_config.get("OPENROUTER_EMBEDDING_MODEL")
        self._embedding_dimensions = int(embedding_config.get("OPENROUTER_EMBEDDING_DIMENSIONS"))
        self._embedding_base_url = embedding_config.get("OPENROUTER_EMBEDDING_BASE_URL")
        self._embedding_api_key = embedding_config.get("OPENROUTER_EMBEDDING_API_KEY", os.environ.get("OPENROUTER_API_KEY"))
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
    async def embed(self, text: str) -> EmbeddingResult:
        response = await self._client.embeddings.create(
            model=self._embedding_model,
            input=text,
        )

        embedding = response.data[0].embedding
        usage = response.usage

        return EmbeddingResult(
            embedding=embedding,
            usage=UsageInfo(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=0,
            ),
        )


# OpenRouter Generator Factory
def create_openrouter_generator(
    logger: Logger, 
    api_key: str, 
    base_url: str, 
    model_name: str,
    max_tokens: int = 128 * 1024,
    schema_type: type | None = None
) -> type[OpenRouterSchematicGenerator]:
    """Create a dynamic OpenRouter generator class with specified parameters."""
    
    class DynamicOpenRouterGenerator(OpenRouterSchematicGenerator):
        def __init__(self, logger: Logger, api_key: str, base_url: str, model_name: str = model_name) -> None:
            super().__init__(model_name=model_name, logger=logger, api_key=api_key, base_url=base_url)

        @property
        @override
        def max_tokens(self) -> int:
            return max_tokens
        
        @property
        def schema(self) -> type:
            return schema_type or super().schema
    
    # Set the __orig_class__ attribute for proper generic type handling
    if schema_type:
        DynamicOpenRouterGenerator.__orig_class__ = schema_type
    
    return DynamicOpenRouterGenerator


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

    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._api_key = os.environ["OPENROUTER_API_KEY"]
        # Load and apply OpenRouter configuration
        config = load_openrouter_config()
        self._base_url = config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self._default_model = config.get("MODEL", "openai/gpt-4o")

        # Load structured schema configuration
        self._schema_config = self._load_schematic_config(config)
        self._logger.info("Initialized OpenRouterService")

    def _load_schematic_config(self, config: dict[str, Any]) -> SchemaConfig:
        """Load structured schema configuration from TOML or use defaults."""
        schematic_config = SchemaConfig.get_default()
        
        if "schematic_config" in config:
            schema_data = config["schematic_config"]
            try:
                # Get fields from SchemaConfig dataclass
                fields = [field.name for field in dataclasses.fields(schematic_config)]
                
                for field in fields:
                    if field in schema_data:
                        data = schema_data[field]
                        current = getattr(schematic_config, field)
                        setattr(schematic_config, field, ModelConfig(
                            model=data.get("model", current.model),
                            max_tokens=data.get("max_tokens", current.max_tokens)
                        ))
                
                self._logger.info("Loaded structured schema configuration from TOML")
            except Exception as e:
                self._logger.warning(f"Failed to load structured schema config: {e}")
        
        return schematic_config

    @override
    async def get_schematic_generator(self, t: type[T]) -> OpenRouterSchematicGenerator[T]:
        # Get the specific model config for the requested schema type
        model_config = self._schema_config.get_config_for_schema(t)
        
        # Create dynamic generator class using factory
        generator_class = create_openrouter_generator(
            logger=self._logger,
            api_key=self._api_key,
            base_url=self._base_url,
            model_name=model_config.model,
            max_tokens=model_config.max_tokens,
            schema_type=t
        )
        
        # Create and return instance
        return generator_class(
            logger=self._logger,
            api_key=self._api_key,
            base_url=self._base_url,
            model_name=model_config.model
        )  # type: ignore

    @override
    async def get_embedder(self) -> Embedder:
        # Create embedder which loads its own configuration
        return OpenRouterEmbedder(self._logger)

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()