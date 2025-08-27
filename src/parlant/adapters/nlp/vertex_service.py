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

# Maintainer: Agam Dubey hello.world.agam@gmail.com

# Moderation service needs to be added
# Usage guidelines - Use gemini-2.5-pro and claude sonnet 4 models for best results
# Set env variables: VERTEX_AI_PROJECT_ID VERTEX_AI_REGION, VERTEX_AI_MODEL

import os
import time
from typing import Any, Mapping, cast
from typing_extensions import override
from enum import Enum

import google.auth
import google.api_core.exceptions
import google.genai  # type: ignore
import google.genai.types  # type: ignore
from google.api_core.exceptions import NotFound, TooManyRequests, ResourceExhausted, ServerError

from anthropic import (
    AsyncAnthropicVertex,
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)  # type: ignore

import jsonfinder  # type: ignore
from pydantic import ValidationError
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    SchematicGenerator,
    FallbackSchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.loggers import Logger


class ModelProvider(Enum):
    """Enum to identify the model provider."""

    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class VertexAIAuthError(Exception):
    """Raised when there are authentication issues with Vertex AI."""

    pass


class VertexAIEstimatingTokenizer(EstimatingTokenizer):
    """Tokenizer that estimates token count for Vertex AI models."""

    def __init__(self, client: google.genai.Client, model_name: str):
        self.model_name = model_name
        self._client = client
        if "claude" in model_name.lower():
            self.encoding: tiktoken.Encoding | None = tiktoken.encoding_for_model(
                "gpt-4o-2024-08-06"
            )
        else:
            self.encoding = None

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        """Estimate token count using tiktoken for Claude, Google API for Gemini."""
        if self.encoding:
            tokens = self.encoding.encode(prompt)
            return int(len(tokens) * 1.15)  # @check - as seen on aws_service for bedrock
        else:
            model_approximation = {
                "text-embedding-004": "gemini-2.5-pro",
            }.get(self.model_name, self.model_name)

            result = await self._client.aio.models.count_tokens(
                model=model_approximation,
                contents=prompt,
            )
            return int(result.total_tokens or 0)


def get_model_provider(model_name: str) -> ModelProvider:
    """Determine the model provider based on model name."""
    if "claude" in model_name.lower():
        return ModelProvider.ANTHROPIC
    elif "gemini" in model_name.lower():
        return ModelProvider.GOOGLE
    else:
        raise ValueError(f"Unknown model provider for model: {model_name}")


class VertexAIClaudeSchematicGenerator(SchematicGenerator[T]):
    """Schematic generator for Claude models via Vertex AI."""

    supported_hints = ["temperature", "max_tokens", "top_p", "top_k"]

    def __init__(
        self,
        project_id: str,
        region: str,
        model_name: str,
        logger: Logger,
    ) -> None:
        self.project_id = project_id
        self.region = region
        self.model_name = model_name
        self._logger = logger

        self._client = AsyncAnthropicVertex(
            project_id=project_id,
            region=region,
        )

        self._genai_client = google.genai.Client(project=project_id, location=region, vertexai=True)
        self._tokenizer = VertexAIEstimatingTokenizer(self._genai_client, model_name)

    @property
    @override
    def id(self) -> str:
        return f"vertex-ai/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        # Claude models support 200k tokens
        return 200_000

    @policy(
        [
            retry(
                exceptions=(
                    APIConnectionError,
                    APITimeoutError,
                    RateLimitError,
                    APIResponseValidationError,
                ),
                max_exceptions=3,
                wait_times=(1.0, 2.0, 4.0),
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
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        anthropic_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        t_start = time.time()
        try:
            response = await self._client.messages.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                max_tokens=hints.get("max_tokens", 8192),
                **anthropic_api_arguments,
            )
        except RateLimitError:
            self._logger.error(
                "Vertex AI rate limit exceeded. Possible reasons:\n"
                "1. Your GCP project may have insufficient quota.\n"
                "2. The model may not be enabled in Vertex AI Model Garden.\n"
                "3. You might have exceeded the requests-per-minute limit.\n\n"
                "Recommended actions:\n"
                "- Check your Vertex AI quotas in the GCP Console.\n"
                "- Ensure the model is enabled in Vertex AI Model Garden.\n"
                "- Review IAM permissions for the service account.\n"
                "- Visit: https://console.cloud.google.com/vertex-ai/model-garden",
            )
            raise
        except Exception as e:
            if "403" in str(e) or "permission" in str(e).lower():
                self._logger.error(
                    f"Permission denied accessing Vertex AI. Ensure:\n"
                    f"1. ADC is properly configured (run 'gcloud auth application-default login')\n"
                    f"2. The service account has 'Vertex AI User' role\n"
                    f"3. The {self.model_name} model is enabled in Vertex AI Model Garden\n"
                    f"Error: {e}"
                )
            raise

        t_end = time.time()

        raw_content = response.content[0].text

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
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                    ),
                ),
            )
        except ValidationError:
            self._logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class VertexAIGeminiSchematicGenerator(SchematicGenerator[T]):
    """Schematic generator for Gemini models"""

    supported_hints = ["temperature", "thinking_config"]

    def __init__(
        self,
        project_id: str,
        region: str,
        model_name: str,
        logger: Logger,
    ) -> None:
        self.project_id = project_id
        self.region = region
        self.model_name = model_name
        self._logger = logger

        self._client = google.genai.Client(project=project_id, location=region, vertexai=True)
        self._tokenizer = VertexAIEstimatingTokenizer(self._client, model_name)

    @property
    @override
    def id(self) -> str:
        return f"vertex-ai/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        if "flash" in self.model_name.lower():
            return 1024 * 1024  # 1M tokens
        else:
            return 2 * 1024 * 1024  # 2M tokens

    @policy(
        [
            retry(
                exceptions=(
                    NotFound,
                    TooManyRequests,
                    ResourceExhausted,
                ),
                max_exceptions=3,
                wait_times=(1.0, 2.0, 4.0),
            ),
            retry(ServerError, max_exceptions=2, wait_times=(1.0, 5.0)),
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

        gemini_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}
        config = {
            "response_mime_type": "application/json",
            "response_schema": self.schema.model_json_schema(),
            **gemini_api_arguments,
        }

        t_start = time.time()
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=cast(google.genai.types.GenerateContentConfigOrDict, config),
            )
        except TooManyRequests:
            self._logger.error(
                "Google API rate limit exceeded.\n\n"
                "Possible reasons:\n"
                "1. Insufficient API credits in your account.\n"
                "2. Using a free-tier account with limited request capacity.\n"
                "3. Exceeded the requests-per-minute limit for your account.\n\n"
                "Recommended actions:\n"
                "- Check your Google API account balance and billing status.\n"
                "- Review your API usage limits in the Google Cloud Console.\n"
                "- Learn more about quotas and limits:\n"
                "  https://cloud.google.com/docs/quota-and-billing/quotas/quotas-overview"
            )
            raise
        except Exception as e:
            if "403" in str(e) or "permission" in str(e).lower():
                self._logger.error(
                    f"Permission denied accessing Google Gen AI. Ensure:\n"
                    f"1. GEMINI_API_KEY is properly configured\n"
                    f"2. The API key has proper permissions\n"
                    f"3. The {self.model_name} model is accessible\n"
                    f"Error: {e}"
                )
            raise

        t_end = time.time()

        raw_content = response.text

        try:
            json_content = normalize_json_output(raw_content or "{}")
            # Fix Gemini's quote issues
            json_content = json_content.replace(""", '"').replace(""", '"')

            # Fix double-escaped sequences
            for control_char in "utn":
                json_content = json_content.replace(f"\\\\{control_char}", f"\\{control_char}")

            json_object = jsonfinder.only_json(json_content)[2]
        except Exception:
            self._logger.error(f"Failed to extract JSON from {self.model_name}:\n{raw_content}")
            raise

        if response.usage_metadata:
            self._logger.trace(response.usage_metadata.model_dump_json(indent=2))

        try:
            model_content = self.schema.model_validate(json_object)

            return SchematicGenerationResult(
                content=model_content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=response.usage_metadata.prompt_token_count or 0,
                        output_tokens=response.usage_metadata.candidates_token_count or 0,
                        extra={
                            "cached_input_tokens": (
                                response.usage_metadata.cached_content_token_count
                                if response.usage_metadata
                                else 0
                            )
                            or 0
                        },
                    )
                    if response.usage_metadata
                    else UsageInfo(input_tokens=0, output_tokens=0, extra={}),
                ),
            )
        except ValidationError:
            self._logger.error(f"JSON from {self.model_name} doesn't match schema:\n{raw_content}")
            raise


class VertexClaudeOpus4(VertexAIClaudeSchematicGenerator[T]):
    def __init__(self, project_id: str, region: str, logger: Logger) -> None:
        super().__init__(
            project_id=project_id,
            region=region,
            model_name="claude-opus-4@20250514",
            logger=logger,
        )


class VertexClaudeSonnet4(VertexAIClaudeSchematicGenerator[T]):
    def __init__(self, project_id: str, region: str, logger: Logger) -> None:
        super().__init__(
            project_id=project_id,
            region=region,
            model_name="claude-sonnet-4@20250514",
            logger=logger,
        )


class VertexClaudeSonnet35(VertexAIClaudeSchematicGenerator[T]):
    def __init__(self, project_id: str, region: str, logger: Logger) -> None:
        super().__init__(
            project_id=project_id,
            region=region,
            model_name="claude-3-5-sonnet-v2@20241022",
            logger=logger,
        )


class VertexClaudeHaiku35(VertexAIClaudeSchematicGenerator[T]):
    def __init__(self, project_id: str, region: str, logger: Logger) -> None:
        super().__init__(
            project_id=project_id,
            region=region,
            model_name="claude-3-5-haiku@20241022",
            logger=logger,
        )


class VertexGemini15Flash(VertexAIGeminiSchematicGenerator[T]):
    def __init__(self, project_id: str, region: str, logger: Logger) -> None:
        super().__init__(
            project_id=project_id,
            region=region,
            model_name="gemini-1.5-flash",
            logger=logger,
        )


class VertexGemini15Pro(VertexAIGeminiSchematicGenerator[T]):
    def __init__(self, project_id: str, region: str, logger: Logger) -> None:
        super().__init__(
            project_id=project_id,
            region=region,
            model_name="gemini-1.5-pro",
            logger=logger,
        )


class VertexGemini20Flash(VertexAIGeminiSchematicGenerator[T]):
    def __init__(self, project_id: str, region: str, logger: Logger) -> None:
        super().__init__(
            project_id=project_id,
            region=region,
            model_name="gemini-2.0-flash",
            logger=logger,
        )


class VertexGemini25Flash(VertexAIGeminiSchematicGenerator[T]):
    def __init__(self, project_id: str, region: str, logger: Logger) -> None:
        super().__init__(
            project_id=project_id,
            region=region,
            model_name="gemini-2.5-flash",
            logger=logger,
        )

    @override
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        return await super().generate(
            prompt,
            {"thinking_config": {"thinking_budget": 0}, **hints},
        )


class VertexGemini25Pro(VertexAIGeminiSchematicGenerator[T]):
    def __init__(self, project_id: str, region: str, logger: Logger) -> None:
        super().__init__(
            project_id=project_id,
            region=region,
            model_name="gemini-2.5-pro",
            logger=logger,
        )


class VertexAIEmbedder(Embedder):
    """Embedder using Google Gen AI text embeddings"""

    supported_hints = ["title", "task_type"]

    def __init__(self, model_name: str, logger: Logger):
        self.project_id = os.environ.get("VERTEX_AI_PROJECT_ID")
        self.region = os.environ.get("VERTEX_AI_REGION", "us-central1")
        self.model_name = model_name
        self._logger = logger

        self._client = google.genai.Client(
            project=self.project_id, location=self.region, vertexai=True
        )
        self._tokenizer = VertexAIEstimatingTokenizer(self._client, model_name)

    @property
    @override
    def id(self) -> str:
        return f"vertex-ai/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return 8192


class VertexTextEmbedding004(VertexAIEmbedder):
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

        self.project_id = os.environ.get("VERTEX_AI_PROJECT_ID")
        self.region = os.environ.get("VERTEX_AI_REGION", "us-central1")
        self.embedding_model = "text-embedding-004"

        if not self.project_id:
            raise ValueError(
                "VERTEX_AI_PROJECT_ID environment variable must be set. "
                "Set this to your Google Cloud Project ID."
            )

        super().__init__(model_name="text-embedding-004", logger=logger)

    @property
    def dimensions(self) -> int:
        return 768

    @policy(
        [
            retry(
                exceptions=(
                    NotFound,
                    TooManyRequests,
                    ResourceExhausted,
                ),
                max_exceptions=3,
                wait_times=(1.0, 2.0, 4.0),
            )
        ]
    )
    @override
    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        gemini_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}
        if "task_type" not in gemini_api_arguments:
            gemini_api_arguments["task_type"] = "RETRIEVAL_DOCUMENT"

        try:
            with self._logger.operation("Embedding text with gemini"):
                response = await self._client.aio.models.embed_content(  # type: ignore
                    model=self.model_name,
                    contents=texts,  # type: ignore
                    config=cast(google.genai.types.EmbedContentConfigDict, gemini_api_arguments),
                )

            vectors = [
                data_point.values for data_point in response.embeddings or [] if data_point.values
            ]
            return EmbeddingResult(vectors=vectors)

        except TooManyRequests:
            self._logger.error(
                (
                    "Google API rate limit exceeded. Possible reasons:\n"
                    "1. Your account may have insufficient API credits.\n"
                    "2. You may be using a free-tier account with limited request capacity.\n"
                    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
                    "Recommended actions:\n"
                    "- Check your Google API account balance and billing status.\n"
                    "- Review your API usage limits in Google's dashboard.\n"
                    "- For more details on rate limits and usage tiers, visit:\n"
                    "  https://cloud.google.com/docs/quota-and-billing/quotas/quotas-overview"
                ),
            )
            raise
        except Exception as e:
            self._logger.error(f"Error during embedding: {e}")
            raise


class VertexAIService(NLPService):
    """NLP Service for Vertex AI supporting both Claude and Gemini models via appropriate APIs."""

    CLAUDE_MODELS = {
        "claude-opus-4": "claude-opus-4@20250514",
        "claude-sonnet-4": "claude-sonnet-4@20250514",
        "claude-sonnet-3.5": "claude-3-5-sonnet-v2@20241022",
        "claude-haiku-3.5": "claude-3-5-haiku@20241022",
    }

    GEMINI_MODELS = {
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-1.5-pro": "gemini-1.5-pro",
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
    }

    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        required_vars = {
            "VERTEX_AI_PROJECT_ID": "your-project-id",
            "VERTEX_AI_REGION": "us-central1",
            "VERTEX_AI_MODEL": "claude-sonnet-3.5",
        }

        missing_vars = []
        for var_name, example_value in required_vars.items():
            if not os.environ.get(var_name):
                missing_vars.append(f"export {var_name}={example_value}")

        if missing_vars:
            return f"""\
    You're using the VERTEX AI service, but required environment variables are not set.
    Please set the following environment variables before running Parlant:

    {chr(10).join(missing_vars)}
    """

        return None

    @staticmethod
    def validate_adc() -> str | None:
        """Validate that Application Default Credentials are configured."""
        try:
            credentials, project = google.auth.default()  # type: ignore
            if not credentials:
                return """\
                        No Application Default Credentials found.
                        Run 'gcloud auth application-default login' for local development.
                        """
        except Exception as e:
            return f"""\
                    Failed to load Application Default Credentials: {e}
                    Run 'gcloud auth application-default login' for local development.
                    """

        return None

    def __init__(
        self,
        logger: Logger,
    ) -> None:
        self.project_id = os.environ.get("VERTEX_AI_PROJECT_ID", "project_id")
        self.region = os.environ.get("VERTEX_AI_REGION", "us-central1")
        self.model_name = self._normalize_model_name(
            os.environ.get("VERTEX_AI_MODEL", "claude-sonnet-3.5")
        )
        self._logger = logger

        self._logger.info(
            f"Initialized VertexAIService with model {self.model_name} "
            f"in project {self.project_id}, region {self.project_id}"
        )

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name to full version string."""
        # Check if it's a short name we recognize
        if model_name in self.CLAUDE_MODELS:
            return self.CLAUDE_MODELS[model_name]
        elif model_name in self.GEMINI_MODELS:
            return self.GEMINI_MODELS[model_name]
        # Otherwise assume it's already a full model name
        return model_name

    @override
    async def get_schematic_generator(self, t: type[T]) -> SchematicGenerator[T]:
        """Get a schematic generator for the specified type."""
        provider = get_model_provider(self.model_name)

        if provider == ModelProvider.ANTHROPIC:
            if "opus-4" in self.model_name:
                primary = VertexClaudeOpus4[t](  # type: ignore
                    project_id=self.project_id, region=self.region, logger=self._logger
                )
                fallback = VertexClaudeSonnet4[t](  # type: ignore
                    project_id=self.project_id, region=self.region, logger=self._logger
                )
                return FallbackSchematicGenerator[t](  # type: ignore
                    primary, fallback, logger=self._logger
                )
            elif "sonnet-4" in self.model_name:
                return VertexClaudeSonnet4[t](  # type: ignore
                    project_id=self.project_id, region=self.region, logger=self._logger
                )
            elif "claude-3-5" in self.model_name:
                return VertexClaudeSonnet35[t](  # type: ignore
                    project_id=self.project_id, region=self.region, logger=self._logger
                )
            elif "haiku" in self.model_name:
                return VertexClaudeHaiku35[t](  # type: ignore
                    project_id=self.project_id, region=self.region, logger=self._logger
                )
            else:
                # Default to Sonnet 3.5
                return VertexClaudeSonnet35[t](  # type: ignore
                    project_id=self.project_id, region=self.region, logger=self._logger
                )

        elif provider == ModelProvider.GOOGLE:
            if "1.5-flash" in self.model_name:
                return VertexGemini15Flash[t](  # type: ignore
                    project_id=self.project_id, region=self.region, logger=self._logger
                )
            elif "1.5-pro" in self.model_name:
                return VertexGemini15Pro[t](  # type: ignore
                    project_id=self.project_id, region=self.region, logger=self._logger
                )
            elif "2.0-flash" in self.model_name:
                return VertexGemini20Flash[t](  # type: ignore
                    project_id=self.project_id, region=self.region, logger=self._logger
                )
            elif "2.5-flash" in self.model_name:
                return VertexGemini25Flash[t](  # type: ignore
                    project_id=self.project_id, region=self.region, logger=self._logger
                )
            elif "2.5-pro" in self.model_name:
                return VertexGemini25Pro[t](  # type: ignore
                    project_id=self.project_id, region=self.region, logger=self._logger
                )
            else:
                # Default to Gemini 2.5-flash
                return VertexGemini25Flash[t](  # type: ignore
                    project_id=self.project_id, region=self.region, logger=self._logger
                )

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    @override
    async def get_embedder(self) -> Embedder:
        """Get an embedder for text embeddings using Google Gen AI."""
        return VertexTextEmbedding004(logger=self._logger)

    @override
    async def get_moderation_service(self) -> ModerationService:  # @Todo - add moderation service
        """Get a moderation service."""
        return NoModeration()
