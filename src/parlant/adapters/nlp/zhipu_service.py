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

from __future__ import annotations
from itertools import chain
import time
from parlant.core.engines.alpha.guideline_matching.generic.journey.journey_backtrack_node_selection import (
    JourneyBacktrackNodeSelectionSchema,
)
from zhipuai import ZhipuAI  # type: ignore
from zhipuai.core._errors import (  # type: ignore
    APIConnectionError,
    APITimeoutError,
    APIReachLimitError,
    APIServerFlowExceedError,
    APIInternalError,
)
from typing import Any, Mapping
from typing_extensions import override
import json
import jsonfinder  # type: ignore
import os

from pydantic import ValidationError
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.core.engines.alpha.canned_response_generator import (
    CannedResponseDraftSchema,
    CannedResponseSelectionSchema,
)
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.engines.alpha.tool_calling.single_tool_batch import SingleToolBatchSchema
from parlant.core.loggers import Logger
from parlant.core.tracer import Tracer
from parlant.core.meter import Meter
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import EmbedderHints, NLPService, SchematicGeneratorHints
from parlant.core.nlp.embedding import BaseEmbedder, Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.moderation import (
    BaseModerationService,
    CustomerModerationContext,
    ModerationCheck,
    ModerationTag,
)


RATE_LIMIT_ERROR_MESSAGE = (
    "Zhipu AI API rate limit exceeded. Possible reasons:\n"
    "1. Your account may have insufficient API credits.\n"
    "2. You may be using a free-tier account with limited request capacity.\n"
    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your Zhipu AI account balance and billing status.\n"
    "- Review your API usage limits in Zhipu AI's dashboard.\n"
    "- For more details on rate limits and usage, visit:\n"
    "  https://open.bigmodel.cn/dev/api\n"
)


class ZhipuEstimatingTokenizer(EstimatingTokenizer):
    """Tokenizer for estimating token count for Zhipu AI models using tiktoken."""

    def __init__(self, model_name: str) -> None:
        """Initialize the tokenizer with a model name.

        Args:
            model_name: The name of the Zhipu AI model (e.g., 'glm-4-plus')
        """
        self.model_name = model_name
        # Use cl100k_base encoding as an approximation for Zhipu AI models
        self.encoding = tiktoken.get_encoding("cl100k_base")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        """Estimate the number of tokens in the given prompt.

        Args:
            prompt: The text to estimate token count for

        Returns:
            The estimated number of tokens
        """
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class ZhipuSchematicGenerator(BaseSchematicGenerator[T]):
    """Base class for Zhipu AI schematic generators that produce structured JSON output."""

    supported_zhipu_params = ["temperature", "max_tokens", "top_p"]
    supported_hints = supported_zhipu_params

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        tokenizer_model_name: str | None = None,
    ) -> None:
        """Initialize the Zhipu AI schematic generator.

        Args:
            model_name: The name of the Zhipu AI model (e.g., 'glm-4-plus')
            logger: Logger instance for logging operations
            meter: Meter instance for metrics
            tokenizer_model_name: Optional model name for tokenizer (defaults to model_name)
        """
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self._client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])

        self._tokenizer = ZhipuEstimatingTokenizer(
            model_name=tokenizer_model_name or self.model_name
        )

    @property
    @override
    def id(self) -> str:
        """Return the model identifier in the format 'zhipu/{model_name}'.

        Returns:
            The model identifier string
        """
        return f"zhipu/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> ZhipuEstimatingTokenizer:
        """Return the tokenizer instance.

        Returns:
            The ZhipuEstimatingTokenizer instance
        """
        return self._tokenizer

    @policy(
        [
            retry(
                exceptions=(
                    APIConnectionError,
                    APITimeoutError,
                    APIReachLimitError,
                    APIServerFlowExceedError,
                ),
            ),
            retry(APIInternalError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        """Generate structured JSON output using Zhipu AI model.

        Args:
            prompt: The prompt string or PromptBuilder instance
            hints: Optional parameters for generation (temperature, max_tokens, top_p)

        Returns:
            SchematicGenerationResult containing the parsed content and generation info
        """
        with self.logger.scope(f"Zhipu LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        """Internal method to handle the actual API call and response processing.

        Args:
            prompt: The prompt string or PromptBuilder instance
            hints: Optional parameters for generation

        Returns:
            SchematicGenerationResult containing the parsed content and generation info
        """
        # Build prompt if it's a PromptBuilder instance
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        # Filter parameters to only include supported ones
        zhipu_api_arguments = {k: v for k, v in hints.items() if k in self.supported_zhipu_params}

        # Track response time
        t_start = time.time()

        try:
            response = self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format={"type": "json_object"},
                **zhipu_api_arguments,
            )
        except (APIReachLimitError, APIServerFlowExceedError):
            self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise

        t_end = time.time()

        # Log usage information if available
        if hasattr(response, "usage") and response.usage:
            self.logger.trace(
                f"Token usage - Input: {response.usage.prompt_tokens}, "
                f"Output: {response.usage.completion_tokens}, "
                f"Total: {response.usage.total_tokens}"
            )

        # Extract raw content from response
        raw_content = response.choices[0].message.content or "{}"

        # Parse JSON from response
        try:
            json_content = json.loads(normalize_json_output(raw_content))
        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content})")
            json_content = jsonfinder.only_json(raw_content)[2]
            self.logger.warning("Found JSON content within model response; continuing...")

        # Validate against schema
        try:
            content = self.schema.model_validate(json_content)

            assert response.usage

            await record_llm_metrics(
                self.meter,
                self.model_name,
                schema_name=self.schema.__name__,
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                cached_input_tokens=0,
            )

            return SchematicGenerationResult(
                content=content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=response.usage.prompt_tokens or 0,
                        output_tokens=response.usage.completion_tokens or 0,
                    ),
                ),
            )

        except ValidationError as e:
            self.logger.error(
                f"Error: {e.json(indent=2)}\nJSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class GLM_4_Plus(ZhipuSchematicGenerator[T]):
    """GLM-4-Plus model for high-performance tasks."""

    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        """Initialize GLM-4-Plus model.

        Args:
            logger: Logger instance for logging operations
            meter: Meter instance for metrics
        """
        super().__init__(model_name="glm-4-plus", logger=logger, tracer=tracer, meter=meter)

    @property
    @override
    def max_tokens(self) -> int:
        """Return the maximum token limit for GLM-4-Plus.

        Returns:
            Maximum token count of 128K
        """
        return 128 * 1024


class GLM_4_Flash(ZhipuSchematicGenerator[T]):
    """GLM-4-Flash model for fast response tasks."""

    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        """Initialize GLM-4-Flash model.

        Args:
            logger: Logger instance for logging operations
            meter: Meter instance for metrics
        """
        super().__init__(model_name="glm-4-flash", logger=logger, tracer=tracer, meter=meter)

    @property
    @override
    def max_tokens(self) -> int:
        """Return the maximum token limit for GLM-4-Flash.

        Returns:
            Maximum token count of 128K
        """
        return 128 * 1024


class GLM_4_Air(ZhipuSchematicGenerator[T]):
    """GLM-4-Air model for lightweight tasks."""

    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        """Initialize GLM-4-Air model.

        Args:
            logger: Logger instance for logging operations
            meter: Meter instance for metrics
        """
        super().__init__(model_name="glm-4-air", logger=logger, tracer=tracer, meter=meter)

    @property
    @override
    def max_tokens(self) -> int:
        """Return the maximum token limit for GLM-4-Air.

        Returns:
            Maximum token count of 128K
        """
        return 128 * 1024


class ZhipuEmbedder(BaseEmbedder):
    """Embedder for generating text embeddings using Zhipu AI models."""

    supported_arguments = ["dimensions"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        """Initialize the Zhipu AI embedder.

        Args:
            model_name: The name of the Zhipu AI embedding model (e.g., 'embedding-3')
            logger: Logger instance for logging operations
            meter: Meter instance for metrics
        """
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self._client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])

        self._tokenizer = ZhipuEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        """Return the embedding model identifier in the format 'zhipu/{model_name}'.

        Returns:
            The model identifier string
        """
        return f"zhipu/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> ZhipuEstimatingTokenizer:
        """Return the tokenizer instance.

        Returns:
            The ZhipuEstimatingTokenizer instance
        """
        return self._tokenizer

    @policy(
        [
            retry(
                exceptions=(
                    APIConnectionError,
                    APITimeoutError,
                    APIReachLimitError,
                    APIServerFlowExceedError,
                ),
            ),
            retry(APIInternalError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        """Generate embeddings for the given texts using Zhipu AI embedding API.

        Args:
            texts: List of text strings to generate embeddings for
            hints: Optional parameters for embedding (dimensions)

        Returns:
            EmbeddingResult containing the list of embedding vectors
        """
        # Filter parameters to only include supported ones
        zhipu_api_arguments = {k: v for k, v in hints.items() if k in self.supported_arguments}

        try:
            response = self._client.embeddings.create(
                model=self.model_name,
                input=texts,
                **zhipu_api_arguments,
            )
        except (APIReachLimitError, APIServerFlowExceedError):
            self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise

        # Log usage information if available
        if hasattr(response, "usage") and response.usage:
            self.logger.trace(f"Token usage - Total: {response.usage.total_tokens}")

        # Extract embeddings from response
        embeddings = [item.embedding for item in response.data]

        return EmbeddingResult(vectors=embeddings)


class Embedding_3(ZhipuEmbedder):
    """Embedding-3 model for generating text embeddings."""

    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        """Initialize Embedding-3 model.

        Args:
            logger: Logger instance for logging operations
            meter: Meter instance for metrics
        """
        super().__init__(model_name="embedding-3", logger=logger, tracer=tracer, meter=meter)

    @property
    @override
    def max_tokens(self) -> int:
        """Return the maximum token limit for Embedding-3.

        Returns:
            Maximum token count of 8192
        """
        return 8192

    @property
    @override
    def dimensions(self) -> int:
        """Return the default embedding dimensions for Embedding-3.

        Returns:
            Default embedding dimensions of 2048
        """
        return 2048


class ZhipuModerationService(BaseModerationService):
    """Moderation service for detecting inappropriate content using Zhipu AI."""

    def __init__(self, model_name: str, logger: Logger, meter: Meter) -> None:
        """Initialize the Zhipu AI moderation service.

        Args:
            model_name: The name of the Zhipu AI moderation model
            logger: Logger instance for logging operations
            meter: Meter instance for metrics
        """
        super().__init__(logger, meter)

        self.model_name = model_name
        self._client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])

        self._hist_moderation_request_duration = meter.create_duration_histogram(
            name="moderation",
            description="Duration of moderation requests in milliseconds",
        )

    @override
    async def do_moderate(self, context: CustomerModerationContext) -> ModerationCheck:
        """Check content for inappropriate material using Zhipu AI moderation API.

        Args:
            context: The moderation context containing the message to check

        Returns:
            ModerationCheck object containing flagged status and tags
        """
        async with self._hist_moderation_request_duration.measure():
            return await self._do_moderate(context)

    async def _do_moderate(self, context: CustomerModerationContext) -> ModerationCheck:
        """Internal method to handle the actual moderation API call.

        Args:
            context: The moderation context containing the message to check

        Returns:
            ModerationCheck object containing flagged status and tags
        """

        def extract_tags(category: str) -> list[ModerationTag]:
            """Map Zhipu AI moderation categories to ModerationTag values.

            Args:
                category: The Zhipu AI category name

            Returns:
                List of corresponding ModerationTag values
            """
            mapping: dict[str, list[ModerationTag]] = {
                "sexual": ["sexual"],
                "hate": ["hate"],
                "harassment": ["harassment"],
                "violence": ["violence"],
                "self_harm": ["self-harm"],
                "self-harm": ["self-harm"],
                "illegal": ["illicit"],
                "illicit": ["illicit"],
            }

            return mapping.get(category.replace("-", "_"), [])

        response = self._client.moderations.create(
            model=self.model_name,
            input=context.message,
        )

        result = response.results[0]

        return ModerationCheck(
            flagged=result.flagged,
            tags=list(
                set(
                    chain.from_iterable(
                        extract_tags(category)
                        for category, detected in result.categories
                        if detected
                    )
                )
            ),
        )


class ZhipuService(NLPService):
    """Main NLP service class for Zhipu AI integration."""

    @staticmethod
    def verify_environment() -> str | None:
        """Verify that the environment is properly configured for Zhipu AI service.

        Returns:
            Error message string if environment is not configured correctly, None otherwise
        """
        if not os.environ.get("ZHIPUAI_API_KEY"):
            return """\
You're using the Zhipu AI NLP service, but ZHIPUAI_API_KEY is not set.
Please set ZHIPUAI_API_KEY in your environment before running Parlant.

To obtain an API key:
1. Visit https://open.bigmodel.cn/
2. Register or log in to your account
3. Create an API key in the console
4. Set the environment variable: export ZHIPUAI_API_KEY=your_api_key_here
"""

        return None

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        """Initialize the Zhipu AI service.

        Args:
            logger: Logger instance for logging operations
            meter: Meter instance for metrics
        """
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._logger.info("Initialized ZhipuService")

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> ZhipuSchematicGenerator[T]:
        """Get the appropriate schematic generator for the given schema type.

        Args:
            t: The schema type to generate for

        Returns:
            A ZhipuSchematicGenerator instance configured for the schema type
        """
        return {
            SingleToolBatchSchema: GLM_4_Flash[SingleToolBatchSchema],
            JourneyBacktrackNodeSelectionSchema: GLM_4_Plus[JourneyBacktrackNodeSelectionSchema],
            CannedResponseDraftSchema: GLM_4_Plus[CannedResponseDraftSchema],
            CannedResponseSelectionSchema: GLM_4_Plus[CannedResponseSelectionSchema],
        }.get(t, GLM_4_Flash[t])(self._logger, self._tracer, self._meter)  # type: ignore

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        """Get the embedder instance for generating text embeddings.

        Returns:
            An Embedding_3 embedder instance
        """
        return Embedding_3(self._logger, self._tracer, self._meter)

    @override
    async def get_moderation_service(self) -> BaseModerationService:
        """Get the moderation service instance for content checking.

        Returns:
            A ZhipuModerationService instance
        """
        return ZhipuModerationService(
            model_name="moderation", logger=self._logger, meter=self._meter
        )
