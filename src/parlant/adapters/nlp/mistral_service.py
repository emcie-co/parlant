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
import time
from typing import Any, Mapping
from typing_extensions import override
import json
import jsonfinder  # type: ignore
import os

from pydantic import ValidationError
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.core.engines.alpha.canned_response_generator import CannedResponseSelectionSchema
from parlant.core.engines.alpha.guideline_matching.generic.disambiguation_batch import (
    DisambiguationGuidelineMatchesSchema,
)
from parlant.core.engines.alpha.guideline_matching.generic.journey.journey_backtrack_node_selection import (
    JourneyBacktrackNodeSelectionSchema,
)
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
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
    ModerationService,
    ModerationTag,
)

try:
    from mistralai import Mistral
    from mistralai.models import SDKError, HTTPValidationError
except ImportError:
    Mistral = None  # type: ignore
    SDKError = Exception  # type: ignore
    HTTPValidationError = Exception  # type: ignore


RATE_LIMIT_ERROR_MESSAGE = (
    "Mistral AI API rate limit exceeded. Possible reasons:\n"
    "1. Your account may have insufficient API credits.\n"
    "2. You may be using a free-tier account with limited request capacity.\n"
    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your Mistral AI account balance and billing status.\n"
    "- Review your API usage limits in Mistral AI's dashboard.\n"
    "- For more details on rate limits and usage tiers, visit:\n"
    "  https://docs.mistral.ai/api/\n"
)


class MistralEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # Use GPT-4o encoding as approximation for Mistral models
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class MistralSchematicGenerator(BaseSchematicGenerator[T]):
    supported_mistral_params = ["temperature", "max_tokens"]
    supported_hints = supported_mistral_params

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self._tokenizer = MistralEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"mistral/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> MistralEstimatingTokenizer:
        return self._tokenizer

    @policy(
        [
            retry(
                exceptions=(
                    ConnectionError,
                    TimeoutError,
                    SDKError,
                    HTTPValidationError,
                ),
            ),
        ]
    )
    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self.logger.scope(f"Mistral LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        mistral_api_arguments = {
            k: v for k, v in hints.items() if k in self.supported_mistral_params
        }

        t_start = time.time()
        try:
            response = await self._client.chat.complete_async(
                messages=[{"role": "user", "content": prompt}],  # type: ignore[arg-type]
                model=self.model_name,
                response_format={"type": "json_object"},  # type: ignore[arg-type]
                **mistral_api_arguments,
            )
        except SDKError as e:
            if "rate" in str(e).lower() or "429" in str(e):
                self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise

        t_end = time.time()

        if response.usage:
            self.logger.trace(
                f"Usage: input_tokens={response.usage.prompt_tokens}, "
                f"output_tokens={response.usage.completion_tokens}"
            )

        raw_content = response.choices[0].message.content or "{}"

        try:
            # Convert content to string if needed
            content_str = raw_content if isinstance(raw_content, str) else str(raw_content)
            json_content = json.loads(normalize_json_output(content_str))
        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content})")
            json_content = jsonfinder.only_json(raw_content)[2]
            self.logger.warning("Found JSON content within model response; continuing...")

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


class Mistral_Large_2411(MistralSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(model_name="mistral-large-2411", logger=logger, tracer=tracer, meter=meter)

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class Mistral_Medium_2508(MistralSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="mistral-medium-2508", logger=logger, tracer=tracer, meter=meter
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class Mistral_Small_2506(MistralSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(model_name="mistral-small-2506", logger=logger, tracer=tracer, meter=meter)

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class MistralEmbedder(BaseEmbedder):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name="mistral-embed")
        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self._tokenizer = MistralEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"mistral/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> MistralEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        return 1024

    @policy(
        [
            retry(
                exceptions=(
                    ConnectionError,
                    TimeoutError,
                    SDKError,
                    HTTPValidationError,
                ),
            ),
        ]
    )
    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        try:
            response = await self._client.embeddings.create_async(
                model=self.model_name,
                inputs=texts,
            )
        except SDKError as e:
            if "rate" in str(e).lower() or "429" in str(e):
                self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise

        vectors = [
            data_point.embedding if data_point.embedding else [] for data_point in response.data
        ]
        return EmbeddingResult(vectors=vectors)


class MistralModerationService(BaseModerationService):
    def __init__(self, logger: Logger, meter: Meter) -> None:
        super().__init__(logger=logger, meter=meter)

        self.model_name = "mistral-moderation-2411"
        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    @override
    async def do_moderate(self, context: CustomerModerationContext) -> ModerationCheck:
        def extract_tags(category: str) -> list[ModerationTag]:
            mapping: dict[str, list[ModerationTag]] = {
                "sexual": ["sexual"],
                "hate_and_discrimination": ["hate"],
                "violence_and_threats": ["violence"],
                "dangerous_and_criminal_content": ["illicit"],
                "selfharm": ["self-harm"],
                "health": ["illicit"],
                "financial": ["illicit"],
                "law": ["illicit"],
                "pii": ["illicit"],
            }

            return mapping.get(category.replace("-", "_").replace(" ", "_").lower(), [])

        response = await self._client.classifiers.moderate_chat_async(
            model=self.model_name,
            inputs=[{"role": "user", "content": context.message}],  # type: ignore[arg-type]
        )

        result = response.results[0]

        flagged = False
        all_tags: list[ModerationTag] = []

        if result.categories:
            for category_result in result.categories:
                # Type check since the API may return different formats
                if hasattr(category_result, "category_scores") and category_result.category_scores:
                    # Check if any score indicates flagged content (threshold can be adjusted)
                    for score_item in category_result.category_scores:
                        if (
                            hasattr(score_item, "score")
                            and score_item.score
                            and score_item.score > 0.5
                        ):
                            flagged = True
                            if hasattr(category_result, "category"):
                                all_tags.extend(extract_tags(str(category_result.category)))
                            break

        return ModerationCheck(
            flagged=flagged,
            tags=list(set(all_tags)),
        )


class MistralService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("MISTRAL_API_KEY"):
            return """\
You're using the Mistral NLP service, but MISTRAL_API_KEY is not set.
Please set MISTRAL_API_KEY in your environment before running Parlant.
"""

        return None

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._logger.info("Initialized MistralService")

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> MistralSchematicGenerator[T]:
        if (
            t == JourneyBacktrackNodeSelectionSchema
            or t == DisambiguationGuidelineMatchesSchema
            or t == CannedResponseSelectionSchema
        ):
            return Mistral_Large_2411[t](self._logger, self._tracer, self._meter)  # type: ignore
        return Mistral_Medium_2508[t](self._logger, self._tracer, self._meter)  # type: ignore

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        return MistralEmbedder(self._logger, self._tracer, self._meter)

    @override
    async def get_moderation_service(self) -> ModerationService:
        return MistralModerationService(self._logger, self._meter)
