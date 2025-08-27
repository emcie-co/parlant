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
from anthropic import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AsyncAnthropic,
    InternalServerError,
    RateLimitError,
)  # type: ignore
from typing import Any, Mapping
from typing_extensions import override
import jsonfinder  # type: ignore
import os

from parlant.adapters.nlp.common import normalize_json_output
from parlant.adapters.nlp.hugging_face import JinaAIEmbedder
from parlant.core.engines.alpha.canned_response_generator import CannedResponseSelectionSchema
from parlant.core.engines.alpha.guideline_matching.generic.disambiguation_batch import (
    DisambiguationGuidelineMatchesSchema,
)
from parlant.core.engines.alpha.guideline_matching.generic.journey_node_selection_batch import (
    JourneyNodeSelectionSchema,
)
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.nlp.embedding import Embedder
from parlant.core.nlp.generation import (
    T,
    SchematicGenerationResult,
    SchematicGenerator,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.loggers import Logger
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.tokenization import EstimatingTokenizer


class AnthropicEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, client: AsyncAnthropic, model_name: str) -> None:
        self._client = client
        self.model_name = model_name

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        result = await self._client.messages.count_tokens(
            model=self.model_name,
            messages=[{"role": "assistant", "content": prompt}],
        )

        return result.input_tokens


class AnthropicAISchematicGenerator(SchematicGenerator[T]):
    supported_hints = ["temperature"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
    ) -> None:
        self.model_name = model_name
        self._logger = logger

        self._client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._estimating_tokenizer = AnthropicEstimatingTokenizer(self._client, model_name)

    @property
    @override
    def id(self) -> str:
        return f"anthropic/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> AnthropicEstimatingTokenizer:
        return self._estimating_tokenizer

    @policy(
        [
            retry(
                exceptions=(
                    APIConnectionError,
                    APITimeoutError,
                    RateLimitError,
                    APIResponseValidationError,
                )
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
                max_tokens=4096,
                **anthropic_api_arguments,
            )
        except RateLimitError:
            self._logger.error(
                (
                    "Anthropic API rate limit exceeded. Possible reasons:\n"
                    "1. Your account may have insufficient API credits.\n"
                    "2. You may be using a free-tier account with limited request capacity.\n"
                    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
                    "Recommended actions:\n"
                    "- Check your Anthropic account balance and billing status.\n"
                    "- Review your API usage limits in Anthropic's dashboard.\n"
                    "- For more details on rate limits and usage tiers, visit:\n"
                    "  https://docs.anthropic.com/claude/reference/rate-limits \n"
                ),
            )
            raise

        t_end = time.time()

        if response.usage:
            self._logger.trace(response.usage.model_dump_json(indent=2))

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


class Claude_Sonnet_3_5(AnthropicAISchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="claude-3-5-sonnet-20241022",
            logger=logger,
        )

    @override
    @property
    def max_tokens(self) -> int:
        return 200 * 1024


class Claude_Sonnet_4(AnthropicAISchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="claude-sonnet-4-20250514",
            logger=logger,
        )

    @override
    @property
    def max_tokens(self) -> int:
        return 200 * 1024


class Claude_Opus_4_1(AnthropicAISchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="claude-opus-4-1-20250805",
            logger=logger,
        )

    @override
    @property
    def max_tokens(self) -> int:
        return 200 * 1024


class AnthropicService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("ANTHROPIC_API_KEY"):
            return """\
You're using the Anthropic NLP service, but ANTHROPIC_API_KEY is not set.
Please set ANTHROPIC_API_KEY in your environment before running Parlant.
"""

        return None

    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._logger.info("Initialized AnthropicService")

    @override
    async def get_schematic_generator(self, t: type[T]) -> AnthropicAISchematicGenerator[T]:
        if (
            t == JourneyNodeSelectionSchema
            or t == DisambiguationGuidelineMatchesSchema
            or t == CannedResponseSelectionSchema
        ):
            return Claude_Opus_4_1[t](self._logger)  # type: ignore
        return Claude_Sonnet_4[t](self._logger)  # type: ignore

    @override
    async def get_embedder(self) -> Embedder:
        return JinaAIEmbedder()

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
