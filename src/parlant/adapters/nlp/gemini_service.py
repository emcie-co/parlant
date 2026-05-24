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

import os
import time
from functools import cached_property
from google.api_core.exceptions import NotFound, TooManyRequests, ResourceExhausted, ServerError
import google.genai  # type: ignore
import google.genai.types  # type: ignore
from typing import Any, Mapping, NoReturn, cast
from typing_extensions import override
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

from parlant.adapters.nlp.common import record_llm_metrics
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.meter import Meter
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.service import (
    EmbedderHints,
    ModelSize,
    NLPService,
    SchematicGeneratorHints,
    StreamingTextGeneratorHints,
)
from parlant.core.nlp.embedding import BaseEmbedder, Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    SchematicGenerationResult,
    StreamingTextGenerator,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.loggers import Logger
from parlant.core.tracer import Tracer
from parlant.core.health import HealthReporter

RATE_LIMIT_ERROR_MESSAGE = (
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


class GoogleEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, client: google.genai.Client, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        model_approximation = {
            "gemini-embedding-001": "gemini-2.5-flash",
        }.get(self._model_name, self._model_name)

        result = await self._client.aio.models.count_tokens(
            model=model_approximation,
            contents=prompt,
        )

        return int(result.total_tokens or 0)


class GeminiSchematicGenerator(BaseSchematicGenerator[T]):
    supported_hints = ["temperature", "thinking_config"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        health_reporter: HealthReporter,
    ) -> None:
        super().__init__(
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
            model_name=model_name,
        )

        self._client = google.genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        self._tokenizer = GoogleEstimatingTokenizer(client=self._client, model_name=self.model_name)

        self._model = GoogleModel(
            self.model_name,
            provider=GoogleProvider(client=self._client),
        )

    @cached_property
    def _agent(self) -> Agent[None, T]:
        # The schema is only available once the generator has been parameterized
        # (i.e. after __init__, via __orig_class__), so the agent is built lazily.
        #
        # retries=0 disables pydantic-ai's internal re-prompting on output/tool
        # validation failures; retries are handled solely by the @policy decorator.
        return Agent(self._model, output_type=self.schema, retries=0)

    @property
    @override
    def id(self) -> str:
        return f"google/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer

    @policy(
        [
            retry(
                exceptions=(
                    NotFound,
                    TooManyRequests,
                    ResourceExhausted,
                )
            ),
            retry(ServerError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self.logger.scope(f"Gemini LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        model_settings = self._build_model_settings(hints)

        t_start = time.time()
        try:
            result = await self._agent.run(prompt, model_settings=model_settings)
        except ModelHTTPError as error:
            self._raise_mapped_error(error)

        t_end = time.time()

        usage = result.usage()
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
        cached_input_tokens = usage.cache_read_tokens or 0

        await record_llm_metrics(
            self.meter,
            self.model_name,
            schema_name=self.schema.__name__,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )

        return SchematicGenerationResult(
            content=result.output,
            info=GenerationInfo(
                schema_name=self.schema.__name__,
                model=self.id,
                duration=(t_end - t_start),
                usage=UsageInfo(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    extra={"cached_input_tokens": cached_input_tokens},
                ),
            ),
        )

    def _build_model_settings(self, hints: Mapping[str, Any]) -> GoogleModelSettings:
        settings: GoogleModelSettings = {}

        if "temperature" in hints:
            settings["temperature"] = hints["temperature"]

        if "thinking_config" in hints:
            settings["google_thinking_config"] = hints["thinking_config"]

        return settings

    def _raise_mapped_error(self, error: ModelHTTPError) -> NoReturn:
        # pydantic-ai collapses Gemini's API errors into ModelHTTPError. We remap them
        # back to the google.api_core exceptions that the retry policy is keyed on, so
        # rate-limit and server errors keep retrying while other 4xx errors propagate.
        if error.status_code == 429:
            self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise TooManyRequests(str(error)) from error  # type: ignore[no-untyped-call]

        if error.status_code == 404:
            raise NotFound(str(error)) from error  # type: ignore[no-untyped-call]

        if error.status_code >= 500:
            raise ServerError(str(error)) from error  # type: ignore[no-untyped-call]

        raise error


class Gemini_2_0_Flash(GeminiSchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="gemini-2.0-flash",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_2_0_Flash_Lite(GeminiSchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="gemini-2.0-flash-lite-preview-02-05",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_2_5_Flash(GeminiSchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="gemini-2.5-flash",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
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

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_2_5_Flash_Lite(GeminiSchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="gemini-2.5-flash-lite",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
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

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_2_5_Pro(GeminiSchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="gemini-2.5-pro",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_3_5_Flash(GeminiSchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="gemini-3.5-flash",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @override
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        return await super().generate(
            prompt,
            {"thinking_config": {"thinking_level": "medium"}, **hints},
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_3_1_Pro(GeminiSchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="gemini-3.1-pro-preview",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class GoogleEmbedder(BaseEmbedder):
    supported_hints = ["title", "task_type"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        health_reporter: HealthReporter,
    ) -> None:
        super().__init__(logger, tracer, meter, model_name, health_reporter)

        self._client = google.genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self._tokenizer = GoogleEstimatingTokenizer(client=self._client, model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"google/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> GoogleEstimatingTokenizer:
        return self._tokenizer

    @policy(
        [
            retry(
                exceptions=(
                    NotFound,
                    TooManyRequests,
                    ResourceExhausted,
                )
            ),
            retry(ServerError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        gemini_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        try:
            response = await self._client.aio.models.embed_content(  # type: ignore
                model=self.model_name,
                contents=texts,  # type: ignore
                config=cast(google.genai.types.EmbedContentConfigDict, gemini_api_arguments),
            )
        except TooManyRequests:
            self.logger.error(
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

        vectors = [
            data_point.values for data_point in response.embeddings or [] if data_point.values
        ]
        return EmbeddingResult(vectors=vectors)


class GeminiTextEmbedding_001(GoogleEmbedder):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="gemini-embedding-001",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 2048

    @property
    def dimensions(self) -> int:
        return 3072


class GeminiService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("GEMINI_API_KEY"):
            return """\
You're using the GEMINI NLP service, but GEMINI_API_KEY is not set.
Please set GEMINI_API_KEY in your environment before running Parlant.
"""

        return None

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        health_reporter: HealthReporter,
    ) -> None:
        self.logger = logger
        self._tracer = tracer
        self._meter = meter

        self._health_reporter = health_reporter

        self.logger.info("Initialized GeminiService")

    @property
    @override
    def supports_streaming(self) -> bool:
        return False

    @override
    async def get_streaming_text_generator(
        self, hints: StreamingTextGeneratorHints = {}
    ) -> StreamingTextGenerator:
        raise NotImplementedError("Streaming is not supported. Check supports_streaming first.")

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> GeminiSchematicGenerator[T]:
        match hints.get("model_size", ModelSize.AUTO):
            case ModelSize.NANO:
                return Gemini_2_5_Flash_Lite[t](
                    self.logger, self._tracer, self._meter, self._health_reporter
                )  # type: ignore
            case ModelSize.MINI:
                return Gemini_2_5_Flash[t](
                    self.logger, self._tracer, self._meter, self._health_reporter
                )  # type: ignore
            case ModelSize.LARGE:
                return Gemini_2_5_Pro[t](
                    self.logger, self._tracer, self._meter, self._health_reporter
                )  # type: ignore
            case _:
                return Gemini_3_5_Flash[t](
                    self.logger, self._tracer, self._meter, self._health_reporter
                )  # type: ignore

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        return GeminiTextEmbedding_001(
            self.logger, self._tracer, self._meter, self._health_reporter
        )

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
