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

import json
import os
import re
import time
from types import MappingProxyType
from typing import Any, AsyncIterator, Callable, Mapping
from urllib.parse import urlparse

import jsonfinder  # type: ignore
import tiktoken
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AsyncClient,
    ConflictError,
    InternalServerError,
    RateLimitError,
)
from pydantic import ValidationError
from typing_extensions import override

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.health import HealthReporter
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.embedding import BaseEmbedder, Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    BaseStreamingTextGenerator,
    SchematicGenerationResult,
    StreamingTextGenerator,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.service import (
    EmbedderHints,
    NLPService,
    SchematicGeneratorHints,
    StreamingTextGeneratorHints,
)
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.tracer import Tracer

AIMLAPI_BASE_URL = "https://api.aimlapi.com/v1"
AIMLAPI_HOST = "api.aimlapi.com"
AIMLAPI_DEFAULT_MODEL = "openai/gpt-4.1"
AIMLAPI_DEFAULT_EMBEDDER_MODEL = "openai/text-embedding-3-large"

# aimlapi.com validates a documented set of optional chat-completion fields as
# "present but wrong type" when they are sent as an explicit JSON null, and answers
# 400 (`{"error": {"details": [{"path": "tools", "reason": "Expected array, received
# null"}]}}`). OpenAI itself accepts null for all of them, so a client that forwards
# an unset optional as None — the natural thing to write — fails on every request.
#
# The set below is what the provider currently rejects. It is kept as documentation;
# `omit_unset_arguments` drops *every* None, which is both simpler and strictly safer.
NULL_REJECTING_REQUEST_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "seed",
        "tools",
        "tool_choice",
        "response_format",
        "stream",
        "stream_options",
        "parallel_tool_calls",
        "max_tokens",
        "max_completion_tokens",
    }
)

_ATTRIBUTION_HEADERS: Mapping[str, str] = MappingProxyType(
    {
        # HTTP-Referer and X-Title identify the *calling* application, not the provider.
        "HTTP-Referer": "https://github.com/emcie-co/parlant",
        "X-Title": "Parlant",
        "X-AIMLAPI-Partner-ID": "part_UOT3mCwOdpOUQKX2gIvYrCmv",
        "X-AIMLAPI-Source": "agent/parlant",
    }
)


def omit_unset_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Drops keys whose value is None instead of serializing them as JSON null.

    See NULL_REJECTING_REQUEST_FIELDS: aimlapi.com answers 400 for several optional
    fields when they arrive as null, so "unset" has to mean "absent from the payload".
    """
    return {k: v for k, v in arguments.items() if v is not None}


def build_attribution_headers(base_url: str = AIMLAPI_BASE_URL) -> dict[str, str]:
    """Builds a fresh header dict for a request to aimlapi.com.

    Returns an empty dict for any other host, so that attribution cannot ride along
    to a different provider or to a proxy that merely fronts the same API.
    """
    if urlparse(base_url).hostname != AIMLAPI_HOST:
        return {}

    headers = dict(_ATTRIBUTION_HEADERS)

    if referer := os.environ.get("AIMLAPI_HTTP_REFERER"):
        headers["HTTP-Referer"] = referer

    if site_name := os.environ.get("AIMLAPI_SITE_NAME"):
        headers["X-Title"] = site_name

    return headers


def _create_client() -> AsyncClient:
    return AsyncClient(
        base_url=AIMLAPI_BASE_URL,
        api_key=os.environ["AIMLAPI_API_KEY"],
        default_headers=build_attribution_headers(),
    )


class AIMLAPIEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # aimlapi.com routes many vendors; gpt-4o's encoding is a good general estimate.
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class AIMLAPISchematicGenerator(BaseSchematicGenerator[T]):
    supported_aimlapi_params = ["temperature", "top_p", "logit_bias", "max_tokens"]
    supported_hints = supported_aimlapi_params + ["strict"]

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

        self._client = _create_client()
        self._tokenizer = AIMLAPIEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"aimlapi/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> AIMLAPIEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024

    def build_request_arguments(self, hints: Mapping[str, Any]) -> dict[str, Any]:
        return omit_unset_arguments(
            {k: v for k, v in hints.items() if k in self.supported_aimlapi_params}
        )

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
        with self.logger.scope(f"AI/ML API LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        aimlapi_api_arguments = self.build_request_arguments(hints)

        t_start = time.time()
        response = await self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            response_format={"type": "json_object"},
            **omit_unset_arguments({"max_tokens": 8192, **aimlapi_api_arguments}),
        )
        t_end = time.time()

        if response.usage:
            self.logger.trace(response.usage.model_dump_json(indent=2))

        raw_content = response.choices[0].message.content or "{}"

        try:
            json_content = json.loads(normalize_json_output(raw_content))
        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content})")
            json_content = jsonfinder.only_json(raw_content)[2]
            self.logger.warning("Found JSON content within model response; continuing...")

        try:
            content = self.schema.model_validate(json_content)

            assert response.usage

            cached_input_tokens = (
                getattr(response.usage, "prompt_cache_hit_tokens", 0)
                or getattr(
                    getattr(response.usage, "prompt_tokens_details", None), "cached_tokens", 0
                )
                or 0
            )

            await record_llm_metrics(
                self.meter,
                self.model_name,
                schema_name=self.schema.__name__,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cached_input_tokens=cached_input_tokens,
            )

            return SchematicGenerationResult(
                content=content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        extra={"cached_input_tokens": cached_input_tokens},
                    ),
                ),
            )
        except ValidationError:
            self.logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class AIMLAPI_GPT_4_1(AIMLAPISchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="openai/gpt-4.1",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1_047_576


class AIMLAPI_GPT_4_1_Mini(AIMLAPISchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="openai/gpt-4.1-mini",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1_000_000


class AIMLAPI_ClaudeSonnet45(AIMLAPISchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="anthropic/claude-sonnet-4.5",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 200 * 1024


class AIMLAPI_Gemini25Flash(AIMLAPISchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="google/gemini-2.5-flash",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1_000_000


# Pattern to detect word boundaries for chunking; matches after any whitespace character.
_WORD_BOUNDARY_PATTERN = re.compile(r"(?<=\s)")

# Number of words to buffer before yielding a chunk.
_WORDS_PER_CHUNK = 3


class AIMLAPIStreamingTextGenerator(BaseStreamingTextGenerator):
    """Streaming text generator over aimlapi.com's OpenAI-compatible streaming API.

    Buffers tokens into word-sized chunks for smoother frontend rendering.
    """

    supported_aimlapi_params = ["temperature", "top_p", "max_tokens"]

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

        self._client = _create_client()
        self._tokenizer = AIMLAPIEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"aimlapi-streaming/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> AIMLAPIEstimatingTokenizer:
        return self._tokenizer

    def build_request_arguments(self, hints: Mapping[str, Any]) -> dict[str, Any]:
        return omit_unset_arguments(
            {k: v for k, v in hints.items() if k in self.supported_aimlapi_params}
        )

    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> tuple[AsyncIterator[str | None], Callable[[], UsageInfo]]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        aimlapi_api_arguments = self.build_request_arguments(hints)

        stream = await self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            stream=True,
            stream_options={"include_usage": True},
            **aimlapi_api_arguments,
        )

        usage_info: UsageInfo | None = None

        async def chunk_generator() -> AsyncIterator[str | None]:
            nonlocal usage_info

            buffer = ""

            async for chunk in stream:
                if chunk.usage is not None:
                    self.logger.trace(chunk.usage.model_dump_json(indent=2))

                    cached_tokens = getattr(chunk.usage, "prompt_cache_hit_tokens", 0) or 0

                    usage_info = UsageInfo(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        extra={"cached_input_tokens": cached_tokens},
                    )

                if chunk.choices and chunk.choices[0].delta.content:
                    buffer += chunk.choices[0].delta.content

                    boundaries = list(_WORD_BOUNDARY_PATTERN.finditer(buffer))
                    if len(boundaries) >= _WORDS_PER_CHUNK:
                        last_boundary = boundaries[_WORDS_PER_CHUNK - 1]
                        chunk_text = buffer[: last_boundary.end()]
                        buffer = buffer[last_boundary.end() :]
                        yield chunk_text

            if buffer:
                yield buffer

            if usage_info is not None:
                await record_llm_metrics(
                    self.meter,
                    self.model_name,
                    schema_name="streaming",
                    input_tokens=usage_info.input_tokens,
                    output_tokens=usage_info.output_tokens,
                    cached_input_tokens=usage_info.extra.get("cached_input_tokens", 0)
                    if usage_info.extra
                    else 0,
                )

            yield None

        def get_usage() -> UsageInfo:
            if usage_info is None:
                return UsageInfo(input_tokens=0, output_tokens=0)
            return usage_info

        return chunk_generator(), get_usage


class AIMLAPIEmbedder(BaseEmbedder):
    supported_arguments = ["dimensions"]

    _KNOWN_DIMENSIONS: Mapping[str, int] = MappingProxyType(
        {
            "openai/text-embedding-3-large": 3072,
            "openai/text-embedding-3-small": 1536,
            "openai/text-embedding-ada-002": 1536,
            "alibaba/text-embedding-v4": 1024,
        }
    )

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        health_reporter: HealthReporter,
    ) -> None:
        super().__init__(logger, tracer, meter, model_name, health_reporter)

        self._client = _create_client()
        self._tokenizer = AIMLAPIEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"aimlapi/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> AIMLAPIEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    @override
    def dimensions(self) -> int:
        if dimensions := os.environ.get("AIMLAPI_EMBEDDER_DIMENSIONS"):
            return int(dimensions)

        return self._KNOWN_DIMENSIONS.get(self.model_name, 1536)

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
        filtered_hints = omit_unset_arguments(
            {k: v for k, v in hints.items() if k in self.supported_arguments}
        )

        response = await self._client.embeddings.create(
            model=self.model_name,
            input=texts,
            **filtered_hints,
        )

        vectors = [data_point.embedding for data_point in response.data]

        return EmbeddingResult(vectors=vectors)


class AIMLAPIService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("AIMLAPI_API_KEY"):
            return """\
You're using the aimlapi.com NLP service, but AIMLAPI_API_KEY is not set.
Please set AIMLAPI_API_KEY in your environment before running Parlant.
"""

        return None

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        health_reporter: HealthReporter,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._health_reporter = health_reporter

        self.model_name = os.environ.get("AIMLAPI_MODEL", AIMLAPI_DEFAULT_MODEL)
        self.embedder_model_name = os.environ.get(
            "AIMLAPI_EMBEDDER_MODEL", AIMLAPI_DEFAULT_EMBEDDER_MODEL
        )

        self._logger.info(f"Initialized AIMLAPIService with model: {self.model_name}")
        self._logger.info(f"aimlapi.com embedder model name: {self.embedder_model_name}")

        embedder_model = self.embedder_model_name

        class DynamicAIMLAPIEmbedder(AIMLAPIEmbedder):
            def __init__(
                self,
                logger: Logger,
                tracer: Tracer,
                meter: Meter,
                health_reporter: HealthReporter,
            ) -> None:
                super().__init__(
                    model_name=embedder_model,
                    logger=logger,
                    tracer=tracer,
                    meter=meter,
                    health_reporter=health_reporter,
                )

        self._embedder_class = DynamicAIMLAPIEmbedder

    @property
    @override
    def supports_streaming(self) -> bool:
        return True

    @override
    async def get_streaming_text_generator(
        self, hints: StreamingTextGeneratorHints = {}
    ) -> StreamingTextGenerator:
        return AIMLAPIStreamingTextGenerator(
            model_name=self.model_name,
            logger=self._logger,
            tracer=self._tracer,
            meter=self._meter,
            health_reporter=self._health_reporter,
        )

    def _get_specialized_generator_class(
        self,
        model_name: str,
        schema_type: type[T],
    ) -> Callable[[Logger, Tracer, Meter, HealthReporter], AIMLAPISchematicGenerator[T]] | None:
        """Returns the specialized generator class for known models, or None for custom models."""
        model_to_class: dict[
            str, Callable[[Logger, Tracer, Meter, HealthReporter], AIMLAPISchematicGenerator[T]]
        ] = {
            "openai/gpt-4.1": AIMLAPI_GPT_4_1[schema_type],  # type: ignore
            "openai/gpt-4.1-mini": AIMLAPI_GPT_4_1_Mini[schema_type],  # type: ignore
            "anthropic/claude-sonnet-4.5": AIMLAPI_ClaudeSonnet45[schema_type],  # type: ignore
            "google/gemini-2.5-flash": AIMLAPI_Gemini25Flash[schema_type],  # type: ignore
        }

        return model_to_class.get(model_name)

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> AIMLAPISchematicGenerator[T]:
        if specialized_class := self._get_specialized_generator_class(
            self.model_name, schema_type=t
        ):
            self._logger.debug(f"Using specialized generator for model: {self.model_name}")
            return specialized_class(self._logger, self._tracer, self._meter, self._health_reporter)

        self._logger.debug(f"Using custom generator for model: {self.model_name}")

        max_tokens = int(os.environ.get("AIMLAPI_MAX_TOKENS", 128 * 1024))

        class DynamicAIMLAPISchematicGenerator(AIMLAPISchematicGenerator[T]):
            @property
            @override
            def max_tokens(self) -> int:
                return max_tokens

        return DynamicAIMLAPISchematicGenerator[t](  # type: ignore
            model_name=self.model_name,
            logger=self._logger,
            tracer=self._tracer,
            meter=self._meter,
            health_reporter=self._health_reporter,
        )

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        return self._embedder_class(
            logger=self._logger,
            tracer=self._tracer,
            meter=self._meter,
            health_reporter=self._health_reporter,
        )

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
