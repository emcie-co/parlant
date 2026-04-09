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
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AsyncClient,
    ConflictError,
    InternalServerError,
    RateLimitError,
)
import re
from typing import Any, AsyncIterator, Callable, Mapping
from typing_extensions import override
import json
import jsonfinder  # type: ignore
import os

from pydantic import ValidationError
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.adapters.nlp.hugging_face import JinaAIEmbedder
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.tracer import Tracer
from parlant.core.meter import Meter
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import (
    EmbedderHints,
    NLPService,
    SchematicGeneratorHints,
    StreamingTextGeneratorHints,
)
from parlant.core.nlp.embedding import Embedder
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    BaseStreamingTextGenerator,
    SchematicGenerationResult,
    StreamingTextGenerator,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.moderation import (
    ModerationService,
    NoModeration,
)


NOVITA_BASE_URL = "https://api.novita.ai/openai"
NOVITA_DEFAULT_MODEL = "moonshotai/kimi-k2.5"


class NovitaEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class NovitaSchematicGenerator(BaseSchematicGenerator[T]):
    supported_novita_params = ["temperature", "logit_bias", "max_tokens"]
    supported_hints = supported_novita_params + ["strict"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self._client = AsyncClient(
            base_url=NOVITA_BASE_URL,
            api_key=os.environ["NOVITA_API_KEY"],
        )

        self._tokenizer = NovitaEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"novita/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> NovitaEstimatingTokenizer:
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
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self.logger.scope(f"Novita LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        novita_api_arguments = {
            k: v for k, v in hints.items() if k in self.supported_novita_params
        }

        t_start = time.time()
        response = await self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            max_tokens=8192,
            response_format={"type": "json_object"},
            **novita_api_arguments,
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

            await record_llm_metrics(
                self.meter,
                self.model_name,
                schema_name=self.schema.__name__,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cached_input_tokens=getattr(
                    response.usage,
                    "prompt_cache_hit_tokens",
                    0,
                ),
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
            self.logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class Novita_KimiK2(NovitaSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="moonshotai/kimi-k2.5", logger=logger, tracer=tracer, meter=meter
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 262_144


class Novita_DeepSeekV3(NovitaSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="deepseek/deepseek-v3.2", logger=logger, tracer=tracer, meter=meter
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 163_840


class Novita_GLM5(NovitaSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="zai-org/glm-5.1", logger=logger, tracer=tracer, meter=meter
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 204_800


class Novita_MinimaxM2(NovitaSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="minimax/minimax-m2.7", logger=logger, tracer=tracer, meter=meter
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 204_800


class CustomNovitaSchematicGenerator(NovitaSchematicGenerator[T]):
    """Generic Novita AI generator that accepts any model name."""

    def __init__(self, model_name: str, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name=model_name,
            logger=logger,
            tracer=tracer,
            meter=meter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


# ============================================================================
# Streaming Text Generators
# ============================================================================

# Pattern to detect word boundaries for chunking
# Matches after any whitespace character
_WORD_BOUNDARY_PATTERN = re.compile(r"(?<=\s)")

# Number of words to buffer before yielding a chunk
_WORDS_PER_CHUNK = 3


class NovitaStreamingTextGenerator(BaseStreamingTextGenerator):
    """Streaming text generator using Novita AI's OpenAI-compatible streaming API.

    Buffers tokens into word-sized chunks for smoother frontend rendering.
    """

    supported_novita_params = ["temperature", "max_tokens"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self._client = AsyncClient(
            base_url=NOVITA_BASE_URL,
            api_key=os.environ["NOVITA_API_KEY"],
        )
        self._tokenizer = NovitaEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"novita-streaming/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> NovitaEstimatingTokenizer:
        return self._tokenizer

    def _list_arguments(self, hints: Mapping[str, Any]) -> Mapping[str, Any]:
        return {k: v for k, v in hints.items() if k in self.supported_novita_params}

    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> tuple[AsyncIterator[str | None], Callable[[], UsageInfo]]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        novita_api_arguments = self._list_arguments(hints)

        stream = await self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            stream=True,
            stream_options={"include_usage": True},
            **novita_api_arguments,
        )

        # Track usage from final chunk
        usage_info: UsageInfo | None = None

        async def chunk_generator() -> AsyncIterator[str | None]:
            nonlocal usage_info

            # Buffer for accumulating tokens into word-sized chunks
            buffer = ""

            async for chunk in stream:
                # Check for usage in final chunk (when stream_options include_usage is set)
                if chunk.usage is not None:
                    self.logger.trace(chunk.usage.model_dump_json(indent=2))

                    cached_tokens = getattr(
                        chunk.usage,
                        "prompt_cache_hit_tokens",
                        0,
                    ) or 0

                    usage_info = UsageInfo(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        extra={"cached_input_tokens": cached_tokens},
                    )

                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    buffer += token

                    # Count word boundaries in buffer
                    boundaries = list(_WORD_BOUNDARY_PATTERN.finditer(buffer))
                    if len(boundaries) >= _WORDS_PER_CHUNK:
                        # Yield up to the last complete word boundary
                        last_boundary = boundaries[_WORDS_PER_CHUNK - 1]
                        chunk_text = buffer[: last_boundary.end()]
                        buffer = buffer[last_boundary.end() :]
                        yield chunk_text

            # Yield any remaining content in the buffer
            if buffer:
                yield buffer

            # Record metrics if we have usage info
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

            # Signal completion
            yield None

        def get_usage() -> UsageInfo:
            if usage_info is None:
                # Fallback if usage wasn't available
                return UsageInfo(input_tokens=0, output_tokens=0)
            return usage_info

        return chunk_generator(), get_usage


class NovitaService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("NOVITA_API_KEY"):
            return """\
You're using the Novita AI NLP service, but NOVITA_API_KEY is not set.
Please set NOVITA_API_KEY in your environment before running Parlant.
"""

        return None

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        self.model_name = os.environ.get("NOVITA_MODEL", NOVITA_DEFAULT_MODEL)
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._logger.info(f"Initialized NovitaService with model: {self.model_name}")

    @property
    @override
    def supports_streaming(self) -> bool:
        return True

    @override
    async def get_streaming_text_generator(
        self, hints: StreamingTextGeneratorHints = {}
    ) -> StreamingTextGenerator:
        return NovitaStreamingTextGenerator(
            model_name=self.model_name,
            logger=self._logger,
            tracer=self._tracer,
            meter=self._meter,
        )

    def _get_specialized_generator_class(
        self,
        model_name: str,
        schema_type: type[T],
    ) -> Callable[[Logger, Tracer, Meter], NovitaSchematicGenerator[T]] | None:
        """Returns the specialized generator class for known models, or None for custom models."""
        model_to_class: dict[
            str, Callable[[Logger, Tracer, Meter], NovitaSchematicGenerator[T]]
        ] = {
            "moonshotai/kimi-k2.5": Novita_KimiK2[schema_type],  # type: ignore
            "deepseek/deepseek-v3.2": Novita_DeepSeekV3[schema_type],  # type: ignore
            "zai-org/glm-5.1": Novita_GLM5[schema_type],  # type: ignore
            "minimax/minimax-m2.7": Novita_MinimaxM2[schema_type],  # type: ignore
        }

        return model_to_class.get(model_name)

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> NovitaSchematicGenerator[T]:
        specialized_class = self._get_specialized_generator_class(self.model_name, schema_type=t)

        if specialized_class:
            self._logger.debug(f"Using specialized generator for model: {self.model_name}")
            return specialized_class(self._logger, self._tracer, self._meter)
        else:
            self._logger.debug(f"Using custom generator for model: {self.model_name}")
            return CustomNovitaSchematicGenerator[t](  # type: ignore
                model_name=self.model_name,
                logger=self._logger,
                tracer=self._tracer,
                meter=self._meter,
            )

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        return JinaAIEmbedder(self._logger, self._tracer, self._meter)

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
