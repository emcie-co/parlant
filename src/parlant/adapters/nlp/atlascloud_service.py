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
import time
from typing import Any, AsyncIterator, Callable, Mapping

import jsonfinder  # type: ignore
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
import tiktoken
from typing_extensions import override

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.adapters.nlp.hugging_face import JinaAIEmbedder
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.health import HealthReporter
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.embedding import Embedder
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


ATLASCLOUD_BASE_URL = "https://api.atlascloud.ai/v1"
ATLASCLOUD_DEFAULT_MODEL = "qwen/qwen3.8-max"
ATLASCLOUD_DEFAULT_MAX_TOKENS = 8192


class AtlasCloudEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        return len(self.encoding.encode(prompt))


class AtlasCloudSchematicGenerator(BaseSchematicGenerator[T]):
    supported_params = ["temperature", "max_tokens"]
    supported_hints = supported_params + ["strict"]

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
        self._client = AsyncClient(
            base_url=ATLASCLOUD_BASE_URL,
            api_key=os.environ["ATLASCLOUD_API_KEY"],
        )
        self._tokenizer = AtlasCloudEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"atlascloud/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> AtlasCloudEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return int(os.environ.get("ATLASCLOUD_MAX_TOKENS", ATLASCLOUD_DEFAULT_MAX_TOKENS))

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

        arguments = {k: v for k, v in hints.items() if k in self.supported_params}
        arguments.setdefault("max_tokens", self.max_tokens)

        started_at = time.time()
        response = await self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            response_format={"type": "json_object"},
            **arguments,
        )
        finished_at = time.time()

        if response.usage:
            self.logger.trace(response.usage.model_dump_json(indent=2))

        raw_content = response.choices[0].message.content or "{}"
        try:
            json_content = json.loads(normalize_json_output(raw_content))
        except json.JSONDecodeError:
            self.logger.warning(
                f"Invalid JSON returned by Atlas Cloud model {self.model_name}:\n{raw_content}"
            )
            json_content = jsonfinder.only_json(raw_content)[2]
            self.logger.warning("Found JSON content within model response; continuing...")

        try:
            content = self.schema.model_validate(json_content)
        except ValidationError:
            self.logger.error(
                f"JSON content returned by Atlas Cloud model {self.model_name} "
                f"does not match the expected schema:\n{raw_content}"
            )
            raise

        assert response.usage
        cached_input_tokens = getattr(response.usage, "prompt_cache_hit_tokens", 0)
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
                duration=finished_at - started_at,
                usage=UsageInfo(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    extra={"cached_input_tokens": cached_input_tokens},
                ),
            ),
        )


class AtlasCloudStreamingTextGenerator(BaseStreamingTextGenerator):
    supported_params = ["temperature", "max_tokens"]

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
        self._client = AsyncClient(
            base_url=ATLASCLOUD_BASE_URL,
            api_key=os.environ["ATLASCLOUD_API_KEY"],
        )
        self._tokenizer = AtlasCloudEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"atlascloud-streaming/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> AtlasCloudEstimatingTokenizer:
        return self._tokenizer

    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> tuple[AsyncIterator[str | None], Callable[[], UsageInfo]]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        arguments = {k: v for k, v in hints.items() if k in self.supported_params}
        arguments.setdefault(
            "max_tokens",
            int(os.environ.get("ATLASCLOUD_MAX_TOKENS", ATLASCLOUD_DEFAULT_MAX_TOKENS)),
        )
        stream = await self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            stream=True,
            stream_options={"include_usage": True},
            **arguments,
        )

        usage_info: UsageInfo | None = None

        async def chunk_generator() -> AsyncIterator[str | None]:
            nonlocal usage_info
            async for chunk in stream:
                if chunk.usage:
                    cached_input_tokens = getattr(chunk.usage, "prompt_cache_hit_tokens", 0)
                    usage_info = UsageInfo(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        extra={"cached_input_tokens": cached_input_tokens},
                    )
                    await record_llm_metrics(
                        self.meter,
                        self.model_name,
                        schema_name="streaming",
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        cached_input_tokens=cached_input_tokens,
                    )

                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

            yield None

        def get_usage() -> UsageInfo:
            return usage_info or UsageInfo(input_tokens=0, output_tokens=0)

        return chunk_generator(), get_usage


class AtlasCloudService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        if not os.environ.get("ATLASCLOUD_API_KEY"):
            return """\
You're using the Atlas Cloud NLP service, but ATLASCLOUD_API_KEY is not set.
Please set ATLASCLOUD_API_KEY in your environment before running Parlant.
"""
        return None

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        health_reporter: HealthReporter,
    ) -> None:
        self.model_name = os.environ.get("ATLASCLOUD_MODEL", ATLASCLOUD_DEFAULT_MODEL)
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._health_reporter = health_reporter
        self._logger.info(f"Initialized AtlasCloudService with model: {self.model_name}")

    @property
    @override
    def supports_streaming(self) -> bool:
        return True

    @override
    async def get_streaming_text_generator(
        self, hints: StreamingTextGeneratorHints = {}
    ) -> StreamingTextGenerator:
        return AtlasCloudStreamingTextGenerator(
            model_name=self.model_name,
            logger=self._logger,
            tracer=self._tracer,
            meter=self._meter,
            health_reporter=self._health_reporter,
        )

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> AtlasCloudSchematicGenerator[T]:
        return AtlasCloudSchematicGenerator[t](  # type: ignore
            model_name=self.model_name,
            logger=self._logger,
            tracer=self._tracer,
            meter=self._meter,
            health_reporter=self._health_reporter,
        )

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        return JinaAIEmbedder(
            self._logger,
            self._tracer,
            self._meter,
            self._health_reporter,
        )

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
