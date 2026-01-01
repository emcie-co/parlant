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
# Maintainer: Ji Qing <jiqing19861123@163.com>

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
from typing import Any, Callable, Mapping
from typing_extensions import override
import json
import jsonfinder  # type: ignore
import os

from pydantic import ValidationError
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
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
    ModerationService,
    NoModeration,
)
from parlant.core.tracer import Tracer

RATE_LIMIT_ERROR_MESSAGE = """\
Qwen API rate limit exceeded. Possible reasons:
1. Your account may have insufficient API credits.
2. You may be using a free-tier account with limited request capacity.
3. You might have exceeded the requests-per-minute limit for your account.

Recommended actions:
- Check your Qwen account balance and billing status.
- Review your API usage limits in Qwen's dashboard.
- For more details on rate limits and usage tiers, visit:
    https://docs.bigmodel.cn/cn/faq/api-code
"""


class QwenEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class QwenEmbedder(BaseEmbedder):
    supported_arguments = ["dimensions"]

    def __init__(self, model_name: str, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self._client = AsyncClient(
            base_url=os.environ.get(
                "BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            ),
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        )
        self._tokenizer = QwenEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"qwen/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> QwenEstimatingTokenizer:
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
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        filtered_hints = {k: v for k, v in hints.items() if k in self.supported_arguments}
        try:
            response = await self._client.embeddings.create(
                model=self.model_name,
                input=texts,
                **filtered_hints,
            )
        except RateLimitError:
            self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise

        vectors = [data_point.embedding for data_point in response.data]
        return EmbeddingResult(vectors=vectors)


class QwenTextEmbedding_V4(QwenEmbedder):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(model_name="text-embedding-v4", logger=logger, tracer=tracer, meter=meter)

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        return 1024


class QwenSchematicGenerator(BaseSchematicGenerator[T]):
    supported_qwen_params = ["temperature", "max_tokens"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self.model_name = model_name
        self._logger = logger
        self._meter = meter

        self._client = AsyncClient(
            base_url=os.environ.get(
                "BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            ),
            api_key=os.environ["DASHSCOPE_API_KEY"],
        )

        self._tokenizer = QwenEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"Qwen/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> QwenEstimatingTokenizer:
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
        with self._logger.scope(f"Qwen LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        qwen_api_arguments = {k: v for k, v in hints.items() if k in self.supported_qwen_params}

        t_start = time.time()
        response = await self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            max_tokens=8 * 1024,
            response_format={"type": "json_object"},
            **qwen_api_arguments,
        )
        t_end = time.time()

        if response.usage:
            self._logger.trace(response.usage.model_dump_json(indent=2))

        raw_content = response.choices[0].message.content or "{}"

        try:
            json_content = json.loads(normalize_json_output(raw_content))
        except json.JSONDecodeError:
            self._logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content})")
            json_content = jsonfinder.only_json(raw_content)[2]
            self._logger.warning("Found JSON content within model response; continuing...")

        try:
            content = self.schema.model_validate(json_content)

            await record_llm_metrics(
                self._meter,
                self.model_name,
                schema_name=self.schema.__name__,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cached_input_tokens=getattr(
                    response,
                    "usage.prompt_cache_hit_tokens",
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
                                response,
                                "usage.prompt_cache_hit_tokens",
                                0,
                            )
                        },
                    ),
                ),
            )
        except ValidationError:
            self._logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class Qwen_MAX(QwenSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(model_name="qwen-max", logger=logger, tracer=tracer, meter=meter)

    @property
    @override
    def max_tokens(self) -> int:
        return 32 * 1024


class Qwen_Plus(QwenSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(model_name="qwen-plus", logger=logger, tracer=tracer, meter=meter)

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class Qwen_2_5_72b(QwenSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="qwen2.5-72b-instruct", logger=logger, tracer=tracer, meter=meter
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class QwenService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("DASHSCOPE_API_KEY"):
            return """\
You're using the Qwen NLP service, but DASHSCOPE_API_KEY is not set.
Please set DASHSCOPE_API_KEY in your environment before running Parlant.
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
        self.model_name = os.environ.get("QWEN_MODEL", "qwen-plus")

        self._logger.info(f"Initialized QwenService with model: {self.model_name}")

    def _get_specialized_generator_class(
        self,
        model_name: str,
        t: type[T],
    ) -> Callable[..., QwenSchematicGenerator[T]] | None:
        """
        Returns the specialized generator class for known models
        """
        model_mapping: dict[str, type[QwenSchematicGenerator[T]]] = {
            "qwen-max": Qwen_MAX[t],  # type: ignore
            "qwen-plus": Qwen_Plus[t],  # type: ignore
            "qwen2.5-72b-instruct": Qwen_2_5_72b[t],  # type: ignore
        }

        if generator_class := model_mapping.get(model_name):
            return generator_class
        else:
            return None

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> QwenSchematicGenerator[T]:
        qwen_generator = self._get_specialized_generator_class(self.model_name, t)
        assert qwen_generator is not None, f"Unsupported Qwen model: {self.model_name}"
        return qwen_generator(self._logger, self._tracer, self._meter)

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        return QwenTextEmbedding_V4(logger=self._logger, tracer=self._tracer, meter=self._meter)

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
