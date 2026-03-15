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
from typing import Any, Callable, Mapping
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
    SchematicGenerationResult,
    StreamingTextGenerator,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.moderation import (
    ModerationService,
    NoModeration,
)

MINIMAX_BASE_URL = "https://api.minimax.io/v1"


class MiniMaxEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class MiniMaxSchematicGenerator(BaseSchematicGenerator[T]):
    supported_minimax_params = ["temperature", "max_tokens"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self._client = AsyncClient(
            base_url=os.environ.get("MINIMAX_BASE_URL", MINIMAX_BASE_URL),
            api_key=os.environ["MINIMAX_API_KEY"],
        )

        self._tokenizer = MiniMaxEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"minimax/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> MiniMaxEstimatingTokenizer:
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
        with self.logger.scope(f"MiniMax LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        minimax_api_arguments: dict[str, Any] = {
            k: v for k, v in hints.items() if k in self.supported_minimax_params
        }

        # MiniMax requires temperature > 0; clamp zero to a small positive value
        if "temperature" in minimax_api_arguments and minimax_api_arguments["temperature"] == 0:
            minimax_api_arguments["temperature"] = 0.01

        t_start = time.time()
        response = await self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            max_tokens=8192,
            response_format={"type": "json_object"},
            **minimax_api_arguments,
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
            self.logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class MiniMax_M2_5(MiniMaxSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(model_name="MiniMax-M2.5", logger=logger, tracer=tracer, meter=meter)

    @property
    @override
    def max_tokens(self) -> int:
        return 204_000


class MiniMax_M2_5_Highspeed(MiniMaxSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="MiniMax-M2.5-highspeed",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 204_000


class MiniMaxService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("MINIMAX_API_KEY"):
            return """\
You're using the MiniMax NLP service, but MINIMAX_API_KEY is not set.
Please set MINIMAX_API_KEY in your environment before running Parlant.
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
        self._model_name = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.5")
        self._logger.info(f"Initialized MiniMaxService with model: {self._model_name}")

    @property
    @override
    def supports_streaming(self) -> bool:
        return False

    @override
    async def get_streaming_text_generator(
        self, hints: StreamingTextGeneratorHints = {}
    ) -> StreamingTextGenerator:
        raise NotImplementedError("Streaming is not supported. Check supports_streaming first.")

    def _get_specialized_generator_class(
        self,
        model_name: str,
        t: type[T],
    ) -> Callable[..., MiniMaxSchematicGenerator[T]] | None:
        """Returns the specialized generator class for known models."""
        model_mapping: dict[str, type[MiniMaxSchematicGenerator[T]]] = {
            "MiniMax-M2.5": MiniMax_M2_5[t],  # type: ignore
            "MiniMax-M2.5-highspeed": MiniMax_M2_5_Highspeed[t],  # type: ignore
        }

        if generator_class := model_mapping.get(model_name):
            return generator_class
        else:
            return None

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> MiniMaxSchematicGenerator[T]:
        minimax_generator = self._get_specialized_generator_class(self._model_name, t)
        assert minimax_generator is not None, f"Unsupported MiniMax model: {self._model_name}"
        return minimax_generator(self._logger, self._tracer, self._meter)

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        return JinaAIEmbedder(self._logger, self._tracer, self._meter)

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
