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
# Maintainer: Rongkun Yan <2493404415@qq.com>

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
from typing import Any, Mapping
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
from parlant.core.nlp.service import EmbedderHints, NLPService, SchematicGeneratorHints
from parlant.core.nlp.embedding import Embedder
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


class ModelScopeEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class ModelScopeSchematicGenerator(BaseSchematicGenerator[T]):
    supported_modelscope_params = ["temperature", "logit_bias", "max_tokens"]
    supported_hints = supported_modelscope_params + ["strict"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self._client = AsyncClient(
            base_url="https://api-inference.modelscope.cn/v1",
            api_key=os.environ["MODELSCOPE_API_KEY"],
        )

        self._tokenizer = ModelScopeEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"modelscope/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> ModelScopeEstimatingTokenizer:
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
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self.logger.scope(f"ModelScope LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        modelscope_api_arguments = {
            k: v for k, v in hints.items() if k in self.supported_modelscope_params
        }

        t_start = time.time()
        response = await self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            stream=True,
            extra_body={"enable_thinking": False},
            max_tokens=8192,
            response_format={"type": "json_object"},
            **modelscope_api_arguments,
        )
        t_end = time.time()

        raw_content = ""
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                raw_content += chunk.choices[0].delta.content

        try:
            json_content = json.loads(normalize_json_output(raw_content))
        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content})")
            json_content = jsonfinder.only_json(raw_content)[2]
            self.logger.warning("Found JSON content within model response; continuing...")

        try:
            content = self.schema.model_validate(json_content)

            input_tokens = await self.tokenizer.estimate_token_count(prompt)
            output_tokens = await self.tokenizer.estimate_token_count(raw_content)

            await record_llm_metrics(
                self.meter,
                self.model_name,
                schema_name=self.schema.__name__,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=0,
            )

            return SchematicGenerationResult(
                content=content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        extra={},
                    ),
                ),
            )
        except ValidationError as ve:
            self.logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            self.logger.error(f"Validation error details: {str(ve)}")
            raise


class ModelScopeChat(ModelScopeSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        model_name = os.environ["MODELSCOPE_MODEL_NAME"]
        super().__init__(model_name=model_name, logger=logger, tracer=tracer, meter=meter)

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class ModelScopeService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("MODELSCOPE_MODEL_NAME"):
            return """\
You're using the ModelScope NLP service, but MODELSCOPE_MODEL_NAME is not set.
Please set MODELSCOPE_MODEL_NAME in your environment before running Parlant.
"""
        if not os.environ.get("MODELSCOPE_API_KEY"):
            return """\
You're using the ModelScope NLP service, but MODELSCOPE_API_KEY is not set.
Please set MODELSCOPE_API_KEY in your environment before running Parlant.
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
        self._logger.info("Initialized ModelScopeService")

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> ModelScopeSchematicGenerator[T]:
        return ModelScopeChat[t](self._logger, self._tracer, self._meter)  # type: ignore

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        return JinaAIEmbedder(self._logger, self._tracer, self._meter)

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
