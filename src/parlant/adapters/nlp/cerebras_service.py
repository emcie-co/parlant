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
from cerebras.cloud.sdk import AsyncCerebras
from cerebras.cloud.sdk import (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)
from typing import Any, Mapping
from typing_extensions import override
import jsonfinder  # type: ignore
import os
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output
from parlant.adapters.nlp.hugging_face import JinaAIEmbedder
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.nlp.embedding import Embedder
from parlant.core.nlp.generation import (
    T,
    SchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.loggers import Logger
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.tokenization import EstimatingTokenizer


class LlamaEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self) -> None:
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens) + 36


class CerebrasSchematicGenerator(SchematicGenerator[T]):
    supported_hints = ["temperature"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
    ) -> None:
        self.model_name = model_name

        self._logger = logger
        self._client = AsyncCerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

    @policy(
        [
            retry(
                exceptions=(
                    APIConnectionError,
                    APITimeoutError,
                    RateLimitError,
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
        with self._logger.scope("CerebrasSchematicGenerator"):
            with self._logger.operation(f"LLM Request ({self.schema.__name__})"):
                return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        cerebras_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        t_start = time.time()
        try:
            response = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "schema": self.schema.model_json_schema(),
                        "name": self.schema.__name__,
                        "strict": True,
                    },
                },
                **cerebras_api_arguments,
            )
        except RateLimitError:
            self._logger.error(
                "Cerebras API rate limit exceeded.\n"
                "Your account may have reached the maximum number of requests allowed per minute for the tier you are using.\n"
                "Please contact with Cerebras support for more information."
            )
            raise

        t_end = time.time()

        if response.usage:  # type: ignore
            self._logger.trace(response.usage.model_dump_json(indent=2))  # type: ignore

        raw_content = response.choices[0].message.content or "{}"  # type: ignore

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
                        input_tokens=response.usage.prompt_tokens,  # type: ignore
                        output_tokens=response.usage.completion_tokens,  # type: ignore
                        extra={},
                    ),
                ),
            )
        except ValidationError:
            self._logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class Llama3_3_8B(CerebrasSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="llama3.1-8b",
            logger=logger,
        )
        self._estimating_tokenizer = LlamaEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        return self.model_name

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    @override
    def tokenizer(self) -> LlamaEstimatingTokenizer:
        return self._estimating_tokenizer


class Llama3_3_70B(CerebrasSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="llama3.3-70b",
            logger=logger,
        )

        self._estimating_tokenizer = LlamaEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        return self.model_name

    @property
    @override
    def tokenizer(self) -> LlamaEstimatingTokenizer:
        return self._estimating_tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return 32 * 1024


class CerebrasService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("CEREBRAS_API_KEY"):
            return """\
You're using the OpenAI NLP service, but CEREBRAS_API_KEY is not set.
Please set CEREBRAS_API_KEY in your environment before running Parlant.
"""

        return None

    def __init__(
        self,
        logger: Logger,
    ) -> None:
        self._logger = logger
        self._logger.info("Initialized CerebrasService")

    @override
    async def get_schematic_generator(self, t: type[T]) -> CerebrasSchematicGenerator[T]:
        return Llama3_3_70B[t](self._logger)  # type: ignore

    @override
    async def get_embedder(self) -> Embedder:
        return JinaAIEmbedder()

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
