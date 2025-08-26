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

from __future__ import annotations
import time
from typing import Any, Mapping
from typing_extensions import override
import json
import jsonfinder  # type: ignore
import os

from pydantic import ValidationError
import tiktoken

import litellm

from parlant.adapters.nlp.common import normalize_json_output
from parlant.adapters.nlp.hugging_face import JinaAIEmbedder
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.embedding import Embedder
from parlant.core.nlp.generation import (
    T,
    SchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.moderation import (
    ModerationService,
    NoModeration,
)

RATE_LIMIT_ERROR_MESSAGE = (
    "LiteLLM to provider API rate limit exceeded. Possible reasons:\n"
    "1. Your account may have insufficient API credits.\n"
    "2. You may be using a free-tier account with limited request capacity.\n"
    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your LLM Provider account balance and billing status.\n"
    "- Review your API usage limits in Provider's dashboard.\n"
    "- For more details on rate limits and usage tiers, visit:\n"
    "  Your Provider's API documentation."
)


class LiteLLMEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class LiteLLMSchematicGenerator(SchematicGenerator[T]):
    supported_litellm_params = [
        "temperature",
        "max_tokens",
        "logit_bias",
        "adapter_id",
        "adapter_soruce",
    ]
    supported_hints = supported_litellm_params + ["strict"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
    ) -> None:
        self.model_name = model_name
        self._logger = logger

        self._client = litellm

        self._tokenizer = LiteLLMEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"litellm/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> LiteLLMEstimatingTokenizer:
        return self._tokenizer

    @override
    async def generate(
        self,
        prompt: PromptBuilder | str,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self._logger.operation(f"LiteLLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        litellm_api_arguments = {
            k: v for k, v in hints.items() if k in self.supported_litellm_params
        }

        t_start = time.time()

        response = self._client.completion(
            api_key=os.environ.get("LITELLM_PROVIDER_API_KEY"),
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            max_tokens=5000,
            response_format={"type": "json_object"},
            **litellm_api_arguments,
        )

        t_end = time.time()

        if response.usage:
            self._logger.trace(response.usage.model_dump_json(indent=2))

        raw_content = response.choices[0].message.content or "{}"

        try:
            json_content = json.loads(normalize_json_output(raw_content))
        except json.JSONDecodeError:
            self._logger.warning(
                f"Invalid JSON returned by litellm/{self.model_name}:\n{raw_content})"
            )
            json_content = jsonfinder.only_json(raw_content)[2]
            self._logger.warning("Found JSON content within model response; continuing...")

        try:
            content = self.schema.model_validate(json_content)
            assert response.usage

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
                f"JSON content returned by litellm/{self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class LiteLLM_Default(LiteLLMSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name=os.environ.get("LITELLM_PROVIDER_MODEL_NAME"),  # type: ignore
            logger=logger,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 5000

    # 8192 16381


class LiteLLMService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("LITELLM_PROVIDER_MODEL_NAME"):
            return """\
You're using the LITELLM NLP service, but LITELLM_PROVIDER_MODEL_NAME is not set.
Please set LITELLM_PROVIDER_MODEL_NAME in your environment before running Parlant.
"""
        if not os.environ.get("LITELLM_PROVIDER_API_KEY"):
            return """\
You're using the LITELLM NLP service, but LITELLM_PROVIDER_API_KEY is not set.
Please set LITELLM_PROVIDER_API_KEY in your environment before running Parlant.
"""

        return None

    def __init__(
        self,
        logger: Logger,
    ) -> None:
        self._model_name = os.environ.get("LITELLM_PROVIDER_MODEL_NAME")
        self._logger = logger
        self._logger.info("Initialized LiteLLMService")

    @override
    async def get_schematic_generator(self, t: type[T]) -> LiteLLMSchematicGenerator[T]:
        return LiteLLM_Default[t](self._logger)  # type: ignore

    @override
    async def get_embedder(self) -> Embedder:
        return JinaAIEmbedder()

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
