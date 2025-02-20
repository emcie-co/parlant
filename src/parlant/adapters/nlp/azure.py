# Copyright 2024 Emcie Co Ltd.
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
from openai import AsyncAzureOpenAI
from typing import Any, Mapping
from typing_extensions import override
import json
import jsonfinder  # type: ignore
import os
from pydantic import ValidationError
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output
from parlant.core.logging import Logger
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    SchematicGenerator,
    GenerationInfo,
    SchematicGenerationResult,
    UsageInfo,
)
from parlant.core.nlp.moderation import ModerationService, NoModeration


class AzureEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model(model_name)

    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class AzureSchematicGenerator(SchematicGenerator[T]):
    supported_azure_params = ["temperature", "logit_bias", "max_tokens"]
    supported_hints = supported_azure_params + ["strict"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        client: AsyncAzureOpenAI,
    ) -> None:
        self.model_name = model_name
        self._logger = logger
        self._client = client
        self._tokenizer = AzureEstimatingTokenizer(model_name=self.model_name)

    @property
    def id(self) -> str:
        return f"azure/{self.model_name}"

    @property
    def tokenizer(self) -> AzureEstimatingTokenizer:
        return self._tokenizer

    async def generate(
        self,
        prompt: str,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self._logger.operation(f"Azure LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        azure_api_arguments = {k: v for k, v in hints.items() if k in self.supported_azure_params}

        if hints.get("strict", False):
            t_start = time.time()
            response = await self._client.beta.chat.completions.parse(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format=self.schema,
                **azure_api_arguments,
            )
            t_end = time.time()

            if response.usage:
                self._logger.debug(response.usage.model_dump_json(indent=2))

            parsed_object = response.choices[0].message.parsed
            assert parsed_object

            assert response.usage

            return SchematicGenerationResult[T](
                content=parsed_object,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        extra=(
                            {
                                "cached_input_tokens": response.usage.prompt_tokens_details.cached_tokens
                                or 0
                            }
                            if response.usage.prompt_tokens_details
                            else {}
                        ),
                    ),
                ),
            )

        else:
            t_start = time.time()
            response = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format={"type": "json_object"},
                **azure_api_arguments,
            )
            t_end = time.time()

            if response.usage:
                self._logger.debug(response.usage.model_dump_json(indent=2))

            raw_content = response.choices[0].message.content or "{}"

            try:
                json_content = json.loads(normalize_json_output(raw_content))
            except json.JSONDecodeError:
                self._logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content})")
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
                            extra=(
                                {
                                    "cached_input_tokens": response.usage.prompt_tokens_details.cached_tokens
                                    or 0
                                }
                                if response.usage.prompt_tokens_details
                                else {}
                            ),
                        ),
                    ),
                )
            except ValidationError:
                self._logger.error(
                    f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
                )
                raise


class GPT_4o(AzureSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        _client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_API_KEY"],
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            api_version="2024-08-01-preview",
        )
        super().__init__(model_name="gpt-4o", logger=logger, client=_client)

    @property
    def max_tokens(self) -> int:
        return 128 * 1024


class GPT_4o_Mini(AzureSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        _client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_API_KEY"],
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            api_version="2024-08-01-preview",
        )
        super().__init__(model_name="gpt-4o-mini", logger=logger, client=_client)
        self._token_estimator = AzureEstimatingTokenizer(model_name=self.model_name)

    @property
    def max_tokens(self) -> int:
        return 128 * 1024


class AzureEmbedder(Embedder):
    supported_arguments = ["dimensions"]

    def __init__(self, model_name: str, client: AsyncAzureOpenAI) -> None:
        self.model_name = model_name
        self._client = client
        self._tokenizer = AzureEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"azure/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> AzureEstimatingTokenizer:
        return self._tokenizer

    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        filtered_hints = {k: v for k, v in hints.items() if k in self.supported_arguments}

        response = await self._client.embeddings.create(
            model=self.model_name,
            input=texts,
            **filtered_hints,
        )

        vectors = [data_point.embedding for data_point in response.data]
        return EmbeddingResult(vectors=vectors)


class AzureTextEmbedding3Large(AzureEmbedder):
    def __init__(self) -> None:
        _client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_API_KEY"],
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            api_version="2023-05-15",
        )
        super().__init__(model_name="text-embedding-3-large", client=_client)

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        return 3072


class AzureTextEmbedding3Small(AzureEmbedder):
    def __init__(self) -> None:
        _client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_API_KEY"],
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            api_version="2023-05-15",
        )
        super().__init__(model_name="text-embedding-3-small", client=_client)

    @property
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        return 3072


class AzureService(NLPService):
    def __init__(
        self,
        logger: Logger,
    ) -> None:
        self._logger = logger

    async def get_schematic_generator(self, t: type[T]) -> AzureSchematicGenerator[T]:
        return GPT_4o[t](self._logger)  # type: ignore

    async def get_embedder(self) -> Embedder:
        return AzureTextEmbedding3Large()

    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
