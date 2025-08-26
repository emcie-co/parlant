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
from openai import (
    AsyncAzureOpenAI,
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)  # type: ignore
from typing import Any, Mapping
from typing_extensions import override
import json
import jsonfinder  # type: ignore
import os
from pydantic import ValidationError
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    SchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
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

    @policy(
        [
            retry(
                exceptions=(
                    APIConnectionError,
                    APITimeoutError,
                    RateLimitError,
                    APIResponseValidationError,
                )
            ),
            retry(InternalServerError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self._logger.operation(f"Azure LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        azure_api_arguments = {k: v for k, v in hints.items() if k in self.supported_azure_params}

        if hints.get("strict", False):
            t_start = time.time()
            try:
                response = await self._client.beta.chat.completions.parse(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    response_format=self.schema,
                    **azure_api_arguments,
                )
            except RateLimitError:
                self._logger.error(
                    "Azure API rate limit exceeded. Possible reasons:\n"
                    "1. Your account may have insufficient API credits.\n"
                    "2. You may be using a free-tier account with limited request capacity.\n"
                    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
                    "Recommended actions:\n"
                    "- Check your Azure account balance and billing status.\n"
                    "- Review your API usage limits in Azure's dashboard.\n"
                    "- For more details on rate limits and usage tiers, visit:\n"
                    "  https://learn.microsoft.com/en-us/azure/ai-services/openai/quotas-limits\n",
                )
                raise

            t_end = time.time()

            if response.usage:
                self._logger.trace(response.usage.model_dump_json(indent=2))

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

            try:
                response = await self._client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    response_format={"type": "json_object"},
                    **azure_api_arguments,
                )
            except RateLimitError:
                self._logger.error(
                    "Azure API rate limit exceeded. Possible reasons:\n"
                    "1. Your account may have insufficient API credits.\n"
                    "2. You may be using a free-tier account with limited request capacity.\n"
                    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
                    "Recommended actions:\n"
                    "- Check your Azure account balance and billing status.\n"
                    "- Review your API usage limits in Azure's dashboard.\n"
                    "- For more details on rate limits and usage tiers, visit:\n"
                    "  https://learn.microsoft.com/en-us/azure/ai-services/openai/quotas-limits\n",
                )
                raise

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


class CustomAzureSchematicGenerator(AzureSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        _client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_API_KEY"],
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            api_version=os.environ.get("AZURE_API_VERSION", "2024-08-01-preview"),
        )

        super().__init__(
            model_name=os.environ["AZURE_GENERATIVE_MODEL_NAME"],
            logger=logger,
            client=_client,
        )

    @property
    def max_tokens(self) -> int:
        return int(os.environ.get("AZURE_GENERATIVE_MODEL_WINDOW", 4096))


class GPT_4o(AzureSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        _client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_API_KEY"],
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            api_version=os.environ.get("AZURE_API_VERSION", "2024-08-01-preview"),
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
            api_version=os.environ.get("AZURE_API_VERSION", "2024-08-01-preview"),
        )
        super().__init__(model_name="gpt-4o-mini", logger=logger, client=_client)
        self._token_estimator = AzureEstimatingTokenizer(model_name=self.model_name)

    @property
    def max_tokens(self) -> int:
        return 128 * 1024


class AzureEmbedder(Embedder):
    supported_arguments = ["dimensions"]

    def __init__(self, model_name: str, logger: Logger, client: AsyncAzureOpenAI) -> None:
        self.model_name = model_name

        self._logger = logger
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

        try:
            response = await self._client.embeddings.create(
                model=self.model_name,
                input=texts,
                **filtered_hints,
            )
        except RateLimitError:
            self._logger.error(
                "Azure API rate limit exceeded. Possible reasons:\n"
                "1. Your account may have insufficient API credits.\n"
                "2. You may be using a free-tier account with limited request capacity.\n"
                "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
                "Recommended actions:\n"
                "- Check your Azure account balance and billing status.\n"
                "- Review your API usage limits in Azure's dashboard.\n"
                "- For more details on rate limits and usage tiers, visit:\n"
                "  https://learn.microsoft.com/en-us/azure/ai-services/openai/quotas-limits\n",
            )
            raise

        vectors = [data_point.embedding for data_point in response.data]
        return EmbeddingResult(vectors=vectors)


class CustomAzureEmbedder(AzureEmbedder):
    def __init__(self, logger: Logger) -> None:
        _client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_API_KEY"],
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            api_version=os.environ.get("AZURE_API_VERSION", "2023-05-15"),
        )
        super().__init__(
            model_name=os.environ["AZURE_EMBEDDING_MODEL_NAME"], logger=logger, client=_client
        )

    @property
    @override
    def max_tokens(self) -> int:
        return int(os.environ["AZURE_EMBEDDING_MODEL_WINDOW"])

    @property
    def dimensions(self) -> int:
        return int(os.environ["AZURE_EMBEDDING_MODEL_DIMS"])


class AzureTextEmbedding3Large(AzureEmbedder):
    def __init__(self, logger: Logger) -> None:
        _client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_API_KEY"],
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            api_version=os.environ.get("AZURE_API_VERSION", "2023-05-15"),
        )
        super().__init__(model_name="text-embedding-3-large", logger=logger, client=_client)

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        return 3072


class AzureTextEmbedding3Small(AzureEmbedder):
    def __init__(self, logger: Logger) -> None:
        _client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_API_KEY"],
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            api_version=os.environ.get("AZURE_API_VERSION", "2023-05-15"),
        )
        super().__init__(model_name="text-embedding-3-small", logger=logger, client=_client)

    @property
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        return 3072


class AzureService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("AZURE_API_KEY"):
            return """\
You're using the Azure NLP service, but AZURE_API_KEY is not set.
Please set AZURE_API_KEY in your environment before running Parlant.

- AZURE_API_KEY
- AZURE_ENDPOINT

You can also set any specific models you'd like to use, using a few more variables:

- AZURE_GENERATIVE_MODEL_NAME (e.g., gpt-4o)
- AZURE_GENERATIVE_MODEL_WINDOW (size of the generative model's context window)

- AZURE_EMBEDDING_MODEL_NAME (e.g., text-embedding-3-large)
- AZURE_EMBEDDING_MODEL_DIMS (dimensions of the embedding model)
- AZURE_EMBEDDING_MODEL_WINDOW (size of of the embedding model's context window)
"""
        return None

    def __init__(
        self,
        logger: Logger,
    ) -> None:
        self._logger = logger

    async def get_schematic_generator(self, t: type[T]) -> AzureSchematicGenerator[T]:
        if os.environ.get("AZURE_GENERATIVE_MODEL_NAME"):
            return CustomAzureSchematicGenerator[t](logger=self._logger)  # type: ignore
        return GPT_4o[t](self._logger)  # type: ignore

    async def get_embedder(self) -> Embedder:
        if os.environ.get("AZURE_EMBEDDING_MODEL_NAME"):
            return CustomAzureEmbedder(self._logger)
        return AzureTextEmbedding3Large(self._logger)

    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
