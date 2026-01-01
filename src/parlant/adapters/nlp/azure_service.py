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
    AsyncAzureOpenAI,
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)  # type: ignore
from azure.identity.aio import DefaultAzureCredential  # type: ignore
from typing import Any, Mapping
from typing_extensions import override
import json
import jsonfinder  # type: ignore
import os
from pydantic import ValidationError
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.tracer import Tracer
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
from parlant.core.nlp.moderation import ModerationService, NoModeration


class AzureEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model(model_name)

    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class AzureSchematicGenerator(BaseSchematicGenerator[T]):
    supported_azure_params = ["temperature", "logit_bias", "max_tokens"]
    supported_hints = supported_azure_params + ["strict"]
    unsupported_params_by_model: dict[str, list[str]] = {
        "gpt-5": ["temperature"],
    }

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        client: AsyncAzureOpenAI,
    ) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self.model_name = model_name
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._client = client
        self._tokenizer = AzureEstimatingTokenizer(model_name=self.model_name)

    @property
    def id(self) -> str:
        return f"azure/{self.model_name}"

    @property
    def tokenizer(self) -> AzureEstimatingTokenizer:
        return self._tokenizer

    def _list_arguments(self, hints: Mapping[str, Any]) -> Mapping[str, Any]:
        exclude_params = [
            k
            for k in self.supported_azure_params
            for prefix, excluded in self.unsupported_params_by_model.items()
            if self.model_name.startswith(prefix) and k in excluded
        ]

        return {
            k: v
            for k, v in hints.items()
            if k in self.supported_azure_params and k not in exclude_params
        }

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
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self._logger.scope(f"Azure LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        azure_api_arguments = self._list_arguments(hints)

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

            await record_llm_metrics(
                self._meter,
                self.model_name,
                schema_name=self.schema.__name__,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cached_input_tokens=response.usage.prompt_tokens_details.cached_tokens or 0
                if response.usage.prompt_tokens_details
                else 0,
            )

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


def create_azure_client() -> AsyncAzureOpenAI:
    """Create an Azure OpenAI client with appropriate authentication."""
    azure_endpoint = os.environ["AZURE_ENDPOINT"]

    # Check if API key is provided (backward compatibility)
    if os.environ.get("AZURE_API_KEY"):
        return AsyncAzureOpenAI(
            api_key=os.environ["AZURE_API_KEY"],
            azure_endpoint=azure_endpoint,
            api_version=os.environ.get("AZURE_API_VERSION", "2024-08-01-preview"),
        )
    else:
        # Use Azure AD authentication
        try:
            credential = DefaultAzureCredential()

            async def token_provider() -> str:
                """Token provider that requests tokens with the correct scope for Azure OpenAI."""
                try:
                    token = await credential.get_token(
                        "https://cognitiveservices.azure.com/.default"
                    )
                    return str(token.token)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to get Azure AD token: {e}\n\n"
                        "Please ensure you are authenticated with Azure AD using one of:\n"
                        "1. Azure CLI: `az login`\n"
                        "2. Service Principal environment variables:\n"
                        "   - AZURE_CLIENT_ID\n"
                        "   - AZURE_CLIENT_SECRET\n"
                        "   - AZURE_TENANT_ID\n"
                        "3. Managed Identity (if running on Azure)\n\n"
                        "For more details, see: https://docs.microsoft.com/en-us/python/api/overview/azure/identity-readme"
                    ) from e

            return AsyncAzureOpenAI(
                azure_ad_token_provider=token_provider,
                azure_endpoint=azure_endpoint,
                api_version=os.environ.get("AZURE_API_VERSION", "2024-08-01-preview"),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Azure AD authentication: {e}\n\n"
                "Please ensure you are authenticated with Azure AD using one of:\n"
                "1. Azure CLI: `az login`\n"
                "2. Service Principal environment variables:\n"
                "   - AZURE_CLIENT_ID\n"
                "   - AZURE_CLIENT_SECRET\n"
                "   - AZURE_TENANT_ID\n"
                "3. Managed Identity (if running on Azure)\n\n"
                "For more details, see: https://docs.microsoft.com/en-us/python/api/overview/azure/identity-readme"
            ) from e


class CustomAzureSchematicGenerator(AzureSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        _client = create_azure_client()

        super().__init__(
            model_name=os.environ["AZURE_GENERATIVE_MODEL_NAME"],
            logger=logger,
            tracer=tracer,
            meter=meter,
            client=_client,
        )

    @property
    def max_tokens(self) -> int:
        return int(os.environ.get("AZURE_GENERATIVE_MODEL_WINDOW", 4096))


class GPT_4o(AzureSchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        _client = create_azure_client()
        super().__init__(
            model_name="gpt-4o", logger=logger, tracer=tracer, meter=meter, client=_client
        )

    @property
    def max_tokens(self) -> int:
        return 128 * 1024


class GPT_4o_Mini(AzureSchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        _client = create_azure_client()
        super().__init__(
            model_name="gpt-4o-mini", logger=logger, tracer=tracer, meter=meter, client=_client
        )
        self._token_estimator = AzureEstimatingTokenizer(model_name=self.model_name)

    @property
    def max_tokens(self) -> int:
        return 128 * 1024


class AzureEmbedder(BaseEmbedder):
    supported_arguments = ["dimensions"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        client: AsyncAzureOpenAI,
    ) -> None:
        self.model_name = model_name

        self._logger = logger
        self._tracer = tracer
        self._meter = meter

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
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        _client = create_azure_client()
        super().__init__(
            model_name=os.environ["AZURE_EMBEDDING_MODEL_NAME"],
            logger=logger,
            tracer=tracer,
            meter=meter,
            client=_client,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return int(os.environ["AZURE_EMBEDDING_MODEL_WINDOW"])

    @property
    def dimensions(self) -> int:
        return int(os.environ["AZURE_EMBEDDING_MODEL_DIMS"])


class AzureTextEmbedding3Large(AzureEmbedder):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        _client = create_azure_client()
        super().__init__(
            model_name="text-embedding-3-large",
            logger=logger,
            tracer=tracer,
            meter=meter,
            client=_client,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        return 3072


class AzureTextEmbedding3Small(AzureEmbedder):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        _client = create_azure_client()
        super().__init__(
            model_name="text-embedding-3-small",
            logger=logger,
            tracer=tracer,
            meter=meter,
            client=_client,
        )

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

        if not os.environ.get("AZURE_ENDPOINT"):
            return """\
You're using the Azure NLP service, but AZURE_ENDPOINT is not set.
Please set AZURE_ENDPOINT in your environment before running Parlant.

Required environment variables:
- AZURE_ENDPOINT

Authentication options (choose one):
1. Azure AD (recommended):
   - Ensure you're authenticated via Azure CLI: `az login`
   - Or set up managed identity/service principal authentication

2. API Key (legacy):
   - AZURE_API_KEY

You can also set any specific models you'd like to use, using a few more variables:

- AZURE_GENERATIVE_MODEL_NAME (e.g., gpt-4o)
- AZURE_GENERATIVE_MODEL_WINDOW (size of the generative model's context window)

- AZURE_EMBEDDING_MODEL_NAME (e.g., text-embedding-3-large)
- AZURE_EMBEDDING_MODEL_DIMS (dimensions of the embedding model)
- AZURE_EMBEDDING_MODEL_WINDOW (size of of the embedding model's context window)

For Azure AD authentication, ensure your identity has the "Cognitive Services OpenAI User" role
on the Azure OpenAI resource.
"""

        # Check authentication method
        has_api_key = bool(os.environ.get("AZURE_API_KEY"))

        if has_api_key:
            # API key authentication is configured
            return None

        # Check Azure AD authentication
        try:
            from azure.identity import DefaultAzureCredential  # type: ignore

            credential = DefaultAzureCredential()

            # Try to get a token to verify authentication works
            import asyncio

            async def test_auth() -> bool:
                try:
                    token = credential.get_token("https://cognitiveservices.azure.com/.default")
                    return token is not None
                except Exception:
                    return False

            # Run the async test
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, we can't test synchronously
                    # Just check if we can create the credential
                    return None
                else:
                    auth_works = loop.run_until_complete(test_auth())
                    if auth_works:
                        return None
            except RuntimeError:
                # No event loop, create a new one
                auth_works = asyncio.run(test_auth())
                if auth_works:
                    return None

        except Exception:
            pass

        # If we get here, neither authentication method is working
        return """\
Azure authentication is not properly configured.

Please choose one of the following authentication methods:

1. API Key Authentication (Legacy):
   Set the AZURE_API_KEY environment variable with your Azure OpenAI API key.

2. Azure AD Authentication (Recommended):
   Ensure you're authenticated using one of these methods:

   a) Azure CLI (for development):
      Run: az login

   b) Service Principal (for production):
      Set these environment variables:
      - AZURE_CLIENT_ID
      - AZURE_CLIENT_SECRET
      - AZURE_TENANT_ID

   c) Managed Identity (if running on Azure):
      Ensure your Azure resource has managed identity enabled

   d) Environment Credential:
      Set these environment variables:
      - AZURE_CLIENT_ID
      - AZURE_CLIENT_SECRET
      - AZURE_TENANT_ID

   e) Workload Identity (for Kubernetes):
      Set these environment variables:
      - AZURE_CLIENT_ID
      - AZURE_TENANT_ID
      - AZURE_FEDERATED_TOKEN_FILE

Important: For Azure AD authentication, ensure your identity has the 
"Cognitive Services OpenAI User" role on the Azure OpenAI resource.

For more details on Azure AD authentication options, see:
https://docs.microsoft.com/en-us/python/api/overview/azure/identity-readme
"""

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter

    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> AzureSchematicGenerator[T]:
        if os.environ.get("AZURE_GENERATIVE_MODEL_NAME"):
            return CustomAzureSchematicGenerator[t](  # type: ignore
                logger=self._logger, tracer=self._tracer, meter=self._meter
            )
        return GPT_4o[t](self._logger, self._tracer, self._meter)  # type: ignore

    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        if os.environ.get("AZURE_EMBEDDING_MODEL_NAME"):
            return CustomAzureEmbedder(self._logger, self._tracer, self._meter)
        return AzureTextEmbedding3Large(self._logger, self._tracer, self._meter)

    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
