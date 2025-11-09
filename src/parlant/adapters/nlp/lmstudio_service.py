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
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AsyncClient,
    ConflictError,
    InternalServerError,
    RateLimitError,
    base_url,
)
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
from parlant.core.meter import Meter
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.embedding import BaseEmbedder, Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.moderation import ModerationService, NoModeration

# TODO: Fix the LMStudioModelVerifier class
# class LMStudioModelVerifier:
#     """Utility class for verifying LMStudio model availability."""

#     @staticmethod
#     def verify_models(base_url: str, generation_model: str, embedding_model: str) -> str | None:
#         """
#         Returns an error string if required LMStudio models are missing,
#         or None if all are available.
#         """
#         client = LMStudio.Client(host=base_url.rstrip("/"))
#         try:
#             models = client.list()

#             model_names = []
#             for model in models.get("models", []):
#                 if hasattr(model, "model"):
#                     model_names.append(model.model)
#                 elif isinstance(model, dict) and "model" in model:
#                     model_names.append(model["model"])
#                 elif isinstance(model, dict) and "name" in model:
#                     model_names.append(model["name"])

#             missing_models = []

#             gen_model_found = any(generation_model in model for model in model_names)
#             if not gen_model_found and generation_model not in model_names:
#                 missing_models.append(f"    LMStudio pull {generation_model}")

#             embed_model_found = any(embedding_model in model for model in model_names)
#             if not embed_model_found and embedding_model not in model_names:
#                 missing_models.append(f"    LMStudio pull {embedding_model}")

#             if missing_models:
#                 return f"""\
#                 The following required models are not available in LMStudio:

#                 {chr(10).join(missing_models)}

#                 Please pull the missing models using the commands above.

#                 Available models: {", ".join(model_names) if model_names else "None"}
#                 """
#             return None

#         except LMStudio.ResponseError as e:
#             if e.status_code in [502, 503, 504]:
#                 return f"""\
#                 Cannot connect to LMStudio server at {base_url}.

#                 Please ensure LMStudio is running:
#                     LMStudio serve

#                 Or check if the LMStudio_BASE_URL is correct: {base_url}
#                 """
#             else:
#                 return f"Error checking LMStudio models: {e.error}"

#         except Exception as e:
#             return f"Error connecting to LMStudio: {str(e)}"

class LMStudioEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # Use GPT-4o encoding as a reasonable approximation
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class LMStudioSchematicGenerator(BaseSchematicGenerator[T]):
    supported_lmstudio_params = ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    supported_hints = supported_lmstudio_params

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        meter: Meter,
        base_url: str = "http://127.0.0.1:1234/v1",
        api_key: str = "EMPTY",
    ) -> None:
        super().__init__(logger=logger, meter=meter, model_name=model_name)

        self._client = AsyncClient(
            api_key=api_key,
            base_url=base_url,
        )

        self._tokenizer = LMStudioEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"lmstudio/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> LMStudioEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        # Default reasonable context window for most local models
        # Users can override this via environment variable if needed
        return int(os.environ.get("LMSTUDIO_MAX_TOKENS", "32768"))

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
        with self.logger.scope(f"LM Studio LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    def _list_arguments(self, hints: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            k: v
            for k, v in hints.items()
            if k in self.supported_lmstudio_params
        }

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        lmstudio_api_arguments = self._list_arguments(hints)

        try:
            t_start = time.time()
            # LM Studio doesn't fully support OpenAI's response_format parameter
            # So we rely on the prompt to request JSON output
            response = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                **lmstudio_api_arguments,
            )
            t_end = time.time()
        except APIConnectionError as e:
            self.logger.error(
                f"Cannot connect to LM Studio server. "
                f"Please ensure LM Studio is running and the model is loaded. Error: {e}"
            )
            raise
        except Exception as e:
            self.logger.error(f"LM Studio API error: {e}")
            raise

        if response.usage:
            self.logger.trace(response.usage.model_dump_json(indent=2))

        raw_content = response.choices[0].message.content or "{}"

        try:
            json_content = json.loads(normalize_json_output(raw_content))
        except json.JSONDecodeError:
            self.logger.debug(f"Invalid JSON returned by {self.model_name}:\n{raw_content})")
            json_content = jsonfinder.only_json(raw_content)[2]
            self.logger.debug("Found JSON content within model response; continuing...")

        try:
            content = self.schema.model_validate(json_content)

            # LM Studio may or may not provide usage info
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            if response.usage:
                await record_llm_metrics(
                    self.meter,
                    self.model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
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
                    ),
                ),
            )

        except ValidationError as e:
            self.logger.error(
                f"Error: {e.json(indent=2)}\nJSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class LMStudioEmbedder(BaseEmbedder):
    supported_arguments = ["dimensions"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        meter: Meter,
    ) -> None:
        super().__init__(logger, meter, model_name)
        self.model_name = model_name
        
        # Read configuration from environment variables
        self.base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        self.api_key = os.environ.get("LMSTUDIO_API_KEY", "EMPTY")

        self._client = AsyncClient(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self._tokenizer = LMStudioEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"lmstudio/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> LMStudioEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return int(os.environ.get("LMSTUDIO_EMBEDDING_MAX_TOKENS", "8192"))

    @property
    def dimensions(self) -> int:
        return int(os.environ.get("LMSTUDIO_EMBEDDING_DIMENSIONS", "768"))

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
        except APIConnectionError as e:
            self.logger.error(
                f"Cannot connect to LM Studio server for embeddings. "
                f"Please ensure LM Studio is running with an embedding model loaded. Error: {e}"
            )
            raise
        except Exception as e:
            self.logger.error(f"LM Studio embedding error: {e}")
            raise

        vectors = [data_point.embedding for data_point in response.data]
        return EmbeddingResult(vectors=vectors)


class LMStudioDefaultEmbedder(LMStudioEmbedder):
    """Default LM Studio embedder that reads configuration from environment variables."""
    
    def __init__(self, logger: Logger, meter: Meter) -> None:
        # Read the embedding model name from environment variables
        model_name = os.environ.get("LMSTUDIO_EMBEDDING_MODEL", "nomic-embed-text")
        super().__init__(model_name=model_name, logger=logger, meter=meter)


class LMStudioService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        required_vars = {
            "LMSTUDIO_BASE_URL": "http://127.0.0.1:1234/v1",
            "LMSTUDIO_MODEL": "unsloth/qwen3-4b-instruct-2507",
            "LMSTUDIO_EMBEDDING_MODEL": "text-embedding-granite-embedding-107m-multilingual",
            "LMSTUDIO_API_TIMEOUT": "300",
        }

        missing_vars = []
        for var_name, default_value in required_vars.items():
            if not os.environ.get(var_name):
                missing_vars.append(f'export {var_name}="{default_value}"')

        if missing_vars:
            return f"""\
You're using the LM Studio NLP service, but the following environment variables are not set:

{chr(10).join(missing_vars)}

Please set these environment variables before running Parlant.

Note: Make sure LM Studio is running and a model is loaded at {required_vars['LMSTUDIO_BASE_URL']}
"""

        return None

    # TODO: Adjust verify model function base on LMStudioModelVerifier class
    # @staticmethod
    # def verify_models() -> str | None:
    #     """
    #     Verify that the required models are available in LM Studio.
    #     Returns an error message if models are missing, None if all are available.
    #     """

    #     base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    #     embedding_model = os.environ.get("LMSTUDIO_EMBEDDING_MODEL", "nomic-embed-text")
    #     generation_model = os.environ.get("LMSTUDIO_MODEL", "gemma3:4b")

    #     if error := LMStudioModelVerifier.verify_models(base_url, generation_model, embedding_model):
    #         return f"Model Verification Issue:\n{error}"

    #     return None

    def __init__(
        self,
        logger: Logger,
        meter: Meter,
    ) -> None:
        self.base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        self.api_key = os.environ.get("LMSTUDIO_API_KEY", "EMPTY")
        self.model_name = os.environ.get("LMSTUDIO_MODEL", "qwen3-4b")
        self.embedding_model = os.environ.get("LMSTUDIO_EMBEDDING_MODEL", "nomic-embed-text")

        self._logger = logger
        self._meter = meter

        self._logger.info(
            f"Initialized LMStudioService with model '{self.model_name}' at {self.base_url}"
        )

    @override
    async def get_schematic_generator(self, t: type[T]) -> LMStudioSchematicGenerator[T]:
        return LMStudioSchematicGenerator[t](  # type: ignore
            model_name=self.model_name,
            logger=self._logger,
            meter=self._meter,
            base_url=self.base_url,
            api_key=self.api_key,
        )

    @override
    async def get_embedder(self) -> Embedder:
        return LMStudioDefaultEmbedder(
            logger=self._logger,
            meter=self._meter,
        )

    @override
    async def get_moderation_service(self) -> ModerationService:
        """Get a moderation service (using no moderation for local models)."""
        return NoModeration()

