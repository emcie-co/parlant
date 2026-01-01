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

import time
from pydantic import ValidationError
from together import AsyncTogether  # type: ignore
from together.error import (  # type: ignore
    RateLimitError,
    Timeout,
    APIConnectionError,
    APIError,
    ServiceUnavailableError,
)
from typing import Any, Callable, Mapping
from typing_extensions import override
import jsonfinder  # type: ignore
import os
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.adapters.nlp.hugging_face import HuggingFaceEstimatingTokenizer
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.tracer import Tracer
from parlant.core.meter import Meter
from parlant.core.nlp.embedding import BaseEmbedder, Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.loggers import Logger
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.service import EmbedderHints, NLPService, SchematicGeneratorHints
from parlant.core.nlp.tokenization import EstimatingTokenizer


RATE_LIMIT_ERROR_MESSAGE = (
    "Together API rate limit exceeded. Possible reasons:\n"
    "1. Your account may have insufficient API credits.\n"
    "2. You may be using a free-tier account with limited request capacity.\n"
    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your Together account balance and billing status.\n"
    "- Review your API usage limits in Together's dashboard.\n"
    "- For more details on rate limits and usage tiers, visit:\n"
    "  https://docs.together.ai/docs/rate-limits"
)


class LlamaEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self) -> None:
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens) + 36


class TogetherAISchematicGenerator(BaseSchematicGenerator[T]):
    supported_hints = ["temperature", "max_tokens", "top_p", "top_k"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self._client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
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
        # Default max tokens, can be overridden by specific model classes
        return 128 * 1024

    @policy(
        [
            retry(
                exceptions=(
                    RateLimitError,
                    Timeout,
                    APIConnectionError,
                    APIError,
                )
            ),
            retry(ServiceUnavailableError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self.logger.scope(f"Together LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        together_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        t_start = time.time()
        try:
            response = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format={"type": "json_object"},
                **together_api_arguments,
            )
        except RateLimitError:
            self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise

        t_end = time.time()

        raw_content = response.choices[0].message.content or "{}"

        try:
            json_content = normalize_json_output(raw_content)
            json_object = jsonfinder.only_json(json_content)[2]
        except Exception:
            self.logger.error(
                f"Failed to extract JSON returned by {self.model_name}:\n{raw_content}"
            )
            raise

        try:
            model_content = self.schema.model_validate(json_object)

            await record_llm_metrics(
                self.meter,
                self.model_name,
                schema_name=self.schema.__name__,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

            return SchematicGenerationResult(
                content=model_content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        extra={},
                    ),
                ),
            )
        except ValidationError:
            self.logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class Llama3_1_8B(TogetherAISchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )


class Llama3_1_70B(TogetherAISchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )


class Llama3_1_405B(TogetherAISchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )


class Llama3_3_70B(TogetherAISchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )


class TogetherAIEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self._client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

    @policy(
        [
            retry(
                exceptions=(
                    RateLimitError,
                    Timeout,
                    APIConnectionError,
                    APIError,
                )
            ),
            retry(ServiceUnavailableError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        _ = hints

        try:
            response = await self._client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
        except RateLimitError:
            self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise

        vectors = [data_point.embedding for data_point in response.data]
        return EmbeddingResult(vectors=vectors)


class M2Bert32K(TogetherAIEmbedder):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="togethercomputer/m2-bert-80M-32k-retrieval",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )
        self._estimating_tokenizer = HuggingFaceEstimatingTokenizer(self.model_name)

    @property
    @override
    def id(self) -> str:
        return self.model_name

    @property
    @override
    def max_tokens(self) -> int:
        return 32 * 1024

    @property
    @override
    def tokenizer(self) -> HuggingFaceEstimatingTokenizer:
        return self._estimating_tokenizer

    @property
    @override
    def dimensions(self) -> int:
        return 768


class CustomTogetherAISchematicGenerator(TogetherAISchematicGenerator[T]):
    """Generic Together AI generator that accepts any model name."""

    def __init__(self, model_name: str, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name=model_name,
            logger=logger,
            tracer=tracer,
            meter=meter,
        )


class CustomTogetherAIEmbedder(TogetherAIEmbedder):
    """Generic Together AI embedder that accepts any model name."""

    def __init__(self, model_name: str, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(model_name=model_name, logger=logger, tracer=tracer, meter=meter)
        self._estimating_tokenizer = HuggingFaceEstimatingTokenizer(model_name)
        self._dimensions = int(os.environ.get("TOGETHER_EMBEDDING_DIMENSIONS", "768"))

    @property
    @override
    def id(self) -> str:
        return self.model_name

    @property
    @override
    def max_tokens(self) -> int:
        return int(os.environ.get("TOGETHER_EMBEDDING_MAX_TOKENS", "32768"))

    @property
    @override
    def tokenizer(self) -> HuggingFaceEstimatingTokenizer:
        return self._estimating_tokenizer

    @property
    @override
    def dimensions(self) -> int:
        return self._dimensions


class TogetherService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        required_vars = {
            "TOGETHER_API_KEY": "your-together-api-key",
            "TOGETHER_MODEL": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "TOGETHER_EMBEDDING_MODEL": "togethercomputer/m2-bert-80M-32k-retrieval",
        }

        missing_vars = []
        for var_name, example_value in required_vars.items():
            if not os.environ.get(var_name):
                missing_vars.append(f'export {var_name}="{example_value}"')

        if missing_vars:
            return f"""\
You're using the Together AI NLP service, but the following environment variables are not set:

{chr(10).join(missing_vars)}

Please set these environment variables before running Parlant.

Available models can be found at: https://docs.together.ai/docs/inference-models
"""

        return None

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        self.model_name = os.environ.get(
            "TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        )
        self.embedding_model = os.environ.get(
            "TOGETHER_EMBEDDING_MODEL", "togethercomputer/m2-bert-80M-32k-retrieval"
        )
        self._logger = logger
        self._tracer = tracer
        self._meter = meter

        self._logger.info(f"Initialized TogetherService with model: {self.model_name}")

    def _get_specialized_generator_class(
        self,
        model_name: str,
        schema_type: type[T],
    ) -> Callable[[Logger], TogetherAISchematicGenerator[T]] | None:
        """
        Returns the specialized generator class for known models, or None for custom models.
        """
        model_to_class: dict[str, Callable[[Logger], TogetherAISchematicGenerator[T]]] = {
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": Llama3_1_8B[schema_type],  # type: ignore
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": Llama3_1_70B[schema_type],  # type: ignore
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": Llama3_1_405B[schema_type],  # type: ignore
            "meta-llama/Llama-3.3-70B-Instruct-Turbo": Llama3_3_70B[schema_type],  # type: ignore
        }

        return model_to_class.get(model_name)

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> TogetherAISchematicGenerator[T]:
        specialized_class = self._get_specialized_generator_class(self.model_name, schema_type=t)

        if specialized_class:
            self._logger.debug(f"Using specialized generator for model: {self.model_name}")
            return specialized_class(self._logger)
        else:
            self._logger.debug(f"Using custom generator for model: {self.model_name}")
            return CustomTogetherAISchematicGenerator[t](  # type: ignore
                model_name=self.model_name,
                logger=self._logger,
                tracer=self._tracer,
                meter=self._meter,
            )

    def _get_specialized_embedder_class(
        self,
        model_name: str,
    ) -> Callable[[Logger, Tracer, Meter], TogetherAIEmbedder] | None:
        """
        Returns the specialized embedder class for known models, or None for custom models.
        """
        model_to_class: dict[str, Callable[[Logger, Tracer, Meter], TogetherAIEmbedder]] = {
            "togethercomputer/m2-bert-80M-32k-retrieval": M2Bert32K,
        }

        return model_to_class.get(model_name)

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        specialized_class = self._get_specialized_embedder_class(self.embedding_model)

        if specialized_class:
            self._logger.debug(f"Using specialized embedder for model: {self.embedding_model}")
            return specialized_class(self._logger, self._tracer, self._meter)
        else:
            self._logger.debug(f"Using custom embedder for model: {self.embedding_model}")
            return CustomTogetherAIEmbedder(
                model_name=self.embedding_model,
                logger=self._logger,
                tracer=self._tracer,
                meter=self._meter,
            )

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
