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
import os
import tiktoken
import jsonfinder  # type: ignore
from pydantic import ValidationError
from fireworks.client import AsyncFireworks  # type: ignore
from typing import Any, Mapping
from typing_extensions import override
from fireworks.client.error import RateLimitError  # type: ignore

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.adapters.nlp.hugging_face import JinaAIEmbedder
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.tracer import Tracer
from parlant.core.meter import Meter
from parlant.core.nlp.embedding import Embedder
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
    "Fireworks API rate limit exceeded. Possible reasons:\n"
    "1. Your account may have insufficient API credits.\n"
    "2. You may be using a free-tier account with limited request capacity.\n"
    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your Fireworks account balance and billing status.\n"
    "- Review your API usage limits in Fireworks dashboard.\n"
    "- For more details on rate limits and usage tiers, visit:\n"
    "  https://fireworks.ai/docs/guides/quotas_usage/rate-limits"
)


class FireworksEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens) + 36


class FireworksSchematicGenerator(BaseSchematicGenerator[T]):
    supported_hints = ["temperature", "max_tokens"]

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
        self._tracer = tracer
        self._meter = meter
        self._client = AsyncFireworks(api_key=os.environ.get("FIREWORKS_API_KEY"))
        self._tokenizer = FireworksEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return self.model_name

    @property
    @override
    def tokenizer(self) -> FireworksEstimatingTokenizer:
        return self._tokenizer

    @policy(
        [
            retry(
                exceptions=(
                    Exception,  # Will handle specific Fireworks exceptions
                ),
                max_exceptions=3,
                wait_times=(1.0, 2.0, 4.0),
            )
        ]
    )
    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self._logger.scope(f"Fireworks LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        fireworks_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        t_start = time.time()
        try:
            response = self._client.chat.completions.create(
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
                **fireworks_api_arguments,
            )
        except RateLimitError:
            self._logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise

        t_end = time.time()

        if response.usage:  # type: ignore
            self._logger.trace(f"Usage: {response.usage.model_dump_json(indent=2)}")  # type: ignore

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

            await record_llm_metrics(
                self._meter,
                self.model_name,
                input_tokens=response.usage.prompt_tokens,  # type: ignore
                output_tokens=response.usage.completion_tokens,  # type: ignore
            )

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


class FireworksLlama3_1_8B(FireworksSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="accounts/fireworks/models/llama-v3p1-8b-instruct",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024

    @property
    @override
    def tokenizer(self) -> FireworksEstimatingTokenizer:
        return self._tokenizer


class FireworksLlama3_1_70B(FireworksSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="accounts/fireworks/models/llama-v3p1-70b-instruct",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024

    @property
    @override
    def tokenizer(self) -> FireworksEstimatingTokenizer:
        return self._tokenizer


class FireworksLlama3_1_405B(FireworksSchematicGenerator[T]):
    """
    @warn: This is an extremely large model (405B parameters).
    Only suitable for high-performance workloads with significant budget considerations.
    """

    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="accounts/fireworks/models/llama-v3p1-405b-instruct",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024

    @property
    @override
    def tokenizer(self) -> FireworksEstimatingTokenizer:
        return self._tokenizer


class FireworksMythoMax(FireworksSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="accounts/fireworks/models/mythomax-l2-13b",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 4096

    @property
    @override
    def tokenizer(self) -> FireworksEstimatingTokenizer:
        return self._tokenizer


class FireworksGemma2_9B(FireworksSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="accounts/fireworks/models/gemma2-9b-it",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    @override
    def tokenizer(self) -> FireworksEstimatingTokenizer:
        return self._tokenizer


class CustomFireworksSchematicGenerator(FireworksSchematicGenerator[T]):
    """Generic Fireworks generator that accepts any model name."""

    def __init__(self, model_name: str, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name=model_name,
            logger=logger,
            tracer=tracer,
            meter=meter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 8192  # Default conservative limit

    @property
    @override
    def tokenizer(self) -> FireworksEstimatingTokenizer:
        return self._tokenizer


# Using JinaAIEmbedder for embeddings since Fireworks focuses on inference


class FireworksService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        required_vars = {
            "FIREWORKS_API_KEY": "<your_api_key_here>",
            "FIREWORKS_MODEL": "accounts/fireworks/models/llama-v3p1-8b-instruct",
        }

        missing_vars = []
        for var_name, default_value in required_vars.items():
            if not os.environ.get(var_name):
                if default_value:
                    missing_vars.append(f'export {var_name}="{default_value}"')
                else:
                    missing_vars.append(f'export {var_name}="<your_{var_name.lower()}>"')

        if missing_vars:
            return f"""\
You're using the Fireworks NLP service, but the following environment variables are not set:

{chr(10).join(missing_vars)}

Please set these environment variables before running Parlant.
You can get your API key from: https://app.fireworks.ai/settings/users/api-keys
"""

        return None

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        self._model_name = os.environ.get(
            "FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct"
        )
        self._embedding_model = os.environ.get(  # Need to be implemented
            "FIREWORKS_EMBEDDING_MODEL", "accounts/fireworks/models/qwen3-embedding-8b"
        )
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._logger.info(f"Initialized FireworksService with {self._model_name}")

    def _get_specialized_generator_class(
        self,
        model_name: str,
        schema_type: type[T],
    ) -> type[FireworksSchematicGenerator[T]] | None:
        model_to_class: dict[str, type[FireworksSchematicGenerator[T]]] = {
            "accounts/fireworks/models/llama-v3p1-8b-instruct": FireworksLlama3_1_8B[schema_type],  # type: ignore
            "accounts/fireworks/models/llama-v3p1-70b-instruct": FireworksLlama3_1_70B[schema_type],  # type: ignore
            "accounts/fireworks/models/llama-v3p1-405b-instruct": FireworksLlama3_1_405B[
                schema_type  # type: ignore
            ],
            "accounts/fireworks/models/mythomax-l2-13b": FireworksMythoMax[schema_type],  # type: ignore
            "accounts/fireworks/models/gemma2-9b-it": FireworksGemma2_9B[schema_type],  # type: ignore
        }

        return model_to_class.get(model_name)

    def _log_model_warnings(self, model_name: str) -> None:
        """Log warnings for resource-intensive models."""
        if "405b" in model_name.lower():
            self._logger.warning(
                f"Using {model_name} - This is an extremely large model with significant cost implications. "
                "Consider using smaller models for development and testing."
            )

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> FireworksSchematicGenerator[T]:
        """Get a schematic generator for the specified type."""
        self._log_model_warnings(self._model_name)

        specialized_class = self._get_specialized_generator_class(self._model_name, schema_type=t)

        if specialized_class:
            self._logger.debug(f"Using specialized generator for model: {self._model_name}")
            return specialized_class(
                model_name=self._model_name,
                logger=self._logger,
                tracer=self._tracer,
                meter=self._meter,
            )
        else:
            self._logger.debug(f"Using custom generator for model: {self._model_name}")
            return CustomFireworksSchematicGenerator[t](  # type: ignore
                model_name=self._model_name,
                logger=self._logger,
                tracer=self._tracer,
                meter=self._meter,
            )

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        return JinaAIEmbedder(self._logger, self._tracer, self._meter)

    @override
    async def get_moderation_service(self) -> ModerationService:
        """Fireworks doesn't provide moderation services, so we use no moderation."""
        return NoModeration()


MODEL_RECOMMENDATIONS = {
    "accounts/fireworks/models/llama-v3p1-8b-instruct": "Fast and cost-effective for most use cases",
    "accounts/fireworks/models/llama-v3p1-70b-instruct": "High accuracy for complex reasoning tasks",
    "accounts/fireworks/models/llama-v3p1-405b-instruct": "@warn: Extremely expensive, use only for critical workloads",
    "accounts/fireworks/models/gemma2-9b-it": "Good balance of speed and accuracy",
    "accounts/fireworks/models/mythomax-l2-13b": "Creative writing and roleplay scenarios",
}
