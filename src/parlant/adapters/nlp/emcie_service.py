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
from pprint import pformat
import time
from typing import Any, Mapping, TypeAlias, cast
from httpx import AsyncClient
import httpx
from typing_extensions import Literal, override
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
from parlant.core.nlp.service import EmbedderHints, ModelSize, NLPService, SchematicGeneratorHints
from parlant.core.nlp.embedding import BaseEmbedder, Embedder, EmbeddingResult
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
from parlant.core.tracer import Tracer
from parlant.core.version import VERSION


ERROR_MESSAGE = (
    "Emcie API rate limit exceeded. Possible reasons:\n"
    "1. Your account may have insufficient API credits.\n"
    "2. You might have exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your Emcie account balance and billing status.\n"
    "- Review your API usage limits in Emcie's dashboard.\n"
    "- For more details on rate limits and usage tiers, visit:\n"
    "  https://docs.emcie.co\n"
)

GenerationModelTier: TypeAlias = Literal["jackal", "bison"]
EmbeddingModelTier: TypeAlias = Literal["jackal-embedding", "bison-embedding"]
ModelRole: TypeAlias = Literal["teacher", "student", "auto"]

BASE_URL = os.environ.get("EMCIE_API_URL", "https://api.emcie.co/inference")


class EmcieEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self) -> None:
        self.encoding = tiktoken.encoding_for_model("gpt-4.1")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class EmcieAPIError(Exception):
    pass


class InsufficientCreditsError(EmcieAPIError):
    pass


class RateLimitError(EmcieAPIError):
    pass


class UnauthorizedError(EmcieAPIError):
    pass


class EmcieSchematicGenerator(BaseSchematicGenerator[T]):
    supported_emcie_params = ["temperature"]

    def __init__(
        self,
        model_name: str,
        model_role: ModelRole,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self._model_role = model_role
        self._tokenizer = EmcieEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        return f"emcie/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> EmcieEstimatingTokenizer:
        return self._tokenizer

    @policy(
        [
            retry(exceptions=(RateLimitError)),
            retry(EmcieAPIError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self.logger.scope(f"Emcie LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            props = prompt.props
            prompt = prompt.build()
        else:
            props = {}

        try:
            t_start = time.time()

            timeout = httpx.Timeout(
                connect=5.0,
                read=120.0,
                write=30.0,
                pool=5.0,
            )

            async with AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{BASE_URL}/v1/completions",
                    headers={
                        "Authorization": f"Bearer {os.environ['EMCIE_API_KEY']}",
                        "X-Parlant-Version": VERSION,
                    },
                    json={
                        "model_tier": self.model_name,
                        "model_role": self._model_role,
                        "prompt": prompt,
                        "schema_name": self.schema.__name__,
                        "hints": {
                            k: v for k, v in hints.items() if k in self.supported_emcie_params
                        },
                        "payload": props,
                    },
                )

                if response.status_code == 429:
                    raise RateLimitError(
                        f"Emcie API rate limit exceeded: {response.json()['detail']['error']['message']} (RID={response.json()['detail']['request_id']})"
                    )
                elif response.status_code == 402:
                    raise InsufficientCreditsError(
                        f"Insufficient API credits for Emcie API: {response.json()['detail']['error']['message']} (RID={response.json()['detail']['request_id']})"
                    )
                elif response.status_code == 403:
                    raise UnauthorizedError(
                        f"Unauthorized access to Emcie API: {response.json()['detail']['error']['message']} (RID={response.json()['detail']['request_id']})"
                    )
                elif response.status_code >= 500:
                    raise EmcieAPIError(
                        f"Emcie API error: {response.status_code} {response.json()['detail']['error']['message']} (RID={response.json()['detail']['request_id']})"
                    )

                response.raise_for_status()

            t_end = time.time()
        except (InsufficientCreditsError, RateLimitError):
            self.logger.error(ERROR_MESSAGE)
        except EmcieAPIError as e:
            self.logger.error(f"Emcie API error occurred: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during Emcie API call: {e}")
            raise

        response_data = response.json()

        usage = response_data["usage"]
        cost = response_data["cost"]

        self.logger.trace(f"Emcie usage data:\n{pformat({**usage, **cost})}")

        raw_content = response_data["completion"]

        try:
            json_content = json.loads(normalize_json_output(raw_content))
        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content})")
            json_content = jsonfinder.only_json(raw_content)[2]
            self.logger.warning("Found JSON content within model response; continuing...")

        try:
            content = self.schema.model_validate(json_content)

            await record_llm_metrics(
                self.meter,
                self.model_name,
                schema_name=self.schema.__name__,
                input_tokens=int(usage["input_tokens"]),
                output_tokens=int(usage["output_tokens"]),
                cached_input_tokens=0,
            )

            return SchematicGenerationResult(
                content=content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=int(usage["input_tokens"]),
                        output_tokens=int(usage["output_tokens"]),
                        extra={},
                    ),
                ),
            )

        except ValidationError as e:
            self.logger.error(
                f"Error: {e.json(indent=2)}\nJSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class Jackal(EmcieSchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        model_role: ModelRole,
    ) -> None:
        super().__init__(
            model_name="jackal",
            logger=logger,
            tracer=tracer,
            meter=meter,
            model_role=model_role,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class Bison(EmcieSchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        model_role: ModelRole,
    ) -> None:
        super().__init__(
            model_name="bison",
            logger=logger,
            tracer=tracer,
            meter=meter,
            model_role=model_role,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class EmcieEmbedder(BaseEmbedder):
    supported_arguments = ["dimensions"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        super().__init__(logger, tracer, meter, model_name)
        self._tokenizer = EmcieEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        return f"emcie/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> EmcieEstimatingTokenizer:
        return self._tokenizer

    @policy(
        [
            retry(exceptions=(RateLimitError)),
            retry(EmcieAPIError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        try:
            timeout = httpx.Timeout(
                connect=5.0,
                read=120.0,
                write=30.0,
                pool=5.0,
            )

            async with AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{BASE_URL}/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {os.environ['EMCIE_API_KEY']}",
                        "X-Parlant-Version": VERSION,
                    },
                    json={
                        "model_tier": self.model_name,
                        "inputs": texts,
                        "hints": {k: v for k, v in hints.items() if k in self.supported_arguments},
                    },
                )

                if response.status_code == 429:
                    raise RateLimitError(
                        f"Emcie API rate limit exceeded: {response.json()['detail']['error']['message']} (RID={response.json()['detail']['request_id']})"
                    )
                elif response.status_code == 402:
                    raise InsufficientCreditsError(
                        f"Insufficient API credits for Emcie API: {response.json()['detail']['error']['message']} (RID={response.json()['detail']['request_id']})"
                    )
                elif response.status_code == 403:
                    raise UnauthorizedError(
                        f"Unauthorized access to Emcie API: {response.json()['detail']['error']['message']} (RID={response.json()['detail']['request_id']})"
                    )
                elif response.status_code >= 500:
                    raise EmcieAPIError(
                        f"Emcie API error: {response.status_code} {response.json()['detail']['error']['message']} (RID={response.json()['detail']['request_id']})"
                    )

                response.raise_for_status()
        except RateLimitError:
            self.logger.error(ERROR_MESSAGE)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during Emcie API call: {e}")
            raise

        response_data = response.json()
        vectors = [data_point["embedding"] for data_point in response_data["data"]]
        return EmbeddingResult(vectors=vectors)


class BisonEmbedding(EmcieEmbedder):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="bison-embedding",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        return 3072


class JackalEmbedding(EmcieEmbedder):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="jackal-embedding",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        return 1536


class EmcieService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("EMCIE_API_KEY"):
            return """\
You're using Emcie's optimized NLP service, but EMCIE_API_KEY is not set.
Please set EMCIE_API_KEY in your environment before running Parlant.

For alternative providers, see https://parlant.io/docs/quickstart/installation.

Get an API key for Emcie by signing up at https://www.emcie.co."""

        return None

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        model_tier: GenerationModelTier | None = None,
        model_role: ModelRole | None = None,
    ) -> None:
        self._logger = logger
        self._meter = meter
        self._tracer = tracer

        self._model_tier = model_tier or os.environ.get("EMCIE_MODEL_TIER", "jackal")
        self._model_role = model_role or os.environ.get("EMCIE_MODEL_ROLE", "auto")

        assert self._model_tier in ("jackal", "bison"), "Invalid EMCIE_MODEL_TIER"
        assert self._model_role in ("teacher", "student", "auto"), "Invalid EMCIE_MODEL_ROLE"

        self._logger.info("Initialized EmcieService")

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> EmcieSchematicGenerator[T]:
        match self._model_tier:
            case "jackal":
                return Jackal[t](  # type: ignore
                    model_role=cast(ModelRole, self._model_role),
                    logger=self._logger,
                    tracer=self._tracer,
                    meter=self._meter,
                )
            case "bison":
                return Bison[t](  # type: ignore
                    model_role=cast(ModelRole, self._model_role),
                    logger=self._logger,
                    tracer=self._tracer,
                    meter=self._meter,
                )
            case _:
                raise ValueError(f"Unsupported model tier: {self._model_tier}")

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        match hints.get("model_size", ModelSize.AUTO):
            case ModelSize.AUTO | ModelSize.LARGE:
                return BisonEmbedding(self._logger, self._tracer, self._meter)
            case _:
                return JackalEmbedding(self._logger, self._tracer, self._meter)

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
