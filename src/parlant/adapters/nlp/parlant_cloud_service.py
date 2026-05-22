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
import asyncio
from pprint import pformat
import re
import time
from typing import Any, AsyncIterator, Callable, Mapping, TypeAlias, cast
from httpx import AsyncClient
import httpx
from typing_extensions import Literal, override
import json
import jsonfinder  # type: ignore
import os

from pydantic import ValidationError
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.core.agents import AgentStore
from parlant.core.application_context import ApplicationContext
from parlant.core.canned_responses import CannedResponseStore
from parlant.core.context_variables import ContextVariableStore
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.glossary import GlossaryStore
from parlant.core.guidelines import GuidelineStore
from parlant.core.journeys import JourneyStore
from parlant.core.loggers import Logger
from parlant.core.relationships import RelationshipStore
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.meter import Meter
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import (
    EmbedderHints,
    ModelSize,
    NLPService,
    SchematicGeneratorHints,
    StreamingTextGeneratorHints,
)
from parlant.core.nlp.embedding import BaseEmbedder, Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    BaseStreamingTextGenerator,
    SchematicGenerationResult,
    StreamingTextGenerator,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.moderation import (
    ModerationService,
    NoModeration,
)
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.services.indexing.indexer import IndexRequest, Indexer
from parlant.core.tracer import Tracer
from parlant.core.version import VERSION
from parlant.core.health import HealthReporter


RATE_LIMIT_ERROR_MESSAGE = (
    "Parlant Cloud API rate limit exceeded. Possible reasons:\n"
    "1. Your account may have insufficient API credits.\n"
    "2. You might have exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your Parlant Cloud account balance and billing status.\n"
    "- Review your API usage limits in Parlant Cloud's dashboard.\n"
    "- For more details on rate limits and usage tiers, visit:\n"
    "  https://parlant.io\n"
)

GenerationModelTier: TypeAlias = Literal["jackal", "bison"]
EmbeddingModelTier: TypeAlias = Literal["jackal-embedding", "bison-embedding"]
ModelRole: TypeAlias = Literal["teacher", "student", "auto"]

BASE_URL = os.environ.get("PARLANT_CLOUD_API_URL", "https://api.parlant.cloud/inference")

# Pattern to detect word boundaries for chunking
# Matches after any whitespace character
_WORD_BOUNDARY_PATTERN = re.compile(r"(?<=\s)")

# Number of words to buffer before yielding a chunk
_WORDS_PER_CHUNK = 3


class ParlantCloudEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self) -> None:
        self.encoding = tiktoken.encoding_for_model("gpt-4.1")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class ParlantCloudAPIError(Exception):
    pass


class InsufficientCreditsError(ParlantCloudAPIError):
    pass


class RateLimitError(ParlantCloudAPIError):
    pass


class UnauthorizedError(ParlantCloudAPIError):
    pass


def _get_error_detail(response: httpx.Response) -> tuple[str, str]:
    try:
        error_message = (
            response.json().get("detail", {}).get("error", {}).get("message", "Unknown error")
        )
        request_id = response.json().get("detail", {}).get("request_id", "N/A")
    except Exception:
        try:
            error_message = response.text
        except Exception:
            error_message = "Unknown error (failed to parse error message)"

        request_id = "N/A"

    return error_message, request_id


class ParlantCloudSchematicGenerator(BaseSchematicGenerator[T]):
    supported_parlant_cloud_params = ["temperature"]

    def __init__(
        self,
        model_name: str,
        model_role: ModelRole,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        health_reporter: HealthReporter,
    ) -> None:
        super().__init__(
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
            model_name=model_name,
        )

        self._model_role = model_role
        self._tokenizer = ParlantCloudEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        return f"parlant-cloud/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> ParlantCloudEstimatingTokenizer:
        return self._tokenizer

    @policy(
        [
            retry(exceptions=(RateLimitError)),
            retry(ParlantCloudAPIError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self.logger.scope(f"Parlant Cloud LLM Request ({self.schema.__name__})"):
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
                connect=30.0,
                read=120.0,
                write=30.0,
                pool=5.0,
            )

            async with AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{BASE_URL}/v1/completions",
                    headers={
                        "Authorization": f"Bearer {os.environ['PARLANT_CLOUD_API_KEY']}",
                        "X-Parlant-Version": VERSION,
                    },
                    json={
                        "model_tier": self.model_name,
                        "model_role": self._model_role,
                        "prompt": prompt,
                        "schema_name": self.schema.__name__,
                        "hints": {
                            k: v
                            for k, v in hints.items()
                            if k in self.supported_parlant_cloud_params
                        },
                        "payload": props,
                    },
                )

                if response.is_error:
                    error_message, request_id = _get_error_detail(response)

                if response.status_code == 429:
                    raise RateLimitError(
                        f"Parlant Cloud API rate limit exceeded: {error_message} (RID={request_id})"
                    )
                elif response.status_code == 402:
                    raise InsufficientCreditsError(
                        f"Insufficient API credits for Parlant Cloud API: {error_message} (RID={request_id})"
                    )
                elif response.status_code == 403:
                    raise UnauthorizedError(
                        f"Unauthorized access to Parlant Cloud API: {error_message} (RID={request_id})"
                    )
                elif response.status_code >= 500:
                    raise ParlantCloudAPIError(
                        f"Parlant Cloud API error: {response.status_code} {error_message} (RID={request_id})"
                    )

                response.raise_for_status()

            t_end = time.time()
        except (InsufficientCreditsError, RateLimitError):
            self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise
        except ParlantCloudAPIError as e:
            self.logger.error(f"Parlant Cloud API error occurred: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during Parlant Cloud API call: {e}")
            raise

        response_data = response.json()

        usage = response_data["usage"]
        cost = response_data["cost"]

        self.logger.trace(f"Parlant Cloud usage data:\n{pformat({**usage, **cost})}")

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


class Jackal(ParlantCloudSchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        health_reporter: HealthReporter,
        model_role: ModelRole,
    ) -> None:
        super().__init__(
            model_name="jackal",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
            model_role=model_role,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class Bison(ParlantCloudSchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        health_reporter: HealthReporter,
        model_role: ModelRole,
    ) -> None:
        super().__init__(
            model_name="bison",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
            model_role=model_role,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


# ============================================================================
# Streaming Text Generators
# ============================================================================


class ParlantCloudStreamingTextGenerator(BaseStreamingTextGenerator):
    """Streaming text generator using Parlant Cloud's streaming API.

    Buffers tokens into word-sized chunks for smoother frontend rendering.
    """

    supported_parlant_cloud_params = ["temperature"]

    def __init__(
        self,
        model_name: str,
        model_role: ModelRole,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        health_reporter: HealthReporter,
    ) -> None:
        super().__init__(
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
            model_name=model_name,
        )
        self._model_role = model_role
        self._tokenizer = ParlantCloudEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        return f"parlant-cloud-streaming/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> ParlantCloudEstimatingTokenizer:
        return self._tokenizer

    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> tuple[AsyncIterator[str | None], Callable[[], UsageInfo]]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        # Track usage from the done event
        usage_info: UsageInfo | None = None

        async def chunk_generator() -> AsyncIterator[str | None]:
            nonlocal usage_info

            timeout = httpx.Timeout(
                connect=30.0,
                read=120.0,
                write=30.0,
                pool=5.0,
            )

            # Buffer for accumulating tokens into word-sized chunks
            buffer = ""

            async with AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{BASE_URL}/v1/completions",
                    headers={
                        "Authorization": f"Bearer {os.environ['PARLANT_CLOUD_API_KEY']}",
                        "X-Parlant-Version": VERSION,
                    },
                    json={
                        "model_tier": self.model_name,
                        "model_role": self._model_role,
                        "prompt": prompt,
                        "stream": True,
                        "hints": {
                            k: v
                            for k, v in hints.items()
                            if k in self.supported_parlant_cloud_params
                        },
                    },
                ) as response:
                    # Check status before iterating to catch auth/rate-limit errors early
                    if response.is_error:
                        await response.aread()
                        error_message, request_id = _get_error_detail(response)

                    if response.status_code == 429:
                        self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
                        raise RateLimitError(
                            f"Parlant Cloud API rate limit exceeded: {error_message} (RID={request_id})"
                        )
                    elif response.status_code == 402:
                        self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
                        raise InsufficientCreditsError(
                            f"Insufficient API credits for Parlant Cloud API: {error_message} (RID={request_id})"
                        )
                    elif response.status_code == 403:
                        raise UnauthorizedError(
                            f"Unauthorized access to Parlant Cloud API: {error_message} (RID={request_id})"
                        )
                    elif response.status_code >= 500:
                        raise ParlantCloudAPIError(
                            f"Parlant Cloud API error: {response.status_code} {error_message} (RID={request_id})"
                        )

                    response.raise_for_status()

                    # Parse SSE events
                    event_type: str | None = None

                    async for line in response.aiter_lines():
                        if line.startswith("event: "):
                            event_type = line[7:]
                        elif line.startswith("data: ") and event_type:
                            data = json.loads(line[6:])

                            if event_type == "chunk":
                                text = data.get("text", "")
                                if text:
                                    buffer += text

                                    # Count word boundaries in buffer
                                    boundaries = list(_WORD_BOUNDARY_PATTERN.finditer(buffer))
                                    if len(boundaries) >= _WORDS_PER_CHUNK:
                                        # Yield up to the last complete word boundary
                                        last_boundary = boundaries[_WORDS_PER_CHUNK - 1]
                                        chunk_text = buffer[: last_boundary.end()]
                                        buffer = buffer[last_boundary.end() :]
                                        yield chunk_text

                            elif event_type == "done":
                                usage = data.get("usage", {})
                                usage_info = UsageInfo(
                                    input_tokens=int(usage.get("input_tokens", 0)),
                                    output_tokens=int(usage.get("output_tokens", 0)),
                                    extra={},
                                )

                                self.logger.trace(
                                    f"Parlant Cloud streaming usage data:\n{pformat(data)}"
                                )

                                # Yield any remaining content in the buffer
                                if buffer:
                                    yield buffer
                                    buffer = ""

                            elif event_type == "error":
                                error_msg = data.get("error", {}).get("message", "Unknown error")
                                raise ParlantCloudAPIError(
                                    f"Parlant Cloud streaming error: {error_msg}"
                                )

            # Record metrics if we have usage info
            if usage_info is not None:
                await record_llm_metrics(
                    self.meter,
                    self.model_name,
                    schema_name="streaming",
                    input_tokens=usage_info.input_tokens,
                    output_tokens=usage_info.output_tokens,
                    cached_input_tokens=0,
                )

            # Signal completion
            yield None

        def get_usage() -> UsageInfo:
            if usage_info is None:
                return UsageInfo(input_tokens=0, output_tokens=0, extra={})
            return usage_info

        return chunk_generator(), get_usage


class JackalStreaming(ParlantCloudStreamingTextGenerator):
    def __init__(
        self,
        model_role: ModelRole,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        health_reporter: HealthReporter,
    ) -> None:
        super().__init__(
            model_name="jackal",
            model_role=model_role,
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )


class BisonStreaming(ParlantCloudStreamingTextGenerator):
    def __init__(
        self,
        model_role: ModelRole,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        health_reporter: HealthReporter,
    ) -> None:
        super().__init__(
            model_name="bison",
            model_role=model_role,
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )


# ============================================================================
# Embedders
# ============================================================================


class ParlantCloudEmbedder(BaseEmbedder):
    supported_arguments = ["dimensions"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        health_reporter: HealthReporter,
    ) -> None:
        super().__init__(logger, tracer, meter, model_name, health_reporter)
        self._tokenizer = ParlantCloudEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        return f"parlant-cloud/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> ParlantCloudEstimatingTokenizer:
        return self._tokenizer

    @policy(
        [
            retry(exceptions=(RateLimitError)),
            retry(ParlantCloudAPIError, max_exceptions=2, wait_times=(1.0, 5.0)),
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
                        "Authorization": f"Bearer {os.environ['PARLANT_CLOUD_API_KEY']}",
                        "X-Parlant-Version": VERSION,
                    },
                    json={
                        "model_tier": self.model_name,
                        "inputs": texts,
                        "hints": {k: v for k, v in hints.items() if k in self.supported_arguments},
                    },
                )

                if response.is_error:
                    error_message, request_id = _get_error_detail(response)

                if response.status_code == 429:
                    raise RateLimitError(
                        f"Parlant Cloud API rate limit exceeded: {error_message} (RID={request_id})"
                    )
                elif response.status_code == 402:
                    raise InsufficientCreditsError(
                        f"Insufficient API credits for Parlant Cloud API: {error_message} (RID={request_id})"
                    )
                elif response.status_code == 403:
                    raise UnauthorizedError(
                        f"Unauthorized access to Parlant Cloud API: {error_message} (RID={request_id})"
                    )
                elif response.status_code >= 500:
                    raise ParlantCloudAPIError(
                        f"Parlant Cloud API error: {response.status_code} {error_message} (RID={request_id})"
                    )

                response.raise_for_status()
        except (RateLimitError, InsufficientCreditsError):
            self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during Parlant Cloud API call: {e}")
            raise

        response_data = response.json()
        vectors = [data_point["embedding"] for data_point in response_data["data"]]
        return EmbeddingResult(vectors=vectors)


class BisonEmbedding(ParlantCloudEmbedder):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="bison-embedding",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        return 3072


class JackalEmbedding(ParlantCloudEmbedder):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="jackal-embedding",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        return 1536


class ParlantCloudIndexer(Indexer):
    def __init__(
        self,
        agent_store: AgentStore,
        guideline_store: GuidelineStore,
        journey_store: JourneyStore,
        relationship_store: RelationshipStore,
        glossary_store: GlossaryStore,
        context_variable_store: ContextVariableStore,
        canned_response_store: CannedResponseStore,
        service_registry: ServiceRegistry,
        application_context: ApplicationContext,
        logger: Logger,
    ) -> None:
        super().__init__(
            agent_store=agent_store,
            guideline_store=guideline_store,
            journey_store=journey_store,
            relationship_store=relationship_store,
            glossary_store=glossary_store,
            context_variable_store=context_variable_store,
            canned_response_store=canned_response_store,
            service_registry=service_registry,
        )
        self._application_context = application_context
        self._logger = logger

    @override
    async def index(
        self,
        payload: Mapping[str, Mapping[str, IndexRequest]],
        progress_report: ProgressReport,
    ) -> None:
        serialized_entities: dict[str, dict[str, dict[str, Any]]] = {
            category: {
                entity_id: {
                    "type": request.type,
                    "id": request.id,
                    "last_modification_utc": request.last_modification_utc.isoformat(),
                    "checksum": request.checksum,
                    "data": request.data,
                }
                for entity_id, request in entities.items()
            }
            for category, entities in payload.items()
        }

        body = {
            "instance_id": self._application_context.instance_id,
            "entities": serialized_entities,
        }

        timeout = httpx.Timeout(connect=30.0, read=60.0, write=120.0, pool=5.0)

        async with AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{BASE_URL}/v1/index",
                headers={
                    "Authorization": f"Bearer {os.environ['PARLANT_CLOUD_API_KEY']}",
                    "X-Parlant-Version": VERSION,
                },
                json=body,
            )

            if response.is_error:
                error_message, request_id = _get_error_detail(response)

                if response.status_code == 429:
                    raise RateLimitError(
                        f"Parlant Cloud API rate limit exceeded: {error_message} (RID={request_id})"
                    )
                elif response.status_code == 402:
                    raise InsufficientCreditsError(
                        f"Insufficient API credits for Parlant Cloud API: {error_message} (RID={request_id})"
                    )
                elif response.status_code == 403:
                    raise UnauthorizedError(
                        f"Unauthorized access to Parlant Cloud API: {error_message} (RID={request_id})"
                    )
                else:
                    raise ParlantCloudAPIError(
                        f"Parlant Cloud API error: {response.status_code} {error_message} (RID={request_id})"
                    )

            status_url = response.json()["status_url"]
            if not status_url.startswith("http"):
                status_url = f"{BASE_URL}{status_url}"

            last_completed = 0

            while True:
                status_response = await client.get(
                    status_url,
                    headers={
                        "Authorization": f"Bearer {os.environ['PARLANT_CLOUD_API_KEY']}",
                        "X-Parlant-Version": VERSION,
                    },
                )

                if status_response.is_error:
                    error_message, request_id = _get_error_detail(status_response)
                    raise ParlantCloudAPIError(
                        f"Parlant Cloud indexing status error: {status_response.status_code} {error_message} (RID={request_id})"
                    )

                status_data = status_response.json()
                completed = int(status_data.get("completed", 0))
                delta = completed - last_completed
                if delta > 0:
                    await progress_report.increment(delta)
                    last_completed = completed

                status = status_data.get("status")
                if status == "completed":
                    return
                if status == "failed":
                    raise ParlantCloudAPIError(f"Parlant Cloud indexing failed: {status_data}")

                await asyncio.sleep(2)


class ParlantCloudService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("PARLANT_CLOUD_API_KEY"):
            return """\
You're using Parlant Cloud's optimized NLP service, but PARLANT_CLOUD_API_KEY is not set.
Please set PARLANT_CLOUD_API_KEY in your environment before running Parlant.

For alternative providers, see https://parlant.io/docs/quickstart/installation.

Get an API key for Parlant Cloud by signing up at https://www.parlant.io."""

        return None

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        health_reporter: HealthReporter,
        model_tier: GenerationModelTier | None = None,
        model_role: ModelRole | None = None,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._health_reporter = health_reporter

        self._model_tier = model_tier or os.environ.get("PARLANT_CLOUD_MODEL_TIER", "jackal")
        self._model_role = model_role or os.environ.get("PARLANT_CLOUD_MODEL_ROLE", "auto")

        assert self._model_tier in ("jackal", "bison"), "Invalid PARLANT_CLOUD_MODEL_TIER"
        assert self._model_role in ("teacher", "student", "auto"), (
            "Invalid PARLANT_CLOUD_MODEL_ROLE"
        )

        self._logger.info("Initialized ParlantCloudService")

    @property
    @override
    def supports_streaming(self) -> bool:
        return True

    @override
    async def get_streaming_text_generator(
        self, hints: StreamingTextGeneratorHints = {}
    ) -> StreamingTextGenerator:
        match self._model_tier:
            case "bison":
                return BisonStreaming(
                    model_role=cast(ModelRole, self._model_role),
                    logger=self._logger,
                    tracer=self._tracer,
                    meter=self._meter,
                    health_reporter=self._health_reporter,
                )
            case _:
                return JackalStreaming(
                    model_role=cast(ModelRole, self._model_role),
                    logger=self._logger,
                    tracer=self._tracer,
                    meter=self._meter,
                    health_reporter=self._health_reporter,
                )

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> ParlantCloudSchematicGenerator[T]:
        match self._model_tier:
            case "jackal":
                return Jackal[t](  # type: ignore
                    model_role=cast(ModelRole, self._model_role),
                    logger=self._logger,
                    tracer=self._tracer,
                    meter=self._meter,
                    health_reporter=self._health_reporter,
                )
            case "bison":
                return Bison[t](  # type: ignore
                    model_role=cast(ModelRole, self._model_role),
                    logger=self._logger,
                    tracer=self._tracer,
                    meter=self._meter,
                    health_reporter=self._health_reporter,
                )
            case _:
                raise ValueError(f"Unsupported model tier: {self._model_tier}")

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        match hints.get("model_size", ModelSize.AUTO):
            case ModelSize.AUTO | ModelSize.LARGE:
                return BisonEmbedding(
                    self._logger, self._tracer, self._meter, self._health_reporter
                )
            case _:
                return JackalEmbedding(
                    self._logger, self._tracer, self._meter, self._health_reporter
                )

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
