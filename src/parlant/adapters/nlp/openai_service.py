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
from itertools import chain
import re
import time
import httpx2
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AsyncClient,
    ConflictError,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)
from typing import Any, AsyncIterator, Iterator, Mapping, Optional, Sequence, cast
from typing_extensions import override
import itertools
import json
import jsonfinder  # type: ignore

from pydantic import ValidationError
import tiktoken
from parlant.core.common import JSONSerializable, to_prompt_json
from parlant.core.container import Container

from parlant.adapters.nlp.common import (
    RESULT_ONLY_TOOL_NAME,
    result_only_tool_output,
    parlant_client_headers,
    with_client_identity,
    decorate_schematic_generator,
    normalize_json_output,
    record_llm_metrics,
    require_api_key,
    validate_model_arguments,
    validate_service_tier,
)

from parlant.core.engines.compass.prompt_builder import PromptBuilder
from parlant.core.loggers import ScopedLogger, Logger
from parlant.core.meter import Meter
from parlant.core.nlp.common import ModelSize, UsageInfo
from parlant.core.nlp.decision import (
    Answer,
    Choice,
    ChoiceAnswer,
    Criterion,
    DecisionError,
    DecisionMaker,
    DecisionResult,
    Proposition,
    PropositionAnswer,
    Question,
    Scale,
    ScaleAnswer,
)
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.embedding import BaseEmbedder, EmbeddingResult
from parlant.core.nlp.generation import (
    REASONING_EFFORT_HINT,
    T,
    BaseSchematicGenerator,
    ReasoningEffort,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.nlp.moderation import (
    BaseModerationService,
    ModerationCheck,
    ModerationContext,
    ModerationTag,
)
from parlant.core.nlp.model_routing import ModelCapability, ModelConfig, ModelImplementationRegistry
from parlant.core.nlp.react import (
    EMPTY_CONVERSATION_USER_TURN,
    CacheConfig,
    FinishReason,
    Message,
    ReactError,
    ReactGenerator,
    ReactGeneratorHints,
    ReasoningConfig,
    Role,
    ServiceTier,
    StreamEvent,
    TextDelta,
    TextPart,
    ReasoningDelta,
    ToolCallPart,
    ToolCallStarted,
    ToolChoice,
    ToolResultPart,
    ToolSpec,
    TurnBuilder,
    Usage,
)
from parlant.core.tracer import Tracer
from parlant.core.usage_reporter import UsageReporter


# Explicit request timeout for the OpenAI client. Without it the SDK defaults to a
# 600s read timeout, so a stalled call hangs ~10 minutes and freezes session
# processing (no agent events → the /events endpoint appears to hang). connect
# stays short. max_retries=0 leaves retrying to our own policy / the loop rather
# than triple-stacking the timeout. For streaming the read timeout is per-chunk
# (max gap between tokens), so it doesn't cut off long-but-active responses.
# An httpx2 object, not an httpx one: openai 3.0 moved the SDK onto HTTPX2, and
# its client rejects a legacy httpx timeout.
_OPENAI_TIMEOUT = httpx2.Timeout(60.0, connect=5.0)
_OPENAI_MAX_RETRIES = 0


RATE_LIMIT_ERROR_MESSAGE = (
    "OpenAI API rate limit exceeded. Possible reasons:\n"
    "1. Your account may have insufficient API credits.\n"
    "2. You may be using a free-tier account with limited request capacity.\n"
    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your OpenAI account balance and billing status.\n"
    "- Review your API usage limits in OpenAI's dashboard.\n"
    "- For more details on rate limits and usage tiers, visit:\n"
    "  https://platform.openai.com/docs/guides/rate-limits/usage-tiers\n"
)


class OpenAIEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

        # All gpt-5.x variants share the gpt-5 (o200k_base) encoding, but tiktoken
        # doesn't know newer point releases (e.g. gpt-5.4-nano) by name — so map
        # any gpt-5* model to plain "gpt-5".
        if model_name.startswith("gpt-5"):
            model_name_query = "gpt-5"
        else:
            model_name_query = model_name

        self.encoding = tiktoken.encoding_for_model(model_name_query)

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class OpenAISchematicGenerator(BaseSchematicGenerator[T]):
    supported_openai_params = ["temperature", "logit_bias", "max_tokens"]
    supported_hints = supported_openai_params + ["strict"]
    unsupported_params_by_model: dict[str, list[str]] = {
        "gpt-5": ["temperature"],
    }

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        tokenizer_model_name: str | None = None,
        usage_reporter: UsageReporter | None = None,
        api_key: str | None = None,
        provider_name: str | None = None,
    ) -> None:
        super().__init__(
            logger=logger,
            tracer=tracer,
            meter=meter,
            model_name=model_name,
            usage_reporter=usage_reporter,
            provider_name=provider_name or "openai",
        )

        self._client = with_client_identity(
            AsyncClient(
                default_headers=parlant_client_headers(),
                api_key=require_api_key("OpenAI", "OPENAI_API_KEY", api_key),
                timeout=_OPENAI_TIMEOUT,
                max_retries=_OPENAI_MAX_RETRIES,
            )
        )

        self._tokenizer = OpenAIEstimatingTokenizer(
            model_name=tokenizer_model_name or self.model_name
        )

    @property
    @override
    def id(self) -> str:
        return f"openai/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> OpenAIEstimatingTokenizer:
        return self._tokenizer

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
        with self.logger.scope(f"OpenAI LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    def _list_arguments(self, hints: Mapping[str, Any]) -> Mapping[str, Any]:
        exclude_params = [
            k
            for k in self.supported_openai_params
            for prefix, excluded in self.unsupported_params_by_model.items()
            if self.model_name.startswith(prefix) and k in excluded
        ]

        return {
            k: v
            for k, v in hints.items()
            if k in self.supported_openai_params and k not in exclude_params
        }

    @staticmethod
    def _supports_reasoning_effort(model: str) -> bool:
        # reasoning_effort applies to the reasoning families (gpt-5*, o-series);
        # standard chat models (gpt-4o, gpt-4.1, ...) reject it.
        return model.startswith("gpt-5") or bool(re.match(r"o\d", model))

    @staticmethod
    def _supports_minimal_effort(model: str) -> bool:
        # The original gpt-5 family exposed reasoning_effort="minimal"; gpt-5.1+ dropped it
        # in favor of "none" (the "as little reasoning as possible" tier).
        return not re.search(r"gpt-5\.\d", model)

    def _reasoning_arguments(self, effort: ReasoningEffort) -> dict[str, Any]:
        """Map the normalized reasoning effort to OpenAI's ``reasoning_effort``
        param, or an empty dict for models without reasoning support."""
        if not self._supports_reasoning_effort(self.model_name):
            return {}
        value: str = effort
        if effort == "minimal" and not self._supports_minimal_effort(self.model_name):
            value = "none"
        return {"reasoning_effort": value}

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        # Route through the Responses API (not Chat Completions): the React generator
        # uses Responses and its prompt cache engages, whereas Chat Completions wasn't
        # caching for the schematic fan-out.
        list_arguments = self._list_arguments(hints)

        request: dict[str, Any] = {
            "model": self.model_name,
            "input": [{"role": "developer", "content": prompt}],
            "store": False,
        }
        if "temperature" in list_arguments:
            request["temperature"] = list_arguments["temperature"]
        if "max_tokens" in list_arguments:
            request["max_output_tokens"] = list_arguments["max_tokens"]

        effort = hints.get(REASONING_EFFORT_HINT)
        if effort is not None:
            reasoning_arguments = self._reasoning_arguments(effort)
            if reasoning_arguments:
                request["reasoning"] = {"effort": reasoning_arguments["reasoning_effort"]}

        # `prompt_cache_key` keeps same-prefix requests (a fan-out and its prefill) on the
        # same cache node so they reuse it. OpenAI ignores the breakpoint — it has no
        # explicit cache resource, it auto-caches the longest common prefix.
        cache_key = (hints.get("cache") or {}).get("key")
        if cache_key is not None:
            request["prompt_cache_key"] = cache_key

        t_start = time.time()
        try:
            if hints.get("strict", False):
                parsed = await self._client.responses.parse(text_format=self.schema, **request)
                content = parsed.output_parsed
                assert content is not None
                usage = parsed.usage
            else:
                response = await self._client.responses.create(
                    text={"format": {"type": "json_object"}}, **request
                )
                raw_content = response.output_text or "{}"
                try:
                    json_content = json.loads(normalize_json_output(raw_content))
                except json.JSONDecodeError:
                    self.logger.warning(
                        f"Invalid JSON returned by {self.model_name}:\n{raw_content})"
                    )
                    json_content = jsonfinder.only_json(raw_content)[2]
                    self.logger.warning("Found JSON content within model response; continuing...")
                content = self.schema.model_validate(json_content)
                usage = response.usage

            t_end = time.time()

            input_tokens = (usage.input_tokens if usage else 0) or 0
            output_tokens = (usage.output_tokens if usage else 0) or 0
            input_details = usage.input_tokens_details if usage else None
            cached_input_tokens = (input_details.cached_tokens if input_details else 0) or 0
            # Billed at a premium on GPT-5.6+; absent (and free) on older models.
            cache_writes = (
                (getattr(input_details, "cache_write_tokens", 0) or 0) if input_details else 0
            )

            await record_llm_metrics(
                self.meter,
                self.model_name,
                schema_name=self.schema.__name__,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
            )

            return SchematicGenerationResult[T](
                content=content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cached_input_tokens=cached_input_tokens,
                        extra={"cache_write_input_tokens": cache_writes} if cache_writes else {},
                    ),
                ),
            )
        except RateLimitError:
            self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise
        except ValidationError as e:
            self.logger.error(
                f"Error: {e.json(indent=2)}\n"
                f"JSON returned by {self.model_name} does not match the expected schema."
            )
            raise


class GPT_4o(OpenAISchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gpt-4o-2024-11-20",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class GPT_4o_24_08_06(OpenAISchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gpt-4o-2024-08-06",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class GPT_4_1(OpenAISchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gpt-4.1",
            logger=logger,
            tracer=tracer,
            meter=meter,
            tokenizer_model_name="gpt-4o-2024-11-20",
            usage_reporter=usage_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class GPT_4o_Mini(OpenAISchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gpt-4o-mini",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )
        self._token_estimator = OpenAIEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class GPT_4_1_Mini(OpenAISchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gpt-4.1-mini",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )
        self._token_estimator = OpenAIEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class GPT_4_1_Nano(OpenAISchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gpt-4.1-nano",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )
        self._token_estimator = OpenAIEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class GPT_5_1(OpenAISchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gpt-5.1",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )
        self._token_estimator = OpenAIEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def max_tokens(self) -> int:
        return 400_000


class GPT_5_Mini(OpenAISchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gpt-5-mini",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )
        self._token_estimator = OpenAIEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def max_tokens(self) -> int:
        return 400_000


class GPT_5_Nano(OpenAISchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gpt-5-nano",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )
        self._token_estimator = OpenAIEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def max_tokens(self) -> int:
        return 400_000


class GPT_5_4_Nano(OpenAISchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gpt-5.4-nano",
            logger=logger,
            tracer=tracer,
            meter=meter,
            # tiktoken doesn't know gpt-5.4-nano; use the gpt-5 tokenizer (same
            # family) for estimation.
            tokenizer_model_name="gpt-5",
            usage_reporter=usage_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 400_000


class GPT_5_4_Mini(OpenAISchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gpt-5.4-mini",
            logger=logger,
            tracer=tracer,
            meter=meter,
            # tiktoken doesn't know gpt-5.4-mini; use the gpt-5 tokenizer (same family).
            tokenizer_model_name="gpt-5",
            usage_reporter=usage_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 400_000


class GPT_5_4(OpenAISchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gpt-5.4",
            logger=logger,
            tracer=tracer,
            meter=meter,
            # tiktoken doesn't know gpt-5.4; use the gpt-5 tokenizer (same family).
            tokenizer_model_name="gpt-5",
            usage_reporter=usage_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 400_000


# ============================================================================
# Streaming Text Generators
# ============================================================================

# Pattern to detect word boundaries for chunking
# Matches after any whitespace character
_WORD_BOUNDARY_PATTERN = re.compile(r"(?<=\s)")

# Number of words to buffer before yielding a chunk
_WORDS_PER_CHUNK = 3


# ============================================================================
# Embedders
# ============================================================================


class OpenAIEmbedder(BaseEmbedder):
    supported_arguments = ["dimensions"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
        api_key: str | None = None,
        provider_name: str | None = None,
    ) -> None:
        super().__init__(
            logger,
            tracer,
            meter,
            model_name,
            usage_reporter,
            provider_name=provider_name or "openai",
        )

        self._client = with_client_identity(
            AsyncClient(
                default_headers=parlant_client_headers(),
                api_key=require_api_key("OpenAI", "OPENAI_API_KEY", api_key),
                timeout=_OPENAI_TIMEOUT,
                max_retries=_OPENAI_MAX_RETRIES,
            )
        )
        self._tokenizer = OpenAIEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"openai/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> OpenAIEstimatingTokenizer:
        return self._tokenizer

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
        except RateLimitError:
            self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise

        vectors = [data_point.embedding for data_point in response.data]
        return EmbeddingResult(
            vectors=vectors,
            usage=UsageInfo(
                input_tokens=(response.usage.prompt_tokens if response.usage else 0) or 0,
                output_tokens=0,
            ),
        )


class OpenAITextEmbedding3Large(OpenAIEmbedder):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="text-embedding-3-large",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        return 3072


class OpenAITextEmbedding3Small(OpenAIEmbedder):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="text-embedding-3-small",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        return 1536


# ── The Decisions API ─────────────────────────────────────────────────────────
#
# One request carries a state plus many named questions and comes back with one
# typed answer each, which is the whole reason to reach for it — N judgments cost
# one round trip instead of N.
#
# The endpoint is stricter than the DecisionMaker port, in three ways that this
# adapter absorbs so callers never have to:
#
#   - `instructions` must be a string, while a Criterion is any JSON value.
#   - A predicate takes no criteria at all; sending them is a 400, so a
#     proposition's true/false meanings are folded into its instructions.
#   - `input` is a string or a message array, never a bare JSON object.
#
# Keeping that here is deliberate: the ranker and every other caller go on
# speaking the structured port, and only the translation is OpenAI-specific.

DEFAULT_DECISION_MODEL = "gpt-6-luna"

# Undocumented for this endpoint, and nothing in the SDK's types states a limit.
# 128k is a conservative stand-in rather than a verified ceiling; pass
# `max_tokens` for a model whose real budget is known to differ.
DEFAULT_DECISION_MAX_TOKENS = 128_000


def _render_criterion(value: Criterion) -> str:
    """A Criterion as text, since every instruction field here is a string.

    A string passes through untouched so authored prose is never mangled into
    JSON; anything structured is rendered the same way prompts render it."""
    return value if isinstance(value, str) else to_prompt_json(value)


def _fold_proposition_criteria(question: Proposition) -> str:
    """A proposition's instructions with its criteria appended as text.

    The endpoint has nowhere to put them, and dropping them would quietly lose a
    signal the caller chose to send — the rule ranker, for one, derives them from
    a rule's signals and anti-signals."""
    instructions = _render_criterion(question.instructions)

    if question.criteria is None:
        return instructions

    return (
        f"{instructions}\n\n"
        f"Answer true when: {_render_criterion(question.criteria.true)}\n"
        f"Answer false when: {_render_criterion(question.criteria.false)}"
    )


def _encode_decision_question(name: str, question: Question) -> dict[str, Any]:
    match question:
        case Proposition():
            return {
                "type": "predicate",
                "name": name,
                "instructions": _fold_proposition_criteria(question),
            }
        case Choice():
            return {
                "type": "choice",
                "name": name,
                "instructions": _render_criterion(question.instructions),
                # An option with no description comes back as a refusal rather
                # than an error, so a None criterion falls back to the option's
                # own name instead of being omitted.
                "choices": [
                    {
                        "value": option,
                        "description": _render_criterion(criterion)
                        if criterion is not None
                        else option,
                    }
                    for option, criterion in question.criteria.items()
                ],
            }
        case Scale():
            return {
                "type": "score",
                "name": name,
                "instructions": _render_criterion(question.instructions),
                # The endpoint indexes levels from zero, so an answer to an
                # N-level scale lands in [0, N-1] and never reaches N.
                "levels": [
                    {"label": rendered, "description": rendered}
                    for rendered in (_render_criterion(c) for c in question.criteria)
                ],
            }

    raise DecisionError(f"Unsupported question type: {type(question).__name__}")


def _decode_decision_answer(name: str, answer: Any) -> Answer:
    match answer.type:
        case "predicate":
            return PropositionAnswer(probability=float(answer.probability))
        case "choice":
            return ChoiceAnswer(
                option=str(answer.choice),
                probabilities={str(p.value): float(p.probability) for p in answer.probabilities},
                confidence=float(answer.confidence),
            )
        case "score":
            return ScaleAnswer(
                score=float(answer.score),
                legend={int(p.value): str(p.label) for p in answer.probabilities},
                probabilities={int(p.value): float(p.probability) for p in answer.probabilities},
                confidence=float(answer.confidence),
            )
        case "refusal":
            # A refusal arrives with a 200, so it would otherwise pass for an
            # answer. The port promises one for every question.
            raise DecisionError(f"The model refused to answer question '{name}'")

    raise DecisionError(f"Question '{name}' came back with an unrecognized answer type")


def _decode_openai_decision_usage(usage: Any) -> UsageInfo:
    """Tokens only: this endpoint reports no cost, unlike OpenRouter's.

    So a decision's price comes from the configured pricing table rather than
    from the provider, and `gpt-6-luna` needs an entry there to be costed.
    ``requests`` stays at 1 however many questions the batch carried — that is
    what makes the batching saving visible in usage reports."""
    if usage is None:
        return UsageInfo(input_tokens=0, output_tokens=0, requests=1)

    input_details = getattr(usage, "input_tokens_details", None)

    return UsageInfo(
        input_tokens=int(usage.input_tokens or 0),
        output_tokens=int(usage.output_tokens or 0),
        cached_input_tokens=int(getattr(input_details, "cached_tokens", 0) or 0),
        requests=1,
    )


class OpenAIDecisionMaker(DecisionMaker):
    """Typed decisions from a model on OpenAI's Decisions endpoint."""

    def __init__(
        self,
        model_name: str = DEFAULT_DECISION_MODEL,
        logger: Logger | None = None,
        tracer: Tracer | None = None,
        meter: Meter | None = None,
        api_key: str | None = None,
        client: AsyncClient | None = None,
        max_tokens: int = DEFAULT_DECISION_MAX_TOKENS,
        usage_reporter: UsageReporter | None = None,
        provider_name: str | None = None,
    ) -> None:
        assert logger is not None
        assert tracer is not None

        self.model_name = model_name
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._max_tokens = max_tokens
        self._usage_reporter = usage_reporter
        self._provider_name = provider_name or "openai"

        self._client = client or with_client_identity(
            AsyncClient(
                default_headers=parlant_client_headers(),
                # Resolved eagerly so a missing key is reported when the component
                # is built, not on the first turn that needs a decision.
                api_key=require_api_key("OpenAI", "OPENAI_API_KEY", api_key),
                timeout=_OPENAI_TIMEOUT,
                max_retries=_OPENAI_MAX_RETRIES,
            )
        )

    @property
    @override
    def id(self) -> str:
        return f"openai/{self.model_name}"

    @property
    def provider_name(self) -> str:
        """The provider this decision maker reports usage under."""
        return self._provider_name

    @property
    @override
    def max_tokens(self) -> int:
        return self._max_tokens

    @override
    async def decide(
        self,
        state: JSONSerializable,
        questions: Mapping[str, Question],
        hints: Mapping[str, Any] = {},
    ) -> DecisionResult:
        with self._tracer.span("openai.decisions"):
            t_start = time.time()
            try:
                decision = await self._client.decisions.create(
                    model=self.model_name,
                    input=state if isinstance(state, str) else to_prompt_json(state),
                    questions=[
                        cast(Any, _encode_decision_question(name, question))
                        for name, question in questions.items()
                    ],
                )
            except OpenAIError as exc:
                raise DecisionError(f"Decisions request failed: {exc}") from exc
            t_end = time.time()

        by_name = {answer.name: answer for answer in decision.answers}

        answers: dict[str, Answer] = {}
        for name in questions:
            if name not in by_name:
                raise DecisionError(f"No answer came back for question '{name}'")
            answers[name] = _decode_decision_answer(name, by_name[name])

        info = GenerationInfo(
            schema_name="Decision",
            model=decision.model or self.model_name,
            duration=t_end - t_start,
            usage=_decode_openai_decision_usage(decision.usage),
        )

        # Report here, as every other port implementation does: nothing downstream
        # reports an evaluator's generation info, so a decision that stayed quiet
        # would be free as far as pricing, budgets and cost control can tell.
        if self._usage_reporter is not None:
            self._usage_reporter.report_usage(
                info.model,
                info.usage,
                provider_name=self.provider_name,
            )

        return DecisionResult(answers=answers, info=info)


class OpenAIModerationService(BaseModerationService):
    def __init__(
        self,
        model_name: str,
        logger: Logger,
        meter: Meter,
        api_key: str | None = None,
    ) -> None:
        super().__init__(logger, meter)

        self.model_name = model_name

        self._client = with_client_identity(
            AsyncClient(
                default_headers=parlant_client_headers(),
                api_key=require_api_key("OpenAI", "OPENAI_API_KEY", api_key),
                timeout=_OPENAI_TIMEOUT,
                max_retries=_OPENAI_MAX_RETRIES,
            )
        )

        self._hist_moderation_request_duration = meter.create_duration_histogram(
            name="moderation",
            description="Duration of moderation requests in milliseconds",
        )

    @override
    async def do_moderate(self, context: ModerationContext) -> ModerationCheck:
        def extract_tags(category: str) -> list[ModerationTag]:
            mapping: dict[str, list[ModerationTag]] = {
                "sexual": ["sexual"],
                "sexual_minors": ["sexual", "illicit"],
                "harassment": ["harassment"],
                "harassment_threatening": ["harassment", "illicit"],
                "hate": ["hate"],
                "hate_threatening": ["hate", "illicit"],
                "illicit": ["illicit"],
                "illicit_violent": ["illicit", "violence"],
                "self_harm": ["self-harm"],
                "self_harm_intent": ["self-harm", "violence"],
                "self_harm_instructions": ["self-harm", "illicit"],
                "violence": ["violence"],
                "violence_graphic": ["violence", "harassment"],
            }

            return mapping.get(category.replace("/", "_").replace("-", "_"), [])

        response = await self._client.moderations.create(
            input=context.message,
            model=self.model_name,
        )

        result = response.results[0]

        return ModerationCheck(
            flagged=result.flagged,
            tags=list(
                set(
                    chain.from_iterable(
                        extract_tags(category)
                        for category, detected in result.categories
                        if detected
                    )
                )
            ),
        )


class OmniModeration(OpenAIModerationService):
    def __init__(self, logger: Logger, meter: Meter) -> None:
        super().__init__(
            model_name="omni-moderation-latest",
            logger=logger,
            meter=meter,
        )


# The key under which an OpenAI Responses output item (reasoning / function_call)
# is preserved verbatim on a canonical Part's provider_data so it can be replayed
# in a later stateless (store=False) request.
OPENAI_ITEM_KEY = "openai_item"


class OpenAIReactGenerator(ReactGenerator):
    """A ReAct generator backed by the OpenAI Responses API (google-genai's peer).

    Implements the ``ReactGenerator`` provider seam against the streaming
    Responses API. Runs statelessly (``store=False``): the caller fully owns the
    history, and reasoning / function-call output items round-trip verbatim via
    each Part's ``provider_data`` (under :data:`OPENAI_ITEM_KEY`), with
    ``include=["reasoning.encrypted_content"]`` so reasoning replays correctly.

    Caching is automatic prefix caching: a :attr:`Message.cache_key` is passed as
    ``prompt_cache_key`` (a routing hint). There is no explicit cache resource to
    manage, so :class:`CacheConfig` ``ttl`` / ``provider_options`` are unused.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-5.4-nano",
        logger: Logger,
        cache: Optional[CacheConfig] = None,
        client: Optional[AsyncClient] = None,
        api_key: Optional[str] = None,
        service_tier: ServiceTier = "standard",
        provider_name: Optional[str] = None,
        usage_reporter: Optional[UsageReporter] = None,
    ) -> None:
        super().__init__(model=model, cache=cache, usage_reporter=usage_reporter)
        self._logger = ScopedLogger(logger, "OpenAIReactGenerator")
        self._default_service_tier = service_tier
        self._provider_name = provider_name or "openai"
        self._client = client or with_client_identity(
            AsyncClient(
                default_headers=parlant_client_headers(),
                api_key=require_api_key("OpenAI", "OPENAI_API_KEY", api_key),
                timeout=_OPENAI_TIMEOUT,
                max_retries=_OPENAI_MAX_RETRIES,
            )
        )

    # Mapping from canonical ModelSize to a concrete OpenAI model id, used to
    # resolve per-call ``hints`` overrides on ``_encode``.
    _MODEL_BY_SIZE: dict[ModelSize, str] = {
        ModelSize.SMALL: "gpt-6-luna",
        ModelSize.MEDIUM: "gpt-5.6-terra",
        ModelSize.LARGE: "gpt-6.1-terra",
    }

    # Cache minimum assumed for models we don't recognize — large enough that
    # prefill is skipped rather than warming a cache that may never engage.
    _UNKNOWN_MIN_CACHE_SIZE = 1 << 20

    @property
    def id(self) -> str:
        return f"openai/{self.model}"

    @property
    @override
    def provider_name(self) -> str:
        return self._provider_name

    def _resolve_model(self, hints: ReactGeneratorHints) -> str:
        """Return the model id for this call, applying ``hints['model_size']``
        if present (falling back to the generator's default)."""
        size = hints.get("model_size", ModelSize.AUTO)
        return self._MODEL_BY_SIZE.get(size, self.model)

    # OpenAI's service_tier accepts "auto"/"default"/"flex"/"fast";
    # "standard" maps to "default", and the seam's "priority" maps to "fast"
    # (OpenAI's new name for its priority tier).
    _SERVICE_TIER: dict[ServiceTier, str] = {
        "standard": "default",
        "flex": "flex",
        "priority": "fast",
    }

    # ---- provider seam -----------------------------------------------------

    @override
    def _encode(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec],
        tool_choice: ToolChoice,
        *,
        reasoning: ReasoningConfig,
        hints: ReactGeneratorHints = {},
    ) -> dict[str, Any]:
        instruction_chunks: list[str] = []
        input_items: list[dict[str, Any]] = []
        cache_key: Optional[str] = None

        seen_non_system = False
        # Call ids issued and not yet answered. A result that answers none of
        # them is sent as a synthetic call plus its output, since the API
        # rejects a function_call_output without its function_call.
        pending_calls: set[str] = set()
        synthetic_ids = itertools.count(1)
        for message in history:
            # OpenAI accepts a single prompt_cache_key (a routing hint). Use the
            # FIRST marked message's key — including a marked system message —
            # since the earliest breakpoint is the most stable prefix boundary
            # and does not shift as the conversation grows.
            if self.cache.enabled and cache_key is None and message.cache_key is not None:
                cache_key = message.cache_key
            if message.role == Role.SYSTEM:
                if not seen_non_system:
                    # Leading system prompt → top-level instructions.
                    if message.text:
                        instruction_chunks.append(message.text)
                elif message.text:
                    # Mid-conversation system message → inline "developer" turn
                    # (OpenAI's role for operator instructions in the input).
                    input_items.append({"role": "developer", "content": message.text})
                continue
            seen_non_system = True
            if message.role == Role.TOOL:
                input_items.extend(self._encode_tool_results(message, pending_calls, synthetic_ids))
                continue
            items = self._encode_message(message)
            pending_calls.update(
                item["call_id"] for item in items if item.get("type") == "function_call"
            )
            input_items.extend(items)

        if not input_items:
            # The Responses API rejects an empty input array; an agent-first
            # greeting over an untouched conversation has only system content.
            input_items.append({"role": "user", "content": EMPTY_CONVERSATION_USER_TURN})

        request: dict[str, Any] = {
            "model": self._resolve_model(hints),
            "input": input_items,
            "instructions": "\n\n".join(chunk for chunk in instruction_chunks if chunk) or None,
        }

        if tools:
            request["tools"] = [self._encode_tool(spec) for spec in tools]
            request["tool_choice"] = self._encode_tool_choice(tool_choice)

        # Always emit the reasoning block — the depth is controlled by ``effort``,
        # mapped to what the resolved model supports. ``include`` is needed so
        # reasoning items round-trip verbatim across stateless calls.
        request["reasoning"] = self._encode_reasoning(reasoning, request["model"])
        request["include"] = ["reasoning.encrypted_content"]

        if cache_key is not None:
            # OpenAI's prefix caching is automatic; the key is only a routing hint.
            request["prompt_cache_key"] = cache_key

        request["service_tier"] = self._SERVICE_TIER[
            hints.get("service_tier", self._default_service_tier)
        ]

        return request

    def _encode_message(self, message: Message) -> list[dict[str, Any]]:
        if message.role == Role.TOOL:
            return [
                self._encode_tool_result(part)
                for part in message.parts
                if isinstance(part, ToolResultPart)
            ]

        if message.role == Role.USER:
            content = [
                {"type": "input_text", "text": part.text}
                for part in message.parts
                if isinstance(part, TextPart)
            ]
            return [{"role": "user", "content": content}] if content else []

        # ASSISTANT: emit items in part order, replaying raw items verbatim.
        items: list[dict[str, Any]] = []
        for part in message.parts:
            raw_item = part.provider_data.get(OPENAI_ITEM_KEY)
            if raw_item is not None:
                items.append(dict(raw_item))
                continue
            if isinstance(part, TextPart) and part.text:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": part.text}],
                    }
                )
            elif isinstance(part, ToolCallPart):
                items.append(
                    {
                        "type": "function_call",
                        "call_id": part.id,
                        "name": part.name,
                        "arguments": to_prompt_json(part.args),
                    }
                )
            # A ReasoningPart without its raw item cannot be reconstructed
            # (encrypted_content is opaque); dropping a reasoning part is safe.
        return items

    def _encode_tool_results(
        self,
        message: Message,
        pending_calls: set[str],
        synthetic_ids: Iterator[int],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for part in message.tool_results:
            if part.call_id in pending_calls:
                pending_calls.remove(part.call_id)
                items.append(self._encode_tool_result(part))
                continue
            call_id = f"call_context_{next(synthetic_ids)}"
            items.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": RESULT_ONLY_TOOL_NAME,
                    "arguments": "{}",
                }
            )
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": to_prompt_json(result_only_tool_output(part)),
                }
            )
        return items

    def _encode_tool_result(self, part: ToolResultPart) -> dict[str, Any]:
        output = part.content if isinstance(part.content, str) else to_prompt_json(part.content)
        return {"type": "function_call_output", "call_id": part.call_id, "output": output}

    def _encode_tool(self, spec: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "name": spec.name,
            "description": spec.description,
            "parameters": self._to_openai_schema(spec.json_schema()),
        }

    def _to_openai_schema(self, schema: Mapping[str, Any]) -> dict[str, Any]:
        """Translate ToolSpec.json_schema()'s OpenAPI-flavored output into the
        JSON Schema dialect the Responses API honors: nullability is a ``"null"``
        member of ``type``, not an OpenAPI ``nullable`` flag (which OpenAI
        silently ignores). Applied recursively to properties and array items."""
        result = {key: value for key, value in schema.items() if key != "nullable"}

        if schema.get("nullable"):
            current = result.get("type")
            if isinstance(current, str):
                result["type"] = [current, "null"]
            elif isinstance(current, list) and "null" not in current:
                result["type"] = [*current, "null"]

        if isinstance(result.get("properties"), dict):
            result["properties"] = {
                name: self._to_openai_schema(sub) for name, sub in result["properties"].items()
            }
        if isinstance(result.get("items"), dict):
            result["items"] = self._to_openai_schema(result["items"])

        return result

    def _encode_tool_choice(self, tool_choice: ToolChoice) -> Any:
        if isinstance(tool_choice, Mapping):
            return {"type": "function", "name": tool_choice.get("name")}
        return tool_choice  # "auto" | "none" | "required"

    @staticmethod
    def _supports_minimal_effort(model: str) -> bool:
        # The original gpt-5 family exposed reasoning_effort="minimal"; gpt-5.1+ dropped it
        # in favor of "none" (the "as little reasoning as possible" tier).
        return not re.search(r"gpt-5\.\d", model)

    def _encode_reasoning(self, reasoning: ReasoningConfig, model: str) -> dict[str, Any]:
        # OpenAI's Responses API controls reasoning depth via ``effort`` tiers.
        # Our canonical "as little as possible" level is ``"minimal"``; newer
        # models call that tier ``"none"`` instead, so remap where unsupported.
        effort: str = reasoning.effort
        if effort == "minimal" and not self._supports_minimal_effort(model):
            effort = "none"

        config: dict[str, Any] = {"effort": effort}
        summary = {"none": None, "summary": "auto", "full": "detailed"}[reasoning.visibility]
        if summary is not None:
            config["summary"] = summary
        return config

    @override
    async def _raw_stream(self, request: Any) -> AsyncIterator[Any]:
        kwargs = {key: value for key, value in request.items() if value is not None}
        try:
            stream = await self._client.responses.create(stream=True, store=False, **kwargs)
            # `async with` guarantees the stream (and its HTTP response) is closed
            # on cancellation or any early exit, rather than leaked.
            async with stream:
                async for event in stream:
                    yield event
        except RateLimitError as exc:
            self._logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise ReactError(str(exc), retryable=True) from exc
        except (
            APIConnectionError,
            APITimeoutError,
            ConflictError,
            APIResponseValidationError,
            InternalServerError,
        ) as exc:
            # Transient — mirror the schematic generator's retry set. The consumer
            # (BaseLoop) retries these before any event of the step is emitted.
            raise ReactError(str(exc), retryable=True) from exc

    def _build_prefill_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Turn an encoded request into a cache-warming request: same input
        prefix / instructions / tools / prompt_cache_key, plus a tiny dummy user
        item and a 16-token output cap. Reasoning is dropped so the small cap is
        valid; it doesn't affect the cached input prefix."""
        input_items = list(request["input"])
        if not input_items or input_items[-1].get("role") != "user":
            input_items = input_items + [{"role": "user", "content": "."}]

        prefill = {
            key: value
            for key, value in request.items()
            if value is not None and key not in ("reasoning", "include")
        }
        prefill["input"] = input_items
        prefill["max_output_tokens"] = 16
        return prefill

    def _min_cache_size(self, model: str) -> int:
        """Minimum prompt size (in tokens) at which OpenAI automatic prompt
        caching engages. It is 1024 tokens across current models; unknown models
        are assumed not to cache cheaply."""
        if model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
            return 1024
        return self._UNKNOWN_MIN_CACHE_SIZE

    @override
    async def _should_prefill(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec],
        hints: ReactGeneratorHints,
        reasoning: Optional[ReasoningConfig] = None,
    ) -> bool:
        token_count = self._estimate_prefill_tokens(history, tools)
        return token_count >= self._min_cache_size(self._resolve_model(hints))

    async def _prefill(self, request: Any) -> Usage:
        prefill_request = self._build_prefill_request(request)
        response = await self._client.responses.create(stream=False, store=False, **prefill_request)
        return self._decode_usage(response.usage, getattr(response, "model", "") or "")

    @override
    def _decode(self, raw_event: Any, builder: TurnBuilder) -> list[StreamEvent]:
        event_type = raw_event.type

        if event_type == "response.output_text.delta":
            builder.text_delta(raw_event.delta)
            return [TextDelta(text=raw_event.delta)]

        if event_type == "response.reasoning_summary_text.delta":
            builder.reasoning_delta(raw_event.delta)
            return [ReasoningDelta(text=raw_event.delta)]

        if event_type == "response.output_item.added":
            item = raw_event.item
            if item.type == "function_call":
                builder.tool_call(item.call_id, name=item.name)
                return [ToolCallStarted(id=item.call_id, name=item.name)]
            return []

        if event_type == "response.output_item.done":
            item = raw_event.item
            if item.type == "function_call":
                builder.tool_call(
                    item.call_id,
                    name=item.name,
                    args=json.loads(item.arguments or "{}"),
                    provider_data={OPENAI_ITEM_KEY: item.model_dump(exclude_none=True)},
                )
            elif item.type == "reasoning":
                # Attach the raw reasoning item (with encrypted_content) for replay,
                # even when no visible summary text was streamed.
                builder.reasoning_delta(
                    "", provider_data={OPENAI_ITEM_KEY: item.model_dump(exclude_none=True)}
                )
            return []

        if event_type == "response.completed":
            response = raw_event.response
            builder.usage = self._decode_usage(response.usage, getattr(response, "model", "") or "")
            builder.finish_reason = self._map_finish_reason(response)
            return []

        return []

    def _decode_usage(self, usage: Any, model_name: str) -> Usage:
        if usage is None:
            return Usage(model_name=model_name)
        output_details = usage.output_tokens_details
        input_details = usage.input_tokens_details
        # GPT-5.6+ models bill cache writes (1.25x) and report them in
        # input_tokens_details; older models neither bill nor report — the
        # defensive read yields no meter there, which prices to zero.
        cache_writes = (
            (getattr(input_details, "cache_write_tokens", 0) or 0) if input_details else 0
        )
        return Usage(
            input_tokens=usage.input_tokens or 0,
            # OpenAI's output_tokens already includes reasoning tokens.
            output_tokens=usage.output_tokens or 0,
            cached_input_tokens=(input_details.cached_tokens if input_details else 0) or 0,
            reasoning_tokens=(output_details.reasoning_tokens if output_details else 0) or 0,
            extra={"cache_write_input_tokens": cache_writes} if cache_writes else {},
            model_name=model_name,
        )

    def _map_finish_reason(self, response: Any) -> FinishReason:
        if response.status == "completed":
            return FinishReason.STOP
        if response.status == "failed":
            return FinishReason.ERROR
        if response.status == "incomplete":
            reason = response.incomplete_details.reason if response.incomplete_details else None
            if reason == "max_output_tokens":
                return FinishReason.MAX_TOKENS
            if reason == "content_filter":
                return FinishReason.CONTENT_FILTER
        return FinishReason.STOP


class RoutedOpenAISchematicGenerator(OpenAISchematicGenerator[T]):
    @property
    def max_tokens(self) -> int:
        return 128 * 1024


class RoutedOpenAIEmbedder(OpenAIEmbedder):
    def __init__(
        self,
        *,
        dimensions: int,
        max_tokens: int,
        **kwargs: Any,
    ) -> None:
        self._dimensions = dimensions
        self._max_tokens = max_tokens
        super().__init__(**kwargs)

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def dimensions(self) -> int:
        return self._dimensions


def register_openai_models(
    registry: ModelImplementationRegistry,
    container: Container,
) -> None:
    logger = container[Logger]
    tracer = container[Tracer]
    meter = container[Meter]
    usage = container[UsageReporter]

    async def schematic(
        model: str,
        schema: type[Any],
        raw_args: Mapping[str, Any],
    ) -> OpenAISchematicGenerator[Any]:
        args = validate_model_arguments(
            "openai",
            raw_args,
            {"api_key", "tokenizer_model_name"},
        )
        generator_type = RoutedOpenAISchematicGenerator[schema]  # type: ignore[valid-type]
        return cast(
            OpenAISchematicGenerator[Any],
            decorate_schematic_generator(
                generator_type(
                    model_name=model,
                    logger=logger,
                    tracer=tracer,
                    meter=meter,
                    usage_reporter=usage,
                    **args,
                ),
                tracer,
            ),
        )

    async def react(model: str, raw_args: Mapping[str, Any]) -> OpenAIReactGenerator:
        args = validate_model_arguments("openai", raw_args, {"api_key", "service_tier"})
        if "service_tier" in args:
            args["service_tier"] = validate_service_tier("openai", args["service_tier"])
        return OpenAIReactGenerator(model=model, logger=logger, usage_reporter=usage, **args)

    async def embedding(
        model: str,
        raw_args: Mapping[str, Any],
    ) -> RoutedOpenAIEmbedder:
        args = validate_model_arguments(
            "openai",
            raw_args,
            {"api_key", "dimensions", "max_tokens"},
        )
        dimensions = int(
            args.pop(
                "dimensions",
                {
                    "text-embedding-3-large": 3072,
                    "text-embedding-3-small": 1536,
                    "text-embedding-ada-002": 1536,
                }.get(model, 1536),
            )
        )
        max_tokens = int(args.pop("max_tokens", 8192))
        return RoutedOpenAIEmbedder(
            model_name=model,
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage,
            dimensions=dimensions,
            max_tokens=max_tokens,
            **args,
        )

    async def moderation(
        model: str,
        raw_args: Mapping[str, Any],
    ) -> OpenAIModerationService:
        args = validate_model_arguments("openai", raw_args, {"api_key"})
        return OpenAIModerationService(
            model_name=model,
            logger=logger,
            meter=meter,
            **args,
        )

    async def decision(
        model: str,
        raw_args: Mapping[str, Any],
    ) -> OpenAIDecisionMaker:
        args = validate_model_arguments("openai", raw_args, {"api_key", "max_tokens"})
        return OpenAIDecisionMaker(
            model_name=model,
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage,
            **args,
        )

    async def tokenizer(
        model: str,
        raw_args: Mapping[str, Any],
    ) -> OpenAIEstimatingTokenizer:
        validate_model_arguments("openai", raw_args, set())
        return OpenAIEstimatingTokenizer(model)

    registry.register_schematic(
        "openai",
        schematic,
        default=_openai_schematic_default,
    )

    def _openai_model_config(
        model: str,
        *,
        args: Mapping[str, Any] | None = None,
        reasoning: ReasoningConfig | ReasoningEffort | None = None,
    ) -> ModelConfig:
        """Create a globally qualified config from an OpenAI-local model name."""
        return ModelConfig(
            f"openai:{model}",
            args=args or {},
            reasoning=reasoning,
        )

    def react_model_resolver(route: str, qualifiers: Mapping[str, str]) -> str | ModelConfig:
        if qualifiers.get("effort") in ("min",):
            return _openai_model_config(
                "gpt-6-luna",
                reasoning="minimal",
            )
        if qualifiers.get("effort") in ("low",):
            return _openai_model_config(
                "gpt-6-luna",
                reasoning="low",
            )
        elif qualifiers.get("effort") in ("max",):
            return _openai_model_config(
                "gpt-6.1-terra",
                reasoning="medium",
            )
        else:  # default
            return _openai_model_config(
                "gpt-6-luna",
                reasoning="medium",
            )

    registry.register(
        "openai",
        ModelCapability.REACT,
        react,
        default=react_model_resolver,
    )
    registry.register(
        "openai",
        ModelCapability.EMBEDDING,
        embedding,
        default="text-embedding-3-small",
    )
    # Intentionally NO default model: the "*" wildcard route resolves each capability via
    # its registered default, so registering a moderation default would silently enable
    # moderation whenever someone sets models={"*": "openai"}. Moderation must be opt-in —
    # route it explicitly, e.g. models={"moderation.customer": "openai:omni-moderation-latest"}.
    registry.register(
        "openai",
        ModelCapability.MODERATION,
        moderation,
    )
    # Registering a decision default is what opts OpenAI into decisions: a bare
    # `{"*": "openai"}` route then serves decisions (e.g. for rule ranking) with
    # this model, instead of the caller keeping to its own schematic path.
    registry.register(
        "openai",
        ModelCapability.DECISION,
        decision,
        default=DEFAULT_DECISION_MODEL,
    )
    registry.register(
        "openai",
        ModelCapability.TOKENIZER,
        tokenizer,
        default="gpt-5.4-nano",
    )


def _openai_schematic_default(route: str, qualifiers: Mapping[str, str]) -> str:
    if route.startswith("review") or route == "compaction":
        return "gpt-5.6-terra"
    return "gpt-5.6-luna"
