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

"""OpenRouter-backed NLP implementations.

OpenRouter model routes accept the common connection arguments ``api_key``,
``site_url``, and ``app_name``, plus a ``provider`` mapping that is forwarded
verbatim as OpenRouter's provider-routing preferences on schematic, ReAct,
streaming, and embedding requests. Configure it per logical model route so
interactive generation and background work can use different trade-offs::

    import parlant.sdk as p

    server = p.Server(
        models={
            "react": p.ModelConfig(
                "openrouter:anthropic/claude-sonnet-5.5",
                args={
                    "provider": {
                        "sort": "latency",
                        "preferred_max_latency": {"p90": 2.0},
                        "allow_fallbacks": True,
                        "require_parameters": True,
                    }
                },
            ),
            "matching.*": p.ModelConfig(
                "openrouter:qwen/qwen3.8-27b",
                args={
                    "provider": {
                        "sort": "throughput",
                        "preferred_min_throughput": {"p90": 50},
                    }
                },
                reasoning="low",
            ),
        }
    )

Use ``order`` or ``only`` to target backend providers, ``ignore`` to exclude
them, and ``allow_fallbacks=False`` when the restriction must be hard. OpenRouter
also supports ``sort`` values ``"price"``, ``"throughput"``, and ``"latency"``;
performance thresholds such as ``preferred_max_latency`` are preferences rather
than guarantees. The exact mapping is intentionally passed through so new
OpenRouter routing fields do not require an adapter release.

Set ``ModelConfig.reasoning`` alongside ``args`` to choose the default reasoning
effort for a route. Schematic requests translate that setting to OpenRouter's
top-level ``reasoning.effort`` field; ReAct requests use the same provider-neutral
configuration directly.

Schematic requests send the generator's Pydantic JSON Schema as a strict
structured-output format. If the selected backend rejects structured output,
the adapter falls back to JSON object mode and finally to an explicit JSON-only
instruction for compatibility. The successful mode is cached for the process
lifetime by model, provider-routing configuration, and output schema, avoiding
repeated negotiation on subsequent requests.

OpenRouter's response ``usage.cost`` is the authoritative amount charged in
USD. Every adapter path preserves it as ``UsageInfo.extra["reported_cost"]`` so
session usage reports can use the provider charge instead of maintaining a
local model price table. Explicit Parlant pricing supplies the fallback for
operations whose provider response does not contain a reported cost.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
import re
import time
from typing import Any, Literal, Mapping, Optional, Sequence, cast

import httpx
import httpx2
from parlant.core.container import Container
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AsyncClient,
    BadRequestError,
    ConflictError,
    InternalServerError,
    RateLimitError,
)
import jsonfinder  # type: ignore
from pydantic import ValidationError
import tiktoken
from typing_extensions import override

from parlant.adapters.nlp.common import (
    client_identity_enabled,
    parlant_client_headers,
    with_client_identity,
    decorate_schematic_generator,
    normalize_json_output,
    record_llm_metrics,
    require_api_key,
    validate_model_arguments,
)
from parlant.adapters.nlp.openai_service import OpenAIReactGenerator
from parlant.core.engines.compass.prompt_builder import PromptBuilder
from parlant.core.loggers import ScopedLogger, Logger
from parlant.core.meter import Meter
from parlant.core.common import JSONSerializable
from parlant.core.nlp.common import UsageInfo, format_reported_cost
from parlant.core.nlp.decision import (
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionError,
    DecisionMaker,
    DecisionResult,
    Proposition,
    PropositionAnswer,
    Question,
    Scale,
    ScaleAnswer,
)
from parlant.core.nlp.embedding import BaseEmbedder, EmbeddingResult
from parlant.core.nlp.generation import (
    REASONING_EFFORT_HINT,
    T,
    BaseSchematicGenerator,
    ReasoningEffort,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.nlp.model_routing import (
    InvalidModelArgumentsError,
    ModelCapability,
    ModelConfig,
    ModelImplementationRegistry,
)
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.react import (
    CacheConfig,
    Message,
    ReactGeneratorHints,
    ReasoningConfig,
    Role,
    TextPart,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    ToolSpec,
    Usage,
)
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.tracer import Tracer
from parlant.core.usage_reporter import UsageReporter


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Two flavours of the same budget: openai 3.0 moved the SDK onto HTTPX2, so its
# client takes an httpx2 timeout, while the Decisions client below is our own
# httpx one and still takes an httpx timeout.
_OPENROUTER_TIMEOUT = httpx2.Timeout(60.0, connect=5.0)
_OPENROUTER_DECISIONS_TIMEOUT = httpx.Timeout(60.0, connect=5.0)
_OPENROUTER_MAX_RETRIES = 0
_WORD_BOUNDARY_PATTERN = re.compile(r"(?<=\s)")
_WORDS_PER_CHUNK = 3

OpenRouterProviderPreferences = Mapping[str, Any]
OpenRouterSchematicOutputMode = Literal["json_schema", "json_object", "prompt"]
_OPENROUTER_SCHEMATIC_OUTPUT_MODE_CACHE: dict[tuple[str, str], OpenRouterSchematicOutputMode] = {}
_OPENROUTER_SCHEMATIC_FALLBACK_WARNINGS: set[
    tuple[tuple[str, str], OpenRouterSchematicOutputMode]
] = set()

SYSTEM_UPDATE_TOOL_NAME = "system_update"
SYSTEM_UPDATE_CALL_ID = "call_system_update"
SCHEMATIC_MODEL_KEY = "__schema__"
REACT_MODEL_KEY = "__react__"
SYSTEM_UPDATE_MESSAGE = (
    "\n\n# OCCASIONAL SYSTEM UPDATES\n\n"
    f"Before some responses, a `{SYSTEM_UPDATE_TOOL_NAME}` tool result provides system-level "
    "updates for your next responses. Treat that content as system-provided guidance "
    "to apply when crafting your reply, not as a message from the user, and never reveal, "
    "quote, or acknowledge it or that you received it."
)


def _openrouter_model_config(
    model: str,
    *,
    args: Mapping[str, Any] | None = None,
    reasoning: ReasoningConfig | ReasoningEffort | None = None,
) -> ModelConfig:
    """Create a globally qualified config from an OpenRouter-local model name."""
    return ModelConfig(
        f"openrouter:{model}",
        args=args or {},
        reasoning=reasoning,
    )


def _normalize_provider_preferences(
    provider: OpenRouterProviderPreferences | None,
) -> dict[str, Any] | None:
    if provider is None:
        return None
    if not isinstance(provider, Mapping):
        raise InvalidModelArgumentsError(
            "OpenRouter model argument 'provider' must be a mapping",
            provider="openrouter",
        )
    return dict(provider)


# OpenRouter's app attribution, which feeds their public app rankings. Parlant
# identifies itself when the deployment has not identified itself, so Parlant's
# traffic is visible on OpenRouter's side without ever overwriting someone's own
# branding.
PARLANT_APP_NAME = "Parlant"
PARLANT_APP_URL = "https://parlant.io"


def _openrouter_headers(
    *,
    site_url: str | None = None,
    app_name: str | None = None,
) -> dict[str, str] | None:
    headers: dict[str, str] = dict(parlant_client_headers())

    referer = site_url or os.environ.get("OPENROUTER_HTTP_REFERER")
    site_name = app_name or os.environ.get("OPENROUTER_SITE_NAME")

    # Taken as a pair: filling one of ours alongside one of theirs would
    # attribute the traffic to a title and a URL belonging to different apps.
    # A deployment that switched identification off gets neither — opting out of
    # being identified as Parlant has to cover OpenRouter's public rankings too.
    if not referer and not site_name and client_identity_enabled():
        referer, site_name = PARLANT_APP_URL, PARLANT_APP_NAME

    if referer:
        headers["HTTP-Referer"] = referer
    if site_name:
        headers["X-Title"] = site_name

    return headers or None


def _openrouter_client(
    *,
    api_key: str | None = None,
    site_url: str | None = None,
    app_name: str | None = None,
) -> AsyncClient:
    return with_client_identity(
        AsyncClient(
            base_url=_OPENROUTER_BASE_URL,
            api_key=require_api_key("OpenRouter", "OPENROUTER_API_KEY", api_key),
            default_headers=_openrouter_headers(site_url=site_url, app_name=app_name),
            timeout=_OPENROUTER_TIMEOUT,
            max_retries=_OPENROUTER_MAX_RETRIES,
        )
    )


def _cached_input_tokens(usage: Any) -> int:
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", 0) if details else 0
    if isinstance(cached_tokens, int) and cached_tokens:
        return cached_tokens

    legacy_cached_tokens = getattr(usage, "prompt_cache_hit_tokens", 0)
    if isinstance(legacy_cached_tokens, int):
        return legacy_cached_tokens
    return cached_tokens if isinstance(cached_tokens, int) else 0


def _openrouter_extra_body(
    provider: OpenRouterProviderPreferences | None,
    *,
    reasoning_effort: ReasoningEffort | None = None,
) -> dict[str, object] | None:
    """Build the OpenAI-SDK escape hatch for OpenRouter-specific request fields."""
    extra: dict[str, object] = {}
    if provider:
        extra["provider"] = dict(provider)
    if reasoning_effort is not None:
        extra["reasoning"] = {"effort": reasoning_effort}
    return extra or None


def _to_openrouter_output_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Make Pydantic JSON Schema explicit enough for strict structured output."""

    def fix(node: Any) -> Any:
        if isinstance(node, dict):
            fixed = {key: fix(value) for key, value in node.items()}
            if fixed.get("type") == "object" and "additionalProperties" not in fixed:
                fixed["additionalProperties"] = False
            return fixed
        if isinstance(node, list):
            return [fix(item) for item in node]
        return node

    return cast(dict[str, Any], fix(dict(schema)))


def _is_json_response_format_error(error: BadRequestError) -> bool:
    text = f"{error} {getattr(error, 'body', None)}".lower()
    return any(
        marker in text
        for marker in (
            "json_schema",
            "json schema",
            "json_object",
            "json mode",
            "response_format",
            "structured output",
        )
    )


def _schematic_output_mode_cache_key(
    model_name: str,
    provider: OpenRouterProviderPreferences | None,
    schema: Mapping[str, Any],
) -> tuple[str, str]:
    configuration = json.dumps(
        {
            "provider": provider or {},
            "schema": schema,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return model_name, hashlib.sha256(configuration.encode()).hexdigest()


def _usage_extra(usage: Any) -> dict[str, int | float | str]:
    """Billable meters beyond the standard token fields.

    OpenRouter reports the authoritative USD cost it charged automatically. It
    is recorded as ``reported_cost`` and preferred over token-derived pricing.
    When the routed provider bills cache writes, their token count is retained
    as an additional billable meter."""
    extra: dict[str, int | float | str] = {}

    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        details = getattr(usage, "input_tokens_details", None)
    writes = getattr(details, "cache_write_tokens", None) if details else None
    if writes is None:
        writes = getattr(usage, "cache_write_tokens", None)
    if isinstance(writes, int) and writes > 0:
        extra["cache_write_input_tokens"] = writes

    cost = getattr(usage, "cost", None)
    if (reported_cost := format_reported_cost(cost)) is not None:
        extra["reported_cost"] = reported_cost

    return extra


def _prompt_tokens(usage: Any) -> int:
    value = getattr(usage, "prompt_tokens", 0)
    return value if isinstance(value, int) else 0


RATE_LIMIT_ERROR_MESSAGE = """\
OpenRouter API rate limit exceeded. Possible reasons:
1. Your account may have insufficient API credits.
2. You may be using a free-tier account with limited request capacity.
3. You might have exceeded the requests-per-minute limit for your account.

Recommended actions:
- Check your OpenRouter account balance and billing status.
- Review your API usage limits in OpenRouter's dashboard.
- For more details on rate limits and usage tiers, visit:
    https://openrouter.ai/docs/api-reference/limits
"""


class OpenRouterEmptyEmbeddingResponseError(Exception):
    """Raised when OpenRouter returns an embedding response with no vectors."""


class OpenRouterModelConfigurationError(ValueError):
    """Raised when no model was configured for an OpenRouter generation path."""


def _encoding_for(model_name: str) -> tiktoken.Encoding:
    """The model's own tiktoken encoding, or a modern default.

    OpenRouter names carry a provider prefix ("openai/gpt-4o"), and most routes
    are models tiktoken has never heard of. These counts are estimates, so an
    approximate encoding beats refusing to build the tokenizer at all.
    """
    for candidate in (model_name, model_name.rsplit("/", 1)[-1]):
        try:
            return tiktoken.encoding_for_model(candidate)
        except KeyError:
            continue

    return tiktoken.get_encoding("o200k_base")


class OpenRouterEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encoding = _encoding_for(model_name)

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class OpenRouterSchematicGenerator(BaseSchematicGenerator[T]):
    supported_openrouter_params = ["temperature", "max_tokens"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
        api_key: str | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
        provider: OpenRouterProviderPreferences | None = None,
        provider_name: str | None = None,
    ) -> None:
        super().__init__(
            logger=logger,
            tracer=tracer,
            meter=meter,
            model_name=model_name,
            usage_reporter=usage_reporter,
            provider_name=provider_name or "openrouter",
        )
        self._logger = ScopedLogger(logger, "OpenRouterSchematicGenerator")
        self._client = _openrouter_client(
            api_key=api_key,
            site_url=site_url,
            app_name=app_name,
        )
        self._provider = _normalize_provider_preferences(provider)

        self._tokenizer = OpenRouterEstimatingTokenizer("gpt-5")

    @property
    @override
    def id(self) -> str:
        return f"openrouter/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> OpenRouterEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        # Default implementation - should be overridden by subclasses
        return 8192

    @policy(
        [
            retry(
                exceptions=(
                    APIConnectionError,
                    APITimeoutError,
                    ConflictError,
                    RateLimitError,
                    APIResponseValidationError,
                    OpenRouterEmptyEmbeddingResponseError,
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
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        openrouter_api_arguments = {
            k: v for k, v in hints.items() if k in self.supported_openrouter_params
        }
        reasoning_effort = cast(ReasoningEffort | None, hints.get(REASONING_EFFORT_HINT))
        extra_body = _openrouter_extra_body(
            self._provider,
            reasoning_effort=reasoning_effort,
        )

        t_start = time.time()

        output_schema = _to_openrouter_output_schema(self.schema.model_json_schema())
        structured_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": self.schema.__name__,
                "strict": True,
                "schema": output_schema,
            },
        }
        json_instruction = (
            "IMPORTANT: You must respond with ONLY valid JSON. No explanatory text before or "
            "after the JSON. The response must be a valid JSON object."
        )
        attempts: list[
            tuple[
                OpenRouterSchematicOutputMode,
                list[dict[str, str]],
                dict[str, Any] | None,
                str,
            ]
        ] = [
            (
                "json_schema",
                [{"role": "user", "content": prompt}],
                structured_response_format,
                "JSON Schema structured output",
            ),
            (
                "json_object",
                [{"role": "user", "content": prompt}],
                {"type": "json_object"},
                "JSON object mode",
            ),
            (
                "prompt",
                [
                    {"role": "system", "content": json_instruction},
                    {"role": "user", "content": prompt},
                ],
                None,
                "prompt-only JSON instruction",
            ),
        ]
        cache_key = _schematic_output_mode_cache_key(
            self.model_name,
            self._provider,
            output_schema,
        )
        cached_mode = _OPENROUTER_SCHEMATIC_OUTPUT_MODE_CACHE.get(cache_key, "json_schema")
        attempts = attempts[
            next(index for index, (mode, *_) in enumerate(attempts) if mode == cached_mode) :
        ]
        response = None

        try:
            for index, (mode, messages, response_format, description) in enumerate(attempts):
                request: dict[str, Any] = {
                    "messages": messages,
                    "model": self.model_name,
                    "extra_body": extra_body,
                    **openrouter_api_arguments,
                }
                if response_format is not None:
                    request["response_format"] = response_format

                try:
                    response = await self._client.chat.completions.create(**request)
                    _OPENROUTER_SCHEMATIC_OUTPUT_MODE_CACHE[cache_key] = mode
                    break
                except BadRequestError as error:
                    if response_format is None or not _is_json_response_format_error(error):
                        raise
                    next_description = attempts[index + 1][3]
                    warning_key = (cache_key, mode)
                    if warning_key not in _OPENROUTER_SCHEMATIC_FALLBACK_WARNINGS:
                        _OPENROUTER_SCHEMATIC_FALLBACK_WARNINGS.add(warning_key)
                        self._logger.warning(
                            f"Model '{self.model_name}' rejected {description}; "
                            f"retrying with {next_description}."
                        )

            if response is None:
                raise RuntimeError(
                    f"OpenRouter model '{self.model_name}' returned no schematic response"
                )
        except BadRequestError as error:
            self._logger.error(f"OpenRouter API BadRequest: {error}")
            raise
        except RateLimitError:
            self._logger.error(
                f"\nRate limit exceeded for model '{self.model_name}'.\n"
                f"{RATE_LIMIT_ERROR_MESSAGE}\n"
                f"Consider:\n"
                f"  - Using a different model\n"
                f"  - Waiting a moment before retrying\n"
                f"  - Adding your own API key for higher limits\n"
            )
            raise
        except Exception as e:
            self._logger.error(
                f"\nOpenRouter API error with model '{self.model_name}': {type(e).__name__}\n"
                f"{e}\n"
                f"Consider switching to a more compatible model.\n"
            )
            raise

        t_end = time.time()

        raw_content = response.choices[0].message.content or "{}"

        # Check if we got empty response
        if not raw_content.strip() or raw_content.strip() == "{}":
            self._logger.error(
                f"\nModel '{self.model_name}' returned empty or invalid JSON.\n"
                f"Response: {raw_content}\n"
                f"This model may not be compatible with structured output requirements.\n"
                f"Please switch to a model that supports JSON mode (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet').\n"
            )
            # Set empty JSON as fallback
            json_content = {}
        else:
            try:
                json_content = json.loads(normalize_json_output(raw_content))
                # Check if parsed JSON is empty
                if not json_content or json_content == {}:
                    self._logger.warning(
                        "Model returned empty JSON object. Attempting to find JSON in response..."
                    )
                    # Try to find JSON in the response
                    try:
                        json_content = jsonfinder.only_json(raw_content)[2]
                        if json_content and json_content != {}:
                            self._logger.info("Found valid JSON content within response.")
                    except Exception:
                        self._logger.error(
                            f"Could not extract valid JSON from response: {raw_content}"
                        )
            except json.JSONDecodeError:
                self._logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content}")
                try:
                    # Try to extract JSON using jsonfinder
                    json_content = jsonfinder.only_json(raw_content)[2]
                    self._logger.warning("Found JSON content within model response; continuing...")
                except Exception as finder_error:
                    self._logger.error(
                        f"\nCould not parse JSON from model response.\n"
                        f"Raw response: {raw_content}\n"
                        f"Error: {finder_error}\n"
                        f"Model '{self.model_name}' may not be compatible.\n"
                        f"Consider switching to a model that supports structured output.\n"
                    )
                    json_content = {}

        try:
            content = self.schema.model_validate(json_content)

            usage = response.usage
            input_tokens = _prompt_tokens(usage)
            output_tokens = getattr(usage, "completion_tokens", 0)
            if not isinstance(output_tokens, int):
                output_tokens = 0
            cached_input_tokens = _cached_input_tokens(usage)

            await record_llm_metrics(
                self.meter,
                self.model_name,
                schema_name=self.schema.__name__,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
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
                        cached_input_tokens=cached_input_tokens,
                        extra=_usage_extra(usage),
                    ),
                ),
            )
        except ValidationError as e:
            self._logger.error(
                f"\nJSON content returned by '{self.model_name}' does not match expected schema.\n"
                f"Schema: {self.schema.__name__}\n"
                f"Raw response: {raw_content}\n"
                f"Parsed JSON: {json.dumps(json_content, indent=2) if json_content else 'Empty'}\n"
                f"Validation errors: {str(e)}\n"
                f"This model may not be producing valid structured output.\n"
                f"Consider switching to a model that supports JSON mode.\n"
            )
            raise


class OpenRouterReactGenerator(OpenAIReactGenerator):
    """OpenRouter's OpenAI-compatible Responses API ReAct adapter.

    The concrete upstream model is intentionally opaque. Dynamic system messages
    therefore always use a synthetic tool call/result pair, rather than relying
    on any model family's support for inline system or developer messages.
    """

    def __init__(
        self,
        *,
        model: str,
        logger: Logger,
        cache: Optional[CacheConfig] = None,
        client: Optional[AsyncClient] = None,
        api_key: Optional[str] = None,
        site_url: str | None = None,
        app_name: str | None = None,
        provider: OpenRouterProviderPreferences | None = None,
        provider_name: str | None = None,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model=model,
            logger=logger,
            cache=cache,
            usage_reporter=usage_reporter,
            client=client
            or _openrouter_client(
                api_key=api_key,
                site_url=site_url,
                app_name=app_name,
            ),
            provider_name=provider_name or "openrouter",
        )
        self._provider = _normalize_provider_preferences(provider)

    @property
    @override
    def id(self) -> str:
        return f"openrouter/{self.model}"

    @override
    def _resolve_model(self, hints: ReactGeneratorHints) -> str:
        return self.model

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
        leading_system: list[Message] = []
        conversation: list[Message] = []
        dynamic_instructions: list[str] = []
        leading_system_captured = False

        for message in history:
            if message.role == Role.SYSTEM:
                if not leading_system_captured:
                    leading_system.append(message)
                    leading_system_captured = True
                elif message.text:
                    dynamic_instructions.append(message.text)
                continue

            conversation.append(message)

        compatible_history = [
            *leading_system,
            Message(role=Role.SYSTEM, parts=[TextPart(text=SYSTEM_UPDATE_MESSAGE)]),
            *conversation,
        ]

        if dynamic_instructions:
            instructions = "\n\n".join(dynamic_instructions)
            compatible_history.extend(
                [
                    Message(
                        role=Role.ASSISTANT,
                        parts=[
                            ToolCallPart(
                                id=SYSTEM_UPDATE_CALL_ID,
                                name=SYSTEM_UPDATE_TOOL_NAME,
                                args={},
                            )
                        ],
                    ),
                    Message(
                        role=Role.TOOL,
                        parts=[
                            ToolResultPart(
                                call_id=SYSTEM_UPDATE_CALL_ID,
                                name=SYSTEM_UPDATE_TOOL_NAME,
                                content=instructions,
                            )
                        ],
                    ),
                ]
            )

        request = super()._encode(
            compatible_history,
            tools,
            tool_choice,
            reasoning=reasoning,
            hints=hints,
        )
        request["extra_body"] = _openrouter_extra_body(self._provider)
        return request

    @override
    def _decode_usage(self, usage: Any, model_name: str) -> Usage:
        decoded = super()._decode_usage(usage, model_name)
        return replace(decoded, extra={**decoded.extra, **_usage_extra(usage)})


class OpenRouterEmbedder(BaseEmbedder):
    supported_arguments = ["dimensions"]

    # Known embedding model dimensions
    _KNOWN_DIMENSIONS: dict[str, int] = {
        "openai/text-embedding-3-large": 3072,
        "openai/text-embedding-3-small": 1536,
        "openai/text-embedding-ada-002": 1536,
        "qwen/qwen3-embedding-8b": 4096,
        "qwen/qwen-embedding-v2": 1536,
    }

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
        api_key: str | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
        provider: OpenRouterProviderPreferences | None = None,
        provider_name: str | None = None,
    ) -> None:
        super().__init__(
            logger,
            tracer,
            meter,
            model_name,
            usage_reporter,
            provider_name=provider_name or "openrouter",
        )
        self._client = _openrouter_client(
            api_key=api_key,
            site_url=site_url,
            app_name=app_name,
        )
        self._provider = _normalize_provider_preferences(provider)
        self._tokenizer = OpenRouterEstimatingTokenizer("gpt-5")
        # Cache dimensions after first API call if not known
        self._cached_dimensions: int | None = None

    @property
    @override
    def id(self) -> str:
        return f"openrouter/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> OpenRouterEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        # Default max tokens for embedding models
        return 8192

    @property
    @override
    def dimensions(self) -> int:
        # Check environment variable override first
        if "OPENROUTER_EMBEDDER_DIMENSIONS" in os.environ:
            return int(os.environ["OPENROUTER_EMBEDDER_DIMENSIONS"])

        # Return cached dimensions if available
        if self._cached_dimensions is not None:
            return self._cached_dimensions

        # Check known dimensions lookup
        for model_key, dims in self._KNOWN_DIMENSIONS.items():
            if model_key in self.model_name:
                return dims

        # Default fallback - most embedding models use 1536 or 3072
        # This will be updated after first API call
        return 1536

    @policy(
        [
            retry(
                exceptions=(
                    APIConnectionError,
                    APITimeoutError,
                    ConflictError,
                    RateLimitError,
                    APIResponseValidationError,
                    OpenRouterEmptyEmbeddingResponseError,
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
                extra_body=_openrouter_extra_body(self._provider),
                **filtered_hints,
            )
        except ValueError as exc:
            if "No embedding data received" in str(exc):
                raise OpenRouterEmptyEmbeddingResponseError(str(exc)) from exc
            raise
        except RateLimitError:
            self.logger.error(
                f"\nRate limit exceeded for embedder model '{self.model_name}'.\n"
                f"{RATE_LIMIT_ERROR_MESSAGE}\n"
                f"Consider:\n"
                f"  - Using a different embedder model\n"
                f"  - Waiting a moment before retrying\n"
                f"  - Adding your own API key for higher limits\n"
            )
            raise

        if not response.data:
            raise OpenRouterEmptyEmbeddingResponseError("No embedding data received")

        vectors = [data_point.embedding for data_point in response.data]

        # Cache dimensions from first response if not already cached and not in known list
        if self._cached_dimensions is None and vectors:
            actual_dims = len(vectors[0])
            # Only cache if different from default or if not found in known dimensions
            if actual_dims != 1536 or not any(
                key in self.model_name for key in self._KNOWN_DIMENSIONS
            ):
                self._cached_dimensions = actual_dims
                self.logger.trace(
                    f"Detected embedding dimensions for '{self.model_name}': {actual_dims}"
                )

        return EmbeddingResult(
            vectors=vectors,
            usage=UsageInfo(
                input_tokens=_prompt_tokens(getattr(response, "usage", None)),
                output_tokens=0,
                extra=_usage_extra(getattr(response, "usage", None)),
            ),
        )


def register_openrouter_models(
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
    ) -> OpenRouterSchematicGenerator[Any]:
        args = validate_model_arguments(
            "openrouter",
            raw_args,
            {"api_key", "site_url", "app_name", "provider"},
        )
        generator_type = OpenRouterSchematicGenerator[schema]  # type: ignore[valid-type]
        return cast(
            OpenRouterSchematicGenerator[Any],
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

    async def react(model: str, raw_args: Mapping[str, Any]) -> OpenRouterReactGenerator:
        args = validate_model_arguments(
            "openrouter",
            raw_args,
            {"api_key", "site_url", "app_name", "provider"},
        )
        return OpenRouterReactGenerator(model=model, logger=logger, usage_reporter=usage, **args)

    async def embedding(
        model: str,
        raw_args: Mapping[str, Any],
    ) -> OpenRouterEmbedder:
        args = validate_model_arguments(
            "openrouter",
            raw_args,
            {"api_key", "site_url", "app_name", "provider"},
        )
        return OpenRouterEmbedder(
            model_name=model,
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage,
            **args,
        )

    async def decision(
        model: str,
        raw_args: Mapping[str, Any],
    ) -> OpenRouterDecisionMaker:
        args = validate_model_arguments(
            "openrouter",
            raw_args,
            {"api_key", "site_url", "app_name", "max_tokens"},
        )
        return OpenRouterDecisionMaker(
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
    ) -> OpenRouterEstimatingTokenizer:
        validate_model_arguments("openrouter", raw_args, set())
        return OpenRouterEstimatingTokenizer("gpt-5")

    def schematic_model_resolver(
        route: str,
        qualifiers: Mapping[str, str],
    ) -> str | ModelConfig:
        """Resolve built-in models and their OpenRouter routing preferences."""
        if route.startswith("indexing"):
            return "openai/gpt-5.6-terra"
        elif route.startswith("review"):
            return _openrouter_model_config(
                "openai/gpt-5.6-terra",
                reasoning="medium",
            )
        elif route.startswith("matching.distill"):
            return _openrouter_model_config(
                "openai/gpt-oss-120b:nitro",
                args={"provider": {"sort": "latency"}},
            )
        else:  # default
            return _openrouter_model_config(
                "openai/gpt-oss-20b:nitro",
                args={"provider": {"sort": "latency"}},
            )

    def react_model_resolver(route: str, qualifiers: Mapping[str, str]) -> str | ModelConfig:
        if qualifiers.get("effort") in ("min",):
            return _openrouter_model_config(
                "anthropic/claude-haiku-5.5",
                reasoning="minimal",
            )
        if qualifiers.get("effort") in ("low",):
            return _openrouter_model_config(
                "anthropic/claude-haiku-5.5",
                reasoning="low",
            )
        elif qualifiers.get("effort") in ("medium",):
            return _openrouter_model_config(
                "anthropic/claude-haiku-5.5",
                reasoning="medium",
            )
        elif qualifiers.get("effort") in ("high",):
            return _openrouter_model_config(
                "openai/gpt-6.1-terra",
                reasoning="medium",
            )
        elif qualifiers.get("effort") in ("max",):
            return _openrouter_model_config(
                "anthropic/claude-sonnet-5.5",
                reasoning="medium",
            )
        else:  # default
            return "anthropic/claude-haiku-5.5"

    registry.register_schematic(
        "openrouter",
        schematic,
        default=schematic_model_resolver,
    )
    registry.register(
        "openrouter",
        ModelCapability.REACT,
        react,
        default=react_model_resolver,
    )
    registry.register(
        "openrouter",
        ModelCapability.EMBEDDING,
        embedding,
        default="openai/text-embedding-3-small",
    )
    registry.register(
        "openrouter",
        ModelCapability.TOKENIZER,
        tokenizer,
        default="openai/gpt-5",
    )
    # Registering a decision default is what opts OpenRouter into decisions: a bare
    # `{"*": "openrouter"}` route then serves decisions (e.g. for rule ranking) with
    # this model, while a provider without a decision default serves none.
    registry.register(
        "openrouter",
        ModelCapability.DECISION,
        decision,
        default=DEFAULT_DECISION_MODEL,
    )


# ── The Decisions API ─────────────────────────────────────────────────────────
#
# A decision model is not a chat model, so it does not ride the OpenAI client:
# OpenRouter serves these on its Decisions endpoint, which sits under /api/alpha
# rather than /api/v1. One request carries a state plus many named questions and
# comes back with one typed answer each, which is the whole reason to reach for
# it — N judgments cost one round trip instead of N.
#
# This adapter speaks the endpoint, not any one model on it. TypeSafe's Jev is
# the model available there today and the default below, but a second decision
# model served on the same endpoint needs only a different `model_name`.

_OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

DEFAULT_DECISION_MODEL = "~typesafe/jev-latest"

# Per-model, not per-endpoint: pass `max_tokens` for a model with a different
# budget. Jev accepts 64k per request, of which 32k may go to the state plus the
# longest question — and the state is what a caller is liable to overrun.
DEFAULT_DECISION_MAX_TOKENS = 32_768


class OpenRouterDecisionMaker(DecisionMaker):
    """Typed decisions from a model on OpenRouter's Decisions endpoint."""

    def __init__(
        self,
        model_name: str = DEFAULT_DECISION_MODEL,
        logger: Logger | None = None,
        tracer: Tracer | None = None,
        meter: Meter | None = None,
        api_key: str | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
        http: httpx.AsyncClient | None = None,
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
        self._provider_name = provider_name or "openrouter"
        # Resolved eagerly so a missing key is reported when the component is
        # built, not on the first turn that needs a decision.
        self._api_key = require_api_key("OpenRouter", "OPENROUTER_API_KEY", api_key)
        self._headers = _openrouter_headers(site_url=site_url, app_name=app_name) or {}
        self._http = http or httpx.AsyncClient(timeout=_OPENROUTER_DECISIONS_TIMEOUT)
        self._max_tokens = max_tokens
        self._usage_reporter = usage_reporter

    @property
    @override
    def id(self) -> str:
        return f"openrouter/{self.model_name}"

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
        payload: dict[str, Any] = {
            "model": self.model_name,
            "state": state,
            "questions": {name: _encode_question(q) for name, q in questions.items()},
        }

        with self._tracer.span("openrouter.decisions"):
            t_start = time.time()
            try:
                response = await self._http.post(
                    _OPENROUTER_DECISIONS_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        **self._headers,
                    },
                )
            except httpx.HTTPError as exc:
                raise DecisionError(f"Decisions request failed: {exc}") from exc
            t_end = time.time()

        if response.status_code != 200:
            raise DecisionError(
                f"The Decisions API returned {response.status_code}: {response.text[:500]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise DecisionError("The Decisions API returned a non-JSON response") from exc

        raw_answers = body.get("answers")
        if not isinstance(raw_answers, dict):
            raise DecisionError("The Decisions API response carried no answers")

        answers = {name: _decode_answer(name, raw_answers, questions[name]) for name in questions}

        info = GenerationInfo(
            schema_name="Decision",
            # The response names the exact dated build that served us, which a
            # floating alias in the request does not.
            model=str(body.get("model") or self.model_name),
            duration=t_end - t_start,
            usage=_decode_decision_usage(body.get("usage") or {}),
        )

        # Report here, as every other port implementation does: nothing downstream
        # reports an evaluator's generation info, so a decision that stays quiet
        # would be free as far as pricing, budgets and cost control can tell. The
        # reported cost is authoritative, so the dated model name needs no entry in
        # the server's pricing table.
        if self._usage_reporter is not None:
            self._usage_reporter.report_usage(
                info.model,
                info.usage,
                provider_name=self.provider_name,
            )

        return DecisionResult(
            answers=answers,
            info=info,
            metadata={"decision_id": str(body.get("id", ""))},
        )


def _encode_question(question: Question) -> dict[str, Any]:
    """Every primitive is instructions plus criteria; only the criteria's shape
    differs — an object of true/false for a proposition, an option-keyed map for a
    choice, an ordered array for a scale."""
    match question:
        case Proposition():
            # "noul" is the endpoint's name for the primitive, short for Bernoulli.
            encoded: dict[str, Any] = {"type": "noul", "instructions": question.instructions}
            if question.criteria is not None:
                encoded["criteria"] = {
                    "true": question.criteria.true,
                    "false": question.criteria.false,
                }
            return encoded
        case Choice():
            return {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": dict(question.criteria),
            }
        case Scale():
            return {
                "type": "score",
                "instructions": question.instructions,
                "criteria": list(question.criteria),
            }

    raise DecisionError(f"Unsupported question type: {type(question).__name__}")


def _decode_answer(
    name: str,
    raw_answers: Mapping[str, Any],
    question: Question,
) -> Answer:
    if name not in raw_answers:
        raise DecisionError(f"No answer came back for question '{name}'")

    answer = raw_answers[name]
    if not isinstance(answer, dict):
        raise DecisionError(f"The answer to '{name}' is not an object")

    match answer.get("type"):
        case "noul":
            return PropositionAnswer(probability=float(answer["noul"]))
        case "choice":
            return ChoiceAnswer(
                option=str(answer["choice"]),
                probabilities={
                    str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()
                },
                confidence=float(answer.get("confidence", 0.0)),
            )
        case "score":
            return ScaleAnswer(
                score=float(answer["score"]),
                legend={int(k): str(v) for k, v in (answer.get("legend") or {}).items()},
                probabilities={
                    int(k): float(v) for k, v in (answer.get("probabilities") or {}).items()
                },
                confidence=float(answer.get("confidence", 0.0)),
            )

    # The Decisions endpoint is in alpha and may grow primitives. A judgment the
    # caller is waiting on must never be silently dropped.
    raise DecisionError(
        f"Question '{name}' came back with an unrecognized answer type: {answer.get('type')!r}"
    )


def _decode_decision_usage(usage: Mapping[str, Any]) -> UsageInfo:
    """The endpoint reports tokens and the authoritative USD cost charged.

    Output tokens are free on Jev today, but they are still reported, and cost is
    preserved the same way every other OpenRouter path preserves it. ``requests``
    stays at 1 no matter how many questions the batch carried — that is what
    makes the saving visible in usage reports."""
    extra: dict[str, int | float | str] = {}
    if (reported_cost := format_reported_cost(usage.get("cost"))) is not None:
        extra["reported_cost"] = reported_cost

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    return UsageInfo(
        input_tokens=int(input_tokens) if isinstance(input_tokens, (int, float)) else 0,
        output_tokens=int(output_tokens) if isinstance(output_tokens, (int, float)) else 0,
        extra=extra,
        requests=1,
    )
