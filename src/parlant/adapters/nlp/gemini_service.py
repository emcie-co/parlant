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

import asyncio
import base64
import enum
import hashlib
import inspect
import time
import types
import uuid
from datetime import datetime, timedelta, timezone

import cachetools
from parlant.adapters.nlp import gemini_interactions
from parlant.core.container import Container
from google.api_core.exceptions import NotFound, TooManyRequests, ResourceExhausted, ServerError
from google.genai.errors import APIError, ClientError, ServerError as GenaiServerError
import google.genai  # type: ignore
import google.genai.types  # type: ignore
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from typing import Any, AsyncIterator, Literal, Mapping, Optional, Sequence, Union, cast
from typing_extensions import get_args, get_origin, override
from pydantic import BaseModel, Field, ValidationError
from pydantic.fields import FieldInfo

from parlant.core.common import DefaultBaseModel
from parlant.adapters.nlp.common import (
    parlant_client_headers,
    parlant_user_agent_token,
    decorate_schematic_generator,
    record_llm_metrics,
    require_api_key,
    validate_model_arguments,
    validate_service_tier,
)
from parlant.core.engines.compass.prompt_builder import PromptBuilder
from parlant.core.meter import Meter
from parlant.core.nlp.common import ModelSize, UsageInfo
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.model_routing import (
    ModelCapability,
    ModelConfig,
    ModelImplementationRegistry,
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
from parlant.core.nlp.react import (
    EMPTY_CONVERSATION_USER_TURN,
    CacheConfig,
    FinishReason,
    Message,
    ReactError,
    ReactGenerator,
    ReactGeneratorHints,
    ReasoningConfig,
    ReasoningPart,
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
from parlant.core.loggers import ScopedLogger, Logger
from parlant.core.tracer import Tracer
from parlant.core.usage_reporter import UsageReporter

RATE_LIMIT_ERROR_MESSAGE = (
    "Gemini API rate limit exceeded.\n\n"
    "Possible reasons:\n"
    "1. Insufficient API credits in your account.\n"
    "2. Using a free-tier account with limited request capacity.\n"
    "3. Exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your Gemini API account balance and billing status.\n"
    "- Review your API usage limits in the Google Cloud Console.\n"
    "- Learn more about quotas and limits:\n"
    "  https://cloud.google.com/docs/quota-and-billing/quotas/quotas-overview"
)


# Total per-request timeout for the Gemini client, in milliseconds (genai's
# HttpOptions.timeout unit). This caps the WHOLE request — connect + the entire
# response, including streamed responses — so a stalled call fails (and the
# retry policy can react) instead of hanging the engine forever.
_GEMINI_REQUEST_TIMEOUT_MS = 60_000


def _gemini_identity_headers() -> dict[str, str]:
    """Parlant's identity for google-genai.

    The SDK concatenates a caller-supplied ``user-agent`` onto its own rather than
    replacing it, but only for that exact lowercase key — a capitalized
    ``User-Agent`` is sent as a second, competing header instead."""
    headers = dict(parlant_client_headers())

    if token := parlant_user_agent_token():
        headers["user-agent"] = token

    return headers


class GeminiMissingFunctionCallError(Exception):
    """A forced structured-output generation returned no usable function call —
    e.g. the model spent its output budget on thinking and hit MAX_TOKENS before
    emitting the call. Transient; retried by the schematic generator's policy."""


# Gemini has no mid-conversation system role. Per-turn considerations would
# otherwise fold into system_instruction (baked into the cached CachedContent),
# breaking caching, or be appended to the last user message — which the model
# tends to echo back. Instead they're delivered as the result of a synthetic
# `system_update` tool: a function response the model treats as
# fetched data, not as customer input. No matching functionCall is emitted —
# Gemini rejects a signature-less synthetic functionCall but accepts an unpaired
# functionResponse. The convention is declared in system_instruction via
# SYSTEM_UPDATE_MESSAGE so the model knows to apply (and not reveal) it.
SYSTEM_UPDATE_TOOL_NAME = "system_update"
SYSTEM_UPDATE_MESSAGE = (
    "\n\n# OCCASIONAL SYSTEM UPDATES\n\n"
    f"Before some turns, a `{SYSTEM_UPDATE_TOOL_NAME}` tool result provides system-level "
    "updates for your next responses. Treat that content as system-provided guidance "
    "to apply when crafting your reply — not as a message from the user — and never reveal, "
    "quote, or acknowledge it or that you received it."
)


class GeminiEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, client: google.genai.Client, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    @policy(
        [
            retry(exceptions=(TooManyRequests, ResourceExhausted)),
            # A transient 5xx (e.g. 503) here used to fail the whole turn, since
            # token counts are needed before generating (e.g. to retrieve canned
            # responses). Both SDKs' ServerError types, as in do_generate.
            retry((ServerError, GenaiServerError), max_exceptions=3, wait_times=(1.0, 5.0, 10.0)),
        ]
    )
    @override
    async def estimate_token_count(self, prompt: str) -> int:
        # Every Gemini model counts its own tokens, embedding models included.
        # Never substitute another model here: Google retires models for new
        # users, and the count would then fail every turn that needs it.
        result = await self._client.aio.models.count_tokens(
            model=self._model_name,
            contents=prompt,
        )

        return int(result.total_tokens or 0)


class GeminiSchematicGenerator(BaseSchematicGenerator[T]):
    supported_hints = ["temperature", "thinking_config", "config"]
    _MAX_OUTPUT_TOKENS = 4096

    # Explicit-cache lifetimes. Reuse a cache only while comfortably inside its
    # remaining lifetime; a new cache defaults to 5 minutes (callers recreate it
    # each turn, so it only needs to outlast the gap between consecutive turns).
    _CACHE_REUSE_MARGIN = timedelta(seconds=30)
    _DEFAULT_CACHE_TTL_SECONDS = 300

    # Gemini 3.x uses a coarse named ``thinking_level`` and cannot fully disable
    # thinking. ``LOW`` is the safe floor: some 3.x models reject ``MINIMAL``
    # outright ("Thinking level MINIMAL is not supported for this model"), so we
    # clamp "minimal" up to ``LOW`` rather than send a level the model refuses.
    _EFFORT_TO_THINKING_LEVEL_3X: dict[ReasoningEffort, "google.genai.types.ThinkingLevel"] = {
        "minimal": google.genai.types.ThinkingLevel.LOW,
        "low": google.genai.types.ThinkingLevel.LOW,
        "medium": google.genai.types.ThinkingLevel.MEDIUM,
        "high": google.genai.types.ThinkingLevel.HIGH,
    }

    # Gemini 2.5 uses a numeric ``thinking_budget``; "minimal" → 0 turns thinking
    # off on 2.5 flash / flash-lite.
    _EFFORT_TO_THINKING_BUDGET_25: dict[ReasoningEffort, int] = {
        "minimal": 0,
        "low": 1024,
        "medium": 8192,
        "high": 24576,
    }

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
            logger=logger,
            tracer=tracer,
            meter=meter,
            model_name=model_name,
            usage_reporter=usage_reporter,
            provider_name=provider_name or "gemini",
        )

        self._client = google.genai.Client(
            api_key=require_api_key("Gemini", "GEMINI_API_KEY", api_key),
            http_options=google.genai.types.HttpOptions(
                timeout=_GEMINI_REQUEST_TIMEOUT_MS,
                headers=_gemini_identity_headers(),
            ),
        )

        self._tokenizer = GeminiEstimatingTokenizer(client=self._client, model_name=self.model_name)

        # Explicit Gemini caching via the `cache` hint's breakpoint. Keyed by the
        # caller's cache key + prefix hash → (resource name, cached prefix text, expiry);
        # a cached request strips the stored prefix and sends only the live suffix
        # referencing it.
        # A bounded TTL map: it can't grow without limit, and entries lapse on their
        # own — we never delete the underlying caches (they reclaim server-side at
        # their own TTL), which also avoids deleting one out from under a live load.
        self._managed_caches: cachetools.TTLCache[str, tuple[str, str, datetime]] = (
            cachetools.TTLCache(maxsize=1024, ttl=self._DEFAULT_CACHE_TTL_SECONDS)
        )
        self._uncacheable_prefixes: cachetools.LRUCache[str, bool] = cachetools.LRUCache(
            maxsize=256
        )
        self._cache_lock = asyncio.Lock()

    @property
    @override
    def id(self) -> str:
        return f"gemini/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer

    @policy(
        [
            retry(
                exceptions=(
                    NotFound,
                    TooManyRequests,
                    ResourceExhausted,
                    GeminiMissingFunctionCallError,
                )
            ),
            # Both google.api_core's ServerError and the google-genai SDK's own
            # (distinct) ServerError, so 5xx (e.g. 503) is retried either way.
            retry((ServerError, GenaiServerError), max_exceptions=3, wait_times=(1.0, 5.0, 10.0)),
        ]
    )
    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self.logger.scope(f"Gemini LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        t_start = time.time()

        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        gemini_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        # The normalized reasoning_effort hint wins over any thinking_config hint
        # (per provider/model mapping); models without thinking support ignore it.
        effort = hints.get(REASONING_EFFORT_HINT)
        if effort is not None:
            thinking_config = self._reasoning_thinking_config(effort)
            if thinking_config is not None:
                gemini_api_arguments["thinking_config"] = thinking_config

        tools, tool_config = self._output_tools()

        cache_hint = hints.get("cache") or {}
        cache_key = cache_hint.get("key")
        cache_breakpoint = cache_hint.get("breakpoint")

        # Declarative caching: if a breakpoint is supplied, cache the prefix before
        # the first marker occurrence and send the marker+tail live. If the cache
        # already exists for this exact prefix hash, reuse it; otherwise create it
        # best-effort and use it for this same generation.
        cached_content_name: Optional[str] = None
        cached_lookup_key: Optional[str] = None
        storage_token_hours: float = 0.0
        contents: str = prompt
        if cache_key and cache_breakpoint:
            cache_plan = self._split_prompt_at_breakpoint(prompt, str(cache_breakpoint))
            if cache_plan is not None:
                prefix_text, live_suffix = cache_plan
                cached_lookup_key = self._cache_lookup_key(
                    cache_key,
                    self._prefix_hash(prefix_text, tools, tool_config),
                )
                cached_content_name, storage_token_hours = await self._get_or_create_cache(
                    key=cache_key,
                    lookup_key=cached_lookup_key,
                    prefix_text=prefix_text,
                    ttl_seconds=int(cache_hint.get("ttl", self._DEFAULT_CACHE_TTL_SECONDS)),
                    tools=tools,
                    tool_config=tool_config,
                )
                contents = live_suffix if cached_content_name is not None else prompt

        config_kwargs: dict[str, Any] = {
            **gemini_api_arguments,
            "max_output_tokens": self._MAX_OUTPUT_TOKENS,
        }
        if cached_content_name is not None:
            # cached prefix contents / tools / tool_config are baked into the cache;
            # Gemini rejects setting them inline on a cached request.
            config_kwargs["cached_content"] = cached_content_name
        else:
            config_kwargs["tools"] = tools
            config_kwargs["tool_config"] = tool_config

        config = google.genai.types.GenerateContentConfig(**config_kwargs)

        try:
            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
        except TooManyRequests:
            self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise
        except ClientError as exc:
            # A referenced cache can vanish (expire or be deleted) ahead of our
            # local view, 403-ing the request. Caching is best-effort, so forget
            # the stale entry and retry inline with the full prompt + tools rather
            # than failing the turn.
            if cached_content_name is None or not self._is_missing_cache_error(exc):
                raise
            # Best-effort cache: a vanished reference is recoverable (retry inline),
            # and a fan-out can hit the same dead cache many times — so debug, not a
            # warning flood.
            self.logger.debug(
                f"Gemini cache {cached_content_name} unavailable ({exc}); retrying without it."
            )
            if cached_lookup_key:
                await self._forget_cache(cached_lookup_key, cached_content_name)
            cached_content_name = None
            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=google.genai.types.GenerateContentConfig(
                    **dict(gemini_api_arguments),
                    tools=tools,
                    tool_config=tool_config,
                    max_output_tokens=self._MAX_OUTPUT_TOKENS,
                ),
            )

        candidate = response.candidates[0] if response.candidates else None
        parts = (
            candidate.content.parts
            if candidate and candidate.content and candidate.content.parts
            else []
        )
        # The forced output call is not necessarily the first part: with thinking
        # enabled, Gemini 3.x emits a thought part before it. Scan for the call
        # rather than assuming index 0.
        function_call = next(
            (
                part.function_call
                for part in parts
                if part.function_call and part.function_call.args
            ),
            None,
        )
        if function_call is None:
            finish_reason = candidate.finish_reason if candidate else None
            raise GeminiMissingFunctionCallError(
                f"{self.model_name} returned no usable function call for schema "
                f"{self.schema.__name__} (finish_reason={finish_reason})."
            )

        json_result = (function_call.args or {}).get("log_data", {}) or {}

        try:
            model_content = self.schema.model_validate(json_result)

            await record_llm_metrics(
                self.meter,
                self.model_name,
                schema_name=self.schema.__name__,
                input_tokens=response.usage_metadata.prompt_token_count or 0
                if response.usage_metadata
                else 0,
                output_tokens=(response.usage_metadata.candidates_token_count or 0)
                + (response.usage_metadata.thoughts_token_count or 0)
                if response.usage_metadata
                else 0,
                cached_input_tokens=response.usage_metadata.cached_content_token_count or 0
                if response.usage_metadata
                else 0,
            )

            reasoning_tokens = (
                response.usage_metadata.thoughts_token_count or 0 if response.usage_metadata else 0
            )
            t_end = time.time()

            # The storage meter attaches to the request that created the cache,
            # whether or not the generate itself reported usage.
            extra: dict[str, int | float | str] = {"reasoning_tokens": reasoning_tokens}
            if storage_token_hours:
                extra["cache_storage_token_hours"] = storage_token_hours

            return SchematicGenerationResult(
                content=model_content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=response.usage_metadata.prompt_token_count or 0,
                        output_tokens=(response.usage_metadata.candidates_token_count or 0)
                        + reasoning_tokens,
                        cached_input_tokens=(
                            response.usage_metadata.cached_content_token_count or 0
                            if response.usage_metadata
                            else 0
                        )
                        or 0,
                        extra=extra,
                    )
                    if response.usage_metadata
                    else UsageInfo(input_tokens=0, output_tokens=0, extra=extra),
                ),
            )
        except ValidationError:
            self.logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{json_result}"
            )
            raise

    def _get_schema_function_declaration(self) -> google.genai.types.FunctionDeclaration:
        # Create a signature from parameters
        sig = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name="log_data",
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=convert_model_to_gemini_compatible_schema(self.schema),
                )
            ],
            return_annotation=bool,
        )

        # Create a fake callable
        def log_data() -> None:
            pass

        # Attach the signature
        log_data.__signature__ = sig  # type: ignore

        fd = google.genai.types.FunctionDeclaration.from_callable(
            callable=log_data,
            client=self._client,  # type: ignore
        )

        return fd

    def _output_tools(
        self,
    ) -> tuple[list[google.genai.types.Tool], google.genai.types.ToolConfig]:
        """The schema function-declaration tool plus a tool_config that forces the
        model to call it. Built once per request so the same value can be sent
        inline or baked into a CachedContent."""
        fd = self._get_schema_function_declaration()
        tools = [google.genai.types.Tool(function_declarations=[fd])]
        tool_config = google.genai.types.ToolConfig(
            function_calling_config=google.genai.types.FunctionCallingConfig(
                mode=google.genai.types.FunctionCallingConfigMode.ANY,
                allowed_function_names=[fd.name],
            )
        )
        return tools, tool_config

    # ---- explicit caching --------------------------------------------------

    def _prefix_hash(
        self,
        prefix_text: str,
        tools: list[google.genai.types.Tool],
        tool_config: google.genai.types.ToolConfig,
    ) -> str:
        # Identifies the cached content independently of the session key, so a
        # prefix Gemini rejects (e.g. below the token minimum) is recognized as
        # uncacheable across every session sharing it. Tools / tool_config are
        # baked into the cache, so they're part of its identity too.
        hasher = hashlib.sha256()
        hasher.update(self.model_name.encode("utf-8"))
        hasher.update(prefix_text.encode("utf-8"))
        for tool in tools:
            hasher.update(tool.model_dump_json(exclude_none=True).encode("utf-8"))
        hasher.update(tool_config.model_dump_json(exclude_none=True).encode("utf-8"))
        return hasher.hexdigest()

    def _cache_lookup_key(self, key: str, prefix_hash: str) -> str:
        return f"{key}:{prefix_hash}"

    def _split_prompt_at_breakpoint(
        self, prompt: str, breakpoint: str
    ) -> Optional[tuple[str, str]]:
        index = prompt.find(breakpoint)
        if index < 0:
            self.logger.debug(
                f"Gemini schematic cache breakpoint {breakpoint!r} was not found; "
                "proceeding without caching."
            )
            return None

        prefix_text = prompt[:index]
        suffix_text = prompt[index:]

        if not prefix_text or not suffix_text:
            self.logger.debug(
                f"Gemini schematic cache breakpoint {breakpoint!r} did not split "
                "the prompt into a cacheable prefix and live suffix; proceeding without caching."
            )
            return None

        return prefix_text, suffix_text

    async def _get_or_create_cache(
        self,
        *,
        key: str,
        lookup_key: str,
        prefix_text: str,
        ttl_seconds: int,
        tools: list[google.genai.types.Tool],
        tool_config: google.genai.types.ToolConfig,
    ) -> tuple[Optional[str], float]:
        """Return ``(cache_name, storage_token_hours)``.

        The second element is nonzero only when this call actually created the
        cache: Gemini bills explicit-cache STORAGE per token-hour, a charge no
        per-request usage ever reports. It is exact at creation time — token
        count from the create response times the requested lifetime — because
        managed caches are never deleted early (they lapse at their TTL)."""
        prefix_hash = self._prefix_hash(prefix_text, tools, tool_config)

        async with self._cache_lock:
            if prefix_hash in self._uncacheable_prefixes:
                return None, 0.0

            entry = self._managed_caches.get(lookup_key)
            if entry is not None:
                name, _prefix_text, expiry = entry
                if expiry - datetime.now(timezone.utc) > self._CACHE_REUSE_MARGIN:
                    return name, 0.0

            created = await self._create_cache(
                display_name=lookup_key,
                prefix_hash=prefix_hash,
                prefix_text=prefix_text,
                ttl_seconds=ttl_seconds,
                tools=tools,
                tool_config=tool_config,
            )
            if created is None:
                return None, 0.0

            _cached_name, storage_token_hours = created
            entry = self._managed_caches.get(lookup_key)
            return (entry[0] if entry is not None else None), storage_token_hours

    async def _create_cache(
        self,
        *,
        display_name: str,
        prefix_hash: str,
        prefix_text: str,
        ttl_seconds: int,
        tools: list[google.genai.types.Tool],
        tool_config: google.genai.types.ToolConfig,
    ) -> Optional[tuple[str, float]]:
        config = google.genai.types.CreateCachedContentConfig(
            display_name=display_name,
            contents=prefix_text,
            tools=tools,
            tool_config=tool_config,
            ttl=f"{ttl_seconds}s",
        )

        try:
            cached = await self._client.aio.caches.create(model=self.model_name, config=config)
        except ClientError as exc:
            # Deterministic rejection (e.g. prefix below the minimum token count):
            # this prefix will never cache, so stop retrying it.
            self.logger.warning(
                f"Gemini rejected schematic caching for key '{display_name}' "
                f"({exc}); proceeding without caching."
            )
            self._uncacheable_prefixes[prefix_hash] = True
            return None
        except Exception as exc:  # noqa: BLE001 - transient: degrade, retry later
            self.logger.warning(
                f"Gemini schematic cache creation failed for key '{display_name}' "
                f"({exc}); proceeding without caching."
            )
            return None

        assert cached.name and cached.expire_time

        # Register the new cache under the fully resolved lookup identity. We
        # deliberately do NOT delete prior caches: a concurrent request may still
        # reference one, and old resources simply lapse at their TTL.
        self._managed_caches[display_name] = (cached.name, prefix_text, cached.expire_time)

        # Storage bills per token-hour for the cache's whole lifetime; the create
        # response reports the cached token count.
        cached_tokens = (
            getattr(getattr(cached, "usage_metadata", None), "total_token_count", 0) or 0
        )
        storage_token_hours = cached_tokens * ttl_seconds / 3600.0
        return cached.name, storage_token_hours

    def _is_missing_cache_error(self, error: ClientError) -> bool:
        """Whether a request failed because its referenced CachedContent is gone
        (expired/deleted), as opposed to some other error."""
        message = str(error).lower()
        return "cachedcontent" in message or "cached content" in message

    async def _forget_cache(self, key: str, name: str) -> None:
        """Drop a managed cache entry that turned out to be unusable, so later loads
        miss (and a later store recreates it) instead of re-hitting the dead name."""
        async with self._cache_lock:
            entry = self._managed_caches.get(key)
            if entry is not None and entry[0] == name:
                del self._managed_caches[key]

    def _reasoning_thinking_config(self, effort: ReasoningEffort) -> dict[str, Any] | None:
        """Map the normalized reasoning effort to this model's thinking config, or
        None for models without thinking support (e.g. Gemini 2.0)."""
        if self.model_name.startswith("gemini-3"):
            return {"thinking_level": self._EFFORT_TO_THINKING_LEVEL_3X[effort]}
        if self.model_name.startswith("gemini-2.5"):
            return {"thinking_budget": self._EFFORT_TO_THINKING_BUDGET_25[effort]}
        return None


class Gemini_2_0_Flash(GeminiSchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gemini-2.0-flash",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_2_0_Flash_Lite(GeminiSchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gemini-2.0-flash-lite-preview-02-05",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_2_5_Flash(GeminiSchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gemini-2.5-flash",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )

    @override
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        return await super().generate(
            prompt,
            {"thinking_config": {"thinking_budget": 0}, **hints},
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_2_5_Flash_Lite(GeminiSchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gemini-2.5-flash-lite",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )

    @override
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        return await super().generate(
            prompt,
            {"thinking_config": {"thinking_budget": 0}, **hints},
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_2_5_Pro(GeminiSchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gemini-2.5-pro",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_3_1_Pro(GeminiSchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gemini-3.1-pro-preview",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_3_1_Flash_Lite(GeminiSchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gemini-3.1-flash-lite",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_3_5_Flash(GeminiSchematicGenerator[T]):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gemini-3.5-flash",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024

    @override
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        return await super().generate(prompt, {**hints})


# The key under which Gemini's per-part ``thought_signature`` is preserved in a
# canonical Part's ``provider_data``. It MUST round-trip verbatim, or replaying
# tool-calling history triggers a 400 "missing thought_signature".
GEMINI_THOUGHT_SIGNATURE_KEY = gemini_interactions.SIGNATURE


def _signature_to_bytes(signature: Union[str, bytes, None]) -> Optional[bytes]:
    if signature is None:
        return None
    if isinstance(signature, bytes):
        return signature
    return signature.encode("utf-8")


class GeminiReactGenerator(ReactGenerator):
    """A ReAct generator backed by Gemini (google-genai).

    Implements the ``ReactGenerator`` provider seam: ``_encode`` builds the
    google-genai request, ``_raw_stream`` opens the streaming call, and
    ``_decode`` folds each native chunk into the shared ``TurnBuilder``.

    Gemini ``thought_signature`` values are preserved verbatim on each Part's
    ``provider_data`` (under :data:`GEMINI_THOUGHT_SIGNATURE_KEY`) so that
    multi-step tool-calling history replays without 400 errors.

    Argon uses stateless Interactions with implicit caching only; older models
    retain generateContent and explicit caching.

    Caching (:class:`CacheConfig` + :attr:`Message.cache`): a marked prefix is
    turned into an explicit Gemini ``CachedContent`` resource, created lazily and
    reused across calls that share the same prefix. Cached resources auto-expire
    at their TTL (the Gemini API's default is ~1h), so cleanup is not required for
    correctness; call :meth:`aclose` to delete the ones this generator created
    early (e.g. on shutdown) for cost control.
    """

    # Gemini Content.role accepts only "user"/"model"; function responses ride in
    # a "user" turn. (A bogus "tool" role is tolerated by generateContent but
    # rejected by cachedContents validation, breaking caching once the history
    # contains tool results.)
    _ROLE_MAP = {Role.USER: "user", Role.ASSISTANT: "model", Role.TOOL: "user"}
    # Don't reuse a cached prefix that's about to expire mid-request.
    _CACHE_REUSE_MARGIN = timedelta(seconds=30)
    _MAX_OUTPUT_TOKENS = 4096

    # Mapping from canonical ModelSize to a concrete Gemini model id, used to
    # resolve per-call ``hints`` overrides on ``_encode``.
    _MODEL_BY_SIZE: dict[ModelSize, str] = {
        ModelSize.SMALL: "gemini-3.1-flash-lite",
        ModelSize.MEDIUM: "gemini-3.5-flash",
        ModelSize.LARGE: "gemini-3.5-flash",
    }

    # Cache minimum assumed for models we don't recognize — large enough that
    # prefill is skipped rather than warming a cache that may never engage.
    _UNKNOWN_MIN_CACHE_SIZE = 1 << 20
    _MISTAKEN_THOUGHT_PREFIX = "thought\n"

    # Gemini's service_tier accepts "standard" / "flex" / "priority" directly.
    _SERVICE_TIER: dict[ServiceTier, str] = {
        "standard": "standard",
        "flex": "flex",
        "priority": "priority",
    }

    def __init__(
        self,
        *,
        model: str = "gemini-3.1-flash-lite",
        logger: Logger,
        cache: Optional[CacheConfig] = None,
        client: Optional[google.genai.Client] = None,
        api_key: Optional[str] = None,
        service_tier: ServiceTier = "standard",
        provider_name: Optional[str] = None,
        usage_reporter: Optional[UsageReporter] = None,
    ) -> None:
        super().__init__(model=model, cache=cache, usage_reporter=usage_reporter)
        self._logger = ScopedLogger(logger, "GeminiReactGenerator")
        self._default_service_tier = service_tier
        self._provider_name = provider_name or "gemini"
        self._client = client or google.genai.Client(
            api_key=require_api_key("Gemini", "GEMINI_API_KEY", api_key),
            http_options=google.genai.types.HttpOptions(
                timeout=_GEMINI_REQUEST_TIMEOUT_MS,
                headers=_gemini_identity_headers(),
            ),
        )
        self._interactions_initialization: asyncio.Task[None] | None = None
        # Caches this generator created and may reuse: key -> (resource name, expiry).
        self._managed_caches: dict[str, tuple[str, datetime]] = {}
        # Prefix keys we know can't be cached (e.g. below the provider minimum),
        # so we don't re-attempt creation on every call.
        self._uncacheable_keys: cachetools.LRUCache[str, bool] = cachetools.LRUCache(maxsize=256)
        self._cache_lock = asyncio.Lock()
        # Storage charges from caches created while building a request, drained
        # into the next decoded usage (see _decode_usage).
        self._pending_storage_token_hours: float = 0.0

    @property
    def id(self) -> str:
        return f"gemini/{self.model}"

    @property
    @override
    def provider_name(self) -> str:
        return self._provider_name

    @override
    def _capture_tool_artifacts(self, calls: Sequence[ToolCallPart], blob: dict[str, Any]) -> None:
        if gemini_interactions.uses_interactions(str(blob["model"])):
            gemini_interactions.capture(calls, blob)
            return
        # Gemini 3.x requires a valid thought_signature on every replayed
        # functionCall; it can't be synthesized, so persist it (per call,
        # index-aligned; base64 since it's bytes). It rides on the tool-call
        # part's provider_data (typically only the first part of the turn).
        signatures: list[Optional[str]] = []
        for call in calls:
            raw = call.provider_data.get(GEMINI_THOUGHT_SIGNATURE_KEY)
            signature = _signature_to_bytes(raw)
            signatures.append(base64.b64encode(signature).decode("ascii") if signature else None)
        blob["thought_signatures"] = signatures

    @override
    def _restore_tool_artifacts(
        self,
        calls: Sequence[ToolCallPart],
        blob: Mapping[str, Any],
        *,
        model: Optional[str] = None,
    ) -> bool:
        if gemini_interactions.uses_interactions(model or self.model):
            return gemini_interactions.restore(calls, blob, model or self.model)
        # The signature is model-bound and unsynthesizable: only replay natively
        # when the model that produced the blob matches the one the replaying turn
        # will run on, and a usable signature is present.
        if blob.get("model") != (model or self.model):
            return False
        signatures = blob.get("thought_signatures") or []
        if len(signatures) != len(calls) or not any(signatures):
            return False
        for call, signature in zip(calls, signatures):
            if signature:
                call.provider_data[GEMINI_THOUGHT_SIGNATURE_KEY] = base64.b64decode(signature)
        return True

    # ---- provider seam -----------------------------------------------------

    def _resolve_model(self, hints: ReactGeneratorHints) -> str:
        """Return the model id for this call, applying ``hints['model_size']``
        if present (falling back to the generator's default)."""
        size = hints.get("model_size", ModelSize.AUTO)
        return self._MODEL_BY_SIZE.get(size, self.model)

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
        resolved_model = self._resolve_model(hints)
        if gemini_interactions.uses_interactions(resolved_model):
            return gemini_interactions.encode(
                history,
                tools,
                tool_choice,
                reasoning,
                resolved_model,
                self._SERVICE_TIER[hints.get("service_tier", self._default_service_tier)],
            )
        system_chunks: list[str] = []
        tail_instruction_chunks: list[str] = []

        # Caching is positional: everything up to and including the last message
        # with a cache_key is the stable prefix to cache; the rest is the live
        # suffix sent on each call. The key names the cache for reuse.
        cache_split = -1
        cache_key: Optional[str] = None
        system_cache_key: Optional[str] = None
        seen_non_system = False
        non_system: list[Message] = []
        for message in history:
            if message.role == Role.SYSTEM:
                if not seen_non_system:
                    # Leading system → the stable, cacheable system_instruction.
                    if message.text:
                        system_chunks.append(message.text)
                    if self.cache.enabled and message.cache_key is not None:
                        system_cache_key = message.cache_key
                elif message.text:
                    # Gemini has no mid-conversation system role: append (wrapped)
                    # to the END of the last user message below so it stays in the
                    # live suffix, out of the cached system_instruction.
                    tail_instruction_chunks.append(message.text)
                continue
            seen_non_system = True
            if self.cache.enabled and message.cache_key is not None:
                cache_split = len(non_system)
                cache_key = message.cache_key
            non_system.append(message)

        contents = [self._encode_message(message) for message in non_system]

        if tail_instruction_chunks:
            self._append_turn_instructions(contents, "\n\n".join(tail_instruction_chunks))

        if not contents:
            # generate_content rejects empty contents; an agent-first greeting
            # over an untouched conversation has only system content.
            contents = [
                google.genai.types.Content(
                    role="user",
                    parts=[google.genai.types.Part(text=EMPTY_CONVERSATION_USER_TURN)],
                )
            ]

        system_instruction = "\n\n".join(chunk for chunk in system_chunks if chunk) or None

        # Declare the tail-instruction convention in the (cached) system prompt.
        # Added unconditionally so system_instruction stays identical across turns
        # and prefills (otherwise the CachedContent would not be reused).
        if system_instruction is not None:
            system_instruction += SYSTEM_UPDATE_MESSAGE

        tool_block: Optional[list[google.genai.types.Tool]] = None
        tool_config: Optional[google.genai.types.ToolConfig] = None
        if tools:
            tool_block = [
                google.genai.types.Tool(
                    function_declarations=[self._encode_tool(spec) for spec in tools]
                )
            ]
            tool_config = self._encode_tool_choice(tool_choice)
        elif tool_choice == "none":
            # No tools to declare, but still forbid function calls explicitly:
            # otherwise the model can keep emitting them from the conversation
            # history (defeating the engine's disable-tools fallback).
            tool_config = self._encode_tool_choice("none")

        resolved_model = self._resolve_model(hints)

        # Always emit a thinking config — depth is controlled by ``effort``.
        # ``"minimal"`` resolves to "off" on Gemini 2.5 (``thinking_budget=0``);
        # on 3.x it routes to the lowest available ``thinking_level``.
        thinking_config = self._encode_thinking(reasoning, resolved_model=resolved_model)

        # Never cache the final (live) turn: it changes on every call and must be
        # sent as the suffix. If the caller marked the last message (e.g. it marks
        # every message with the same cache_key), clamp the split so at least the
        # last content stays live — otherwise the suffix would be empty and Gemini
        # rejects the request with "contents are required".
        if cache_split >= len(non_system) - 1:
            cache_split = len(non_system) - 2

        prefix_contents: Optional[list[google.genai.types.Content]] = None
        suffix_contents = contents
        if cache_split >= 0:
            prefix_contents = contents[: cache_split + 1]
            suffix_contents = contents[cache_split + 1 :]
        elif system_cache_key is not None:
            # Only the system is marked: cache the system instruction alone, and
            # send the whole conversation as the suffix referencing it.
            prefix_contents = []
            cache_key = system_cache_key

        explicit_cache = self.cache.provider_options.get("gemini_cached_content")

        return {
            "model": resolved_model,
            "system_instruction": system_instruction,
            "tools": tool_block,
            "tool_config": tool_config,
            "thinking_config": thinking_config,
            "service_tier": self._SERVICE_TIER[
                hints.get("service_tier", self._default_service_tier)
            ],
            "max_output_tokens": self._MAX_OUTPUT_TOKENS,
            "all_contents": contents,
            "prefix_contents": prefix_contents,  # cache this (None => no managed cache)
            "suffix_contents": suffix_contents,  # send this when a cache is used
            "cache_key": cache_key,  # caller-provided reuse identity for the prefix
            "explicit_cache_name": explicit_cache,
        }

    @override
    def capture_assistant_artifacts(self, message: Message) -> Mapping[str, Any] | None:
        return gemini_interactions.capture_assistant(message, self.model)

    @override
    def restore_assistant_message(self, *, text: str, replay_data: Any = None) -> Message:
        return gemini_interactions.restore_assistant(text, replay_data, self.model)

    def _encode_message(self, message: Message) -> google.genai.types.Content:
        parts = [self._encode_part(part) for part in message.parts]
        return google.genai.types.Content(
            role=self._ROLE_MAP[message.role],
            parts=[p for p in parts if p is not None],
        )

    def _append_turn_instructions(
        self, contents: list[google.genai.types.Content], instructions: str
    ) -> None:
        """Deliver per-turn considerations as the result of a synthetic
        ``system_update`` tool — a function response the model
        treats as fetched data rather than as customer input (which Gemini tends to
        echo). Appended as a trailing turn so it stays in the live suffix, out of
        the cached prefix. No functionCall accompanies it: Gemini accepts an
        unpaired functionResponse, while a synthetic functionCall would be rejected
        for lacking a thought_signature."""
        contents.append(
            google.genai.types.Content(
                role=self._ROLE_MAP[Role.TOOL],
                parts=[
                    google.genai.types.Part(
                        function_response=google.genai.types.FunctionResponse(
                            name=SYSTEM_UPDATE_TOOL_NAME,
                            response={"content": instructions},
                        )
                    )
                ],
            )
        )

    def _encode_part(self, part: Any) -> Optional[google.genai.types.Part]:
        signature = part.provider_data.get(GEMINI_THOUGHT_SIGNATURE_KEY)

        if isinstance(part, TextPart):
            return google.genai.types.Part(text=part.text, thought_signature=signature)

        if isinstance(part, ReasoningPart):
            return google.genai.types.Part(
                text=part.text,
                thought=True,
                thought_signature=_signature_to_bytes(part.signature) or signature,
            )

        if isinstance(part, ToolCallPart):
            return google.genai.types.Part(
                function_call=google.genai.types.FunctionCall(
                    id=part.id or None,
                    name=part.name,
                    args=part.args,
                ),
                thought_signature=signature,
            )

        if isinstance(part, ToolResultPart):
            return google.genai.types.Part(
                function_response=google.genai.types.FunctionResponse(
                    id=part.call_id or None,
                    name=part.name,
                    response=self._encode_tool_response(part.content),
                )
            )

        return None

    def _encode_tool_response(self, content: Any) -> dict[str, Any]:
        # Gemini requires the function response to be a JSON object.
        if isinstance(content, MappingABC):
            return dict(content)
        return {"result": content}

    def _encode_tool(self, spec: ToolSpec) -> google.genai.types.FunctionDeclaration:
        return google.genai.types.FunctionDeclaration(
            name=spec.name,
            description=spec.description,
            # The JSON Schema dict is coerced to a google-genai Schema.
            parameters=spec.json_schema() if spec.parameters else None,
        )

    def _encode_tool_choice(self, tool_choice: ToolChoice) -> google.genai.types.ToolConfig:
        mode_enum = google.genai.types.FunctionCallingConfigMode

        if isinstance(tool_choice, MappingABC):
            name = tool_choice.get("name")
            function_calling_config = google.genai.types.FunctionCallingConfig(
                mode=mode_enum.ANY,
                allowed_function_names=[name] if name else None,
            )
        else:
            mode = {
                "auto": mode_enum.AUTO,
                "none": mode_enum.NONE,
                "required": mode_enum.ANY,
            }[tool_choice]
            function_calling_config = google.genai.types.FunctionCallingConfig(mode=mode)

        return google.genai.types.ToolConfig(function_calling_config=function_calling_config)

    # Maps ``ReasoningConfig.effort`` to a thinking-token budget for Gemini 2.5
    # models. ``"minimal"`` resolves to 0, which is the documented way to turn
    # thinking off on 2.5 flash / flash-lite. (2.5 Pro silently ignores 0 and
    # always thinks — there's nothing the adapter can do about that.)
    _EFFORT_TO_BUDGET_25: dict[str, int] = {
        "minimal": 0,
        "low": 1024,
        "medium": 8192,
        "high": 24576,
    }

    # Maps ``ReasoningConfig.effort`` to Gemini 3.x's ``thinking_level``. 3.x
    # uses a coarse named-level knob instead of a numeric budget and cannot fully
    # disable thinking. ``LOW`` is the safe floor: some 3.x models reject
    # ``MINIMAL`` outright, so "minimal" is clamped up to ``LOW``.
    _EFFORT_TO_LEVEL_3X: dict[str, google.genai.types.ThinkingLevel] = {
        "minimal": google.genai.types.ThinkingLevel.LOW,
        "low": google.genai.types.ThinkingLevel.LOW,
        "medium": google.genai.types.ThinkingLevel.MEDIUM,
        "high": google.genai.types.ThinkingLevel.HIGH,
    }

    def _encode_thinking(
        self, reasoning: ReasoningConfig, *, resolved_model: str
    ) -> google.genai.types.ThinkingConfig:
        thinking_kwargs: dict[str, Any] = {
            "include_thoughts": reasoning.visibility != "none",
        }
        if resolved_model.startswith("gemini-3"):
            # Gemini 3.x prefers ``thinking_level``; ``thinking_budget`` is
            # accepted for backward compat but may produce unexpected results
            # on 3.x Pro, so we don't emit it.
            thinking_kwargs["thinking_level"] = self._EFFORT_TO_LEVEL_3X[reasoning.effort]
        else:
            # Gemini 2.5 uses an explicit numeric budget.
            thinking_kwargs["thinking_budget"] = self._EFFORT_TO_BUDGET_25[reasoning.effort]
        return google.genai.types.ThinkingConfig(**thinking_kwargs)

    def _min_cache_size(self, model: str) -> int:
        """Minimum prompt size (in tokens) at which Gemini context caching
        engages for ``model``. Pro models require 2048 tokens, Flash models 1024;
        unknown models are assumed not to cache cheaply."""
        if "pro" in model:
            return 2048
        if "flash" in model:
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
        if gemini_interactions.uses_interactions(self._resolve_model(hints)):
            return False
        token_count = self._estimate_prefill_tokens(history, tools)
        return token_count >= self._min_cache_size(self._resolve_model(hints))

    @override
    async def _prefill(self, request: Any) -> Usage:
        # Gemini supports explicit caching: create the CachedContent resource for
        # the marked prefix now, so a later call reuses it. No inference needed.
        prefix_contents = request.get("prefix_contents")
        cache_key = request.get("cache_key")
        if prefix_contents is None or not cache_key:
            return Usage(model_name=request["model"])

        _name, storage_token_hours = await self._get_or_create_cache(
            model=request["model"],
            system_instruction=request["system_instruction"],
            prefix_contents=prefix_contents,
            tools=request["tools"],
            tool_config=request["tool_config"],
            cache_key=cache_key,
        )
        return Usage(
            model_name=request["model"],
            extra={"cache_storage_token_hours": storage_token_hours} if storage_token_hours else {},
        )

    def _initialize_interactions(self) -> None:
        # Google's first property access synchronously imports the Interactions
        # SDK. Keep that work off the serving loop, including under CPU quotas.
        _ = self._client.aio.interactions

    async def _ensure_interactions_initialized(self) -> None:
        if self._interactions_initialization is None:
            self._interactions_initialization = asyncio.create_task(
                asyncio.to_thread(self._initialize_interactions)
            )
        initialization = self._interactions_initialization
        try:
            # A cancelled request must not cancel initialization for other
            # requests, or let them race a worker that is still importing.
            await asyncio.shield(initialization)
        except Exception:
            if self._interactions_initialization is initialization:
                self._interactions_initialization = None
            raise

    async def _raw_stream(self, request: Any) -> AsyncIterator[Any]:
        if gemini_interactions.uses_interactions(request["model"]):
            await self._ensure_interactions_initialized()
            async for interaction_chunk in gemini_interactions.stream(
                self._client, request, self._decode_visible_text_part
            ):
                yield interaction_chunk
            return
        # thinking_config is always set on the request (it's allowed alongside a
        # CachedContent). system_instruction / tools / tool_config are NOT: Gemini
        # rejects a cached request that also sets them, so when a cache is used
        # they're baked into the cache instead and only set inline otherwise.
        config_kwargs: dict[str, Any] = {}
        if request["thinking_config"] is not None:
            config_kwargs["thinking_config"] = request["thinking_config"]
        if request.get("service_tier") is not None:
            config_kwargs["service_tier"] = request["service_tier"]
        if request.get("max_output_tokens") is not None:
            config_kwargs["max_output_tokens"] = request["max_output_tokens"]

        cached_content_name: Optional[str] = request["explicit_cache_name"]
        contents = request["all_contents"]

        if cached_content_name is not None:
            # Reuse a caller-provided cache: assume it holds the system prompt,
            # tools, and tool_config.
            contents = request["all_contents"]
        elif request["prefix_contents"] is not None:
            # Managed cache: cache the marked prefix (system + prefix contents +
            # tools/tool_config), then send only the live suffix referencing it.
            cached_content_name, created_storage_token_hours = await self._get_or_create_cache(
                model=request["model"],
                system_instruction=request["system_instruction"],
                prefix_contents=request["prefix_contents"],
                tools=request["tools"],
                tool_config=request["tool_config"],
                cache_key=request["cache_key"],
            )
            if created_storage_token_hours:
                # Accumulated, then drained into the step's decoded usage —
                # total-preserving even if steps interleave on this instance.
                self._pending_storage_token_hours += created_storage_token_hours
            if cached_content_name is not None:
                contents = request["suffix_contents"]

        if cached_content_name is not None:
            config_kwargs["cached_content"] = cached_content_name
        else:
            # No cache in play: set system_instruction / tools / tool_config inline.
            if request["system_instruction"] is not None:
                config_kwargs["system_instruction"] = request["system_instruction"]
            if request["tools"] is not None:
                config_kwargs["tools"] = request["tools"]
            if request["tool_config"] is not None:
                config_kwargs["tool_config"] = request["tool_config"]

        config = google.genai.types.GenerateContentConfig(**config_kwargs)

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=request["model"],
                contents=contents,
                config=config,
            )
            try:
                async for chunk in stream:
                    yield chunk
            finally:
                # On cancellation (or any early exit) close the underlying stream
                # so the HTTP response/connection is released rather than leaked.
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    await aclose()
        except TooManyRequests as exc:
            self._logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise ReactError(str(exc), retryable=True) from exc
        except (NotFound, ResourceExhausted, ServerError) as exc:
            # Transient — mirror the schematic generator's retry set. The consumer
            # (BaseLoop) retries these before any event of the step is emitted.
            raise ReactError(str(exc), retryable=True) from exc
        except APIError as exc:
            # The google-genai SDK raises its OWN ClientError/ServerError (distinct
            # from google.api_core's — not subclasses), so the above never catch
            # them. Retry transient HTTP statuses (rate limit + 5xx, e.g. 503);
            # let other 4xx (bad request, auth) propagate unretried.
            if exc.code in (429, 500, 502, 503, 504):
                raise ReactError(str(exc), retryable=True) from exc
            raise

    # ---- explicit caching --------------------------------------------------

    def _cache_key(
        self,
        cache_key: str,
        model: str,
        system_instruction: Optional[str],
        prefix_contents: list[google.genai.types.Content],
        tools: Optional[list[google.genai.types.Tool]] = None,
        tool_config: Optional[google.genai.types.ToolConfig] = None,
    ) -> str:
        # Fold the caller's key together with the actual content: the key gives
        # intentional identity, the content hash guarantees a key reused after an
        # edited prefix never serves stale content. Tools / tool_config are baked
        # into the cache (Gemini forbids passing them on a cached request), so
        # they're part of the identity too.
        hasher = hashlib.sha256()
        hasher.update(cache_key.encode("utf-8"))
        hasher.update(model.encode("utf-8"))
        hasher.update((system_instruction or "").encode("utf-8"))
        for content in prefix_contents:
            hasher.update(content.model_dump_json(exclude_none=True).encode("utf-8"))
        for tool in tools or []:
            hasher.update(tool.model_dump_json(exclude_none=True).encode("utf-8"))
        if tool_config is not None:
            hasher.update(tool_config.model_dump_json(exclude_none=True).encode("utf-8"))
        return hasher.hexdigest()

    async def _get_or_create_cache(
        self,
        *,
        model: str,
        system_instruction: Optional[str],
        prefix_contents: list[google.genai.types.Content],
        cache_key: str,
        tools: Optional[list[google.genai.types.Tool]] = None,
        tool_config: Optional[google.genai.types.ToolConfig] = None,
    ) -> tuple[Optional[str], float]:
        """Return ``(cache_name, storage_token_hours)`` — name ``None`` if caching
        is unavailable for this prefix. Caching is an optimization, so this never
        raises for cache problems — the caller falls back to an inline request.

        ``storage_token_hours`` is nonzero only when this call created the cache:
        explicit-cache storage bills per token-hour for the resource's lifetime
        and never appears in per-request usage, so it is accounted exactly once,
        at creation, from the create response's token count and expiry.

        ``system_instruction`` / ``tools`` / ``tool_config`` are baked into the
        cached content: Gemini rejects a GenerateContent request that both uses a
        CachedContent and sets those fields, so they must live in the cache."""
        key = self._cache_key(
            cache_key, model, system_instruction, prefix_contents, tools, tool_config
        )

        async with self._cache_lock:
            if key in self._uncacheable_keys:
                return None, 0.0

            existing = self._managed_caches.get(key)
            if existing is not None:
                name, expiry = existing
                # Reuse only while comfortably within the resource's lifetime.
                if expiry - datetime.now(timezone.utc) > self._CACHE_REUSE_MARGIN:
                    return name, 0.0

            config_kwargs: dict[str, Any] = {"display_name": cache_key}
            if prefix_contents:
                config_kwargs["contents"] = prefix_contents
            if system_instruction is not None:
                config_kwargs["system_instruction"] = system_instruction
            if tools is not None:
                config_kwargs["tools"] = tools
            if tool_config is not None:
                config_kwargs["tool_config"] = tool_config
            if self.cache.ttl is not None:
                config_kwargs["ttl"] = f"{int(self.cache.ttl.total_seconds())}s"

            try:
                cached = await self._client.aio.caches.create(
                    model=model,
                    config=google.genai.types.CreateCachedContentConfig(**config_kwargs),
                )
            except ClientError as exc:
                # Deterministic rejection (e.g. prefix below the minimum token
                # count): this prefix will never cache, so stop retrying it.
                self._logger.warning(
                    f"Gemini rejected caching for key '{cache_key}' "
                    f"({exc}); proceeding without caching."
                )
                self._uncacheable_keys[key] = True
                return None, 0.0
            except Exception as exc:  # noqa: BLE001 - transient: degrade, retry later
                self._logger.warning(
                    f"Gemini cache creation failed for key '{cache_key}' "
                    f"({exc}); proceeding without caching."
                )
                return None, 0.0

            assert cached.name and cached.expire_time
            self._managed_caches[key] = (cached.name, cached.expire_time)

            cached_tokens = (
                getattr(getattr(cached, "usage_metadata", None), "total_token_count", 0) or 0
            )
            lifetime_hours = (
                max((cached.expire_time - datetime.now(timezone.utc)).total_seconds(), 0.0) / 3600.0
            )
            return cached.name, cached_tokens * lifetime_hours

    async def aclose(self) -> None:
        """Delete every cache this generator created. Optional: cached content
        auto-expires at its TTL, but deleting early frees the resource (and cost)
        sooner. Best-effort; failures are logged and swallowed."""
        async with self._cache_lock:
            for name, _ in self._managed_caches.values():
                try:
                    await self._client.aio.caches.delete(name=name)
                except Exception as exc:  # noqa: BLE001 - cleanup must not raise
                    self._logger.warning(f"Failed to delete Gemini cache {name}: {exc}")
            self._managed_caches.clear()

    @override
    def _decode(self, raw_event: Any, builder: TurnBuilder) -> list[StreamEvent]:
        if isinstance(raw_event, gemini_interactions.Chunk):
            return raw_event.state.decode(raw_event.event, builder)
        events: list[StreamEvent] = []

        candidate = raw_event.candidates[0] if raw_event.candidates else None
        if candidate is not None:
            if candidate.finish_reason is not None:
                builder.finish_reason = self._map_finish_reason(candidate.finish_reason)

            content = candidate.content
            for part in content.parts if content and content.parts else []:
                events.extend(self._decode_part(part, builder))

        if raw_event.usage_metadata is not None:
            model_name = getattr(raw_event, "model_version", "") or ""
            builder.usage = self._decode_usage(raw_event.usage_metadata, model_name)

        return events

    def _decode_part(self, part: Any, builder: TurnBuilder) -> list[StreamEvent]:
        signature = part.thought_signature
        provider_data = {GEMINI_THOUGHT_SIGNATURE_KEY: signature} if signature else None

        if part.function_call is not None:
            function_call = part.function_call
            call_id = function_call.id or uuid.uuid4().hex
            name = function_call.name or ""
            builder.tool_call(
                call_id,
                name=name,
                args=dict(function_call.args or {}),
                provider_data=provider_data,
            )
            return [ToolCallStarted(id=call_id, name=name)]

        if part.text is not None:
            if part.thought:
                builder.reasoning_delta(
                    part.text,
                    visibility="summary",
                    provider_data=provider_data,
                )
                return [ReasoningDelta(text=part.text)] if part.text else []

            return self._decode_visible_text_part(part.text, builder, provider_data)

        if provider_data:
            # Streaming can deliver the final signature without a text delta.
            builder.text_delta("", provider_data=provider_data)

        return []

    def _decode_visible_text_part(
        self,
        text: str,
        builder: TurnBuilder,
        provider_data: Mapping[str, Any] | None,
    ) -> list[StreamEvent]:
        if not text:
            if provider_data:
                builder.text_delta("", provider_data=provider_data)
            return []

        current_text = builder.finish().message.text
        combined_text = current_text + text

        if combined_text.startswith(self._MISTAKEN_THOUGHT_PREFIX):
            raise ReactError(
                "Gemini produced a visible message starting with 'thought\\n'",
                retryable=True,
            )

        if self._MISTAKEN_THOUGHT_PREFIX.startswith(combined_text):
            builder.text_delta(text, provider_data=provider_data)
            return []

        if current_text and self._MISTAKEN_THOUGHT_PREFIX.startswith(current_text):
            builder.text_delta(text, provider_data=provider_data)
            return [TextDelta(text=combined_text)] if combined_text else []

        builder.text_delta(text, provider_data=provider_data)
        return [TextDelta(text=text)] if text else []

    def _map_finish_reason(self, finish_reason: Any) -> FinishReason:
        name = getattr(finish_reason, "name", str(finish_reason))
        if name == "STOP":
            return FinishReason.STOP
        if name == "MAX_TOKENS":
            return FinishReason.MAX_TOKENS
        if name in {
            "SAFETY",
            "PROHIBITED_CONTENT",
            "BLOCKLIST",
            "SPII",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
        }:
            return FinishReason.CONTENT_FILTER
        if name == "MALFORMED_FUNCTION_CALL":
            return FinishReason.ERROR
        return FinishReason.STOP

    def _decode_usage(self, usage_metadata: Any, model_name: str) -> Usage:
        candidates_tokens = usage_metadata.candidates_token_count or 0
        reasoning_tokens = usage_metadata.thoughts_token_count or 0
        # Drain storage charges from caches created for this request.
        storage_token_hours = self._pending_storage_token_hours
        self._pending_storage_token_hours = 0.0
        return Usage(
            input_tokens=usage_metadata.prompt_token_count or 0,
            # Keep reasoning_tokens a subset of output_tokens.
            output_tokens=candidates_tokens + reasoning_tokens,
            cached_input_tokens=usage_metadata.cached_content_token_count or 0,
            reasoning_tokens=reasoning_tokens,
            extra={"cache_storage_token_hours": storage_token_hours} if storage_token_hours else {},
            model_name=model_name,
        )


class GeminiEmbedder(BaseEmbedder):
    supported_hints = ["title", "task_type"]
    _MAX_BATCH_SIZE = 100

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
            provider_name=provider_name or "gemini",
        )

        self._client = google.genai.Client(
            api_key=require_api_key("Gemini", "GEMINI_API_KEY", api_key),
            http_options=google.genai.types.HttpOptions(
                timeout=_GEMINI_REQUEST_TIMEOUT_MS,
                headers=_gemini_identity_headers(),
            ),
        )
        self._tokenizer = GeminiEstimatingTokenizer(client=self._client, model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"gemini/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> GeminiEstimatingTokenizer:
        return self._tokenizer

    @policy(
        [
            retry(
                exceptions=(
                    NotFound,
                    TooManyRequests,
                    ResourceExhausted,
                )
            ),
            retry((ServerError, GenaiServerError), max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        gemini_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}
        vectors: list[Sequence[float]] = []
        input_tokens = 0

        for offset in range(0, len(texts), self._MAX_BATCH_SIZE):
            batch = texts[offset : offset + self._MAX_BATCH_SIZE]

            # One Content per text: gemini-embedding-2 combines a list of plain
            # strings into a single embedding.
            contents: list[google.genai.types.ContentUnionDict] = [
                google.genai.types.Content(parts=[google.genai.types.Part(text=text)])
                for text in batch
            ]

            try:
                response = await self._client.aio.models.embed_content(  # type: ignore
                    model=self.model_name,
                    contents=contents,
                    config=cast(google.genai.types.EmbedContentConfigDict, gemini_api_arguments),
                )
            except TooManyRequests:
                self.logger.error(
                    (
                        "Gemini API rate limit exceeded. Possible reasons:\n"
                        "1. Your account may have insufficient API credits.\n"
                        "2. You may be using a free-tier account with limited request capacity.\n"
                        "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
                        "Recommended actions:\n"
                        "- Check your Gemini API account balance and billing status.\n"
                        "- Review your API usage limits in Google's dashboard.\n"
                        "- For more details on rate limits and usage tiers, visit:\n"
                        "  https://cloud.google.com/docs/quota-and-billing/quotas/quotas-overview"
                    ),
                )
                raise

            vectors.extend(
                data_point.values for data_point in response.embeddings or [] if data_point.values
            )
            usage_metadata = getattr(response, "usage_metadata", None)
            input_tokens += (
                getattr(usage_metadata, "prompt_token_count", None)
                or getattr(usage_metadata, "total_token_count", None)
                or 0
            )

        return EmbeddingResult(
            vectors=vectors,
            usage=UsageInfo(input_tokens=input_tokens, output_tokens=0),
        )


class GeminiTextEmbedding_001(GeminiEmbedder):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            model_name="gemini-embedding-001",
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 2048

    @property
    def dimensions(self) -> int:
        return 3072


def convert_type_annotation_to_gemini_compatible_schema(annotation: Any) -> Any:
    origin = get_origin(annotation)

    if origin is None:
        if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
            enum_values = tuple(member.value for member in annotation)
            if len(enum_values) == 1:
                return Literal[enum_values[0]]
            return Literal.__getitem__(enum_values)

        if inspect.isclass(annotation) and issubclass(annotation, DefaultBaseModel):
            return convert_model_to_gemini_compatible_schema(annotation)

        return annotation

    args = get_args(annotation)
    converted_args = tuple(convert_type_annotation_to_gemini_compatible_schema(arg) for arg in args)

    if origin is Mapping or origin is MappingABC:
        return dict[converted_args] if converted_args else dict  # type: ignore

    if origin is Sequence or origin is SequenceABC:
        return list[converted_args] if converted_args else list  # type: ignore

    if origin is types.UnionType:
        return Union[converted_args]

    if converted_args:
        return origin[converted_args]

    return annotation


def convert_model_to_gemini_compatible_schema(model_cls: type[DefaultBaseModel]) -> type[BaseModel]:
    if hasattr(model_cls, "_conversion_cache"):
        return cast(type[BaseModel], model_cls._conversion_cache)

    new_annotations = {}
    new_fields = {}

    for field_name, field_info in model_cls.model_fields.items():
        converted_annotation = convert_type_annotation_to_gemini_compatible_schema(
            field_info.annotation
        )
        new_annotations[field_name] = converted_annotation

        field_kwargs = {}
        if field_info.default is not None and field_info.default is not FieldInfo:
            field_kwargs["default"] = field_info.default
        elif field_info.default_factory is not None:
            field_kwargs["default_factory"] = field_info.default_factory

        if field_info.description is not None:
            field_kwargs["description"] = field_info.description
        if field_info.title is not None:
            field_kwargs["title"] = field_info.title
        if field_info.examples is not None:
            field_kwargs["examples"] = field_info.examples
        if field_kwargs:
            new_fields[field_name] = Field(**field_kwargs)

    new_model_attrs = {"__annotations__": new_annotations, **new_fields}
    if hasattr(model_cls, "model_config"):
        new_model_attrs["model_config"] = model_cls.model_config

    converted_model = type(f"{model_cls.__name__}Converted", (DefaultBaseModel,), new_model_attrs)
    setattr(model_cls, "_conversion_cache", converted_model)
    return converted_model


class RoutedGeminiSchematicGenerator(GeminiSchematicGenerator[T]):
    @property
    def max_tokens(self) -> int:
        return 1024 * 1024


# Input limits of the Gemini embedding models. Newer models (gemini-embedding-2
# and later) take 8192 tokens; gemini-embedding-001 takes 2048.
_GEMINI_EMBEDDING_MAX_TOKENS = {
    "gemini-embedding-001": 2048,
    "gemini-embedding-2": 8192,
}


class RoutedGeminiEmbedder(GeminiEmbedder):
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


def register_gemini_models(
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
    ) -> GeminiSchematicGenerator[Any]:
        args = validate_model_arguments("gemini", raw_args, {"api_key"})
        generator_type = RoutedGeminiSchematicGenerator[schema]  # type: ignore[valid-type]
        return cast(
            GeminiSchematicGenerator[Any],
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

    async def react(model: str, raw_args: Mapping[str, Any]) -> GeminiReactGenerator:
        args = validate_model_arguments("gemini", raw_args, {"api_key", "service_tier"})
        if "service_tier" in args:
            args["service_tier"] = validate_service_tier("gemini", args["service_tier"])
        return GeminiReactGenerator(model=model, logger=logger, usage_reporter=usage, **args)

    async def embedding(
        model: str,
        raw_args: Mapping[str, Any],
    ) -> RoutedGeminiEmbedder:
        args = validate_model_arguments(
            "gemini",
            raw_args,
            {"api_key", "dimensions", "max_tokens"},
        )
        dimensions = int(args.pop("dimensions", 3072))
        max_tokens = int(args.pop("max_tokens", _GEMINI_EMBEDDING_MAX_TOKENS.get(model, 8192)))
        return RoutedGeminiEmbedder(
            model_name=model,
            logger=logger,
            tracer=tracer,
            meter=meter,
            usage_reporter=usage,
            dimensions=dimensions,
            max_tokens=max_tokens,
            **args,
        )

    async def tokenizer(
        model: str,
        raw_args: Mapping[str, Any],
    ) -> GeminiEstimatingTokenizer:
        args = validate_model_arguments("gemini", raw_args, {"api_key"})
        client = google.genai.Client(api_key=cast(str | None, args.get("api_key")))
        return GeminiEstimatingTokenizer(client, model)

    registry.register_schematic(
        "gemini",
        schematic,
        default=_gemini_schematic_default,
    )
    registry.register(
        "gemini",
        ModelCapability.REACT,
        react,
        default=_gemini_react_default,
    )
    registry.register(
        "gemini",
        ModelCapability.EMBEDDING,
        embedding,
        default="gemini-embedding-2",
    )
    registry.register(
        "gemini",
        ModelCapability.TOKENIZER,
        tokenizer,
        default="gemini-3.1-flash-lite",
    )


def _gemini_schematic_default(route: str, qualifiers: Mapping[str, str]) -> str | ModelConfig:
    if route.startswith("review"):
        return "gemini-3.8-flash"
    elif route.startswith("matching.distill"):
        # Distillation is state-tracking-heavy (which policy clauses apply at
        # THIS point of the conversation), so give the small model a thinking
        # budget instead of relying on its reflexive read.
        return ModelConfig(model="gemini:gemini-3.5-flash-lite", reasoning="medium")
    else:
        return "gemini-3.1-flash-lite"


def _gemini_react_default(route: str, qualifiers: Mapping[str, str]) -> str | ModelConfig:
    if effort := qualifiers.get("effort"):
        if effort == "min":
            return "gemini-3.5-flash-lite"
        if effort in ["low", "medium"]:
            # At its LOW thinking level Flash-Lite often skips its thought channel
            # and writes its deliberation into the reply. MEDIUM keeps that
            # deliberation out of most replies.
            return ModelConfig(
                model="gemini:gemini-3.5-flash-lite",
                reasoning=ReasoningConfig(effort="medium", visibility="summary"),
            )
    return "gemini-3.8-flash"
