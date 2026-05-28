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
import enum
import hashlib
import inspect
import os
import time
import types
import uuid
from datetime import datetime, timedelta, timezone
from google.api_core.exceptions import NotFound, TooManyRequests, ResourceExhausted, ServerError
from google.genai.errors import ClientError
import google.genai  # type: ignore
import google.genai.types  # type: ignore
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from typing import Any, AsyncIterator, Literal, Mapping, Optional, Sequence, Union, cast
from typing_extensions import get_args, get_origin, override
from pydantic import BaseModel, Field, ValidationError
from pydantic.fields import FieldInfo

from parlant.core.common import DefaultBaseModel
from parlant.adapters.nlp.common import record_llm_metrics
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.meter import Meter
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.service import (
    EmbedderHints,
    ModelSize,
    NLPService,
    ReactGeneratorHints,
    SchematicGeneratorHints,
    StreamingTextGeneratorHints,
)
from parlant.core.nlp.embedding import BaseEmbedder, Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    FallbackSchematicGenerator,
    SchematicGenerationResult,
    StreamingTextGenerator,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.react import (
    CacheConfig,
    FinishReason,
    Message,
    ReactGenerator,
    ReasoningConfig,
    ReasoningPart,
    Role,
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
from parlant.core.loggers import Logger
from parlant.core.tracer import Tracer
from parlant.core.health import HealthReporter

RATE_LIMIT_ERROR_MESSAGE = (
    "Google API rate limit exceeded.\n\n"
    "Possible reasons:\n"
    "1. Insufficient API credits in your account.\n"
    "2. Using a free-tier account with limited request capacity.\n"
    "3. Exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your Google API account balance and billing status.\n"
    "- Review your API usage limits in the Google Cloud Console.\n"
    "- Learn more about quotas and limits:\n"
    "  https://cloud.google.com/docs/quota-and-billing/quotas/quotas-overview"
)


class GoogleEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, client: google.genai.Client, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        model_approximation = {
            "gemini-embedding-001": "gemini-2.5-flash",
        }.get(self._model_name, self._model_name)

        result = await self._client.aio.models.count_tokens(
            model=model_approximation,
            contents=prompt,
        )

        return int(result.total_tokens or 0)


class GeminiSchematicGenerator(BaseSchematicGenerator[T]):
    supported_hints = ["temperature", "thinking_config"]

    def __init__(
        self,
        model_name: str,
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

        self._client = google.genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        self._tokenizer = GoogleEstimatingTokenizer(client=self._client, model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"google/{self.model_name}"

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
                )
            ),
            retry(ServerError, max_exceptions=2, wait_times=(1.0, 5.0)),
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
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        gemini_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        fd = self._get_schema_function_declaration()

        config = google.genai.types.GenerateContentConfig(
            tools=[google.genai.types.Tool(function_declarations=[fd])],
            tool_config=google.genai.types.ToolConfig(
                function_calling_config=google.genai.types.FunctionCallingConfig(
                    mode=google.genai.types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=[fd.name],
                )
            ),
            **gemini_api_arguments,  # type: ignore
        )

        t_start = time.time()
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
        except TooManyRequests:
            self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise

        t_end = time.time()

        assert response.candidates
        assert response.candidates[0].content
        assert response.candidates[0].content.parts
        assert response.candidates[0].content.parts[0].function_call
        assert response.candidates[0].content.parts[0].function_call.args

        json_result = (
            response.candidates[0].content.parts[0].function_call.args.get("log_data", {}) or {}
        )

        if response.usage_metadata:
            self.logger.trace(response.usage_metadata.model_dump_json(indent=2))

        try:
            model_content = self.schema.model_validate(json_result)

            await record_llm_metrics(
                self.meter,
                self.model_name,
                schema_name=self.schema.__name__,
                input_tokens=response.usage_metadata.prompt_token_count or 0
                if response.usage_metadata
                else 0,
                output_tokens=response.usage_metadata.candidates_token_count or 0
                if response.usage_metadata
                else 0,
                cached_input_tokens=response.usage_metadata.cached_content_token_count or 0
                if response.usage_metadata
                else 0,
            )

            return SchematicGenerationResult(
                content=model_content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=response.usage_metadata.prompt_token_count or 0,
                        output_tokens=response.usage_metadata.candidates_token_count or 0,
                        extra={
                            "cached_input_tokens": (
                                response.usage_metadata.cached_content_token_count or 0
                                if response.usage_metadata
                                else 0
                            )
                            or 0
                        },
                    )
                    if response.usage_metadata
                    else UsageInfo(input_tokens=0, output_tokens=0, extra={}),
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


class Gemini_2_0_Flash(GeminiSchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="gemini-2.0-flash",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_2_0_Flash_Lite(GeminiSchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="gemini-2.0-flash-lite-preview-02-05",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_2_5_Flash(GeminiSchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="gemini-2.5-flash",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
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
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="gemini-2.5-flash-lite",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
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
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="gemini-2.5-pro",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_3_1_Flash_Lite(GeminiSchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="gemini-3.1-flash-lite",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_3_5_Flash(GeminiSchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="gemini-3.5-flash",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


# The key under which Gemini's per-part ``thought_signature`` is preserved in a
# canonical Part's ``provider_data``. It MUST round-trip verbatim, or replaying
# tool-calling history triggers a 400 "missing thought_signature".
GEMINI_THOUGHT_SIGNATURE_KEY = "gemini_thought_signature"


def _signature_to_bytes(signature: Union[str, bytes, None]) -> Optional[bytes]:
    if signature is None:
        return None
    if isinstance(signature, bytes):
        return signature
    return signature.encode("utf-8")


class GeminiReactGenerator(ReactGenerator):
    """A ReAct generator backed by Google Gemini (google-genai).

    Implements the ``ReactGenerator`` provider seam: ``_encode`` builds the
    google-genai request, ``_raw_stream`` opens the streaming call, and
    ``_decode`` folds each native chunk into the shared ``TurnBuilder``.

    Gemini ``thought_signature`` values are preserved verbatim on each Part's
    ``provider_data`` (under :data:`GEMINI_THOUGHT_SIGNATURE_KEY`) so that
    multi-step tool-calling history replays without 400 errors.

    Caching (:class:`CacheConfig` + :attr:`Message.cache`): a marked prefix is
    turned into an explicit Gemini ``CachedContent`` resource, created lazily and
    reused across calls that share the same prefix. Cached resources auto-expire
    at their TTL (Google's default is ~1h), so cleanup is not required for
    correctness; call :meth:`aclose` to delete the ones this generator created
    early (e.g. on shutdown) for cost control.
    """

    _ROLE_MAP = {Role.USER: "user", Role.ASSISTANT: "model", Role.TOOL: "tool"}
    # Don't reuse a cached prefix that's about to expire mid-request.
    _CACHE_REUSE_MARGIN = timedelta(seconds=30)

    def __init__(
        self,
        *,
        model: str = "gemini-3.1-flash-lite",
        logger: Logger,
        cache: Optional[CacheConfig] = None,
        client: Optional[google.genai.Client] = None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, cache=cache)
        self._logger = logger
        self._client = client or google.genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY")
        )
        # Caches this generator created and may reuse: key -> (resource name, expiry).
        self._managed_caches: dict[str, tuple[str, datetime]] = {}
        # Prefix keys we know can't be cached (e.g. below the provider minimum),
        # so we don't re-attempt creation on every call.
        self._uncacheable_keys: set[str] = set()
        self._cache_lock = asyncio.Lock()

    @property
    def id(self) -> str:
        return f"google/{self.model}"

    # ---- provider seam -----------------------------------------------------

    @override
    def _encode(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec],
        tool_choice: ToolChoice,
        *,
        reasoning: ReasoningConfig,
    ) -> dict[str, Any]:
        system_chunks: list[str] = []

        # Caching is positional: everything up to and including the last message
        # with a cache_key is the stable prefix to cache; the rest is the live
        # suffix sent on each call. The key names the cache for reuse.
        cache_split = -1
        cache_key: Optional[str] = None
        system_cache_key: Optional[str] = None
        non_system: list[Message] = []
        for message in history:
            if message.role == Role.SYSTEM:
                if message.text:
                    system_chunks.append(message.text)
                if self.cache.enabled and message.cache_key is not None:
                    system_cache_key = message.cache_key
                continue
            if self.cache.enabled and message.cache_key is not None:
                cache_split = len(non_system)
                cache_key = message.cache_key
            non_system.append(message)

        contents = [self._encode_message(message) for message in non_system]
        system_instruction = "\n\n".join(chunk for chunk in system_chunks if chunk) or None

        tool_block: Optional[list[google.genai.types.Tool]] = None
        tool_config: Optional[google.genai.types.ToolConfig] = None
        if tools:
            tool_block = [
                google.genai.types.Tool(
                    function_declarations=[self._encode_tool(spec) for spec in tools]
                )
            ]
            tool_config = self._encode_tool_choice(tool_choice)

        thinking_config = self._encode_thinking(reasoning) if reasoning.enabled else None

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
            "model": self.model,
            "system_instruction": system_instruction,
            "tools": tool_block,
            "tool_config": tool_config,
            "thinking_config": thinking_config,
            "all_contents": contents,
            "prefix_contents": prefix_contents,  # cache this (None => no managed cache)
            "suffix_contents": suffix_contents,  # send this when a cache is used
            "cache_key": cache_key,  # caller-provided reuse identity for the prefix
            "explicit_cache_name": explicit_cache,
        }

    def _encode_message(self, message: Message) -> google.genai.types.Content:
        parts = [self._encode_part(part) for part in message.parts]
        return google.genai.types.Content(
            role=self._ROLE_MAP[message.role],
            parts=[p for p in parts if p is not None],
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

    def _encode_thinking(self, reasoning: ReasoningConfig) -> google.genai.types.ThinkingConfig:
        thinking_kwargs: dict[str, Any] = {
            "include_thoughts": reasoning.visibility != "none",
        }
        if reasoning.budget_tokens is not None:
            thinking_kwargs["thinking_budget"] = reasoning.budget_tokens
        return google.genai.types.ThinkingConfig(**thinking_kwargs)

    @override
    async def _raw_stream(self, request: Any) -> AsyncIterator[Any]:
        config_kwargs: dict[str, Any] = {}
        if request["tool_config"] is not None:
            config_kwargs["tool_config"] = request["tool_config"]
        if request["thinking_config"] is not None:
            config_kwargs["thinking_config"] = request["thinking_config"]
        if request["tools"] is not None:
            config_kwargs["tools"] = request["tools"]

        cached_content_name: Optional[str] = request["explicit_cache_name"]
        contents = request["all_contents"]

        if cached_content_name is not None:
            # Reuse a caller-provided cache: assume it holds the system prompt.
            contents = request["all_contents"]
        elif request["prefix_contents"] is not None:
            # Managed cache: cache the marked prefix (system + prefix contents),
            # then send only the live suffix referencing it.
            cached_content_name = await self._get_or_create_cache(
                model=request["model"],
                system_instruction=request["system_instruction"],
                prefix_contents=request["prefix_contents"],
                cache_key=request["cache_key"],
            )
            if cached_content_name is not None:
                contents = request["suffix_contents"]
            elif request["system_instruction"] is not None:
                # Caching unavailable (e.g. prefix below the provider minimum):
                # fall back to sending the full content inline, uncached.
                config_kwargs["system_instruction"] = request["system_instruction"]
        elif request["system_instruction"] is not None:
            config_kwargs["system_instruction"] = request["system_instruction"]

        if cached_content_name is not None:
            config_kwargs["cached_content"] = cached_content_name

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
        except TooManyRequests:
            self._logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise

    # ---- explicit caching --------------------------------------------------

    def _cache_key(
        self,
        cache_key: str,
        model: str,
        system_instruction: Optional[str],
        prefix_contents: list[google.genai.types.Content],
    ) -> str:
        # Fold the caller's key together with the actual content: the key gives
        # intentional identity, the content hash guarantees a key reused after an
        # edited prefix never serves stale content.
        hasher = hashlib.sha256()
        hasher.update(cache_key.encode("utf-8"))
        hasher.update(model.encode("utf-8"))
        hasher.update((system_instruction or "").encode("utf-8"))
        for content in prefix_contents:
            hasher.update(content.model_dump_json(exclude_none=True).encode("utf-8"))
        return hasher.hexdigest()

    async def _get_or_create_cache(
        self,
        *,
        model: str,
        system_instruction: Optional[str],
        prefix_contents: list[google.genai.types.Content],
        cache_key: str,
    ) -> Optional[str]:
        """Return a usable cache resource name, or ``None`` if caching is
        unavailable for this prefix. Caching is an optimization, so this never
        raises for cache problems — the caller falls back to an inline request."""
        key = self._cache_key(cache_key, model, system_instruction, prefix_contents)

        async with self._cache_lock:
            if key in self._uncacheable_keys:
                return None

            existing = self._managed_caches.get(key)
            if existing is not None:
                name, expiry = existing
                # Reuse only while comfortably within the resource's lifetime.
                if expiry - datetime.now(timezone.utc) > self._CACHE_REUSE_MARGIN:
                    return name

            config_kwargs: dict[str, Any] = {"display_name": cache_key}
            if prefix_contents:
                config_kwargs["contents"] = prefix_contents
            if system_instruction is not None:
                config_kwargs["system_instruction"] = system_instruction
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
                self._uncacheable_keys.add(key)
                return None
            except Exception as exc:  # noqa: BLE001 - transient: degrade, retry later
                self._logger.warning(
                    f"Gemini cache creation failed for key '{cache_key}' "
                    f"({exc}); proceeding without caching."
                )
                return None

            assert cached.name and cached.expire_time
            self._managed_caches[key] = (cached.name, cached.expire_time)
            return cached.name

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
        events: list[StreamEvent] = []

        candidate = raw_event.candidates[0] if raw_event.candidates else None
        if candidate is not None:
            if candidate.finish_reason is not None:
                builder.finish_reason = self._map_finish_reason(candidate.finish_reason)

            content = candidate.content
            for part in content.parts if content and content.parts else []:
                events.extend(self._decode_part(part, builder))

        if raw_event.usage_metadata is not None:
            builder.usage = self._decode_usage(raw_event.usage_metadata)

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

            builder.text_delta(part.text, provider_data=provider_data)
            return [TextDelta(text=part.text)] if part.text else []

        return []

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

    def _decode_usage(self, usage_metadata: Any) -> Usage:
        candidates_tokens = usage_metadata.candidates_token_count or 0
        reasoning_tokens = usage_metadata.thoughts_token_count or 0
        return Usage(
            input_tokens=usage_metadata.prompt_token_count or 0,
            # Keep reasoning_tokens a subset of output_tokens.
            output_tokens=candidates_tokens + reasoning_tokens,
            cached_input_tokens=usage_metadata.cached_content_token_count or 0,
            reasoning_tokens=reasoning_tokens,
        )


class GoogleEmbedder(BaseEmbedder):
    supported_hints = ["title", "task_type"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        health_reporter: HealthReporter,
    ) -> None:
        super().__init__(logger, tracer, meter, model_name, health_reporter)

        self._client = google.genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self._tokenizer = GoogleEstimatingTokenizer(client=self._client, model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"google/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> GoogleEstimatingTokenizer:
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
            retry(ServerError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        gemini_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        try:
            response = await self._client.aio.models.embed_content(  # type: ignore
                model=self.model_name,
                contents=texts,  # type: ignore
                config=cast(google.genai.types.EmbedContentConfigDict, gemini_api_arguments),
            )
        except TooManyRequests:
            self.logger.error(
                (
                    "Google API rate limit exceeded. Possible reasons:\n"
                    "1. Your account may have insufficient API credits.\n"
                    "2. You may be using a free-tier account with limited request capacity.\n"
                    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
                    "Recommended actions:\n"
                    "- Check your Google API account balance and billing status.\n"
                    "- Review your API usage limits in Google's dashboard.\n"
                    "- For more details on rate limits and usage tiers, visit:\n"
                    "  https://cloud.google.com/docs/quota-and-billing/quotas/quotas-overview"
                ),
            )
            raise

        vectors = [
            data_point.values for data_point in response.embeddings or [] if data_point.values
        ]
        return EmbeddingResult(vectors=vectors)


class GeminiTextEmbedding_001(GoogleEmbedder):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="gemini-embedding-001",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 2048

    @property
    def dimensions(self) -> int:
        return 3072


class GeminiService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("GEMINI_API_KEY"):
            return """\
You're using the GEMINI NLP service, but GEMINI_API_KEY is not set.
Please set GEMINI_API_KEY in your environment before running Parlant.
"""

        return None

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        health_reporter: HealthReporter,
    ) -> None:
        self.logger = logger
        self._tracer = tracer
        self._meter = meter

        self._health_reporter = health_reporter

        self.logger.info("Initialized GeminiService")

    @property
    @override
    def supports_streaming(self) -> bool:
        return False

    @override
    async def get_streaming_text_generator(
        self, hints: StreamingTextGeneratorHints = {}
    ) -> StreamingTextGenerator:
        raise NotImplementedError("Streaming is not supported. Check supports_streaming first.")

    @property
    @override
    def supports_react(self) -> bool:
        return True

    @override
    async def get_react_generator(self, hints: ReactGeneratorHints = {}) -> ReactGenerator:
        model = {
            ModelSize.NANO: "gemini-3.1-flash-lite",
            ModelSize.MINI: "gemini-2.5-flash",
            ModelSize.LARGE: "gemini-2.5-pro",
        }.get(hints.get("model_size", ModelSize.AUTO), "gemini-3.1-flash-lite")
        return GeminiReactGenerator(model=model, logger=self.logger)

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> GeminiSchematicGenerator[T]:
        match hints.get("model_size", ModelSize.AUTO):
            case ModelSize.NANO:
                return Gemini_3_1_Flash_Lite[t](  # type: ignore
                    self.logger, self._tracer, self._meter, self._health_reporter
                )
            case ModelSize.MINI:
                return Gemini_3_1_Flash_Lite[t](  # type: ignore
                    self.logger, self._tracer, self._meter, self._health_reporter
                )
            case ModelSize.LARGE:
                return Gemini_3_5_Flash[t](  # type: ignore
                    self.logger, self._tracer, self._meter, self._health_reporter
                )
            case _:
                return FallbackSchematicGenerator[t](  # type: ignore
                    Gemini_3_1_Flash_Lite[t](  # type: ignore
                        self.logger, self._tracer, self._meter, self._health_reporter
                    ),
                    Gemini_3_5_Flash[t](  # type: ignore
                        self.logger, self._tracer, self._meter, self._health_reporter
                    ),
                    logger=self.logger,
                )

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        return GeminiTextEmbedding_001(
            self.logger, self._tracer, self._meter, self._health_reporter
        )

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()


def convert_type_annotation_to_gemini_compatible_schema(annotation: Any) -> Any:
    origin = get_origin(annotation)

    # If not a generic type, check if it's a BaseModel or Enum
    if origin is None:
        # If it's an Enum class, convert to Literal of its values
        if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
            enum_values = tuple(member.value for member in annotation)
            if len(enum_values) == 1:
                return Literal[enum_values[0]]
            return Literal.__getitem__(enum_values)

        # If it's a BaseModel class, recursively convert it
        if inspect.isclass(annotation) and issubclass(annotation, DefaultBaseModel):
            return convert_model_to_gemini_compatible_schema(annotation)

        return annotation

    # Get the type arguments
    args = get_args(annotation)

    # Convert nested types recursively
    converted_args = tuple(convert_type_annotation_to_gemini_compatible_schema(arg) for arg in args)

    # Check if origin is Mapping or Sequence
    if origin is Mapping or origin is MappingABC:
        return dict[converted_args] if converted_args else dict  # type: ignore

    if origin is Sequence or origin is SequenceABC:
        return list[converted_args] if converted_args else list  # type: ignore

    # Handle UnionType (X | Y syntax) - not subscriptable!
    if origin is types.UnionType:
        return Union[converted_args]

    # For other generic types, preserve the origin with converted args
    if converted_args:
        return origin[converted_args]

    return annotation


def convert_model_to_gemini_compatible_schema(model_cls: type[DefaultBaseModel]) -> type[BaseModel]:
    """
    Create a new BaseModel class with converted annotations.
    Returns a new class without modifying the original.
    """
    # Avoid infinite recursion - check if already converted
    if hasattr(model_cls, "_conversion_cache"):
        return cast(type[BaseModel], model_cls._conversion_cache)

    # Build new annotations
    new_annotations = {}
    new_fields = {}

    for field_name, field_info in model_cls.model_fields.items():
        # Convert the annotation
        converted_annotation = convert_type_annotation_to_gemini_compatible_schema(
            field_info.annotation
        )
        new_annotations[field_name] = converted_annotation

        # Preserve field metadata (default, description, etc.)
        # We need to recreate the field with the new annotation
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

        # Add other field properties as needed
        if field_kwargs:
            new_fields[field_name] = Field(**field_kwargs)

    # Create new model class
    new_model_attrs = {"__annotations__": new_annotations, **new_fields}

    # Preserve model config if present
    if hasattr(model_cls, "model_config"):
        new_model_attrs["model_config"] = model_cls.model_config

    # Create the new class
    converted_model = type(f"{model_cls.__name__}Converted", (DefaultBaseModel,), new_model_attrs)

    # Cache the conversion to avoid infinite recursion
    setattr(model_cls, "_conversion_cache", converted_model)

    return converted_model
