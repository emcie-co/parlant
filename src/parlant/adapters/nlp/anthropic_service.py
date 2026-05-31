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
from anthropic import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AsyncAnthropic,
    InternalServerError,
    RateLimitError,
)  # type: ignore
from typing import Any, AsyncIterator, Mapping, Optional, Sequence
from typing_extensions import override
import jsonfinder  # type: ignore
import json
import os

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.adapters.nlp.hugging_face import JinaAIEmbedder
from parlant.core.engines.alpha.canned_response_generator import CannedResponseSelectionSchema
from parlant.core.engines.alpha.guideline_matching.generic.disambiguation_batch import (
    DisambiguationGuidelineMatchesSchema,
)

from parlant.core.engines.alpha.guideline_matching.generic.journey.journey_backtrack_node_selection import (
    JourneyBacktrackNodeSelectionSchema,
)
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.tracer import Tracer
from parlant.core.meter import Meter
from parlant.core.nlp.embedding import Embedder, EmbedderHints
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.loggers import Logger
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.service import (
    ModelSize,
    NLPService,
    SchematicGeneratorHints,
    StreamingTextGeneratorHints,
)
from parlant.core.nlp.generation import StreamingTextGenerator
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.react import (
    CacheConfig,
    FinishReason,
    Message,
    ReactGenerator,
    ReactGeneratorHints,
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
from parlant.core.health import HealthReporter


class AnthropicEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, client: AsyncAnthropic, model_name: str) -> None:
        self._client = client
        self.model_name = model_name

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        result = await self._client.messages.count_tokens(
            model=self.model_name,
            messages=[{"role": "assistant", "content": prompt}],
        )

        return result.input_tokens  # type: ignore[no-any-return]


class AnthropicAISchematicGenerator(BaseSchematicGenerator[T]):
    supported_hints = ["temperature"]

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

        self._client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._estimating_tokenizer = AnthropicEstimatingTokenizer(self._client, model_name)

    @property
    @override
    def id(self) -> str:
        return f"anthropic/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> AnthropicEstimatingTokenizer:
        return self._estimating_tokenizer

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
    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self.logger.scope(f"Anthropic LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        anthropic_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        t_start = time.time()
        try:
            response = await self._client.messages.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                max_tokens=4096,
                **anthropic_api_arguments,
            )
        except RateLimitError:
            self.logger.error(
                (
                    "Anthropic API rate limit exceeded. Possible reasons:\n"
                    "1. Your account may have insufficient API credits.\n"
                    "2. You may be using a free-tier account with limited request capacity.\n"
                    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
                    "Recommended actions:\n"
                    "- Check your Anthropic account balance and billing status.\n"
                    "- Review your API usage limits in Anthropic's dashboard.\n"
                    "- For more details on rate limits and usage tiers, visit:\n"
                    "  https://docs.anthropic.com/claude/reference/rate-limits \n"
                ),
            )
            raise

        t_end = time.time()

        if response.usage:
            self.logger.trace(response.usage.model_dump_json(indent=2))

        raw_content = response.content[0].text

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
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            return SchematicGenerationResult(
                content=model_content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                    ),
                ),
            )
        except ValidationError:
            self.logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class Claude_Sonnet_3_5(AnthropicAISchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="claude-3-5-sonnet-20241022",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 200 * 1024


class Claude_Sonnet_4(AnthropicAISchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="claude-sonnet-4-20250514",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 200 * 1024


class Claude_Opus_4_1(AnthropicAISchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        super().__init__(
            model_name="claude-opus-4-1-20250805",
            logger=logger,
            tracer=tracer,
            meter=meter,
            health_reporter=health_reporter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 200 * 1024


# The key under which an Anthropic content block (thinking / tool_use) is
# preserved verbatim on a canonical Part's provider_data so it can be replayed.
# The thinking block's signature MUST round-trip, or replaying a turn that had
# thinking before a tool_use errors with 400.
ANTHROPIC_BLOCK_KEY = "anthropic_block"

ANTHROPIC_RATE_LIMIT_ERROR_MESSAGE = (
    "Anthropic API rate limit exceeded. Check your plan and billing, and review "
    "https://docs.anthropic.com/en/api/rate-limits"
)


class _AnthropicFinal:
    """Sentinel wrapping the fully-accumulated message, yielded after the raw
    stream so _decode can build the authoritative turn from complete blocks."""

    def __init__(self, message: Any) -> None:
        self.message = message


class AnthropicReactGenerator(ReactGenerator):
    """A ReAct generator backed by the Anthropic Messages API.

    Implements the ``ReactGenerator`` provider seam against the streaming
    Messages API. Thinking blocks (with their signature) and tool_use blocks
    round-trip verbatim via each Part's ``provider_data`` (under
    :data:`ANTHROPIC_BLOCK_KEY`) — Anthropic rejects a replayed turn whose
    thinking block before a tool_use is missing or altered.

    Caching is positional: a :attr:`Message.cache_key` marks the prefix whose
    last block gets ``cache_control={"type": "ephemeral"}``.

    Provider constraints honored by callers (not worked around silently):
    - Extended thinking is incompatible with a forced ``tool_choice``
      (``"required"`` / ``{"name": ...}``); Anthropic 400s on that combination.
      Use ``"auto"`` with reasoning enabled.
    - ``ReasoningConfig.effort`` has no effect: Anthropic controls thinking via
      ``budget_tokens`` (the mirror of OpenAI ignoring ``budget_tokens``).
    - Claude 4 returns only a SUMMARY of its thinking (never verbatim, though it
      bills the full thinking tokens). ``visibility`` maps to the ``display``
      knob: ``"none"`` -> ``"omitted"`` (reason internally, return nothing);
      ``"summary"``/``"full"`` -> ``"summarized"`` (no verbatim option exists).
    - Anthropic does not report a separate thinking-token count, so
      :attr:`Usage.reasoning_tokens` is always 0 for this provider.
    """

    _ROLE = {Role.USER: "user", Role.ASSISTANT: "assistant", Role.TOOL: "user"}

    # Maps ``ReasoningConfig.effort`` to a thinking-token budget for *manual*
    # thinking mode (Sonnet 4.5 / Haiku 4.5). ``"minimal"`` means "no thinking" —
    # the adapter skips the ``thinking`` block entirely so the model runs in
    # standard mode with zero thinking tokens spent.
    _EFFORT_TO_BUDGET: dict[str, int] = {
        "minimal": 0,  # sentinel — skip thinking block
        "low": 2048,
        "medium": 8192,
        "high": 16384,
    }

    # Mapping from canonical ModelSize to a concrete Anthropic model id, used
    # to resolve per-call ``hints`` overrides on ``_encode``.
    _MODEL_BY_SIZE: dict[ModelSize, str] = {
        ModelSize.SMALL: "claude-haiku-4-5-20251001",
        ModelSize.MEDIUM: "claude-sonnet-4-6",
        ModelSize.LARGE: "claude-opus-4-8",
    }

    def __init__(
        self,
        *,
        model: str = "claude-haiku-4-5-20251001",
        logger: Logger,
        cache: Optional[CacheConfig] = None,
        client: Optional[AsyncAnthropic] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 8192,
    ) -> None:
        super().__init__(model=model, cache=cache)
        self._logger = logger
        self._client = client or AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self._max_tokens = max_tokens

    @property
    def id(self) -> str:
        return f"anthropic/{self.model}"

    def _resolve_model(self, hints: ReactGeneratorHints) -> str:
        """Return the model id for this call, applying ``hints['model_size']``
        if present (falling back to the generator's default)."""
        size = hints.get("model_size", ModelSize.AUTO)
        return self._MODEL_BY_SIZE.get(size, self.model)

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
        system_chunks: list[str] = []

        cache_split = -1
        system_marked = False
        non_system: list[Message] = []
        for message in history:
            if message.role == Role.SYSTEM:
                if message.text:
                    system_chunks.append(message.text)
                if self.cache.enabled and message.cache_key is not None:
                    system_marked = True
                continue
            if self.cache.enabled and message.cache_key is not None:
                cache_split = len(non_system)
            non_system.append(message)

        messages = [
            self._encode_message(message, cache=(index == cache_split))
            for index, message in enumerate(non_system)
        ]

        max_tokens = self._max_tokens
        resolved_model = self._resolve_model(hints)
        request: dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        system_text = "\n\n".join(chunk for chunk in system_chunks if chunk)
        if system_text:
            if system_marked:
                # A marked system message caches the system prefix; Anthropic
                # caches system via cache_control on a system text block.
                request["system"] = [
                    {"type": "text", "text": system_text, "cache_control": self._cache_control()}
                ]
            else:
                request["system"] = system_text

        if tools:
            request["tools"] = [self._encode_tool(spec) for spec in tools]
            request["tool_choice"] = self._encode_tool_choice(tool_choice)

        # Claude 4 only ever returns a SUMMARY of its thinking (it is billed
        # the full thinking tokens). ``display`` is the visibility knob:
        # "omitted" reasons internally without returning the block; "summarized"
        # returns the summary. There is no verbatim option, so "full" maps to
        # "summarized" (the closest Anthropic offers).
        display = "omitted" if reasoning.visibility == "none" else "summarized"

        if self._uses_adaptive_thinking(resolved_model):
            # Opus 4.6+/4.7+ exclusively use adaptive thinking — they never run
            # without a thinking block, so "minimal" maps to the lowest level
            # rather than disabling. Effort routes through ``output_config``.
            request["thinking"] = {"type": "adaptive", "display": display}
            request["output_config"] = {"effort": self._map_effort(reasoning.effort)}
        else:
            # Sonnet 4.5 / Haiku 4.5 use manual budgeted thinking. "minimal"
            # means "no thinking" — skip the block entirely so the model runs
            # in standard mode with zero thinking tokens spent.
            budget = self._EFFORT_TO_BUDGET[reasoning.effort]
            if budget > 0:
                request["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                    "display": display,
                }
                # Anthropic requires max_tokens > budget_tokens; leave headroom
                # for the visible answer after the thinking budget.
                request["max_tokens"] = max(max_tokens, budget + 2048)
        # NOTE: a ``thinking`` block is rejected by Anthropic alongside a
        # forced ``tool_choice`` ("any"/"tool"); callers must use "auto" when
        # effort > "minimal" on Sonnet/Haiku or on any Opus 4.6+ call.

        return request

    @staticmethod
    def _uses_adaptive_thinking(model: str) -> bool:
        """Opus 4.x exclusively uses adaptive thinking (4.7+ rejects ``enabled``
        outright; 4.5/4.6 deprecate it). Sonnet and Haiku 4.5 still use the
        manual ``enabled`` shape."""
        return model.startswith("claude-opus-4")

    @staticmethod
    def _map_effort(effort: str) -> str:
        """Map ``ReasoningConfig.effort`` to Anthropic's adaptive effort levels
        ("low" | "medium" | "high"). Adaptive thinking can't be disabled, so
        our ``"minimal"`` collapses to ``"low"``."""
        return "low" if effort == "minimal" else effort

    def _encode_message(self, message: Message, *, cache: bool) -> dict[str, Any]:
        blocks = self._encode_blocks(message)
        if cache and blocks:
            blocks[-1] = {**blocks[-1], "cache_control": self._cache_control()}
        return {"role": self._ROLE[message.role], "content": blocks}

    def _encode_blocks(self, message: Message) -> list[dict[str, Any]]:
        if message.role == Role.TOOL:
            return [
                {
                    "type": "tool_result",
                    "tool_use_id": part.call_id,
                    "content": part.content
                    if isinstance(part.content, str)
                    else json.dumps(part.content),
                    "is_error": part.is_error,
                }
                for part in message.parts
                if isinstance(part, ToolResultPart)
            ]

        if message.role == Role.USER:
            return [
                {"type": "text", "text": part.text}
                for part in message.parts
                if isinstance(part, TextPart)
            ]

        # ASSISTANT: replay raw blocks verbatim (preserving thinking signatures),
        # reconstruct where a raw block is absent.
        blocks: list[dict[str, Any]] = []
        for part in message.parts:
            raw_block = part.provider_data.get(ANTHROPIC_BLOCK_KEY)
            if raw_block is not None:
                blocks.append(dict(raw_block))
                continue
            if isinstance(part, TextPart) and part.text:
                blocks.append({"type": "text", "text": part.text})
            elif isinstance(part, ToolCallPart):
                blocks.append(
                    {"type": "tool_use", "id": part.id, "name": part.name, "input": part.args}
                )
            elif isinstance(part, ReasoningPart) and part.signature is not None:
                blocks.append(
                    {"type": "thinking", "thinking": part.text, "signature": part.signature}
                )
        return blocks

    def _cache_control(self) -> dict[str, Any]:
        control: dict[str, Any] = {"type": "ephemeral"}
        if self.cache.ttl is not None:
            control["ttl"] = "1h" if self.cache.ttl.total_seconds() > 300 else "5m"
        return control

    def _encode_tool(self, spec: ToolSpec) -> dict[str, Any]:
        return {
            "name": spec.name,
            "description": spec.description,
            "input_schema": self._to_anthropic_schema(spec.json_schema()),
        }

    def _to_anthropic_schema(self, schema: Mapping[str, Any]) -> dict[str, Any]:
        """Anthropic's input_schema is JSON Schema and (like OpenAI) ignores
        OpenAPI's ``"nullable": true``; nullability must be a ``"null"`` member
        of ``type``. Translate recursively."""
        result = {key: value for key, value in schema.items() if key != "nullable"}
        if schema.get("nullable"):
            current = result.get("type")
            if isinstance(current, str):
                result["type"] = [current, "null"]
            elif isinstance(current, list) and "null" not in current:
                result["type"] = [*current, "null"]
        if isinstance(result.get("properties"), dict):
            result["properties"] = {
                name: self._to_anthropic_schema(sub) for name, sub in result["properties"].items()
            }
        if isinstance(result.get("items"), dict):
            result["items"] = self._to_anthropic_schema(result["items"])
        return result

    def _encode_tool_choice(self, tool_choice: ToolChoice) -> dict[str, Any]:
        if isinstance(tool_choice, Mapping):
            return {"type": "tool", "name": tool_choice.get("name")}
        return {
            "auto": {"type": "auto"},
            "none": {"type": "none"},
            "required": {"type": "any"},
        }[tool_choice]

    @override
    async def _raw_stream(self, request: Any) -> AsyncIterator[Any]:
        try:
            async with self._client.messages.stream(**request) as stream:
                async for event in stream:
                    yield event
                yield _AnthropicFinal(await stream.get_final_message())
        except RateLimitError:
            self._logger.error(ANTHROPIC_RATE_LIMIT_ERROR_MESSAGE)
            raise

    @override
    def _decode(self, raw_event: Any, builder: TurnBuilder) -> list[StreamEvent]:
        if isinstance(raw_event, _AnthropicFinal):
            return self._decode_final(raw_event.message, builder)

        event_type = raw_event.type

        if event_type == "content_block_start" and raw_event.content_block.type == "tool_use":
            block = raw_event.content_block
            return [ToolCallStarted(id=block.id, name=block.name)]

        if event_type == "content_block_delta":
            delta = raw_event.delta
            if delta.type == "text_delta":
                return [TextDelta(text=delta.text)]
            if delta.type == "thinking_delta":
                return [ReasoningDelta(text=delta.thinking)]

        return []

    def _decode_final(self, message: Any, builder: TurnBuilder) -> list[StreamEvent]:
        for block in message.content:
            raw_block = block.model_dump(exclude_none=True)
            if block.type == "text":
                builder.text_delta(block.text)
            elif block.type == "thinking":
                builder.reasoning_delta(
                    block.thinking,
                    signature=block.signature,
                    visibility="summary",  # Claude 4 returns a summary, not verbatim
                    provider_data={ANTHROPIC_BLOCK_KEY: raw_block},
                )
            elif block.type == "redacted_thinking":
                builder.reasoning_delta("", provider_data={ANTHROPIC_BLOCK_KEY: raw_block})
            elif block.type == "tool_use":
                builder.tool_call(
                    block.id,
                    name=block.name,
                    args=dict(block.input or {}),
                    provider_data={ANTHROPIC_BLOCK_KEY: raw_block},
                )

        builder.usage = self._decode_usage(message.usage, getattr(message, "model", "") or "")
        builder.finish_reason = self._map_finish_reason(message.stop_reason)
        return []

    def _decode_usage(self, usage: Any, model_name: str) -> Usage:
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        return Usage(
            # input_tokens excludes cached/created prompt tokens; fold them back
            # in so cached_input_tokens is a subset of input_tokens.
            input_tokens=(usage.input_tokens or 0) + cache_read + cache_creation,
            output_tokens=usage.output_tokens or 0,  # includes thinking tokens
            cached_input_tokens=cache_read,
            reasoning_tokens=0,  # Anthropic does not report thinking tokens separately
            model_name=model_name,
        )

    def _map_finish_reason(self, stop_reason: Optional[str]) -> FinishReason:
        if stop_reason == "max_tokens":
            return FinishReason.MAX_TOKENS
        if stop_reason == "refusal":
            return FinishReason.CONTENT_FILTER
        if stop_reason == "pause_turn":
            return FinishReason.PAUSE
        # end_turn, stop_sequence, tool_use (builder derives TOOL_CALLS) -> STOP
        return FinishReason.STOP


class AnthropicService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("ANTHROPIC_API_KEY"):
            return """\
You're using the Anthropic NLP service, but ANTHROPIC_API_KEY is not set.
Please set ANTHROPIC_API_KEY in your environment before running Parlant.
"""

        return None

    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, health_reporter: HealthReporter
    ) -> None:
        self.logger = logger
        self._tracer = tracer
        self._meter = meter

        self._health_reporter = health_reporter

        self.logger.info("Initialized AnthropicService")

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
    async def get_react_generator(self) -> ReactGenerator:
        return AnthropicReactGenerator(logger=self.logger, cache=CacheConfig(enabled=True))

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> AnthropicAISchematicGenerator[T]:
        if (
            t == JourneyBacktrackNodeSelectionSchema
            or t == DisambiguationGuidelineMatchesSchema
            or t == CannedResponseSelectionSchema
        ):
            return Claude_Opus_4_1[t](self.logger, self._tracer, self._meter, self._health_reporter)  # type: ignore
        return Claude_Sonnet_4[t](self.logger, self._tracer, self._meter, self._health_reporter)  # type: ignore

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        return JinaAIEmbedder(self.logger, self._tracer, self._meter, self._health_reporter)

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
