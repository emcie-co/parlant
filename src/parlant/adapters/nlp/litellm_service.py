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

"""LiteLLM-backed NLP service.

LiteLLM is a unified gateway over 100+ providers (OpenAI, Anthropic, Bedrock,
Vertex, Azure, Ollama, ...). This adapter drives it through the LiteLLM Python
SDK (``litellm.acompletion`` / ``litellm.aembedding``) rather than an HTTP proxy,
so a single provider reaches any model.

Cost correctness is the point: LiteLLM prices every response it can, and the
adapter surfaces that authoritative figure as ``UsageInfo.extra["reported_cost"]``
— which :mod:`parlant.core.pricing` prefers over token-derived pricing. Usage
and cost reports, budgets/cost-control, and the frontend therefore work without
any provider-specific pricing table, because they all key off ``UsageInfo``.
"""

import itertools
import json
import time
from typing import Any, AsyncIterator, Iterator, Mapping, Optional, Sequence, cast
from weakref import WeakKeyDictionary

import jsonfinder  # type: ignore
import litellm
from litellm.exceptions import BadRequestError
from pydantic import ValidationError

from parlant.core.container import Container

from parlant.adapters.nlp.common import (
    RESULT_ONLY_TOOL_NAME,
    result_only_tool_output,
    parlant_client_headers,
    parlant_user_agent_token,
    decorate_schematic_generator,
    normalize_json_output,
    record_llm_metrics,
    validate_model_arguments,
)
from parlant.core.common import to_prompt_json
from parlant.core.engines.compass.prompt_builder import PromptBuilder
from parlant.core.loggers import ScopedLogger, Logger
from parlant.core.meter import Meter
from parlant.core.nlp.common import UsageInfo, format_reported_cost
from parlant.core.nlp.embedding import BaseEmbedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.nlp.model_routing import ModelCapability, ModelImplementationRegistry
from parlant.core.nlp.react import (
    CacheConfig,
    FinishReason,
    Message,
    ReactGenerator,
    ReactGeneratorHints,
    ReasoningConfig,
    ReasoningDelta,
    Role,
    StreamEvent,
    TextDelta,
    ToolCallStarted,
    ToolChoice,
    ToolSpec,
    TurnBuilder,
    Usage,
)
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.tracer import Tracer
from parlant.core.usage_reporter import UsageReporter

# LiteLLM chatters on stdout/stderr by default; keep it quiet and let Parlant's
# logger own diagnostics.
litellm.suppress_debug_info = True

_JSON_INSTRUCTION = (
    "IMPORTANT: You must respond with ONLY valid JSON — a single JSON object, "
    "with no explanatory text before or after it."
)


def _reported_cost(response: Any) -> float | None:
    """LiteLLM's authoritative USD cost for a response, if it could price it.

    Primary source is ``response._hidden_params["response_cost"]`` (populated by
    LiteLLM's cost tracking); ``litellm.completion_cost`` is the fallback. Absent
    or non-positive → ``None`` (pricing falls back to token derivation)."""
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, Mapping):
        cost = hidden.get("response_cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost > 0:
            return float(cost)

    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        return None

    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost > 0:
        return float(cost)
    return None


def _cached_input_tokens(usage: Any) -> int:
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details else 0
    return cached if isinstance(cached, int) and cached > 0 else 0


def _usage_info(response: Any) -> UsageInfo:
    """Map a LiteLLM response to Parlant's provider-neutral ``UsageInfo``.

    The token fields come from the OpenAI-compatible ``usage`` block; the
    provider-reported cost (when LiteLLM could price the model) rides in
    ``extra["reported_cost"]`` and overrides token-derived pricing downstream."""
    usage = getattr(response, "usage", None)

    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    if not isinstance(input_tokens, int):
        input_tokens = 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    if not isinstance(output_tokens, int):
        output_tokens = 0
    cached_input_tokens = _cached_input_tokens(usage) if usage else 0

    extra: dict[str, int | float | str] = {}
    if (cost := _reported_cost(response)) is not None:
        if (reported_cost := format_reported_cost(cost)) is not None:
            extra["reported_cost"] = reported_cost

    return UsageInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        extra=extra,
    )


def _extract_embedding(data_point: Any) -> list[float]:
    if isinstance(data_point, Mapping):
        return list(data_point.get("embedding", []))
    return list(getattr(data_point, "embedding", []))


def _parse_json_content(raw: str, logger: Logger) -> dict[str, Any]:
    text = raw.strip() or "{}"
    try:
        parsed = json.loads(normalize_json_output(text))
        if isinstance(parsed, dict) and parsed:
            return parsed
    except json.JSONDecodeError:
        logger.warning(f"LiteLLM returned non-JSON content; attempting recovery:\n{text}")

    try:
        return cast(dict[str, Any], jsonfinder.only_json(text)[2])
    except Exception:
        logger.error(f"Could not extract JSON from LiteLLM response:\n{text}")
        return {}


def _litellm_identity_headers() -> dict[str, str]:
    """Parlant's identity for a LiteLLM call.

    LiteLLM exposes no client object to carry default headers, so every call site
    passes these explicitly. The User-Agent is set rather than appended: the
    underlying provider SDK's own agent is not reachable from here."""
    headers = dict(parlant_client_headers())

    if token := parlant_user_agent_token():
        headers["User-Agent"] = token

    return headers


class LiteLLMEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def estimate_token_count(self, prompt: str) -> int:
        try:
            return int(litellm.token_counter(model=self.model_name, text=prompt))
        except Exception:
            # Unknown model tokenizer — fall back to a coarse char estimate.
            return max(1, len(prompt) // 4)


class LiteLLMSchematicGenerator(BaseSchematicGenerator[T]):
    # reasoning_effort is litellm-native: forwarded to models that support
    # it and dropped (via drop_params below) for those that do not.
    supported_arguments = ["temperature", "max_tokens", "reasoning_effort"]

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
            provider_name=provider_name or "litellm",
        )
        self._logger = ScopedLogger(logger, "LiteLLMSchematicGenerator")
        self._api_key = api_key
        self._tokenizer = LiteLLMEstimatingTokenizer(model_name=model_name)

    @property
    def id(self) -> str:
        return f"litellm/{self.model_name}"

    @property
    def tokenizer(self) -> LiteLLMEstimatingTokenizer:
        return self._tokenizer

    @property
    def max_tokens(self) -> int:
        return 8192

    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        args = {k: v for k, v in hints.items() if k in self.supported_arguments}
        if self._api_key is not None:
            args["api_key"] = self._api_key

        t_start = time.time()
        try:
            response = await litellm.acompletion(
                extra_headers=_litellm_identity_headers(),
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                num_retries=2,
                drop_params=True,
                **args,
            )
        except BadRequestError:
            # The routed model may not support JSON mode — retry with an explicit
            # instruction instead of forcing response_format.
            self._logger.warning(
                f"Model '{self.model_name}' rejected JSON mode; retrying with a "
                "JSON instruction instead."
            )
            response = await litellm.acompletion(
                extra_headers=_litellm_identity_headers(),
                model=self.model_name,
                messages=[
                    {"role": "system", "content": _JSON_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                num_retries=2,
                drop_params=True,
                **args,
            )
        t_end = time.time()

        raw_content = response.choices[0].message.content or "{}"
        json_content = _parse_json_content(raw_content, self._logger)

        try:
            content = self.schema.model_validate(json_content)
        except ValidationError as exc:
            self._logger.error(
                f"LiteLLM content from '{self.model_name}' does not match "
                f"{self.schema.__name__}:\n{raw_content}\nErrors: {exc}"
            )
            raise

        usage = _usage_info(response)

        await record_llm_metrics(
            self.meter,
            self.model_name,
            schema_name=self.schema.__name__,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cached_input_tokens,
        )

        return SchematicGenerationResult(
            content=content,
            info=GenerationInfo(
                schema_name=self.schema.__name__,
                model=self.id,
                duration=(t_end - t_start),
                usage=usage,
            ),
        )


class LiteLLMEmbedder(BaseEmbedder):
    supported_arguments = ["dimensions"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        usage_reporter: UsageReporter | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
        provider_name: str | None = None,
    ) -> None:
        super().__init__(
            logger=logger,
            tracer=tracer,
            meter=meter,
            model_name=model_name,
            usage_reporter=usage_reporter,
            provider_name=provider_name or "litellm",
        )
        self._api_key = api_key
        self._dimensions = dimensions
        self._tokenizer = LiteLLMEstimatingTokenizer(model_name=model_name)

    @property
    def id(self) -> str:
        return f"litellm/{self.model_name}"

    @property
    def tokenizer(self) -> LiteLLMEstimatingTokenizer:
        return self._tokenizer

    @property
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        # Detected from the first response when not configured; 1536 is a safe
        # default matching most OpenAI-family embedding models.
        return self._dimensions or 1536

    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        args = {k: v for k, v in hints.items() if k in self.supported_arguments}
        if self._api_key is not None:
            args["api_key"] = self._api_key
        if self._dimensions is not None and "dimensions" not in args:
            args["dimensions"] = self._dimensions

        response = await litellm.aembedding(
            extra_headers=_litellm_identity_headers(),
            model=self.model_name,
            input=texts,
            num_retries=2,
            **args,
        )

        vectors = [_extract_embedding(data_point) for data_point in response.data]

        if self._dimensions is None and vectors:
            self._dimensions = len(vectors[0])

        return EmbeddingResult(vectors=vectors, usage=_usage_info(response))


def _cost_from_tokens(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Token-derived USD cost from LiteLLM's pricing tables.

    Used on the streaming paths (React / streaming text), where the response is
    assembled from chunks and ``_hidden_params["response_cost"]`` isn't reliably
    present. LiteLLM prices the model from its own table; unknown model → None
    (pricing then falls back to Parlant's token derivation downstream)."""
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
    except Exception:
        return None
    total = float(prompt_cost or 0.0) + float(completion_cost or 0.0)
    return total if total > 0 else None


def _stream_usage_info(usage: Any, model: str) -> UsageInfo:
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    extra: dict[str, int | float | str] = {}
    if (cost := _cost_from_tokens(model, int(input_tokens), int(output_tokens))) is not None:
        if (reported_cost := format_reported_cost(cost)) is not None:
            extra["reported_cost"] = reported_cost
    return UsageInfo(
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        cached_input_tokens=_cached_input_tokens(usage),
        extra=extra,
    )


# Per-stream tool-call bookkeeping for the React decoder. Chat-completions streams
# tool calls by ``index`` — the id/name arrive on the first delta, arguments then
# stream as JSON fragments under the same index. We remember index → id (and which
# calls we've already announced) per TurnBuilder; the WeakKeyDictionary drops the
# entry once the builder is collected at the end of the stream.
_REACT_STREAM_STATE: "WeakKeyDictionary[TurnBuilder, dict[str, Any]]" = WeakKeyDictionary()

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "length": FinishReason.MAX_TOKENS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


def _stringify_tool_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return to_prompt_json(content)
    except (TypeError, ValueError):
        return str(content)


def _react_usage(usage: Any, model: str) -> Usage:
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0

    extra: dict[str, int | float | str] = {}
    if (cost := _cost_from_tokens(model, input_tokens, output_tokens)) is not None:
        if (reported_cost := format_reported_cost(cost)) is not None:
            extra["reported_cost"] = reported_cost

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=_cached_input_tokens(usage),
        reasoning_tokens=reasoning_tokens,
        extra=extra,
        model_name=model,
        requests=1,
    )


class LiteLLMReactGenerator(ReactGenerator):
    """A native ReAct adapter over LiteLLM's streaming chat-completions API.

    LiteLLM speaks the OpenAI chat-completions wire format across every provider
    it fronts, so the encode/decode here is provider-neutral: history becomes
    ``messages`` + ``tools``, and the streamed deltas (text, reasoning, tool-call
    fragments, and the closing usage block) fold into the shared ``TurnBuilder``.
    """

    def __init__(
        self,
        *,
        model: str,
        logger: Logger,
        cache: Optional[CacheConfig] = None,
        api_key: str | None = None,
        provider_name: Optional[str] = None,
        usage_reporter: Optional[UsageReporter] = None,
    ) -> None:
        super().__init__(model=model, cache=cache, usage_reporter=usage_reporter)
        self._logger = ScopedLogger(logger, "LiteLLMReactGenerator")
        self._api_key = api_key
        self._provider_name = provider_name or "litellm"
        self._tokenizer = LiteLLMEstimatingTokenizer(model_name=model)

    @property
    def id(self) -> str:
        return f"litellm/{self.model}"

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def tokenizer(self) -> LiteLLMEstimatingTokenizer:
        return self._tokenizer

    @property
    def max_tokens(self) -> int:
        return 8192

    def _encode(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec],
        tool_choice: ToolChoice,
        *,
        reasoning: ReasoningConfig,
        hints: ReactGeneratorHints = {},
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        # Call ids issued and not yet answered. A result that answers none of
        # them is sent as a synthetic call plus its tool message, since
        # providers reject a tool message that answers no tool call.
        pending_calls: set[str] = set()
        synthetic_ids = itertools.count(1)
        for message in history:
            if message.role == Role.TOOL:
                messages.extend(self._encode_tool_results(message, pending_calls, synthetic_ids))
                continue
            pending_calls.update(call.id for call in message.tool_calls)
            messages.extend(self._encode_message(message))

        request: dict[str, Any] = {
            "model": self._resolve_model(hints),
            "messages": messages,
            "num_retries": 2,
        }
        if tools:
            request["tools"] = [self._encode_tool(spec) for spec in tools]
            request["tool_choice"] = self._encode_tool_choice(tool_choice)
        if self._api_key is not None:
            request["api_key"] = self._api_key
        return request

    def _encode_tool_results(
        self,
        message: Message,
        pending_calls: set[str],
        synthetic_ids: Iterator[int],
    ) -> list[dict[str, Any]]:
        """A tool message's results: those answering an issued call first, as
        they must directly follow it, then a synthetic call and its tool
        message for each result that answers none."""
        answered: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        synthetic: list[dict[str, Any]] = []
        for part in message.tool_results:
            if part.call_id in pending_calls:
                pending_calls.remove(part.call_id)
                answered.append(
                    {
                        "role": "tool",
                        "tool_call_id": part.call_id,
                        "content": _stringify_tool_content(part.content),
                    }
                )
                continue
            call_id = f"call_context_{next(synthetic_ids)}"
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": RESULT_ONLY_TOOL_NAME, "arguments": "{}"},
                }
            )
            synthetic.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": to_prompt_json(result_only_tool_output(part)),
                }
            )
        if calls:
            return [
                *answered,
                {"role": "assistant", "content": None, "tool_calls": calls},
                *synthetic,
            ]
        return answered

    def _encode_message(self, message: Message) -> list[dict[str, Any]]:
        if message.role == Role.SYSTEM:
            return [{"role": "system", "content": message.text}]

        if message.role == Role.USER:
            return [{"role": "user", "content": message.text}]

        if message.role == Role.TOOL:
            return [
                {
                    "role": "tool",
                    "tool_call_id": part.call_id,
                    "content": _stringify_tool_content(part.content),
                }
                for part in message.tool_results
            ]

        # Assistant: text and/or tool calls.
        payload: dict[str, Any] = {"role": "assistant", "content": message.text or None}
        if calls := message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": to_prompt_json(dict(call.args)),
                    },
                }
                for call in calls
            ]
        return [payload]

    def _encode_tool(self, spec: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.json_schema(),
            },
        }

    def _encode_tool_choice(self, tool_choice: ToolChoice) -> Any:
        if isinstance(tool_choice, Mapping):
            return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
        return tool_choice

    async def _raw_stream(self, request: Any) -> AsyncIterator[Any]:
        stream = await litellm.acompletion(
            extra_headers=_litellm_identity_headers(),
            **request,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            yield chunk

    def _decode(self, raw_event: Any, builder: TurnBuilder) -> Sequence[StreamEvent]:
        events: list[StreamEvent] = []
        state = _REACT_STREAM_STATE.setdefault(builder, {"ids": {}, "started": set()})
        index_ids: dict[int, str] = state["ids"]
        started: set[str] = state["started"]

        if (usage := getattr(raw_event, "usage", None)) is not None:
            builder.usage = _react_usage(usage, self.model)

        choices = getattr(raw_event, "choices", None)
        if not choices:
            return events

        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is not None:
            if content := getattr(delta, "content", None):
                builder.text_delta(content)
                events.append(TextDelta(text=content))

            if reasoning := getattr(delta, "reasoning_content", None):
                builder.reasoning_delta(reasoning)
                events.append(ReasoningDelta(text=reasoning))

            for tool_call in getattr(delta, "tool_calls", None) or []:
                index = getattr(tool_call, "index", 0) or 0
                if tc_id := getattr(tool_call, "id", None):
                    index_ids[index] = tc_id
                call_id = index_ids.get(index) or getattr(tool_call, "id", None) or f"call_{index}"
                index_ids.setdefault(index, call_id)

                function = getattr(tool_call, "function", None)
                name = getattr(function, "name", None) if function else None
                args_delta = (getattr(function, "arguments", None) if function else None) or ""

                builder.tool_call(call_id, name=name or None, args_delta=args_delta)
                if name and call_id not in started:
                    started.add(call_id)
                    events.append(ToolCallStarted(id=call_id, name=name))

        if finish := getattr(choice, "finish_reason", None):
            builder.finish_reason = _FINISH_REASONS.get(finish, FinishReason.STOP)

        return events


def register_litellm_models(
    registry: ModelImplementationRegistry,
    container: Container,
) -> None:
    """Register the LiteLLM provider's model factories + default route resolvers.

    The default model choices mirror the OpenRouter provider's, translated into
    LiteLLM's namespace (``google/…`` → ``gemini/…``; ``openai/…`` /
    ``anthropic/…`` unchanged). Any LiteLLM model string works via the routing
    layer; these are only the defaults a bare route resolves to.
    """
    logger = container[Logger]
    tracer = container[Tracer]
    meter = container[Meter]
    usage = container[UsageReporter]

    async def schematic(
        model: str,
        schema: type[Any],
        raw_args: Mapping[str, Any],
    ) -> LiteLLMSchematicGenerator[Any]:
        args = validate_model_arguments("litellm", raw_args, {"api_key"})
        generator_type = LiteLLMSchematicGenerator[schema]  # type: ignore[valid-type]
        return cast(
            LiteLLMSchematicGenerator[Any],
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

    async def react(model: str, raw_args: Mapping[str, Any]) -> LiteLLMReactGenerator:
        args = validate_model_arguments("litellm", raw_args, {"api_key"})
        return LiteLLMReactGenerator(model=model, logger=logger, usage_reporter=usage, **args)

    async def embedding(model: str, raw_args: Mapping[str, Any]) -> LiteLLMEmbedder:
        args = validate_model_arguments("litellm", raw_args, {"api_key", "dimensions"})
        return LiteLLMEmbedder(
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
    ) -> LiteLLMEstimatingTokenizer:
        validate_model_arguments("litellm", raw_args, set())
        return LiteLLMEstimatingTokenizer(model)

    def schematic_model_resolver(route: str, qualifiers: Mapping[str, str]) -> str:
        if route.startswith("review"):
            return "gemini/gemini-3.6-flash"
        elif route.startswith("matching.distill"):
            return "gemini/gemini-3.5-flash-lite"
        else:  # default
            return "gemini/gemini-3.1-flash-lite"

    def react_model_resolver(route: str, qualifiers: Mapping[str, str]) -> str:
        if qualifiers.get("effort") in ("min", "low"):
            return "openai/gpt-5.6-luna"
        else:  # default
            return "anthropic/claude-sonnet-5"

    registry.register_schematic(
        "litellm",
        schematic,
        default=schematic_model_resolver,
    )
    registry.register(
        "litellm",
        ModelCapability.REACT,
        react,
        default=react_model_resolver,
    )
    registry.register(
        "litellm",
        ModelCapability.EMBEDDING,
        embedding,
        default="openai/text-embedding-3-large",
    )
    registry.register(
        "litellm",
        ModelCapability.TOKENIZER,
        tokenizer,
        default="openai/gpt-4o",
    )
