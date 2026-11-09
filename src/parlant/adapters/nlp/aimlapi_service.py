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

"""AI/ML API adapter using the OpenAI-compatible Chat Completions API."""

import itertools
import json
import time
from typing import Any, AsyncIterator, Iterator, Mapping, Optional, Sequence, cast
from weakref import WeakKeyDictionary

import jsonfinder  # type: ignore
import os
from urllib.parse import urlparse
import httpx2
import tiktoken
from openai import AsyncClient
from pydantic import ValidationError

from parlant.core.container import Container

from parlant.adapters.nlp.common import (
    RESULT_ONLY_TOOL_NAME,
    result_only_tool_output,
    parlant_client_headers,
    client_identity_enabled,
    require_api_key,
    with_client_identity,
    decorate_schematic_generator,
    normalize_json_output,
    record_llm_metrics,
    validate_model_arguments,
)
from parlant.core.common import to_prompt_json
from parlant.core.engines.compass.prompt_builder import PromptBuilder
from parlant.core.loggers import ScopedLogger, Logger
from parlant.core.meter import Meter
from parlant.core.nlp.common import UsageInfo
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

AIMLAPI_BASE_URL = "https://api.aimlapi.com/v1"
AIMLAPI_DEFAULT_MODEL = "openai/gpt-4.1"
AIMLAPI_DEFAULT_EMBEDDER_MODEL = "openai/text-embedding-3-large"


def build_attribution_headers(base_url: str = AIMLAPI_BASE_URL) -> dict[str, str]:
    if urlparse(base_url).hostname != "api.aimlapi.com" or not client_identity_enabled():
        return {}
    return {
        **parlant_client_headers(),
        "HTTP-Referer": os.environ.get(
            "AIMLAPI_HTTP_REFERER", "https://github.com/emcie-co/parlant"
        ),
        "X-Title": os.environ.get("AIMLAPI_SITE_NAME", "Parlant"),
        "X-AIMLAPI-Partner-ID": "part_UOT3mCwOdpOUQKX2gIvYrCmv",
        "X-AIMLAPI-Source": "agent/parlant",
    }


def _create_client(api_key: str | None = None) -> AsyncClient:
    return with_client_identity(
        AsyncClient(
            api_key=require_api_key("AI/ML API", "AIMLAPI_API_KEY", api_key),
            base_url=AIMLAPI_BASE_URL,
            default_headers=build_attribution_headers(),
            timeout=httpx2.Timeout(120.0, connect=10.0),
            max_retries=2,
        )
    )


def _cached_input_tokens(usage: Any) -> int:
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details else 0
    return cached if isinstance(cached, int) and cached > 0 else 0


def _usage_info(response: Any) -> UsageInfo:
    usage = getattr(response, "usage", None)

    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    if not isinstance(input_tokens, int):
        input_tokens = 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    if not isinstance(output_tokens, int):
        output_tokens = 0
    cached_input_tokens = _cached_input_tokens(usage) if usage else 0

    return UsageInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
    )


def _extract_embedding(data_point: Any) -> list[float]:
    if isinstance(data_point, Mapping):
        return list(data_point.get("embedding", []))
    return list(getattr(data_point, "embedding", []))


def _parse_json_content(raw: str, logger: Logger) -> dict[str, Any]:
    text = normalize_json_output(raw.strip())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("AI/ML API returned non-JSON content; attempting JSON extraction")
        try:
            parsed = jsonfinder.only_json(text)[2]
        except Exception as exc:
            raise ValueError("AI/ML API returned no valid JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError("AI/ML API schematic output must be a JSON object")
    return cast(dict[str, Any], parsed)


class AIMLAPIEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def estimate_token_count(self, prompt: str) -> int:
        return len(tiktoken.get_encoding("o200k_base").encode(prompt))


class AIMLAPISchematicGenerator(BaseSchematicGenerator[T]):
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
            provider_name=provider_name or "aimlapi",
        )
        self._logger = ScopedLogger(logger, "AIMLAPISchematicGenerator")
        self._client = _create_client(api_key)
        self._tokenizer = AIMLAPIEstimatingTokenizer(model_name=model_name)

    @property
    def id(self) -> str:
        return f"aimlapi/{self.model_name}"

    @property
    def tokenizer(self) -> AIMLAPIEstimatingTokenizer:
        return self._tokenizer

    @property
    def max_tokens(self) -> int:
        return 1047576 if self.model_name == AIMLAPI_DEFAULT_MODEL else 131072

    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        args = {k: v for k, v in hints.items() if k in self.supported_arguments and v is not None}

        t_start = time.time()
        response = await self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            **args,
        )
        t_end = time.time()

        if not response.choices or not response.choices[0].message.content:
            raise ValueError("AI/ML API returned an empty schematic response")
        raw_content = response.choices[0].message.content
        json_content = _parse_json_content(raw_content, self._logger)

        try:
            content = self.schema.model_validate(json_content)
        except ValidationError as exc:
            self._logger.error(
                f"AIMLAPI content from '{self.model_name}' does not match "
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


class AIMLAPIEmbedder(BaseEmbedder):
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
            provider_name=provider_name or "aimlapi",
        )
        self._dimensions = dimensions
        self._expected_dimensions = dimensions or {
            "openai/text-embedding-3-large": 3072,
            "openai/text-embedding-3-small": 1536,
            "openai/text-embedding-ada-002": 1536,
        }.get(model_name)
        if self._expected_dimensions is None or self._expected_dimensions <= 0:
            raise ValueError(
                "Set a positive dimensions argument for this AI/ML API embedding model"
            )
        self._client = _create_client(api_key)
        self._tokenizer = AIMLAPIEstimatingTokenizer(model_name=model_name)

    @property
    def id(self) -> str:
        return f"aimlapi/{self.model_name}"

    @property
    def tokenizer(self) -> AIMLAPIEstimatingTokenizer:
        return self._tokenizer

    @property
    def max_tokens(self) -> int:
        return 8192

    @property
    def dimensions(self) -> int:
        assert self._expected_dimensions is not None
        return self._expected_dimensions

    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        args = {k: v for k, v in hints.items() if k in self.supported_arguments and v is not None}
        if self._dimensions is not None and "dimensions" not in args:
            args["dimensions"] = self._dimensions

        response = await self._client.embeddings.create(
            model=self.model_name,
            input=texts,
            **args,
        )

        vectors = [_extract_embedding(data_point) for data_point in response.data]

        if len(vectors) != len(texts) or any(len(v) != self.dimensions for v in vectors):
            raise ValueError("AI/ML API returned an unexpected embedding count or dimension")

        return EmbeddingResult(vectors=vectors, usage=_usage_info(response))


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

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=_cached_input_tokens(usage),
        reasoning_tokens=reasoning_tokens,
        model_name=model,
        requests=1,
    )


class AIMLAPIReactGenerator(ReactGenerator):
    """Streaming Chat Completions with text, tool calls and token usage."""

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
        self._logger = ScopedLogger(logger, "AIMLAPIReactGenerator")
        self._client = _create_client(api_key)
        self._provider_name = provider_name or "aimlapi"
        self._tokenizer = AIMLAPIEstimatingTokenizer(model_name=model)

    @property
    def id(self) -> str:
        return f"aimlapi/{self.model}"

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def tokenizer(self) -> AIMLAPIEstimatingTokenizer:
        return self._tokenizer

    @property
    def max_tokens(self) -> int:
        return 1047576 if self.model == AIMLAPI_DEFAULT_MODEL else 131072

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
        }
        if tools:
            request["tools"] = [self._encode_tool(spec) for spec in tools]
            request["tool_choice"] = self._encode_tool_choice(tool_choice)
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
        stream = await self._client.chat.completions.create(
            **request,
            stream=True,
            stream_options={"include_usage": True},
        )
        try:
            async for chunk in stream:
                yield chunk
        finally:
            await stream.close()

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


def register_aimlapi_models(
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
    ) -> AIMLAPISchematicGenerator[Any]:
        args = validate_model_arguments("aimlapi", raw_args, {"api_key"})
        generator_type = AIMLAPISchematicGenerator[schema]  # type: ignore[valid-type]
        return cast(
            AIMLAPISchematicGenerator[Any],
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

    async def react(model: str, raw_args: Mapping[str, Any]) -> AIMLAPIReactGenerator:
        args = validate_model_arguments("aimlapi", raw_args, {"api_key"})
        return AIMLAPIReactGenerator(model=model, logger=logger, usage_reporter=usage, **args)

    async def embedding(model: str, raw_args: Mapping[str, Any]) -> AIMLAPIEmbedder:
        args = validate_model_arguments("aimlapi", raw_args, {"api_key", "dimensions"})
        return AIMLAPIEmbedder(
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
    ) -> AIMLAPIEstimatingTokenizer:
        validate_model_arguments("aimlapi", raw_args, set())
        return AIMLAPIEstimatingTokenizer(model)

    registry.register_schematic(
        "aimlapi",
        schematic,
        default=AIMLAPI_DEFAULT_MODEL,
    )
    registry.register(
        "aimlapi",
        ModelCapability.REACT,
        react,
        default=AIMLAPI_DEFAULT_MODEL,
    )
    registry.register(
        "aimlapi",
        ModelCapability.EMBEDDING,
        embedding,
        default=AIMLAPI_DEFAULT_EMBEDDER_MODEL,
    )
    registry.register(
        "aimlapi",
        ModelCapability.TOKENIZER,
        tokenizer,
        default="openai/gpt-4o",
    )
