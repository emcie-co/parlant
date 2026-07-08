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

"""Provider-agnostic infrastructure for modern ReAct agent loops.

This module is the *port* (in hexagonal terms): it defines the canonical,
provider-neutral message model and the abstract ``ReactGenerator`` contract.
Concrete adapters (Gemini, OpenAI, Anthropic, ...) live under
``parlant.adapters.nlp`` and implement the three-method provider seam:
``_encode``, ``_raw_stream``, and ``_decode``. Everything else here is
concrete and shared.

Core contract
-------------
* ``step()`` is ONE model inference. It NEVER executes tools. It returns the
  assistant turn (text + reasoning + pending tool calls); the caller decides
  whether and how to run them. This per-step seam is what gives callers full
  control over the loop.
* History is a ``Sequence[Message]`` the caller fully owns and may mutate,
  splice, or rebuild between steps (see :meth:`ReactGenerator.run` and
  ``StepHook``). The generator only reads it.
* Every :class:`Part` carries an opaque ``provider_data`` blob. Anthropic
  thinking signatures, Gemini ``thought_signature`` values, and OpenAI
  reasoning-item ids / encrypted content round-trip through it. PRESERVE IT
  VERBATIM when replaying history — dropping it breaks tool calling on Gemini
  and Anthropic. Never edit values inside it.
* Cancellation uses plain asyncio: cancel the task awaiting ``step``/``run``
  (or break out of ``stream_step``) and the in-flight provider stream is torn
  down via the usual ``CancelledError`` propagation. There is no separate
  abort flag.

Manual step loop (the caller controls tool execution and may edit history)::

    history = [Message(role=Role.USER, parts=[TextPart(text="Fix the bug in app.py")])]
    total = Usage()
    while True:
        result = await generator.step(history, tools, tool_choice="auto")
        history.append(result.message)  # append the turn AS-IS (keeps signatures)
        total = total + result.usage  # usage reports aggregate across steps
        if not result.needs_tools:
            break
        # The caller decides how to run tools: approve, deny, edit args, etc.
        outputs = [
            ToolResultPart(call_id=c.id, name=c.name, content=await run_tool(c.name, c.args))
            for c in result.tool_calls
        ]
        history.append(Message(role=Role.TOOL, parts=outputs))
        # History surgery is safe as long as each kept Part's provider_data /
        # signature is kept with it (dropping a whole part is fine).
"""

import abc
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from functools import lru_cache
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import tiktoken
from typing_extensions import NotRequired, TypedDict

from parlant.core.nlp.common import ModelGeneration, ModelSize, ModelType
from parlant.core.tools import Tool


ServiceTier = Literal["standard", "flex", "priority"]
"""Provider-neutral request service tier, mapped per provider in ``_encode``:

- ``"standard"``: the normal tier.
- ``"flex"``: cheaper, slower, best-effort (OpenAI/Gemini have it; Anthropic
  has no flex tier, so it maps to standard).
- ``"priority"``: faster, premium, higher reliability.
"""


class ReactGeneratorHints(TypedDict, total=False):
    model_size: NotRequired[ModelSize]
    model_generation: NotRequired[ModelGeneration]
    model_type: NotRequired[ModelType]
    service_tier: NotRequired[ServiceTier]
    # Seconds to wait for the first content-bearing event before hedging: if no
    # event arrives within this window, an identical second stream is opened and
    # whichever emits its first event first wins (the loser is cancelled). Absent
    # / ``None`` / ``<= 0`` disables hedging. See ``ReactGenerator.stream_step``.
    hedge_timeout: NotRequired[float]


# Key under which a tool-call part records, in its ``provider_data``, the concrete
# model that produced it (the resolved per-call model, not the generator's static
# identity model). Stamped at generation, read at serialize time to build the
# tool-event blob, and compared on replay by providers whose native replay is
# model-bound (e.g. Gemini's thought_signature). Transient: it travels on the live
# part within a turn and is not itself persisted.
REACT_MODEL_KEY = "__react_model__"


class ReactError(Exception):
    """An error raised by a ReactGenerator while running a step. ``retryable``
    marks transient provider failures (rate limits, connection/server errors) the
    caller may safely retry — but only before any event of the step has been
    emitted, since a stream can't be replayed mid-flight."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


# ───────────────────────────── canonical message model ─────────────────────


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(kw_only=True)
class Part:
    """Base for every content part.

    ``provider_data`` is opaque passthrough that MUST survive any history edit
    and be sent back to the provider unchanged (e.g. Gemini
    ``thought_signature``, OpenAI reasoning item id).
    """

    provider_data: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class TextPart(Part):
    text: str = ""
    type: Literal["text"] = "text"


@dataclass(kw_only=True)
class ReasoningPart(Part):
    """Model reasoning.

    ``text`` is what is visible to the caller: Anthropic returns the full
    thinking text; OpenAI and Gemini return a summary (raw chain-of-thought is
    never returned). ``signature`` carries Anthropic's thinking-block
    signature; providers that prefer a different channel (Gemini, OpenAI) use
    ``provider_data``. Echo whichever channel is set back unmodified.
    """

    text: str = ""
    signature: Union[str, bytes, None] = None
    visibility: Literal["full", "summary"] = "summary"
    type: Literal["reasoning"] = "reasoning"


@dataclass(kw_only=True)
class ToolCallPart(Part):
    id: str = ""  # echo this on the matching result
    name: str = ""
    args: Mapping[str, Any] = field(default_factory=dict)
    type: Literal["tool_call"] = "tool_call"
    # NOTE (Gemini): the model's thought_signature rides on the FIRST tool_call
    # part of a turn and lives in provider_data. Keep it.


@dataclass(kw_only=True)
class ToolResultPart(Part):
    call_id: str = ""  # must match the originating ToolCallPart.id
    name: str = ""
    content: Any = None  # str or any JSON-serializable value
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


@dataclass(kw_only=True)
class Message:
    role: Role
    parts: Sequence[Part] = field(default_factory=list)
    provider_data: dict[str, Any] = field(default_factory=dict)  # message-level passthrough
    cache_key: Optional[str] = None
    """Cache breakpoint + identity. ``None`` means this message is not a cache
    breakpoint. A non-``None`` key marks the prefix up to and including this
    message as cacheable and names that cache. Caching is positional, so this
    one provider-neutral field maps onto each provider's mechanism:

    - OpenAI: passed as ``prompt_cache_key`` (a routing hint for its automatic
      prefix cache; correctness does not depend on it).
    - Gemini: the marked prefix becomes a ``CachedContent`` resource; the key is
      its reuse identity / display name.
    - Anthropic: marks a ``cache_control`` breakpoint here; the key value has no
      native meaning and is used only as a reuse identity.

    See :class:`CacheConfig` for the master on/off and TTL."""

    @property
    def text(self) -> str:
        """Concatenation of all visible text parts (excludes reasoning)."""
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    @property
    def reasoning(self) -> str:
        """Concatenation of all reasoning parts' visible text."""
        return "".join(p.text for p in self.parts if isinstance(p, ReasoningPart))

    @property
    def tool_calls(self) -> Sequence[ToolCallPart]:
        return [p for p in self.parts if isinstance(p, ToolCallPart)]

    @property
    def tool_results(self) -> Sequence[ToolResultPart]:
        return [p for p in self.parts if isinstance(p, ToolResultPart)]

    def __repr__(self) -> str:
        return f"Message(role={self.role}, text={self.text!r}, reasoning={self.reasoning!r}, tool_calls={self.tool_calls}, tool_results={self.tool_results})"


# ──────────────────────── tool-event persistence ports ─────────────────────
#
# A tool turn (assistant calls + tool results) is persisted outside this module
# (e.g. in a session) so it can be replayed into the history of a LATER turn.
# Two problems make that provider-specific: providers disagree on tool-call id
# shape, and some require native artifacts on replay (Gemini's thought_signature,
# etc.) that aren't synthesizable. So the *provider* owns turning a tool turn
# into a JSON-safe blob and back into Messages, while the *caller* owns where the
# bytes live — expressed through these two ports. The caller implements them; the
# provider (ReactGenerator) drives them. Neither imports the other's domain.
#
# Index-alignment invariant: calls[i], results[i], and the i-th per-call slice of
# the provider-data blob all describe the same call. Calls↔results are 1:1 (a
# failed call still yields a result with is_error=True).


class ToolMessageSerializer(abc.ABC):
    """Sink a ReactGenerator pushes a tool turn into, on persist."""

    @abc.abstractmethod
    def write_calls(self, calls: Sequence[ToolCallPart]) -> None:
        """The turn's tool calls. May be a no-op where the durable record is built
        from execution rather than from these parts — kept for future providers
        that need to persist call-side data."""

    @abc.abstractmethod
    def write_results(self, results: Sequence[ToolResultPart]) -> None:
        """The turn's results. Same no-op caveat as :meth:`write_calls`; exists so
        a future provider can persist result-side artifacts."""

    @abc.abstractmethod
    def write_provider_data(self, data: Mapping[str, Any]) -> None:
        """The single per-turn provider blob (always carries ``provider`` and
        ``model``, plus any native replay artifacts). Opaque to the caller; stored
        verbatim and handed back unchanged on read."""


class ToolMessageDeserializer(abc.ABC):
    """Source a ReactGenerator pulls a stored tool turn out of, on replay."""

    @abc.abstractmethod
    def read_calls(self) -> Sequence[ToolCallPart]:
        """Calls as parts with name + args populated (id/provider_data are left
        for the provider to assign)."""

    @abc.abstractmethod
    def read_results(self) -> Sequence[ToolResultPart]:
        """Results as parts with name + content + is_error populated (call_id is
        left for the provider to match to its synthesized ids)."""

    @abc.abstractmethod
    def read_provider_data(self) -> Mapping[str, Any]:
        """The blob written by :meth:`ToolMessageSerializer.write_provider_data`,
        or ``{}`` if none was stored."""


# ───────────────────────────── tools & config ──────────────────────────────


JSONSchemaType = Literal["string", "number", "integer", "boolean", "array", "object"]

# Sentinel distinguishing "no default" from an explicit default of ``None``.
_UNSET: Any = object()


@dataclass(kw_only=True)
class ParameterSpec:
    """A single tool parameter, described in a provider-neutral way.

    A list of these is turned into JSON Schema by :meth:`ToolSpec.json_schema`,
    which adapters feed to their provider. Use ``items`` for the element type of
    an ``array``, and ``properties`` for the fields of a nested ``object``.
    """

    name: str
    type: JSONSchemaType = "string"
    description: str = ""
    required: bool = True
    enum: Optional[Sequence[Any]] = None
    nullable: bool = False
    default: Any = _UNSET  # leave as _UNSET to omit; pass any value (incl. None) to set
    items: Optional["ParameterSpec"] = None  # element type when type == "array"
    properties: Optional[Sequence["ParameterSpec"]] = None  # fields when type == "object"

    def value_schema(self) -> Mapping[str, Any]:
        """JSON Schema for this parameter's *value* (excludes its name)."""
        schema: dict[str, Any] = {"type": self.type}

        if self.description:
            schema["description"] = self.description

        if self.enum is not None:
            schema["enum"] = list(self.enum)

        if self.nullable:
            schema["nullable"] = True

        if self.default is not _UNSET:
            schema["default"] = self.default

        if self.type == "array" and self.items is not None:
            schema["items"] = self.items.value_schema()

        if self.type == "object" and self.properties is not None:
            schema["properties"] = {p.name: p.value_schema() for p in self.properties}
            required = [p.name for p in self.properties if p.required]
            if required:
                schema["required"] = required

        return schema


@dataclass(kw_only=True)
class ToolSpec:
    name: str
    description: str
    parameters: Sequence[ParameterSpec] = ()

    def json_schema(self) -> Mapping[str, Any]:
        """The tool's argument object rendered as JSON Schema."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {p.name: p.value_schema() for p in self.parameters},
        }

        required = [p.name for p in self.parameters if p.required]
        if required:
            schema["required"] = required

        return schema


# Parlant tool-parameter types that have no direct JSON Schema primitive are
# presented to the model as strings.
_PARLANT_PARAM_TYPE_TO_JSON_SCHEMA: dict[str, JSONSchemaType] = {
    "string": "string",
    "number": "number",
    "integer": "integer",
    "boolean": "boolean",
    "array": "array",
}


def tool_specs_from_tools(tools: Sequence[Tool]) -> list[ToolSpec]:
    """Convert Parlant :class:`~parlant.core.tools.Tool` definitions into the
    provider-neutral :class:`ToolSpec`s the generator understands."""
    return [_tool_spec_from_tool(tool) for tool in tools]


def _tool_spec_from_tool(tool: Tool) -> ToolSpec:
    return ToolSpec(
        name=tool.name,
        description=tool.description,
        parameters=[
            _parameter_spec_from_descriptor(name, descriptor, required=name in tool.required)
            for name, (descriptor, _options) in tool.parameters.items()
        ],
    )


def _parameter_spec_from_descriptor(
    name: str,
    descriptor: Mapping[str, Any],
    *,
    required: bool,
) -> ParameterSpec:
    param_type = _PARLANT_PARAM_TYPE_TO_JSON_SCHEMA.get(descriptor.get("type", "string"), "string")

    items: Optional[ParameterSpec] = None
    if param_type == "array":
        item_type = _PARLANT_PARAM_TYPE_TO_JSON_SCHEMA.get(
            descriptor.get("item_type", "string"), "string"
        )
        items = ParameterSpec(name="item", type=item_type)

    enum = descriptor.get("enum")
    return ParameterSpec(
        name=name,
        type=param_type,
        description=descriptor.get("description", ""),
        required=required,
        enum=list(enum) if enum is not None else None,
        items=items,
    )


# "auto" | "none" | "required", or {"name": "<tool>"} to force one specific tool.
ToolChoice = Union[Literal["auto", "none", "required"], Mapping[str, str]]


@dataclass(kw_only=True)
class ReasoningConfig:
    """Provider-neutral reasoning knobs, mapped per provider in ``_encode``.

    ``effort`` is the only depth control. ``"minimal"`` means "as little
    reasoning as possible" and resolves to *off* on providers that support
    disabling reasoning (Anthropic Sonnet/Haiku 4.5: no ``thinking`` block;
    Gemini 2.5 flash/flash-lite: ``thinking_budget=0``; OpenAI: native
    ``effort="minimal"``). On providers that can't be fully disabled (Anthropic
    Opus 4.6+ adaptive; Gemini 3.x), it routes to the lowest available level.

    ``visibility`` is what the caller sees of the model's reasoning. It is
    orthogonal to ``effort``: visibility is what comes BACK, effort is what
    happens server-side. ``"none"`` requests omission of any returned reasoning
    where the provider supports it.
    """

    effort: Literal["minimal", "low", "medium", "high"] = "medium"
    visibility: Literal["none", "summary", "full"] = "summary"


@dataclass(kw_only=True)
class CacheConfig:
    """Provider-neutral prompt-caching policy.

    *Where* to cache is expressed positionally via :attr:`Message.cache`
    breakpoints. This config only carries the *how*:

    - ``enabled``: master on/off. When off, cache breakpoints are ignored.
    - ``ttl``: how long a cached prefix should live. Honored where the provider
      supports an explicit lifetime (e.g. Gemini ``CachedContent``); ignored by
      providers with purely automatic caching (OpenAI).
    - ``provider_options``: a typed escape hatch for genuinely provider-specific
      handles that cannot be unified — e.g. ``{"gemini_cached_content": "<name>"}``
      to reuse a pre-created Gemini cache instead of letting the adapter manage
      one.

    Always read :attr:`Usage.cached_input_tokens` to confirm cache hits.
    """

    enabled: bool = True
    ttl: Optional[timedelta] = None
    provider_options: Mapping[str, Any] = field(default_factory=dict)


# ───────────────────────────── results & events ────────────────────────────


class FinishReason(str, Enum):
    TOOL_CALLS = "tool_calls"  # model wants tools run -> continue the loop
    STOP = "stop"  # natural completion
    MAX_TOKENS = "max_tokens"
    CONTENT_FILTER = "content_filter"
    PAUSE = "pause"  # e.g. Anthropic pause_turn / long-running operations
    ERROR = "error"


@dataclass(kw_only=True)
class Usage:
    """Token accounting for a step. Aggregate across steps with ``+``.

    ``model_name`` records the concrete provider model id that produced the
    usage (e.g. ``"claude-haiku-4-5-20251001"``, ``"gpt-5.4-nano"``,
    ``"gemini-3.1-flash-lite"``). When summing usages from steps that ran on
    different models, the LEFT operand's ``model_name`` wins; if it is empty,
    the right operand's ``model_name`` is used.

    ``ttft`` is the wall-clock time in seconds from the moment the provider
    request is opened to the first content-bearing event (the first text,
    reasoning, or tool-call signal). It is ``0.0`` when no first token was
    observed (e.g. an empty response or a cancelled stream). When aggregating
    usages across steps, ``__add__`` keeps the minimum non-zero ``ttft`` — that
    is, the earliest "first token" of any step.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # subset of input_tokens served from cache
    reasoning_tokens: int = 0  # subset of output_tokens spent on reasoning
    model_name: str = ""
    ttft: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        # Treat 0.0 as "unset": min of non-zero values, or 0.0 if both unset.
        nonzero_ttfts = [t for t in (self.ttft, other.ttft) if t > 0.0]
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            model_name=self.model_name or other.model_name,
            ttft=min(nonzero_ttfts) if nonzero_ttfts else 0.0,
        )

    def __repr__(self) -> str:
        return json.dumps(self.__dict__, indent=2)


@dataclass(kw_only=True)
class StepResult:
    message: Message  # the assistant turn — append to history as-is
    finish_reason: FinishReason
    usage: Usage

    @property
    def tool_calls(self) -> Sequence[ToolCallPart]:
        return self.message.tool_calls

    @property
    def needs_tools(self) -> bool:
        return bool(self.message.tool_calls)

    def __repr__(self) -> str:
        return f"StepResult(message={self.message}, finish_reason={self.finish_reason}, usage={self.usage})"


@dataclass(kw_only=True)
class StreamEvent:
    """Base class for normalized streaming events."""


@dataclass(kw_only=True)
class TextDelta(StreamEvent):
    text: str


@dataclass(kw_only=True)
class ReasoningDelta(StreamEvent):
    text: str


@dataclass(kw_only=True)
class ToolCallStarted(StreamEvent):
    id: str
    name: str


@dataclass(kw_only=True)
class StepCompleted(StreamEvent):
    result: StepResult


# Runs a single tool call and returns its result part. Caller-supplied.
ToolDispatcher = Callable[[ToolCallPart], Awaitable[ToolResultPart]]
# Optional per-step hook: inspect/approve/EDIT history. Return False to stop
# the loop; return None/True to continue.
StepHook = Callable[[StepResult, list[Message]], Awaitable[Optional[bool]]]


# ───────────────────────────── turn builder (shared) ───────────────────────


class TurnBuilder:
    """Accumulates streamed deltas from ``_decode`` into one assistant Message.

    Providers fold their native events into this builder; reasoning signatures,
    usage, and the finish reason are captured here as the stream progresses.
    Part ordering is preserved as first-seen.
    """

    def __init__(self) -> None:
        self._order: list[Part] = []
        self._text: Optional[TextPart] = None
        self._reasoning: Optional[ReasoningPart] = None
        self._calls: dict[str, ToolCallPart] = {}
        self._argbuf: dict[str, str] = {}
        self.finish_reason = FinishReason.STOP
        self.usage = Usage()

    def text_delta(self, s: str, *, provider_data: Optional[Mapping[str, Any]] = None) -> None:
        if self._text is None:
            self._text = TextPart(text="")
            self._order.append(self._text)
        self._text.text += s
        if provider_data:
            self._text.provider_data.update(provider_data)

    def reasoning_delta(
        self,
        s: str,
        *,
        signature: Union[str, bytes, None] = None,
        visibility: Literal["full", "summary"] = "summary",
        provider_data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self._reasoning is None:
            self._reasoning = ReasoningPart(text="", visibility=visibility)
            self._order.append(self._reasoning)
        self._reasoning.text += s
        if signature is not None:
            self._reasoning.signature = signature
        if provider_data:
            self._reasoning.provider_data.update(provider_data)

    def tool_call(
        self,
        id: str,
        *,
        name: Optional[str] = None,
        args_delta: str = "",
        args: Optional[dict[str, Any]] = None,  # becomes the mutable ToolCallPart.args
        provider_data: Optional[Mapping[str, Any]] = None,
    ) -> ToolCallPart:
        call = self._calls.get(id)
        if call is None:
            call = ToolCallPart(id=id, name=name or "")
            self._calls[id] = call
            self._argbuf[id] = ""
            self._order.append(call)
        if name:
            call.name = name
        if args is not None:  # providers that deliver complete args at once
            call.args = args
        if args_delta:  # providers that stream JSON deltas
            self._argbuf[id] += args_delta
        if provider_data:
            call.provider_data.update(provider_data)
        return call

    def finish(self) -> StepResult:
        for cid, call in self._calls.items():
            if not call.args and self._argbuf.get(cid):
                call.args = json.loads(self._argbuf[cid] or "{}")

        # Drop empty text parts, but keep one that carries provider_data (e.g. a
        # trailing signature-only part some providers emit at end of turn).
        parts = [
            p
            for p in self._order
            if not (isinstance(p, TextPart) and not p.text and not p.provider_data)
        ]

        reason = (
            FinishReason.TOOL_CALLS
            if any(isinstance(p, ToolCallPart) for p in parts)
            else self.finish_reason
        )

        return StepResult(
            message=Message(role=Role.ASSISTANT, parts=parts),
            finish_reason=reason,
            usage=self.usage,
        )


@lru_cache(maxsize=1)
def _prefill_tokenizer() -> tiktoken.Encoding:
    """The gpt-5 (o200k_base) encoding, used as a fast, provider-agnostic local
    estimator for the prefill cache-minimum check."""
    return tiktoken.encoding_for_model("gpt-5")


# ───────────────────────────── the abstract base ───────────────────────────


@dataclass
class _StreamAttempt:
    """One in-flight stream within a hedged ``stream_step``.

    Its events are pumped onto ``queue`` (terminated by a ``None`` sentinel).
    ``first_event`` resolves ``True`` the moment the attempt emits a
    content-bearing event (or completes cleanly without any) and ``False`` if it
    errors before emitting — this is the signal the hedge race selects on. An
    attempt that errors *after* winning records the exception in ``error`` so the
    consumer can re-raise it once the buffered events have drained.
    """

    queue: asyncio.Queue[StreamEvent | None]
    first_event: asyncio.Future[bool]
    task: asyncio.Task[None] | None = None
    error: BaseException | None = None


class ReactGenerator(abc.ABC):
    """Provider-agnostic ReAct generator.

    Subclasses implement the three-method provider seam. The orchestration
    methods (:meth:`stream_step`, :meth:`step`, :meth:`run`) are concrete and
    shared across all providers.
    """

    def __init__(
        self,
        *,
        model: str,
        cache: Optional[CacheConfig] = None,
    ) -> None:
        self.model = model
        self.cache = cache or CacheConfig()

    # ---- model resolution --------------------------------------------------

    def resolve_model(self, hints: ReactGeneratorHints) -> str:
        """The concrete model id this call would use under ``hints`` — public so
        callers (e.g. the engine, when gating tool-event replay) can ask which
        model a turn runs on. Delegates to the :meth:`_resolve_model` hook."""
        return self._resolve_model(hints)

    def _resolve_model(self, hints: ReactGeneratorHints) -> str:
        """Hook: map ``hints['model_size']`` to a concrete model id. Default: the
        generator's identity model; providers that serve several sizes override
        this to consult their ``ModelSize`` → model mapping."""
        return self.model

    # ---- tool-event persistence (provider-decided) -------------------------

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Short, stable provider id ("anthropic" / "openai" / "gemini"). Used to
        decide whether a stored tool-event blob was produced by — and so can be
        replayed by — this generator."""

    def serialize_tool_messages(
        self, messages: Sequence[Message], serializer: ToolMessageSerializer
    ) -> None:
        """Persist a tool turn (the assistant call message + the tool result
        message) through ``serializer``: push the calls/results and a JSON-safe
        provider blob (``provider``/``model`` + any native replay artifacts)."""
        calls = [p for m in messages for p in m.parts if isinstance(p, ToolCallPart)]
        results = [p for m in messages for p in m.parts if isinstance(p, ToolResultPart)]
        serializer.write_calls(calls)
        serializer.write_results(results)
        # Record the model that actually produced the call (stamped at generation),
        # falling back to the identity model for parts that predate the stamp.
        produced_by = (calls[0].provider_data.get(REACT_MODEL_KEY) if calls else None) or self.model
        blob: dict[str, Any] = {"provider": self.provider_name, "model": produced_by}
        self._capture_tool_artifacts(calls, blob)
        serializer.write_provider_data(blob)

    def deserialize_tool_messages(
        self, deserializer: ToolMessageDeserializer, *, model: Optional[str] = None
    ) -> Optional[Sequence[Message]]:
        """Rebuild a tool turn for replay: the assistant ``ToolCallPart`` message
        + the matching ``ToolResultPart`` message, with consistent synthesized
        ids. Returns ``None`` (caller should skip + warn) when the stored blob
        isn't ours or can't be faithfully replayed under the current model.

        ``model`` is the concrete model the replaying turn will run on; providers
        whose native replay is model-bound compare the blob's producing model
        against it. Defaults to the generator's identity model when unknown."""
        blob = deserializer.read_provider_data()
        if blob.get("provider") != self.provider_name:
            return None

        calls = list(deserializer.read_calls())
        results = list(deserializer.read_results())
        if not calls:
            return None

        call_parts: list[ToolCallPart] = []
        result_parts: list[ToolResultPart] = []
        for call, result in zip(calls, results):
            call_id = uuid.uuid4().hex
            call_parts.append(ToolCallPart(id=call_id, name=call.name, args=call.args))
            result_parts.append(
                ToolResultPart(
                    call_id=call_id,
                    name=result.name,
                    content=result.content,
                    is_error=result.is_error,
                )
            )

        if not self._restore_tool_artifacts(call_parts, blob, model=model):
            return None

        return [
            Message(role=Role.ASSISTANT, parts=call_parts),
            Message(role=Role.TOOL, parts=result_parts),
        ]

    def _capture_tool_artifacts(self, calls: Sequence[ToolCallPart], blob: dict[str, Any]) -> None:
        """Hook: add provider-native replay artifacts (per-call, index-aligned) to
        ``blob``. Default: none — Anthropic/OpenAI replay with synthesized ids and
        need nothing stored."""
        return None

    def _restore_tool_artifacts(
        self,
        calls: Sequence[ToolCallPart],
        blob: Mapping[str, Any],
        *,
        model: Optional[str] = None,
    ) -> bool:
        """Hook: reattach artifacts from ``blob`` onto the rebuilt call parts.
        Return ``False`` if the turn can't be faithfully replayed under ``model``
        (the model the replaying turn will run on; ``None`` means unknown — fall
        back to the identity model). Caller skips + warns. Default: ``True``."""
        return True

    # ---- provider seam: implement these three per provider -----------------

    @abc.abstractmethod
    def _encode(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec],
        tool_choice: ToolChoice,
        *,
        reasoning: ReasoningConfig,
        hints: ReactGeneratorHints = {},
    ) -> Any:
        """Translate canonical history + tools into the provider request
        payload. MUST preserve every ``Part.provider_data`` / signature and
        emit the correct block/item type per part. The system prompt, if any, is
        a leading ``Role.SYSTEM`` message in ``history``; ``reasoning`` is per
        call; ``hints`` lets callers override model selection and service tier
        per call."""

    @abc.abstractmethod
    def _raw_stream(self, request: Any) -> AsyncIterator[Any]:
        """Open the provider streaming call and yield its native events.

        Cancellation is cooperative via asyncio: if the consuming task is
        cancelled, the ``async for`` driving this iterator raises
        ``CancelledError`` and the underlying stream is closed.
        """

    @abc.abstractmethod
    def _decode(self, raw_event: Any, builder: TurnBuilder) -> Sequence[StreamEvent]:
        """Map ONE native event to zero or more normalized ``StreamEvent``s and
        fold its content into ``builder`` (capturing signatures, usage, and the
        finish reason)."""

    async def _prefill(self, request: Any) -> Usage:
        """Warm the provider's cache for an encoded request, without producing a
        real turn. Default no-op (for providers/fakes that don't support
        prefilling); the concrete adapters override it. See :meth:`prefill`."""
        return Usage()

    async def _should_prefill(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec],
        hints: ReactGeneratorHints,
        reasoning: Optional[ReasoningConfig] = None,
    ) -> bool:
        """Whether warming the cache is worth it for this prefix. Providers only
        cache prompts above a per-model token minimum; below it, ``cache_control``
        / implicit caching is ignored and a prefill round-trip is wasted. The
        ``reasoning`` config is provided because some providers can only warm a
        cache the real call will read when the prefill matches its thinking
        settings. Default always prefills; the concrete adapters override with a
        token-count check against the resolved model's minimum. See
        :meth:`prefill`."""
        return True

    @staticmethod
    def _estimate_prefill_tokens(
        history: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> int:
        """Estimate the cacheable prefix's token count locally with the gpt-5
        (o200k_base) tokenizer. It is provider-agnostic and within ~10% of every
        provider's own count — close enough to compare against a cache minimum,
        and ~1000x cheaper than a provider ``count_tokens`` round-trip. It runs a
        touch low, so the rare miss is a skipped (not a wasted) prefill."""
        return len(_prefill_tokenizer().encode(ReactGenerator._prefix_text(history, tools)))

    @staticmethod
    def _prefix_text(history: Sequence[Message], tools: Sequence[ToolSpec]) -> str:
        """Flatten the cacheable prefix (system + conversation + tool schemas)
        into plain text for token estimation. Approximate — it ignores
        provider-specific framing tokens, which is fine for comparing against a
        model's cache-size minimum."""
        chunks: list[str] = []

        for message in history:
            for part in message.parts:
                if isinstance(part, (TextPart, ReasoningPart)):
                    chunks.append(part.text)
                elif isinstance(part, ToolCallPart):
                    chunks.append(part.name)
                    chunks.append(str(dict(part.args)))
                elif isinstance(part, ToolResultPart):
                    chunks.append(str(part.content))

        for tool in tools:
            chunks.append(tool.name)
            chunks.append(tool.description)
            for parameter in tool.parameters:
                chunks.append(parameter.name)
                if parameter.description:
                    chunks.append(parameter.description)

        return "\n".join(chunks)

    # ---- concrete orchestration: shared by all providers -------------------

    async def prefill(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        tool_choice: ToolChoice = "auto",
        reasoning: Optional[ReasoningConfig] = None,
        hints: Optional[ReactGeneratorHints] = None,
    ) -> Usage:
        """Warm the provider cache for ``history`` so a subsequent
        :meth:`step` / :meth:`stream_step` reads it instead of paying full input
        cost. Mark the prefix to cache via :attr:`Message.cache_key`, as usual.

        Pass the SAME ``tools`` / ``reasoning`` / ``hints`` you'll use in the real
        call: the cached prefix must be byte-identical for the cache to hit.

        Returns the prefill :class:`Usage` (e.g. ``cache_creation_input_tokens``
        on Anthropic). Best-effort: providers that can't prefill return an empty
        ``Usage``.
        """
        if not await self._should_prefill(history, tools, hints or {}, reasoning):
            return Usage()

        request = self._encode(
            history,
            tools,
            tool_choice,
            reasoning=reasoning or ReasoningConfig(),
            hints=hints or {},
        )

        # Prefill is a single blocking round-trip rather than a stream, so there
        # is no first-token moment to clock. Report the whole operation's
        # wall-clock duration as ``ttft`` — that's the latency a caller pays to
        # warm the cache.
        started_at = time.monotonic()
        usage = await self._prefill(request)
        usage.ttft = time.monotonic() - started_at

        return usage

    async def stream_step(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        tool_choice: ToolChoice = "auto",
        reasoning: Optional[ReasoningConfig] = None,
        hints: Optional[ReactGeneratorHints] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run one inference, yielding normalized events; ends with
        :class:`StepCompleted` carrying the assembled :class:`StepResult`.

        The system prompt, if any, is supplied as a leading ``Role.SYSTEM``
        message in ``history`` (the caller owns it). ``reasoning`` is per call so
        a single generator can serve many turns with different thinking settings.
        ``hints`` may override the underlying model and service tier per call.
        """
        hints = hints or {}
        request = self._encode(
            history,
            tools,
            tool_choice,
            reasoning=reasoning or ReasoningConfig(),
            hints=hints,
        )

        # The concrete model this call runs on. Stamped onto the turn's tool-call
        # parts as they exit, so serialize can record the producing model in the
        # tool-event blob (and a later turn can decide whether to natively replay).
        resolved_model = self._resolve_model(hints)

        # When ``hints['hedge_timeout']`` is a positive number, hedge on TTFT: if
        # the first event hasn't arrived within that window, open a second stream
        # and forward whichever produces its first event first. Otherwise run a
        # single stream with no hedging overhead.
        hedge_timeout = hints.get("hedge_timeout")
        if not (
            isinstance(hedge_timeout, (int, float))
            and not isinstance(hedge_timeout, bool)
            and hedge_timeout > 0
        ):
            async for event in self._stream_attempt(request):
                yield self._stamp_model(event, resolved_model)
            return

        async for event in self._stream_hedged(request, float(hedge_timeout)):
            yield self._stamp_model(event, resolved_model)

    @staticmethod
    def _stamp_model(event: StreamEvent, model: str) -> StreamEvent:
        """Record the producing model on each tool-call part of a completed turn,
        so it rides through to serialization. A no-op for non-terminal events."""
        if isinstance(event, StepCompleted):
            for part in event.result.message.parts:
                if isinstance(part, ToolCallPart):
                    part.provider_data.setdefault(REACT_MODEL_KEY, model)
        return event

    async def _stream_attempt(self, request: Any) -> AsyncIterator[StreamEvent]:
        """Drive one provider stream to completion: yield its normalized events,
        ending with :class:`StepCompleted`.

        TTFT is measured from the moment we open the provider stream to the first
        content-bearing event (text, reasoning, or tool-call signal). It stays
        0.0 if no first token arrives (empty / cancelled stream). We track it in a
        local and stamp it onto ``builder.usage`` *after* the stream loop so
        adapters' final ``builder.usage = self._decode_usage(...)`` assignment
        (which typically lands on the closing usage event) can't clobber it.
        """
        builder = TurnBuilder()
        request_started_at = time.monotonic()
        ttft: float = 0.0
        async for raw in self._raw_stream(request):
            for event in self._decode(raw, builder):
                if ttft == 0.0 and isinstance(event, (TextDelta, ReasoningDelta, ToolCallStarted)):
                    ttft = time.monotonic() - request_started_at
                yield event
        builder.usage.ttft = ttft
        yield StepCompleted(result=builder.finish())

    async def _stream_hedged(
        self,
        request: Any,
        hedge_timeout: float,
    ) -> AsyncIterator[StreamEvent]:
        """Run :meth:`stream_step` with TTFT hedging.

        Open one stream; if it hasn't emitted its first event within
        ``hedge_timeout``, open a second identical stream and forward whichever
        emits first, cancelling the loser. Safe to double-open because a step is a
        pure read with no side effects.
        """
        attempts: list[_StreamAttempt] = []
        try:
            attempts.append(self._spawn_attempt(request))

            done, _ = await asyncio.wait({attempts[0].first_event}, timeout=hedge_timeout)
            if not done:
                # First event didn't arrive in time — fire the hedge.
                attempts.append(self._spawn_attempt(request))

            winner = await self._select_winner(attempts)

            for attempt in attempts:
                if attempt is not winner and attempt.task is not None:
                    attempt.task.cancel()

            while True:
                event = await winner.queue.get()
                if event is None:
                    break
                yield event

            if winner.error is not None:
                raise winner.error
        finally:
            for attempt in attempts:
                if attempt.task is not None and not attempt.task.done():
                    attempt.task.cancel()
            await asyncio.gather(
                *(a.task for a in attempts if a.task is not None),
                return_exceptions=True,
            )

    def _spawn_attempt(self, request: Any) -> _StreamAttempt:
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        first_event: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        attempt = _StreamAttempt(queue=queue, first_event=first_event)
        attempt.task = asyncio.ensure_future(self._pump_attempt(request, attempt))
        return attempt

    async def _pump_attempt(self, request: Any, attempt: _StreamAttempt) -> None:
        """Forward one attempt's events onto its queue, signalling ``first_event``
        on the first content-bearing event so the hedge race can pick a winner the
        instant tokens start."""
        produced_content = False
        try:
            async for event in self._stream_attempt(request):
                attempt.queue.put_nowait(event)
                if not produced_content and isinstance(
                    event, (TextDelta, ReasoningDelta, ToolCallStarted)
                ):
                    produced_content = True
                    if not attempt.first_event.done():
                        attempt.first_event.set_result(True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            attempt.error = exc
        finally:
            attempt.queue.put_nowait(None)
            if not attempt.first_event.done():
                # Terminated before any content event: viable iff it ended cleanly
                # (an empty turn still yields a StepCompleted); not viable if it
                # errored — its error is recorded for the all-failed path.
                attempt.first_event.set_result(attempt.error is None)

    @staticmethod
    async def _select_winner(attempts: list[_StreamAttempt]) -> _StreamAttempt:
        """Return the first attempt to become viable (emit content or complete
        cleanly). If every attempt errors before producing content, re-raise the
        primary (first) attempt's error."""
        by_future = {a.first_event: a for a in attempts}
        pending: set[asyncio.Future[bool]] = set(by_future)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                if future.result():
                    return by_future[future]
        primary_error = attempts[0].error
        assert primary_error is not None
        raise primary_error

    async def step(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        tool_choice: ToolChoice = "auto",
        reasoning: Optional[ReasoningConfig] = None,
        hints: Optional[ReactGeneratorHints] = None,
    ) -> StepResult:
        """Run one inference and return the assembled :class:`StepResult`."""
        result: Optional[StepResult] = None
        async for event in self.stream_step(
            history,
            tools,
            tool_choice=tool_choice,
            reasoning=reasoning,
            hints=hints,
        ):
            if isinstance(event, StepCompleted):
                result = event.result
        assert result is not None, "stream_step must yield a StepCompleted event"
        return result

    async def run(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec],
        dispatch: ToolDispatcher,
        *,
        max_steps: int = 20,
        tool_choice: ToolChoice = "auto",
        reasoning: Optional[ReasoningConfig] = None,
        hints: Optional[ReactGeneratorHints] = None,
        on_step: Optional[StepHook] = None,
    ) -> Sequence[Message]:
        """Convenience ReAct loop.

        Works on a private copy of ``history`` and returns the full conversation
        (the input followed by the turns produced); the caller's ``history`` is
        not mutated. The system prompt, if any, is a leading ``Role.SYSTEM``
        message in ``history``. ``dispatch`` runs a single tool call; the tool
        calls within one turn run concurrently. ``on_step`` receives the working
        conversation (a mutable list it may EDIT in place) and may return
        ``False`` to stop early. ``reasoning`` applies to every step.
        Cancellation is via asyncio (cancel the awaiting task).
        """
        conversation: list[Message] = list(history)

        for _ in range(max_steps):
            result = await self.step(
                conversation,
                tools,
                tool_choice=tool_choice,
                reasoning=reasoning,
                hints=hints,
            )
            conversation.append(result.message)

            if on_step is not None and (await on_step(result, conversation)) is False:
                return conversation

            if not result.needs_tools:
                return conversation

            results = await asyncio.gather(*(dispatch(call) for call in result.tool_calls))
            conversation.append(Message(role=Role.TOOL, parts=list(results)))

        return conversation
