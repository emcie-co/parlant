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
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
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
    args: dict[str, Any] = field(default_factory=dict)
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
    parts: list[Part] = field(default_factory=list)
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


# "auto" | "none" | "required", or {"name": "<tool>"} to force one specific tool.
ToolChoice = Union[Literal["auto", "none", "required"], Mapping[str, str]]


@dataclass(kw_only=True)
class ReasoningConfig:
    """Reasoning/thinking knobs, mapped per provider in ``_encode``.

    Anthropic -> ``thinking={"type": "enabled", "budget_tokens": budget_tokens}``
    OpenAI    -> ``reasoning={"effort": effort, "summary": "auto"}``
    Gemini    -> ``thinking_config(include_thoughts=True, thinking_budget=...)``
                 on 2.5; 3.x keys ``thinking_level`` off ``effort``.
    """

    enabled: bool = False
    effort: Literal["minimal", "low", "medium", "high"] = "medium"
    budget_tokens: Optional[int] = None
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
    """Token accounting for a step. Aggregate across steps with ``+``."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # subset of input_tokens served from cache
    reasoning_tokens: int = 0  # subset of output_tokens spent on reasoning

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


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


# ───────────────────────────── the abstract base ───────────────────────────


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

    # ---- provider seam: implement these three per provider -----------------

    @abc.abstractmethod
    def _encode(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec],
        tool_choice: ToolChoice,
        *,
        system: Optional[str],
        reasoning: ReasoningConfig,
    ) -> Any:
        """Translate canonical history + tools into the provider request
        payload. MUST preserve every ``Part.provider_data`` / signature and
        emit the correct block/item type per part. ``system`` and ``reasoning``
        are supplied per call."""

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

    # ---- concrete orchestration: shared by all providers -------------------

    async def stream_step(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        tool_choice: ToolChoice = "auto",
        system: Optional[str] = None,
        reasoning: Optional[ReasoningConfig] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run one inference, yielding normalized events; ends with
        :class:`StepCompleted` carrying the assembled :class:`StepResult`.

        ``system`` and ``reasoning`` are per call so a single generator can serve
        many agents/turns with different prompts and thinking settings.
        """
        request = self._encode(
            history,
            tools,
            tool_choice,
            system=system,
            reasoning=reasoning or ReasoningConfig(),
        )
        builder = TurnBuilder()
        async for raw in self._raw_stream(request):
            for event in self._decode(raw, builder):
                yield event
        yield StepCompleted(result=builder.finish())

    async def step(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        tool_choice: ToolChoice = "auto",
        system: Optional[str] = None,
        reasoning: Optional[ReasoningConfig] = None,
    ) -> StepResult:
        """Run one inference and return the assembled :class:`StepResult`."""
        result: Optional[StepResult] = None
        async for event in self.stream_step(
            history, tools, tool_choice=tool_choice, system=system, reasoning=reasoning
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
        system: Optional[str] = None,
        reasoning: Optional[ReasoningConfig] = None,
        on_step: Optional[StepHook] = None,
    ) -> Sequence[Message]:
        """Convenience ReAct loop.

        Works on a private copy of ``history`` and returns the full conversation
        (the input followed by the turns produced); the caller's ``history`` is
        not mutated. ``dispatch`` runs a single tool call; the tool calls within
        one turn run concurrently. ``on_step`` receives the working conversation
        (a mutable list it may EDIT in place) and may return ``False`` to stop
        early. ``system`` and ``reasoning`` apply to every step. Cancellation is
        via asyncio (cancel the awaiting task).
        """
        conversation: list[Message] = list(history)

        for _ in range(max_steps):
            result = await self.step(
                conversation, tools, tool_choice=tool_choice, system=system, reasoning=reasoning
            )
            conversation.append(result.message)

            if on_step is not None and (await on_step(result, conversation)) is False:
                return conversation

            if not result.needs_tools:
                return conversation

            results = await asyncio.gather(*(dispatch(call) for call in result.tool_calls))
            conversation.append(Message(role=Role.TOOL, parts=list(results)))

        return conversation
