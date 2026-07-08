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

"""Provider-agnostic tests for the ReAct infrastructure (parlant.core.nlp.react).

These exercise the shared core via a scripted fake provider (_FakeReactGenerator):
the canonical message model, ToolSpec/ParameterSpec schema building, TurnBuilder
assembly, the step/stream_step/run orchestration, usage aggregation, history
editing, error propagation, and cancellation. They never touch any real provider;
provider adapters (Gemini, OpenAI, ...) are tested in tests/adapters/nlp/.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence, cast

import pytest

from parlant.core.nlp.react import (
    FinishReason,
    Message,
    ParameterSpec,
    REACT_MODEL_KEY,
    ReactGenerator,
    ReactGeneratorHints,
    ReasoningConfig,
    ReasoningPart,
    Role,
    StepCompleted,
    StepResult,
    StreamEvent,
    TextDelta,
    TextPart,
    ReasoningDelta,
    ToolCallPart,
    ToolCallStarted,
    ToolMessageDeserializer,
    ToolMessageSerializer,
    ToolResultPart,
    ToolSpec,
    TurnBuilder,
    Usage,
    tool_specs_from_tools,
)
from parlant.core.nlp.common import ModelSize
from parlant.core.tools import Tool, ToolOverlap


WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters=[ParameterSpec(name="city", type="string", description="The city name")],
)


# A scripted "raw event" is a callable that folds content into the builder and
# returns the normalized events it produced. This lets each test describe a
# provider's streaming behavior precisely and deterministically.
ScriptedRawEvent = Callable[[TurnBuilder], list[StreamEvent]]


class _FakeReactGenerator(ReactGenerator):
    """A ReactGenerator whose provider stream is a fixed list of scripted events."""

    def __init__(self, script: list[ScriptedRawEvent], **kwargs: Any) -> None:
        super().__init__(model="fake-model", **kwargs)
        self._script = script
        self.encoded_requests: list[dict[str, Any]] = []
        self.raw_stream_started = 0
        self.raw_stream_cancelled = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    def _encode(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec],
        tool_choice: Any,
        *,
        reasoning: Any = None,
        hints: Any = None,
    ) -> Any:
        request = {
            "history": list(history),
            "tools": list(tools),
            "tool_choice": tool_choice,
            "reasoning": reasoning,
            "hints": hints,
        }
        self.encoded_requests.append(request)
        return request

    async def _raw_stream(self, request: Any) -> Any:
        self.raw_stream_started += 1
        try:
            for scripted in self._script:
                yield scripted
        except asyncio.CancelledError:
            self.raw_stream_cancelled += 1
            raise

    def _decode(self, raw_event: Any, builder: TurnBuilder) -> list[StreamEvent]:
        return cast(list[StreamEvent], raw_event(builder))


def _text_event(text: str) -> ScriptedRawEvent:
    def _apply(builder: TurnBuilder) -> list[StreamEvent]:
        builder.text_delta(text)
        return [TextDelta(text=text)]

    return _apply


def _reasoning_event(text: str) -> ScriptedRawEvent:
    def _apply(builder: TurnBuilder) -> list[StreamEvent]:
        builder.reasoning_delta(text)
        return [ReasoningDelta(text=text)]

    return _apply


def _tool_call_event(call_id: str, name: str, args: dict[str, Any]) -> ScriptedRawEvent:
    def _apply(builder: TurnBuilder) -> list[StreamEvent]:
        builder.tool_call(call_id, name=name, args=args)
        return [ToolCallStarted(id=call_id, name=name)]

    return _apply


# ════════════════════════════ 1. SHARED INFRA ══════════════════════════════


async def test_that_turn_builder_merges_consecutive_text_deltas_into_one_part() -> None:
    builder = TurnBuilder()
    builder.text_delta("Hello, ")
    builder.text_delta("world!")

    result = builder.finish()

    assert len(result.message.parts) == 1
    assert isinstance(result.message.parts[0], TextPart)
    assert result.message.text == "Hello, world!"
    assert result.finish_reason == FinishReason.STOP


async def test_that_turn_builder_preserves_part_order_across_kinds() -> None:
    builder = TurnBuilder()
    builder.reasoning_delta("thinking...")
    builder.text_delta("the answer is ")
    builder.tool_call("c1", name="get_weather", args={"city": "Paris"})

    result = builder.finish()
    kinds = [type(p).__name__ for p in result.message.parts]

    assert kinds == ["ReasoningPart", "TextPart", "ToolCallPart"]
    assert result.finish_reason == FinishReason.TOOL_CALLS


async def test_that_turn_builder_accumulates_streamed_json_args() -> None:
    builder = TurnBuilder()
    builder.tool_call("c1", name="search", args_delta='{"qu')
    builder.tool_call("c1", args_delta='ery": "cats"}')

    result = builder.finish()
    call = result.message.tool_calls[0]

    assert call.args == {"query": "cats"}


async def test_that_turn_builder_drops_empty_text_but_keeps_signature_only_part() -> None:
    builder = TurnBuilder()
    builder.text_delta("real content")
    builder.text_delta("", provider_data={"sig": "abc"})  # trailing signature-only delta

    result = builder.finish()

    assert len(result.message.parts) == 1
    assert result.message.parts[0].provider_data == {"sig": "abc"}
    assert result.message.text == "real content"


async def test_that_turn_builder_finish_reason_is_overridden_by_tool_calls() -> None:
    builder = TurnBuilder()
    builder.finish_reason = FinishReason.MAX_TOKENS
    builder.tool_call("c1", name="x", args={})

    assert builder.finish().finish_reason == FinishReason.TOOL_CALLS


async def test_that_usage_aggregates_with_addition() -> None:
    total = Usage(
        input_tokens=10, output_tokens=5, cached_input_tokens=2, reasoning_tokens=1
    ) + Usage(input_tokens=3, output_tokens=4, cached_input_tokens=1, reasoning_tokens=2)

    assert total == Usage(
        input_tokens=13, output_tokens=9, cached_input_tokens=3, reasoning_tokens=3
    )


async def test_that_message_properties_partition_parts_by_kind() -> None:
    message = Message(
        role=Role.ASSISTANT,
        parts=[
            ReasoningPart(text="hmm"),
            TextPart(text="hi "),
            TextPart(text="there"),
            ToolCallPart(id="c1", name="t", args={}),
        ],
    )

    assert message.text == "hi there"
    assert message.reasoning == "hmm"
    assert [c.id for c in message.tool_calls] == ["c1"]


def test_that_tool_spec_renders_object_json_schema() -> None:
    spec = ToolSpec(
        name="get_weather",
        description="d",
        parameters=[
            ParameterSpec(name="city", type="string", description="The city name"),
            ParameterSpec(name="units", type="string", enum=["c", "f"], required=False),
        ],
    )

    assert spec.json_schema() == {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city name"},
            "units": {"type": "string", "enum": ["c", "f"]},
        },
        "required": ["city"],  # only required params are listed
    }


def test_that_tool_spec_with_no_parameters_has_no_required_key() -> None:
    schema = ToolSpec(name="now", description="d").json_schema()

    assert schema == {"type": "object", "properties": {}}


def test_that_parameter_spec_renders_arrays_and_nested_objects() -> None:
    spec = ParameterSpec(
        name="filters",
        type="object",
        properties=[
            ParameterSpec(
                name="tags",
                type="array",
                items=ParameterSpec(name="tag", type="string"),
            ),
            ParameterSpec(name="limit", type="integer", required=False),
        ],
    )

    assert spec.value_schema() == {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer"},
        },
        "required": ["tags"],
    }


def test_that_parameter_spec_renders_nullable_and_default() -> None:
    # An explicit default of None must be emitted (distinct from "no default").
    with_none_default = ParameterSpec(name="p", type="string", nullable=True, default=None)
    assert with_none_default.value_schema() == {
        "type": "string",
        "nullable": True,
        "default": None,
    }

    # A non-None default is emitted; an unset default is omitted entirely.
    assert ParameterSpec(name="p", type="integer", default=5).value_schema() == {
        "type": "integer",
        "default": 5,
    }
    assert ParameterSpec(name="p", type="integer").value_schema() == {"type": "integer"}


async def test_that_step_assembles_a_step_result_from_scripted_stream() -> None:
    generator = _FakeReactGenerator(
        [_reasoning_event("let me think"), _text_event("The answer is 42.")]
    )

    result = await generator.step([Message(role=Role.USER, parts=[TextPart(text="q")])])

    assert result.message.text == "The answer is 42."
    assert result.message.reasoning == "let me think"
    assert result.finish_reason == FinishReason.STOP
    assert not result.needs_tools


async def test_that_step_reports_tool_calls_as_needs_tools() -> None:
    generator = _FakeReactGenerator([_tool_call_event("c1", "get_weather", {"city": "Paris"})])

    result = await generator.step([Message(role=Role.USER, parts=[TextPart(text="weather?")])])

    assert result.needs_tools
    assert result.finish_reason == FinishReason.TOOL_CALLS
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].args == {"city": "Paris"}


async def test_that_stream_step_yields_events_in_order_ending_with_step_completed() -> None:
    generator = _FakeReactGenerator([_text_event("a"), _text_event("b")])

    events = [
        event
        async for event in generator.stream_step(
            [Message(role=Role.USER, parts=[TextPart(text="q")])]
        )
    ]

    assert isinstance(events[0], TextDelta) and events[0].text == "a"
    assert isinstance(events[1], TextDelta) and events[1].text == "b"
    assert isinstance(events[-1], StepCompleted)
    assert events[-1].result.message.text == "ab"


async def test_that_stream_step_surfaces_tool_call_started_before_completion() -> None:
    generator = _FakeReactGenerator(
        [_text_event("calling now"), _tool_call_event("c1", "get_weather", {"city": "Paris"})]
    )

    events = [
        event
        async for event in generator.stream_step(
            [Message(role=Role.USER, parts=[TextPart(text="weather?")])]
        )
    ]
    kinds = [type(e).__name__ for e in events]

    assert kinds == ["TextDelta", "ToolCallStarted", "StepCompleted"]
    started = next(e for e in events if isinstance(e, ToolCallStarted))
    assert started.id == "c1" and started.name == "get_weather"


async def test_that_run_executes_tools_and_loops_until_no_tools() -> None:
    # Step 1 asks for a tool; step 2 (after the tool result) answers in text.
    script_by_step = [
        [_tool_call_event("c1", "get_weather", {"city": "Paris"})],
        [_text_event("It is sunny in Paris.")],
    ]

    class _MultiStep(_FakeReactGenerator):
        def __init__(self) -> None:
            super().__init__([])
            self._step_index = 0

        async def _raw_stream(self, request: Any) -> Any:
            script = script_by_step[self._step_index]
            self._step_index += 1
            for scripted in script:
                yield scripted

    generator = _MultiStep()
    dispatched: list[ToolCallPart] = []

    async def dispatch(call: ToolCallPart) -> ToolResultPart:
        dispatched.append(call)
        return ToolResultPart(call_id=call.id, name=call.name, content="sunny")

    history: list[Message] = [Message(role=Role.USER, parts=[TextPart(text="weather in Paris?")])]
    result_history = await generator.run(history, [WEATHER_TOOL], dispatch)

    assert dispatched[0].name == "get_weather"
    # user, assistant(tool_call), tool(result), assistant(text)
    assert [m.role for m in result_history] == [
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
    ]
    assert result_history[-1].text == "It is sunny in Paris."
    assert result_history[2].tool_results[0].content == "sunny"


async def test_that_run_dispatches_parallel_tool_calls_in_a_single_turn() -> None:
    script_by_step = [
        [
            _tool_call_event("c1", "get_weather", {"city": "Paris"}),
            _tool_call_event("c2", "get_weather", {"city": "Tokyo"}),
        ],
        [_text_event("Paris and Tokyo are both sunny.")],
    ]

    class _MultiStep(_FakeReactGenerator):
        def __init__(self) -> None:
            super().__init__([])
            self._step_index = 0

        async def _raw_stream(self, request: Any) -> Any:
            script = script_by_step[self._step_index]
            self._step_index += 1
            for scripted in script:
                yield scripted

    generator = _MultiStep()
    dispatched: list[str] = []

    async def dispatch(call: ToolCallPart) -> ToolResultPart:
        dispatched.append(call.args["city"])
        return ToolResultPart(
            call_id=call.id, name=call.name, content=f"sunny in {call.args['city']}"
        )

    history: list[Message] = [Message(role=Role.USER, parts=[TextPart(text="weather in both?")])]
    result_history = await generator.run(history, [WEATHER_TOOL], dispatch)

    assert sorted(dispatched) == ["Paris", "Tokyo"]
    tool_message = next(m for m in result_history if m.role == Role.TOOL)
    assert len(tool_message.tool_results) == 2
    assert {r.call_id for r in tool_message.tool_results} == {"c1", "c2"}


async def test_that_tool_call_provider_data_merges_across_deltas() -> None:
    builder = TurnBuilder()
    builder.tool_call("c1", name="t", args={"a": 1}, provider_data={"sig": "first"})
    builder.tool_call("c1", provider_data={"extra": "second"})

    call = builder.finish().message.tool_calls[0]

    assert call.provider_data == {"sig": "first", "extra": "second"}


async def test_that_run_stops_early_when_on_step_returns_false() -> None:
    generator = _FakeReactGenerator([_tool_call_event("c1", "get_weather", {"city": "Paris"})])

    async def dispatch(call: ToolCallPart) -> ToolResultPart:
        raise AssertionError("dispatch should not run when on_step stops the loop")

    async def on_step(result: StepResult, history: list[Message]) -> bool:
        return False

    history: list[Message] = [Message(role=Role.USER, parts=[TextPart(text="q")])]
    result_history = await generator.run(history, [WEATHER_TOOL], dispatch, on_step=on_step)

    assert [m.role for m in result_history] == [Role.USER, Role.ASSISTANT]


async def test_that_on_step_can_edit_history_in_place() -> None:
    generator = _FakeReactGenerator([_text_event("final")])

    async def dispatch(call: ToolCallPart) -> ToolResultPart:  # pragma: no cover - no tools here
        raise AssertionError

    async def on_step(result: StepResult, history: list[Message]) -> None:
        # Surgery: drop everything except the latest assistant turn.
        history[:] = history[-1:]

    history: list[Message] = [
        Message(role=Role.USER, parts=[TextPart(text="one")]),
        Message(role=Role.USER, parts=[TextPart(text="two")]),
    ]
    result_history = await generator.run(history, [], dispatch, on_step=on_step)

    assert len(result_history) == 1
    assert result_history[0].role == Role.ASSISTANT


async def test_that_run_respects_max_steps() -> None:
    # Every step asks for a tool, so without a cap the loop would never end.
    class _AlwaysTools(_FakeReactGenerator):
        def __init__(self) -> None:
            super().__init__([])
            self.steps = 0

        async def _raw_stream(self, request: Any) -> Any:
            self.steps += 1
            yield _tool_call_event(f"c{self.steps}", "get_weather", {"city": "X"})

    generator = _AlwaysTools()

    async def dispatch(call: ToolCallPart) -> ToolResultPart:
        return ToolResultPart(call_id=call.id, name=call.name, content="ok")

    history: list[Message] = [Message(role=Role.USER, parts=[TextPart(text="q")])]
    await generator.run(history, [WEATHER_TOOL], dispatch, max_steps=3)

    assert generator.steps == 3


async def test_that_provider_errors_propagate_out_of_step() -> None:
    class _Boom(ReactGenerator):
        def __init__(self) -> None:
            super().__init__(model="fake")

        @property
        def provider_name(self) -> str:
            return "fake"

        def _encode(
            self,
            history: Sequence[Message],
            tools: Sequence[ToolSpec],
            tool_choice: Any,
            *,
            reasoning: Any = None,
            hints: Any = None,
        ) -> Any:
            return {}

        async def _raw_stream(self, request: Any) -> Any:
            yield None  # one event, then fail
            raise RuntimeError("provider exploded")

        def _decode(self, raw_event: Any, builder: TurnBuilder) -> list[StreamEvent]:
            return []

    with pytest.raises(RuntimeError, match="provider exploded"):
        await _Boom().step([Message(role=Role.USER, parts=[TextPart(text="q")])])


async def test_that_call_options_are_threaded_through_to_encode() -> None:
    generator = _FakeReactGenerator([_text_event("ok")])
    reasoning = ReasoningConfig(effort="medium")

    await generator.step(
        [Message(role=Role.USER, parts=[TextPart(text="q")])],
        [WEATHER_TOOL],
        tool_choice={"name": "get_weather"},
        reasoning=reasoning,
    )

    request = generator.encoded_requests[0]
    assert request["tool_choice"] == {"name": "get_weather"}
    assert request["tools"] == [WEATHER_TOOL]
    assert request["reasoning"] is reasoning


async def test_that_prefill_encodes_the_history_and_defaults_to_a_no_op_usage() -> None:
    generator = _FakeReactGenerator([_text_event("ok")])

    usage = await generator.prefill([Message(role=Role.USER, parts=[TextPart(text="q")])])

    # prefill encodes (so the provider can warm its cache) and, with no override,
    # reports no token usage — but ttft always covers the operation's duration.
    assert len(generator.encoded_requests) == 1
    assert usage.ttft >= 0.0
    assert usage == Usage(ttft=usage.ttft)


def test_that_prefix_text_flattens_messages_and_tool_schemas_for_token_counting() -> None:
    history = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="you are helpful")]),
        Message(role=Role.USER, parts=[TextPart(text="weather in paris?")]),
    ]

    text = _FakeReactGenerator([])._prefix_text(history, [WEATHER_TOOL])

    assert "you are helpful" in text
    assert "weather in paris?" in text
    # Tool name + description feed into the count too.
    assert WEATHER_TOOL.name in text
    assert WEATHER_TOOL.description in text


async def test_that_cancelling_a_step_propagates_and_tears_down_the_stream() -> None:
    started = asyncio.Event()

    class _Hanging(ReactGenerator):
        def __init__(self) -> None:
            super().__init__(model="fake")
            self.cancelled = False

        @property
        def provider_name(self) -> str:
            return "fake"

        def _encode(
            self,
            history: Sequence[Message],
            tools: Sequence[ToolSpec],
            tool_choice: Any,
            *,
            reasoning: Any = None,
            hints: Any = None,
        ) -> Any:
            return {}

        async def _raw_stream(self, request: Any) -> Any:
            started.set()
            try:
                await asyncio.Event().wait()  # block forever
                yield None  # pragma: no cover
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        def _decode(self, raw_event: Any, builder: TurnBuilder) -> list[StreamEvent]:
            return []  # pragma: no cover

    generator = _Hanging()
    task = asyncio.ensure_future(
        generator.step([Message(role=Role.USER, parts=[TextPart(text="q")])])
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert generator.cancelled


class _StreamingProvider(ReactGenerator):
    """A fake provider that opens a 'connection', streams forever, and closes it
    in a finally — mirroring how real adapters close the provider stream. The
    ``stream_closed`` flag lets tests assert cancellation tears the stream down."""

    def __init__(self) -> None:
        super().__init__(model="fake")
        self.started = asyncio.Event()
        self.stream_closed = False

    @property
    def provider_name(self) -> str:
        return "fake"

    def _encode(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec],
        tool_choice: Any,
        *,
        reasoning: Any = None,
        hints: Any = None,
    ) -> Any:
        return {}

    async def _raw_stream(self, request: Any) -> Any:
        self.started.set()
        try:
            while True:
                await asyncio.sleep(0.005)
                yield _text_event("tick")
        finally:
            self.stream_closed = True

    def _decode(self, raw_event: Any, builder: TurnBuilder) -> list[StreamEvent]:
        return cast(list[StreamEvent], raw_event(builder))


async def test_that_cancelling_mid_step_closes_the_provider_stream() -> None:
    generator = _StreamingProvider()
    task = asyncio.ensure_future(
        generator.step([Message(role=Role.USER, parts=[TextPart(text="q")])])
    )
    await generator.started.wait()
    await asyncio.sleep(0.02)  # let it stream a bit
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    # The _raw_stream finally ran — a real adapter closes the HTTP stream here.
    assert generator.stream_closed


async def test_that_cancelling_mid_stream_closes_the_provider_stream() -> None:
    generator = _StreamingProvider()
    received: list[StreamEvent] = []

    async def consume() -> None:
        async for event in generator.stream_step(
            [Message(role=Role.USER, parts=[TextPart(text="q")])]
        ):
            received.append(event)

    task = asyncio.ensure_future(consume())
    await generator.started.wait()
    await asyncio.sleep(0.02)  # receive several events mid-stream
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert received  # events were delivered before cancellation
    assert generator.stream_closed  # ...and the stream was still torn down


def _parlant_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    required: list[str],
) -> Tool:
    return Tool(
        name=name,
        creation_utc=datetime.now(timezone.utc),
        description=description,
        metadata={},
        parameters={k: (v, None) for k, v in parameters.items()},  # type: ignore[misc]
        required=required,
        consequential=False,
        overlap=ToolOverlap.NONE,
    )


def test_that_parlant_tools_convert_to_tool_specs() -> None:
    tool = _parlant_tool(
        "charge_card",
        "Charge the customer's card.",
        parameters={
            "amount": {"type": "number", "description": "The amount to charge."},
            "tags": {"type": "array", "item_type": "string"},
            "when": {"type": "datetime"},  # no JSON Schema primitive -> string
        },
        required=["amount"],
    )

    spec = tool_specs_from_tools([tool])[0]

    assert spec.name == "charge_card"
    assert spec.description == "Charge the customer's card."
    by_name = {p.name: p for p in spec.parameters}
    assert by_name["amount"].type == "number" and by_name["amount"].required
    assert by_name["tags"].type == "array" and by_name["tags"].items is not None
    assert by_name["tags"].items.type == "string"
    assert not by_name["tags"].required
    # Exotic Parlant types fall back to "string".
    assert by_name["when"].type == "string"


# ============================================================================
# stream_step TTFT Hedging
# ============================================================================


@dataclass
class _AttemptScript:
    """Scripts one `_raw_stream` invocation: how long until its first event,
    the events it then emits, and whether it errors instead (after the delay)."""

    first_event_delay: float
    events: list[ScriptedRawEvent] = field(default_factory=list)
    error: BaseException | None = None


class _HedgingProvider(ReactGenerator):
    """A provider whose Nth `_raw_stream` call runs the Nth scripted attempt (the
    last repeats), so the base class's TTFT hedging can be exercised."""

    def __init__(self, attempts: list[_AttemptScript]) -> None:
        super().__init__(model="fake-model")
        self._attempts = attempts
        self.raw_stream_started = 0
        self.cancelled_attempts: list[int] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    def _encode(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec],
        tool_choice: Any,
        *,
        reasoning: Any = None,
        hints: Any = None,
    ) -> Any:
        return {}

    async def _raw_stream(self, request: Any) -> Any:
        index = self.raw_stream_started
        self.raw_stream_started += 1
        spec = self._attempts[min(index, len(self._attempts) - 1)]
        try:
            await asyncio.sleep(spec.first_event_delay)
            if spec.error is not None:
                raise spec.error
            for scripted in spec.events:
                yield scripted
        except asyncio.CancelledError:
            self.cancelled_attempts.append(index)
            raise

    def _decode(self, raw_event: Any, builder: TurnBuilder) -> list[StreamEvent]:
        return cast(list[StreamEvent], raw_event(builder))


async def _collect(generator: ReactGenerator, **hints: Any) -> StepResult:
    result: StepResult | None = None
    async for event in generator.stream_step(
        [Message(role=Role.USER, parts=[TextPart(text="q")])],
        hints=cast(Any, hints) if hints else None,
    ):
        if isinstance(event, StepCompleted):
            result = event.result
    assert result is not None
    return result


async def test_that_a_slow_first_event_triggers_a_hedge_and_the_faster_stream_wins() -> None:
    generator = _HedgingProvider(
        [
            _AttemptScript(first_event_delay=5.0, events=[_text_event("slow")]),
            _AttemptScript(first_event_delay=0.0, events=[_text_event("fast")]),
        ]
    )

    result = await _collect(generator, hedge_timeout=0.05)

    assert generator.raw_stream_started == 2
    assert result.message.text == "fast"
    assert generator.cancelled_attempts == [0]


async def test_that_a_fast_first_event_is_not_hedged() -> None:
    generator = _HedgingProvider(
        [_AttemptScript(first_event_delay=0.0, events=[_text_event("quick")])]
    )

    result = await _collect(generator, hedge_timeout=0.5)

    assert generator.raw_stream_started == 1
    assert result.message.text == "quick"


async def test_that_no_hedge_is_attempted_without_the_hint() -> None:
    generator = _HedgingProvider(
        [_AttemptScript(first_event_delay=0.05, events=[_text_event("only")])]
    )

    result = await _collect(generator)

    assert generator.raw_stream_started == 1
    assert result.message.text == "only"


async def test_that_the_original_stream_still_wins_if_it_emits_before_the_hedge() -> None:
    # Primary is slow enough to trigger the hedge, but still emits before the
    # (even slower) hedge — so the primary wins and the hedge is cancelled.
    generator = _HedgingProvider(
        [
            _AttemptScript(first_event_delay=0.1, events=[_text_event("primary")]),
            _AttemptScript(first_event_delay=0.5, events=[_text_event("hedge")]),
        ]
    )

    result = await _collect(generator, hedge_timeout=0.05)

    assert generator.raw_stream_started == 2
    assert result.message.text == "primary"
    assert generator.cancelled_attempts == [1]


async def test_that_a_hedge_raises_the_primary_error_when_both_streams_fail() -> None:
    generator = _HedgingProvider(
        [
            _AttemptScript(first_event_delay=0.1, error=RuntimeError("primary-failure")),
            _AttemptScript(first_event_delay=0.01, error=RuntimeError("hedge-failure")),
        ]
    )

    with pytest.raises(RuntimeError, match="primary-failure"):
        await _collect(generator, hedge_timeout=0.05)

    assert generator.raw_stream_started == 2


# ═══════════════════ 5. TOOL-EVENT MODEL STAMPING / REPLAY ══════════════════
#
# A persisted tool event records, in its provider blob, the model that produced
# it — the *resolved* per-call model (model_size hint), not the generator's
# static identity model. On replay the provider may gate native replay on that
# model. These exercise the provider-agnostic plumbing: stamping the resolved
# model onto the call part, serializing it into the blob, and forwarding the
# replay model down to the restore hook.


class _SizedFakeReactGenerator(_FakeReactGenerator):
    """A fake whose ``model_size`` hint resolves to a distinct concrete model,
    mirroring how real providers map ``ModelSize`` onto a model id."""

    def _resolve_model(self, hints: ReactGeneratorHints) -> str:
        if hints.get("model_size") == ModelSize.MEDIUM:
            return "big-model"
        return self.model


class _RecordingRestoreReactGenerator(_FakeReactGenerator):
    """Captures the ``model`` argument forwarded into the restore hook."""

    def __init__(self, script: list[ScriptedRawEvent], **kwargs: Any) -> None:
        super().__init__(script, **kwargs)
        self.restored_with_model: str | None = None

    def _restore_tool_artifacts(
        self, calls: Sequence[ToolCallPart], blob: Any, *, model: str | None = None
    ) -> bool:
        self.restored_with_model = model
        return True


class _DictToolMessageStore(ToolMessageSerializer, ToolMessageDeserializer):
    """In-memory round-trip store backing both the serialize and replay sides."""

    def __init__(self) -> None:
        self._calls: list[ToolCallPart] = []
        self._results: list[ToolResultPart] = []
        self._provider_data: dict[str, Any] = {}

    def write_calls(self, calls: Sequence[ToolCallPart]) -> None:
        self._calls = list(calls)

    def write_results(self, results: Sequence[ToolResultPart]) -> None:
        self._results = list(results)

    def write_provider_data(self, data: Any) -> None:
        self._provider_data = dict(data)

    def read_calls(self) -> Sequence[ToolCallPart]:
        return self._calls

    def read_results(self) -> Sequence[ToolResultPart]:
        return self._results

    def read_provider_data(self) -> Any:
        return self._provider_data


async def test_that_a_tool_call_part_is_stamped_with_the_resolved_model() -> None:
    generator = _SizedFakeReactGenerator([_tool_call_event("c1", "get_weather", {"city": "Paris"})])

    result = await generator.step(
        [Message(role=Role.USER, parts=[TextPart(text="weather?")])],
        hints={"model_size": ModelSize.MEDIUM},
    )

    call = next(p for p in result.message.parts if isinstance(p, ToolCallPart))
    assert call.provider_data[REACT_MODEL_KEY] == "big-model"


async def test_that_serialize_records_the_stamped_model_not_the_default() -> None:
    generator = _FakeReactGenerator([])  # self.model == "fake-model"
    store = _DictToolMessageStore()

    call = ToolCallPart(id="c1", name="get_weather", args={"city": "Paris"})
    call.provider_data[REACT_MODEL_KEY] = "big-model"
    result = ToolResultPart(call_id="c1", name="get_weather", content="sunny")

    generator.serialize_tool_messages(
        [Message(role=Role.ASSISTANT, parts=[call]), Message(role=Role.TOOL, parts=[result])],
        store,
    )

    assert store.read_provider_data()["model"] == "big-model"


async def test_that_serialize_falls_back_to_self_model_without_a_stamp() -> None:
    generator = _FakeReactGenerator([])  # self.model == "fake-model"
    store = _DictToolMessageStore()

    call = ToolCallPart(id="c1", name="get_weather", args={"city": "Paris"})
    result = ToolResultPart(call_id="c1", name="get_weather", content="sunny")

    generator.serialize_tool_messages(
        [Message(role=Role.ASSISTANT, parts=[call]), Message(role=Role.TOOL, parts=[result])],
        store,
    )

    assert store.read_provider_data()["model"] == "fake-model"


async def test_that_deserialize_forwards_the_replay_model_to_restore_artifacts() -> None:
    generator = _RecordingRestoreReactGenerator([])
    store = _DictToolMessageStore()

    call = ToolCallPart(id="c1", name="get_weather", args={"city": "Paris"})
    result = ToolResultPart(call_id="c1", name="get_weather", content="sunny")
    generator.serialize_tool_messages(
        [Message(role=Role.ASSISTANT, parts=[call]), Message(role=Role.TOOL, parts=[result])],
        store,
    )

    generator.deserialize_tool_messages(store, model="replay-model")

    assert generator.restored_with_model == "replay-model"
