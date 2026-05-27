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
from typing import Any, Callable, cast

import pytest

from parlant.core.nlp.react import (
    FinishReason,
    Message,
    ParameterSpec,
    ReactGenerator,
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
    ToolResultPart,
    ToolSpec,
    TurnBuilder,
    Usage,
)


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

    def _encode(
        self,
        history: list[Message],
        tools: list[ToolSpec],
        tool_choice: Any,
        *,
        system: Any = None,
        reasoning: Any = None,
    ) -> Any:
        request = {
            "history": list(history),
            "tools": list(tools),
            "tool_choice": tool_choice,
            "system": system,
            "reasoning": reasoning,
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

        def _encode(
            self,
            history: list[Message],
            tools: list[ToolSpec],
            tool_choice: Any,
            *,
            system: Any = None,
            reasoning: Any = None,
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
    reasoning = ReasoningConfig(enabled=True)

    await generator.step(
        [Message(role=Role.USER, parts=[TextPart(text="q")])],
        [WEATHER_TOOL],
        tool_choice={"name": "get_weather"},
        system="be helpful",
        reasoning=reasoning,
    )

    request = generator.encoded_requests[0]
    assert request["tool_choice"] == {"name": "get_weather"}
    assert request["tools"] == [WEATHER_TOOL]
    assert request["system"] == "be helpful"
    assert request["reasoning"] is reasoning


async def test_that_cancelling_a_step_propagates_and_tears_down_the_stream() -> None:
    started = asyncio.Event()

    class _Hanging(ReactGenerator):
        def __init__(self) -> None:
            super().__init__(model="fake")
            self.cancelled = False

        def _encode(
            self,
            history: list[Message],
            tools: list[ToolSpec],
            tool_choice: Any,
            *,
            system: Any = None,
            reasoning: Any = None,
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
