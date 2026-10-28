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
from typing import Any, cast

import pytest

from parlant.core import metadata as metadata_utils
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.compass.hooks import CompassHookResult
from parlant.core.engines.compass.loop.base_loop import _LoopState, _ToolPreambleState
from parlant.core.engines.compass.loop.loop import LoopJob
from parlant.core.engines.compass.loop.streaming_loop import StreamingLoop
from parlant.core.engines.compass.preambles import (
    DEFAULT_PREAMBLE_INTERVAL_SECONDS,
    PreambleConfiguration,
)
from parlant.core.engines.compass.context import EngineContext, ResponseState
from parlant.core.loggers import StdoutLogger
from parlant.core.nlp.react import (
    FinishReason,
    Message,
    ReasoningConfig,
    Role,
    StepCompleted,
    StepResult,
    TextDelta,
    TextPart,
    ToolCallPart,
    ToolCallStarted,
    Usage,
)
from parlant.core.sessions import EventKind, EventSource
from parlant.core.nlp.tokenization import ZeroEstimatingTokenizer
from parlant.core.tracer import LocalTracer
from parlant.core.cost_control import NullCostControlPolicy
from parlant.core.usage_reporter import UsageReporter

from tests.core.stable.engines.compass.matching.utils import create_engine_context
from parlant.core.application_context import ApplicationContext
from parlant.core.health import NullHealthReporter


def _health_reporter() -> NullHealthReporter:
    return NullHealthReporter(ApplicationContext(instance_id="test-instance"))


class _ModelResolvingStubReact:
    """Minimal react stub for history replay model resolution."""

    provider_name = "fake"

    def resolve_model(self, hints: Any) -> str:
        return "fake-model"


class _ModelRouterStub:
    def __init__(self) -> None:
        self.react = _ModelResolvingStubReact()
        self.tokenizer = ZeroEstimatingTokenizer()
        self.react_reasoning: ReasoningConfig | None = None

    async def get_react_generator(self, **kwargs: Any) -> Any:
        return self.react

    async def get_estimating_tokenizer(self, **kwargs: Any) -> Any:
        return self.tokenizer

    async def resolve_react_reasoning(self, **kwargs: Any) -> ReasoningConfig | None:
        return self.react_reasoning


def _make_streaming_loop(tracer: LocalTracer | None = None) -> StreamingLoop:
    # _build_history only reads the LoopJob, so the heavier collaborators
    # (meter/optimization_policy/react/tool_runner) aren't exercised here — except
    # the react's resolve_model, which history building uses for tool-event replay.
    tracer = tracer or LocalTracer()
    logger = StdoutLogger(tracer)

    return StreamingLoop(
        logger=logger,
        tracer=tracer,
        meter=cast(Any, None),
        model_router=cast(Any, _ModelRouterStub()),
        tool_runner=cast(Any, None),
        reviewer=cast(Any, None),
        usage_reporter=UsageReporter(tracer),
        cost_control_policy=NullCostControlPolicy(),
        health_reporter=_health_reporter(),
    )


def _encouraged_preamble_state() -> _ToolPreambleState:
    return _ToolPreambleState(PreambleConfiguration.encourage())


def test_that_the_base_loop_uses_the_full_message_text_in_history_regardless_of_voice_metadata() -> (
    None
):
    # The base loop is voice-agnostic: even when an event carries voice delivery
    # metadata, it renders the full generated text, never the delivered subset.
    loop = _make_streaming_loop()

    text = loop._historical_assistant_text(
        cast(Any, {"message": "The whole sentence."}),
        {"voice": {"spoken_text": "The whole"}},
    )

    assert text == "The whole sentence."


async def test_that_turn_instructions_are_placed_before_the_last_customer_message() -> None:
    context = create_engine_context(
        conversation=[
            (EventSource.CUSTOMER, "hi"),
            (EventSource.AI_AGENT, "hello, how can I help?"),
            (EventSource.CUSTOMER, "buying a house for the first time, what do I need to know?"),
        ],
    )
    context.state = ResponseState()

    marker = "TURN_INSTRUCTIONS_MARKER_12345"

    async def turn_instructions(_: EngineContext) -> str:
        return marker

    job = LoopJob(
        context=context,
        system_instructions="SYSTEM_INSTRUCTIONS",
        step_instructions=turn_instructions,
    )

    history, instructions_index = await _make_streaming_loop()._build_history(job)

    # The model's most recent turn must be the customer's message — not the
    # imperative instructions note, which it otherwise tends to answer / echo.
    assert history[-1].role == Role.USER
    assert "buying a house" in history[-1].text

    # The turn instructions appear exactly once, immediately before that last
    # customer message, and _build_history reports their index (so the loop can
    # replace them in place when reevaluating).
    instruction_indices = [i for i, m in enumerate(history) if marker in m.text]
    assert instruction_indices == [len(history) - 2]
    assert instructions_index == len(history) - 2


async def test_that_reviewer_adjusted_reasoning_is_injected_even_without_step_instructions() -> (
    None
):
    # When step_instructions is None (e.g. low-effort agents) there is no turn-
    # instructions message in history. The reviewer's adjusted reasoning (step_notes)
    # must STILL reach the retried prompt — otherwise the retry re-streams the same
    # output and the review loop never converges.
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(step_notes="Do not call charge_card; ask for confirmation first.")

    loop = _make_streaming_loop()
    job = LoopJob(context=context, system_instructions="SYSTEM", step_instructions=None)

    history, instructions_index = await loop._build_history(job)
    assert instructions_index is None  # no step-instructions message exists for this config

    state = _LoopState(history=history, instructions_index=instructions_index)
    await loop._update_step_instructions(job, state)

    # The adjusted reasoning must be present in the (to-be-re-streamed) history.
    assert any("ask for confirmation first" in m.text for m in state.history)


async def test_that_default_preamble_configuration_does_not_inject_tool_preamble_note() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "find flights")])
    context.state = ResponseState()

    loop = _make_streaming_loop()
    job = LoopJob(context=context, system_instructions="SYSTEM", step_instructions=None)

    history, instructions_index = await loop._build_history(job)
    state = _LoopState(history=history, instructions_index=instructions_index)

    await loop._update_step_instructions(job, state)
    instructions_text = "\n".join(message.text for message in state.history)
    assert "Tool communication before tool use" not in instructions_text


async def test_that_tool_preamble_note_is_injected_initially_and_after_fifteen_seconds() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "find flights")])
    context.state = ResponseState()

    loop = _make_streaming_loop()
    job = LoopJob(
        context=context,
        system_instructions="SYSTEM",
        step_instructions=None,
        preamble_config=PreambleConfiguration.encourage(),
    )

    history, instructions_index = await loop._build_history(job)
    state = _LoopState(
        history=history,
        instructions_index=instructions_index,
        preamble=_encouraged_preamble_state(),
    )

    await loop._update_step_instructions(job, state)
    instructions_text = "\n".join(message.text for message in state.history)
    assert "Tool communication before tool use" in instructions_text
    assert "exactly one short, natural sentence" in instructions_text

    loop._mark_user_visible_message_emitted(context, state)
    assert state.preamble.last_user_visible_message_at is not None
    state.preamble.last_user_visible_message_at -= DEFAULT_PREAMBLE_INTERVAL_SECONDS - 1
    await loop._update_step_instructions(job, state)
    instructions_text = "\n".join(message.text for message in state.history)
    assert "Tool communication before tool use" in instructions_text
    assert "Less than 15 seconds have passed since your last message" in instructions_text
    assert "Run the next tool silently" in instructions_text

    assert state.preamble.last_user_visible_message_at is not None
    state.preamble.last_user_visible_message_at -= 2
    await loop._update_step_instructions(job, state)

    instructions_text = "\n".join(message.text for message in state.history)
    assert "Tool communication before tool use" in instructions_text
    assert "More than 15 seconds have passed since your last message" in instructions_text


async def test_that_adjusted_reasoning_does_not_override_tool_preamble_interval() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "find flights")])
    context.state = ResponseState(step_notes="Ask for the missing airport before searching.")

    loop = _make_streaming_loop()
    job = LoopJob(
        context=context,
        system_instructions="SYSTEM",
        step_instructions=None,
        preamble_config=PreambleConfiguration.encourage(),
    )

    history, instructions_index = await loop._build_history(job)
    state = _LoopState(
        history=history,
        instructions_index=instructions_index,
        preamble=_encouraged_preamble_state(),
    )
    loop._mark_user_visible_message_emitted(context, state)

    await loop._update_step_instructions(job, state)

    instructions_text = "\n".join(message.text for message in state.history)
    assert "Ask for the missing airport before searching." in instructions_text
    assert "Less than 15 seconds have passed since your last message" in instructions_text
    assert "Run the next tool silently" in instructions_text


async def test_that_a_restarted_step_finalizes_and_resets_the_streamed_message() -> None:
    # When a step is restarted (reviewer rejection), the already-streamed preamble must
    # be finalized as its own message and the streaming state reset, so the retry begins
    # a fresh message instead of concatenating onto the rejected one.
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    loop = _make_streaming_loop()
    state = _LoopState(preamble=_encouraged_preamble_state())

    await loop._surface_message_event(context, state, TextDelta(text="Let me check that for you."))
    assert state.message.buffer is not None

    await loop._reset_message_after_restart(context, state)
    assert state.message.handle is None
    assert state.message.buffer is None
    assert state.message.chunks == []
    assert state.in_the_middle_of_running_tools is False

    context.state.step_notes = "Ask for confirmation instead of calling the tool."
    assert state.preamble.last_user_visible_message_at is not None
    state.preamble.last_user_visible_message_at -= DEFAULT_PREAMBLE_INTERVAL_SECONDS + 1

    # The retry's text starts a NEW message, not appended to the rejected preamble.
    await loop._surface_message_event(
        context, state, TextDelta(text="Actually, here's the answer.")
    )
    assert state.message.buffer is not None
    assert state.message.buffer.getvalue() == "Actually, here's the answer."

    # Two separate message events were emitted; the latest carries no preamble.
    emitter = cast(EventBuffer, context.session_event_emitter)
    message_texts = [
        cast(dict[str, Any], e.data)["message"]
        for e in emitter.events
        if e.kind == EventKind.MESSAGE
    ]
    assert "Let me check that for you." not in message_texts[-1]


async def test_that_text_after_a_tool_call_in_one_step_is_suppressed_after_the_preamble() -> None:
    # Within ONE step the model can emit text, call a tool, then emit more text
    # (TEXT, TOOL, TEXT). Once the pre-tool preamble is surfaced, further text in
    # that same tool-call step should be suppressed so the user doesn't receive a
    # chain of progress updates before tools actually run.
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    loop = _make_streaming_loop()
    state = _LoopState(preamble=_encouraged_preamble_state())

    events = [
        TextDelta(text="Let me search for direct flights. "),
        ToolCallStarted(id="call-1", name="search_flights"),
        TextDelta(text="There are no direct flights."),
    ]
    for event in events:
        await loop._surface_message_event(context, state, event)

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_texts = [
        cast(dict[str, Any], e.data)["message"]
        for e in emitter.events
        if e.kind == EventKind.MESSAGE
    ]

    assert message_texts == ["Let me search for direct flights."]


async def test_that_streamed_tool_preamble_is_trimmed_to_one_sentence() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    loop = _make_streaming_loop()
    state = _LoopState(preamble=_encouraged_preamble_state())

    events = [
        TextDelta(text="I am checking the reservation status. "),
        TextDelta(text="Let me check the next flight too."),
        ToolCallStarted(id="call-1", name="get_reservation_details"),
    ]
    for event in events:
        await loop._surface_message_event(context, state, event)

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]

    assert len(message_events) == 1
    assert cast(dict[str, Any], message_events[0].data)["message"] == (
        "I am checking the reservation status."
    )


async def test_that_default_preamble_configuration_passes_pre_tool_text_through_untrimmed() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    loop = _make_streaming_loop()
    state = _LoopState()

    events = [
        TextDelta(text="I am checking the reservation status. "),
        TextDelta(text="Let me check the next flight too."),
        ToolCallStarted(id="call-1", name="get_reservation_details"),
    ]
    for event in events:
        await loop._surface_message_event(context, state, event)

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]

    assert len(message_events) == 1
    assert cast(dict[str, Any], message_events[0].data)["message"] == (
        "I am checking the reservation status. Let me check the next flight too."
    )


async def test_that_discourage_preamble_configuration_suppresses_pre_tool_text() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    loop = _make_streaming_loop()
    state = _LoopState(preamble=_ToolPreambleState(PreambleConfiguration.discourage()))

    events = [
        TextDelta(text="I am checking the reservation status."),
        ToolCallStarted(id="call-1", name="get_reservation_details"),
    ]
    for event in events:
        await loop._surface_message_event(context, state, event)

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]

    assert message_events == []


async def test_that_subsequent_streamed_tool_preambles_are_suppressed_and_not_committed() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    loop = _make_streaming_loop()
    state = _LoopState(preamble=_encouraged_preamble_state())
    loop._mark_user_visible_message_emitted(context, state)

    tool_call = ToolCallPart(id="call-1", name="search_flights")
    result = StepResult(
        message=Message(
            role=Role.ASSISTANT,
            parts=[TextPart(text="I'll check another airport."), tool_call],
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(),
    )

    events = [
        TextDelta(text="I'll check another airport."),
        ToolCallStarted(id="call-1", name="search_flights"),
        StepCompleted(result=result),
    ]
    for event in events:
        await loop._surface_message_event(context, state, event)
        await loop._commit_react_event(state, event, cast(Any, loop._model_router).react)

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]

    assert message_events == []
    assert state.history[-1].text == ""
    assert state.history[-1].tool_calls == [tool_call]


async def test_that_streamed_tool_preambles_are_allowed_after_fifteen_seconds() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    loop = _make_streaming_loop()
    state = _LoopState(preamble=_encouraged_preamble_state())
    loop._mark_user_visible_message_emitted(context, state)
    assert state.preamble.last_user_visible_message_at is not None
    state.preamble.last_user_visible_message_at -= DEFAULT_PREAMBLE_INTERVAL_SECONDS + 1

    tool_call = ToolCallPart(id="call-1", name="search_flights")
    result = StepResult(
        message=Message(
            role=Role.ASSISTANT,
            parts=[TextPart(text="I'll check another airport."), tool_call],
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(),
    )

    events = [
        TextDelta(text="I'll check another airport."),
        ToolCallStarted(id="call-1", name="search_flights"),
        StepCompleted(result=result),
    ]
    for event in events:
        await loop._surface_message_event(context, state, event)
        await loop._commit_react_event(state, event, cast(Any, loop._model_router).react)

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]

    assert len(message_events) == 1
    assert cast(dict[str, Any], message_events[0].data)["message"] == (
        "I'll check another airport."
    )
    assert state.history[-1].text == "I'll check another airport."
    assert state.history[-1].tool_calls == [tool_call]


async def test_that_the_loop_prefers_the_model_configs_reasoning_over_the_default() -> None:
    # ModelConfig.reasoning (resolved per route/effort) overrides the engine's
    # built-in effort→reasoning default.
    loop = _make_streaming_loop()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    cast(Any, loop._model_router).react_reasoning = ReasoningConfig(
        effort="high", visibility="full"
    )

    resolved = await loop._resolve_reasoning(
        context, ReasoningConfig(effort="low", visibility="summary")
    )

    assert resolved == ReasoningConfig(effort="high", visibility="full")


async def test_that_the_loop_falls_back_to_the_default_reasoning_when_the_route_sets_none() -> None:
    loop = _make_streaming_loop()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    # Stub resolves no route reasoning (react_reasoning stays None).
    default = ReasoningConfig(effort="low", visibility="summary")
    resolved = await loop._resolve_reasoning(context, default)

    assert resolved == default


async def test_that_streaming_fires_on_message_updated_with_the_accumulated_text() -> None:
    # Each streamed delta fires on_message_updated with the full message text so far
    # (streaming only) — never a bare chunk.
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    loop = _make_streaming_loop()
    state = _LoopState()

    updates: list[str] = []

    async def on_updated(_: Any, payload: str, __: Any) -> CompassHookResult:
        updates.append(payload)
        return CompassHookResult.CALL_NEXT

    context.dependencies.hooks.on_message_updated.append(on_updated)

    for delta in [TextDelta(text="Hello "), TextDelta(text="world")]:
        await loop._surface_message_event(context, state, delta)

    assert updates == ["Hello ", "Hello world"]


async def test_that_a_pending_supersede_before_the_first_streamed_chunk_abandons_the_reply() -> (
    None
):
    # A newer customer message superseded this turn while the step shield
    # deferred its cancellation: nothing user-visible may go out.
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    loop = _make_streaming_loop()
    state = _LoopState()
    state.cancellation_pending = lambda: True

    with pytest.raises(asyncio.CancelledError):
        await loop._surface_message_event(context, state, TextDelta(text="Hello"))

    emitter = cast(EventBuffer, context.session_event_emitter)
    assert not any(e.kind == EventKind.MESSAGE for e in emitter.events)


async def test_that_a_supersede_arriving_mid_stream_lets_the_streamed_message_complete() -> None:
    # Once streaming has begun the message is already on the wire, so a
    # supersede arriving mid-stream must NOT abandon it: remaining deltas
    # flow and the message finalizes normally (terminated chunk list).
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    loop = _make_streaming_loop()
    state = _LoopState()

    await loop._surface_message_event(context, state, TextDelta(text="Hello "))
    assert state.message.handle is not None

    state.cancellation_pending = lambda: True

    await loop._surface_message_event(context, state, TextDelta(text="world"))

    await loop._surface_message_event(
        context,
        state,
        StepCompleted(
            result=StepResult(
                message=Message(role=Role.ASSISTANT, parts=[TextPart(text="Hello world")]),
                finish_reason=FinishReason.STOP,
                usage=Usage(),
            )
        ),
    )

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]
    assert len(message_events) == 1

    data = cast(dict[str, Any], message_events[0].data)
    assert data["message"] == "Hello world"
    assert data["chunks"][-1] is None


class _EventRecordingTracer(LocalTracer):
    def __init__(self) -> None:
        super().__init__()
        self.recorded_events: list[tuple[str, dict[str, Any]]] = []

    def add_event(self, name: str, attributes: Any = {}) -> None:
        self.recorded_events.append((name, dict(attributes)))


def _dashboard_output_ref(event: Any) -> str:
    metadata = cast(dict[str, Any], event.metadata)
    return cast(str, metadata["__dashboard__"]["output_ref"])


async def test_that_a_streamed_bubble_carries_an_output_ref_joined_to_the_loop_message_event() -> (
    None
):
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    tracer = _EventRecordingTracer()
    loop = _make_streaming_loop(tracer)
    state = _LoopState()

    await loop._surface_message_event(context, state, TextDelta(text="Hello "))
    await loop._surface_message_event(context, state, TextDelta(text="world"))
    await loop._surface_message_event(
        context,
        state,
        StepCompleted(
            result=StepResult(
                message=Message(role=Role.ASSISTANT, parts=[TextPart(text="Hello world")]),
                finish_reason=FinishReason.STOP,
                usage=Usage(),
            )
        ),
    )

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]
    assert len(message_events) == 1

    output_ref = _dashboard_output_ref(message_events[0])
    assert output_ref

    loop_messages = [a for n, a in tracer.recorded_events if n == "loop.message"]
    assert len(loop_messages) == 1
    assert loop_messages[0]["output_ref"] == output_ref
    assert loop_messages[0]["mode"] == "stream"


async def test_that_a_restarted_attempt_and_its_retry_carry_distinct_output_refs() -> None:
    # A reviewer-rejected attempt is finalized as its own bubble; the retry
    # starts a NEW bubble. Each persisted bubble gets its own output_ref and
    # its own authoritative loop.message event.
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    tracer = _EventRecordingTracer()
    loop = _make_streaming_loop(tracer)
    state = _LoopState(preamble=_encouraged_preamble_state())

    await loop._surface_message_event(context, state, TextDelta(text="Let me check."))
    await loop._reset_message_after_restart(context, state)

    assert state.preamble.last_user_visible_message_at is not None
    state.preamble.last_user_visible_message_at -= DEFAULT_PREAMBLE_INTERVAL_SECONDS + 1

    await loop._surface_message_event(context, state, TextDelta(text="Here's the answer."))
    await loop._surface_message_event(
        context,
        state,
        StepCompleted(
            result=StepResult(
                message=Message(role=Role.ASSISTANT, parts=[TextPart(text="Here's the answer.")]),
                finish_reason=FinishReason.STOP,
                usage=Usage(),
            )
        ),
    )

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]
    assert len(message_events) == 2

    first_ref = _dashboard_output_ref(message_events[0])
    second_ref = _dashboard_output_ref(message_events[1])
    assert first_ref != second_ref

    loop_message_refs = [a["output_ref"] for n, a in tracer.recorded_events if n == "loop.message"]
    assert loop_message_refs == [first_ref, second_ref]


async def test_that_an_abandoned_streamed_reply_produces_no_loop_message_event() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    tracer = _EventRecordingTracer()
    loop = _make_streaming_loop(tracer)
    state = _LoopState()
    state.cancellation_pending = lambda: True

    with pytest.raises(asyncio.CancelledError):
        await loop._surface_message_event(context, state, TextDelta(text="Hello"))

    assert not any(n == "loop.message" for n, _ in tracer.recorded_events)


async def test_that_on_messages_emitted_receives_the_final_text_of_a_streamed_message() -> None:
    # A streamed bubble is opened on its FIRST chunk and only reaches its final
    # text through handle updates, so recording it at emission time would capture
    # "Hello " instead of the whole message.
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    loop = _make_streaming_loop()
    state = _LoopState()

    await loop._surface_message_event(context, state, TextDelta(text="Hello "))
    await loop._surface_message_event(context, state, TextDelta(text="world"))
    await loop._surface_message_event(
        context,
        state,
        StepCompleted(
            result=StepResult(
                message=Message(role=Role.ASSISTANT, parts=[TextPart(text="Hello world")]),
                finish_reason=FinishReason.STOP,
                usage=Usage(),
            )
        ),
    )

    assert context.state.emitted_messages == ["Hello world"]


async def test_that_native_assistant_artifacts_stay_out_of_visible_message_content() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.state = ResponseState()
    loop = _make_streaming_loop()
    state = _LoopState()
    replay = {
        "provider": "gemini",
        "text": "Hello.",
        "native_parts": [{"signature": "opaque-native-artifact"}],
    }
    # Exercise the metadata carrier with an opaque blob. Provider validation
    # belongs to the separate real-Google-client protocol tests.
    setattr(state.message, "assistant_replay", replay)
    await loop._surface_message_event(context, state, TextDelta(text="Hello."))
    await loop._surface_message_event(
        context,
        state,
        StepCompleted(
            result=StepResult(
                message=Message(role=Role.ASSISTANT, parts=[TextPart(text="Hello.")]),
                finish_reason=FinishReason.STOP,
                usage=Usage(),
            )
        ),
    )
    messages = [
        e
        for e in cast(EventBuffer, context.session_event_emitter).events
        if e.kind == EventKind.MESSAGE
    ]
    assert len(messages) == 1
    assert (
        metadata_utils.internal_value(messages[0].metadata, ("assistant_replay",), None) == replay
    )
    assert cast(dict[str, Any], messages[0].data)["message"] == "Hello."
    assert "opaque-native-artifact" not in repr(messages[0].data)
    assert "assistant_replay" not in repr(metadata_utils.public_metadata(messages[0].metadata))


async def test_that_committed_assistant_replay_artifacts_survive_split_tool_preambles() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "look it up")])
    context.state = ResponseState()
    loop = _make_streaming_loop()
    state = _LoopState(preamble=_encouraged_preamble_state())
    replay = {
        "provider": "gemini",
        "text": "I'll check.",
        "native_parts": [{"signature": "preamble-native-artifact"}],
    }
    setattr(state.message, "assistant_replay", replay)
    call = ToolCallPart(id="lookup-1", name="lookup")
    await loop._surface_message_event(context, state, TextDelta(text="I'll check."))
    await loop._surface_message_event(context, state, ToolCallStarted(id=call.id, name=call.name))
    await loop._surface_message_event(
        context,
        state,
        StepCompleted(
            result=StepResult(
                message=Message(role=Role.ASSISTANT, parts=[TextPart(text="I'll check."), call]),
                finish_reason=FinishReason.TOOL_CALLS,
                usage=Usage(),
            )
        ),
    )
    # A different following step must not inherit the prior bubble's blob.
    await loop._surface_message_event(context, state, TextDelta(text="It is ready."))
    await loop._surface_message_event(
        context,
        state,
        StepCompleted(
            result=StepResult(
                message=Message(role=Role.ASSISTANT, parts=[TextPart(text="It is ready.")]),
                finish_reason=FinishReason.STOP,
                usage=Usage(),
            )
        ),
    )
    messages = [
        e
        for e in cast(EventBuffer, context.session_event_emitter).events
        if e.kind == EventKind.MESSAGE
    ]
    assert [cast(dict[str, Any], e.data)["message"] for e in messages] == [
        "I'll check.",
        "It is ready.",
    ]
    assert (
        metadata_utils.internal_value(messages[0].metadata, ("assistant_replay",), None) == replay
    )
    assert metadata_utils.internal_value(messages[1].metadata, ("assistant_replay",), None) is None
