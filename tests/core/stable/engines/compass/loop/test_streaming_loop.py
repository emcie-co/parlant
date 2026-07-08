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

from typing import Any, cast

from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.compass.loop.base_loop import _LoopState, _ToolPreambleState
from parlant.core.engines.compass.loop.loop import LoopJob
from parlant.core.engines.compass.loop.streaming_loop import StreamingLoop
from parlant.core.engines.compass.preambles import (
    DEFAULT_PREAMBLE_INTERVAL_SECONDS,
    PreambleConfiguration,
)
from parlant.core.engines.compass.response_state import EngineContext, ResponseState
from parlant.core.loggers import StdoutLogger
from parlant.core.nlp.react import (
    FinishReason,
    Message,
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
from parlant.core.cost_control import AdvisoryCostControlPolicy
from parlant.core.usage_reporter import UsageReporter

from tests.core.stable.engines.compass.matching.utils import create_engine_context


class _ModelResolvingStubReact:
    """Minimal react stub for history-only tests: building history resolves the
    replay model from the job's model_size, so even these tests need that hook."""

    def resolve_model(self, hints: Any) -> str:
        return "fake-model"


def _make_streaming_loop() -> StreamingLoop:
    # _build_history only reads the LoopJob, so the heavier collaborators
    # (meter/optimization_policy/react/tool_runner) aren't exercised here — except
    # the react's resolve_model, which history building uses to pick the replay model.
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)

    return StreamingLoop(
        logger=logger,
        tracer=tracer,
        meter=cast(Any, None),
        optimization_policy=cast(Any, None),
        react=cast(Any, _ModelResolvingStubReact()),
        tokenizer=ZeroEstimatingTokenizer(),
        tool_runner=cast(Any, None),
        reviewer=cast(Any, None),
        hooks=EngineHooks(),
        usage_reporter=UsageReporter(tracer),
        cost_control_policy=AdvisoryCostControlPolicy(UsageReporter(tracer)),
    )


def _encouraged_preamble_state() -> _ToolPreambleState:
    return _ToolPreambleState(PreambleConfiguration.encourage())


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

    loop._mark_user_visible_message_emitted(state)
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
    loop._mark_user_visible_message_emitted(state)

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
    loop._mark_user_visible_message_emitted(state)

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
        await loop._commit_react_event(state, event)

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
    loop._mark_user_visible_message_emitted(state)
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
        await loop._commit_react_event(state, event)

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]

    assert len(message_events) == 1
    assert cast(dict[str, Any], message_events[0].data)["message"] == (
        "I'll check another airport."
    )
    assert state.history[-1].text == "I'll check another airport."
    assert state.history[-1].tool_calls == [tool_call]
