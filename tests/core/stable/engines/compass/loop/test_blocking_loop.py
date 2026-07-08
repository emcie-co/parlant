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
from dataclasses import replace
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest

from parlant.core.agents import Effort
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.compass.tracing import format_json_attr
from parlant.core.engines.compass.loop.base_loop import (
    _InteractionHistoryBuilder,
    _LoopState,
    _PROVIDER_DATA_KEY,
    _ToolPreambleState,
    _ToolStepController,
)
from parlant.core.nlp.common import ModelSize
from parlant.core.engines.compass.loop.blocking_loop import BlockingLoop
from parlant.core.engines.compass.loop.loop import LoopJob
from parlant.core.engines.compass.preambles import (
    DEFAULT_PREAMBLE_INTERVAL_SECONDS,
    PreambleConfiguration,
)
from parlant.core.engines.compass.response_state import ResponseState
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core.nlp.react import (
    FinishReason,
    Message,
    ReasoningDelta,
    ReasoningPart,
    Role,
    StepCompleted,
    StepResult,
    TextDelta,
    TextPart,
    ToolCallPart,
    ToolCallStarted,
    ToolResultPart,
    Usage,
)
from parlant.core.sessions import EventKind, EventSource, ToolEventData
from parlant.core.tools import Tool, ToolId, ToolOverlap, ToolResult
from parlant.core.nlp.tokenization import ZeroEstimatingTokenizer
from parlant.core.tracer import LocalTracer
from parlant.core.cost_control import AdvisoryCostControlPolicy
from parlant.core.usage_reporter import UsageReporter

from tests.core.stable.engines.compass.matching.utils import (
    RecordingTracer,
    create_agent,
    create_engine_context,
)


class _ModelResolvingStubReact:
    """Minimal react stub: building history resolves the replay model from the
    job's model_size, so even tests that don't drive a full turn need this hook.
    Build-driving fakes inherit it to get resolve_model for free."""

    def resolve_model(self, hints: Any) -> str:
        return "fake-model"


def _make_blocking_loop(
    logger: Logger | None = None,
    tracer: LocalTracer | None = None,
) -> BlockingLoop:
    # _surface_message_event only touches the session event emitter and the loop state,
    # so the heavier collaborators aren't exercised here.
    tracer = tracer or LocalTracer()
    logger = logger or StdoutLogger(tracer)

    return BlockingLoop(
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


class _RecordingLogger:
    def __init__(self) -> None:
        self.trace_messages: list[str] = []
        self.debug_messages: list[str] = []

    def trace(self, message: str) -> None:
        self.trace_messages.append(message)

    def debug(self, message: str) -> None:
        self.debug_messages.append(message)

    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass

    def critical(self, message: str) -> None:
        pass

    def set_level(self, level: str) -> None:
        pass

    @contextmanager
    def scope(self, scope_id: str) -> Any:
        yield


def _encouraged_preamble_state() -> _ToolPreambleState:
    return _ToolPreambleState(PreambleConfiguration.encourage())


def _tool(name: str, *, consequential: bool) -> Tool:
    return Tool(
        name=name,
        creation_utc=datetime.now(timezone.utc),
        description="",
        metadata={},
        parameters={},
        required=[],
        consequential=consequential,
        overlap=ToolOverlap.NONE,
    )


def _offer_tool(context: Any, tool: Tool) -> None:
    """Put a tool in the offered catalog so the loop can resolve it (and read its
    consequential flag for review gating)."""
    context.state.available_tools = [tool]
    context.state.tool_ids_by_name = {
        tool.name: ToolId(service_name="test_service", tool_name=tool.name)
    }


class _NoReplayReact:
    """Stands in for a generator that can't natively replay a stored tool turn
    (e.g. Gemini when the persisted thought_signature is unusable)."""

    def deserialize_tool_messages(self, deserializer: Any, *, model: str | None = None) -> None:
        return None


class _NoopToolMessageReact:
    def serialize_tool_messages(self, messages: list[Message], serializer: Any) -> None:
        return None


class _StubToolRunner:
    async def run_tool(self, context: Any, tool_id: ToolId, args: dict[str, Any]) -> ToolResult:
        return ToolResult(data={"ok": True})


class _CancellingToolRunner:
    def __init__(self) -> None:
        self.running = asyncio.Event()
        self.loop_task: asyncio.Task[Any] | None = None

    async def run_tool(self, context: Any, tool_id: ToolId, args: dict[str, Any]) -> ToolResult:
        self.running.set()
        assert self.loop_task is not None
        self.loop_task.cancel()
        return ToolResult(data={"ok": True})


class _EmptyThenMessageReact(_ModelResolvingStubReact):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def stream_step(
        self,
        *,
        history: list[Message],
        tools: list[Any],
        tool_choice: str,
        reasoning: Any,
        hints: dict[str, Any],
    ) -> Any:
        self.calls.append(
            {
                "history": list(history),
                "tools": list(tools),
                "tool_choice": tool_choice,
            }
        )

        if len(self.calls) == 1:
            yield StepCompleted(
                result=StepResult(
                    message=Message(role=Role.ASSISTANT),
                    finish_reason=FinishReason.STOP,
                    usage=Usage(),
                )
            )
        else:
            yield StepCompleted(
                result=StepResult(
                    message=Message(
                        role=Role.ASSISTANT,
                        parts=[
                            TextPart(text="I'm sorry, I'm not able to help with that right now.")
                        ],
                    ),
                    finish_reason=FinishReason.STOP,
                    usage=Usage(),
                )
            )


class _MessageReact(_ModelResolvingStubReact):
    async def stream_step(
        self,
        *,
        history: list[Message],
        tools: list[Any],
        tool_choice: str,
        reasoning: Any,
        hints: dict[str, Any],
    ) -> Any:
        yield TextDelta(text="Hello there!")
        yield StepCompleted(
            result=StepResult(
                message=Message(
                    role=Role.ASSISTANT,
                    parts=[TextPart(text="Hello there!")],
                ),
                finish_reason=FinishReason.STOP,
                usage=Usage(),
            )
        )


class _RejectingReviewer:
    async def review_tool_calls(
        self,
        context: Any,
        reasoning: str,
        tool_calls: list[ToolCallPart],
    ) -> Any:
        return type(
            "ToolCallReviewResult",
            (),
            {
                "todo": "",
                "adjusted_reasoning": "Ask the user for the missing confirmation instead.",
            },
        )()


class _TodoReviewer:
    async def review_tool_calls(
        self,
        context: Any,
        reasoning: str,
        tool_calls: list[ToolCallPart],
    ) -> Any:
        return type(
            "ToolCallReviewResult",
            (),
            {
                "todo": "Verify the tool result before sending a final answer.",
                "adjusted_reasoning": None,
            },
        )()


class _SpyReviewer:
    def __init__(self) -> None:
        self.called = False

    async def review_tool_calls(
        self,
        context: Any,
        reasoning: str,
        tool_calls: list[ToolCallPart],
    ) -> Any:
        self.called = True
        return type("ToolCallReviewResult", (), {"todo": "", "adjusted_reasoning": None})()


def _controller(reviewer: _SpyReviewer) -> _ToolStepController:
    return _ToolStepController(
        logger=StdoutLogger(LocalTracer()),
        react=cast(Any, None),
        tool_runner=cast(Any, None),
        reviewer=cast(Any, lambda: reviewer),
    )


async def _review(reviewer: _SpyReviewer, *, effort: Effort, consequential: bool) -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "please charge it")])
    context.state = ResponseState(agent_effort=effort)
    _offer_tool(context, _tool("charge_card", consequential=consequential))
    await _controller(reviewer).review_tool_calls(
        context, "reasoning", [ToolCallPart(id="c1", name="charge_card", args={})]
    )


async def test_that_a_consequential_tool_call_triggers_review_below_max_effort() -> None:
    reviewer = _SpyReviewer()
    await _review(reviewer, effort=Effort.HIGH, consequential=True)
    assert reviewer.called


async def test_that_a_non_consequential_tool_call_skips_review_below_max_effort() -> None:
    reviewer = _SpyReviewer()
    await _review(reviewer, effort=Effort.HIGH, consequential=False)
    assert not reviewer.called


async def test_that_max_effort_reviews_even_non_consequential_tool_calls() -> None:
    reviewer = _SpyReviewer()
    await _review(reviewer, effort=Effort.MAX, consequential=False)
    assert reviewer.called


class _RejectedToolsThenMessageReact(_ModelResolvingStubReact):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def stream_step(
        self,
        *,
        history: list[Message],
        tools: list[Any],
        tool_choice: str,
        reasoning: Any,
        hints: dict[str, Any],
    ) -> Any:
        self.calls.append(
            {
                "history": list(history),
                "tools": list(tools),
                "tool_choice": tool_choice,
            }
        )

        if tool_choice == "none":
            yield StepCompleted(
                result=StepResult(
                    message=Message(
                        role=Role.ASSISTANT,
                        parts=[
                            TextPart(text="I'm sorry, I'm not able to help with that right now.")
                        ],
                    ),
                    finish_reason=FinishReason.STOP,
                    usage=Usage(),
                )
            )
            return

        tool_call = ToolCallPart(id="call-1", name="charge_card", args={})
        yield ToolCallStarted(id=tool_call.id, name=tool_call.name)
        yield StepCompleted(
            result=StepResult(
                message=Message(role=Role.ASSISTANT, parts=[tool_call]),
                finish_reason=FinishReason.TOOL_CALLS,
                usage=Usage(),
            )
        )


class _TodoToolsThenMessageReact(_ModelResolvingStubReact):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def serialize_tool_messages(self, messages: list[Message], serializer: Any) -> None:
        return None

    async def stream_step(
        self,
        *,
        history: list[Message],
        tools: list[Any],
        tool_choice: str,
        reasoning: Any,
        hints: dict[str, Any],
    ) -> Any:
        self.calls.append(
            {
                "history": list(history),
                "tools": list(tools),
                "tool_choice": tool_choice,
            }
        )

        if len(self.calls) == 1:
            tool_call = ToolCallPart(id="call-1", name="lookup_account", args={})
            yield ToolCallStarted(id=tool_call.id, name=tool_call.name)
            yield StepCompleted(
                result=StepResult(
                    message=Message(role=Role.ASSISTANT, parts=[tool_call]),
                    finish_reason=FinishReason.TOOL_CALLS,
                    usage=Usage(),
                )
            )
            return

        yield StepCompleted(
            result=StepResult(
                message=Message(
                    role=Role.ASSISTANT,
                    parts=[TextPart(text="I checked that and can help.")],
                ),
                finish_reason=FinishReason.STOP,
                usage=Usage(),
            )
        )


class _ToolThenMessageReact(_ModelResolvingStubReact):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def serialize_tool_messages(self, messages: list[Message], serializer: Any) -> None:
        return None

    async def stream_step(
        self,
        *,
        history: list[Message],
        tools: list[Any],
        tool_choice: str,
        reasoning: Any,
        hints: dict[str, Any],
    ) -> Any:
        self.calls.append(
            {
                "history": list(history),
                "tools": list(tools),
                "tool_choice": tool_choice,
            }
        )

        if len(self.calls) == 1:
            tool_call = ToolCallPart(id="call-1", name="lookup_account", args={})
            yield ToolCallStarted(id=tool_call.id, name=tool_call.name)
            yield StepCompleted(
                result=StepResult(
                    message=Message(role=Role.ASSISTANT, parts=[tool_call]),
                    finish_reason=FinishReason.TOOL_CALLS,
                    usage=Usage(),
                )
            )
            return

        yield StepCompleted(
            result=StepResult(
                message=Message(
                    role=Role.ASSISTANT,
                    parts=[TextPart(text="I checked that and can help.")],
                ),
                finish_reason=FinishReason.STOP,
                usage=Usage(),
            )
        )


def test_that_an_unreplayable_tool_event_is_rendered_as_a_result_not_dropped() -> None:
    # A prior-turn tool event whose provider blob can't be natively replayed must
    # NOT be discarded — that would make the model forget the tool's data across
    # turns. Fall back to a result-only rendering so the data survives.
    loop = _make_blocking_loop()
    loop._react = cast(Any, _NoReplayReact())

    data: ToolEventData = {
        "tool_calls": [
            {
                "tool_id": "get_order_details",
                "rationale": "",
                "arguments": {"order_id": "#W2378156"},
                "result": cast(Any, {"data": {"status": "delivered"}, "metadata": {}}),
            }
        ]
    }
    metadata = {_PROVIDER_DATA_KEY: {"provider": "gemini", "model": "stale-model"}}

    messages = loop._build_tool_event_messages(data, metadata, "sess.compass")

    assert len(messages) == 1
    assert messages[0].role == Role.TOOL
    result_part = cast(ToolResultPart, messages[0].parts[0])
    # Rendered as a result (not dropped); the result data is carried through, whatever
    # the exact rendering (raw value or a descriptive string).
    assert "delivered" in str(result_part.content)


class _RecordingReplayReact:
    """Records the replay model forwarded into ``deserialize_tool_messages`` and
    exposes a ``model_size`` → model resolution like the real generators."""

    def __init__(self) -> None:
        self.deserialized_with_model: str | None = "UNSET"

    def resolve_model(self, hints: dict[str, Any]) -> str:
        return f"resolved:{hints.get('model_size')}"

    def deserialize_tool_messages(
        self, deserializer: Any, *, model: str | None = None
    ) -> list[Message]:
        self.deserialized_with_model = model
        return [Message(role=Role.TOOL, parts=[ToolResultPart(call_id="1", name="t", content="x")])]


async def test_that_history_building_resolves_the_replay_model_from_the_job_model_size() -> None:
    # A persisted tool event is replayed against the model THIS turn will use,
    # resolved from the job's model_size — not the generator's static default.
    react = _RecordingReplayReact()
    builder = _InteractionHistoryBuilder(StdoutLogger(LocalTracer()), lambda: cast(Any, react))

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    context.state.tool_events = [
        cast(
            Any,
            SimpleNamespace(
                kind=EventKind.TOOL,
                data={
                    "tool_calls": [
                        {
                            "tool_id": "get_order_details",
                            "rationale": "",
                            "arguments": {"order_id": "#W2378156"},
                            "result": cast(Any, {"data": {"status": "delivered"}, "metadata": {}}),
                        }
                    ]
                },
                metadata={_PROVIDER_DATA_KEY: {"provider": "gemini", "model": "stale-model"}},
            ),
        )
    ]

    job = LoopJob(
        context=context,
        system_instructions="SYSTEM",
        model_size=ModelSize.SMALL,
    )

    await builder.build(job)

    # The model forwarded to replay is the one resolved from THIS turn's model_size.
    assert react.deserialized_with_model == react.resolve_model({"model_size": ModelSize.SMALL})
    assert react.deserialized_with_model != react.resolve_model({"model_size": ModelSize.MEDIUM})


async def test_that_tool_call_step_is_committed_before_tool_results_are_appended() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(tool_ids_by_name={"test_tool": ToolId("local", "test_tool")})

    loop = _make_blocking_loop()
    loop._react = cast(Any, _NoopToolMessageReact())
    loop._tool_runner = cast(Any, _StubToolRunner())
    state = _LoopState()

    tool_call = ToolCallPart(
        id="call-1",
        name="test_tool",
        args={"value": 1},
    )
    result = StepResult(
        message=Message(role=Role.ASSISTANT, parts=[tool_call]),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(),
    )

    job = LoopJob(context=context, system_instructions="SYSTEM")

    committed = await loop._process_tool_event(job, state, StepCompleted(result=result))

    assert committed is True
    assert [m.role for m in state.history] == [Role.ASSISTANT, Role.TOOL]
    assert state.history[0].tool_calls == [tool_call]
    assert state.history[1].tool_results[0].content == {"ok": True}
    assert state.steps == [result]


async def test_that_tool_call_step_records_tool_events_under_current_spans() -> None:
    tracer = RecordingTracer()
    context = replace(
        create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")]),
        tracer=tracer,
    )
    context.state = ResponseState(tool_ids_by_name={"test_tool": ToolId("local", "test_tool")})

    loop = _make_blocking_loop(tracer=tracer)
    loop._react = cast(Any, _NoopToolMessageReact())
    loop._tool_runner = cast(Any, _StubToolRunner())
    state = _LoopState()

    tool_call = ToolCallPart(
        id="call-1",
        name="test_tool",
        args={"value": 1},
    )
    result = StepResult(
        message=Message(role=Role.ASSISTANT, parts=[tool_call]),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(),
    )

    job = LoopJob(context=context, system_instructions="SYSTEM")

    with tracer.span("loop.step"):
        committed = await loop._process_tool_event(job, state, StepCompleted(result=result))

    assert committed is True
    assert [span.name for span in tracer.started_spans] == [
        "loop.step",
        "tools.batch",
    ]
    assert [event.name for event in tracer.events] == [
        "loop.tools.started",
        "tool.requested",
        "loop.tool.persistent",
        "loop.tools.finished",
    ]
    assert tracer.events[1].attributes == {
        "tool_call_id": "call-1",
        "tool_name": "test_tool",
        "arguments": format_json_attr({"value": 1}),
    }
    assert tracer.events[0].span_id == "loop.step"
    assert tracer.events[2].span_id == "tools.batch"
    assert tracer.events[-1].span_id == "loop.step"


async def test_that_step_completed_reasoning_records_reasoning_event_under_current_span() -> None:
    tracer = RecordingTracer()
    context = replace(
        create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")]),
        tracer=tracer,
    )
    context.state = ResponseState()
    loop = _make_blocking_loop(tracer=tracer)
    state = _LoopState()
    result = StepResult(
        message=Message(
            role=Role.ASSISTANT,
            parts=[ReasoningPart(text="The customer needs a concise answer.")],
        ),
        finish_reason=FinishReason.STOP,
        usage=Usage(),
    )

    with tracer.span("loop.step"):
        await loop._process_reasoning_event(context, state, StepCompleted(result=result))

    assert [span.name for span in tracer.started_spans] == ["loop.step"]
    assert [event.name for event in tracer.events] == [
        "loop.reasoning.started",
        "loop.reasoning",
        "loop.reasoning.finished",
    ]
    assert tracer.events[1].attributes == {
        "reasoning": "The customer needs a concise answer.",
        "chunk_count": 0,
    }
    assert tracer.events[0].span_id == "loop.step"


async def test_that_streamed_reasoning_completion_uses_step_completed_reasoning_value() -> None:
    tracer = RecordingTracer()
    context = replace(
        create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")]),
        tracer=tracer,
    )
    context.state = ResponseState()
    loop = _make_blocking_loop(tracer=tracer)
    state = _LoopState()
    result = StepResult(
        message=Message(role=Role.ASSISTANT, parts=[]),
        finish_reason=FinishReason.STOP,
        usage=Usage(),
    )

    with tracer.span("loop.step"):
        await loop._process_reasoning_event(
            context,
            state,
            ReasoningDelta(text="draft reasoning"),
        )
        await loop._process_reasoning_event(context, state, StepCompleted(result=result))

    assert [event.name for event in tracer.events] == [
        "loop.reasoning.started",
        "loop.reasoning",
        "loop.reasoning.finished",
    ]
    assert tracer.events[1].attributes == {
        "reasoning": "",
        "chunk_count": 1,
    }


def test_that_system_instructions_are_logged_only_when_inserted_or_changed() -> None:
    logger = _RecordingLogger()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    loop = _make_blocking_loop(cast(Logger, logger))
    state = _LoopState()
    job = LoopJob(context=context, system_instructions="SYSTEM A")

    loop._update_system_instructions(job, state)
    assert state.history[0].text == "SYSTEM A"
    assert logger.trace_messages == ["BlockingLoop inserted system instructions:\nSYSTEM A"]

    loop._update_system_instructions(job, state)
    assert logger.trace_messages == ["BlockingLoop inserted system instructions:\nSYSTEM A"]

    updated_job = replace(job, system_instructions="SYSTEM B")
    loop._update_system_instructions(updated_job, state)

    assert state.history[0].text == "SYSTEM B"
    assert logger.trace_messages == [
        "BlockingLoop inserted system instructions:\nSYSTEM A",
        "BlockingLoop updated system instructions:\nSYSTEM B",
    ]


def test_that_semantic_history_log_omits_provider_data() -> None:
    loop = _make_blocking_loop()
    history = [
        Message(
            role=Role.USER,
            parts=[TextPart(text="Book a flight", provider_data={"opaque": "user-blob"})],
            provider_data={"opaque": "message-blob"},
        ),
        Message(
            role=Role.ASSISTANT,
            parts=[
                TextPart(text="I will search."),
                ToolCallPart(
                    id="call-1",
                    name="search_flights",
                    args={"origin": "JFK", "destination": "SFO"},
                    provider_data={"thought_signature": "huge-provider-blob"},
                ),
            ],
        ),
        Message(
            role=Role.TOOL,
            parts=[
                ToolResultPart(
                    call_id="call-1",
                    name="search_flights",
                    content={"flights": ["AA100"]},
                    provider_data={"provider": "native-tool-replay-blob"},
                )
            ],
        ),
    ]

    rendered = loop._render_semantic_history(history)

    assert "Book a flight" in rendered
    assert "Tool call: search_flights({" in rendered
    assert '"origin": "JFK"' in rendered
    assert "Tool result: search_flights[call_id=call-1]" in rendered
    assert '"AA100"' in rendered

    assert "provider_data" not in rendered
    assert "huge-provider-blob" not in rendered
    assert "native-tool-replay-blob" not in rendered
    assert "message-blob" not in rendered


async def test_that_reviewer_todo_refreshes_turn_instructions_before_next_step() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "please check")])
    context.state = ResponseState(agent_effort=Effort.HIGH)
    _offer_tool(context, _tool("lookup_account", consequential=True))

    async def step_instructions(_: Any) -> str:
        return "Base turn instructions."

    react = _TodoToolsThenMessageReact()
    loop = _make_blocking_loop()
    loop._react = cast(Any, react)
    loop._tool_runner = cast(Any, _StubToolRunner())
    loop._reviewer = cast(Any, _TodoReviewer())

    await loop.run(
        LoopJob(
            context=context,
            system_instructions="SYSTEM",
            step_instructions=step_instructions,
        )
    )

    assert len(react.calls) == 2
    next_step_history_text = "\n".join(message.text for message in react.calls[1]["history"])
    assert "TODO LIST: Remaining tasks before responding to the user" in next_step_history_text
    assert "Verify the tool result before sending a final answer." in next_step_history_text


async def test_that_empty_reviewer_todo_clears_stale_todo_in_turn_instructions() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "please check")])
    context.state = ResponseState(agent_effort=Effort.HIGH)
    context.state.todo = "Stale task from a previous tool review."
    _offer_tool(context, _tool("lookup_account", consequential=True))

    async def step_instructions(_: Any) -> str:
        return "Base turn instructions."

    react = _TodoToolsThenMessageReact()
    loop = _make_blocking_loop()
    loop._react = cast(Any, react)
    loop._tool_runner = cast(Any, _StubToolRunner())
    loop._reviewer = cast(Any, _SpyReviewer())

    await loop.run(
        LoopJob(
            context=context,
            system_instructions="SYSTEM",
            step_instructions=step_instructions,
        )
    )

    assert context.state.todo == ""
    assert len(react.calls) == 2
    next_step_history_text = "\n".join(message.text for message in react.calls[1]["history"])
    assert "Stale task from a previous tool review." not in next_step_history_text
    assert "TODO LIST: Remaining tasks before responding to the user" not in next_step_history_text


async def test_that_cancellation_during_a_tool_step_is_raised_after_tool_result_is_emitted() -> (
    None
):
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "please check")])
    context.state = ResponseState(agent_effort=Effort.MIN)
    _offer_tool(context, _tool("lookup_account", consequential=False))

    react = _ToolThenMessageReact()
    tool_runner = _CancellingToolRunner()
    loop = _make_blocking_loop()
    loop._react = cast(Any, react)
    loop._tool_runner = cast(Any, tool_runner)

    run_task = asyncio.create_task(loop.run(LoopJob(context=context, system_instructions="SYSTEM")))
    tool_runner.loop_task = run_task

    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert tool_runner.running.is_set()
    assert len(react.calls) == 1

    emitter = cast(EventBuffer, context.session_event_emitter)
    tool_events = [e for e in emitter.events if e.kind == EventKind.TOOL]
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]
    status_events = [e for e in emitter.events if e.kind == EventKind.STATUS]

    assert len(tool_events) == 1
    tool_data = cast(ToolEventData, tool_events[0].data)
    assert tool_data["tool_calls"][0]["tool_id"] == "lookup_account"
    assert tool_data["tool_calls"][0]["result"]["data"] == {"ok": True}

    assert not message_events
    statuses = [cast(dict[str, Any], e.data)["status"] for e in status_events]
    assert "cancelled" not in statuses
    assert "ready" not in statuses


async def test_that_blocking_loop_emits_a_single_complete_message_event_without_chunks() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    loop = _make_blocking_loop()
    state = _LoopState()

    result = StepResult(
        message=Message(role=Role.ASSISTANT, parts=[TextPart(text="Hello there!")]),
        finish_reason=FinishReason.STOP,
        usage=Usage(),
    )

    # Block mode still receives the provider's text deltas; it just doesn't emit
    # them incrementally — the whole message is emitted once on step completion.
    events = [TextDelta(text="Hello "), TextDelta(text="there!"), StepCompleted(result=result)]
    for event in events:
        await loop._surface_message_event(context, state, event)
        await loop._commit_react_event(state, event)

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]

    # Exactly one message event, carrying the full text and NO `chunks` key (so
    # consumers render it as a complete, non-streamed message).
    assert len(message_events) == 1
    data = cast(dict[str, Any], message_events[0].data)
    assert data["message"] == "Hello there!"
    assert "chunks" not in data

    # A completed message with no tool calls ends the loop.
    assert context.state.prepared_to_respond is True


async def test_that_blocking_loop_records_run_step_message_spans_and_events() -> None:
    tracer = RecordingTracer()
    context = replace(
        create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")]),
        tracer=tracer,
    )
    context.state = ResponseState()

    loop = _make_blocking_loop(tracer=tracer)
    loop._react = cast(Any, _MessageReact())

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    assert [span.name for span in tracer.started_spans] == [
        "loop.run",
        "loop.step",
    ]
    assert [event.name for event in tracer.events] == [
        "loop.message.started",
        "loop.message",
        "loop.message.finished",
    ]
    assert tracer.events[1].attributes == {
        "mode": "blocking",
        "message": "Hello there!",
    }
    assert tracer.events[1].span_id == "loop.step"


async def test_that_max_engine_iterations_forces_a_final_message_with_tools_disabled() -> None:
    agent = replace(create_agent(), max_engine_iterations=1)
    context = create_engine_context(
        conversation=[(EventSource.CUSTOMER, "please help")],
        agent=agent,
    )
    context.state = ResponseState()

    react = _EmptyThenMessageReact()
    loop = _make_blocking_loop()
    loop._react = cast(Any, react)

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    assert len(react.calls) == 2
    assert react.calls[0]["tool_choice"] == "auto"
    assert react.calls[1]["tool_choice"] == "none"
    assert react.calls[1]["tools"] == []
    assert any(
        "You must now explain to the user" in message.text for message in react.calls[1]["history"]
    )

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]
    assert cast(dict[str, Any], message_events[-1].data)["message"] == (
        "I'm sorry, I'm not able to help with that right now."
    )


async def test_that_max_semantic_failures_force_a_final_message_with_tools_disabled() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "please charge it")])
    context.state = ResponseState(agent_effort=Effort.HIGH)
    # The reviewer only runs (and can reject) when a consequential tool is called.
    _offer_tool(context, _tool("charge_card", consequential=True))

    react = _RejectedToolsThenMessageReact()
    loop = _make_blocking_loop()
    loop._react = cast(Any, react)
    loop._reviewer = cast(Any, _RejectingReviewer())

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    assert len(react.calls) == 6
    assert [call["tool_choice"] for call in react.calls[:-1]] == ["auto"] * 5
    assert react.calls[-1]["tool_choice"] == "none"
    assert react.calls[-1]["tools"] == []
    assert any(
        "Tool use is disabled for this step" in message.text
        for message in react.calls[-1]["history"]
    )

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]
    assert cast(dict[str, Any], message_events[-1].data)["message"] == (
        "I'm sorry, I'm not able to help with that right now."
    )


async def test_that_text_after_a_tool_call_in_one_step_is_suppressed_after_the_preamble() -> None:
    # TEXT, TOOL, TEXT within ONE step: once the pre-tool preamble is surfaced,
    # further text in that same tool-call step should be suppressed so the user
    # doesn't receive a chain of progress updates before tools actually run.
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    loop = _make_blocking_loop()
    state = _LoopState(preamble=_encouraged_preamble_state())

    # As TurnBuilder assembles it: both text segments fold into ONE TextPart, with
    # the tool call sitting after them in part order.
    result = StepResult(
        message=Message(
            role=Role.ASSISTANT,
            parts=[
                TextPart(text="Let me search for direct flights. There are no direct flights."),
                ToolCallPart(id="call-1", name="search_flights"),
            ],
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(),
    )

    events = [
        TextDelta(text="Let me search for direct flights. "),
        ToolCallStarted(id="call-1", name="search_flights"),
        TextDelta(text="There are no direct flights."),
        StepCompleted(result=result),
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


async def test_that_blocking_tool_preamble_is_trimmed_to_one_sentence() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    loop = _make_blocking_loop()
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


async def test_that_subsequent_blocking_tool_preambles_are_suppressed_and_not_committed() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    loop = _make_blocking_loop()
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


async def test_that_blocking_tool_preambles_are_allowed_after_fifteen_seconds() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    loop = _make_blocking_loop()
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
