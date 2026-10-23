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
import time
from collections.abc import Mapping
from dataclasses import replace
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest

from parlant.core.agents import Effort
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.compass.hooks import (
    CompassHookResult,
    CompassHooks,
    GaveUp,
    GiveUpKind,
    MessageReviewed,
    ToolCallOutcome,
    ToolCallsReviewed,
)
from parlant.core.async_utils import safe_gather
from parlant.core.engines.compass.tracing import CompassTracer
from parlant.core.engines.compass.loop.base_loop import (
    _GiveUp,
    _InteractionHistoryBuilder,
    _LoopState,
    _PROVIDER_DATA_KEY,
    _SemanticFailure,
    _ToolPreambleState,
    _ToolStepController,
    _customer_message_text,
)
from parlant.core.engines.compass.loop.blocking_loop import BlockingLoop
from parlant.core.engines.compass.loop.loop import LoopJob
from parlant.core.engines.compass.preambles import (
    DEFAULT_PREAMBLE_INTERVAL_SECONDS,
    PreambleConfiguration,
)
from parlant.core.engines.compass.context import CompassDependencies, ResponseState
from parlant.core.engines.compass.optimization_policy import CompassOptimizationPolicy
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core import metadata as metadata_utils
from parlant.core.nlp.react import (
    FinishReason,
    Message,
    ReasoningDelta,
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
from parlant.core.agents import Moderation
from parlant.core.nlp.moderation import (
    AgentModerationContext,
    CustomerModerationContext,
    ModerationCheck,
    ModerationService,
)
from parlant.core.tracer import LocalTracer
from parlant.core.cost_control import NullCostControlPolicy
from parlant.core.usage_reporter import UsageReporter

from tests.core.stable.engines.compass.matching.utils import (
    create_agent,
    create_engine_context,
)
from parlant.core.application_context import ApplicationContext
from parlant.core.health import NullHealthReporter


def _health_reporter() -> NullHealthReporter:
    return NullHealthReporter(ApplicationContext(instance_id="test-instance"))


class _ModelResolvingStubReact:
    """Minimal react stub for history replay model resolution."""

    provider_name = "fake"

    def resolve_model(self, hints: Any) -> str:
        return "fake-model"


@pytest.mark.parametrize(
    ("attachment", "expected_note"),
    [
        (
            {
                "url": "https://private.example/image.jpg?token=secret",
                "media_type": "image/jpeg",
                "caption": "The damaged package",
                "filename": "damage.jpg",
                "metadata": {"private": "value"},
            },
            "[image: The damaged package]",
        ),
        (
            {
                "url": "https://private.example/invoice.pdf?token=secret",
                "media_type": "application/pdf",
                "filename": "invoice.pdf",
            },
            "[file: invoice.pdf]",
        ),
        (
            {"url": "https://private.example/file?token=secret"},
            "[file attached]",
        ),
    ],
)
def test_that_customer_attachment_urls_never_enter_model_history(
    attachment: Any,
    expected_note: str,
) -> None:
    message = cast(
        Any,
        {
            "message": "Please inspect this.",
            "participant": {"display_name": "Customer"},
            "attachments": [attachment],
        },
    )

    rendered = _customer_message_text(message)

    assert rendered == f"Please inspect this.\n{expected_note}"
    assert attachment["url"] not in rendered


class _ModelRouterStub:
    def __init__(self, react: Any | None = None) -> None:
        self.react = react or _ModelResolvingStubReact()
        self.tokenizer = ZeroEstimatingTokenizer()
        self.react_requests: list[tuple[str, Mapping[str, str]]] = []
        self.moderation_service: ModerationService | None = None
        self.moderation_error: Exception | None = None
        self.moderation_requests: list[str] = []

    async def get_react_generator(
        self,
        *,
        route: str,
        qualifiers: Mapping[str, str],
        **kwargs: Any,
    ) -> Any:
        self.react_requests.append((route, qualifiers))
        return self.react

    async def get_estimating_tokenizer(self, **kwargs: Any) -> Any:
        return self.tokenizer

    async def resolve_react_reasoning(self, **kwargs: Any) -> None:
        return None

    async def get_moderation_service(self, *, route: str, **kwargs: Any) -> ModerationService:
        self.moderation_requests.append(route)
        if self.moderation_error is not None:
            raise self.moderation_error
        assert self.moderation_service is not None
        return self.moderation_service


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
        model_router=cast(Any, _ModelRouterStub()),
        tool_runner=cast(Any, None),
        reviewer=cast(Any, None),
        usage_reporter=UsageReporter(tracer),
        cost_control_policy=NullCostControlPolicy(),
        health_reporter=_health_reporter(),
    )


async def test_that_each_step_resolves_the_current_effort_route() -> None:
    loop = _make_blocking_loop()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.state = ResponseState(agent_effort=Effort.LOW)

    low_react = _ModelResolvingStubReact()
    high_react = _ModelResolvingStubReact()
    router = cast(_ModelRouterStub, loop._model_router)
    router.react = low_react

    assert cast(Any, await loop._resolve_react(context, step=0)) is low_react

    context.state = ResponseState(agent_effort=Effort.HIGH)
    router.react = high_react

    assert cast(Any, await loop._resolve_react(context, step=1)) is high_react
    assert router.react_requests == [
        ("react", {"effort": "low"}),
        ("react", {"effort": "high"}),
    ]


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

    provider_name = "fake"

    def deserialize_tool_messages(self, deserializer: Any, *, model: str | None = None) -> None:
        return None


class _NoopToolMessageReact:
    provider_name = "fake"

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
                "hints": dict(hints),
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
        tool_runner=cast(Any, None),
        reviewer=cast(Any, lambda: reviewer),
        before_tool_calls=cast(Any, None),  # never invoked: these tests review, not run
    )


def _review_requesting_match(*, message_review: bool = False, tool_review: bool = False) -> Any:
    from parlant.core.common import Weight, generate_id
    from parlant.core.engines.rule_evaluation import RuleEvaluation
    from parlant.core.rules import Rule, RuleId

    now = datetime.now(timezone.utc)
    rule = Rule(
        id=RuleId(generate_id()),
        creation_utc=now,
        modified_utc=now,
        condition="c",
        content="a",
        enabled=True,
        groups=[],
        metadata={},
        weight=Weight.MEDIUM,
        message_review=message_review,
        tool_review=tool_review,
    )
    return RuleEvaluation(rule=rule, reasoning="matched", metadata={})


async def _review(
    reviewer: _SpyReviewer,
    *,
    effort: Effort,
    consequential: bool,
    agent_tool_review: bool = False,
    rule_requests_tool_review: bool = False,
) -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "please charge it")])
    context.state = ResponseState(agent_effort=effort, agent_tool_review=agent_tool_review)
    if rule_requests_tool_review:
        context.state.ordinary_rule_matches = [_review_requesting_match(tool_review=True)]
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


async def test_that_max_effort_alone_no_longer_triggers_tool_review() -> None:
    reviewer = _SpyReviewer()
    await _review(reviewer, effort=Effort.MAX, consequential=False)
    assert not reviewer.called


async def test_that_tool_review_runs_when_a_matched_rule_requests_it() -> None:
    reviewer = _SpyReviewer()
    await _review(
        reviewer, effort=Effort.MEDIUM, consequential=False, rule_requests_tool_review=True
    )
    assert reviewer.called


async def test_that_tool_review_runs_when_the_agent_defaults_to_tool_review() -> None:
    reviewer = _SpyReviewer()
    await _review(reviewer, effort=Effort.MEDIUM, consequential=False, agent_tool_review=True)
    assert reviewer.called


async def test_that_min_effort_skips_tool_review_even_when_requested() -> None:
    reviewer = _SpyReviewer()
    await _review(
        reviewer,
        effort=Effort.MIN,
        consequential=True,
        agent_tool_review=True,
        rule_requests_tool_review=True,
    )
    assert not reviewer.called


# --- Message review gating -------------------------------------------------------
#
# Message review is requested declaratively: by the agent-level default or by any
# turn-matched rule. Effort's only whether-role is the MIN floor.


class _MessageSpyReviewer:
    def __init__(self) -> None:
        self.called = False

    async def review_message(self, context: Any, message: str) -> Any:
        self.called = True
        return None


def _message_step_result() -> StepResult:
    return StepResult(
        message=Message(role=Role.ASSISTANT, parts=[TextPart(text="Here you go.")]),
        finish_reason=FinishReason.STOP,
        usage=Usage(),
    )


def _message_review_context(
    *,
    effort: Effort,
    agent_message_review: bool = False,
    rule_requests: bool = False,
) -> Any:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(agent_effort=effort, agent_message_review=agent_message_review)
    if rule_requests:
        context.state.ordinary_rule_matches = [_review_requesting_match(message_review=True)]
    return context


async def _review_message(
    reviewer: _MessageSpyReviewer,
    *,
    effort: Effort,
    agent_message_review: bool = False,
    rule_requests: bool = False,
) -> None:
    context = _message_review_context(
        effort=effort, agent_message_review=agent_message_review, rule_requests=rule_requests
    )
    loop = _make_blocking_loop()
    loop._reviewer = cast(Any, reviewer)
    await loop._maybe_review_message(context, _message_step_result())


async def test_that_a_message_is_reviewed_when_the_agent_defaults_to_message_review() -> None:
    reviewer = _MessageSpyReviewer()
    await _review_message(reviewer, effort=Effort.MEDIUM, agent_message_review=True)
    assert reviewer.called


async def test_that_a_message_is_reviewed_when_a_matched_rule_requests_it() -> None:
    reviewer = _MessageSpyReviewer()
    await _review_message(reviewer, effort=Effort.MEDIUM, rule_requests=True)
    assert reviewer.called


async def test_that_a_message_is_not_reviewed_when_nobody_requests_it() -> None:
    reviewer = _MessageSpyReviewer()
    await _review_message(reviewer, effort=Effort.MAX)
    assert not reviewer.called


async def test_that_min_effort_skips_message_review_even_when_requested() -> None:
    reviewer = _MessageSpyReviewer()
    await _review_message(
        reviewer, effort=Effort.MIN, agent_message_review=True, rule_requests=True
    )
    assert not reviewer.called


async def test_that_streaming_buffers_the_message_exactly_when_review_will_run() -> None:
    loop = _make_blocking_loop()

    assert loop._should_buffer_message_for_review(
        _message_review_context(effort=Effort.MEDIUM, agent_message_review=True)
    )
    assert loop._should_buffer_message_for_review(
        _message_review_context(effort=Effort.MEDIUM, rule_requests=True)
    )
    assert not loop._should_buffer_message_for_review(_message_review_context(effort=Effort.MAX))
    assert not loop._should_buffer_message_for_review(
        _message_review_context(effort=Effort.MIN, agent_message_review=True)
    )


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
    cast(_ModelRouterStub, loop._model_router).react = _NoReplayReact()

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
    metadata = metadata_utils.set_internal_value(
        {},
        (_PROVIDER_DATA_KEY,),
        {"provider": "gemini", "model": "stale-model"},
    )

    messages = loop._build_tool_event_messages(data, metadata, "sess.compass")

    assert len(messages) == 1
    assert messages[0].role == Role.TOOL
    result_part = cast(ToolResultPart, messages[0].parts[0])
    # Rendered as a result (not dropped); the result data is carried through, whatever
    # the exact rendering (raw value or a descriptive string).
    assert "delivered" in str(result_part.content)


class _RecordingReplayReact:
    """Records the routed model forwarded into ``deserialize_tool_messages``."""

    provider_name = "fake"

    def __init__(self) -> None:
        self.deserialized_with_model: str | None = "UNSET"

    def resolve_model(self, hints: dict[str, Any]) -> str:
        assert hints == {}
        return "configured-model"

    def deserialize_tool_messages(
        self, deserializer: Any, *, model: str | None = None
    ) -> list[Message]:
        self.deserialized_with_model = model
        return [Message(role=Role.TOOL, parts=[ToolResultPart(call_id="1", name="t", content="x")])]


async def test_that_history_building_uses_the_routed_model_for_tool_replay() -> None:
    # Persisted tool events are replayed against the routed generator's model.
    react = _RecordingReplayReact()
    builder = _InteractionHistoryBuilder(
        StdoutLogger(LocalTracer()),
        lambda data, metadata: data["message"],
    )

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
                metadata=metadata_utils.set_internal_value(
                    {},
                    (_PROVIDER_DATA_KEY,),
                    {"provider": "gemini", "model": "stale-model"},
                ),
            ),
        )
    ]

    job = LoopJob(
        context=context,
        system_instructions="SYSTEM",
    )

    await builder.build(job, react=cast(Any, react))

    assert react.deserialized_with_model == "configured-model"


async def test_that_tool_call_step_is_committed_before_tool_results_are_appended() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(tool_ids_by_name={"test_tool": ToolId("local", "test_tool")})

    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = _NoopToolMessageReact()
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

    router = cast(_ModelRouterStub, loop._model_router)
    committed = await loop._process_tool_event(
        job,
        state,
        StepCompleted(result=result),
        cast(Any, router.react),
        router.tokenizer,
    )

    assert committed is True
    assert [m.role for m in state.history] == [Role.ASSISTANT, Role.TOOL]
    assert state.history[0].tool_calls == [tool_call]
    assert state.history[1].tool_results[0].content == {"ok": True}
    assert state.steps == [result]


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
    cast(_ModelRouterStub, loop._model_router).react = react
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
    cast(_ModelRouterStub, loop._model_router).react = react
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
    cast(_ModelRouterStub, loop._model_router).react = react
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
    # Tool events persist the qualified "service:tool" id (consumers parse it
    # with ToolId.from_string); the react layer maps it back to the bare name
    # on history replay.
    assert tool_data["tool_calls"][0]["tool_id"] == "test_service:lookup_account"
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
        await loop._commit_react_event(state, event, cast(Any, loop._model_router).react)

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


async def test_that_max_steps_forces_a_final_message_with_tools_disabled() -> None:
    class _ReactHedgePolicy(CompassOptimizationPolicy):
        def get_hedge_timeout(
            self,
            route: str,
            *,
            qualifiers: Mapping[str, str] | None = None,
        ) -> float | None:
            assert route == "react"
            return 3.25

    agent = replace(create_agent(), max_steps=1)
    context = create_engine_context(
        conversation=[(EventSource.CUSTOMER, "please help")],
        agent=agent,
    )
    context.state = ResponseState()
    context.dependencies = CompassDependencies(_ReactHedgePolicy())

    react = _EmptyThenMessageReact()
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = react

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    assert len(react.calls) == 2
    assert react.calls[0]["tool_choice"] == "auto"
    assert react.calls[1]["tool_choice"] == "none"
    assert react.calls[1]["tools"] == []
    assert all(call["hints"]["hedge_timeout"] == 3.25 for call in react.calls)
    assert any(
        "You must now explain to the user" in message.text for message in react.calls[1]["history"]
    )

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]
    assert cast(dict[str, Any], message_events[-1].data)["message"] == (
        "I'm sorry, I'm not able to help with that right now."
    )


async def test_that_the_review_retry_budget_comes_from_the_optimization_policy() -> None:
    class _TightBudgetPolicy(CompassOptimizationPolicy):
        def get_max_review_restarts(self, effort: Effort) -> int:
            assert effort == Effort.HIGH
            return 2

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "please charge it")])
    context.state = ResponseState(agent_effort=Effort.HIGH)
    context.dependencies = CompassDependencies(_TightBudgetPolicy())
    _offer_tool(context, _tool("charge_card", consequential=True))

    react = _RejectedToolsThenMessageReact()
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = react
    loop._reviewer = cast(Any, _RejectingReviewer())

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    # 2 policy-budgeted restarts, then the forced tool-less final message.
    assert len(react.calls) == 3
    assert [call["tool_choice"] for call in react.calls[:-1]] == ["auto"] * 2
    assert react.calls[-1]["tool_choice"] == "none"


async def test_that_max_semantic_failures_force_a_final_message_with_tools_disabled() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "please charge it")])
    context.state = ResponseState(agent_effort=Effort.HIGH)
    # The reviewer only runs (and can reject) when a consequential tool is called.
    _offer_tool(context, _tool("charge_card", consequential=True))

    react = _RejectedToolsThenMessageReact()
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = react
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


async def test_that_blocking_tool_preambles_are_allowed_after_fifteen_seconds() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    loop = _make_blocking_loop()
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


# --- Engine hooks --------------------------------------------------------------
#
# The Compass loop fires hooks through ``context.dependencies.hooks``. These tests
# register recorders on that shared object and drive the loop (or a single method)
# to assert each hook fires with the expected payload.


def _record_hooks(hooks: CompassHooks) -> dict[str, list[Any]]:
    recorded: dict[str, list[Any]] = {
        "step_started": [],
        "step_completed": [],
        "giving_up": [],
        "reviewing_tool_calls": [],
        "tool_calls_reviewed": [],
        "calling_tools": [],
        "tool_calls_completed": [],
        "reviewing_message": [],
        "message_reviewed": [],
    }

    def _recorder(key: str) -> Any:
        async def hook(_: Any, payload: Any, __: Any) -> CompassHookResult:
            recorded[key].append(payload)
            return CompassHookResult.CALL_NEXT

        return hook

    hooks.on_step_started.append(_recorder("step_started"))
    hooks.on_step_completed.append(_recorder("step_completed"))
    hooks.on_giving_up.append(_recorder("giving_up"))
    hooks.on_reviewing_tool_calls.append(_recorder("reviewing_tool_calls"))
    hooks.on_tool_calls_reviewed.append(_recorder("tool_calls_reviewed"))
    hooks.on_calling_tools.append(_recorder("calling_tools"))
    hooks.on_tool_calls_completed.append(_recorder("tool_calls_completed"))
    hooks.on_reviewing_message.append(_recorder("reviewing_message"))
    hooks.on_message_reviewed.append(_recorder("message_reviewed"))

    return recorded


async def test_that_the_loop_fires_step_and_tool_hooks_across_a_tool_then_message_turn() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "please check")])
    context.state = ResponseState(agent_effort=Effort.HIGH)
    _offer_tool(context, _tool("lookup_account", consequential=True))
    recorded = _record_hooks(context.dependencies.hooks)

    react = _ToolThenMessageReact()
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = react
    loop._tool_runner = cast(Any, _StubToolRunner())
    loop._reviewer = cast(Any, _SpyReviewer())

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    # Two react steps ran (tool step, then message step). step_number is the count
    # of committed steps when the step begins, and completion fires on the success
    # path of each.
    assert recorded["step_started"] == [0, 1]
    assert recorded["step_completed"] == [0, 1]

    # The tool step reviews, then calls the tool.
    assert [tc.name for tc in recorded["reviewing_tool_calls"][0].tool_calls] == ["lookup_account"]
    assert [tc.name for tc in recorded["calling_tools"][0]] == ["lookup_account"]

    reviewed = recorded["tool_calls_reviewed"][0]
    assert isinstance(reviewed, ToolCallsReviewed)
    assert reviewed.passed is True
    assert [tc.name for tc in reviewed.tool_calls] == ["lookup_account"]

    outcomes = recorded["tool_calls_completed"][0]
    assert isinstance(outcomes, dict)
    assert list(outcomes.keys()) == ["lookup_account"]
    (outcome,) = outcomes["lookup_account"]
    assert isinstance(outcome, ToolCallOutcome)
    assert outcome.result is not None and outcome.result.data == {"ok": True}


async def test_that_a_reviewed_and_passing_message_fires_the_message_review_hooks() -> None:
    context = _message_review_context(effort=Effort.MEDIUM, agent_message_review=True)
    recorded = _record_hooks(context.dependencies.hooks)

    loop = _make_blocking_loop()
    loop._reviewer = cast(Any, _MessageSpyReviewer())  # returns None => passes

    await loop._maybe_review_message(context, _message_step_result())

    assert [p.message for p in recorded["reviewing_message"]] == ["Here you go."]
    (reviewed,) = recorded["message_reviewed"]
    assert isinstance(reviewed, MessageReviewed)
    assert reviewed.message == "Here you go."
    assert reviewed.passed is True
    assert reviewed.adjusted_reasoning is None


async def test_that_a_rejected_message_review_reports_the_adjusted_reasoning() -> None:
    class _RejectingMessageReviewer:
        async def review_message(self, context: Any, message: str) -> Any:
            return type("MessageReviewResult", (), {"adjusted_reasoning": "Soften the tone."})()

    context = _message_review_context(effort=Effort.MEDIUM, agent_message_review=True)
    recorded = _record_hooks(context.dependencies.hooks)

    loop = _make_blocking_loop()
    loop._reviewer = cast(Any, _RejectingMessageReviewer())

    with pytest.raises(Exception):  # _SemanticFailure restarts the step
        await loop._maybe_review_message(context, _message_step_result())

    (reviewed,) = recorded["message_reviewed"]
    assert reviewed.passed is False
    assert reviewed.adjusted_reasoning == "Soften the tone."


async def test_that_giving_up_fires_the_on_giving_up_hook_with_reason_and_step_count() -> None:
    # Exhausting the engine steps forces a final message via _give_up, which
    # fires on_giving_up before it forces the message.
    agent = replace(create_agent(), max_steps=1)
    context = create_engine_context(
        conversation=[(EventSource.CUSTOMER, "please help")], agent=agent
    )
    context.state = ResponseState()
    recorded = _record_hooks(context.dependencies.hooks)

    react = _EmptyThenMessageReact()
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = react

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    (gave_up,) = recorded["giving_up"]
    assert isinstance(gave_up, GaveUp)
    assert gave_up.reason  # a non-empty explanation
    # One empty step was committed before the loop gave up.
    assert gave_up.steps == 1


# --- RESOLVE skips the reviewer -------------------------------------------------


async def _resolve_hook(context: Any, payload: Any, exc: Any) -> CompassHookResult:
    return CompassHookResult.RESOLVE


async def test_that_resolve_on_reviewing_message_skips_the_reviewer() -> None:
    # A hook returning RESOLVE on on_reviewing_message bypasses the reviewer and
    # the message is treated as passed.
    context = _message_review_context(effort=Effort.MEDIUM, agent_message_review=True)
    context.dependencies.hooks.on_reviewing_message.append(_resolve_hook)

    reviewer = _MessageSpyReviewer()
    loop = _make_blocking_loop()
    loop._reviewer = cast(Any, reviewer)

    # Passes (no _SemanticFailure) without ever invoking the reviewer.
    await loop._maybe_review_message(context, _message_step_result())

    assert reviewer.called is False


async def test_that_resolve_on_reviewing_tool_calls_skips_the_reviewer() -> None:
    # A consequential tool at HIGH effort would normally be reviewed; a RESOLVE
    # hook on on_reviewing_tool_calls skips the reviewer and approves the calls.
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "please check")])
    context.state = ResponseState(agent_effort=Effort.HIGH)
    _offer_tool(context, _tool("lookup_account", consequential=True))
    context.dependencies.hooks.on_reviewing_tool_calls.append(_resolve_hook)

    react = _ToolThenMessageReact()
    reviewer = _SpyReviewer()
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = react
    loop._tool_runner = cast(Any, _StubToolRunner())
    loop._reviewer = cast(Any, reviewer)

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    # The reviewer was skipped, and the tool step still completed (react ran twice:
    # the tool step, then the message step).
    assert reviewer.called is False
    assert len(react.calls) == 2


# --- BAIL on the review hooks rejects immediately --------------------------------

_DEFAULT_REJECTION_NOTE = "The review was rejected - please look into it."


async def test_that_bail_on_reviewing_message_rejects_and_regenerates_with_the_hook_note() -> None:
    context = _message_review_context(effort=Effort.MEDIUM, agent_message_review=True)

    async def reject(ctx: Any, payload: Any, exc: Any) -> CompassHookResult:
        payload.note = "Never quote another customer's balance."
        return CompassHookResult.BAIL

    context.dependencies.hooks.on_reviewing_message.append(reject)

    reviewer = _MessageSpyReviewer()
    loop = _make_blocking_loop()
    loop._reviewer = cast(Any, reviewer)

    with pytest.raises(_SemanticFailure):
        await loop._maybe_review_message(context, _message_step_result())

    assert reviewer.called is False  # the reviewer was skipped
    assert context.state.step_notes == "Never quote another customer's balance."


async def test_that_bail_on_reviewing_message_uses_the_default_note() -> None:
    context = _message_review_context(effort=Effort.MEDIUM, agent_message_review=True)

    async def reject(ctx: Any, payload: Any, exc: Any) -> CompassHookResult:
        return CompassHookResult.BAIL  # note left at its default

    context.dependencies.hooks.on_reviewing_message.append(reject)

    loop = _make_blocking_loop()
    loop._reviewer = cast(Any, _MessageSpyReviewer())

    with pytest.raises(_SemanticFailure):
        await loop._maybe_review_message(context, _message_step_result())

    assert context.state.step_notes == _DEFAULT_REJECTION_NOTE


async def test_that_bail_with_give_up_on_reviewing_message_forces_give_up() -> None:
    context = _message_review_context(effort=Effort.MEDIUM, agent_message_review=True)

    async def reject(ctx: Any, payload: Any, exc: Any) -> CompassHookResult:
        payload.give_up = True
        return CompassHookResult.BAIL

    context.dependencies.hooks.on_reviewing_message.append(reject)

    reviewer = _MessageSpyReviewer()
    loop = _make_blocking_loop()
    loop._reviewer = cast(Any, reviewer)

    with pytest.raises(_GiveUp):
        await loop._maybe_review_message(context, _message_step_result())

    assert reviewer.called is False


async def test_that_bail_with_give_up_on_reviewing_tool_calls_forces_a_final_message() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "please check")])
    context.state = ResponseState(agent_effort=Effort.HIGH)
    _offer_tool(context, _tool("lookup_account", consequential=True))

    async def reject(ctx: Any, payload: Any, exc: Any) -> CompassHookResult:
        payload.give_up = True
        return CompassHookResult.BAIL

    context.dependencies.hooks.on_reviewing_tool_calls.append(reject)

    react = _ToolThenMessageReact()
    reviewer = _SpyReviewer()
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = react
    loop._tool_runner = cast(Any, _StubToolRunner())
    loop._reviewer = cast(Any, reviewer)

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    # The reviewer was skipped and the step gave up (no tool was run) before a final
    # message was produced.
    assert reviewer.called is False
    emitter = cast(EventBuffer, context.session_event_emitter)
    assert [e for e in emitter.events if e.kind == EventKind.MESSAGE]


# --- Agent-side moderation gate ---------------------------------------------------


class _FlaggingModerationService(ModerationService):
    def __init__(self) -> None:
        self.seen_contexts: list[AgentModerationContext] = []

    async def moderate_customer(self, context: CustomerModerationContext) -> ModerationCheck:
        raise AssertionError("agent moderation must not call moderate_customer here")

    async def moderate_agent(self, context: AgentModerationContext) -> ModerationCheck:
        self.seen_contexts.append(context)
        return ModerationCheck(flagged=True, tags=["harassment"])


class _CleanModerationService(ModerationService):
    async def moderate_customer(self, context: CustomerModerationContext) -> ModerationCheck:
        return ModerationCheck(flagged=False, tags=[])


class _FlagFirstNModerationService(ModerationService):
    """Flags the first ``flag_count`` agent messages, then passes the rest."""

    def __init__(self, flag_count: int) -> None:
        self.flag_count = flag_count
        self.calls = 0

    async def moderate_customer(self, context: CustomerModerationContext) -> ModerationCheck:
        raise AssertionError("agent moderation must not call moderate_customer here")

    async def moderate_agent(self, context: AgentModerationContext) -> ModerationCheck:
        self.calls += 1
        flagged = self.calls <= self.flag_count
        return ModerationCheck(flagged=flagged, tags=["harassment"] if flagged else [])


class _CountingMessageReact(_ModelResolvingStubReact):
    """Yields one message step per call, walking ``texts`` (repeating the last)."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls = 0

    async def stream_step(self, **kwargs: Any) -> Any:
        index = min(self.calls, len(self.texts) - 1)
        self.calls += 1
        yield StepCompleted(
            result=StepResult(
                message=Message(role=Role.ASSISTANT, parts=[TextPart(text=self.texts[index])]),
                finish_reason=FinishReason.STOP,
                usage=Usage(),
            )
        )


def _agent_message_result(text: str) -> StepResult:
    return StepResult(
        message=Message(role=Role.ASSISTANT, parts=[TextPart(text=text)]),
        finish_reason=FinishReason.STOP,
        usage=Usage(),
    )


def _emitted_message_texts(context: Any) -> list[str]:
    emitter = cast(EventBuffer, context.session_event_emitter)
    return [
        cast(dict[str, Any], e.data)["message"]
        for e in emitter.events
        if e.kind == EventKind.MESSAGE
    ]


async def test_that_a_flagged_agent_message_regenerates_with_the_moderation_reason() -> None:
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).moderation_service = _FlaggingModerationService()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.state.reasoning_steps = ["original reasoning"]

    # A flag on a normal (non-give-up) message regenerates the step: it installs the
    # reason as guidance and raises _SemanticFailure rather than emitting a fallback.
    with pytest.raises(_SemanticFailure):
        await loop._maybe_moderate_message(context, _agent_message_result("something vile"))

    assert "flagged by moderation" in context.state.step_notes
    assert "harassment" in context.state.step_notes
    assert "something vile" not in context.state.step_notes  # the flagged text is never fed back
    assert context.state.reasoning_steps[-1] == context.state.step_notes


async def test_that_a_flagged_give_up_message_is_replaced_with_the_configured_fallback() -> None:
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).moderation_service = _FlaggingModerationService()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.state.giving_up = True
    context.state.moderation_fallback = "CONFIGURED FALLBACK"

    override = await loop._maybe_moderate_message(context, _agent_message_result("still vile"))

    assert override is not None
    text, moderation_metadata = override
    assert text == "CONFIGURED FALLBACK"
    assert metadata_utils.internal_value(moderation_metadata, ("moderation",)) == {
        "blocked": True,
        "tags": ["harassment"],
    }


async def test_that_agent_moderation_receives_the_turn_engine_context() -> None:
    loop = _make_blocking_loop()
    service = _FlaggingModerationService()
    cast(_ModelRouterStub, loop._model_router).moderation_service = service
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])

    with pytest.raises(_SemanticFailure):
        await loop._maybe_moderate_message(context, _agent_message_result("text"))

    assert service.seen_contexts[0].engine_context is context
    assert service.seen_contexts[0].session is context.session


async def test_that_an_unflagged_agent_message_passes_through() -> None:
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).moderation_service = _CleanModerationService()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])

    override = await loop._maybe_moderate_message(context, _agent_message_result("a fine reply"))

    assert override is None


async def test_that_agent_moderation_is_off_when_no_model_is_routed() -> None:
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).moderation_error = Exception(
        "no moderation model for route 'moderation.agent'"
    )
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])

    override = await loop._maybe_moderate_message(context, _agent_message_result("a fine reply"))

    assert override is None


async def test_that_agent_moderation_off_skips_the_gate_entirely() -> None:
    loop = _make_blocking_loop()
    router = cast(_ModelRouterStub, loop._model_router)
    router.moderation_service = _FlaggingModerationService()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.agent = replace(context.agent, moderation=Moderation.OFF)

    override = await loop._maybe_moderate_message(context, _agent_message_result("anything"))

    assert override is None
    assert router.moderation_requests == []


async def test_that_a_flagged_message_is_regenerated_and_the_clean_retry_is_emitted() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    react = _CountingMessageReact(["flagged attempt", "clean retry"])
    loop = _make_blocking_loop()
    router = cast(_ModelRouterStub, loop._model_router)
    router.react = react
    router.moderation_service = _FlagFirstNModerationService(flag_count=1)

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    texts = _emitted_message_texts(context)
    assert texts == ["clean retry"]  # the flagged attempt was never shown
    assert router.moderation_service.calls == 2  # attempt 1 flagged, attempt 2 clean


async def test_that_persistently_flagged_messages_give_up_to_the_configured_fallback() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(agent_effort=Effort.MIN)  # smallest retry budget
    react = _CountingMessageReact(["bad text"])  # always regenerates the same bad text
    loop = _make_blocking_loop()
    router = cast(_ModelRouterStub, loop._model_router)
    router.react = react
    router.moderation_service = _FlaggingModerationService()  # always flags

    await loop.run(
        LoopJob(
            context=context,
            system_instructions="SYSTEM",
            moderation_fallback="CUSTOM FALLBACK",
        )
    )

    texts = _emitted_message_texts(context)
    assert texts and texts[-1] == "CUSTOM FALLBACK"
    assert "bad text" not in texts  # the flagged text never reaches the customer


async def test_that_a_clean_give_up_message_is_emitted_rather_than_the_fallback() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(agent_effort=Effort.MIN)  # smallest retry budget
    react = _CountingMessageReact(["bad text", "here is why I couldn't help"])
    loop = _make_blocking_loop()
    router = cast(_ModelRouterStub, loop._model_router)
    router.react = react
    # Flag only the first (normal) attempt; the forced give-up message passes moderation.
    router.moderation_service = _FlagFirstNModerationService(flag_count=1)

    await loop.run(
        LoopJob(
            context=context,
            system_instructions="SYSTEM",
            moderation_fallback="UNUSED FALLBACK",
        )
    )

    texts = _emitted_message_texts(context)
    # A clean give-up message is shown as-is; the fallback is only for a flagged give-up.
    assert texts == ["here is why I couldn't help"]
    assert "UNUSED FALLBACK" not in texts
    assert "bad text" not in texts


async def test_that_the_give_up_fallback_event_carries_the_moderation_metadata() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(agent_effort=Effort.MIN)
    react = _CountingMessageReact(["bad text"])
    loop = _make_blocking_loop()
    router = cast(_ModelRouterStub, loop._model_router)
    router.react = react
    router.moderation_service = _FlaggingModerationService()  # always flags

    await loop.run(
        LoopJob(context=context, system_instructions="SYSTEM", moderation_fallback="FALLBACK")
    )

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]
    assert cast(dict[str, Any], message_events[-1].data)["message"] == "FALLBACK"
    assert metadata_utils.internal_value(message_events[-1].metadata, ("moderation",)) == {
        "blocked": True,
        "tags": ["harassment"],
    }


async def test_that_message_review_and_moderation_share_one_retry_budget() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(agent_effort=Effort.MEDIUM)  # budget of 3
    context.state.agent_message_review = True  # turn on message review
    react = _CountingMessageReact(["attempt one", "attempt two", "attempt three"])
    loop = _make_blocking_loop()
    router = cast(_ModelRouterStub, loop._model_router)
    router.react = react
    router.moderation_service = _FlagFirstNModerationService(flag_count=1)

    review_calls = 0

    async def reject_first_review(_ctx: Any, _payload: Any, _exc: Any) -> CompassHookResult:
        nonlocal review_calls
        review_calls += 1
        # Reject the first message via review; pass (skipping the null reviewer) after.
        return CompassHookResult.BAIL if review_calls == 1 else CompassHookResult.RESOLVE

    context.dependencies.hooks.on_reviewing_message.append(reject_first_review)

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    # Attempt 1 rejected by review, attempt 2 flagged by moderation, attempt 3 clean.
    # Both failures counted against the single shared budget (3), leaving room to succeed.
    texts = _emitted_message_texts(context)
    assert texts == ["attempt three"]


async def test_that_moderation_is_skipped_for_a_tool_call_step() -> None:
    loop = _make_blocking_loop()
    router = cast(_ModelRouterStub, loop._model_router)
    router.moderation_service = _FlaggingModerationService()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])

    tool_result = StepResult(
        message=Message(
            role=Role.ASSISTANT, parts=[ToolCallPart(id="c1", name="do_thing", args={})]
        ),
        finish_reason=FinishReason.STOP,
        usage=Usage(),
    )

    override = await loop._maybe_moderate_message(context, tool_result)

    assert override is None
    assert router.moderation_requests == []  # the gate returned before resolving a service


async def test_that_moderation_is_skipped_for_an_empty_message() -> None:
    loop = _make_blocking_loop()
    router = cast(_ModelRouterStub, loop._model_router)
    router.moderation_service = _FlaggingModerationService()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])

    override = await loop._maybe_moderate_message(context, _agent_message_result("   "))

    assert override is None
    assert router.moderation_requests == []


class _PausingMessageReact(_ModelResolvingStubReact):
    """Streams one message step, pausing mid-generation until released."""

    def __init__(self) -> None:
        self.generation_started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_step(self, **kwargs: Any) -> Any:
        self.generation_started.set()
        await self.release.wait()
        yield StepCompleted(
            result=StepResult(
                message=Message(
                    role=Role.ASSISTANT,
                    parts=[TextPart(text="A reply the customer already superseded.")],
                ),
                finish_reason=FinishReason.STOP,
                usage=Usage(),
            )
        )


async def test_that_a_cancellation_during_generation_abandons_the_reply_before_emission() -> None:
    """A newer customer message cancels the in-flight turn while the step
    shield defers the cancellation; the loop must notice the pending
    cancellation before emitting anything user-visible, so the superseded
    reply never reaches the session."""
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "what are bananas?")])
    context.state = ResponseState(agent_effort=Effort.MIN)

    react = _PausingMessageReact()
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = react

    run_task = asyncio.create_task(loop.run(LoopJob(context=context, system_instructions="SYSTEM")))
    await react.generation_started.wait()

    run_task.cancel()
    react.release.set()

    with pytest.raises(asyncio.CancelledError):
        await run_task

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]
    assert not message_events


class _EventRecordingTracer(LocalTracer):
    def __init__(self) -> None:
        super().__init__()
        self.recorded_events: list[tuple[str, dict[str, Any]]] = []

    def add_event(self, name: str, attributes: Any = {}) -> None:
        self.recorded_events.append((name, dict(attributes)))


class _SimpleMessageReact(_ModelResolvingStubReact):
    async def stream_step(
        self,
        *,
        history: list[Message],
        tools: list[Any],
        tool_choice: str,
        reasoning: Any,
        hints: dict[str, Any],
    ) -> Any:
        yield StepCompleted(
            result=StepResult(
                message=Message(role=Role.ASSISTANT, parts=[TextPart(text="Here you go.")]),
                finish_reason=FinishReason.STOP,
                usage=Usage(),
            )
        )


async def test_that_a_completed_reply_carries_an_output_ref_joined_to_the_loop_message_event() -> (
    None
):
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(agent_effort=Effort.MIN)

    tracer = _EventRecordingTracer()
    loop = _make_blocking_loop(tracer=tracer)
    cast(_ModelRouterStub, loop._model_router).react = _SimpleMessageReact()

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]
    assert len(message_events) == 1

    metadata = cast(dict[str, Any], message_events[0].metadata)
    output_ref = metadata["__dashboard__"]["output_ref"]
    assert output_ref

    loop_messages = [a for n, a in tracer.recorded_events if n == "loop.message"]
    assert len(loop_messages) == 1
    assert loop_messages[0]["output_ref"] == output_ref
    assert loop_messages[0]["mode"] == "block"


async def test_that_an_abandoned_blocking_reply_produces_no_loop_message_event() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "what are bananas?")])
    context.state = ResponseState(agent_effort=Effort.MIN)

    tracer = _EventRecordingTracer()
    react = _PausingMessageReact()
    loop = _make_blocking_loop(tracer=tracer)
    cast(_ModelRouterStub, loop._model_router).react = react

    run_task = asyncio.create_task(loop.run(LoopJob(context=context, system_instructions="SYSTEM")))
    await react.generation_started.wait()

    run_task.cancel()
    react.release.set()

    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert not any(n == "loop.message" for n, _ in tracer.recorded_events)


async def _noop_before_tool_calls(context: Any, tool_calls: Any) -> None: ...


async def test_that_tool_execution_carries_the_requested_call_id_through_the_scope() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(agent_effort=Effort.MIN)
    context.state.tool_ids_by_name = {"get_price": ToolId("svc", "get_price")}

    tracer = _EventRecordingTracer()
    context.tracer = tracer

    observed_scope_ids: list[str | None] = []

    class _ScopeObservingRunner:
        async def run_tool(self, ctx: Any, tool_id: Any, arguments: Any) -> ToolResult:
            observed_scope_ids.append(CompassTracer(tracer).current_tool_call_id())
            return ToolResult(data={"price": 3})

    controller = _ToolStepController(
        logger=StdoutLogger(LocalTracer()),
        tool_runner=lambda: cast(Any, _ScopeObservingRunner()),
        reviewer=lambda: cast(Any, None),
        before_tool_calls=_noop_before_tool_calls,
    )

    calls = [
        ToolCallPart(id="call-a", name="get_price", args={"item": "apple"}),
        ToolCallPart(id="call-b", name="get_price", args={"item": "pear"}),
    ]

    results = await safe_gather(
        *(controller.run_tool_call(context, tool_call) for tool_call in calls)
    )
    assert all(r is not None for r in results)
    assert set(observed_scope_ids) == {"call-a", "call-b"}


async def test_that_a_tool_step_emits_requested_events_for_each_call() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(agent_effort=Effort.MIN)
    context.state.tool_ids_by_name = {"get_price": ToolId("svc", "get_price")}

    tracer = _EventRecordingTracer()
    context.tracer = tracer

    class _NullRunner:
        async def run_tool(self, ctx: Any, tool_id: Any, arguments: Any) -> ToolResult:
            return ToolResult(data={"ok": True})

    class _SerializingReact(_ModelResolvingStubReact):
        def serialize_tool_messages(self, *args: Any, **kwargs: Any) -> None:
            return None

    controller = _ToolStepController(
        logger=StdoutLogger(LocalTracer()),
        tool_runner=lambda: cast(Any, _NullRunner()),
        reviewer=lambda: cast(Any, None),
        before_tool_calls=_noop_before_tool_calls,
    )

    state = _LoopState()

    await controller.run_tool_calls(
        context,
        state,
        [ToolCallPart(id="call-1", name="get_price", args={"item": "apple"})],
        cast(Any, _SerializingReact()),
        ZeroEstimatingTokenizer(),
    )

    requested = [a for n, a in tracer.recorded_events if n == "tool.requested"]
    assert len(requested) == 1
    assert requested[0]["tool_call_id"] == "call-1"
    assert requested[0]["tool_name"] == "get_price"


async def test_that_on_messages_emitted_receives_the_messages_the_turn_delivered() -> None:
    # The hook's payload is the turn's delivered message texts, in order — so a
    # handler can act on what the customer actually received without re-reading
    # the session (the interaction on the context is a turn-start snapshot).
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "please check")])
    context.state = ResponseState(agent_effort=Effort.HIGH)
    _offer_tool(context, _tool("lookup_account", consequential=True))

    emitted: list[Any] = []

    async def on_messages_emitted(_: Any, payload: Any, __: Any) -> CompassHookResult:
        emitted.append(payload)
        return CompassHookResult.CALL_NEXT

    context.dependencies.hooks.on_messages_emitted.append(on_messages_emitted)

    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = _ToolThenMessageReact()
    loop._tool_runner = cast(Any, _StubToolRunner())
    loop._reviewer = cast(Any, _SpyReviewer())

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    assert emitted == [["I checked that and can help."]]


async def test_that_a_tool_preamble_and_the_final_message_are_both_recorded_in_order() -> None:
    # A turn can deliver more than one bubble: the pre-tool preamble is its own
    # message, and the reply that follows is another. Both belong in the payload,
    # in the order the customer saw them.
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    loop = _make_blocking_loop()
    state = _LoopState(preamble=_encouraged_preamble_state())

    result = StepResult(
        message=Message(
            role=Role.ASSISTANT,
            parts=[TextPart(text="Let me check that.It's all set.")],
        ),
        finish_reason=FinishReason.STOP,
        usage=Usage(),
    )

    events = [
        TextDelta(text="Let me check that."),
        ToolCallStarted(id="call-1", name="lookup_account"),
        TextDelta(text="It's all set."),
        StepCompleted(result=result),
    ]
    for event in events:
        await loop._surface_message_event(context, state, event)

    assert context.state.emitted_messages == ["Let me check that.", "It's all set."]


# --- on_giving_up takes over the final message ----------------------------------
#
# When the loop runs out of road it normally spends one more model call producing
# an "I couldn't help" message. RESOLVE means that work is handled: the hook emits
# whatever it wants and the loop skips its own message. BAIL has no negative branch
# to take here (the turn is ending either way), so it falls back to CALL_NEXT.


def _give_up_taker(
    verdict: CompassHookResult,
    *,
    say: str | None = None,
    seen: list[Any] | None = None,
) -> Any:
    async def hook(ctx: Any, payload: Any, exc: Any) -> CompassHookResult:
        if seen is not None:
            seen.append(payload)
        if say is not None:
            await ctx.session_event_emitter.emit_message_event(
                trace_id=ctx.tracer.trace_id,
                data=say,
            )
        return verdict

    return hook


def _max_steps_context() -> Any:
    """A turn that will exhaust its single engine step without answering."""
    agent = replace(create_agent(), max_steps=1)
    context = create_engine_context(
        conversation=[(EventSource.CUSTOMER, "please help")], agent=agent
    )
    context.state = ResponseState()
    return context


def _message_texts(context: Any) -> list[str]:
    emitter = cast(EventBuffer, context.session_event_emitter)
    return [
        cast(dict[str, Any], e.data)["message"]
        for e in emitter.events
        if e.kind == EventKind.MESSAGE
    ]


async def test_that_on_giving_up_can_take_over_the_final_message() -> None:
    context = _max_steps_context()
    context.dependencies.hooks.on_giving_up.append(
        _give_up_taker(CompassHookResult.RESOLVE, say="A colleague will take this from here.")
    )

    react = _EmptyThenMessageReact()
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = react

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    # The hook's message is the only one, and the forced tools-disabled step that
    # would have generated Parlant's own apology never ran.
    assert _message_texts(context) == ["A colleague will take this from here."]
    assert all(call["tool_choice"] != "none" for call in react.calls)


async def test_that_bail_on_giving_up_is_treated_as_call_next() -> None:
    # BAIL is the negative branch of whatever a hook brackets, and giving up has
    # no such branch — the turn is ending regardless. So it must not silently
    # double as a take-over; RESOLVE is the verdict that claims the response.
    context = _max_steps_context()
    context.dependencies.hooks.on_giving_up.append(_give_up_taker(CompassHookResult.BAIL))

    react = _EmptyThenMessageReact()
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = react

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    assert react.calls[-1]["tool_choice"] == "none"  # the forced step still ran
    assert _message_texts(context) == ["I'm sorry, I'm not able to help with that right now."]


async def test_that_parlant_forces_its_own_message_when_a_hook_declines_to_take_over() -> None:
    # CALL_NEXT is the observer's return value, so a plain audit hook must not
    # accidentally suppress the final message.
    context = _max_steps_context()
    seen: list[Any] = []
    context.dependencies.hooks.on_giving_up.append(
        _give_up_taker(CompassHookResult.CALL_NEXT, seen=seen)
    )

    react = _EmptyThenMessageReact()
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = react

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    assert len(seen) == 1
    assert react.calls[-1]["tool_choice"] == "none"  # the forced step still ran
    assert _message_texts(context) == ["I'm sorry, I'm not able to help with that right now."]


async def test_that_a_take_over_that_emits_nothing_ends_the_turn_silently() -> None:
    # A deliberate silence: a human is already watching the thread, so the hook
    # claims the response and says nothing. The turn must still finish cleanly.
    context = _max_steps_context()
    context.dependencies.hooks.on_giving_up.append(_give_up_taker(CompassHookResult.RESOLVE))

    react = _EmptyThenMessageReact()
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = react

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    assert _message_texts(context) == []
    assert context.state.prepared_to_respond is True


async def test_that_hooks_after_the_one_taking_over_do_not_run() -> None:
    context = _max_steps_context()
    later_ran: list[Any] = []
    context.dependencies.hooks.on_giving_up.append(
        _give_up_taker(CompassHookResult.RESOLVE, say="Taken.")
    )
    context.dependencies.hooks.on_giving_up.append(
        _give_up_taker(CompassHookResult.CALL_NEXT, seen=later_ran)
    )

    react = _EmptyThenMessageReact()
    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = react

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    assert later_ran == []
    assert _message_texts(context) == ["Taken."]


# --- why the loop gave up -------------------------------------------------------


async def test_that_the_give_up_payload_names_exhausted_steps() -> None:
    context = _max_steps_context()
    seen: list[Any] = []
    context.dependencies.hooks.on_giving_up.append(
        _give_up_taker(CompassHookResult.RESOLVE, say="ok", seen=seen)
    )

    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = _EmptyThenMessageReact()

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    (gave_up,) = seen
    assert isinstance(gave_up, GaveUp)
    assert gave_up.kind is GiveUpKind.MAX_STEPS
    assert gave_up.steps == 1
    assert gave_up.reason  # the human-readable explanation is still there


async def test_that_the_give_up_payload_names_exhausted_retries() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "please charge it")])
    context.state = ResponseState(agent_effort=Effort.HIGH)
    _offer_tool(context, _tool("charge_card", consequential=True))
    seen: list[Any] = []
    context.dependencies.hooks.on_giving_up.append(
        _give_up_taker(CompassHookResult.RESOLVE, say="ok", seen=seen)
    )

    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = _RejectedToolsThenMessageReact()
    loop._reviewer = cast(Any, _RejectingReviewer())

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    (gave_up,) = seen
    assert gave_up.kind is GiveUpKind.RETRIES_EXHAUSTED


async def test_that_the_give_up_payload_names_a_hook_request() -> None:
    # A review hook that BAILs with give_up set is a different situation from
    # running out of budget, and a handler should be able to tell them apart.
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "please check")])
    context.state = ResponseState(agent_effort=Effort.HIGH)
    _offer_tool(context, _tool("lookup_account", consequential=True))

    async def demand_give_up(ctx: Any, payload: Any, exc: Any) -> CompassHookResult:
        payload.give_up = True
        return CompassHookResult.BAIL

    context.dependencies.hooks.on_reviewing_tool_calls.append(demand_give_up)

    seen: list[Any] = []
    context.dependencies.hooks.on_giving_up.append(
        _give_up_taker(CompassHookResult.RESOLVE, say="ok", seen=seen)
    )

    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = _ToolThenMessageReact()
    loop._tool_runner = cast(Any, _StubToolRunner())
    loop._reviewer = cast(Any, _SpyReviewer())

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    (gave_up,) = seen
    assert gave_up.kind is GiveUpKind.HOOK_REQUESTED


# --- the forced message is not reviewed -----------------------------------------


async def test_that_message_review_is_skipped_while_giving_up() -> None:
    # The forced message is the last thing the turn can say. Reviewing it can only
    # reject it, and a rejection there has nowhere to go but silence.
    context = _message_review_context(effort=Effort.MEDIUM, agent_message_review=True)
    loop = _make_blocking_loop()

    assert loop._should_review_message(context) is True

    context.state.giving_up = True

    assert loop._should_review_message(context) is False


async def test_that_a_rejecting_reviewer_cannot_silence_the_forced_message() -> None:
    # End to end on the same point: a reviewer that rejects everything exhausts the
    # budget, and the forced message still reaches the customer.
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "please charge it")])
    context.state = ResponseState(agent_effort=Effort.HIGH, agent_message_review=True)
    _offer_tool(context, _tool("charge_card", consequential=True))

    loop = _make_blocking_loop()
    cast(_ModelRouterStub, loop._model_router).react = _RejectedToolsThenMessageReact()
    loop._reviewer = cast(Any, _RejectingReviewer())

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    assert _message_texts(context) == ["I'm sorry, I'm not able to help with that right now."]


class _PhaseRecordingTracer(LocalTracer):
    """Records each event with the step it was emitted in, the way the collector
    reads it: the step_index of the enclosing span."""

    def __init__(self) -> None:
        super().__init__()
        self.phases: list[tuple[str, object, dict[str, Any]]] = []
        self.times: list[float] = []

    def add_event(self, name: str, attributes: Any = {}) -> None:
        if name.startswith("loop.") and name.split(".")[-1] in ("started", "finished"):
            self.phases.append((name, self.get_attribute("step_index"), dict(attributes)))
            self.times.append(time.monotonic())


class _ReasonToolThenReasonAndAnswerReact(_ModelResolvingStubReact):
    """Step 0 reasons and calls a tool; step 1 reasons and answers."""

    def __init__(self, fail_after_reasoning: bool = False) -> None:
        self.calls = 0
        self.fail_after_reasoning = fail_after_reasoning

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
        self.calls += 1
        yield ReasoningDelta(text="Thinking it through.")
        if self.fail_after_reasoning:
            raise RuntimeError("provider failed mid-step")

        if self.calls == 1:
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

        yield TextDelta(text="Your account is active.")
        yield StepCompleted(
            result=StepResult(
                message=Message(
                    role=Role.ASSISTANT, parts=[TextPart(text="Your account is active.")]
                ),
                finish_reason=FinishReason.STOP,
                usage=Usage(),
            )
        )


async def _run_traced_turn(
    react: _ReasonToolThenReasonAndAnswerReact,
    tracer: _PhaseRecordingTracer | None = None,
) -> _PhaseRecordingTracer:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "is my account ok?")])
    context.state = ResponseState()
    _offer_tool(context, _tool("lookup_account", consequential=False))
    tracer = tracer or _PhaseRecordingTracer()
    loop = _make_blocking_loop(tracer=tracer)
    cast(_ModelRouterStub, loop._model_router).react = react
    loop._tool_runner = cast(Any, _StubToolRunner())
    context.tracer = tracer

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))
    return tracer


async def test_that_a_tool_step_traces_its_tools_phase_within_the_step() -> None:
    # The Dashboard's per-step durations are paired from loop.<phase>.started /
    # .finished events by the step_index of the span they were emitted in.
    tracer = await _run_traced_turn(_ReasonToolThenReasonAndAnswerReact())

    tools = [(name, step, attrs) for name, step, attrs in tracer.phases if ".tools." in name]
    assert [(name, step) for name, step, _ in tools] == [
        ("loop.tools.started", 0),
        ("loop.tools.finished", 0),
    ]
    assert tools[0][2]["tool_count"] == 1
    assert tools[1][2]["outcome"] == "completed"


async def test_that_each_step_traces_its_reasoning_and_message_phases_with_its_step_index() -> None:
    tracer = await _run_traced_turn(_ReasonToolThenReasonAndAnswerReact())

    reasoning_and_message = [
        (name, step) for name, step, _ in tracer.phases if ".tools." not in name
    ]
    assert reasoning_and_message == [
        ("loop.reasoning.started", 0),
        ("loop.reasoning.finished", 0),
        ("loop.reasoning.started", 1),
        ("loop.reasoning.finished", 1),
        ("loop.message.started", 1),
        ("loop.message.finished", 1),
    ]


async def test_that_a_phase_open_when_its_step_fails_is_still_finished() -> None:
    # Without a finish, the Dashboard would show the phase as running forever.
    tracer = _PhaseRecordingTracer()
    try:
        await _run_traced_turn(
            _ReasonToolThenReasonAndAnswerReact(fail_after_reasoning=True), tracer
        )
    except Exception:
        pass  # Whether the loop gives up or raises, the phase must be closed.

    reasoning = [
        (name, step, attrs) for name, step, attrs in tracer.phases if ".reasoning." in name
    ]
    assert [(name, step) for name, step, _ in reasoning][:2] == [
        ("loop.reasoning.started", 0),
        ("loop.reasoning.finished", 0),
    ]
    assert reasoning[1][2]["outcome"] == "failed"


class _ScriptedStepsReact(_ModelResolvingStubReact):
    """Plays one scripted list of stream events per step; a float is a pause."""

    def __init__(self, *steps: list[Any]) -> None:
        self.steps = list(steps)

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
        for item in self.steps.pop(0):
            if isinstance(item, float):
                await asyncio.sleep(item)
            else:
                yield item


def _completed(*parts: Any, finish: FinishReason = FinishReason.STOP) -> StepCompleted:
    return StepCompleted(
        result=StepResult(
            message=Message(role=Role.ASSISTANT, parts=list(parts)),
            finish_reason=finish,
            usage=Usage(),
        )
    )


_ANSWER = [
    TextDelta(text="Your account is active."),
    _completed(TextPart(text="Your account is active.")),
]


async def test_that_reasoning_is_timed_from_when_the_model_call_starts() -> None:
    # Gemini sends its thought summary only once it has finished thinking; timing
    # reasoning from that chunk records the thinking as almost instant.
    tracer = await _run_traced_turn(
        cast(Any, _ScriptedStepsReact([0.2, ReasoningDelta(text="Thought it through."), *_ANSWER]))
    )

    started = tracer.times[[name for name, _, _ in tracer.phases].index("loop.reasoning.started")]
    finished = tracer.times[[name for name, _, _ in tracer.phases].index("loop.reasoning.finished")]
    assert finished - started >= 0.2


async def test_that_a_step_with_no_reasoning_chunks_still_times_its_thinking() -> None:
    tracer = await _run_traced_turn(cast(Any, _ScriptedStepsReact([0.2, *_ANSWER])))

    names = [name for name, _, _ in tracer.phases]
    assert names == [
        "loop.reasoning.started",
        "loop.reasoning.finished",
        "loop.message.started",
        "loop.message.finished",
    ]
    assert tracer.phases[1][2]["chunk_count"] == 0
    assert tracer.times[1] - tracer.times[0] >= 0.2


async def test_that_a_preamble_then_a_tool_call_traces_one_message_phase_for_the_step() -> None:
    tool_call = ToolCallPart(id="call-1", name="lookup_account", args={})
    tracer = await _run_traced_turn(
        cast(
            Any,
            _ScriptedStepsReact(
                [
                    TextDelta(text="Let me check."),
                    ToolCallStarted(id=tool_call.id, name=tool_call.name),
                    _completed(
                        TextPart(text="Let me check."), tool_call, finish=FinishReason.TOOL_CALLS
                    ),
                ],
                _ANSWER,
            ),
        )
    )

    step_0_messages = [name for name, step, _ in tracer.phases if ".message." in name and step == 0]
    assert step_0_messages == ["loop.message.started", "loop.message.finished"]
