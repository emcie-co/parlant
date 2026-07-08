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
from abc import abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from io import StringIO
import json
import re
import time
from typing import Any, Literal, Optional, cast

from parlant.core.agents import Effort
from parlant.core.async_utils import CancellationSuppressionLatch, latched_shield, safe_gather
from parlant.core.common import DISABLE_WARNINGS, JSONSerializable
from parlant.core.emissions import MessageEventHandle, StatusEventHandle
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.compass.preambles import PreambleConfiguration, PreambleDecision
from parlant.core.engines.compass.response_state import EngineContext, IterationState
from parlant.core.engines.compass.tracing import CompassTracer
from parlant.core.engines.compass.loop.loop import Loop, LoopJob, LoopResult
from parlant.core.engines.compass.tool_runner import ToolRunner
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.common import ModelSize, UsageInfo
from parlant.core.nlp.react import (
    Message,
    Part,
    ReactError,
    ReactGenerator,
    ReasoningDelta,
    Role,
    StepCompleted,
    StepResult,
    StreamEvent,
    TextPart,
    ToolCallPart,
    ToolCallStarted,
    ToolMessageDeserializer,
    ToolMessageSerializer,
    ToolResultPart,
    ToolSpec,
    Usage,
    tool_specs_from_tools,
)
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.sessions import (
    EventKind,
    EventSource,
    MessageEventData,
    Participant,
    StatusEventData,
    ToolCall,
    ToolEventData,
)
from parlant.core.tools import ToolResult
from parlant.core.tracer import Tracer
from parlant.core.cost_control import CostContext, CostControlPolicy, WorkKind
from parlant.core.usage_reporter import UsageReporter
from parlant.core.engines.compass.reviewer import Reviewer, ToolCallReviewResult


# Key under which the provider's opaque tool-call replay blob is stored in a tool
# event's metadata. Read back at history-build time so the originating provider
# can faithfully replay the call (ids, signatures, …) on a later turn.
_PROVIDER_DATA_KEY = "__provider_data__"


class SessionToolMessageSerializer(ToolMessageSerializer):
    """Session-backed sink. The neutral record (tool_id/args/result) is built by
    the engine from execution and stored in the event's ``data``, so the call/
    result writes are no-ops here; we only capture the provider blob for metadata."""

    def __init__(self) -> None:
        self.provider_data: Mapping[str, JSONSerializable] = {}

    def write_calls(self, calls: Sequence[ToolCallPart]) -> None:
        return None

    def write_results(self, results: Sequence[ToolResultPart]) -> None:
        return None

    def write_provider_data(self, data: Mapping[str, Any]) -> None:
        self.provider_data = cast(Mapping[str, JSONSerializable], dict(data))


class SessionToolMessageDeserializer(ToolMessageDeserializer):
    """Session-backed source. Serves calls/results from the neutral ``ToolEventData``
    and the provider blob from the event's metadata."""

    def __init__(self, data: ToolEventData, provider_data: Mapping[str, Any]) -> None:
        self._data = data
        self._provider_data = provider_data

    def read_calls(self) -> Sequence[ToolCallPart]:
        return [
            ToolCallPart(name=call["tool_id"], args=call["arguments"])
            for call in self._data["tool_calls"]
        ]

    def read_results(self) -> Sequence[ToolResultPart]:
        results: list[ToolResultPart] = []
        for call in self._data["tool_calls"]:
            result = call.get("result", {}) or {}
            results.append(
                ToolResultPart(
                    name=call["tool_id"],
                    content=result.get("data", {}),
                    is_error="error_details" in (result.get("metadata", {}) or {}),
                )
            )
        return results

    def read_provider_data(self) -> Mapping[str, Any]:
        return self._provider_data


@dataclass
class _ToolPreambleState:
    configuration: PreambleConfiguration = field(default_factory=PreambleConfiguration.default)
    user_visible_message_emitted: bool = False
    last_user_visible_message_at: float | None = None

    def decide(self, now: float) -> PreambleDecision:
        if not self.user_visible_message_emitted or self.last_user_visible_message_at is None:
            return PreambleDecision.ALLOW_INITIAL

        if now - self.last_user_visible_message_at > self.configuration.interval_seconds:
            return PreambleDecision.ALLOW_INTERVAL_UPDATE

        return PreambleDecision.SUPPRESS

    def mark_user_visible_message_emitted(self, now: float) -> None:
        self.user_visible_message_emitted = True
        self.last_user_visible_message_at = now


class _MessageCommitMode(Enum):
    KEEP_FULL_MESSAGE = auto()
    KEEP_NO_TEXT = auto()
    KEEP_TEXT_PREFIX = auto()


@dataclass
class _MessageCommitPolicy:
    mode: _MessageCommitMode = _MessageCommitMode.KEEP_FULL_MESSAGE
    prefix_len: int = 0

    @classmethod
    def keep_full_message(cls) -> "_MessageCommitPolicy":
        return cls(_MessageCommitMode.KEEP_FULL_MESSAGE)

    @classmethod
    def keep_no_text(cls) -> "_MessageCommitPolicy":
        return cls(_MessageCommitMode.KEEP_NO_TEXT)

    @classmethod
    def keep_text_prefix(cls, prefix_len: int) -> "_MessageCommitPolicy":
        return cls(_MessageCommitMode.KEEP_TEXT_PREFIX, prefix_len)


@dataclass
class _MessageEmissionState:
    handle: MessageEventHandle | None = None
    buffer: StringIO | None = None
    chunks: list[str | None] = field(default_factory=list)
    # Chars of THIS step's message already emitted as their own (complete) bubbles
    # via interrupt-splits (text -> tool -> text). The step-completion emit covers
    # `result.message.text[emitted_len:]` — the authoritative remainder.
    emitted_len: int = 0
    commit_policy: _MessageCommitPolicy = field(
        default_factory=_MessageCommitPolicy.keep_full_message
    )

    def clear_transient_output(self) -> None:
        self.handle = None
        self.buffer = None
        self.chunks = []

    def reset_step_output(self) -> None:
        self.clear_transient_output()
        self.emitted_len = 0

    def suppress_tool_text(self) -> None:
        self.commit_policy = _MessageCommitPolicy.keep_no_text()

    def keep_tool_text_prefix(self, prefix_len: int) -> None:
        self.commit_policy = _MessageCommitPolicy.keep_text_prefix(prefix_len)

    def reset_commit_policy(self) -> None:
        self.commit_policy = _MessageCommitPolicy.keep_full_message()

    def has_custom_commit_policy(self) -> bool:
        return self.commit_policy.mode != _MessageCommitMode.KEEP_FULL_MESSAGE


PhaseName = Literal["reasoning", "message", "tools"]


@dataclass
class _LoopState:
    history: list[Message] = field(default_factory=list)
    # The first message is the stable system-instructions block. Track the last
    # logged text so the loop, not prompt construction, owns instruction logging.
    system_instructions_index: int = 0
    logged_system_instructions: str | None = None
    # Index of the turn-instructions message in `history`, so it can be replaced
    # in place when rules are reevaluated between steps. Stable because the
    # instructions sit before the last customer message and all later events are
    # appended after them.
    instructions_index: int | None = None
    in_the_middle_of_running_tools: bool = False
    active_phases: set[PhaseName] = field(default_factory=set)

    reasoning_handle: StatusEventHandle | None = None
    reasoning_buffer: StringIO | None = None
    reasoning_chunks: list[str | None] = field(default_factory=list)

    message: _MessageEmissionState = field(default_factory=_MessageEmissionState)
    preamble: _ToolPreambleState = field(default_factory=_ToolPreambleState)

    disable_tools: bool = False
    force_message_note: str = ""

    steps: list[StepResult] = field(default_factory=list)

    def mark_phase_started(self, phase: PhaseName) -> bool:
        if phase in self.active_phases:
            return False

        self.active_phases.add(phase)
        return True

    def mark_phase_finished(self, phase: PhaseName) -> bool:
        if phase not in self.active_phases:
            return False

        self.active_phases.remove(phase)
        return True


class _SemanticFailure(Exception):
    pass


class _GiveUp(Exception):
    pass


def _finish_open_loop_phases(context: EngineContext, state: _LoopState) -> None:
    compass_tracer = CompassTracer(context.tracer)

    if state.mark_phase_finished("reasoning"):
        compass_tracer.loop_reasoning_finished()

    if state.mark_phase_finished("message"):
        compass_tracer.loop_message_finished()

    if state.mark_phase_finished("tools"):
        compass_tracer.loop_tools_finished()


def _instructions_message(turn_instructions: str, cache_key: str) -> Message:
    return Message(
        role=Role.SYSTEM,
        cache_key=cache_key,
        parts=[
            TextPart(
                text=f"""\
The following is notes and context about the current state of the conversation — the rules, glossary, and tools relevant to it. Treat it as background that informs your next reply; it is NOT itself a message addressed to you, so never respond to it, acknowledge it, or refer to it.:

{turn_instructions}"""
            )
        ],
    )


def _system_instructions_message(system_instructions: str, cache_key: str) -> Message:
    return Message(
        role=Role.SYSTEM,
        cache_key=cache_key,
        parts=[TextPart(text=system_instructions)],
    )


class _InteractionHistoryBuilder:
    def __init__(self, logger: Logger, react: Callable[[], ReactGenerator]) -> None:
        self._logger = logger
        self._react = react

    async def build(
        self,
        job: LoopJob,
        *,
        include_turn_instructions: bool = True,
    ) -> tuple[list[Message], int | None]:
        cache_key = job.context.session.id

        # The model this turn will run on — used to decide whether a persisted
        # tool event can be natively replayed (some providers' replay is bound to
        # the producing model).
        replay_model = self._react().resolve_model({"model_size": job.model_size})

        system_message = _system_instructions_message(job.system_instructions, cache_key)

        history = [system_message]

        if job.context.state.session_summary:
            history.append(
                Message(
                    role=Role.SYSTEM,
                    cache_key=cache_key,
                    parts=[
                        TextPart(
                            text=f"""\
The earlier part of this session was compacted into the following summary.
Treat it as factual background context for the current interaction. It is not a new
message from the user and should not be acknowledged directly:

### Summary

{job.context.state.session_summary.strip()}
"""
                        )
                    ],
                )
            )

        for event in job.context.interaction.events:
            if event.kind == EventKind.MESSAGE and event.source == EventSource.CUSTOMER:
                history.append(
                    Message(
                        role=Role.USER,
                        cache_key=cache_key,
                        parts=[TextPart(text=cast(MessageEventData, event.data)["message"])],
                    )
                )
            elif event.source == EventSource.CUSTOMER_UI:
                history.append(
                    Message(
                        role=Role.USER,
                        cache_key=cache_key,
                        parts=[TextPart(text=f"[Customer UI Event]: {event.data}")],
                    )
                )
            elif event.kind == EventKind.MESSAGE and event.source in (
                EventSource.AI_AGENT,
                EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT,
            ):
                history.append(
                    Message(
                        role=Role.ASSISTANT,
                        cache_key=cache_key,
                        parts=[TextPart(text=cast(MessageEventData, event.data)["message"])],
                    )
                )
            elif event.source == EventSource.HUMAN_AGENT:
                message_data = cast(MessageEventData, event.data)

                history.append(
                    Message(
                        role=Role.ASSISTANT,
                        cache_key=cache_key,
                        parts=[
                            TextPart(
                                text=f"[Intervention by human agent. Name: {message_data['participant']['display_name']}]: {message_data['message']}"
                            )
                        ],
                    )
                )
            elif event.kind == EventKind.TOOL:
                # Reconstruct every tool event regardless of source (matching the
                # alpha engine). Persisted tool events can carry source=AI_AGENT,
                # so gating on SYSTEM silently dropped prior-turn tool results.
                history.extend(
                    self.build_tool_event_messages(
                        cast(ToolEventData, event.data),
                        event.metadata,
                        cache_key,
                        model=replay_model,
                    )
                )

        # Providers (e.g. Gemini, Anthropic) require at least one non-system
        # turn. When the agent speaks first — a greeting before any customer
        # message — the interaction is empty and the history holds only the
        # system message; give the model a user turn to respond to.
        if len(history) == 1:
            history.append(
                Message(
                    role=Role.USER,
                    cache_key=cache_key,
                    parts=[TextPart(text="[The conversation has not started yet.]")],
                )
            )

        # Tool events staged for this turn (e.g. retriever results emitted during
        # the on_generating_messages hook) aren't in the interaction history yet,
        # so fold them in here so the model sees the retrieved context.
        for tool_event in job.context.state.tool_events:
            history.extend(
                self.build_tool_event_messages(
                    cast(ToolEventData, tool_event.data),
                    tool_event.metadata,
                    cache_key,
                    model=replay_model,
                )
            )

        instructions_index: int | None = None

        if include_turn_instructions and job.step_instructions:
            turn_instructions = await job.step_instructions(job.context)

            # Place the instructions immediately BEFORE the last customer message
            # rather than at the very end. Ending the prompt on an imperative note
            # makes the model treat it as the turn to answer — it paraphrases or
            # echoes the instructions back instead of replying. Keeping the
            # customer's message last keeps the model answering the customer.
            instructions_index = self._last_user_message_index(history)
            history.insert(instructions_index, _instructions_message(turn_instructions, cache_key))

        return history, instructions_index

    def build_tool_event_messages(
        self,
        data: ToolEventData,
        metadata: Optional[Mapping[str, JSONSerializable]],
        cache_key: str,
        *,
        model: Optional[str] = None,
    ) -> list[Message]:
        provider_data = (metadata or {}).get(_PROVIDER_DATA_KEY)
        if isinstance(provider_data, Mapping) and provider_data:
            # A model-issued tool call carrying its provider's replay blob: let the
            # originating provider rebuild a faithful, native tool_use/tool_result
            # pair (consistent ids, signatures, …). ``model`` is the model THIS turn
            # will run on, so a provider can gate model-bound native replay on it.
            messages = self._react().deserialize_tool_messages(
                SessionToolMessageDeserializer(data, provider_data), model=model
            )
            if messages is None:
                # The current generator can't natively replay this blob (e.g. a
                # Gemini tool call whose stored thought_signature is unusable).
                # Degrade — don't drop: render the result so the model still sees
                # the tool's data across turns, rather than forgetting it entirely.
                self._logger.debug(
                    "Can't natively replay a tool event while building history "
                    f"({provider_data.get('provider')}/{provider_data.get('model')}); "
                    "falling back to a result-only rendering."
                )
                return self.build_result_only_tool_event_messages(data, cache_key)
            for message in messages:
                message.cache_key = cache_key
            return list(messages)

        # No provider blob (e.g. retriever-staged results, or events from before
        # this was introduced): fall back to the prior tool-result-only rendering.
        return self.build_result_only_tool_event_messages(data, cache_key)

    def build_result_only_tool_event_messages(
        self, data: ToolEventData, cache_key: str
    ) -> list[Message]:
        def build_content_with_args(call: ToolCall) -> str:
            return f"{call['tool_id']}({', '.join(f'{k}={v}' for k, v in call['arguments'].items())}) returned: {call.get('result', {}).get('data', {})}"

        messages: list[Message] = []

        call_id = 0
        for call in data["tool_calls"]:
            call_id += 1
            is_error = "error_details" in call.get("result", {}).get("metadata", {})

            messages.append(
                Message(
                    role=Role.TOOL,
                    cache_key=cache_key,
                    parts=[
                        ToolResultPart(
                            call_id=str(call_id),
                            name=call["tool_id"],
                            content=build_content_with_args(call),
                            is_error=is_error,
                        )
                    ],
                )
            )

        return messages

    def _last_user_message_index(self, history: Sequence[Message]) -> int:
        return next(
            (i for i in range(len(history) - 1, -1, -1) if history[i].role == Role.USER),
            len(history),
        )


class _TurnInstructionBuilder:
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    async def refresh(
        self,
        job: LoopJob,
        state: _LoopState,
        preamble_decision: PreambleDecision,
        loop_name: str,
    ) -> None:
        if job.step_instructions is not None:
            instructions = await job.step_instructions(job.context)
        else:
            instructions = ""

        reviewer_notes = self._build_reviewer_notes(job, state, preamble_decision)

        if reviewer_notes:
            instructions += (
                "\n\n### IMPORTANT: Please mind the following notes for your next step:\n\n"
                + "\n\n".join(reviewer_notes)
            )

        refreshed_instructions = _instructions_message(
            instructions,
            job.context.session.id,
        )

        if state.instructions_index is not None:
            if state.history[state.instructions_index].text != refreshed_instructions.text:
                self._log_instruction_change(loop_name, "updated", refreshed_instructions.text)

                state.history[state.instructions_index] = refreshed_instructions
        elif instructions:
            # No instructions message exists (e.g. a config with no per-turn
            # step_instructions, like low-effort agents), but there's content to inject
            # — notably the reviewer's adjusted reasoning / TODO on a retry. Insert one,
            # positioned before the last customer message like history building, so it
            # actually reaches the model. Without this the review loop would re-stream
            # the same output and never converge.
            index = next(
                (
                    i
                    for i in range(len(state.history) - 1, -1, -1)
                    if state.history[i].role == Role.USER
                ),
                len(state.history),
            )
            state.history.insert(index, refreshed_instructions)
            state.instructions_index = index

            self._log_instruction_change(loop_name, "inserted", refreshed_instructions.text)

    def _log_instruction_change(
        self,
        loop_name: str,
        action: str,
        instructions: str,
    ) -> None:
        self._logger.debug(f"{loop_name} {action} turn instructions:\n\n{instructions}")

    def _build_reviewer_notes(
        self,
        job: LoopJob,
        state: _LoopState,
        preamble_decision: PreambleDecision,
    ) -> list[str]:
        reviewer_notes: list[str] = []

        if job.context.state.todo:
            reviewer_notes.append(
                "#### TODO LIST: Remaining tasks before responding to the user\n\n"
                + job.context.state.todo
            )

        if job.context.state.step_notes:
            reviewer_notes.append(
                "#### Suggested reasoning for the next step\n\n" + job.context.state.step_notes
            )

        if preamble_note := job.preamble_config.note_for(job.context, preamble_decision):
            reviewer_notes.append(preamble_note)

        if state.force_message_note:
            reviewer_notes.append("#### Required final response\n\n" + state.force_message_note)

        return reviewer_notes


class _ReasoningEventProcessor:
    async def process(
        self,
        context: EngineContext,
        state: _LoopState,
        event: StreamEvent,
    ) -> None:
        match event:
            case ReasoningDelta(text=text):
                if state.reasoning_handle is None:  # First reasoning chunk
                    if state.mark_phase_started("reasoning"):
                        CompassTracer(context.tracer).loop_reasoning_started()

                    state.reasoning_buffer = StringIO()
                    state.reasoning_buffer.write(text)
                    state.reasoning_chunks = [text]

                    status_data = StatusEventData(
                        status="processing",
                        message=state.reasoning_buffer.getvalue(),
                        chunks=state.reasoning_chunks,
                    )
                    state.reasoning_handle = await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=status_data,
                        metadata={"reasoning": True},
                    )
                else:  # Subsequent reasoning chunk
                    assert state.reasoning_buffer is not None

                    state.reasoning_buffer.write(text)
                    state.reasoning_chunks.append(text)

                    state.reasoning_handle = await state.reasoning_handle.update(
                        StatusEventData(
                            status="processing",
                            message=state.reasoning_buffer.getvalue(),
                            chunks=state.reasoning_chunks,
                        )
                    )
            case StepCompleted(result=result):
                if result.message.reasoning:
                    if state.reasoning_handle is None:
                        if state.mark_phase_started("reasoning"):
                            CompassTracer(context.tracer).loop_reasoning_started()

                        CompassTracer(context.tracer).loop_reasoning(
                            result.message.reasoning,
                            len(state.reasoning_chunks),
                        )

                    context.state.reasoning_steps.append(result.message.reasoning)

                if state.reasoning_handle is not None:
                    status_data = StatusEventData(
                        status="processing",
                        message=result.message.reasoning,
                        chunks=[*state.reasoning_chunks, None],
                    )
                    CompassTracer(context.tracer).loop_reasoning(
                        result.message.reasoning,
                        len(state.reasoning_chunks),
                    )
                    await state.reasoning_handle.update(status_data)

                if state.mark_phase_finished("reasoning"):
                    CompassTracer(context.tracer).loop_reasoning_finished(
                        len(state.reasoning_chunks),
                    )

                state.reasoning_buffer = None
                state.reasoning_chunks = []
                state.reasoning_handle = None
            case _ if (
                state.reasoning_handle is not None
            ):  # In case reasoning is followed by events other than StepCompletion
                assert state.reasoning_buffer is not None

                if state.reasoning_buffer.getvalue().strip():
                    context.state.reasoning_steps.append(state.reasoning_buffer.getvalue())

                status_data = StatusEventData(
                    status="processing",
                    message=state.reasoning_buffer.getvalue(),
                    chunks=[*state.reasoning_chunks, None],
                )
                CompassTracer(context.tracer).loop_reasoning(
                    state.reasoning_buffer.getvalue(),
                    len(state.reasoning_chunks),
                )
                await state.reasoning_handle.update(status_data)

                if state.mark_phase_finished("reasoning"):
                    CompassTracer(context.tracer).loop_reasoning_finished(
                        len(state.reasoning_chunks),
                    )

                state.reasoning_buffer = None
                state.reasoning_chunks = []
                state.reasoning_handle = None


class _ToolStepController:
    def __init__(
        self,
        logger: Logger,
        react: Callable[[], ReactGenerator],
        tool_runner: Callable[[], ToolRunner],
        reviewer: Callable[[], Reviewer],
    ) -> None:
        self._logger = logger
        self._react = react
        self._tool_runner = tool_runner
        self._reviewer = reviewer

    async def process(
        self,
        context: EngineContext,
        state: _LoopState,
        event: StreamEvent,
        commit_step: Callable[[_LoopState, StreamEvent], Awaitable[None]],
        tokenizer: EstimatingTokenizer,
        refresh_step_instructions: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        match event:
            case ToolCallStarted():
                if state.mark_phase_started("tools"):
                    CompassTracer(context.tracer).loop_tools_started()

                if not state.in_the_middle_of_running_tools:
                    status_data = StatusEventData(
                        status="processing",
                        message="Evaluating tools",
                    )
                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=status_data,
                    )

                state.in_the_middle_of_running_tools = True
                return False
            case StepCompleted(result=result) if result.needs_tools:
                if state.mark_phase_started("tools"):
                    CompassTracer(context.tracer).loop_tools_started(len(result.tool_calls))

                CompassTracer(context.tracer).tool_calls_requested(result.tool_calls)

                review_result = await self.review_tool_calls(
                    context,
                    result.message.reasoning,
                    result.tool_calls,
                )

                todo = review_result.todo if review_result else None
                adjusted_reasoning = review_result.adjusted_reasoning if review_result else None

                if todo is not None:
                    context.state.todo = todo

                if adjusted_reasoning:
                    context.state.step_notes = adjusted_reasoning

                    # Was reasoning emitted by the model for this step?
                    if result.message.reasoning:
                        # We need to replace this step's reasoning in the state
                        assert len(context.state.reasoning_steps) > 0
                        context.state.reasoning_steps[-1] = adjusted_reasoning

                    # Restart the step with the adjusted reasoning in place.
                    raise _SemanticFailure()
                else:
                    context.state.step_notes = ""

                if todo is not None and refresh_step_instructions:
                    await refresh_step_instructions()

                # Approved tool calls must be committed before their tool results
                # are appended, preserving the provider-required assistant-tool order.
                await commit_step(state, event)

                status_data = StatusEventData(status="processing", message="Running tools")
                await context.session_event_emitter.emit_status_event(
                    trace_id=context.tracer.trace_id,
                    data=status_data,
                )

                await self.run_tool_calls(context, state, result.tool_calls, tokenizer)

                if state.mark_phase_finished("tools"):
                    CompassTracer(context.tracer).loop_tools_finished(len(result.tool_calls))

                return True
            case _:
                if state.mark_phase_finished("tools"):
                    CompassTracer(context.tracer).loop_tools_finished()

                state.in_the_middle_of_running_tools = False
                return False

    async def review_tool_calls(
        self,
        context: EngineContext,
        reasoning: str,
        tool_calls: Sequence[ToolCallPart],
    ) -> ToolCallReviewResult | None:
        effort = context.state.dynamic_effort_level

        if effort == Effort.MIN:
            # Skip the review for minimal-effort agents
            return None

        # Max-effort agents review every tool call. Below max effort, only review
        # when at least one consequential tool was called.
        if effort != Effort.MAX:
            if not self._any_consequential_tool_called(context, tool_calls):
                return None

        status_data = StatusEventData(status="processing", message="Reviewing tool use")
        await context.session_event_emitter.emit_status_event(
            trace_id=context.tracer.trace_id,
            data=status_data,
        )

        review_result = await self._reviewer().review_tool_calls(
            context,
            reasoning,
            tool_calls,
        )

        return review_result

    def _any_consequential_tool_called(
        self,
        context: EngineContext,
        tool_calls: Sequence[ToolCallPart],
    ) -> bool:
        # tool_call.name matches Tool.name (both are tool_ids_by_name keys), so the
        # offered tool catalog gives us each call's consequential flag without a
        # service round-trip.
        consequential_by_name = {
            tool.name: tool.consequential for tool in context.state.available_tools
        }
        return any(consequential_by_name.get(call.name, False) for call in tool_calls)

    async def run_tool_calls(
        self,
        context: EngineContext,
        state: _LoopState,
        tool_calls: Sequence[ToolCallPart],
        tokenizer: EstimatingTokenizer,
    ) -> None:
        with context.tracer.span("tools.batch", {"tool_count": len(tool_calls)}):
            await self._run_tool_calls(context, state, tool_calls, tokenizer)

    async def _run_tool_calls(
        self,
        context: EngineContext,
        state: _LoopState,
        tool_calls: Sequence[ToolCallPart],
        tokenizer: EstimatingTokenizer,
    ) -> None:
        # Run all of the step's tool calls concurrently; results keep call order.
        results: tuple[ToolResult | None] = await safe_gather(
            *(self.run_tool_call(context, tool_call) for tool_call in tool_calls)
        )

        calls_and_results = list(zip(tool_calls, results))

        step_parts: list[ToolResultPart] = []
        transient_call_ids: list[str] = []
        persisted_call_ids: list[str] = []

        for tool_call, result in calls_and_results:
            if result is not None:
                lifespan = result.control.get("lifespan", "auto")

                if lifespan == "session":
                    persisted_call_ids.append(tool_call.id)
                elif lifespan == "response":
                    transient_call_ids.append(tool_call.id)
                else:
                    if await tokenizer.estimate_token_count(json.dumps(result.data)) > 1_000:
                        if not DISABLE_WARNINGS:
                            self._logger.warning(
                                f"Tool result for {tool_call.name} exceeds 1,000 tokens; "
                                "defaulting to response lifespan."
                            )

                        transient_call_ids.append(tool_call.id)
                    else:
                        persisted_call_ids.append(tool_call.id)

                step_parts.append(
                    ToolResultPart(
                        call_id=tool_call.id,
                        name=tool_call.name,
                        content=result.data,
                        is_error="error_details" in result.metadata,
                    )
                )
            else:
                step_parts.append(
                    ToolResultPart(
                        call_id=tool_call.id,
                        name=tool_call.name,
                        content=f"Unknown tool: {tool_call.name}",
                        is_error=True,
                    )
                )

        def tool_event_data(call_ids: list[str]) -> ToolEventData:
            return ToolEventData(
                tool_calls=[
                    ToolCall(
                        tool_id=tool_call.name,
                        arguments=tool_call.args,
                        result={
                            "data": result.data,
                            "metadata": result.metadata,
                            "control": result.control,
                            "rules": result.rules,
                            "canned_responses": result.canned_responses,
                            "canned_response_fields": result.canned_response_fields,
                        },
                        rationale=state.reasoning_buffer.getvalue()
                        if state.reasoning_buffer
                        else "Not provided.",
                    )
                    for tool_call, result in calls_and_results
                    if result and tool_call.id in call_ids
                ]
            )

        # Emit transient results into the transient response context. Skip empty
        # steps: a tool event with no calls carries nothing.
        if transient_call_ids:
            transient_data = tool_event_data(transient_call_ids)
            CompassTracer(context.tracer).loop_tool_transient(transient_data)
            transient_tool_event = await context.response_event_emitter.emit_tool_event(
                trace_id=context.tracer.trace_id,
                data=transient_data,
            )
            context.state.tool_events.append(transient_tool_event)

        # Persist session-lifespan results together with the provider replay blob,
        # so a later turn can rebuild a faithful native tool turn from the stored
        # event. The original ToolCallParts carry the provider artifacts (e.g.
        # Gemini's thought_signature) the serializer needs. Skip entirely when
        # there are no persisted calls — otherwise we'd persist an empty tool event
        # whose blob model falls back to the generator's default identity model and
        # spuriously fails native replay (warning) on every later turn.
        if persisted_call_ids:
            persisted_calls = [
                tool_call
                for tool_call, result in calls_and_results
                if result and tool_call.id in persisted_call_ids
            ]
            persisted_results = [part for part in step_parts if part.call_id in persisted_call_ids]
            tool_message_serializer = SessionToolMessageSerializer()
            self._react().serialize_tool_messages(
                [
                    Message(role=Role.ASSISTANT, parts=persisted_calls),
                    Message(role=Role.TOOL, parts=persisted_results),
                ],
                tool_message_serializer,
            )
            persisted_data = tool_event_data(persisted_call_ids)
            CompassTracer(context.tracer).loop_tool_persistent(persisted_data)
            persisted_tool_event = await context.session_event_emitter.emit_tool_event(
                trace_id=context.tracer.trace_id,
                data=persisted_data,
                metadata={_PROVIDER_DATA_KEY: tool_message_serializer.provider_data},
            )
            context.state.tool_events.append(persisted_tool_event)

        # Finally, append all the react step parts
        state.history.append(
            Message(
                role=Role.TOOL,
                cache_key=context.session.id,
                parts=list(step_parts),
            )
        )

    async def run_tool_call(
        self, context: EngineContext, tool_call: ToolCallPart
    ) -> ToolResult | None:
        tool_id = context.state.tool_ids_by_name.get(tool_call.name)

        if tool_id is None:
            message = f"Model requested an unknown tool: {tool_call.name}"
            self._logger.warning(message)
            CompassTracer(context.tracer).tool_call_error(tool_call, "UnknownTool", message)
            return None

        return await self._tool_runner().run_tool(context, tool_id, tool_call.args)


class BaseLoop(Loop):
    """Shared agentic generation loop. Streaming vs blocking output differs only in
    how the assistant message is surfaced to the session, so that single step —
    :meth:`_surface_message_event` — is left abstract for concrete loops to implement;
    everything else (history building, reasoning/tool handling, the react loop) is
    output-mode-agnostic and lives here."""

    # Retry a step on a transient ReactError, but only before any event has been
    # emitted (a stream can't be replayed mid-flight). Waits between attempts.
    _RETRY_TIMEOUT_PER_ATTTEMPT = (2.0, 8.0, 32.0)
    _SENTENCE_END_PATTERN = re.compile(r"(?<=[.!?])(?:\s+|$)")

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        optimization_policy: OptimizationPolicy,
        react: ReactGenerator,
        tokenizer: EstimatingTokenizer,
        tool_runner: ToolRunner,
        reviewer: Reviewer,
        hooks: EngineHooks,
        usage_reporter: UsageReporter,
        cost_control_policy: CostControlPolicy,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._optimization_policy = optimization_policy
        self._react = react
        self._tokenizer = tokenizer
        self._tool_runner = tool_runner
        self._reviewer = reviewer
        self._hooks = hooks
        self._cost_control_policy = cost_control_policy
        self._usage_reporter = usage_reporter
        self._history_builder = _InteractionHistoryBuilder(logger, lambda: self._react)
        self._turn_instruction_builder = _TurnInstructionBuilder(logger)
        self._reasoning_event_processor = _ReasoningEventProcessor()
        self._tool_step_controller = _ToolStepController(
            logger,
            lambda: self._react,
            lambda: self._tool_runner,
            lambda: self._reviewer,
        )

    async def warm_up(self, job: LoopJob) -> Usage:
        self._logger.debug(f"Prefilling job for session {job.context.session.id}")

        # Warm the cache for the stable prefix only — the system instructions and
        # the conversation so far. The per-turn instructions are dynamic and sit
        # past the cache breakpoint, so we leave them out here.
        history, _ = await self._history_builder.build(job, include_turn_instructions=False)

        usage = await self._react.prefill(
            history=history,
            tools=await self._get_tools(job.context),
            tool_choice="auto",
            reasoning=job.reasoning_config,
            hints={"model_size": job.model_size},
        )

        if usage.input_tokens > 0:
            self._logger.debug(f"{self.__class__.__name__} prefill usage:\n {usage}")

        return usage

    async def run(self, job: LoopJob) -> LoopResult:
        with self._tracer.span(
            "loop.run",
            {
                "session_id": job.context.session.id,
                "agent_id": job.context.agent.id,
                "loop": self.__class__.__name__,
            },
        ):
            return await self._run(job)

    async def _run(self, job: LoopJob) -> LoopResult:
        # Give hooks (e.g. retrievers) a chance to stage tool events before we
        # build the history — history building folds context.state.tool_events in.
        if not await self._hooks.call_on_generating_messages(job.context):
            # A hook requested that we not proceed with generating a response.
            return LoopResult(job=job, steps=[])

        # Don't seed the turn instructions here: the first _run_step's
        # _update_step_instructions builds and places them (and on high/max effort,
        # which don't cache step_instructions, seeding here would build them twice).
        # _last_user_message_index (the seed's placement) is identical to refresh's
        # insertion index, so placement is unchanged.
        history, instructions_index = await self._history_builder.build(
            job, include_turn_instructions=False
        )
        state = _LoopState(
            history=history,
            instructions_index=instructions_index,
            preamble=_ToolPreambleState(job.preamble_config),
        )

        while not job.context.state.prepared_to_respond:
            # STEP cost-control choke point: a deny stops iterating after the
            # last completed step (never truncating in-flight output) and falls
            # through to the loop's normal terminal path below.
            if not await self._gate_step(job.context):
                break

            try:
                max_semantic_failures = self._max_semantic_failures(job)
                semantic_failure_count = 0

                while semantic_failure_count < max_semantic_failures:
                    try:
                        await self._run_step_protected(job, state)
                    except _SemanticFailure:
                        semantic_failure_count += 1
                        await self._reset_message_after_restart(job.context, state)
                        continue
                    except Exception as exc:
                        if not isinstance(exc.__cause__, _SemanticFailure):
                            raise

                        semantic_failure_count += 1
                        await self._reset_message_after_restart(job.context, state)
                        continue
                    else:
                        break

                if semantic_failure_count == max_semantic_failures:
                    raise _GiveUp(
                        "The agent repeatedly proposed tool calls that were rejected by "
                        "policy review."
                    )

                job.context.state.iterations.append(
                    IterationState(
                        matched_rules=[],
                        ruled_out=[],
                        resolved_rules=[],
                        tool_insights=ToolInsights(evaluations={}, missing_data={}),
                        executed_tools=[],
                    )
                )

                if len(job.context.state.iterations) >= 30:
                    self._logger.warning(
                        f"Large number of engine iterations on session {job.context.session.id} ({job.context.session.title or 'Untitled'})"
                    )

                if len(job.context.state.iterations) == job.context.agent.max_engine_iterations:
                    raise _GiveUp(
                        "The agent reached the maximum number of engine iterations "
                        "without preparing a response."
                    )
            except asyncio.CancelledError:
                self._logger.warning(
                    f"{self.__class__.__name__} run cancelled on session {job.context.session.id}"
                )
                raise
            except _GiveUp as exc:
                await self._give_up(job, state, str(exc))

        status_data = StatusEventData(status="ready", data={"stage": "completed"})
        await job.context.session_event_emitter.emit_status_event(
            trace_id=job.context.tracer.trace_id,
            data=status_data,
        )

        await self._hooks.call_on_messages_emitted(job.context)

        return LoopResult(job=job, steps=state.steps)

    async def _gate_step(self, context: EngineContext) -> bool:
        """The STEP cost-control choke point.

        Consulted at step boundaries only, so a denial never truncates in-flight
        streamed text — it stops the loop before the next chargeable step. Fails
        open: a policy error is logged and the step proceeds."""
        try:
            verdict = await self._cost_control_policy.check(
                CostContext(
                    agent_id=context.agent.id,
                    session_id=context.session.id,
                    customer_id=context.customer.id,
                    trace_id=context.tracer.trace_id,
                ),
                WorkKind.STEP,
            )
        except Exception as exc:
            self._logger.warning(f"Cost-control check failed (failing open): {exc}")
            return True

        for warning in verdict.warnings:
            self._logger.warning(f"Cost-control warning: {warning}")

        if not verdict.allowed:
            self._logger.warning(
                f"Response loop stopped by cost-control policy on session "
                f"{context.session.id} (reason: {verdict.reason or 'unspecified'})"
            )

        return verdict.allowed

    def _max_semantic_failures(self, job: LoopJob) -> int:
        """The number of times a step can be restarted due to a reviewer-provided
        policy-adjusted reasoning before we give up and propagate the failure."""
        match job.context.state.dynamic_effort_level:
            case Effort.MIN:
                return 1
            case Effort.LOW:
                return 2
            case Effort.MEDIUM:
                return 3
            case Effort.HIGH:
                return 5
            case Effort.MAX:
                return 10

    async def _run_step_protected(self, job: LoopJob, state: _LoopState) -> None:
        async def defer_cancellation(latch: CancellationSuppressionLatch[None]) -> None:
            latch.enable()
            await self._run_step_internal(job, state)
            latch.disable()  # Raise any internally-caught exception

        return await latched_shield(defer_cancellation)

    async def _run_step_internal(self, job: LoopJob, state: _LoopState) -> None:
        """Run one react step, processing each event. A transient ReactError is
        retried — but only while NO event has been produced yet: once events have
        been emitted, the stream can't be replayed (it would re-emit chunks and
        re-run side effects), so the error propagates. The transient errors we
        retry are raised when the stream is opened, before any event."""

        with self._tracer.span(
            "loop.step",
            {
                "step_index": len(state.steps) + 1,
                "disable_tools": state.disable_tools,
            },
        ):
            try:
                self._update_system_instructions(job, state)
                await self._update_step_instructions(job, state)

                for attempt in range(len(self._RETRY_TIMEOUT_PER_ATTTEMPT) + 1):
                    produced = False
                    try:
                        async for event in self._react.stream_step(
                            history=state.history,
                            tools=[] if state.disable_tools else await self._get_tools(job.context),
                            tool_choice="none" if state.disable_tools else "auto",
                            reasoning=job.reasoning_config,
                            hints={"model_size": job.model_size, "hedge_timeout": 10.0},
                        ):
                            produced = True
                            await self._process_reasoning_event(job.context, state, event)
                            # Surface/close the message BEFORE the tool status: a text->tool
                            # transition finalizes the preamble as its own bubble first, so the
                            # tool status follows the message (no flicker), and post-tool text
                            # starts a fresh message instead of gluing onto the preamble.
                            await self._surface_message_event(job.context, state, event)
                            committed = await self._process_tool_event(job, state, event)
                            if not committed:
                                await self._commit_react_event(state, event)

                        return
                    except ReactError as exc:
                        if (
                            not exc.retryable
                            or produced
                            or attempt == len(self._RETRY_TIMEOUT_PER_ATTTEMPT)
                        ):
                            raise
                        wait = self._RETRY_TIMEOUT_PER_ATTTEMPT[attempt]
                        self._logger.warning(
                            f"{self.__class__.__name__} retrying step after a transient error "
                            f"({exc}); retrying in {wait}s (attempt {attempt + 2})."
                        )
                        await asyncio.sleep(wait)
            finally:
                _finish_open_loop_phases(job.context, state)

    async def _give_up(self, job: LoopJob, state: _LoopState, reason: str) -> None:
        CompassTracer(job.context.tracer).loop_give_up(
            reason,
            len(job.context.state.iterations),
        )
        self._logger.warning(
            f"{reason} Forcing a final message on session {job.context.session.id} "
            f"({job.context.session.title or 'Untitled'}). Reasoning: \n"
            f"{json.dumps(job.context.state.reasoning_steps, indent=2)}"
        )

        await self._reset_message_after_restart(job.context, state)
        self._drop_trailing_empty_assistant_messages(state)

        state.disable_tools = True
        state.force_message_note = (
            "You've failed to address the current request after repeated attempts. "
            "You must now explain to the user why you were not able to help currently. "
            "Do not claim that the request was completed. Tool use is disabled for this "
            "step, so you must send a concise message to the user now."
        )

        # Strip the task-pushing context so the forced step isn't fighting itself: the
        # reviewer's TODO and adjusted reasoning tell the model to keep working, and the
        # step prompt still renders the tool catalog and matched rules. Left in
        # place, the model keeps attempting the task (re-emitting tool calls the reviewer
        # then rejects, or stalling on reasoning) and never produces the final message.
        job.context.state.giving_up = True
        job.context.state.todo = ""
        job.context.state.step_notes = ""

        try:
            await self._run_step_protected(job, state)
        except _SemanticFailure:
            self._logger.error(
                f"{self.__class__.__name__} could not force a final message because the "
                "model still attempted tool use with tools disabled."
            )
        except Exception as exc:
            if not isinstance(exc.__cause__, _SemanticFailure):
                raise

            self._logger.error(
                f"{self.__class__.__name__} could not force a final message because the "
                "model still attempted tool use with tools disabled."
            )
        finally:
            state.disable_tools = False
            state.force_message_note = ""
            job.context.state.giving_up = False

        job.context.state.prepared_to_respond = True

    def _drop_trailing_empty_assistant_messages(self, state: _LoopState) -> None:
        while state.history:
            last_message = state.history[-1]
            if (
                last_message.role == Role.ASSISTANT
                and not last_message.text
                and not last_message.tool_calls
            ):
                state.history.pop()
                continue

            break

    async def _get_tools(self, context: EngineContext) -> list[ToolSpec]:
        return [*tool_specs_from_tools(context.state.available_tools)]

    def _tool_preamble_decision(self, state: _LoopState) -> PreambleDecision:
        return state.preamble.decide(time.monotonic())

    def _tool_preamble_is_allowed(self, state: _LoopState) -> bool:
        return state.preamble.configuration.allows_emission(self._tool_preamble_decision(state))

    def _should_trim_tool_preamble_text(self, state: _LoopState) -> bool:
        return state.preamble.configuration.trims_preamble_text()

    def _mark_user_visible_message_emitted(self, state: _LoopState) -> None:
        state.preamble.mark_user_visible_message_emitted(time.monotonic())

    def _tool_preamble_text(self, text: str) -> str:
        match = self._SENTENCE_END_PATTERN.search(text)
        if match is None:
            return text

        return text[: match.end()].strip()

    def _message_with_text_prefix(self, message: Message, prefix_len: int) -> Message:
        remaining = prefix_len
        parts: list[Part] = []

        for part in message.parts:
            if isinstance(part, TextPart):
                if remaining <= 0:
                    continue

                text = part.text[:remaining]
                remaining -= len(text)
                if text:
                    parts.append(TextPart(text=text, provider_data=part.provider_data))
            else:
                parts.append(part)

        return Message(
            role=message.role,
            parts=parts,
            provider_data=message.provider_data,
            cache_key=message.cache_key,
        )

    def _apply_message_commit_policy(
        self,
        message: Message,
        policy: _MessageCommitPolicy,
    ) -> Message:
        match policy.mode:
            case _MessageCommitMode.KEEP_FULL_MESSAGE:
                return message
            case _MessageCommitMode.KEEP_NO_TEXT:
                return self._message_with_text_prefix(message, 0)
            case _MessageCommitMode.KEEP_TEXT_PREFIX:
                return self._message_with_text_prefix(message, policy.prefix_len)

    async def _commit_react_event(self, state: _LoopState, event: StreamEvent) -> None:
        if isinstance(event, StepCompleted):
            message = event.result.message
            if event.result.needs_tools:
                message = self._apply_message_commit_policy(
                    message,
                    state.message.commit_policy,
                )

            state.history.append(message)
            state.steps.append(event.result)
            state.message.reset_commit_policy()

            if event.result.message.reasoning:
                self._logger.trace(
                    f"{self.__class__.__name__} step reasoning:\n {event.result.message.reasoning}"
                )

            self._logger.debug(f"{self.__class__.__name__} step usage:\n {event.result.usage}")
            self._usage_reporter.report_usage(
                event.result.usage.model_name,
                UsageInfo(
                    input_tokens=event.result.usage.input_tokens,
                    output_tokens=event.result.usage.output_tokens,
                    cached_input_tokens=event.result.usage.cached_input_tokens,
                    extra={"reasoning_tokens": event.result.usage.reasoning_tokens},
                ),
            )

    async def _process_reasoning_event(
        self,
        context: EngineContext,
        state: _LoopState,
        event: StreamEvent,
    ) -> None:
        await self._reasoning_event_processor.process(context, state, event)

    async def _process_tool_event(
        self,
        job: LoopJob,
        state: _LoopState,
        event: StreamEvent,
    ) -> bool:
        return await self._tool_step_controller.process(
            job.context,
            state,
            event,
            self._commit_react_event,
            self._tokenizer,
            refresh_step_instructions=lambda: self._update_step_instructions(job, state),
        )

    async def _review_tool_calls(
        self,
        context: EngineContext,
        reasoning: str,
        tool_calls: Sequence[ToolCallPart],
    ) -> ToolCallReviewResult | None:
        return await self._tool_step_controller.review_tool_calls(
            context,
            reasoning,
            tool_calls,
        )

    async def _run_tool_calls(
        self,
        context: EngineContext,
        state: _LoopState,
        tool_calls: Sequence[ToolCallPart],
    ) -> None:
        await self._tool_step_controller.run_tool_calls(context, state, tool_calls, self._tokenizer)

    async def _run_tool_call(
        self, context: EngineContext, tool_call: ToolCallPart
    ) -> ToolResult | None:
        return await self._tool_step_controller.run_tool_call(context, tool_call)

    def _record_message_event(
        self,
        context: EngineContext,
        state: _LoopState,
        data: MessageEventData,
        mode: str,
    ) -> None:
        CompassTracer(context.tracer).loop_message(data, mode)

    @abstractmethod
    async def _surface_message_event(
        self,
        context: EngineContext,
        state: _LoopState,
        event: StreamEvent,
    ) -> None:
        """Surface the assistant's message for the current stream event.

        This is the sole output-mode-specific step: a streaming loop emits the
        message incrementally (chunked) as deltas arrive, while a blocking loop
        emits it once, complete, on step completion. Concrete loops should call
        :meth:`_complete_message_step` when a step that produced a message
        completes, to share the loop's termination + hook logic.
        """
        ...

    async def _complete_message_step(self, context: EngineContext, result: StepResult) -> None:
        """Shared step-completion tail for a step that produced a message: a message
        with no pending tool calls ends the loop, and the message-generated hook
        fires regardless of output mode."""
        if not result.needs_tools:
            context.state.prepared_to_respond = True

        await self._hooks.call_on_message_generated(context, result.message.text)

    async def _reset_message_after_restart(self, context: EngineContext, state: _LoopState) -> None:
        """A reviewer-rejected step is being restarted. Finalize the message already
        streamed during the rejected attempt as its own (complete) bubble, then reset
        the streaming state so the retry begins a fresh message instead of appending to
        it. A blocking loop doesn't stream a partial message (its handle is None here),
        so this only clears the per-attempt flags there."""
        if state.message.handle is not None:
            await state.message.handle.update(
                MessageEventData(
                    message=state.message.buffer.getvalue() if state.message.buffer else "",
                    participant=Participant(id=context.agent.id, display_name=context.agent.name),
                    chunks=[*state.message.chunks, None],
                )
            )

        state.message.reset_step_output()
        state.in_the_middle_of_running_tools = False

    def _get_model_size(self, context: EngineContext, state: _LoopState) -> ModelSize:
        return ModelSize.MEDIUM

    async def _build_history(
        self,
        job: LoopJob,
        *,
        include_turn_instructions: bool = True,
    ) -> tuple[list[Message], int | None]:
        return await self._history_builder.build(
            job,
            include_turn_instructions=include_turn_instructions,
        )

    def _instructions_message(self, turn_instructions: str, cache_key: str) -> Message:
        return _instructions_message(turn_instructions, cache_key)

    async def _update_step_instructions(
        self,
        job: LoopJob,
        state: _LoopState,
    ) -> None:
        await self._turn_instruction_builder.refresh(
            job,
            state,
            self._tool_preamble_decision(state),
            self.__class__.__name__,
        )

    def _update_system_instructions(
        self,
        job: LoopJob,
        state: _LoopState,
    ) -> None:
        refreshed_instructions = _system_instructions_message(
            job.system_instructions,
            job.context.session.id,
        )

        if not state.history:
            state.history.append(refreshed_instructions)
            state.system_instructions_index = 0
            self._logger.trace(
                f"{self.__class__.__name__} inserted system instructions:\n"
                f"{refreshed_instructions.text}"
            )
            state.logged_system_instructions = refreshed_instructions.text
            return

        current_instructions = state.history[state.system_instructions_index]
        if current_instructions.text != refreshed_instructions.text:
            state.history[state.system_instructions_index] = refreshed_instructions

        if state.logged_system_instructions is None:
            self._logger.trace(
                f"{self.__class__.__name__} inserted system instructions:\n"
                f"{refreshed_instructions.text}"
            )
        elif state.logged_system_instructions != refreshed_instructions.text:
            self._logger.trace(
                f"{self.__class__.__name__} updated system instructions:\n"
                f"{refreshed_instructions.text}"
            )

        state.logged_system_instructions = refreshed_instructions.text

    def _render_semantic_history(self, history: Sequence[Message]) -> str:
        if not history:
            return "[No messages in loop history.]"

        rendered_messages: list[str] = []

        for index, message in enumerate(history, start=1):
            lines = [f"## Event {index}: {message.role.value}"]

            if text := message.text.strip():
                lines.append(f"Text: {text}")

            if reasoning := message.reasoning.strip():
                lines.append(f"Reasoning: {reasoning}")

            for tool_call in message.tool_calls:
                lines.append(
                    f"Tool call: {tool_call.name}({self._render_json_for_log(tool_call.args)})"
                )

            for tool_result in message.tool_results:
                result_prefix = "Tool error" if tool_result.is_error else "Tool result"
                lines.append(
                    f"{result_prefix}: {tool_result.name}"
                    f"[call_id={tool_result.call_id}] -> "
                    f"{self._render_json_for_log(tool_result.content)}"
                )

            if len(lines) == 1:
                lines.append("[No semantic content.]")

            rendered_messages.append("\n".join(lines))

        return "\n\n".join(rendered_messages)

    def _render_json_for_log(self, value: Any) -> str:
        try:
            return json.dumps(value, indent=2, default=str)
        except (TypeError, ValueError):
            return repr(value)

    def _build_tool_event_messages(
        self,
        data: ToolEventData,
        metadata: Optional[Mapping[str, JSONSerializable]],
        cache_key: str,
    ) -> list[Message]:
        return self._history_builder.build_tool_event_messages(data, metadata, cache_key)

    def _build_result_only_tool_event_messages(
        self, data: ToolEventData, cache_key: str
    ) -> list[Message]:
        return self._history_builder.build_result_only_tool_event_messages(data, cache_key)
