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
from typing import Any, Optional, cast

from parlant.core.agents import AgentId, Effort
from parlant.core.async_utils import CancellationSuppressionLatch, latched_shield, safe_gather
from parlant.core.common import DISABLE_WARNINGS, JSONSerializable, generate_id, to_prompt_json
from parlant.core.emissions import MessageEventHandle, StatusEventHandle
from parlant.core.engines.compass.preambles import PreambleConfiguration, PreambleDecision
from parlant.core.engines.compass.context import EngineContext, IterationState
from parlant.core.engines.compass.loop.loop import Loop, LoopJob, LoopResult
from parlant.core.engines.compass.tool_runner import ToolRunner
from parlant.core.engines.compass.tracing import CompassTracer
from parlant.core.loggers import Logger
from parlant.core import metadata as metadata_utils
from parlant.core.meter import Meter
from parlant.core.health import (
    ENGINE_TTFM_KIND,
    EngineHealthView,
    HealthReporter,
    NLP_REQUEST_KIND,
    NLP_REQUESTS_COUNTER,
    NLP_TOKENS_COUNTER,
    NLPHealthView,
)
from parlant.core.nlp.model_routing import ModelRouter
from parlant.core.agents import Moderation
from parlant.core.nlp.moderation import AgentModerationContext
from parlant.core.nlp.react import (
    Message,
    Part,
    ReactError,
    ReactGenerator,
    ReactGeneratorHints,
    ReasoningConfig,
    ReasoningDelta,
    Role,
    StepCompleted,
    StepResult,
    StreamEvent,
    TextDelta,
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
    AttachmentRef,
    censored_message_notice,
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
from parlant.core.pricing import PricingConfigurationError
from parlant.core.usage_reporter import UsageReporter
from parlant.core.engines.compass.hooks import (
    GiveUpKind,
    CompassHookResult,
    GaveUp,
    MessageReviewed,
    ReviewingMessage,
    ReviewingToolCalls,
    ToolCallOutcome,
    ToolCallsReviewed,
)
from parlant.core.engines.compass.reviewer import Reviewer, ToolCallReviewResult


# Key under which the provider's opaque tool-call replay blob is stored in a tool
# event's metadata. Read back at history-build time so the originating provider
# can faithfully replay the call (ids, signatures, …) on a later turn.
_PROVIDER_DATA_KEY = "provider_data"


def _render_attachment_note(attachment: AttachmentRef) -> str:
    """A short, model-visible stand-in for a file the customer attached.

    Uses the caption when present, then the filename. The URL is deliberately
    excluded from model context; an unlabeled attachment gets a neutral placeholder."""
    media_type = attachment.get("media_type") or ""
    kind = "image" if media_type.startswith("image/") else "file"
    label = attachment.get("caption") or attachment.get("filename")
    return f"[{kind}: {label}]" if label else f"[{kind} attached]"


def _customer_message_text(message_data: MessageEventData) -> str:
    """The customer message text the model sees, with a note per attachment appended.

    A moderation-flagged message renders as its censor notice: the flagged text
    must never reach the model, but the fact (and reasons) of the censorship
    must — rules can condition on it, and the agent may need to explain it."""
    if (notice := censored_message_notice(message_data)) is not None:
        return notice

    text = message_data["message"]

    attachments = message_data.get("attachments") or []
    if not attachments:
        return text

    notes = "\n".join(_render_attachment_note(a) for a in attachments)
    return f"{text}\n{notes}" if text else notes


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


def _bare_tool_name(tool_id: str) -> str:
    """The model-facing tool name for a stored tool-event id.

    Tool events persist the qualified "service:tool" id; the react layer
    replays history by the bare name the model actually called. Bare names in
    older events pass through unchanged."""
    return tool_id.split(":", 1)[1] if ":" in tool_id else tool_id


class SessionToolMessageDeserializer(ToolMessageDeserializer):
    """Session-backed source. Serves calls/results from the neutral ``ToolEventData``
    and the provider blob from the event's metadata."""

    def __init__(self, data: ToolEventData, provider_data: Mapping[str, Any]) -> None:
        self._data = data
        self._provider_data = provider_data

    def read_calls(self) -> Sequence[ToolCallPart]:
        return [
            ToolCallPart(name=_bare_tool_name(call["tool_id"]), args=call["arguments"])
            for call in self._data["tool_calls"]
        ]

    def read_results(self) -> Sequence[ToolResultPart]:
        results: list[ToolResultPart] = []
        for call in self._data["tool_calls"]:
            result = call.get("result", {}) or {}
            results.append(
                ToolResultPart(
                    name=_bare_tool_name(call["tool_id"]),
                    content=result.get("data", {}),
                    is_error=metadata_utils.internal_value(
                        result.get("metadata", {}) or {},
                        ("error_details",),
                    )
                    is not None,
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
    # Opaque per-bubble id joining the persisted message event to its
    # authoritative loop.message telemetry event (Dashboard correlation).
    output_ref: str | None = None

    def ensure_output_ref(self) -> str:
        if self.output_ref is None:
            self.output_ref = str(generate_id())

        return self.output_ref

    def output_metadata(
        self,
        existing: Mapping[str, JSONSerializable] | None = None,
    ) -> dict[str, JSONSerializable]:
        base = dict(existing or {})
        namespace = base.get("__dashboard__")
        dashboard_metadata = dict(cast(Mapping[str, JSONSerializable], namespace or {}))
        dashboard_metadata["output_ref"] = self.ensure_output_ref()
        return {**base, "__dashboard__": dashboard_metadata}

    def consume_output_ref(self) -> str:
        output_ref = self.ensure_output_ref()
        self.output_ref = None
        return output_ref

    def clear_transient_output(self) -> None:
        self.handle = None
        self.buffer = None
        self.chunks = []
        self.output_ref = None

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
    # True once the current step's cancellation shield has absorbed a cancel
    # request (a newer customer message superseded this turn). Emission points
    # poll it to abandon before anything user-visible goes out.
    cancellation_pending: Callable[[], bool] = lambda: False

    reasoning_handle: StatusEventHandle | None = None
    reasoning_buffer: StringIO | None = None
    reasoning_chunks: list[str | None] = field(default_factory=list)

    message: _MessageEmissionState = field(default_factory=_MessageEmissionState)
    preamble: _ToolPreambleState = field(default_factory=_ToolPreambleState)

    disable_tools: bool = False
    force_message_note: str = ""

    steps: list[StepResult] = field(default_factory=list)


class _SemanticFailure(Exception):
    pass


class _GiveUp(Exception):
    """Ends the turn's attempts. ``kind`` travels to the on_giving_up hook so a
    handler can respond differently depending on why the loop stopped."""

    def __init__(self, reason: str, kind: GiveUpKind) -> None:
        super().__init__(reason)
        self.kind = kind


def _instructions_message(turn_instructions: str, cache_key: str) -> Message:
    return Message(
        role=Role.SYSTEM,
        cache_key=cache_key,
        parts=[
            TextPart(
                text=f"""\
The following is notes and context about the current state of the conversation — the rules, glossary, and tools relevant to it. Treat it as background that informs your next reply; it is NOT itself a message addressed to you, so never respond to it, acknowledge it, or refer to it. (One exception: approved style reference examples below, if any, are meant to be reused in your reply — reusing their text is not "referring to" these notes.):

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
    def __init__(
        self,
        logger: Logger,
        assistant_text: Callable[[MessageEventData, Mapping[str, JSONSerializable] | None], str],
    ) -> None:
        self._logger = logger
        self._assistant_text = assistant_text

    async def build(
        self,
        job: LoopJob,
        *,
        react: ReactGenerator,
        include_turn_instructions: bool = True,
    ) -> tuple[list[Message], int | None]:
        cache_key = job.context.session.id

        # The model this turn will run on — used to decide whether a persisted
        # tool event can be natively replayed (some providers' replay is bound to
        # the producing model).
        replay_model = react.resolve_model({})

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
                message_data = cast(MessageEventData, event.data)
                history.append(
                    Message(
                        role=Role.USER,
                        cache_key=cache_key,
                        parts=[TextPart(text=_customer_message_text(message_data))],
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
                # The text to render is decided by the loop: the base loop uses
                # the full generated text; the voice loop narrows it to what was
                # actually delivered. Empty text (e.g. a fully-interrupted voice
                # message that was never heard) drops out of history — as far as
                # the conversation is concerned, it never happened.
                text = self._assistant_text(cast(MessageEventData, event.data), event.metadata)

                if not text.strip():
                    continue

                history.append(
                    Message(
                        role=Role.ASSISTANT,
                        cache_key=cache_key,
                        parts=[TextPart(text=text)],
                    )
                )
            elif event.kind == EventKind.MESSAGE and event.source == EventSource.HUMAN_AGENT:
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
                # Reconstruct every tool event regardless of source. Persisted
                # tool events can carry source=AI_AGENT, so gating on SYSTEM
                # silently dropped prior-turn tool results.
                history.extend(
                    self.build_tool_event_messages(
                        cast(ToolEventData, event.data),
                        event.metadata,
                        cache_key,
                        react=react,
                        model=replay_model,
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
                    react=react,
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
        react: ReactGenerator | None = None,
        model: Optional[str] = None,
    ) -> list[Message]:
        provider_data = metadata_utils.internal_value(metadata, (_PROVIDER_DATA_KEY,), {})
        if isinstance(provider_data, Mapping) and provider_data and react is not None:
            # A model-issued tool call carrying its provider's replay blob: let the
            # originating provider rebuild a faithful, native tool_use/tool_result
            # pair (consistent ids, signatures, …). ``model`` is the model THIS turn
            # will run on, so a provider can gate model-bound native replay on it.
            messages = react.deserialize_tool_messages(
                SessionToolMessageDeserializer(data, provider_data), model=model
            )
            if messages is None:
                # The current generator can't natively replay this blob (e.g. a
                # tool call whose stored native replay artifact is unusable).
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
            # Persisted events are integrator-writable JSON: arguments is
            # normally a mapping, but render any other shape as-is rather than
            # failing the whole turn on a malformed historical event.
            arguments = call["arguments"]
            args_text = (
                ", ".join(f"{k}={v}" for k, v in arguments.items())
                if isinstance(arguments, dict)
                else str(arguments)
            )
            return (
                f"{call['tool_id']}({args_text}) returned: {call.get('result', {}).get('data', {})}"
            )

        messages: list[Message] = []

        call_id = 0
        for call in data["tool_calls"]:
            call_id += 1
            is_error = (
                metadata_utils.internal_value(
                    call.get("result", {}).get("metadata", {}),
                    ("error_details",),
                )
                is not None
            )

            messages.append(
                Message(
                    role=Role.TOOL,
                    cache_key=cache_key,
                    parts=[
                        ToolResultPart(
                            call_id=str(call_id),
                            name=_bare_tool_name(call["tool_id"]),
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

                # A changed mid-turn update carries fresh directives (tool-
                # returned rules, suppression notes) reacting to the tool turns
                # that were appended AFTER the original insertion point — so an
                # in-place rewrite would leave them buried behind the very
                # events they respond to. Relocate the instructions to the end
                # of the history, where the model reads them freshest.
                state.history.pop(state.instructions_index)
                state.history.append(refreshed_instructions)
                state.instructions_index = len(state.history) - 1
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
            # If a model turn follows that user message — e.g. utter(), where
            # the history ends with the agent's own previous reply — inserting
            # before the user message would leave the request ENDING with a
            # model turn, which some providers reject outright (Gemini:
            # "Requests ending with a model turn are not supported"). Append
            # at the end instead; the instructions concern the NEXT reply, so
            # last is also the semantically right place.
            if any(message.role == Role.ASSISTANT for message in state.history[index:]):
                index = len(state.history)

            state.history.insert(index, refreshed_instructions)
            state.instructions_index = index

            self._log_instruction_change(loop_name, "inserted", refreshed_instructions.text)

    def _log_instruction_change(
        self,
        loop_name: str,
        action: str,
        instructions: str,
    ) -> None:
        self._logger.trace(f"{action.capitalize()} turn instructions:\n\n{instructions}")

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


class _StepPhaseTracer:
    """Marks when a step's reasoning and message phases start and finish, so
    traces carry per-phase durations.

    Reasoning runs from the model call's start until its first text or tool
    call: providers such as Gemini send their thought summary only once they're
    done thinking, so timing from the first reasoning chunk would record the
    thinking as almost instant, and a step with no chunks would record none.
    A step has at most one message phase. Phases are closed when the next kind
    of output begins, when the step completes, or with the step's failure, so
    none is left open."""

    def __init__(self, tracer: CompassTracer, message_mode: str) -> None:
        self._tracer = tracer
        self._message_mode = message_mode
        self._reasoning_chunks: int | None = None
        self._message_open = False
        self._message_traced = False

    def begin(self) -> None:
        """The model call is starting: it's thinking until it writes or calls a tool."""
        self._reasoning_chunks = 0
        self._tracer.loop_reasoning_started()

    def observe(self, event: StreamEvent) -> None:
        if isinstance(event, ReasoningDelta):
            if self._reasoning_chunks is not None:
                self._reasoning_chunks += 1
            return

        self._finish_reasoning("completed")

        if isinstance(event, TextDelta) or (
            isinstance(event, StepCompleted) and event.result.message.text
        ):
            # Text that arrives only with the completed step is a phase with no gap.
            self._start_message()

        if isinstance(event, (ToolCallStarted, StepCompleted)):
            self._finish_message("completed")

    def close(self, outcome: str) -> None:
        self._finish_reasoning(outcome)
        self._finish_message(outcome)

    def _start_message(self) -> None:
        if not self._message_traced:
            self._message_traced = True
            self._message_open = True
            self._tracer.loop_message_started(self._message_mode)

    def _finish_reasoning(self, outcome: str) -> None:
        if self._reasoning_chunks is not None:
            self._tracer.loop_reasoning_finished(self._reasoning_chunks, outcome=outcome)
            self._reasoning_chunks = None

    def _finish_message(self, outcome: str) -> None:
        if self._message_open:
            self._tracer.loop_message_finished(self._message_mode, outcome=outcome)
            self._message_open = False


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
                    state.reasoning_buffer = StringIO()
                    state.reasoning_buffer.write(text)
                    state.reasoning_chunks = [text]

                    state.reasoning_handle = await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(
                            status="processing",
                            message=state.reasoning_buffer.getvalue(),
                            chunks=state.reasoning_chunks,
                        ),
                        metadata=metadata_utils.set_internal_value({}, ("reasoning",), True),
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
                    context.state.reasoning_steps.append(result.message.reasoning)

                if state.reasoning_handle is not None:
                    await state.reasoning_handle.update(
                        StatusEventData(
                            status="processing",
                            message=result.message.reasoning,
                            chunks=[*state.reasoning_chunks, None],
                        )
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

                await state.reasoning_handle.update(
                    StatusEventData(
                        status="processing",
                        message=state.reasoning_buffer.getvalue(),
                        chunks=[*state.reasoning_chunks, None],
                    )
                )

                state.reasoning_buffer = None
                state.reasoning_chunks = []
                state.reasoning_handle = None


def _any_consequential_tool_called(
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


class _ToolStepController:
    def __init__(
        self,
        logger: Logger,
        tool_runner: Callable[[], ToolRunner],
        reviewer: Callable[[], Reviewer],
        before_tool_calls: Callable[[EngineContext, Sequence[ToolCallPart]], Awaitable[None]],
    ) -> None:
        self._logger = logger
        self._tool_runner = tool_runner
        self._reviewer = reviewer
        self._before_tool_calls = before_tool_calls

    async def process(
        self,
        context: EngineContext,
        state: _LoopState,
        event: StreamEvent,
        commit_step: Callable[[_LoopState, StreamEvent], Awaitable[None]],
        react: ReactGenerator,
        tokenizer: EstimatingTokenizer,
        refresh_step_instructions: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        match event:
            case ToolCallStarted():
                if not state.in_the_middle_of_running_tools:
                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(status="processing", message="Evaluating tools"),
                    )

                state.in_the_middle_of_running_tools = True
                return False
            case StepCompleted(result=result) if result.needs_tools:
                # A hook may intervene: RESOLVE skips the reviewer and approves the
                # calls; BAIL skips it and rejects them (regenerate with the hook's
                # note, or give up if it asked to).
                review_payload = ReviewingToolCalls(tool_calls=result.tool_calls)
                verdict = await context.dependencies.hooks.call_on_reviewing_tool_calls_result(
                    context, review_payload
                )

                give_up = False
                if verdict is CompassHookResult.RESOLVE:
                    todo = None
                    adjusted_reasoning = None
                elif verdict is CompassHookResult.BAIL:
                    todo = None
                    adjusted_reasoning = review_payload.note
                    give_up = review_payload.give_up
                else:
                    review_result = await self.review_tool_calls(
                        context,
                        result.message.reasoning,
                        result.tool_calls,
                    )
                    todo = review_result.todo if review_result else None
                    adjusted_reasoning = review_result.adjusted_reasoning if review_result else None

                await context.dependencies.hooks.call_on_tool_calls_reviewed(
                    context,
                    ToolCallsReviewed(
                        tool_calls=result.tool_calls,
                        passed=adjusted_reasoning is None,
                        todo=todo,
                        adjusted_reasoning=adjusted_reasoning,
                    ),
                )

                if todo is not None:
                    context.state.todo = todo

                if adjusted_reasoning:
                    context.state.step_notes = adjusted_reasoning

                    # Was reasoning emitted by the model for this step?
                    if result.message.reasoning:
                        # We need to replace this step's reasoning in the state
                        assert len(context.state.reasoning_steps) > 0
                        context.state.reasoning_steps[-1] = adjusted_reasoning

                    if give_up:
                        raise _GiveUp(adjusted_reasoning, GiveUpKind.HOOK_REQUESTED)
                    raise _SemanticFailure()  # Restart the step with the adjusted reasoning in place
                else:
                    context.state.step_notes = ""

                if todo is not None and refresh_step_instructions:
                    await refresh_step_instructions()

                # Approved tool calls must be committed before their tool results
                # are appended, preserving the provider-required assistant-tool order.
                await commit_step(state, event)

                await context.session_event_emitter.emit_status_event(
                    trace_id=context.tracer.trace_id,
                    data=StatusEventData(status="processing", message="Running tools"),
                )

                await context.dependencies.hooks.call_on_calling_tools(context, result.tool_calls)

                await self.run_tool_calls(
                    context,
                    state,
                    result.tool_calls,
                    react,
                    tokenizer,
                )

                return True
            case _:
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
            # The MIN cost floor: no reviews, whatever was requested.
            return None

        # Tool review is requested declaratively (agent default or a matched
        # rule), or triggered by the tool author's own stakes marker: a
        # consequential tool in the call set.
        if not (
            context.state.tool_review_requested
            or _any_consequential_tool_called(context, tool_calls)
        ):
            return None

        await context.session_event_emitter.emit_status_event(
            trace_id=context.tracer.trace_id,
            data=StatusEventData(status="processing", message="Reviewing tool use"),
        )

        review_result = await self._reviewer().review_tool_calls(
            context,
            reasoning,
            tool_calls,
        )

        return review_result

    async def run_tool_calls(
        self,
        context: EngineContext,
        state: _LoopState,
        tool_calls: Sequence[ToolCallPart],
        react: ReactGenerator,
        tokenizer: EstimatingTokenizer,
    ) -> None:
        compass_tracer = CompassTracer(context.tracer)
        compass_tracer.tool_calls_requested(tool_calls)
        compass_tracer.loop_tools_started(len(tool_calls))
        try:
            await self._run_tool_calls_traced(context, state, tool_calls, react, tokenizer)
        except asyncio.CancelledError:
            compass_tracer.loop_tools_finished(len(tool_calls), outcome="cancelled")
            raise
        except Exception:
            compass_tracer.loop_tools_finished(len(tool_calls), outcome="failed")
            raise
        compass_tracer.loop_tools_finished(len(tool_calls))

    async def _run_tool_calls_traced(
        self,
        context: EngineContext,
        state: _LoopState,
        tool_calls: Sequence[ToolCallPart],
        react: ReactGenerator,
        tokenizer: EstimatingTokenizer,
    ) -> None:
        # A loop-provided gate before the step's tools run. The base loop's is a
        # no-op; the voice loop holds consequential tools while the triggering
        # utterance is still an unconfirmed speculation.
        await self._before_tool_calls(context, tool_calls)

        # Run all of the step's tool calls concurrently; results keep call order.
        results: tuple[ToolResult | None] = await safe_gather(
            *(self.run_tool_call(context, tool_call) for tool_call in tool_calls)
        )

        calls_and_results = list(zip(tool_calls, results))

        # Report the outcomes by tool name (a tool may be called more than once
        # in a step). An unresolved tool surfaces as a ``None`` result; a tool
        # that ran but failed carries its error inside the ``ToolResult`` itself.
        outcomes: dict[str, list[ToolCallOutcome]] = {}
        for tool_call, result in calls_and_results:
            outcomes.setdefault(tool_call.name, []).append(
                ToolCallOutcome(args=tool_call.args, result=result)
            )
        await context.dependencies.hooks.call_on_tool_calls_completed(context, outcomes)

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
                    if await tokenizer.estimate_token_count(to_prompt_json(result.data)) > 1_000:
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
                        is_error=metadata_utils.internal_value(
                            result.metadata,
                            ("error_details",),
                        )
                        is not None,
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

        def qualified_tool_id(name: str) -> str:
            # Tool events persist the QUALIFIED "service:tool" id — the react
            # layer only knows the model-facing bare name. Consumers (e.g. the
            # SDK's staged_tool_calls) parse the stored id with
            # ToolId.from_string, so a bare name would break them.
            if tool_id := context.state.tool_ids_by_name.get(name):
                return tool_id.to_string()
            return name

        def tool_event_data(call_ids: list[str]) -> ToolEventData:
            return ToolEventData(
                tool_calls=[
                    ToolCall(
                        tool_id=qualified_tool_id(tool_call.name),
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
            transient_tool_event = await context.response_event_emitter.emit_tool_event(
                trace_id=context.tracer.trace_id,
                data=tool_event_data(transient_call_ids),
            )
            context.state.tool_events.append(transient_tool_event)

        # Persist session-lifespan results together with the provider replay blob,
        # so a later turn can rebuild a faithful native tool turn from the stored
        # event. The original ToolCallParts carry the provider artifacts the
        # serializer needs. Skip entirely when
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
            react.serialize_tool_messages(
                [
                    Message(role=Role.ASSISTANT, parts=persisted_calls),
                    Message(role=Role.TOOL, parts=persisted_results),
                ],
                tool_message_serializer,
            )
            persisted_tool_event = await context.session_event_emitter.emit_tool_event(
                trace_id=context.tracer.trace_id,
                data=tool_event_data(persisted_call_ids),
                metadata=metadata_utils.set_internal_value(
                    {},
                    (_PROVIDER_DATA_KEY,),
                    cast(JSONSerializable, tool_message_serializer.provider_data),
                ),
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
            self._logger.warning(f"Model requested an unknown tool: {tool_call.name}")
            return None

        with CompassTracer(context.tracer).tool_call_scope(tool_call.id):
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

    # The mode attribute stamped on each bubble's authoritative loop.message
    # telemetry event; concrete loops override.
    _LOOP_MESSAGE_MODE = "block"

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        model_router: ModelRouter,
        tool_runner: ToolRunner,
        reviewer: Reviewer,
        usage_reporter: UsageReporter,
        cost_control_policy: CostControlPolicy,
        health_reporter: HealthReporter,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._compass_tracer = CompassTracer(tracer)
        self._meter = meter
        self._model_router = model_router
        self._tool_runner = tool_runner
        self._reviewer = reviewer
        self._cost_control_policy = cost_control_policy
        self._usage_reporter = usage_reporter
        self._health_reporter = health_reporter
        self._history_builder = _InteractionHistoryBuilder(
            logger,
            self._historical_assistant_text,
        )
        self._turn_instruction_builder = _TurnInstructionBuilder(logger)
        self._reasoning_event_processor = _ReasoningEventProcessor()
        self._tool_step_controller = _ToolStepController(
            logger,
            lambda: self._tool_runner,
            lambda: self._reviewer,
            self._before_consequential_tools,
        )

    def _historical_assistant_text(
        self,
        data: MessageEventData,
        metadata: Mapping[str, JSONSerializable] | None,
    ) -> str:
        """The text to render for a past assistant message when building history.

        The base loop is voice-agnostic and renders the full generated text; the
        voice loop overrides this to render only what was actually delivered
        (barge-in truth)."""
        return data["message"]

    async def _before_consequential_tools(
        self,
        context: EngineContext,
        tool_calls: Sequence[ToolCallPart],
    ) -> None:
        """Gate invoked before a step's tool calls run. A no-op by default; the
        voice loop overrides it to hold consequential tools while the triggering
        utterance is an unconfirmed speculation."""

    async def warm_up(self, job: LoopJob) -> Usage:
        self._logger.trace(f"Prefilling job for session {job.context.session.id}")
        react = await self._resolve_react(job.context, step=0)

        # Warm the cache for the stable prefix only — the system instructions and
        # the conversation so far. The per-turn instructions are dynamic and sit
        # past the cache breakpoint, so we leave them out here.
        history, _ = await self._history_builder.build(
            job,
            react=react,
            include_turn_instructions=False,
        )

        usage = await react.prefill(
            history=history,
            tools=await self._get_tools(job.context),
            tool_choice="auto",
            reasoning=await self._resolve_reasoning(job.context, job.reasoning_config),
        )

        if usage.input_tokens > 0:
            self._logger.debug(f"Prefill usage:\n {usage}")

        return usage

    async def run(self, job: LoopJob) -> LoopResult:
        with self._logger.scope(self.__class__.__name__):
            return await self._run(job)

    async def _run(self, job: LoopJob) -> LoopResult:
        # Give hooks (e.g. retrievers) a chance to stage tool events before we
        # build the history — history building folds context.state.tool_events in.
        if not await job.context.dependencies.hooks.call_on_generating_messages(job.context):
            # A hook requested that we not proceed with generating a response.
            return LoopResult(job=job, steps=[])

        # Don't seed the turn instructions here: the first _run_step's
        # _update_step_instructions builds and places them (and on high/max effort,
        # which don't cache step_instructions, seeding here would build them twice).
        # _last_user_message_index (the seed's placement) is identical to refresh's
        # insertion index, so placement is unchanged.
        react = await self._resolve_react(job.context, step=0)
        history, instructions_index = await self._history_builder.build(
            job,
            react=react,
            include_turn_instructions=False,
        )
        state = _LoopState(
            history=history,
            instructions_index=instructions_index,
            preamble=_ToolPreambleState(job.preamble_config),
        )

        # The DI-configurable last-resort fallback travels on the job; stash it on the
        # per-turn state so the moderation gate can reach it deep in the step handling.
        job.context.state.moderation_fallback = job.moderation_fallback

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
                        with job.context.tracer.span(
                            "compass.loop.step",
                            {"loop": self.__class__.__name__, "step_index": len(state.steps)},
                        ):
                            await self._run_step_protected(job, state)
                    except _SemanticFailure:
                        semantic_failure_count += 1
                        await self._reset_message_after_restart(job.context, state)
                        continue
                    except Exception as exc:
                        # A hook may reject a review with an immediate give-up; the
                        # latch wraps it, so unwrap it here to reach the give-up path.
                        if isinstance(exc.__cause__, _GiveUp):
                            raise exc.__cause__

                        if not isinstance(exc.__cause__, _SemanticFailure):
                            raise

                        semantic_failure_count += 1
                        await self._reset_message_after_restart(job.context, state)
                        continue
                    else:
                        break

                if semantic_failure_count == max_semantic_failures:
                    # The shared retry budget is exhausted by any mix of rejected tool
                    # calls, rejected messages (review), and moderation-blocked messages.
                    raise _GiveUp(
                        "The agent's output was repeatedly rejected by policy review or moderation.",
                        GiveUpKind.RETRIES_EXHAUSTED,
                    )

                job.context.state.iterations.append(
                    IterationState(
                        matched_rules=[],
                        ruled_out=[],
                        resolved_rules=[],
                        executed_tools=[],
                    )
                )

                if len(job.context.state.iterations) >= 30:
                    self._logger.warning(
                        f"Large number of engine steps on session {job.context.session.id} ({job.context.session.title or 'Untitled'})"
                    )

                # Only give up if the step that just ran didn't itself prepare the
                # response — a message produced on the last allowed iteration is a
                # success, not a reason to force an apology on top of it.
                if (
                    not job.context.state.prepared_to_respond
                    and len(job.context.state.iterations) == job.context.agent.max_steps
                ):
                    raise _GiveUp(
                        "The agent reached the maximum number of engine steps "
                        "without preparing a response.",
                        GiveUpKind.MAX_STEPS,
                    )
            except asyncio.CancelledError:
                self._logger.warning(
                    f"{self.__class__.__name__} run cancelled on session {job.context.session.id}"
                )
                raise
            except _GiveUp as exc:
                await self._give_up(job, state, str(exc), exc.kind)

        await job.context.session_event_emitter.emit_status_event(
            trace_id=job.context.tracer.trace_id,
            data=StatusEventData(status="ready", data={"stage": "completed"}),
        )

        await job.context.dependencies.hooks.call_on_messages_emitted(
            job.context,
            job.context.state.emitted_messages,
        )

        return LoopResult(job=job, steps=state.steps)

    async def _maybe_moderate_message(
        self,
        context: EngineContext,
        result: StepResult,
    ) -> tuple[str, Mapping[str, JSONSerializable]] | None:
        """Gate a completed message on agent-side moderation before it is emitted.

        Mirrors ``_maybe_review_message``: only runs for a pure message step (no
        pending tool calls). The agent's moderation property governs the gate: OFF
        skips regardless of routing; AUTO runs iff a "moderation.agent" model is
        routed (silently skipping otherwise); ON requires the route (a resolution
        failure here is a mid-run configuration regression, logged as an error). A
        service failure fails open — moderation is a safety layer, not a delivery gate.

        On a flag the message is regenerated: the moderation reason (never the flagged
        text) is installed as the next step's guidance and ``_SemanticFailure`` is
        raised, so nothing is shown and the shared retry budget converts repeated flags
        into a give-up. The one exception is the terminal give-up step — it cannot
        retry, so if its forced message is itself flagged, this returns the
        DI-configurable fallback text and the ``blocked`` verdict metadata for the
        caller to emit instead. Returns ``None`` when the message may be emitted as-is.
        """
        if result.needs_tools or not result.message.text.strip():
            return None

        if context.agent.moderation is Moderation.OFF:
            return None

        text = result.message.text

        try:
            service = await self._model_router.get_moderation_service(
                context=context,
                route="moderation.agent",
                agent_id=context.agent.id,
            )
        except Exception as exc:
            if context.agent.moderation is Moderation.ON:
                self._logger.error(
                    "Agent moderation is ON but no model resolves for the "
                    f"'moderation.agent' route; message NOT moderated ({exc})"
                )
            return None

        try:
            check = await service.moderate_agent(
                AgentModerationContext(
                    session=context.session,
                    message=text,
                    engine_context=context,
                )
            )
        except Exception as exc:
            self._logger.warning(f"Agent-message moderation failed (failing open): {exc}")
            return None

        if not check.flagged:
            return None

        blocked_metadata = metadata_utils.set_internal_value(
            {}, ("moderation",), {"blocked": True, "tags": list(check.tags)}
        )

        if context.state.giving_up:
            # The terminal give-up message is itself flagged and cannot be regenerated:
            # emit the configurable last-resort fallback rather than the flagged text.
            self._logger.warning(
                f"The forced give-up message was blocked by moderation (tags: {check.tags}) "
                f"on session {context.session.id}; emitting the configured fallback."
            )
            return context.state.moderation_fallback, blocked_metadata

        # Regenerate with the reason (never the flagged text) as the next step's guidance.
        reason = ", ".join(str(tag) for tag in check.tags) or "policy"
        note = (
            "Your last attempt to generate the message was flagged by moderation "
            f"(reasons: {reason}). Reconsider how to respond so it won't be flagged again."
        )
        self._logger.warning(
            f"Agent message blocked by moderation (tags: {check.tags}) on session "
            f"{context.session.id}; regenerating."
        )
        context.state.step_notes = note
        if context.state.reasoning_steps:
            context.state.reasoning_steps[-1] = note

        raise _SemanticFailure()

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
        except PricingConfigurationError:
            raise
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
        return job.context.dependencies.optimization_policy.get_max_review_restarts(
            job.context.state.dynamic_effort_level
        )

    async def _run_step_protected(self, job: LoopJob, state: _LoopState) -> None:
        async def defer_cancellation(latch: CancellationSuppressionLatch[None]) -> None:
            latch.enable()
            state.cancellation_pending = lambda: latch.cancellation_pending
            await self._run_step_internal(job, state)
            latch.disable()  # Raise any internally-caught exception

        return await latched_shield(defer_cancellation)

    async def _run_step_internal(self, job: LoopJob, state: _LoopState) -> None:
        """Run one react step, processing each event. A transient ReactError is
        retried — but only while NO event has been produced yet: once events have
        been emitted, the stream can't be replayed (it would re-emit chunks and
        re-run side effects), so the error propagates. The transient errors we
        retry are raised when the stream is opened, before any event."""

        self._update_system_instructions(job, state)
        await self._update_step_instructions(job, state)

        tokenizer = await self._resolve_tokenizer(job.context, step=len(state.steps))
        react = await self._resolve_react(job.context, step=len(state.steps))
        react_hints: ReactGeneratorHints = {
            "hedge_timeout": job.context.dependencies.optimization_policy.get_hedge_timeout(
                "react", qualifiers=job.context.state.model_route_qualifiers
            )
        }

        step_number = len(state.steps)
        await job.context.dependencies.hooks.call_on_step_started(job.context, step_number)

        for attempt in range(len(self._RETRY_TIMEOUT_PER_ATTTEMPT) + 1):
            produced = False
            step_started = time.monotonic()
            phases = _StepPhaseTracer(CompassTracer(job.context.tracer), self._LOOP_MESSAGE_MODE)
            phases.begin()
            try:
                async for event in react.stream_step(
                    history=state.history,
                    tools=[] if state.disable_tools else await self._get_tools(job.context),
                    tool_choice="none" if state.disable_tools else "auto",
                    reasoning=await self._resolve_reasoning(job.context, job.reasoning_config),
                    hints=react_hints,
                ):
                    produced = True
                    phases.observe(event)
                    if isinstance(event, StepCompleted):
                        self._report_step_health(
                            time.monotonic() - step_started,
                            agent_id=job.context.agent.id,
                            success=True,
                            error=None,
                            usage=event.result.usage,
                        )
                    await self._process_reasoning_event(job.context, state, event)
                    # Surface/close the message BEFORE the tool status: a text->tool
                    # transition finalizes the preamble as its own bubble first, so the
                    # tool status follows the message (no flicker), and post-tool text
                    # starts a fresh message instead of gluing onto the preamble.
                    await self._surface_message_event(job.context, state, event)
                    committed = await self._process_tool_event(
                        job,
                        state,
                        event,
                        react,
                        tokenizer,
                    )
                    if not committed:
                        await self._commit_react_event(state, event, react)

                phases.close("completed")
                await job.context.dependencies.hooks.call_on_step_completed(
                    job.context, step_number
                )
                return
            except asyncio.CancelledError:
                phases.close("cancelled")
                raise
            except ReactError as exc:
                phases.close("failed")
                if (
                    not exc.retryable
                    or produced
                    or attempt == len(self._RETRY_TIMEOUT_PER_ATTTEMPT)
                ):
                    self._report_step_health(
                        time.monotonic() - step_started,
                        agent_id=job.context.agent.id,
                        success=False,
                        error=exc,
                        usage=None,
                    )
                    raise
                wait = self._RETRY_TIMEOUT_PER_ATTTEMPT[attempt]
                self._logger.warning(
                    f"Retrying step after a transient error "
                    f"({exc}); retrying in {wait}s (attempt {attempt + 2})."
                )
                await asyncio.sleep(wait)
            except Exception:
                phases.close("failed")
                raise

    async def _give_up(
        self,
        job: LoopJob,
        state: _LoopState,
        reason: str,
        kind: GiveUpKind,
    ) -> None:
        verdict = await job.context.dependencies.hooks.call_on_giving_up_result(
            job.context, GaveUp(reason=reason, steps=len(state.steps), kind=kind)
        )

        # Housekeeping runs either way: the failed attempt may have left a partial
        # message and empty assistant turns behind, and nobody should inherit those.
        await self._reset_message_after_restart(job.context, state)
        self._drop_trailing_empty_assistant_messages(state)

        if verdict is CompassHookResult.RESOLVE:
            # A hook claimed the response — it has said whatever it wanted to say (or
            # deliberately said nothing). Skip the forced message, and mark the turn
            # answered so the loop doesn't take another run at the task.
            self._logger.info(
                f"{reason} A hook took over the final response on session "
                f"{job.context.session.id} ({job.context.session.title or 'Untitled'})."
            )
            job.context.state.prepared_to_respond = True
            return

        self._logger.warning(
            f"{reason} Forcing a final message on session {job.context.session.id} "
            f"({job.context.session.title or 'Untitled'}). Reasoning: \n"
            f"{json.dumps(job.context.state.reasoning_steps, indent=2)}"
        )

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

    async def _resolve_react(
        self,
        context: EngineContext,
        *,
        step: int,
    ) -> ReactGenerator:
        return await self._model_router.get_react_generator(
            context=context,
            route="react",
            qualifiers=context.state.model_route_qualifiers,
        )

    async def _resolve_reasoning(
        self,
        context: EngineContext,
        default: ReasoningConfig | None,
    ) -> ReasoningConfig | None:
        """The reasoning for this react call: the resolved ReAct ModelConfig's
        reasoning if it sets one (per agent, per effort qualifier), otherwise the
        engine's built-in effort→reasoning ``default`` (``job.reasoning_config``)."""
        configured = await self._model_router.resolve_react_reasoning(
            context=context,
            route="react",
            qualifiers=context.state.model_route_qualifiers,
        )
        return configured or default

    async def _resolve_tokenizer(
        self,
        context: EngineContext,
        *,
        step: int,
    ) -> EstimatingTokenizer:
        return await self._model_router.get_estimating_tokenizer(
            context=context,
            route="tokenizer",
            qualifiers=context.state.model_route_qualifiers,
        )

    def _tool_preamble_decision(self, state: _LoopState) -> PreambleDecision:
        return state.preamble.decide(time.monotonic())

    def _tool_preamble_is_allowed(self, state: _LoopState) -> bool:
        return state.preamble.configuration.allows_emission(self._tool_preamble_decision(state))

    def _should_trim_tool_preamble_text(self, state: _LoopState) -> bool:
        return state.preamble.configuration.trims_preamble_text()

    def _report_step_health(
        self,
        duration_seconds: float,
        *,
        agent_id: AgentId,
        success: bool,
        error: BaseException | None,
        usage: Usage | None,
    ) -> None:
        """Report one react step as an NLP request health sample.

        The react path bypasses the router's health-attributing wrapper (which
        reports schematic/embedding requests), so the loop is the producer for
        the conversation's actual generation traffic — attributed to the react
        route and the responding agent, like every other NLP health sample."""
        self._health_reporter.report(
            NLP_REQUEST_KIND,
            {
                NLPHealthView.ATTR_ROUTE: "react",
                NLPHealthView.ATTR_AGENT: str(agent_id),
                NLPHealthView.ATTR_SCHEMA: "React",
                NLPHealthView.ATTR_MODEL: usage.model_name if usage is not None else "",
                NLPHealthView.ATTR_SUCCESS: success,
                NLPHealthView.ATTR_LATENCY_MS: duration_seconds * 1000.0,
                NLPHealthView.ATTR_ERROR_CLASS: (
                    type(error).__name__ if error is not None else None
                ),
            },
        )
        self._health_reporter.increment_counter(
            NLP_REQUESTS_COUNTER, max(usage.requests, 1) if usage is not None else 1
        )
        if usage is not None:
            self._health_reporter.increment_counter(
                NLP_TOKENS_COUNTER, usage.input_tokens + usage.output_tokens
            )

    def _record_emitted_message(self, context: EngineContext, data: MessageEventData) -> None:
        """Record a finalized message bubble for the ``on_messages_emitted`` payload.

        Called once per bubble, with the data it was finalized with — for a streamed
        message that is the handle update, not the event's opening chunk."""
        context.state.emitted_messages.append(data["message"])

    def _mark_user_visible_message_emitted(self, context: EngineContext, state: _LoopState) -> None:
        now = time.monotonic()

        # The turn's FIRST user-visible message is the TTFM health sample —
        # measured from the moment the engine began processing the turn.
        if (
            not state.preamble.user_visible_message_emitted
            and context.state.turn_started_monotonic is not None
        ):
            self._health_reporter.report(
                ENGINE_TTFM_KIND,
                {
                    EngineHealthView.ATTR_TTFM_MS: (
                        (now - context.state.turn_started_monotonic) * 1000.0
                    )
                },
            )

        state.preamble.mark_user_visible_message_emitted(now)

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

    async def _commit_react_event(
        self,
        state: _LoopState,
        event: StreamEvent,
        react: ReactGenerator,
    ) -> None:
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

            self._logger.debug(f"Step usage:\n {event.result.usage}")
            self._usage_reporter.report_usage(
                event.result.usage.model_name,
                provider_name=react.provider_name,
                usage=event.result.usage.to_usage_info(),
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
        react: ReactGenerator,
        tokenizer: EstimatingTokenizer,
    ) -> bool:
        return await self._tool_step_controller.process(
            job.context,
            state,
            event,
            lambda s, e: self._commit_react_event(s, e, react),
            react,
            tokenizer,
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
        react = await self._resolve_react(context, step=len(state.steps))
        tokenizer = await self._resolve_tokenizer(context, step=len(state.steps))
        await self._tool_step_controller.run_tool_calls(
            context,
            state,
            tool_calls,
            react,
            tokenizer,
        )

    async def _run_tool_call(
        self, context: EngineContext, tool_call: ToolCallPart
    ) -> ToolResult | None:
        return await self._tool_step_controller.run_tool_call(context, tool_call)

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

    def _should_buffer_message_for_review(self, context: EngineContext) -> bool:
        """When the message will be reviewed, it is reviewed whole before it may
        be shown, so even a streaming loop must buffer it (no live deltas)
        rather than stream it out."""
        return self._should_review_message(context)

    def _should_review_message(self, context: EngineContext) -> bool:
        """Message review runs when requested (agent default or a turn-matched
        rule), except under the MIN-effort cost floor."""
        if context.state.giving_up:
            # The forced message is the turn's last word. A rejection here has
            # nowhere to go — the step would fail and the customer would get
            # silence — so the give-up message is never reviewed.
            return False

        if context.state.dynamic_effort_level == Effort.MIN:
            return False

        return context.state.message_review_requested

    async def _maybe_review_message(self, context: EngineContext, result: StepResult) -> None:
        """Gate a completed message on the reviewer before it is emitted.

        Only runs when message review was requested and only for a pure message step
        (no pending tool calls). On a breach the reviewer's adjusted reasoning is
        installed as the next step's guidance and ``_SemanticFailure`` is raised so
        the step regenerates — nothing is shown to the user. Concrete loops must
        call this at the top of their step-completion handling, before any emit."""
        if result.needs_tools or not result.message.text.strip():
            return

        if not self._should_review_message(context):
            return

        await context.session_event_emitter.emit_status_event(
            trace_id=context.tracer.trace_id,
            data=StatusEventData(status="processing", message="Reviewing message"),
        )

        # A hook may intervene on the review: RESOLVE skips the reviewer and treats
        # the message as passed; BAIL skips it and rejects the message (regenerate
        # with the hook's note, or give up if it asked to).
        review_payload = ReviewingMessage(message=result.message.text)
        verdict = await context.dependencies.hooks.call_on_reviewing_message_result(
            context, review_payload
        )

        if verdict is CompassHookResult.BAIL:
            await context.dependencies.hooks.call_on_message_reviewed(
                context,
                MessageReviewed(
                    message=result.message.text,
                    passed=False,
                    adjusted_reasoning=review_payload.note,
                ),
            )
            context.state.step_notes = review_payload.note
            if context.state.reasoning_steps:
                context.state.reasoning_steps[-1] = review_payload.note
            if review_payload.give_up:
                raise _GiveUp(review_payload.note, GiveUpKind.HOOK_REQUESTED)
            raise _SemanticFailure()

        if verdict is CompassHookResult.RESOLVE:
            review = None
        else:
            review = await self._reviewer.review_message(context, result.message.text)

        passed = review is None or not review.adjusted_reasoning
        await context.dependencies.hooks.call_on_message_reviewed(
            context,
            MessageReviewed(
                message=result.message.text,
                passed=passed,
                adjusted_reasoning=review.adjusted_reasoning if review else None,
            ),
        )

        if passed:
            return

        assert review is not None and review.adjusted_reasoning is not None
        context.state.step_notes = review.adjusted_reasoning

        if context.state.reasoning_steps:
            context.state.reasoning_steps[-1] = review.adjusted_reasoning

        raise _SemanticFailure()

    async def _complete_message_step(self, context: EngineContext, result: StepResult) -> bool:
        """Shared step-completion tail for a step that produced a message: a message
        with no pending tool calls ends the loop, and the message-generated hook
        fires regardless of output mode.

        Returns the hook verdict: ``False`` when a hook BAILed. Only the
        blocking loop can act on it — it consults the hooks *before* emitting
        the final message, so a BAIL suppresses the emission (the handler may
        emit a replacement itself). Streamed output is already on the wire by
        the time the step completes, so there the hook is observational."""
        if not result.needs_tools:
            context.state.prepared_to_respond = True

        return await context.dependencies.hooks.call_on_message_generated(
            context, result.message.text
        )

    async def _reset_message_after_restart(self, context: EngineContext, state: _LoopState) -> None:
        """A reviewer-rejected step is being restarted. Finalize the message already
        streamed during the rejected attempt as its own (complete) bubble, then reset
        the streaming state so the retry begins a fresh message instead of appending to
        it. A blocking loop doesn't stream a partial message (its handle is None here),
        so this only clears the per-attempt flags there."""
        if state.message.handle is not None:
            data = MessageEventData(
                message=state.message.buffer.getvalue() if state.message.buffer else "",
                participant=Participant(id=context.agent.id, display_name=context.agent.name),
                chunks=[*state.message.chunks, None],
            )
            await state.message.handle.update(data)
            self._compass_tracer.loop_message(
                data,
                self._LOOP_MESSAGE_MODE,
                state.message.consume_output_ref(),
            )

        state.message.reset_step_output()
        state.in_the_middle_of_running_tools = False

    async def _build_history(
        self,
        job: LoopJob,
        *,
        include_turn_instructions: bool = True,
    ) -> tuple[list[Message], int | None]:
        react = await self._resolve_react(
            job.context,
            step=len(job.context.state.iterations),
        )
        return await self._history_builder.build(
            job,
            react=react,
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
