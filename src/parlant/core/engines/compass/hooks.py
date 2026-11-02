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

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence, TypeAlias

# Engine-agnostic: a bare EngineContext is EngineContext[Any, Any], so hook/retriever
# authors see state typed as Any (the concrete ResponseState is hidden).
from parlant.core.engines.engine_context import EngineContext
from parlant.core.engines.rule_evaluation import RuleEvaluation
from parlant.core.nlp.react import ToolCallPart
from parlant.core.rules import RuleId as GuidelineId
from parlant.core.tools import ToolResult


# ── Payloads for the Compass hooks (passed as the hook's `payload` argument) ──


class GiveUpKind(Enum):
    """Why the loop stopped trying, for handlers that respond differently per cause."""

    RETRIES_EXHAUSTED = auto()
    """The shared retry budget ran out: output was rejected by review or blocked by
    moderation once too often (see ``get_max_review_restarts``)."""

    MAX_STEPS = auto()
    """The turn used up ``agent.max_steps`` without preparing a response."""

    HOOK_REQUESTED = auto()
    """A review hook rejected the step with ``payload.give_up`` set, so the turn was
    told to stop rather than retry."""


@dataclass(frozen=True)
class GaveUp:
    """Payload of ``on_giving_up``: why the loop stopped and how far it got."""

    reason: str
    steps: int
    kind: GiveUpKind


@dataclass(frozen=True)
class ToolCallOutcome:
    """One tool call's arguments and what it produced.

    ``result`` is ``None`` when the model named a tool that couldn't be resolved
    (an unknown tool). A tool that ran but failed still returns a ``ToolResult``,
    carrying its error under ``result.metadata`` (``error_details``) — the loop
    never surfaces a raised exception here, since a raising tool aborts the whole
    step rather than producing a per-call outcome."""

    args: Mapping[str, Any]
    result: Optional[ToolResult] = None


@dataclass(frozen=True)
class ToolCallsReviewed:
    """Payload of ``on_tool_calls_reviewed``. ``passed`` is False when the
    reviewer sent the step back to regenerate (an ``adjusted_reasoning`` breach)."""

    tool_calls: Sequence[ToolCallPart]
    passed: bool
    todo: Optional[str] = None
    adjusted_reasoning: Optional[str] = None


@dataclass(frozen=True)
class MessageReviewed:
    """Payload of ``on_message_reviewed``. ``passed`` is False when the reviewer
    sent the message back to regenerate."""

    message: str
    passed: bool
    adjusted_reasoning: Optional[str] = None


@dataclass(frozen=True)
class Compacted:
    """Payload of ``on_compacted``: session token counts before and after."""

    before: int
    after: int


# Default correction fed to the next attempt when a hook rejects a review via BAIL
# without setting its own note.
DEFAULT_REVIEW_REJECTION_NOTE = "The review was rejected - please look into it."


@dataclass
class ReviewingMessage:
    """Payload of ``on_reviewing_message`` — a two-way channel. The engine passes
    the drafted ``message`` in; a hook that returns BAIL rejects it, and may first
    edit ``note`` (the correction fed to the next attempt) and ``give_up`` (True to
    force the turn to give up now instead of regenerating)."""

    message: str
    note: str = DEFAULT_REVIEW_REJECTION_NOTE
    give_up: bool = False


@dataclass
class ReviewingToolCalls:
    """Payload of ``on_reviewing_tool_calls`` — a two-way channel. The engine passes
    the proposed ``tool_calls`` in; a hook that returns BAIL rejects them, and may
    first edit ``note`` (the correction fed to the next attempt) and ``give_up``
    (True to force the turn to give up now instead of regenerating)."""

    tool_calls: Sequence[ToolCallPart]
    note: str = DEFAULT_REVIEW_REJECTION_NOTE
    give_up: bool = False


class CompassHookResult(Enum):
    CALL_NEXT = auto()
    """Runs the next hook in the chain, if any"""

    RESOLVE = auto()
    """Skip this section's normal work and jump ahead to the next phase (without
    running the remaining hooks in the chain). What "this section" is depends on the
    hook: RESOLVE on ``on_preparing`` skips discovery/matching and responds
    directly; on ``on_reviewing_tool_calls`` / ``on_reviewing_message`` it skips the
    reviewer, treating the calls/message as approved/passed. Hooks that define no
    RESOLVE behavior treat it like CALL_NEXT."""

    BAIL = auto()
    """Stop the chain and take the negative/stop branch of whatever this hook brackets:

    - Early turn gates (``on_acknowledging`` / ``on_acknowledged`` / ``on_preparing`` /
      ``on_generating_messages``): stop the turn — the engine does nothing further, so the
      hook should emit any fallback itself first.
    - Review hooks (``on_reviewing_tool_calls`` / ``on_reviewing_message``): reject the
      calls/message — the step regenerates with the payload's ``note``, or gives up if the
      hook set ``payload.give_up``.
    - ``on_message_generated`` (blocking output only): suppress the message emission; the
      handler may emit a replacement itself.
    - Other points ignore it (treated like ``CALL_NEXT``)."""


CompassHook: TypeAlias = Callable[
    [EngineContext, Any, Optional[Exception]], Awaitable[CompassHookResult]
]
"""A callable that takes an EngineContext and an optional Exception, and returns a CompassHookResult."""


@dataclass(frozen=False)
class CompassHooks:
    on_error: list[CompassHook] = field(default_factory=list)
    """Called when the engine has encountered a runtime error"""

    on_acknowledging: list[CompassHook] = field(default_factory=list)
    """Called just before emitting an acknowledgement status event"""

    on_acknowledged: list[CompassHook] = field(default_factory=list)
    """Called right after emitting an acknowledgement status event"""

    on_preparing: list[CompassHook] = field(default_factory=list)
    """Called just before beginning the preparation iterations"""

    on_generating_messages: list[CompassHook] = field(default_factory=list)
    """Called just before generating messages"""

    on_message_generated: list[CompassHook] = field(default_factory=list)
    """Called right after a message was generated (but not yet emitted)"""

    on_messages_emitted: list[CompassHook] = field(default_factory=list)
    """Called right after all messages were emitted into the session. The payload is
    the text of each message the turn delivered (``Sequence[str]``), in the order the
    customer saw it — a tool preamble and the reply that follows are two entries."""

    on_rule_discovered_handlers: dict[
        GuidelineId, list[Callable[[EngineContext, RuleEvaluation], Awaitable[None]]]
    ] = field(default_factory=lambda: defaultdict(list))
    """Handlers called when a rule enters Compass's session working set."""

    on_rule_matched_handlers: dict[
        GuidelineId, list[Callable[[EngineContext, RuleEvaluation], Awaitable[None]]]
    ] = field(default_factory=lambda: defaultdict(list))
    """Map from GuidelineId to handlers called when that guideline is matched."""

    on_step_started: list[CompassHook] = field(default_factory=list)
    """Called just before a ReAct step runs. Payload: the step number (int)."""

    on_step_completed: list[CompassHook] = field(default_factory=list)
    """Called after a ReAct step completes (not on a reviewer-forced restart).
    Payload: the step number (int)."""

    on_giving_up: list[CompassHook] = field(default_factory=list)
    """Called when the loop gives up (step/failure limit). Payload: ``GaveUp``."""

    on_calling_tools: list[CompassHook] = field(default_factory=list)
    """Called just before the step's approved tool calls run. Payload:
    ``Sequence[ToolCallPart]``."""

    on_tool_calls_completed: list[CompassHook] = field(default_factory=list)
    """Called right after the step's tool calls finish. Payload:
    ``Mapping[str, list[ToolCallOutcome]]`` (by tool name)."""

    on_reviewing_tool_calls: list[CompassHook] = field(default_factory=list)
    """Called just before the reviewer checks proposed tool calls. Payload:
    ``ReviewingToolCalls`` (RESOLVE approves as-is, BAIL rejects with its ``note`` /
    ``give_up``)."""

    on_tool_calls_reviewed: list[CompassHook] = field(default_factory=list)
    """Called right after the reviewer checks proposed tool calls (before any
    regenerate). Payload: ``ToolCallsReviewed``."""

    on_reviewing_message: list[CompassHook] = field(default_factory=list)
    """Called just before the reviewer checks a drafted message. Payload:
    ``ReviewingMessage`` (RESOLVE accepts as passed, BAIL rejects with its ``note`` /
    ``give_up``)."""

    on_message_reviewed: list[CompassHook] = field(default_factory=list)
    """Called right after the reviewer checks a drafted message (before any
    regenerate). Payload: ``MessageReviewed``."""

    on_message_updated: list[CompassHook] = field(default_factory=list)
    """Called as a streamed message grows. Payload: the full message text so far
    (str). Streaming output only."""

    on_compacting: list[CompassHook] = field(default_factory=list)
    """Called just before session-history compaction. Payload: token count before
    compaction (int)."""

    on_compacted: list[CompassHook] = field(default_factory=list)
    """Called right after session-history compaction. Payload: ``Compacted``."""

    async def call_on_error(self, context: EngineContext, exception: Exception) -> bool:
        return await self.call_hooks(self.on_error, context, None, exception)

    async def call_on_acknowledging(self, context: EngineContext) -> bool:
        return await self.call_hooks(self.on_acknowledging, context, None)

    async def call_on_acknowledged(self, context: EngineContext) -> bool:
        return await self.call_hooks(self.on_acknowledged, context, None)

    async def call_on_preparing(self, context: EngineContext) -> bool:
        return await self.call_hooks(self.on_preparing, context, None)

    async def call_on_generating_messages(self, context: EngineContext) -> bool:
        return await self.call_hooks(self.on_generating_messages, context, None)

    async def call_on_message_generated(self, context: EngineContext, payload: str) -> bool:
        return await self.call_hooks(self.on_message_generated, context, payload)

    async def call_on_messages_emitted(
        self,
        context: EngineContext,
        messages: Sequence[str],
    ) -> bool:
        return await self.call_hooks(self.on_messages_emitted, context, messages)

    async def call_on_step_started(self, context: EngineContext, step: int) -> bool:
        return await self.call_hooks(self.on_step_started, context, step)

    async def call_on_step_completed(self, context: EngineContext, step: int) -> bool:
        return await self.call_hooks(self.on_step_completed, context, step)

    async def call_on_giving_up_result(
        self, context: EngineContext, payload: GaveUp
    ) -> CompassHookResult:
        return await self._call_hooks_result(self.on_giving_up, context, payload)

    async def call_on_calling_tools(
        self, context: EngineContext, tool_calls: Sequence[ToolCallPart]
    ) -> bool:
        return await self.call_hooks(self.on_calling_tools, context, tool_calls)

    async def call_on_tool_calls_completed(
        self, context: EngineContext, outcomes: Mapping[str, list[ToolCallOutcome]]
    ) -> bool:
        return await self.call_hooks(self.on_tool_calls_completed, context, outcomes)

    async def call_on_tool_calls_reviewed(
        self, context: EngineContext, payload: ToolCallsReviewed
    ) -> bool:
        return await self.call_hooks(self.on_tool_calls_reviewed, context, payload)

    async def call_on_message_reviewed(
        self, context: EngineContext, payload: MessageReviewed
    ) -> bool:
        return await self.call_hooks(self.on_message_reviewed, context, payload)

    async def call_on_message_updated(self, context: EngineContext, message: str) -> bool:
        return await self.call_hooks(self.on_message_updated, context, message)

    async def call_on_compacting(self, context: EngineContext, tokens_before: int) -> bool:
        return await self.call_hooks(self.on_compacting, context, tokens_before)

    async def call_on_compacted(self, context: EngineContext, payload: Compacted) -> bool:
        return await self.call_hooks(self.on_compacted, context, payload)

    async def call_on_preparing_result(self, context: EngineContext) -> CompassHookResult:
        return await self._call_hooks_result(self.on_preparing, context, None)

    async def call_on_reviewing_tool_calls_result(
        self, context: EngineContext, payload: ReviewingToolCalls
    ) -> CompassHookResult:
        return await self._call_hooks_result(self.on_reviewing_tool_calls, context, payload)

    async def call_on_reviewing_message_result(
        self, context: EngineContext, payload: ReviewingMessage
    ) -> CompassHookResult:
        return await self._call_hooks_result(self.on_reviewing_message, context, payload)

    async def _call_hooks_result(
        self,
        hooks: Sequence[CompassHook],
        context: EngineContext,
        payload: Any,
        exc: Optional[Exception] = None,
    ) -> CompassHookResult:
        """Run the chain and return the first non-CALL_NEXT verdict (RESOLVE stops
        the chain and jumps ahead; BAIL stops the turn), or CALL_NEXT if the chain
        is exhausted."""
        for callable in hooks:
            result = await callable(context, payload, exc)
            if result is not CompassHookResult.CALL_NEXT:
                return result
        return CompassHookResult.CALL_NEXT

    async def call_hooks(
        self,
        hooks: Sequence[CompassHook],
        context: EngineContext,
        payload: Any,
        exc: Optional[Exception] = None,
    ) -> bool:
        # Bool view for the many call sites that only distinguish "keep going"
        # (CALL_NEXT/RESOLVE) from BAIL. Sites that also need RESOLVE call
        # _call_hooks_result (or a call_on_*_result wrapper) directly.
        return await self._call_hooks_result(hooks, context, payload, exc) is not (
            CompassHookResult.BAIL
        )
