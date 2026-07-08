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

"""The cost-control policy port.

Cost control is a policy layer around chargeable NLP work, separate from
``UsageReporter`` (which remains pure accounting). The engine calls
:meth:`CostControlPolicy.check` at coarse choke points before doing chargeable
work, and observed usage flows into :meth:`CostControlPolicy.report`.

The interface can express denial, and the engine honors it — deployments may
bind an enforcing policy. The built-in :class:`AdvisoryCostControlPolicy` never
denies: it provides visibility and advisory warnings, not automatic blocking.
"""

import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Callable, Mapping, Optional, Sequence

from parlant.core.agents import AgentId
from parlant.core.customers import CustomerId
from parlant.core.nlp.common import UsageInfo
from parlant.core.sessions import SessionId
from parlant.core.usage_reporter import UsageReporter


@dataclass(frozen=True)
class CostContext:
    """Identity of the work being charged.

    Deliberately carries only entities that exist in Parlant's data model; scope
    resolution (including mapping to external project/tenant hierarchies) lives
    entirely inside policy implementations, keeping this interface stable."""

    agent_id: AgentId
    session_id: SessionId
    customer_id: CustomerId
    trace_id: str


class WorkKind(Enum):
    """The coarse choke points chargeable work is gated at.

    Checks happen in engine code at these boundaries — never inside provider
    adapters."""

    TURN = auto()
    """A whole user-facing turn, checked after acknowledgement and before
    matching/preparation begins. A deny produces the cooldown status protocol."""

    STEP = auto()
    """One response-loop step. A deny mid-turn finishes the current step
    gracefully and stops iterating — in-flight streamed text is never truncated."""

    BACKGROUND = auto()
    """Post-response work that costs money but has no visible turn (cache
    warm-ups, session pruning). Compaction is exempt by design: it reduces
    future cost."""


@dataclass(frozen=True)
class CostVerdict:
    """The outcome of a :meth:`CostControlPolicy.check` call.

    Richer than a boolean so the cooldown protocol and advisory warnings ride
    the same call."""

    allowed: bool

    warnings: Sequence[str] = field(default=())
    """Advisory notices (e.g. a soft threshold was crossed). May accompany an
    allowed verdict."""

    retry_after_utc: Optional[datetime] = None
    """For denials with a known recovery time (e.g. a cooldown's expiry): the
    absolute time after which retrying is expected to succeed. Advisory —
    retrying early simply yields another denial with an updated hint."""

    reason: Optional[str] = None
    """Human-readable explanation of a denial or warning, for logs/diagnostics."""

    scope: Optional[str] = None
    """Which scope produced the denial/warning (e.g. "session" for a per-session
    circuit breaker). Surfaces in the client-visible status payload so frontends
    can distinguish scopes; the engine defaults it sensibly when absent."""


@dataclass(frozen=True)
class CostWeights:
    """Weights turning raw token counts into cost units.

    Thresholds are defined over weighted units rather than raw tokens: the
    engine is deliberately cache-heavy (a healthy session re-reads a large
    cached prompt prefix many times per turn), and cached input is roughly an
    order of magnitude cheaper than uncached. Raw token counts would flag
    exactly the sessions the architecture makes cheap.

    Dollar-denominated budgets later become a per-model weights (pricing)
    table — the same mechanism, different coefficients."""

    uncached_input: float = 1.0
    cached_input: float = 0.1
    output: float = 4.0

    model_multipliers: Mapping[str, float] = field(default_factory=dict)
    """Optional per-model scaling (e.g. a large model costing 10x a nano one)."""

    def weighted_cost(self, model: str, usage: UsageInfo) -> float:
        # Adapters report input_tokens INCLUSIVE of cached_input_tokens.
        cached = min(usage.cached_input_tokens, usage.input_tokens)
        uncached = usage.input_tokens - cached

        units = (
            uncached * self.uncached_input
            + cached * self.cached_input
            + usage.output_tokens * self.output
        )

        return units * self.model_multipliers.get(model, 1.0)


class SlidingCostWindow:
    """A decaying accumulator of cost units: entries older than the span fall
    out. Rate-based (units within the window), which is what makes
    "cool down, then recover" semantics meaningful for enforcement policies —
    and keeps advisory warnings tied to *current* behavior rather than session
    lifetime totals."""

    def __init__(self, span_seconds: float) -> None:
        self._span_seconds = span_seconds
        self._entries: deque[tuple[float, float]] = deque()
        self._total = 0.0

    def add(self, at: float, units: float) -> None:
        self._entries.append((at, units))
        self._total += units
        self._evict(at)

    def total(self, now: float) -> float:
        self._evict(now)
        return self._total

    def _evict(self, now: float) -> None:
        cutoff = now - self._span_seconds
        while self._entries and self._entries[0][0] <= cutoff:
            _, units = self._entries.popleft()
            self._total -= units


class CostControlPolicy(ABC):
    """Decides whether chargeable NLP work may proceed, and observes its cost.

    Interface contracts (binding on ALL implementations):

    - ``check`` FAILS OPEN: if it raises, the engine logs and proceeds with the
      work. Availability beats enforcement when small overages are tolerable by
      design. Implementations needing fail-closed semantics must handle their
      own failures internally and return a denial.
    - ``report`` is NON-BLOCKING: it is called on the hot path after provider
      calls and must not add latency. Counters are eventually consistent;
      implementations that persist state should queue internally.
    - The trace→context association is established by ``check``: the engine's
      TURN/BACKGROUND checks carry the full context, letting the policy bind the
      current trace to it. Usage reported for traces the policy has never seen
      should be accounted to an "unattributed" bucket, not dropped.
    """

    @abstractmethod
    async def check(self, context: CostContext, work: WorkKind) -> CostVerdict: ...

    @abstractmethod
    def report(self, trace_id: str, model: str, usage: UsageInfo) -> None: ...


class WindowedCostControlPolicy(CostControlPolicy):
    """Shared accounting base for window-based policies.

    Handles the mechanics every windowed policy needs: self-subscription to the
    usage reporter, trace→session binding (established by `check` per the port
    contract), per-session decaying windows of weighted cost units, and the
    unattributed bucket. Subclasses implement `check` (calling
    `_observe_check` first) to turn the window level into a verdict — advisory
    warnings, denials, or anything else."""

    _MAX_TRACE_BINDINGS = 512
    _UNATTRIBUTED_KEY = "unattributed"

    def __init__(
        self,
        usage_reporter: UsageReporter,
        weights: Optional[CostWeights] = None,
        window_seconds: float = 300.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        # Self-subscribe to observed usage (late-bound so tests may shadow
        # `report`); no external wiring is needed anywhere.
        usage_reporter.add_listener(
            lambda trace_id, model, usage: self.report(trace_id, model, usage)
        )

        self._weights = weights or CostWeights()
        self._window_seconds = window_seconds
        self._clock = clock

        # Trace -> session bindings, established by `check` (LRU-capped: stale
        # traces age out; their late usage lands in the unattributed bucket).
        self._sessions_by_trace: OrderedDict[str, SessionId] = OrderedDict()
        self._windows: dict[str, SlidingCostWindow] = {}

    def report(self, trace_id: str, model: str, usage: UsageInfo) -> None:
        session_id = self._sessions_by_trace.get(trace_id)
        key = str(session_id) if session_id is not None else self._UNATTRIBUTED_KEY

        self._window(key).add(self._clock(), self._weights.weighted_cost(model, usage))

    def session_units(self, session_id: SessionId) -> float:
        """The session's current weighted cost within the window (visibility)."""
        return self._window(str(session_id)).total(self._clock())

    def unattributed_units(self) -> float:
        """Weighted cost observed on traces no check() ever bound (visibility)."""
        return self._window(self._UNATTRIBUTED_KEY).total(self._clock())

    def _observe_check(self, context: CostContext) -> float:
        """Bind the trace and return the session's current window level.
        Subclass `check` implementations call this first."""
        self._bind_trace(context.trace_id, context.session_id)
        return self._window(str(context.session_id)).total(self._clock())

    def _bind_trace(self, trace_id: str, session_id: SessionId) -> None:
        self._sessions_by_trace[trace_id] = session_id
        self._sessions_by_trace.move_to_end(trace_id)

        while len(self._sessions_by_trace) > self._MAX_TRACE_BINDINGS:
            self._sessions_by_trace.popitem(last=False)

    def _window(self, key: str) -> SlidingCostWindow:
        if key not in self._windows:
            self._windows[key] = SlidingCostWindow(self._window_seconds)
        return self._windows[key]


class AdvisoryCostControlPolicy(WindowedCostControlPolicy):
    """The built-in policy: real accounting and advisory warnings, never blocks.

    Maintains a decaying per-session window of weighted cost units, fed by
    observed usage. When a configured advisory threshold is crossed, `check`
    returns warnings — but always allows. The ability to deny is deliberately
    left to plugged-in implementations (which can reuse the same accounting
    base)."""

    def __init__(
        self,
        usage_reporter: UsageReporter,
        weights: Optional[CostWeights] = None,
        window_seconds: float = 300.0,
        advisory_threshold_units: Optional[float] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """``advisory_threshold_units`` is None by default: accounting always
        runs, warnings only when a threshold is configured."""
        super().__init__(usage_reporter, weights, window_seconds, clock)
        self._advisory_threshold_units = advisory_threshold_units

    async def check(self, context: CostContext, work: WorkKind) -> CostVerdict:
        units = self._observe_check(context)

        if self._advisory_threshold_units is None or units <= self._advisory_threshold_units:
            return CostVerdict(allowed=True)

        return CostVerdict(
            allowed=True,
            warnings=(
                f"Session {context.session_id} crossed the advisory cost threshold: "
                f"{units:.0f} weighted units in the last {self._window_seconds:.0f}s "
                f"(threshold: {self._advisory_threshold_units:.0f})",
            ),
            scope="session",
        )
