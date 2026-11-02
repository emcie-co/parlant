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
from dataclasses import dataclass, field
from functools import cached_property
from itertools import chain
from typing import Any, TypeAlias

from parlant.core.agents import Effort
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.compass.tracing import RuleDecision
from parlant.core.engines.rule_evaluation import RuleEvaluation
from parlant.core.engines.engine_context import EngineContext as _EngineContext
from parlant.core.engines.compass.turn_directives import TurnDirective
from parlant.core.engines.compass.hooks import CompassHooks
from parlant.core.engines.compass.optimization_policy import CompassOptimizationPolicy
from parlant.core.glossary import Term, TermId
from parlant.core.rules import Rule, RuleId
from parlant.core.common import Weight
from parlant.core.tools import Tool, ToolId


@dataclass(frozen=True)
class CompassDependencies:
    optimization_policy: CompassOptimizationPolicy
    # DI injects the bound CompassHooks singleton (verified: lagom resolves the
    # bound type even past the default); the default only serves direct
    # construction in tests.
    hooks: CompassHooks = field(default_factory=CompassHooks)


@dataclass(frozen=True)
class IterationState:
    """State of a single iteration in the response process"""

    matched_rules: list[RuleEvaluation]
    ruled_out: list[RuleEvaluation]
    resolved_rules: list[RuleEvaluation]
    executed_tools: list[ToolId]


@dataclass
class MatcherState:
    """Transient state shared between evaluation phases of one matcher call.

    ``pending_discovered_rules`` contains rules that fresh discovery
    judged relevant but that have not yet passed dependency resolution and
    durable admission. It lets later evaluation phases interpret candidates
    using discoveries made earlier in the same matcher call without pretending
    those discoveries are already part of the session's durable state.

    ``pending_matched_rules`` contains rules that an evaluation round
    this turn judged relevant (ranker, code evaluator, or distiller) but that
    have not yet been recorded as matches. It exists for the same reason as
    the discovery set — later rounds must see this turn's evidence — but
    covers the evidence recall cannot produce: a cross-referential rule
    deferred to the final round needs the rules that ranking caught even
    though embedding recall missed them (see matching/common.py: "Policy
    context").

    The matcher owns both sets' lifecycles: it clears them before a batch,
    populates them from positive discovery/evaluation results per round, and
    clears them after results are recorded or abandoned.
    """

    pending_discovered_rules: set[Rule] = field(default_factory=set)
    pending_matched_rules: set[Rule] = field(default_factory=set)
    # Authoritative per-rule decisions staged this turn; flushed as telemetry
    # events by the engine only after the behavioral workspace commits, so
    # telemetry never claims a selection a rolled-back turn discarded.
    pending_rule_decisions: list[RuleDecision] = field(default_factory=list)


# Default last-resort text for a message blocked by agent-side moderation. Kept here
# (a low-level module) so LoopJob, ResponseState, and the Responder can share it
# without an import cycle. The Responder exposes it as a DI-configurable override.
DEFAULT_AGENT_MODERATION_FALLBACK = (
    "I'm sorry, but I can't send my last response. Let me know how else I can help."
)


@dataclass
class ResponseState:
    agent_effort: Effort = Effort.MEDIUM
    agent_message_review: bool = False
    agent_tool_review: bool = False
    # Explicit instructions that apply only to this response. Unlike rule
    # matches, directives are not inferred, resolved, persisted, or handled.
    turn_directives: list[TurnDirective] = field(default_factory=list)
    ordinary_rule_matches: list[RuleEvaluation] = field(default_factory=list)
    tool_enabled_rule_matches: dict[RuleEvaluation, list[ToolId]] = field(default_factory=dict)
    # Rules that matched this turn but lost priority resolution (relational or
    # numeric). They stay in the discovered working set — and therefore remain
    # visible in the responder's full instruction set — so the responder
    # renders an explicit "do not apply this turn" note for them.
    turn_deprioritized_rules: list[Rule] = field(default_factory=list)
    # The session's matched-rule ledger: every rule that has turn-matched at
    # least once this session (verdict-level — recorded before priority
    # resolution, so deprioritized matches count). Dependencies are satisfied
    # from this ledger, never from mere discovery admission, and eviction from
    # the working set does not erase it.
    rules_ever_matched_in_session: set[RuleId] = field(default_factory=set)
    # Snapshot of the rule ids actually rendered into the system-wide
    # instructions when they were built at turn start. The discovered set can
    # grow mid-turn (e.g. a reevaluation match admits its rule), but the
    # system-wide message is never re-rendered mid-turn — so the step
    # instructions consult this snapshot, not the live discovered set, to know
    # whether a matched rule's content is visible to the model.
    rendered_rule_ids: set[RuleId] = field(default_factory=set)
    # Same idea for glossary terms: the ids rendered into the system-wide
    # glossary section at turn start. Tool results can admit terms mid-turn
    # (see GlossaryRecaller.recall_from_tool_results), and since the
    # system-wide message is never re-rendered mid-turn, the step instructions
    # render any terms beyond this snapshot in the live tail instead.
    rendered_term_ids: set[TermId] = field(default_factory=set)
    # tools the matched rules enabled this turn (described in the prompt)
    matched_tools: list[Tool] = field(default_factory=list)
    # per-turn relevance scores for candidate tools, computed before final selection
    tool_relevance_scores: dict[ToolId, float] = field(default_factory=dict)
    # final tool catalog offered to the model, capped and emitted by name for cache stability
    available_tools: list[Tool] = field(default_factory=list)
    tool_ids_by_name: dict[str, ToolId] = field(default_factory=dict)  # to run a tool by its name
    # Pairs of AVAILABLE tools declared as overlapping alternatives (via OVERLAP
    # tool relationships or ToolOverlap.ALWAYS), sorted for prompt-cache
    # stability. Empty when no overlap is configured — the responder renders
    # its overlapping-tools note only when this is non-empty.
    overlapping_tool_ids: list[tuple[ToolId, ToolId]] = field(default_factory=list)
    # initial tool catalog selected during fill(); update() logs only deltas from this baseline
    fill_available_tool_ids: set[ToolId] = field(default_factory=set)

    # The agent's reasoning from each step of the response loop so far this turn,
    # in order. The loop appends to it after every step; the matching components
    # (turn evaluators) feed it into their per-rule prompts so each step's
    # evaluation is aware of what the agent has already concluded. Empty on the
    # initial match (no steps have run yet).
    reasoning_steps: list[str] = field(default_factory=list)

    # The text of every message this turn delivered, in the order the customer saw
    # it — a tool preamble and the reply that follows are two entries. Recorded as
    # each bubble is finalized (so a streamed message is stored with its final text,
    # not its first chunk) and handed to the on_messages_emitted hook.
    emitted_messages: list[str] = field(default_factory=list)

    # Durable summary of session events before the latest compaction marker.
    # Empty means no compaction summary is active for this loaded interaction.
    session_summary: str = ""

    # Reviewer-provided replacement reasoning when pending tool calls would breach
    # policy. Empty means no breach was found, or the reviewer has not run yet.
    step_notes: str = ""

    # Reviewer-provided summary of what the agent still needs to do before
    # responding to the user. Empty means the reviewer has not run yet.
    todo: str = ""

    # Monotonic timestamp of when this turn's processing began (stamped by the
    # engine after context load). The loop reads it to report time-to-first-
    # message health samples; None outside a processed turn (e.g. warm-ups).
    turn_started_monotonic: float | None = None

    # Set when the loop has exhausted its attempts and is forcing a final message.
    # While true, the step prompt is stripped of the tool catalog, matched rules,
    # and reviewer TODO so the model stops attempting the task and simply explains to
    # the user why it couldn't help. Reset once the give-up step completes.
    giving_up: bool = False

    # Last-resort text emitted if even the forced give-up message is itself flagged
    # by agent-side moderation. Sourced from the Responder (DI-configurable) via the
    # LoopJob and stashed here at the start of the run.
    moderation_fallback: str = DEFAULT_AGENT_MODERATION_FALLBACK

    # Per-turn signals the matcher precomputes (once) so its per-rule strategy
    # selection can stay synchronous: rules that carry tools, rules that
    # participate in a dependency relationship (an attachment signal), and
    # dependency targets whose match verdict gates dependent rules (a strategy
    # signal).
    # Whether the turn evaluator would answer this turn's whole batch in one
    # model request. Resolved alongside the signals above, for the same reason:
    # the answer needs a router lookup, and strategy selection is synchronous.
    batched_rule_evaluation: bool = False
    rule_ids_with_tools: set[RuleId] = field(default_factory=set)
    rule_ids_with_dependencies: set[RuleId] = field(default_factory=set)
    rule_ids_with_dependents: set[RuleId] = field(default_factory=set)
    # Rules participating in a PRIORITY relationship (either side, directly or
    # via a group). Both sides need real match verdicts: the source's verdict
    # decides whether suppression fires, the target's whether there is
    # anything to suppress.
    rule_ids_with_priority_relationships: set[RuleId] = field(default_factory=set)

    # Tools attached to a rule's action (e.g. a journey condensed into a
    # rule carries the tools of its tool-using steps). Read by turn evaluators
    # to surface what each tool does; empty for rules without attached tools.
    tools_by_rule: dict[RuleId, set[tuple[ToolId, Tool]]] = field(default_factory=dict)

    # Glossary terms each evaluated rule depends on to be interpreted
    # correctly (top-k by the rule's own query; see Matcher._load_rule_terms).
    # Rendered with the rule in the per-rule prompt tails.
    terms_by_rule: dict[RuleId, list[Term]] = field(default_factory=dict)

    # The rules each evaluated rule depends on (its dependency targets, with
    # group endpoints resolved to member rules). Rendered with the rule in the
    # per-rule prompt tails so a dependent rule is judged in the context it
    # operates within, not in isolation. Computed by the matcher alongside the
    # dependency strategy signals; keyed by source rule, deterministic order.
    dependency_targets_by_rule: dict[RuleId, list[Rule]] = field(default_factory=dict)

    # The dependency targets that should actually be rendered when evaluating
    # each source rule. The matcher derives this per evaluation round from the
    # scheduler's effective evidence: normally only satisfied targets; when any
    # target remains indeterminate, all configured targets as a best-effort
    # fallback. An absent key means no scheduler snapshot was prepared (e.g. a
    # direct prompt build), which uses the same explicitly uncertain all-targets
    # fallback.
    dependency_prompt_targets_by_rule: dict[RuleId, list[Rule]] = field(default_factory=dict)
    indeterminate_dependency_context_rule_ids: set[RuleId] = field(default_factory=set)

    # TODO: Remove what isn't needed
    context_variables: list[tuple[ContextVariable, ContextVariableValue]] = field(
        default_factory=list
    )
    glossary_terms: set[Term] = field(default_factory=set)
    tool_events: list[EmittedEvent] = field(default_factory=list)
    prepared_to_respond: bool = False
    message_events: list[EmittedEvent] = field(default_factory=list)
    usable_rules: list[Rule] = field(default_factory=list)
    # Bare cross-referential rules (no attachments; see matching/common.py for
    # the terminology): always rendered in the system instructions like
    # LOW-weight rules, never discovered or ranked. Computed by the matcher in
    # preload from the full inventory — NOT from the discovered set — because
    # these rules deliberately bypass discovery.
    statically_included_rules: list[Rule] = field(default_factory=list)

    # Rules admitted on this or an earlier turn. Unlike MatcherState's pending
    # set, this is the durable session working set loaded from session metadata.
    discovered_rules: set[Rule] = field(default_factory=set)
    matcher: MatcherState = field(default_factory=MatcherState)
    early_evaluated_rule_ids: set[RuleId] = field(default_factory=set)
    early_rule_evaluations: list[RuleEvaluation] = field(default_factory=list)
    discovered_rule_evaluations: list[RuleEvaluation] = field(default_factory=list)
    matched_handler_rule_ids: set[RuleId] = field(default_factory=set)
    # Eviction ledger for discovered rules: {id: last event offset at eviction}.
    # A ledgered rule may only be readmitted by conversation that arrived
    # after its eviction (see Matcher.prune_discovered_rules).
    evicted_discovered_rules: dict[RuleId, int] = field(default_factory=dict)
    additional_canned_response_fields: dict[str, Any] = field(default_factory=dict)
    iterations: list[IterationState] = field(default_factory=list)
    # SDK-managed retrievers started during preparation. Compass cancels and
    # awaits any survivors before finalizing an abandoned workspace.
    background_tasks: set[asyncio.Task[Any]] = field(default_factory=set)

    @property
    def ordinary_rules(self) -> list[Rule]:
        return [gp.rule for gp in self.ordinary_rule_matches]

    @property
    def tool_enabled_rules(self) -> list[Rule]:
        return [gp.rule for gp in self.tool_enabled_rule_matches.keys()]

    @property
    def rules(self) -> list[Rule]:
        return self.ordinary_rules + self.tool_enabled_rules

    @cached_property
    def dynamic_effort_level(self) -> Effort:
        """Resolve effective effort from the agent default and matched rule effort levels."""
        efforts = [
            self.agent_effort,
            *(
                match.rule.effort
                for match in chain(
                    self.ordinary_rule_matches,
                    self.tool_enabled_rule_matches.keys(),
                )
                if match.rule.effort is not None
            ),
        ]

        return max(efforts)

    @property
    def model_route_qualifiers(self) -> dict[str, str]:
        return {"effort": self.dynamic_effort_level.value}

    @cached_property
    def has_matched_high_weight_rules(self) -> bool:
        return any(
            match.rule.weight == Weight.HIGH
            for match in chain(
                self.ordinary_rule_matches,
                self.tool_enabled_rule_matches.keys(),
            )
        )

    @cached_property
    def message_review_requested(self) -> bool:
        """Message review is requested declaratively: by the agent-level default
        or by any turn-matched rule. Monotonic — a rule can request, never veto."""
        return self.agent_message_review or any(
            match.rule.message_review
            for match in chain(
                self.ordinary_rule_matches,
                self.tool_enabled_rule_matches.keys(),
            )
        )

    @cached_property
    def tool_review_requested(self) -> bool:
        """Tool review requests from the agent default or any turn-matched rule.
        The loop additionally reviews consequential tool calls regardless."""
        return self.agent_tool_review or any(
            match.rule.tool_review
            for match in chain(
                self.ordinary_rule_matches,
                self.tool_enabled_rule_matches.keys(),
            )
        )

    def invalidate_cached_properties(self) -> None:
        self.__dict__.pop("dynamic_effort_level", None)
        self.__dict__.pop("has_matched_high_weight_rules", None)
        self.__dict__.pop("message_review_requested", None)
        self.__dict__.pop("tool_review_requested", None)


# Compass components see both their state and container-resolved dependencies.
EngineContext: TypeAlias = _EngineContext[ResponseState, CompassDependencies]
