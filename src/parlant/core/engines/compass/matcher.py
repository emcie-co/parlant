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

"""Prepares each turn's response state: which rules apply, which tools are
offered, and which rules are rendered without matching at all.

Terminology (full definitions in ``matching/common.py``):

- **Inventory / usable rules**: every enabled rule available this turn.
- **Discovered rules**: the session working set — rules admitted on this or an
  earlier turn, rendered with "use your judgment" guidance.
- **Cross-referential rule**: flagged via
  ``rule.metadata["references_other_policies"]`` — its meaning depends on the
  agent's other rules, so it can never be judged blindly.
- **Attachment**: anything that fires only on an official turn match (tools,
  dependencies, labels, triggered effort, review requests, handlers, code
  evaluators). Cross-referential rules WITH attachments are escalated to RANK
  and evaluated with their policy context; those WITHOUT ("bare") are
  **statically included** — always rendered alongside LOW-weight rules,
  never discovered or ranked.
"""

from collections import Counter, defaultdict
from collections.abc import Awaitable, Iterable, Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from io import StringIO
from itertools import chain
from typing import cast

from parlant.core.agents import Effort
from parlant.core.async_utils import safe_gather
from parlant.core.common import Weight, JSONSerializable, xxh3_checksum
from parlant.core.engines.rule_evaluation import RuleEvaluation
from parlant.core.engines.rule_evaluator_registry import RuleEvaluatorPhase, RuleEvaluatorRegistry
from parlant.core.engines.compass.matching.rule_function_evaluator import (
    RuleFunctionEvaluator,
)
from parlant.core.engines.compass.matching.glossary_recaller import GlossaryRecaller
from parlant.core.engines.compass.matching.evaluation_scheduler import (
    EvaluationPlan,
    EvaluationPreconditionState,
    EvaluationScheduler,
)
from parlant.core.engines.compass.matching.rule_evaluation import (
    RuleEvaluationResult,
    TurnEvaluator,
)
from parlant.core.engines.compass.matching.rule_discovery import (
    RuleDiscoverer,
    RuleDiscoveryResult,
)
from parlant.core.engines.compass.matching.rule_pruner import (
    RulePruner,
)
from parlant.core.engines.compass.matching.tool_selection import ToolSelector
from parlant.core.engines.compass.relational_resolver import (
    RelationalResolver,
    ResolutionKind,
    ResolvedEntity,
)
from parlant.core.engines.compass.context import (
    EngineContext,
)
from parlant.core.engines.compass.tracing import CompassTracer, RuleDecision
from parlant.core.engines.compass.variable_loader import VariableLoader
from parlant.core.entity_cq import EntityQueries
from parlant.core.glossary import TermId
from datetime import datetime, timezone

from parlant.core.rules import Rule, RuleId
from parlant.core.loggers import Logger
from parlant.core import metadata as metadata_utils
from parlant.core.relationships import (
    RelationshipEntity,
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.sessions import EventKind, EventSource, ToolEventData
from parlant.core.groups import GroupId
from parlant.core.tools import ToolId, ToolOverlap

# Memo for per-rule glossary-term lookups, keyed by (rule id, rule
# modified time, glossary inventory checksum) - so edits to either side recompute
# on next use, and the steady state is a dict hit (no embedding search).
_RULE_TERM_IDS: dict[tuple[RuleId, str, str], tuple[TermId, ...]] = {}
_DISCOVERED_RULE_IDS_PATH = ("compass", "discovered_rule_ids")
# The session's matched-rule ledger: every rule that has turn-matched at least
# once this session (verdict-level — recorded before priority resolution).
# Dependencies are satisfied by THIS ledger, not by discovery admission.
_RULES_EVER_MATCHED_IN_SESSION_PATH = ("compass", "rules_ever_matched_in_session")
# Eviction ledger: {rule id: last event offset at eviction}. A ledgered
# rule may only be readmitted by conversation that arrives AFTER its
# eviction (honored by the RuleDiscoverer via state.evicted_discovered_rules).
_EVICTED_DISCOVERED_RULES_PATH = ("compass", "evicted_discovered_rules")
# The event offset at which the last pruning ran — the pruning rate limiter's
# reference point.
_LAST_DISCOVERED_RULE_PRUNING_OFFSET_PATH = (
    "compass",
    "last_discovered_rule_pruning_offset",
)


class _ContextUsage(Enum):
    """How to use a rule within the engine's context"""

    ADMIT_TO_DISCOVERED = auto()
    MATCH_CURRENT_TURN = auto()


class _MatcherStrategy(IntEnum):
    """How much effort to spend deciding whether a rule applies, cheapest to
    most thorough."""

    NONE = auto()
    RECALL = auto()
    RANK = auto()


@dataclass(frozen=True)
class _EvaluationRoundResult:
    code_rules: Sequence[Rule]
    code_evaluations: Sequence[RuleEvaluation]
    evaluator_results: Sequence[tuple[TurnEvaluator, Sequence[Rule], RuleEvaluationResult]]
    discovery: RuleDiscoveryResult | None


class Matcher:
    """Prepares the turn's response state: which rules apply and which tools
    are offered to the model.

    ``fill`` does the initial preparation; ``update`` refreshes it after a step
    (reevaluating rules gated on the tools that just ran). Both leave
    ``context.state`` carrying the matched rules, the matched rules'
    tools, and the offered tool catalog.
    """

    # Glossary terms fetched per evaluated rule (top-k by the rule's
    # own query) so each per-rule evaluation can interpret the terms the
    # rule depends on.
    _TERMS_PER_RULE = 10

    # Discovered rule cap: pruning fires only above the high-water mark and
    # evicts down to the target, so evictions (responder-prefix cache misses) are
    # batched and rare rather than continuous.
    _DISCOVERED_RULES_HIGH_WATER_MARK = 30
    _DISCOVERED_RULES_TARGET = 20
    # And at most once per this many customer messages, whatever the set size.
    _MIN_CUSTOMER_MESSAGES_BETWEEN_PRUNINGS = 10

    # At or below this many usable rules, discovery has nothing to decide: the
    # whole inventory fits in the responder's instructions, so a rule the
    # recaller rejected would be rendered anyway. Such an inventory is admitted
    # wholesale instead (see `_get_strategy`), sparing the embedding round trip.
    _SMALL_INVENTORY_THRESHOLD = 5

    def __init__(
        self,
        logger: Logger,
        rule_discoverer: RuleDiscoverer,
        tool_selector: ToolSelector,
        rule_evaluator: TurnEvaluator,
        rule_function_evaluator: RuleFunctionEvaluator,
        rule_pruner: RulePruner,
        glossary_recaller: GlossaryRecaller,
        evaluation_scheduler: EvaluationScheduler,
        relational_resolver: RelationalResolver,
        evaluator_registry: RuleEvaluatorRegistry,
        relationship_store: RelationshipStore,
        entity_queries: EntityQueries,
        variable_loader: VariableLoader,
    ) -> None:
        self._logger = logger
        self._rule_discoverer = rule_discoverer
        self._tool_selector = tool_selector
        self._rule_evaluator = rule_evaluator
        self._rule_function_evaluator = rule_function_evaluator
        self._rule_pruner = rule_pruner
        self._glossary_recaller = glossary_recaller
        self._evaluation_scheduler = evaluation_scheduler
        self._relational_resolver = relational_resolver
        self._evaluator_registry = evaluator_registry
        self._relationship_store = relationship_store
        self._entity_queries = entity_queries
        self._variable_loader = variable_loader

    async def preload(self, context: EngineContext) -> None:
        # Load the shared prompt/matching inputs that matcher-owned preparation
        # controls before matching and cache warm-up.
        with self._logger.scope("Matcher"):
            await self._preload(context)

    async def _preload(self, context: EngineContext) -> None:
        context.state.context_variables, rules = await safe_gather(
            self._variable_loader.load(context),
            self._entity_queries.find_rules_for_context(context.agent.id),
        )

        context.state.usable_rules = list(rules)

        compass_tracer = CompassTracer(context.tracer)
        compass_tracer.rules_loaded(context.state.usable_rules)
        compass_tracer.context_variables_loaded(context.state.context_variables)

        await self._load_discovered_rules(context)
        await self._load_static_inclusion(context)

    async def match_early(self, context: EngineContext) -> None:
        """Evaluate deterministic budget predicates before chargeable work."""
        rules = await self._entity_queries.find_rules_for_context(context.agent.id)
        context.state.usable_rules = list(rules)
        CompassTracer(context.tracer).rules_loaded(context.state.usable_rules)
        candidates = [
            g for g in rules if self._evaluator_registry.phase(g.id) == RuleEvaluatorPhase.EARLY
        ]
        context.state.early_evaluated_rule_ids = {g.id for g in candidates}

        if not candidates:
            return

        await self._load_discovered_rules(context)
        evaluations = await self._rule_function_evaluator.evaluate(context, candidates)
        context.state.matcher.pending_rule_decisions.extend(
            RuleDecision(rule=evaluation.rule, selected=False, rationale=evaluation.reasoning)
            for evaluation in evaluations
            if not evaluation.is_relevant
        )
        # Budget initialization may have updated usage metadata.
        context.session = await self._entity_queries.read_session(context.session.id)
        filtered = await self._resolve_admission_relationships(
            context,
            [
                (evaluation, _ContextUsage.MATCH_CURRENT_TURN)
                for evaluation in evaluations
                if evaluation.is_relevant
            ],
        )
        context.state.early_rule_evaluations = [evaluation for evaluation, _ in filtered]
        await self._record(
            context,
            filtered,
            append=False,
        )

    async def fill(self, context: EngineContext) -> None:
        """Initial preparation: match all usable rules, score the agent's tool
        pool, and load the relevant glossary (all independent, so in parallel), then
        select the offered tools."""
        with self._logger.scope("Matcher"), context.tracer.span("compass.matcher.fill"):
            await self._fill(context)

    async def _fill(self, context: EngineContext) -> None:
        await safe_gather(
            self._match(context),
            self._tool_selector.prepare(context),
            self._load_glossary(context),
        )
        await self._select_tools(context, log_delta_from_fill=False)

    async def update(self, context: EngineContext) -> None:
        """Refresh after a step: reevaluate rules gated on the tools that
        just ran, inject any rules the tools returned, then re-select tools.
        Tool relevance depends on the unchanged conversation, so it isn't
        rescored."""
        with self._logger.scope("Matcher"), context.tracer.span("compass.matcher.update"):
            await self._update(context)

    async def _update(self, context: EngineContext) -> None:
        await self._reevaluate(context)
        await self._inject_tool_returned_rules(context)
        # Tool results can introduce glossary terms mid-turn ("Terry offers
        # leaf"); admit them here (lexical-only — see GlossaryRecaller) so the
        # response step that consumes these results can define them.
        await self._glossary_recaller.recall_from_tool_results(context)
        await self._select_tools(context, log_delta_from_fill=True)

    async def _inject_tool_returned_rules(self, context: EngineContext) -> None:
        """Inject rules returned by this turn's tool results as turn matches.

        A tool can return ``TransientRule`` payloads ("do X for this
        response", optionally with a priority) in ``ToolResult(rules=[...])``.
        They exist only in this turn's state: never stored, admitted, or
        ledgered (durable recording filters the ``transient`` metadata flag),
        but they compete in priority resolution like any other turn match and
        render in full in the step reminders (they're never part of the
        system-wide prompt). Ids are derived deterministically from the staged
        tool events, so repeated post-step refreshes stay idempotent."""
        recorded_ids = {
            match.rule.id
            for match in chain(
                context.state.ordinary_rule_matches,
                context.state.tool_enabled_rule_matches,
            )
        }

        evaluations: list[tuple[RuleEvaluation, _ContextUsage]] = []
        rule_index = 0

        for event in context.state.tool_events:
            if event.kind != EventKind.TOOL:
                continue

            for tool_call in cast(ToolEventData, event.data)["tool_calls"]:
                for payload in tool_call["result"].get("rules", []):
                    rule_index += 1
                    rule_id = RuleId(f"<tool-rule-{rule_index}>")

                    if rule_id in recorded_ids:
                        continue

                    now = datetime.now(timezone.utc)
                    rule = Rule(
                        id=rule_id,
                        creation_utc=now,
                        modified_utc=now,
                        content=payload["content"],
                        condition=payload.get("condition") or None,
                        enabled=True,
                        groups=[],
                        metadata={},
                        weight=Weight(payload["weight"]) if "weight" in payload else Weight.MEDIUM,
                        priority=payload.get("priority", 0),
                    )
                    evaluations.append(
                        (
                            RuleEvaluation(
                                rule=rule,
                                reasoning=f"Returned by tool '{tool_call['tool_id']}'",
                                metadata={"transient": True},
                            ),
                            _ContextUsage.MATCH_CURRENT_TURN,
                        )
                    )

        if evaluations:
            await self._record(context, evaluations, append=True)

    async def warm_up(self, context: EngineContext) -> None:
        """Warm only the evaluators the current strategy would actually select."""
        evaluators = await self._get_prefill_targets(context)

        if not evaluators:
            return

        # The glossary is part of the cached shared prefix, so it must be loaded before
        # warming — otherwise the warmed prefix omits it and the first real turn (which
        # has loaded it) misses. At end-of-turn it's already loaded by `fill`, so skip.
        if not context.state.glossary_terms:
            await self._load_glossary(context)

        await safe_gather(*(evaluator.warm_up(context) for evaluator in evaluators))

    async def _get_prefill_targets(self, context: EngineContext) -> list[TurnEvaluator]:
        """The turn evaluators whose fan-out is large enough to be worth warming.

        Prefill buys cheaper repeats by paying for one extra request up front,
        so it only pays off once an evaluator is about to issue several. Each
        evaluator is judged on its own rule count, never on the total.
        """
        if not context.state.usable_rules:
            return []

        minimum = context.dependencies.optimization_policy.get_min_prefill_fan_out()

        if minimum <= 0:
            return []

        # Dependency-target status is one of the signals that can make an
        # otherwise LOW-weight rule require a real verdict, so load it before
        # applying the LOW-weight exclusion.
        await self._load_strategy_choice_signals(context, context.state.usable_rules)

        rules = [
            g
            for g in context.state.usable_rules
            if (
                g.weight != Weight.LOW
                or self._check_if_raises_effort(context, g)
                or self._check_if_has_dependents(context, g)
                or self._check_if_has_priority_relationships(context, g)
            )
            and not self._check_if_statically_included(context, g)
            and self._evaluator_registry.get(g.id) is None
        ]

        if not rules:
            return []

        fan_out: list[tuple[TurnEvaluator, int]] = []
        for rule in rules:
            evaluator = self._select_evaluator(context, rule)

            if evaluator is None:
                continue

            for index, (candidate, count) in enumerate(fan_out):
                if candidate is evaluator:
                    fan_out[index] = (candidate, count + 1)
                    break
            else:
                fan_out.append((evaluator, 1))

        # An evaluator that answers its whole batch in one request issues no
        # fan-out for a warm prefix to serve, so warming it would buy nothing —
        # and OpenAI's Decisions endpoint does no prompt caching either way.
        # Filtered here rather than inside the evaluator so an empty result also
        # spares `warm_up` its glossary load.
        #
        # Note the coupling: should the ranker ever gain a fallback from a failed
        # decision to the per-rule fan-out, that fallback would run cold.
        return [
            evaluator
            for evaluator, count in fan_out
            if count >= minimum and not await evaluator.evaluates_batch_in_one_request(context)
        ]

    # --- rule matching ---

    async def _load_discovered_rules(self, context: EngineContext) -> None:
        rule_ids = self._read_discovered_rule_ids(context.session.metadata)

        rules_by_id = {g.id: g for g in context.state.usable_rules}

        context.state.discovered_rules = {
            rules_by_id[rule_id] for rule_id in rule_ids if rule_id in rules_by_id
        }
        context.state.rules_ever_matched_in_session = set(
            self._read_rules_ever_matched_in_session(context.session.metadata)
        )

        ledger = cast(
            Mapping[str, int],
            metadata_utils.internal_value(
                context.session.metadata, _EVICTED_DISCOVERED_RULES_PATH, {}
            ),
        )
        context.state.evicted_discovered_rules = {
            RuleId(rule_id): int(offset)
            for rule_id, offset in ledger.items()
            if RuleId(rule_id) in rules_by_id
        }

    async def _store_discovered_rules(
        self,
        context: EngineContext,
        evaluations: Sequence[tuple[RuleEvaluation, _ContextUsage]],
    ) -> None:
        rules_by_id = {g.id: g for g in context.state.usable_rules}
        previous_rule_ids = {rule.id for rule in context.state.discovered_rules}
        # Transient (tool-returned) rules are per-turn only: they never enter
        # the durable discovered set or session metadata.
        rule_ids = {
            *previous_rule_ids,
            *(
                evaluation.rule.id
                for evaluation, _ in evaluations
                if not evaluation.metadata.get("transient")
            ),
        }

        context.state.discovered_rules = {rules_by_id[gid] for gid in rule_ids}
        newly_discovered_ids = rule_ids - previous_rule_ids
        queued_ids: set[RuleId] = set()
        for evaluation, _ in evaluations:
            if evaluation.rule.id in newly_discovered_ids and evaluation.rule.id not in queued_ids:
                context.state.discovered_rule_evaluations.append(evaluation)
                queued_ids.add(evaluation.rule.id)

        # A rule (re)entering the discovered set leaves the eviction ledger — it
        # earned readmission with a fresh trigger.
        context.state.evicted_discovered_rules = {
            gid: offset
            for gid, offset in context.state.evicted_discovered_rules.items()
            if gid not in rule_ids
        }

        current_rule_ids = self._read_discovered_rule_ids(context.session.metadata)
        current_ledger = metadata_utils.internal_value(
            context.session.metadata,
            _EVICTED_DISCOVERED_RULES_PATH,
            {},
        )
        ledger = {
            str(gid): offset for gid, offset in context.state.evicted_discovered_rules.items()
        }

        if rule_ids == current_rule_ids and ledger == current_ledger:
            return

        workspace = context.session_metadata
        workspace.stage_rule_discovery([str(rule_id) for rule_id in rule_ids])
        context.session = workspace.session

    def _read_discovered_rule_ids(
        self,
        metadata: Mapping[str, JSONSerializable],
    ) -> set[RuleId]:
        last_known_state = set(
            cast(
                Iterable[str],
                metadata_utils.internal_value(metadata, _DISCOVERED_RULE_IDS_PATH, []),
            )
        )

        return {RuleId(rule_id) for rule_id in last_known_state}

    def _store_rules_ever_matched(
        self,
        context: EngineContext,
        verdict_match_ids: AbstractSet[RuleId],
    ) -> None:
        """Union this turn's verdict-level matches into the session's
        matched-rule ledger (state + staged session metadata)."""
        new_ids = set(verdict_match_ids) - context.state.rules_ever_matched_in_session
        if not new_ids:
            return

        context.state.rules_ever_matched_in_session |= new_ids

        workspace = context.session_metadata
        workspace.stage_rule_matches(sorted(str(rule_id) for rule_id in new_ids))
        context.session = workspace.session

    def _read_rules_ever_matched_in_session(
        self,
        metadata: Mapping[str, JSONSerializable],
    ) -> set[RuleId]:
        return {
            RuleId(rule_id)
            for rule_id in cast(
                Iterable[str],
                metadata_utils.internal_value(metadata, _RULES_EVER_MATCHED_IN_SESSION_PATH, []),
            )
        }

    async def prune_discovered_rules(self, context: EngineContext) -> None:
        """Cap the discovered rule set.

        When the set exceeds the high-water mark, ask the pruner which members are
        still relevant to the conversation going forward, and evict the stalest down
        to the target — exempting HIGH-weight members and this turn's matches.
        Runs post-response (finalize), so the responder-prefix cache miss the
        eviction causes lands between turns; rate-limited to once per
        ``_MIN_CUSTOMER_MESSAGES_BETWEEN_PRUNINGS`` customer messages. Best-effort:
        a pruning failure must never fail the turn.
        """
        try:
            members = context.state.discovered_rules

            if len(members) <= self._DISCOVERED_RULES_HIGH_WATER_MARK:
                return

            if not self._pruning_rate_limit_elapsed(context):
                return

            exempt_ids = {
                g.id for g in members if g.weight == Weight.HIGH
            } | self._matched_rule_ids(context)

            candidates = sorted(
                (g for g in members if g.id not in exempt_ids),
                key=lambda g: str(g.id),
            )

            if not candidates:
                return

            result = await self._rule_pruner.prune(context, candidates)

            # Land exactly at the target (subject to exemptions), whether the
            # pruner flagged too many members as stale or too few: evict the
            # lowest-scoring first, breaking ties by weight, then id.
            weight_rank = {
                Weight.LOW: 0,
                Weight.MEDIUM: 1,
                Weight.HIGH: 2,
            }
            # Defensive: never evict an exempt member, whatever the pruner returned.
            ranked = sorted(
                (c for c in result.pruned_rules if c.rule.id not in exempt_ids),
                key=lambda c: (
                    c.score,
                    weight_rank[c.rule.weight],
                    str(c.rule.id),
                ),
            )

            needed = len(members) - self._DISCOVERED_RULES_TARGET
            evicted = ranked[: min(needed, len(ranked))]

            if not evicted:
                return

            # Offsets are session-global and monotonic (they survive compaction),
            # so the interaction's last event marks "now" for the ledger and the
            # rate limiter alike.
            last_offset = max((e.offset for e in context.interaction.events), default=0)
            evicted_ids = {c.rule.id for c in evicted}

            context.state.discovered_rules = {g for g in members if g.id not in evicted_ids}
            context.state.evicted_discovered_rules = {
                **context.state.evicted_discovered_rules,
                **{rule_id: last_offset for rule_id in evicted_ids},
            }

            workspace = context.session_metadata
            workspace.stage_rule_pruning(
                [str(rule_id) for rule_id in evicted_ids],
                last_offset,
            )
            context.session = workspace.session

            context.state.invalidate_cached_properties()

            pruning_results = StringIO()
            for idx, pruned in enumerate(evicted, start=1):
                g = pruned.rule
                pruning_results.write(
                    f"### {idx} {g.title or g.condition or ''} [Score: {pruned.score:.2f}]\n"
                    f"    Reasoning: {pruned.reasoning.strip()}\n\n"
                )
            self._logger.debug(
                f"Evicted {len(evicted)} of {len(members)} discovered "
                f"rules (target {self._DISCOVERED_RULES_TARGET}):\n{pruning_results.getvalue()}"
            )
        except Exception as exc:
            self._logger.warning(f"Discovered rule pruning failed (continuing): {exc}")

    def _pruning_rate_limit_elapsed(self, context: EngineContext) -> bool:
        last_pruning_offset = metadata_utils.internal_value(
            context.session.metadata,
            _LAST_DISCOVERED_RULE_PRUNING_OFFSET_PATH,
        )

        if last_pruning_offset is None:
            return True

        new_customer_messages = sum(
            1
            for event in context.interaction.events
            if event.kind == EventKind.MESSAGE
            and event.source == EventSource.CUSTOMER
            and event.offset > cast(int, last_pruning_offset)
        )

        return new_customer_messages >= self._MIN_CUSTOMER_MESSAGES_BETWEEN_PRUNINGS

    async def _match(self, context: EngineContext) -> None:
        evaluations = await self._run_batches(context, context.state.usable_rules)
        await self._record(context, evaluations, append=False)

    async def _reevaluate(self, context: EngineContext) -> None:
        executed_tool_ids = self._executed_tool_ids(context)
        if not executed_tool_ids:
            return

        gated = await self._find_rules_gated_on_tools(context, executed_tool_ids)
        already_matched = self._matched_rule_ids(context)
        candidates = [g for g in gated if g.id not in already_matched]
        if not candidates:
            return

        evaluations = await self._run_batches(context, candidates)

        if not evaluations:
            context.state.matcher.pending_discovered_rules.clear()
            context.state.matcher.pending_matched_rules.clear()
            return

        await self._record(context, evaluations, append=True)

    def _select_evaluator(self, context: EngineContext, rule: Rule) -> TurnEvaluator | None:
        """The turn-evaluator selection seam.

        Returns the evaluator that should judge this rule for the current turn,
        or None when discovery alone suffices. Subclasses may override to route
        specific rules to their own evaluators; `_load_strategy_choice_signals`
        is guaranteed to have run for the rules being selected."""
        if self._get_strategy(context, rule) >= _MatcherStrategy.RANK:
            return self._rule_evaluator
        return None

    def _get_strategy(self, context: EngineContext, rule: Rule) -> _MatcherStrategy:
        strategy = _MatcherStrategy.RECALL

        match context.state.dynamic_effort_level:
            case Effort.MIN:
                match rule.weight:
                    case Weight.LOW:
                        strategy = _MatcherStrategy.RECALL
                    case Weight.MEDIUM:
                        strategy = _MatcherStrategy.RECALL
                    case Weight.HIGH:
                        strategy = _MatcherStrategy.RECALL
            case Effort.LOW:
                match rule.weight:
                    case Weight.LOW:
                        strategy = _MatcherStrategy.RECALL
                    case Weight.MEDIUM:
                        strategy = _MatcherStrategy.RECALL
                    case Weight.HIGH:
                        # Worth a real verdict as soon as the batch costs one
                        # request rather than one per rule.
                        strategy = (
                            _MatcherStrategy.RANK
                            if context.state.batched_rule_evaluation
                            else _MatcherStrategy.RECALL
                        )
            case Effort.MEDIUM:
                match rule.weight:
                    case Weight.LOW:
                        strategy = _MatcherStrategy.RECALL
                    case Weight.MEDIUM:
                        # Same trade as LOW/HIGH above: a batched verdict is
                        # cheap enough that a middling rule need not be left to
                        # embedding recall.
                        strategy = (
                            _MatcherStrategy.RANK
                            if context.state.batched_rule_evaluation
                            else _MatcherStrategy.RECALL
                        )
                    case Weight.HIGH:
                        strategy = _MatcherStrategy.RANK
            case Effort.HIGH:
                match rule.weight:
                    case Weight.LOW:
                        strategy = _MatcherStrategy.RECALL
                    case Weight.MEDIUM:
                        strategy = _MatcherStrategy.RANK
                    case Weight.HIGH:
                        strategy = _MatcherStrategy.RANK
            case Effort.MAX:
                match rule.weight:
                    case Weight.LOW:
                        strategy = _MatcherStrategy.RECALL
                    case Weight.MEDIUM:
                        strategy = _MatcherStrategy.RANK
                    case Weight.HIGH:
                        strategy = _MatcherStrategy.RANK

        if strategy < _MatcherStrategy.RANK:
            # There are some special conditions under which we want
            # to ensure a baseline of matching effort, since their
            # matching carries important implications.

            if rule.labels:
                # If the rule has labels, we want to ensure high-quality analytics,
                # so we should at least rank - not use embeddings.
                return _MatcherStrategy.RANK

            if self._check_if_has_dependents(context, rule):
                # If other rules depend on this rule, its verdict gates those
                # follow-ups, so we should at least rank - not use embeddings.
                return _MatcherStrategy.RANK

            if self._check_if_has_priority_relationships(context, rule):
                # Priority resolution operates on turn-match verdicts, so both
                # sides of a PRIORITY relationship need one: the source's
                # verdict decides whether suppression fires, the target's
                # whether there is anything to suppress.
                return _MatcherStrategy.RANK

            if context.state.dynamic_effort_level >= Effort.MEDIUM and self._check_if_has_tools(
                context, rule
            ):
                # If the rule has tools, it may be gating important interactions
                # with data and/or actions. At MEDIUM effort and above, we should
                # at least rank rather than rely only on embeddings; MIN and LOW
                # deliberately retain the ordinary strategy to control cost.
                return _MatcherStrategy.RANK

            if self._check_if_raises_effort(context, rule):
                # If a matching rule would raise the turn's effort level,
                # it must be matched more thoroughly
                return _MatcherStrategy.RANK

            if rule.message_review or rule.tool_review:
                # A review request only triggers from a turn MATCH, and recall
                # produces no per-rule verdict — a review-requesting rule left
                # at the recall tier would never trigger its review. The
                # request declares stakes; honor it with a real verdict.
                return _MatcherStrategy.RANK

            if context.dependencies.hooks.on_rule_matched_handlers.get(rule.id):
                # Match handlers (on_matched callbacks, canned-response field
                # providers) fire only from a turn-match verdict. Attaching
                # code to a match declares the match consequential — honor it
                # with a real verdict, same as a review request. Discovery
                # handlers deliberately do NOT escalate: on_discovered is
                # recall's native, cheap-tier event.
                return _MatcherStrategy.RANK

            if rule.metadata.get("references_other_policies") or metadata_utils.internal_value(
                rule.metadata,
                ("references_other_policies",),
                False,
            ):
                # A cross-referential rule reaching strategy selection is
                # attached by construction (bare ones were filtered into
                # static inclusion before matching), and embedding recall is
                # structurally weakest on exactly its condition type
                # (complement sets have no stable embedding neighborhood).
                # Rank it — with its policy context (see rule_ranker.py and
                # matching/common.py).
                return _MatcherStrategy.RANK

            if metadata_utils.internal_value(rule.metadata, ("evaluation_method",)) in (
                "rank",
                "distill",
            ):
                # An explicit evaluation request (stamped by the SDK for
                # rule-attached retrievers, or set deliberately): the rule
                # needs a real per-turn verdict, whatever the matrix says.
                return _MatcherStrategy.RANK

            if len(context.state.usable_rules) <= self._SMALL_INVENTORY_THRESHOLD:
                # A tiny inventory is rendered in full regardless, so recalling it
                # buys nothing but latency. Keyed off the inventory, never the
                # discovery batch — the batch shrinks as rules join the discovered
                # set, so a large agent would otherwise slip into this path on a
                # later turn.
                return _MatcherStrategy.NONE

        return strategy

    async def _load_static_inclusion(self, context: EngineContext) -> None:
        """Compute the statically-included set: bare cross-referential rules.

        A cross-referential rule with no attachments (see matching/common.py
        for both terms) is not matched at all — it is rendered in the
        always-applied instructions every turn, like LOW-weight rules, and the
        responder judges it with the surrounding rules present in the same
        prompt. This sidesteps embedding recall entirely, which is
        structurally unreliable for complement-type conditions ("anything we
        don't support"), and costs nothing per turn.

        Computed in ``preload`` — before prompt warm-up — so the always-
        rendered set is identical between the warmed prefix and the live
        turn. Sourced from the full inventory, NOT the discovered set,
        because these rules deliberately bypass discovery."""
        flagged = [
            rule
            for rule in context.state.usable_rules
            if rule.metadata.get("references_other_policies")
            or metadata_utils.internal_value(rule.metadata, ("references_other_policies",), False)
        ]

        if not flagged:
            context.state.statically_included_rules = []
            return

        # Attachment signals (tools/dependencies) come from the stores; load
        # them for the flagged rules before classifying.
        await self._load_strategy_choice_signals(context, flagged)

        context.state.statically_included_rules = sorted(
            (rule for rule in flagged if not self._check_if_has_match_attachments(context, rule)),
            key=lambda rule: str(rule.id),
        )

    def _check_if_statically_included(self, context: EngineContext, rule: Rule) -> bool:
        return any(rule.id == included.id for included in context.state.statically_included_rules)

    def _check_if_has_match_attachments(self, context: EngineContext, rule: Rule) -> bool:
        """Whether anything mechanical fires only when this rule matches.

        Tools, dependencies, labels, triggered effort, a non-default priority,
        review requests, match or discovery handlers, and code evaluators all
        key off an official match verdict; a rule carrying any of them must go
        through matching so those attachments can actually trigger. Canned-response associations are a
        known v1 gap in this check (not cheaply attributable per rule here);
        such rules can be forced into matching by giving them any listed
        attachment or an explicit metadata override."""
        return bool(
            rule.labels
            or rule.effort is not None
            or rule.priority != 0
            or metadata_utils.internal_value(rule.metadata, ("evaluation_method",)) is not None
            or rule.message_review
            or rule.tool_review
            or self._check_if_has_tools(context, rule)
            or self._check_if_has_dependencies(context, rule)
            or self._check_if_has_priority_relationships(context, rule)
            or self._evaluator_registry.get(rule.id) is not None
            or context.dependencies.hooks.on_rule_matched_handlers.get(rule.id)
            or context.dependencies.hooks.on_rule_discovered_handlers.get(rule.id)
        )

    def _check_if_raises_effort(self, context: EngineContext, rule: Rule) -> bool:
        return rule.effort is not None and rule.effort > context.state.dynamic_effort_level

    async def _load_strategy_choice_signals(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
    ) -> None:
        associations = await self._entity_queries.find_rule_tool_associations()
        context.state.rule_ids_with_tools = {a.rule_id for a in associations}

        # One lookup per batch, never per rule: `_get_strategy` reads the answer
        # synchronously below, and the evaluator owns both the route it asks on
        # and whether this turn's effort could use a batched verdict at all.
        context.state.batched_rule_evaluation = (
            await self._rule_evaluator.evaluates_batch_in_one_request(context)
        )

        # Both endpoints count as dependency attachments: a source must not bypass
        # dependency enforcement through static inclusion, while a target's verdict
        # gates its dependents. Only targets get the RANK strategy floor; a source's
        # dependency can be enforced authoritatively after ordinary recall.
        #
        # Endpoints may be rules directly, or groups (GROUP_ALL/GROUP_ANY) standing
        # in for every rule that carries the group. Resolve them against the full
        # usable inventory, not merely the current evaluation batch: reevaluation
        # and early-evaluation paths intentionally pass partial batches, but their
        # model prompts still need the complete dependency context.
        dependency_relationships = [
            *await self._relationship_store.list_relationships(
                kind=RelationshipKind.DEPENDENCY, indirect=False
            ),
            *await self._relationship_store.list_relationships(
                kind=RelationshipKind.DEPENDENCY_ANY, indirect=False
            ),
        ]

        participating_rule_ids: set[RuleId] = set()
        participating_groups: set[GroupId] = set()
        target_rule_ids: set[RuleId] = set()
        target_groups: set[GroupId] = set()
        for relationship in dependency_relationships:
            for endpoint in (relationship.source, relationship.target):
                if endpoint.kind == RelationshipEntityKind.RULE:
                    participating_rule_ids.add(cast(RuleId, endpoint.id))
                elif endpoint.kind.is_group:
                    participating_groups.add(cast(GroupId, endpoint.id))

            if relationship.target.kind == RelationshipEntityKind.RULE:
                target_rule_ids.add(cast(RuleId, relationship.target.id))
            elif relationship.target.kind.is_group:
                target_groups.add(cast(GroupId, relationship.target.id))

        inventory_rules = context.state.usable_rules or list(rules)

        context.state.rule_ids_with_dependencies = {
            rule.id
            for rule in inventory_rules
            if rule.id in participating_rule_ids or not participating_groups.isdisjoint(rule.groups)
        }
        context.state.rule_ids_with_dependents = {
            rule.id
            for rule in inventory_rules
            if rule.id in target_rule_ids or not target_groups.isdisjoint(rule.groups)
        }

        # Priority participants (either side, directly or via a group) need
        # real match verdicts for relational priority to resolve anything:
        # the source's verdict decides whether suppression fires, the
        # target's whether there is anything to suppress.
        priority_relationships = await self._relationship_store.list_relationships(
            kind=RelationshipKind.PRIORITY, indirect=False
        )

        priority_rule_ids: set[RuleId] = set()
        priority_groups: set[GroupId] = set()
        for relationship in priority_relationships:
            for endpoint in (relationship.source, relationship.target):
                if endpoint.kind == RelationshipEntityKind.RULE:
                    priority_rule_ids.add(cast(RuleId, endpoint.id))
                elif endpoint.kind.is_group:
                    priority_groups.add(cast(GroupId, endpoint.id))

        context.state.rule_ids_with_priority_relationships = {
            rule.id
            for rule in inventory_rules
            if rule.id in priority_rule_ids or not priority_groups.isdisjoint(rule.groups)
        }

        # Retain the source → targets mapping (not just the flat sets): a
        # dependent rule is evaluated with its targets' content rendered in
        # its prompt tail, so it is judged in the context it operates within.
        rules_by_id = {rule.id: rule for rule in inventory_rules}

        def resolve_endpoint(endpoint: RelationshipEntity) -> list[Rule]:
            if endpoint.kind == RelationshipEntityKind.RULE:
                resolved = rules_by_id.get(cast(RuleId, endpoint.id))
                return [resolved] if resolved else []
            if endpoint.kind.is_group:
                return [
                    rule for rule in inventory_rules if cast(GroupId, endpoint.id) in rule.groups
                ]
            return []

        targets_by_rule: dict[RuleId, dict[RuleId, Rule]] = {}
        for relationship in dependency_relationships:
            for source in resolve_endpoint(relationship.source):
                for target in resolve_endpoint(relationship.target):
                    if target.id != source.id:
                        targets_by_rule.setdefault(source.id, {})[target.id] = target

        context.state.dependency_targets_by_rule = {
            source_id: sorted(targets.values(), key=lambda rule: str(rule.id))
            for source_id, targets in targets_by_rule.items()
        }
        context.state.dependency_prompt_targets_by_rule.clear()
        context.state.indeterminate_dependency_context_rule_ids.clear()

    def _prepare_dependency_prompt_context(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
        effective_states: Mapping[RuleId, EvaluationPreconditionState],
    ) -> None:
        """Select dependency targets to render for this evaluation round.

        Determinate evidence renders only satisfied targets. If any target is
        still unknown, preserve the scheduler's best-effort fallback by
        rendering every configured target and warning once for that dependent
        evaluation. Known-unsatisfied alternatives are therefore omitted only
        when the complete target set is determinate.
        """
        for rule in rules:
            targets = context.state.dependency_targets_by_rule.get(rule.id, [])
            if not targets:
                continue

            indeterminate_targets = [
                target
                for target in targets
                if effective_states.get(target.id, EvaluationPreconditionState.UNKNOWN)
                == EvaluationPreconditionState.UNKNOWN
            ]
            if indeterminate_targets:
                context.state.dependency_prompt_targets_by_rule[rule.id] = list(targets)
                context.state.indeterminate_dependency_context_rule_ids.add(rule.id)
                self._logger.warning(
                    f"Evaluating dependent rule {rule.id} with indeterminate dependency "
                    f"target(s): {', '.join(str(target.id) for target in indeterminate_targets)}; "
                    "rendering all configured dependency targets as fallback context"
                )
                continue

            context.state.dependency_prompt_targets_by_rule[rule.id] = [
                target
                for target in targets
                if effective_states.get(target.id) == EvaluationPreconditionState.SATISFIED
            ]
            context.state.indeterminate_dependency_context_rule_ids.discard(rule.id)

    def _check_if_has_tools(self, context: EngineContext, rule: Rule) -> bool:
        return rule.id in context.state.rule_ids_with_tools

    def _check_if_has_dependencies(self, context: EngineContext, rule: Rule) -> bool:
        return rule.id in context.state.rule_ids_with_dependencies

    def _check_if_has_dependents(self, context: EngineContext, rule: Rule) -> bool:
        return rule.id in context.state.rule_ids_with_dependents

    def _check_if_has_priority_relationships(self, context: EngineContext, rule: Rule) -> bool:
        return rule.id in context.state.rule_ids_with_priority_relationships

    async def _run_batches(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
    ) -> Sequence[tuple[RuleEvaluation, _ContextUsage]]:
        """Evaluate a matcher batch, potentially over several ordered rounds.

        A rule with a code (Python) matcher always goes to the function
        evaluator, regardless of strategy: an explicit evaluator is
        authoritative. The rest are bucketed by ``_get_strategy``.

        The control flow is:

        1. Route rules to code evaluation, discovery, direct admission, and/or a
           turn evaluator.
        2. Count every source expected to produce relevance evidence for each
           rule and ask ``EvaluationScheduler`` for a bounded plan.
        3. Run all work selected for one round concurrently. Discovery always
           participates in the first round so its results can influence later
           rounds.
        4. Expose positively discovered rules through
           ``MatcherState.pending_discovered_rules`` before starting
           the next round.
        5. Merge the evidence, enforce dependencies authoritatively, and return
           the surviving turn matches and durable-admission candidates.

        The pending-admission set is intentionally not the durable discovered
        set. Later phases may use it as fresh context, but a rule only becomes
        durable in ``_record`` after dependency filtering succeeds.
        """
        # A MatcherState belongs to the response, which can outlive one call to
        # this method during mid-turn rematching. Never let provisional evidence
        # from an earlier call leak into a new plan.
        context.state.matcher.pending_discovered_rules.clear()
        context.state.matcher.pending_matched_rules.clear()

        early_evaluations = [
            (evaluation, _ContextUsage.MATCH_CURRENT_TURN)
            for evaluation in context.state.early_rule_evaluations
            if evaluation.rule.id in {rule.id for rule in rules}
        ]

        if not rules:
            return early_evaluations

        # Load against the original batch before filtering LOW and early rules:
        # a LOW dependency target earns the RANK floor, and an early target's
        # content must remain available to a dependent model prompt.
        await self._load_strategy_choice_signals(context, rules)

        rules = [rule for rule in rules if rule.id not in context.state.early_evaluated_rule_ids]

        # Low-weight rules are always included in the system instructions,
        # unless they carry an effort override that can affect the current turn
        # or their verdict gates a dependent rule.
        rules = [
            g
            for g in rules
            if (
                g.weight != Weight.LOW
                or self._check_if_raises_effort(context, g)
                or self._check_if_has_dependents(context, g)
                or self._check_if_has_priority_relationships(context, g)
            )
            # Bare cross-referential rules are statically included in the
            # always-rendered instructions instead of being matched at all
            # (see matching/common.py: "Attachment").
            and not self._check_if_statically_included(context, g)
        ]

        if not rules:
            return early_evaluations

        code_batch: list[Rule] = []
        discovery_batch: list[Rule] = []
        # Rules admitted to the discovered set without any relevance work at all
        # (_MatcherStrategy.NONE).
        admission_batch: list[Rule] = []
        # Rules grouped by their selected turn evaluator — the evaluator is
        # natively multi-evaluator; subclasses may route rules to their own
        # evaluators via `_select_evaluator`.
        evaluator_by_rule_id: dict[RuleId, TurnEvaluator] = {}

        for rule in rules:
            if self._evaluator_registry.get(rule.id) is not None:
                code_batch.append(rule)
                continue

            if self._get_strategy(context, rule) == _MatcherStrategy.NONE:
                # Admitted outright, with no relevance work spent on it. Only
                # discovery is skipped — the `_select_evaluator` seam below still
                # runs, so a subclass routing this rule to its own evaluator is
                # honored (the base matcher never selects one at this tier).
                if rule not in context.state.discovered_rules:
                    admission_batch.append(rule)
            elif rule not in context.state.discovered_rules:
                # Discovery runs for every rule not already in the discovered working set,
                # regardless of whether a turn evaluator also judges it.
                discovery_batch.append(rule)

            if (evaluator := self._select_evaluator(context, rule)) is not None:
                evaluator_by_rule_id[rule.id] = evaluator

        code_rule_ids = {rule.id for rule in code_batch}
        evaluator_candidates = [
            rule for rule in rules if rule.id in code_rule_ids or rule.id in evaluator_by_rule_id
        ]

        # Recall, code evaluation, and a model evaluator can all judge the same
        # rule. A negative result is only conclusive after all expected sources
        # report, so the plan receives a count rather than a boolean.
        pending_evidence_sources: Counter[RuleId] = Counter()
        pending_evidence_sources.update(rule.id for rule in code_batch)
        pending_evidence_sources.update(evaluator_by_rule_id.keys())
        pending_evidence_sources.update(rule.id for rule in discovery_batch)

        # Durable discoveries and early deterministic matches are authoritative
        # prior evidence. Direct admissions are merely relevant: they still
        # need their own scheduling preconditions to become effective.
        initially_satisfied_rule_ids = {
            *(rule.id for rule in context.state.discovered_rules),
            *(evaluation.rule.id for evaluation in context.state.early_rule_evaluations),
        }
        plan = await self._evaluation_scheduler.create_plan(
            context,
            candidates=evaluator_candidates,
            rules=context.state.usable_rules,
            pending_evidence_sources=pending_evidence_sources,
            initially_satisfied_rule_ids=initially_satisfied_rule_ids,
            initially_relevant_rule_ids={rule.id for rule in admission_batch},
        )

        code_evaluations: list[RuleEvaluation] = []
        evaluator_results: list[tuple[TurnEvaluator, Sequence[Rule], RuleEvaluationResult]] = []
        discovered = RuleDiscoveryResult(discovered_rules=[], duration=0.0)
        round_number = 0

        while plan.can_start_round or (round_number == 0 and discovery_batch):
            round_number += 1
            round_rules = plan.next_round() if plan.can_start_round else []
            self._prepare_dependency_prompt_context(
                context,
                [rule for rule in round_rules if rule.id in evaluator_by_rule_id],
                plan.effective_rule_states(),
            )
            if plan.enabled:
                self._log_evaluation_round(round_number, round_rules)

            round_result = await self._run_evaluation_round(
                context,
                round_rules,
                code_rule_ids=code_rule_ids,
                evaluator_by_rule_id=evaluator_by_rule_id,
                discovery_batch=discovery_batch if round_number == 1 else None,
            )
            code_evaluations.extend(round_result.code_evaluations)
            evaluator_results.extend(round_result.evaluator_results)

            # Make this round's positive verdicts visible to later rounds: a
            # deferred cross-referential rule's policy context must include
            # the rules that ranking caught even where recall missed them —
            # the embedding blind spot is the very case it serves.
            context.state.matcher.pending_matched_rules.update(
                evaluation.rule
                for evaluation in round_result.code_evaluations
                if evaluation.is_relevant
            )
            context.state.matcher.pending_matched_rules.update(
                evaluation.rule
                for _, _, result in round_result.evaluator_results
                for evaluation in result.evaluations
                if evaluation.is_relevant
            )
            if round_result.discovery is not None:
                discovered = round_result.discovery

                # Make first-round recall visible immediately to deferred
                # evaluators. This state remains provisional until the final
                # dependency filter admits the corresponding evaluations.
                context.state.matcher.pending_discovered_rules.update(
                    result.rule for result in discovered.discovered_rules if result.is_relevant
                )

            self._observe_evaluation_round(
                plan,
                round_result,
                discovery_batch=discovery_batch if round_number == 1 else None,
            )

        if plan.enabled and (skipped_rules := plan.skipped_rules()):
            self._logger.debug(
                f"Multi-phase evaluation skipped "
                f"{len(skipped_rules)} rule(s) with unsatisfied preconditions: "
                f"{', '.join(str(rule.id) for rule in skipped_rules)}"
            )

        discovery_results = StringIO()

        if discovered.discovered_rules:
            discovery_results.write(f"Duration: {discovered.duration:.3f}s\n\n")

            for idx, discovery_result in enumerate(discovered.discovered_rules, start=1):
                g = discovery_result.rule

                discovery_results.write(
                    f"### {idx} {g.title or ''} [Score: {discovery_result.score:.2f} ({'Relevant' if discovery_result.is_relevant else 'Not Relevant'})]\n\n"
                )
                if g.condition:
                    discovery_results.write(f"    Condition: {g.condition}\n")
                if g.content:
                    discovery_results.write(f"    Description: {g.content.strip()}\n")
                discovery_results.write("\n")

            self._logger.debug(f"Rule discovery results:\n{discovery_results.getvalue()}")

        for evaluator, _, evaluated in evaluator_results:
            self._log_evaluations(type(evaluator).__name__, evaluated)

        context.state.matcher.pending_rule_decisions.extend(
            RuleDecision(rule=evaluation.rule, selected=False, rationale=evaluation.reasoning)
            for evaluation in code_evaluations
            if not evaluation.is_relevant
        )
        context.state.matcher.pending_rule_decisions.extend(
            RuleDecision(rule=evaluation.rule, selected=False, rationale=evaluation.reasoning)
            for _, _, evaluated in evaluator_results
            for evaluation in evaluated.evaluations
            if not evaluation.is_relevant
        )

        evaluations: list[tuple[RuleEvaluation, _ContextUsage]] = list(early_evaluations)
        evaluations.extend(
            (evaluation, _ContextUsage.MATCH_CURRENT_TURN)
            for evaluation in code_evaluations
            if evaluation.is_relevant
        )

        evaluations += [
            (
                evaluation,
                _ContextUsage.MATCH_CURRENT_TURN,
            )
            for _, _, evaluated in evaluator_results
            for evaluation in evaluated.evaluations
            if evaluation.is_relevant
        ]

        turn_matched_rule_ids = {
            evaluation.rule.id
            for evaluation, context_usage in evaluations
            if context_usage == _ContextUsage.MATCH_CURRENT_TURN
        }

        evaluations += [
            (
                RuleEvaluation(
                    rule=rc.rule,
                    reasoning="This may or may not be relevant right now - use your judgment.",
                    is_relevant=True,
                    score=rc.score,
                ),
                _ContextUsage.ADMIT_TO_DISCOVERED,
            )
            for rc in discovered.discovered_rules
            if rc.is_relevant and rc.rule.id not in turn_matched_rule_ids
        ]

        evaluations += [
            (
                RuleEvaluation(
                    rule=rule,
                    reasoning="This may or may not be relevant right now - use your judgment.",
                    is_relevant=True,
                ),
                _ContextUsage.ADMIT_TO_DISCOVERED,
            )
            for rule in admission_batch
            if rule.id not in turn_matched_rule_ids
        ]

        return await self._resolve_admission_relationships(context, evaluations)

    async def _run_evaluation_round(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
        *,
        code_rule_ids: set[RuleId],
        evaluator_by_rule_id: Mapping[RuleId, TurnEvaluator],
        discovery_batch: Sequence[Rule] | None,
    ) -> _EvaluationRoundResult:
        """Run all evaluator families selected for one round concurrently.

        ``discovery_batch`` is provided only in the first round. Evaluator
        groups are formed by object identity so one evaluator can retain its
        native batch behavior while different evaluator implementations still
        run in parallel.

        This method executes work but does not interpret cross-evaluator
        evidence. ``_observe_evaluation_round`` performs that translation after
        all results are available.
        """
        code_rules = [rule for rule in rules if rule.id in code_rule_ids]
        evaluator_groups: list[tuple[TurnEvaluator, list[Rule]]] = []
        for rule in rules:
            if (evaluator := evaluator_by_rule_id.get(rule.id)) is None:
                continue
            for existing, group in evaluator_groups:
                if existing is evaluator:
                    group.append(rule)
                    break
            else:
                evaluator_groups.append((evaluator, [rule]))

        await self._load_rule_terms(
            context,
            [rule for _, group in evaluator_groups for rule in group],
        )

        tasks: list[Awaitable[object]] = []
        if code_rules:
            tasks.append(self._rule_function_evaluator.evaluate(context, code_rules))
        if discovery_batch is not None:
            tasks.append(self._discover(context, discovery_batch))
        tasks.extend(evaluator.evaluate(context, group) for evaluator, group in evaluator_groups)

        gathered = await safe_gather(*tasks)
        result_offset = 0
        code_evaluations: Sequence[RuleEvaluation] = []
        if code_rules:
            code_evaluations = cast(Sequence[RuleEvaluation], gathered[result_offset])
            result_offset += 1

        discovered: RuleDiscoveryResult | None = None
        if discovery_batch is not None:
            discovered = cast(RuleDiscoveryResult, gathered[result_offset])
            result_offset += 1

        evaluation_results = cast(Sequence[RuleEvaluationResult], gathered[result_offset:])
        return _EvaluationRoundResult(
            code_rules=code_rules,
            code_evaluations=code_evaluations,
            evaluator_results=[
                (evaluator, group, evaluated)
                for (evaluator, group), evaluated in zip(evaluator_groups, evaluation_results)
            ],
            discovery=discovered,
        )

    @staticmethod
    def _observe_evaluation_round(
        plan: EvaluationPlan,
        result: _EvaluationRoundResult,
        *,
        discovery_batch: Sequence[Rule] | None,
    ) -> None:
        """Report each independent evidence source back to the plan.

        A rule present in both discovery and a turn evaluator is observed
        twice. This exactly balances the source count supplied when the plan was
        created, preventing one negative source from prematurely rejecting it.
        """
        code_rule_ids = {rule.id for rule in result.code_rules}
        if code_rule_ids:
            plan.observe(
                checked_rule_ids=code_rule_ids,
                relevant_rule_ids={
                    evaluation.rule.id
                    for evaluation in result.code_evaluations
                    if evaluation.is_relevant
                },
            )

        for _, rules, evaluated in result.evaluator_results:
            plan.observe(
                checked_rule_ids={rule.id for rule in rules},
                relevant_rule_ids={
                    evaluation.rule.id
                    for evaluation in evaluated.evaluations
                    if evaluation.is_relevant
                },
            )

        if discovery_batch is not None and result.discovery is not None:
            plan.observe(
                checked_rule_ids={rule.id for rule in discovery_batch},
                relevant_rule_ids={
                    discovered.rule.id
                    for discovered in result.discovery.discovered_rules
                    if discovered.is_relevant
                },
            )

    def _log_evaluation_round(
        self,
        round_number: int,
        rules: Sequence[Rule],
    ) -> None:
        self._logger.debug(
            f"Multi-phase evaluation round {round_number}: "
            f"{len(rules)} rule(s)"
            + (f" ({', '.join(str(rule.id) for rule in rules)})" if rules else "")
        )

    async def _discover(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
    ) -> RuleDiscoveryResult:
        """Discovery on an empty batch is a foregone conclusion, and discoverers
        are not free to ask (the recaller embeds the conversation). Skip it."""
        if not rules:
            return RuleDiscoveryResult(discovered_rules=[], duration=0.0)

        return await self._rule_discoverer.discover(context, rules)

    async def _resolve_admission_relationships(
        self,
        context: EngineContext,
        evaluations: Sequence[tuple[RuleEvaluation, _ContextUsage]],
    ) -> Sequence[tuple[RuleEvaluation, _ContextUsage]]:
        """Expand entailments, then gate all evaluations (turn matches and
        discovered admissions alike) on their dependencies.

        Entailment expansion runs first so that an entailed rule is itself
        dependency-gated, and so it can satisfy the dependencies of other
        evaluations in the same pass. An entailed evaluation inherits the
        strongest usage among its entailing sources: it becomes a turn match
        when any source matched this turn, otherwise a discovered admission.

        A dependency is met by another match this turn (same-turn chains
        resolve in dependency order) or by the session's matched-rule ledger —
        the "has ever actually matched" latch. Discovery admission never
        satisfies a dependency: recall is deliberately generous, and letting
        its false positives unlock dependents would make depend_on()
        decorative. A rule whose dependency was never met stays out of the
        discovered set entirely, so it can keep being judged fresh every turn
        until its dependency arrives.

        Scheduler preconditions reduce unnecessary evaluation work, but this
        resolver remains authoritative. New scheduling prerequisite providers
        must not be relied on as the sole enforcement mechanism for a business
        invariant.
        """
        if not evaluations:
            return evaluations

        usage_by_id = {evaluation.rule.id: usage for evaluation, usage in evaluations}

        expanded = await self._relational_resolver.expand_entailment(
            context.state.usable_rules,
            [evaluation for evaluation, _ in evaluations],
        )

        entailed_pairs: list[tuple[RuleEvaluation, _ContextUsage]] = []
        for evaluation in expanded.evaluations:
            if evaluation.rule.id in usage_by_id:
                continue

            source_usages = {
                usage_by_id.get(RuleId(counterpart.entity.id))
                for resolution in expanded.resolutions.get(ResolvedEntity.rule(evaluation.rule), [])
                if resolution.kind == ResolutionKind.ENTAILED
                for counterpart in resolution.details.counterparts
                if counterpart.entity_type == "rule"
            }
            usage = (
                _ContextUsage.MATCH_CURRENT_TURN
                if _ContextUsage.MATCH_CURRENT_TURN in source_usages
                else _ContextUsage.ADMIT_TO_DISCOVERED
            )
            entailed_pairs.append((evaluation, usage))

        all_pairs = [*evaluations, *entailed_pairs]

        # Dependencies are satisfied by the matched-rule ledger ("has this
        # ever actually applied in the session?") plus this batch's turn
        # matches — never by discovery admission, whose embedding-level
        # false positives would make depend_on() decorative.
        result = await self._relational_resolver.filter_unmet_dependencies(
            context.state.usable_rules,
            [evaluation for evaluation, _ in all_pairs],
            satisfied_rule_ids=context.state.rules_ever_matched_in_session,
            same_turn_satisfier_ids={
                evaluation.rule.id
                for evaluation, usage in all_pairs
                if usage == _ContextUsage.MATCH_CURRENT_TURN
            },
        )

        surviving_ids = {evaluation.rule.id for evaluation in result.evaluations}

        context.state.matcher.pending_rule_decisions.extend(
            RuleDecision(
                rule=evaluation.rule,
                selected=False,
                rationale="Ruled out: the rule's dependencies have not matched in this session.",
            )
            for evaluation, usage in all_pairs
            if evaluation.rule.id not in surviving_ids
            and usage == _ContextUsage.MATCH_CURRENT_TURN
            and not evaluation.metadata.get("transient")
        )

        return [
            (evaluation, usage)
            for evaluation, usage in all_pairs
            if evaluation.rule.id in surviving_ids
        ]

    async def _resolve_turn_priority(
        self,
        context: EngineContext,
        evaluations: Sequence[tuple[RuleEvaluation, _ContextUsage]],
        *,
        append: bool,
    ) -> Sequence[tuple[RuleEvaluation, _ContextUsage]]:
        """Keep only the priority-winning matches in the turn-matched state;
        outranked turn matches are demoted to discovered admission (they were
        relevant — they just lost this turn).

        Relational priority (PRIORITY relationships) resolves first, then the
        numeric filter over the survivors. Entailed matches are exempt from the
        numeric bar — they owe their activation to their entailer and must not
        suppress it.

        On mid-turn updates (``append``), the already-recorded turn matches
        compete as well: a higher-priority rule matching mid-turn drops them
        from the turn state. That breaks the responder prompt's byte-identical
        prefix for the rest of the turn — accepted, since the match changes
        behavior just as strongly either way."""
        new_turn_matches = [
            evaluation
            for evaluation, usage in evaluations
            if usage == _ContextUsage.MATCH_CURRENT_TURN
        ]

        recorded: list[RuleEvaluation] = []
        if append:
            recorded = [
                *context.state.ordinary_rule_matches,
                *context.state.tool_enabled_rule_matches,
            ]

        if not new_turn_matches and not recorded:
            return evaluations

        competing = [*recorded, *new_turn_matches]

        relational = await self._relational_resolver.resolve_priority_relationships(
            context.state.usable_rules,
            competing,
        )

        entailed_rule_ids = {
            evaluation.rule.id for evaluation in competing if evaluation.metadata.get("entailed")
        }
        result = self._relational_resolver.keep_highest_priority(
            list(relational.evaluations),
            entailed_rule_ids=entailed_rule_ids,
        )
        winner_ids = {evaluation.rule.id for evaluation in result.evaluations}

        # Track this turn's priority losers so the responder can render an
        # explicit "do not apply this turn" note — they remain visible in its
        # full instruction set through the discovered working set.
        losers = {
            evaluation.rule.id: evaluation.rule
            for evaluation in competing
            if evaluation.rule.id not in winner_ids
        }
        if append:
            retained = {
                rule.id: rule
                for rule in context.state.turn_deprioritized_rules
                if rule.id not in winner_ids and rule.id not in losers
            }
            context.state.turn_deprioritized_rules = [*retained.values(), *losers.values()]
        else:
            context.state.turn_deprioritized_rules = list(losers.values())

        if append and recorded:
            context.state.ordinary_rule_matches = [
                match
                for match in context.state.ordinary_rule_matches
                if match.rule.id in winner_ids
            ]
            context.state.tool_enabled_rule_matches = {
                match: tools
                for match, tools in context.state.tool_enabled_rule_matches.items()
                if match.rule.id in winner_ids
            }

        return [
            (evaluation, usage)
            if usage != _ContextUsage.MATCH_CURRENT_TURN or evaluation.rule.id in winner_ids
            else (evaluation, _ContextUsage.ADMIT_TO_DISCOVERED)
            for evaluation, usage in evaluations
        ]

    def _log_evaluations(self, evaluator_name: str, evaluated: RuleEvaluationResult) -> None:
        results = StringIO()

        if evaluated.generation_info:
            results.write(f"Usage: {evaluated.generation_info}\n\n")

        if not evaluated.evaluations:
            return

        for idx, evaluation in enumerate(evaluated.evaluations, start=1):
            g = evaluation.rule

            score_text = f"Score: {evaluation.score:.2f} " if evaluation.score is not None else ""
            results.write(
                f"### {idx} {g.title or ''} [{score_text}({'Relevant' if evaluation.is_relevant else 'Not Relevant'})]\n\n"
            )
            if g.condition:
                results.write(f"    Condition: {g.condition}\n")
            if g.content:
                results.write(f"    Description: {g.content.strip()}\n")
            if evaluation.highlights:
                points_text = "\n".join(
                    f"      - {point.strip()}" for point in evaluation.highlights
                )
                results.write(f"    Highlights:\n{points_text}\n")
            results.write(f"    Reasoning: {evaluation.reasoning.strip()}\n\n")

        self._logger.debug(f"Rule evaluation results ({evaluator_name}):\n{results.getvalue()}")

    async def _record(
        self,
        context: EngineContext,
        evaluations: Sequence[tuple[RuleEvaluation, _ContextUsage]],
        *,
        append: bool,
    ) -> None:
        # The matched-rule ledger records verdict-level matches — captured
        # BEFORE priority resolution, so a match demoted by priority still
        # counts as "has matched" and keeps satisfying its dependents.
        # Transient (tool-returned) rules are per-turn and never ledgered.
        verdict_match_ids = {
            evaluation.rule.id
            for evaluation, usage in evaluations
            if usage == _ContextUsage.MATCH_CURRENT_TURN
            and not evaluation.metadata.get("transient")
        }

        # Decisions mirror exactly what persists: the verdict-level match set
        # (pre-priority), so a priority-demoted rule is still "selected".
        staged_decision_ids = set()
        for evaluation, usage in evaluations:
            if (
                usage == _ContextUsage.MATCH_CURRENT_TURN
                and not evaluation.metadata.get("transient")
                and evaluation.rule.id not in staged_decision_ids
            ):
                staged_decision_ids.add(evaluation.rule.id)
                context.state.matcher.pending_rule_decisions.append(
                    RuleDecision(
                        rule=evaluation.rule,
                        selected=True,
                        rationale=evaluation.reasoning,
                    )
                )

        lifting_rules = sorted(
            {
                evaluation.rule
                for evaluation, usage in evaluations
                if usage == _ContextUsage.MATCH_CURRENT_TURN
                and evaluation.rule.effort is not None
                and evaluation.rule.effort > context.state.agent_effort
            },
            key=lambda rule: str(rule.id),
        )
        if lifting_rules:
            CompassTracer(context.tracer).effort_raised(
                from_effort=context.state.agent_effort,
                to_effort=max(rule.effort for rule in lifting_rules if rule.effort is not None),
                rule_ids=[str(rule.id) for rule in lifting_rules],
            )

        evaluations = await self._resolve_turn_priority(context, evaluations, append=append)

        turn_matches = [
            evaluation
            for evaluation in evaluations
            if evaluation[1] == _ContextUsage.MATCH_CURRENT_TURN
        ]

        # Classify into ordinary vs tool-enabled (the latter carries the tool ids
        # the matched rules enable).
        tool_enabled = await self._find_tool_enabled_rule_matches(turn_matches)
        ordinary = [m for m in turn_matches if m not in tool_enabled]

        if append:
            # Never reorder, so the already-rendered rules stay a
            # byte-identical prefix.
            context.state.ordinary_rule_matches.extend([key[0] for key in ordinary])
            context.state.tool_enabled_rule_matches.update(
                {key[0]: tool_enabled[key] for key in tool_enabled}
            )
        else:
            context.state.tool_enabled_rule_matches = {
                key[0]: tool_enabled[key] for key in tool_enabled
            }
            context.state.ordinary_rule_matches = [
                m[0] for m in set(turn_matches).difference(set(tool_enabled.keys()))
            ]

        try:
            await self._store_discovered_rules(context, evaluations)
            self._store_rules_ever_matched(context, verdict_match_ids)
        finally:
            # Pending admission is only useful while evaluation phases are in
            # progress. Clear it even if persistence fails, otherwise a later
            # rematch could mistake stale provisional evidence for this call's
            # discovery.
            context.state.matcher.pending_discovered_rules.clear()
            context.state.matcher.pending_matched_rules.clear()

        context.state.invalidate_cached_properties()

    async def _find_tool_enabled_rule_matches(
        self,
        rule_matches: Sequence[tuple[RuleEvaluation, _ContextUsage]],
    ) -> dict[tuple[RuleEvaluation, _ContextUsage], list[ToolId]]:
        matches_by_id = {m[0].rule.id: m for m in rule_matches}

        tools_for_rules: dict[tuple[RuleEvaluation, _ContextUsage], list[ToolId]] = defaultdict(
            list
        )

        for association in await self._entity_queries.find_rule_tool_associations():
            if association.rule_id in matches_by_id:
                tools_for_rules[matches_by_id[association.rule_id]].append(association.tool_id)

        return dict(tools_for_rules)

    def _matched_rule_ids(self, context: EngineContext) -> set[RuleId]:
        return {
            m.rule.id
            for m in chain(
                context.state.ordinary_rule_matches,
                context.state.tool_enabled_rule_matches,
            )
        }

    def _executed_tool_ids(self, context: EngineContext) -> set[ToolId]:
        # Tool events record the qualified "service:tool" id; fall back to the
        # per-turn name->id table for any bare name (older events).
        executed: set[ToolId] = set()
        for event in context.state.tool_events:
            for call in cast(ToolEventData, event.data)["tool_calls"]:
                raw_tool_id = call["tool_id"]
                if ":" in raw_tool_id:
                    executed.add(ToolId.from_string(raw_tool_id))
                elif tool_id := context.state.tool_ids_by_name.get(raw_tool_id):
                    executed.add(tool_id)
        return executed

    async def _find_rules_gated_on_tools(
        self,
        context: EngineContext,
        tool_ids: set[ToolId],
    ) -> list[Rule]:
        usable_by_id = {g.id: g for g in context.state.usable_rules}
        gated: dict[RuleId, Rule] = {}

        for tool_id in tool_ids:
            relationships = await self._relationship_store.list_relationships(
                kind=RelationshipKind.REEVALUATION,
                indirect=False,
                target_id=tool_id,
            )

            for relationship in relationships:
                # Source is a rule (match by id prefix, as elsewhere) or a group
                # (match by group membership).
                matched = [
                    g for gid, g in usable_by_id.items() if gid.startswith(relationship.source.id)
                ]

                if not matched and relationship.source.kind.is_group:
                    matched = [
                        g for g in usable_by_id.values() if relationship.source.id in g.groups
                    ]

                for rule in matched:
                    gated[rule.id] = rule

        return list(gated.values())

    # --- tool selection ---

    async def _select_tools(
        self,
        context: EngineContext,
        *,
        log_delta_from_fill: bool,
    ) -> None:
        await self._tool_selector.select(context)
        CompassTracer(context.tracer).tools_selected(context.state.matched_tools)

        current_tool_ids = self._available_tool_ids(context)

        await self._collect_tool_overlaps(context)

        if log_delta_from_fill:
            added = current_tool_ids - context.state.fill_available_tool_ids
            removed = context.state.fill_available_tool_ids - current_tool_ids
            if added or removed:
                self._log_available_tool_delta(context, added, removed)
        else:
            context.state.fill_available_tool_ids = current_tool_ids
            self._log_available_tools(context)

    async def _collect_tool_overlaps(self, context: EngineContext) -> None:
        """Collect overlapping-alternative pairs among the AVAILABLE tools into
        ``state.overlapping_tool_ids`` (the responder renders them as an
        explicit pick-one note — only when any pair exists, so unconfigured
        deployments pay no prompt tokens for this).

        A pair overlaps when both sides allow it (``ToolOverlap.NONE`` opts a
        tool out entirely) and either an OVERLAP tool relationship declares it
        or one side is ``ToolOverlap.ALWAYS`` (which pairs with every other
        available tool). The relationship store is only consulted when at least
        two eligible tools are present."""
        overlap_by_id: dict[str, tuple[ToolId, ToolOverlap]] = {}
        for tool in context.state.available_tools:
            if (tool_id := context.state.tool_ids_by_name.get(tool.name)) is not None:
                overlap_by_id[tool_id.to_string()] = (tool_id, tool.overlap)

        eligible = {
            key: tool_id
            for key, (tool_id, overlap) in overlap_by_id.items()
            if overlap is not ToolOverlap.NONE
        }

        pairs: dict[tuple[str, str], tuple[ToolId, ToolId]] = {}

        def admit(a_key: str, b_key: str) -> None:
            (first_key, second_key) = sorted((a_key, b_key))
            pairs[(first_key, second_key)] = (eligible[first_key], eligible[second_key])

        for key, (tool_id, overlap) in overlap_by_id.items():
            if overlap is ToolOverlap.ALWAYS:
                for other_key in eligible:
                    if other_key != key:
                        admit(key, other_key)

        if len(eligible) > 1:
            declared = await self._relationship_store.list_relationships(
                kind=RelationshipKind.OVERLAP
            )
            for relationship in declared:
                if (
                    relationship.source.kind is RelationshipEntityKind.TOOL
                    and relationship.target.kind is RelationshipEntityKind.TOOL
                ):
                    source_key = relationship.source.id_to_string()
                    target_key = relationship.target.id_to_string()
                    if source_key in eligible and target_key in eligible:
                        admit(source_key, target_key)

        context.state.overlapping_tool_ids = [pairs[key] for key in sorted(pairs)]

    def _available_tool_ids(self, context: EngineContext) -> set[ToolId]:
        return {
            tool_id
            for tool in context.state.available_tools
            if (tool_id := context.state.tool_ids_by_name.get(tool.name)) is not None
        }

    def _log_available_tool_delta(
        self,
        context: EngineContext,
        added: set[ToolId],
        removed: set[ToolId],
    ) -> None:
        tools_by_id = {
            tool_id: tool
            for tool in context.state.available_tools
            if (tool_id := context.state.tool_ids_by_name.get(tool.name)) is not None
        }

        tools_log = StringIO()

        if added:
            tools_log.write("Added:\n")
            for idx, tool_id in enumerate(sorted(added, key=lambda tid: tid.to_string()), start=1):
                tool = tools_by_id.get(tool_id)
                score = context.state.tool_relevance_scores.get(tool_id, 0.0)
                tools_log.write(
                    f"### +{idx} {tool.name if tool else tool_id.tool_name} "
                    f"[Score: {score:.2f}]\n\n"
                )
                tools_log.write(f"    Tool ID: {tool_id.to_string()}\n")
                if tool and tool.description:
                    tools_log.write(f"    Description: {tool.description.strip()}\n")
                tools_log.write("\n")

        if removed:
            tools_log.write("Removed:\n")
            for idx, tool_id in enumerate(
                sorted(removed, key=lambda tid: tid.to_string()), start=1
            ):
                tools_log.write(f"### -{idx} {tool_id.tool_name}\n\n")
                tools_log.write(f"    Tool ID: {tool_id.to_string()}\n\n")

        self._logger.debug(f"Tool recall:\n{tools_log.getvalue()}")

    def _log_available_tools(self, context: EngineContext) -> None:
        if not context.state.available_tools:
            self._logger.debug("Tool recall:\n[None]")
            return

        matched_tool_names = {tool.name for tool in context.state.matched_tools}
        discovered_rule_ids = {rule.id for rule in context.state.discovered_rules}
        session_tool_ids = {
            tool_id
            for rule_id in discovered_rule_ids
            for tool_id, _ in context.state.tools_by_rule.get(rule_id, set())
        }

        tool_log_entries = []
        for tool in context.state.available_tools:
            tool_id = context.state.tool_ids_by_name.get(tool.name)
            score = context.state.tool_relevance_scores.get(tool_id, 0.0) if tool_id else 0.0

            if tool.name in matched_tool_names:
                bucket_priority = 0
                bucket = "Matched to turn"
            elif tool_id in session_tool_ids:
                bucket_priority = 1
                bucket = "Matched to session"
            else:
                bucket_priority = 2
                bucket = "Complementary"

            tool_log_entries.append(
                (bucket_priority, -score, tool.name, tool, tool_id, score, bucket)
            )

        tools_log = StringIO()
        for idx, (_, _, _, tool, tool_id, score, bucket) in enumerate(
            sorted(tool_log_entries),
            start=1,
        ):
            tools_log.write(f"### {idx} {tool.name} [Score: {score:.2f} ({bucket})]\n\n")

            if tool_id:
                tools_log.write(f"    Tool ID: {tool_id.to_string()}\n")

            if tool.description:
                tools_log.write(f"    Description: {tool.description.strip()}\n")

            tools_log.write("\n")

        self._logger.trace(f"Available tools:\n{tools_log.getvalue()}")

    # --- glossary ---

    async def _load_rule_terms(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
    ) -> None:
        """Resolve the glossary terms each rule depends on (top-k by the
        rule's own query) into ``context.state.terms_by_rule``, so the
        turn evaluator can render them next to the rule. Memoized on
        (rule, glossary inventory) - always up-to-date under runtime edits
        to either side, at a dict-hit steady-state cost."""
        if not rules:
            return

        inventory = await self._entity_queries.list_glossary_terms_for_context(context.agent.id)

        if not inventory:
            return

        terms_by_id = {term.id: term for term in inventory}
        inventory_checksum = xxh3_checksum(
            "".join(sorted(f"{term.id}:{term.modified_utc.isoformat()}" for term in inventory))
        )

        async def resolve(rule: Rule) -> tuple[RuleId, tuple[TermId, ...]]:
            key = (rule.id, rule.modified_utc.isoformat(), inventory_checksum)

            if (term_ids := _RULE_TERM_IDS.get(key)) is None:
                found = await self._entity_queries.find_glossary_terms_for_context(
                    context.agent.id,
                    query=rule.query,
                    max_terms=self._TERMS_PER_RULE,
                )
                term_ids = tuple(term.id for term in found)
                _RULE_TERM_IDS[key] = term_ids

            return rule.id, term_ids

        results = await safe_gather(*(resolve(rule) for rule in rules))

        for rule_id, term_ids in results:
            terms = [terms_by_id[term_id] for term_id in term_ids if term_id in terms_by_id]
            if terms:
                context.state.terms_by_rule[rule_id] = terms

    async def _load_glossary(self, context: EngineContext) -> None:
        # Load the session's sticky glossary working set into the state (discovering
        # new terms from this turn's messages), so the responder can surface it in
        # its (cached) system instructions. Loaded once here, not per response step.
        await self._glossary_recaller.recall(context)

    async def prune_session_glossary(self, context: EngineContext) -> None:
        """Cap the session glossary working set (post-response; see
        :meth:`GlossaryRecaller.prune`). Best-effort: a pruning failure must never
        fail the turn."""
        try:
            await self._glossary_recaller.prune(context)
        except Exception as exc:
            self._logger.warning(f"Session glossary pruning failed (continuing): {exc}")
