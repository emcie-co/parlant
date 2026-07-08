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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from enum import Enum, IntEnum, auto
from io import StringIO
from itertools import chain
from typing import cast

from parlant.core.agents import Effort
from parlant.core.async_utils import safe_gather
from parlant.core.common import Weight, JSONSerializable, xxh3_checksum
from parlant.core.engines.rule_match import RuleMatch
from parlant.core.engines.rule_matcher_registry import RuleMatcherRegistry
from parlant.core.engines.compass.matching.rule_function_matcher import (
    RuleFunctionMatcher,
)
from parlant.core.engines.compass.matching.glossary_recaller import GlossaryRecaller
from parlant.core.engines.compass.matching.rule_evaluation import (
    RuleEvaluationResult,
    TurnEvaluator,
)
from parlant.core.engines.compass.matching.rule_discovery import RuleDiscoverer
from parlant.core.engines.compass.matching.rule_pruner import (
    RulePruner,
)
from parlant.core.engines.compass.matching.tool_recaller import ToolRecaller
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.engines.compass.tracing import CompassTracer
from parlant.core.engines.compass.variable_loader import VariableLoader
from parlant.core.entity_cq import EntityCommands, EntityQueries
from parlant.core.glossary import TermId
from parlant.core.rules import Rule, RuleId
from parlant.core.loggers import Logger
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.sessions import EventKind, EventSource, ToolEventData
from parlant.core.groups import GroupId
from parlant.core.tools import ToolId

# Memo for per-rule glossary-term lookups, keyed by (rule id, rule
# modified time, glossary inventory checksum) - so edits to either side recompute
# on next use, and the steady state is a dict hit (no embedding search).
_RULE_TERM_IDS: dict[tuple[RuleId, str, str], tuple[TermId, ...]] = {}
_SESSION_RULE_IDS_METADATA_KEY = "compass.session_rule_ids"
# Eviction ledger: {rule id: last event offset at eviction}. A ledgered
# rule may only be readmitted by conversation that arrives AFTER its
# eviction (honored by the RuleDiscoverer via state.evicted_session_rules).
_EVICTED_SESSION_RULES_METADATA_KEY = "compass.evicted_session_rules"
# The event offset at which the last pruning ran — the pruning rate limiter's
# reference point.
_LAST_SESSION_PRUNING_OFFSET_METADATA_KEY = "compass.last_session_pruning_offset"


class _ContextUsage(Enum):
    """How to use a rule within the engine's context"""

    INCLUDE_IN_SESSION = auto()
    MATCH_CURRENT_TURN = auto()


class _MatcherStrategy(IntEnum):
    """How much effort to spend deciding whether a rule applies, cheapest to
    most thorough."""

    NONE = auto()
    RECALL = auto()
    RANK = auto()


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

    # Session rule cap: pruning fires only above the high-water mark and
    # evicts down to the target, so evictions (responder-prefix cache misses) are
    # batched and rare rather than continuous.
    _SESSION_RULES_HIGH_WATER_MARK = 30
    _SESSION_RULES_TARGET = 20
    # And at most once per this many customer messages, whatever the set size.
    _MIN_CUSTOMER_MESSAGES_BETWEEN_PRUNINGS = 10

    def __init__(
        self,
        logger: Logger,
        rule_discoverer: RuleDiscoverer,
        tool_recaller: ToolRecaller,
        rule_evaluator: TurnEvaluator,
        rule_function_matcher: RuleFunctionMatcher,
        rule_pruner: RulePruner,
        glossary_recaller: GlossaryRecaller,
        matcher_registry: RuleMatcherRegistry,
        relationship_store: RelationshipStore,
        entity_queries: EntityQueries,
        entity_commands: EntityCommands,
        variable_loader: VariableLoader,
    ) -> None:
        self._logger = logger
        self._rule_discoverer = rule_discoverer
        self._tool_recaller = tool_recaller
        self._rule_evaluator = rule_evaluator
        self._rule_function_matcher = rule_function_matcher
        self._rule_pruner = rule_pruner
        self._glossary_recaller = glossary_recaller
        self._matcher_registry = matcher_registry
        self._relationship_store = relationship_store
        self._entity_queries = entity_queries
        self._entity_commands = entity_commands
        self._variable_loader = variable_loader

    async def preload(self, context: EngineContext) -> None:
        with context.tracer.span("match.preload"):
            # Load the shared prompt/matching inputs that matcher-owned preparation
            # controls before matching and cache warm-up.
            context.state.context_variables, rules = await safe_gather(
                self._variable_loader.load(context),
                self._entity_queries.find_rules_for_context(context.agent.id, []),
            )

            context.state.usable_rules = list(rules)

            CompassTracer(context.tracer).rules_loaded(context.state.usable_rules)

            await self._load_session_rules(context)

    async def fill(self, context: EngineContext) -> None:
        """Initial preparation: match all usable rules, score the agent's tool
        pool, and load the relevant glossary (all independent, so in parallel), then
        select the offered tools."""
        with context.tracer.span("match.fill"):
            await safe_gather(
                self._match(context),
                self._tool_recaller.prepare(context),
                self._load_glossary(context),
            )
            await self._select_tools(context, log_delta_from_fill=False)

    async def update(self, context: EngineContext) -> None:
        """Refresh after a step: reevaluate rules gated on the tools that
        just ran, then re-select tools. Tool relevance depends on the unchanged
        conversation, so it isn't rescored."""
        with context.tracer.span("match.update"):
            await self._reevaluate(context)
            await self._select_tools(context, log_delta_from_fill=True)

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
        """The distinct turn evaluators selected across the usable inventory."""
        rules = [
            g
            for g in context.state.usable_rules
            if (g.weight != Weight.LOW or self._check_if_raises_effort(context, g))
            and self._matcher_registry.get(g.id) is None
        ]

        if not rules:
            return []

        await self._load_strategy_choice_signals(context, rules)

        evaluators: list[TurnEvaluator] = []
        for rule in rules:
            evaluator = self._select_evaluator(context, rule)
            if evaluator is not None and not any(evaluator is e for e in evaluators):
                evaluators.append(evaluator)

        return evaluators

    # --- rule matching ---

    async def _load_session_rules(self, context: EngineContext) -> None:
        rule_ids = self._read_session_rule_ids(context.session.metadata)

        rules_by_id = {g.id: g for g in context.state.usable_rules}

        context.state.session_rules = {
            rules_by_id[rule_id] for rule_id in rule_ids if rule_id in rules_by_id
        }

        ledger = cast(
            Mapping[str, int],
            context.session.metadata.get(_EVICTED_SESSION_RULES_METADATA_KEY, {}),
        )
        context.state.evicted_session_rules = {
            RuleId(rule_id): int(offset)
            for rule_id, offset in ledger.items()
            if RuleId(rule_id) in rules_by_id
        }

    async def _store_session_rules(
        self,
        context: EngineContext,
        matches: Sequence[tuple[RuleMatch, _ContextUsage]],
    ) -> None:
        rules_by_id = {g.id: g for g in context.state.usable_rules}
        rule_ids = {
            *(g.id for g in context.state.session_rules),
            *(m[0].rule.id for m in matches),
        }

        context.state.session_rules = {rules_by_id[gid] for gid in rule_ids}

        # A rule (re)entering the session set leaves the eviction ledger — it
        # earned readmission with a fresh trigger.
        context.state.evicted_session_rules = {
            gid: offset
            for gid, offset in context.state.evicted_session_rules.items()
            if gid not in rule_ids
        }

        current_rule_ids = self._read_session_rule_ids(context.session.metadata)
        current_ledger = context.session.metadata.get(_EVICTED_SESSION_RULES_METADATA_KEY, {})
        ledger = {str(gid): offset for gid, offset in context.state.evicted_session_rules.items()}

        if rule_ids == current_rule_ids and ledger == current_ledger:
            return

        metadata = dict(context.session.metadata)

        metadata[_SESSION_RULE_IDS_METADATA_KEY] = [str(rule_id) for rule_id in rule_ids]
        metadata[_EVICTED_SESSION_RULES_METADATA_KEY] = cast(JSONSerializable, ledger)

        await self._entity_commands.update_session(context.session.id, {"metadata": metadata})
        context.session = replace(context.session, metadata=metadata)

    def _read_session_rule_ids(
        self,
        metadata: Mapping[str, JSONSerializable],
    ) -> set[RuleId]:
        last_known_state = set(
            cast(Iterable[str], metadata.get(_SESSION_RULE_IDS_METADATA_KEY, []))
        )

        return {RuleId(rule_id) for rule_id in last_known_state}

    async def prune_session_rules(self, context: EngineContext) -> None:
        """Cap the session rule set.

        When the set exceeds the high-water mark, ask the pruner which members are
        still relevant to the conversation going forward, and evict the stalest down
        to the target — exempting HIGH-criticality members and this turn's matches.
        Runs post-response (finalize), so the responder-prefix cache miss the
        eviction causes lands between turns; rate-limited to once per
        ``_MIN_CUSTOMER_MESSAGES_BETWEEN_PRUNINGS`` customer messages. Best-effort:
        a pruning failure must never fail the turn.
        """
        try:
            members = context.state.session_rules

            if len(members) <= self._SESSION_RULES_HIGH_WATER_MARK:
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
            # lowest-scoring first, breaking ties by criticality, then id.
            criticality_rank = {
                Weight.LOW: 0,
                Weight.MEDIUM: 1,
                Weight.HIGH: 2,
            }
            # Defensive: never evict an exempt member, whatever the pruner returned.
            ranked = sorted(
                (c for c in result.pruned_rules if c.rule.id not in exempt_ids),
                key=lambda c: (
                    c.score,
                    criticality_rank[c.rule.weight],
                    str(c.rule.id),
                ),
            )

            needed = len(members) - self._SESSION_RULES_TARGET
            evicted = ranked[: min(needed, len(ranked))]

            if not evicted:
                return

            # Offsets are session-global and monotonic (they survive compaction),
            # so the interaction's last event marks "now" for the ledger and the
            # rate limiter alike.
            last_offset = max((e.offset for e in context.interaction.events), default=0)
            evicted_ids = {c.rule.id for c in evicted}

            context.state.session_rules = {g for g in members if g.id not in evicted_ids}
            context.state.evicted_session_rules = {
                **context.state.evicted_session_rules,
                **{rule_id: last_offset for rule_id in evicted_ids},
            }

            metadata = dict(context.session.metadata)
            metadata[_SESSION_RULE_IDS_METADATA_KEY] = [
                str(g.id) for g in context.state.session_rules
            ]
            metadata[_EVICTED_SESSION_RULES_METADATA_KEY] = {
                str(rule_id): offset
                for rule_id, offset in context.state.evicted_session_rules.items()
            }
            metadata[_LAST_SESSION_PRUNING_OFFSET_METADATA_KEY] = last_offset

            await self._entity_commands.update_session(context.session.id, {"metadata": metadata})
            context.session = replace(context.session, metadata=metadata)

            context.state.invalidate_cached_properties()

            pruning_results = StringIO()
            for idx, pruned in enumerate(evicted, start=1):
                g = pruned.rule
                pruning_results.write(
                    f"### {idx} {g.title or g.content.condition or ''} [Score: {pruned.score:.2f}]\n"
                    f"    Reasoning: {pruned.reasoning.strip()}\n\n"
                )
            self._logger.debug(
                f"{self.__class__.__name__} evicted {len(evicted)} of {len(members)} session "
                f"rules (target {self._SESSION_RULES_TARGET}):\n{pruning_results.getvalue()}"
            )
        except Exception as exc:
            self._logger.warning(f"Session rule pruning failed (continuing): {exc}")

    def _pruning_rate_limit_elapsed(self, context: EngineContext) -> bool:
        last_pruning_offset = context.session.metadata.get(
            _LAST_SESSION_PRUNING_OFFSET_METADATA_KEY
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
        matches = await self._run_batches(context, context.state.usable_rules)
        await self._record(context, matches, append=False)

    async def _reevaluate(self, context: EngineContext) -> None:
        executed_tool_ids = self._executed_tool_ids(context)
        if not executed_tool_ids:
            return

        gated = await self._find_rules_gated_on_tools(context, executed_tool_ids)
        already_matched = self._matched_rule_ids(context)
        candidates = [g for g in gated if g.id not in already_matched]
        if not candidates:
            return

        matches = await self._run_batches(context, candidates)

        if not matches:
            return

        await self._record(context, matches, append=True)

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
                        strategy = _MatcherStrategy.RECALL
            case Effort.MEDIUM:
                match rule.weight:
                    case Weight.LOW:
                        strategy = _MatcherStrategy.RECALL
                    case Weight.MEDIUM:
                        strategy = _MatcherStrategy.RECALL
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

            if self._check_if_has_dependencies(context, rule):
                # If the rule has dependencies, it may be gating important follow-up
                # rules, so we should at least rank - not use embeddings.
                return _MatcherStrategy.RANK

            if self._check_if_has_tools(context, rule):
                # If the rule has tools, it may be gating important interactions
                # with data and/or actions, so we should at least rank - not use embeddings.
                return _MatcherStrategy.RANK

            if self._check_if_raises_effort(context, rule):
                # If a matching rule would raise the turn's effort level,
                # it must be matched more thoroughly
                return _MatcherStrategy.RANK

        return strategy

    def _check_if_raises_effort(self, context: EngineContext, rule: Rule) -> bool:
        return (
            rule.effort_lift is not None and rule.effort_lift > context.state.dynamic_effort_level
        )

    async def _load_strategy_choice_signals(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
    ) -> None:
        associations = await self._entity_queries.find_rule_tool_associations()
        context.state.rule_ids_with_tools = {a.rule_id for a in associations}

        # A rule "has dependencies" if it takes part in a dependency
        # relationship — either as the dependent (source) or as the rule being
        # depended on (target, which gates its dependents). Either side means
        # getting its match right carries downstream consequences.
        #
        # Endpoints may be rules directly, or groups (GROUP_ALL/GROUP_ANY) standing
        # in for every rule that carries the group; we resolve both against the
        # candidate rules so the per-rule check can stay a plain lookup.
        dependency_relationships = chain(
            await self._relationship_store.list_relationships(
                kind=RelationshipKind.DEPENDENCY, indirect=False
            ),
            await self._relationship_store.list_relationships(
                kind=RelationshipKind.DEPENDENCY_ANY, indirect=False
            ),
        )

        dependency_rule_ids: set[RuleId] = set()
        dependency_groups: set[GroupId] = set()
        for relationship in dependency_relationships:
            for endpoint in (relationship.source, relationship.target):
                if endpoint.kind == RelationshipEntityKind.RULE:
                    dependency_rule_ids.add(cast(RuleId, endpoint.id))
                elif endpoint.kind.is_group:
                    dependency_groups.add(cast(GroupId, endpoint.id))

        context.state.rule_ids_with_dependencies = {
            rule.id
            for rule in rules
            if rule.id in dependency_rule_ids or not dependency_groups.isdisjoint(rule.groups)
        }

    def _check_if_has_tools(self, context: EngineContext, rule: Rule) -> bool:
        return rule.id in context.state.rule_ids_with_tools

    def _check_if_has_dependencies(self, context: EngineContext, rule: Rule) -> bool:
        return rule.id in context.state.rule_ids_with_dependencies

    async def _run_batches(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
    ) -> Sequence[tuple[RuleMatch, _ContextUsage]]:
        """Decide which of `rules` apply by routing each to a strategy and
        running the resulting batches in parallel.

        A rule with a code (Python) matcher always goes to the function
        matcher, regardless of strategy — an explicit matcher is authoritative. The
        rest are bucketed by `_get_strategy`.
        """
        # Low-criticality rules are always included in the system instructions,
        # unless they carry an effort override that can affect the current turn.
        rules = [
            g for g in rules if g.weight != Weight.LOW or self._check_if_raises_effort(context, g)
        ]

        if not rules:
            return []

        # _get_strategy runs per rule and is synchronous, so precompute the
        # store-backed signals it consults (which rules have tools / are in a
        # dependency relationship) once, here.
        await self._load_strategy_choice_signals(context, rules)

        code_batch: list[Rule] = []
        discovery_batch: list[Rule] = []
        # Rules grouped by their selected turn evaluator — the matcher is
        # natively multi-evaluator; subclasses may route rules to their own
        # evaluators via `_select_evaluator`.
        evaluator_groups: list[tuple[TurnEvaluator, list[Rule]]] = []

        for rule in rules:
            if self._matcher_registry.get(rule.id) is not None:
                code_batch.append(rule)
                continue

            # Discovery runs for every rule not already in the session working set,
            # regardless of whether a turn evaluator also judges it.
            if rule not in context.state.session_rules:
                discovery_batch.append(rule)

            if (evaluator := self._select_evaluator(context, rule)) is not None:
                for existing, group in evaluator_groups:
                    if existing is evaluator:
                        group.append(rule)
                        break
                else:
                    evaluator_groups.append((evaluator, [rule]))

        # Evaluated rules may depend on glossary terms to be interpreted
        # correctly; resolve them into the state before the prompts are built.
        await self._load_rule_terms(
            context, [rule for _, group in evaluator_groups for rule in group]
        )

        gathered = await safe_gather(
            self._rule_function_matcher.match(context, code_batch),
            self._rule_discoverer.discover(context, discovery_batch),
            *(evaluator.evaluate(context, group) for evaluator, group in evaluator_groups),
        )
        code_matches = gathered[0]
        discovered = gathered[1]
        evaluation_results = cast(list[RuleEvaluationResult], list(gathered[2:]))

        discovery_results = StringIO()

        if discovered.discovered_rules:
            discovery_results.write(f"Duration: {discovered.duration:.3f}s\n\n")

            for idx, discovery_result in enumerate(discovered.discovered_rules, start=1):
                g = discovery_result.rule

                discovery_results.write(
                    f"### {idx} {g.title or ''} [Score: {discovery_result.score:.2f} ({'Relevant' if discovery_result.is_relevant else 'Not Relevant'})]\n\n"
                )
                if g.content.condition:
                    discovery_results.write(f"    Condition: {g.content.condition}\n")
                if g.content.action:
                    discovery_results.write(f"    Action: {g.content.action}\n")
                if g.content.description:
                    discovery_results.write(f"    Description: {g.content.description.strip()}\n")
                discovery_results.write("\n")

            self._logger.debug(
                f"{self.__class__.__name__} rule discovery results:\n{discovery_results.getvalue()}"
            )

        for (evaluator, _), evaluated in zip(evaluator_groups, evaluation_results):
            self._log_evaluations(type(evaluator).__name__, evaluated)

        matches: list[tuple[RuleMatch, _ContextUsage]] = [
            (m, _ContextUsage.MATCH_CURRENT_TURN) for m in code_matches
        ]

        matches += [
            (
                RuleMatch(
                    rule=evaluation.rule,
                    rationale=evaluation.reasoning
                    or "This may or may not be relevant right now - use your judgment.",
                    metadata={"highlights": list(evaluation.highlights)}
                    if evaluation.highlights
                    else {},
                ),
                _ContextUsage.MATCH_CURRENT_TURN,
            )
            for evaluated in evaluation_results
            for evaluation in evaluated.evaluations
            if evaluation.is_relevant
        ]

        turn_matched_rule_ids = {
            match.rule.id
            for match, context_usage in matches
            if context_usage == _ContextUsage.MATCH_CURRENT_TURN
        }

        matches += [
            (
                RuleMatch(
                    rule=rc.rule,
                    rationale="This may or may not be relevant right now - use your judgment.",
                ),
                _ContextUsage.INCLUDE_IN_SESSION,
            )
            for rc in discovered.discovered_rules
            if rc.is_relevant and rc.rule.id not in turn_matched_rule_ids
        ]

        return matches

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
            if g.content.condition:
                results.write(f"    Condition: {g.content.condition}\n")
            if g.content.action:
                results.write(f"    Action: {g.content.action}\n")
            if g.content.description:
                results.write(f"    Description: {g.content.description.strip()}\n")
            if evaluation.highlights:
                points_text = "\n".join(
                    f"      - {point.strip()}" for point in evaluation.highlights
                )
                results.write(f"    Highlights:\n{points_text}\n")
            results.write(f"    Reasoning: {evaluation.reasoning.strip()}\n\n")

        self._logger.debug(
            f"{self.__class__.__name__} rule evaluation results ({evaluator_name}):\n{results.getvalue()}"
        )

    async def _record(
        self,
        context: EngineContext,
        matches: Sequence[tuple[RuleMatch, _ContextUsage]],
        *,
        append: bool,
    ) -> None:
        old_effort = context.state.dynamic_effort_level
        turn_matches = [match for match in matches if match[1] == _ContextUsage.MATCH_CURRENT_TURN]

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

        await self._store_session_rules(context, matches)

        await self._update_session_labels(context, turn_matches)

        context.state.invalidate_cached_properties()

        new_effort = context.state.dynamic_effort_level
        if new_effort > old_effort:
            CompassTracer(context.tracer).effort_raised(
                old_effort,
                new_effort,
                [
                    str(match.rule.id)
                    for match, usage in matches
                    if usage == _ContextUsage.MATCH_CURRENT_TURN
                    and match.rule.effort_lift == new_effort
                ],
            )

    async def _update_session_labels(
        self,
        context: EngineContext,
        matches: Sequence[tuple[RuleMatch, _ContextUsage]],
    ) -> None:
        existing_labels = set(context.session.labels)
        labels_to_add: set[str] = set()
        source_rule_ids: set[str] = set()

        for match, usage in matches:
            if usage != _ContextUsage.MATCH_CURRENT_TURN:
                continue

            new_rule_labels = set(match.rule.labels) - existing_labels
            if not new_rule_labels:
                continue

            labels_to_add.update(new_rule_labels)
            source_rule_ids.add(str(match.rule.id))

        if not labels_to_add:
            return

        context.session = await self._entity_commands.upsert_session_labels(
            context.session.id,
            labels_to_add,
        )
        CompassTracer(context.tracer).labels_added(labels_to_add, source_rule_ids)

    async def _find_tool_enabled_rule_matches(
        self,
        rule_matches: Sequence[tuple[RuleMatch, _ContextUsage]],
    ) -> dict[tuple[RuleMatch, _ContextUsage], list[ToolId]]:
        matches_by_id = {m[0].rule.id: m for m in rule_matches}

        tools_for_rules: dict[tuple[RuleMatch, _ContextUsage], list[ToolId]] = defaultdict(list)

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
        # The react loop records executed tools (by name) in state.tool_events;
        # map them back to ToolIds via the per-turn name->id table.
        executed: set[ToolId] = set()
        for event in context.state.tool_events:
            for call in cast(ToolEventData, event.data)["tool_calls"]:
                if tool_id := context.state.tool_ids_by_name.get(call["tool_id"]):
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
        await self._tool_recaller.select(context)

        current_tool_ids = self._available_tool_ids(context)

        if log_delta_from_fill:
            added = current_tool_ids - context.state.fill_available_tool_ids
            removed = context.state.fill_available_tool_ids - current_tool_ids
            if added or removed:
                self._log_available_tool_delta(context, added, removed)
        else:
            context.state.fill_available_tool_ids = current_tool_ids
            self._log_available_tools(context)

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

        self._logger.debug(f"{self.__class__.__name__} tool recall:\n{tools_log.getvalue()}")

    def _log_available_tools(self, context: EngineContext) -> None:
        if not context.state.available_tools:
            self._logger.debug(f"{self.__class__.__name__} tool recall:\n[None]")
            return

        matched_tool_names = {tool.name for tool in context.state.matched_tools}
        session_rule_ids = {rule.id for rule in context.state.session_rules}
        session_tool_ids = {
            tool_id
            for rule_id in session_rule_ids
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

        self._logger.debug(f"{self.__class__.__name__} available tools:\n{tools_log.getvalue()}")

    def _build_tool_query(self, context: EngineContext) -> str:
        return (
            f"{context.agent.description or ''}\n\n{self._build_interaction_query_lines(context)}"
        )

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

    def _build_interaction_query_lines(self, context: EngineContext) -> list[str]:
        lines: list[str] = []

        if context.state.session_summary:
            lines.append(f"Session summary: {context.state.session_summary}")

        lines.extend(f"{m.source}: {m.content}" for m in context.interaction.messages)

        if not lines:
            # No conversation yet (the initialize-time warm-up). Rank against a neutral
            # greeting so the glossary still loads — letting the warmed prefix include it
            # and match the first real turn. With a glossary under the cap the full set
            # is returned regardless of query, so it matches that turn exactly.
            lines.append("User: Hello")

        return lines
