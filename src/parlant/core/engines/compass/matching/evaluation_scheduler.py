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

"""Plans multi-phase rule evaluation without owning rule evaluation itself.

Terminology
-----------

Rule inventory
    Every usable rule known to the matcher. Preconditions may refer to any rule
    in this inventory, including one that is not an evaluation candidate.

Candidate
    A rule that one or more evaluators may evaluate during this matcher call.
    The scheduling threshold counts candidates, not the full inventory.

Evidence source
    One independent mechanism expected to report whether a rule is relevant,
    such as recall, a code evaluator, a ranker, or a distiller. A rule may have
    several sources, and one negative result is not conclusive while another
    source remains outstanding.

Evidence
    A snapshot of rule relevance results available between evaluation rounds.
    Evidence is not streamed into the plan and the scheduler does not monitor
    background tasks. The matcher runs the selected evidence sources
    concurrently, waits for their calls to complete, and then updates the plan
    by calling :meth:`EvaluationPlan.observe` once for each source.

    A plan starts with evidence supplied by the matcher, such as rules durably
    discovered on earlier turns. Current-turn evidence is added after each
    round. Preconditions interpret the resulting snapshot synchronously when
    the plan selects the next round.

Relevance
    Raw positive evidence reported by a source. Relevance alone does not mean a
    rule's own prerequisites have been satisfied.

Precondition (or scheduling prerequisite)
    A tri-state scheduling predicate attached to a candidate. It reads the
    evidence accumulated for rules in the inventory and returns ``satisfied``,
    ``unsatisfied``, or ``unknown``:

    * ``satisfied`` means the candidate is ready to be evaluated;
    * ``unsatisfied`` means current evidence conclusively rules it out;
    * ``unknown`` means outstanding evidence may still make it ready.

    Several preconditions attached to one candidate are combined
    conjunctively: all of them must be satisfied before an ordinary round can
    select that candidate. A precondition performs no I/O and does not evaluate
    the candidate itself; a provider constructs it before execution begins.
    Preconditions only order or avoid evaluation work and do not replace final
    business-rule enforcement. Rule dependencies are one concrete kind of
    precondition.

Satisfied rule
    A relevant rule whose own preconditions are satisfied, or a rule already
    durably discovered on an earlier turn. Only effective satisfaction can
    release candidates that depend on that rule.

Forcing precondition
    A precondition whose ``forces_scheduling`` flag marks it as a correctness
    requirement rather than a load optimization: its presence on any candidate
    enables phased evaluation even below the candidate threshold. The provider
    emitting it is responsible for only doing so where the ordering genuinely
    changes the outcome (see CrossReferenceEvaluationPreconditionProvider,
    which self-gates on the policy-landscape size defined in
    matching/common.py).

Round
    One concurrently executed batch of candidates. Ordering exists between
    rounds, while code and model evaluators selected within a round still fan
    out in parallel.

Evaluation plan
    Turn-local mutable scheduling state. It selects each round, receives the
    resulting evidence, and tracks candidates that were evaluated or skipped.

Flow
----

1. The matcher identifies candidates and counts the evidence sources expected
   to report on each one. It also supplies initial evidence from durable and
   early matches.
2. :class:`EvaluationScheduler` checks the candidate threshold and asks all
   registered providers for applicable preconditions. Provider ``provide``
   methods are asynchronous and run concurrently because they may query stores.
3. If the set is small or no candidate has a precondition, the plan selects all
   candidates in one round, preserving ordinary one-pass execution.
4. Otherwise, the plan initially selects candidates whose preconditions are
   already satisfied. The matcher starts the selected code/model evaluators and
   first-round discovery concurrently, then waits for the entire batch.
5. Once those asynchronous calls complete, the matcher reports each source's
   result through :meth:`EvaluationPlan.observe`. The plan synchronously
   converts the updated evidence snapshot into effective satisfaction, releases
   newly eligible candidates, and starts another asynchronous round. This
   repeats until work is exhausted or ``max_rounds`` is reached.
6. The final permitted round also evaluates candidates whose state remains
   unknown. This bounded fallback covers cycles and dependency chains deeper
   than the configured number of rounds. Candidates known to be unsatisfied
   remain skipped.
7. The matcher merges all results and performs authoritative dependency
   resolution before any rule is durably admitted.

Extending scheduling
--------------------

Scheduling preconditions are deliberately more general than rule dependencies.
To add another reason for ordering evaluations:

1. Implement :class:`EvaluationPrecondition` for the pure condition that
   decides whether a candidate is ready.
2. Implement :class:`EvaluationPreconditionProvider` to derive those conditions
   from rules, relationships, metadata, or another source.
3. Register the provider in :class:`EvaluationPreconditionRegistry` at the
   application's composition root.

Providers only influence when a rule is evaluated. They do not evaluate rules,
admit rules to the discovered set, or replace final policy enforcement. A
provider failure therefore safely degrades to the original one-pass behavior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import AbstractSet, cast

from parlant.core.async_utils import safe_gather
from parlant.core.engines.compass.matching.common import is_small_policy_landscape
from parlant.core.engines.compass.context import EngineContext
from parlant.core.groups import GroupId
from parlant.core.loggers import Logger
from parlant.core.relationships import (
    Relationship,
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.rules import Rule, RuleId
from parlant.core import metadata


@dataclass(frozen=True)
class EvaluationSchedulingPolicy:
    """Bounds when and how aggressively evaluation is split into phases.

    ``threshold`` counts actual evaluation candidates, not the total rule
    inventory. Below it, the scheduler preserves one-pass execution. The
    ``max_rounds`` bound prevents long dependency chains or custom
    preconditions from creating unbounded model-call sequences.
    """

    threshold: int = 8
    max_rounds: int = 2

    def __post_init__(self) -> None:
        if self.threshold < 1:
            raise ValueError("Evaluation scheduling threshold must be positive")
        if self.max_rounds < 1:
            raise ValueError("Evaluation scheduling max_rounds must be positive")


class EvaluationPreconditionState(Enum):
    """Current scheduling state of a prerequisite.

    ``SATISFIED`` means the candidate may be evaluated now.
    ``UNSATISFIED`` means available evidence rules the candidate out.
    ``UNKNOWN`` means a prior evaluation can still change the answer, so the
    candidate should normally wait for a later round.
    """

    SATISFIED = "satisfied"
    UNSATISFIED = "unsatisfied"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvaluationEvidence:
    """Read-only evidence exposed to scheduling preconditions.

    Evidence is rule-oriented because all current evaluators ultimately report
    relevance for rules. A custom precondition may combine these sets however
    it needs, but its ``evaluate`` method should remain pure and inexpensive.

    ``known_rule_ids`` is the complete usable inventory. Referencing a rule
    outside that inventory is treated as unsatisfied rather than indefinitely
    unknown.
    """

    satisfied_rule_ids: AbstractSet[RuleId]
    rejected_rule_ids: AbstractSet[RuleId]
    known_rule_ids: AbstractSet[RuleId]

    def status(self, rule_id: RuleId) -> EvaluationPreconditionState:
        """Return the normalized evidence state for one rule."""

        if rule_id in self.satisfied_rule_ids:
            return EvaluationPreconditionState.SATISFIED
        if rule_id in self.rejected_rule_ids:
            return EvaluationPreconditionState.UNSATISFIED
        if rule_id not in self.known_rule_ids:
            return EvaluationPreconditionState.UNSATISFIED
        return EvaluationPreconditionState.UNKNOWN


class EvaluationPrecondition(ABC):
    """A pure condition controlling when one or more candidates may run.

    Implementations should only interpret ``evidence``. Any I/O or discovery
    needed to construct a condition belongs in an
    :class:`EvaluationPreconditionProvider`, whose ``provide`` method is async.

    ``forces_scheduling`` marks a precondition as a correctness requirement
    rather than a load optimization: its presence on any candidate enables
    phased evaluation even below the policy threshold. Providers emitting
    forcing preconditions are responsible for only doing so where the ordering
    genuinely changes the outcome (see
    :class:`CrossReferenceEvaluationPreconditionProvider`, which self-gates on
    the policy-landscape size so forcing can never produce a pointless round).
    """

    forces_scheduling: bool = False

    @abstractmethod
    def evaluate(self, evidence: EvaluationEvidence) -> EvaluationPreconditionState: ...


class EvaluationPreconditionProvider(ABC):
    """Builds scheduling preconditions for rules.

    ``rules`` contains the full usable inventory, allowing a provider to
    resolve references to rules that are not themselves evaluation candidates
    this turn. The return mapping is keyed by the candidate whose scheduling is
    constrained. Multiple returned conditions for the same rule, including
    conditions from different providers, must all be satisfied.

    Providers may query stores or inspect ``context``. They are prepared in
    parallel whenever there are candidates: forcing preconditions must be
    discoverable below the scheduling threshold too. Phasing itself still only
    engages per the policy threshold unless a forcing precondition applies.
    """

    @abstractmethod
    async def provide(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
    ) -> Mapping[RuleId, Sequence[EvaluationPrecondition]]: ...


class EvaluationPreconditionRegistry:
    """Collection of independent sources of evaluation preconditions.

    Core matching knows only this registry, not concrete provider types. Add
    application-specific providers at the composition root, for example::

        registry = EvaluationPreconditionRegistry([
            DependencyEvaluationPreconditionProvider(relationship_store),
            MyDomainPreconditionProvider(...),
        ])

    ``register`` is also available for environments that assemble extensions
    incrementally before matching begins.
    """

    def __init__(
        self,
        providers: Sequence[EvaluationPreconditionProvider] = (),
    ) -> None:
        self._providers = list(providers)

    def register(self, provider: EvaluationPreconditionProvider) -> None:
        self._providers.append(provider)

    @property
    def providers(self) -> Sequence[EvaluationPreconditionProvider]:
        return tuple(self._providers)


class _RequirementMode(Enum):
    ALL = "all"
    ANY = "any"


@dataclass(frozen=True)
class _RuleRequirement(EvaluationPrecondition):
    """Requires all or any of a concrete set of rule IDs."""

    mode: _RequirementMode
    rule_ids: frozenset[RuleId]

    def evaluate(self, evidence: EvaluationEvidence) -> EvaluationPreconditionState:
        if not self.rule_ids:
            return EvaluationPreconditionState.UNSATISFIED

        statuses = [evidence.status(rule_id) for rule_id in self.rule_ids]
        if self.mode == _RequirementMode.ALL:
            if EvaluationPreconditionState.UNSATISFIED in statuses:
                return EvaluationPreconditionState.UNSATISFIED
            if all(status == EvaluationPreconditionState.SATISFIED for status in statuses):
                return EvaluationPreconditionState.SATISFIED
            return EvaluationPreconditionState.UNKNOWN

        if EvaluationPreconditionState.SATISFIED in statuses:
            return EvaluationPreconditionState.SATISFIED
        if all(status == EvaluationPreconditionState.UNSATISFIED for status in statuses):
            return EvaluationPreconditionState.UNSATISFIED
        return EvaluationPreconditionState.UNKNOWN


@dataclass(frozen=True)
class _AnyRequirement(EvaluationPrecondition):
    """Groups alternatives where one satisfied condition is sufficient."""

    alternatives: tuple[EvaluationPrecondition, ...]

    def evaluate(self, evidence: EvaluationEvidence) -> EvaluationPreconditionState:
        if not self.alternatives:
            return EvaluationPreconditionState.UNSATISFIED

        statuses = [alternative.evaluate(evidence) for alternative in self.alternatives]
        if EvaluationPreconditionState.SATISFIED in statuses:
            return EvaluationPreconditionState.SATISFIED
        if all(status == EvaluationPreconditionState.UNSATISFIED for status in statuses):
            return EvaluationPreconditionState.UNSATISFIED
        return EvaluationPreconditionState.UNKNOWN


@dataclass(frozen=True)
class _DependencyPrecondition(EvaluationPrecondition):
    """Combines all dependency clauses attached to a source rule."""

    requirements: tuple[EvaluationPrecondition, ...]

    def evaluate(self, evidence: EvaluationEvidence) -> EvaluationPreconditionState:
        statuses = [requirement.evaluate(evidence) for requirement in self.requirements]
        if EvaluationPreconditionState.UNSATISFIED in statuses:
            return EvaluationPreconditionState.UNSATISFIED
        if all(status == EvaluationPreconditionState.SATISFIED for status in statuses):
            return EvaluationPreconditionState.SATISFIED
        return EvaluationPreconditionState.UNKNOWN


class DependencyEvaluationPreconditionProvider(EvaluationPreconditionProvider):
    """Translates dependency relationships into scheduling preconditions.

    This is the first concrete provider and an example for future prerequisite
    types. Relationship loading and group expansion happen here; the resulting
    condition objects contain no stores and perform no I/O.

    Ordinary dependencies are conjunctive. ``DEPENDENCY_ANY`` relationships
    sharing a ``group_id`` form one alternative group, while separate groups
    must each be satisfied.
    """

    def __init__(self, relationship_store: RelationshipStore) -> None:
        self._relationship_store = relationship_store

    async def provide(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
    ) -> Mapping[RuleId, Sequence[EvaluationPrecondition]]:
        del context

        relationships = [
            *await self._relationship_store.list_relationships(
                kind=RelationshipKind.DEPENDENCY,
                indirect=False,
            ),
            *await self._relationship_store.list_relationships(
                kind=RelationshipKind.DEPENDENCY_ANY,
                indirect=False,
            ),
        ]
        if not relationships:
            return {}

        rules_by_id = {rule.id: rule for rule in rules}
        rules_by_group: dict[GroupId, set[RuleId]] = defaultdict(set)
        for rule in rules:
            for group_id in rule.groups:
                rules_by_group[group_id].add(rule.id)

        relationships_by_source: dict[RuleId, list[Relationship]] = defaultdict(list)
        for relationship in relationships:
            if relationship.source.kind == RelationshipEntityKind.RULE:
                source_ids = [cast(RuleId, relationship.source.id)]
            elif relationship.source.kind.is_group:
                source_ids = list(rules_by_group.get(cast(GroupId, relationship.source.id), set()))
            else:
                continue

            for source_id in source_ids:
                if source_id in rules_by_id:
                    relationships_by_source[source_id].append(relationship)

        result: dict[RuleId, Sequence[EvaluationPrecondition]] = {}
        for source_id, source_relationships in relationships_by_source.items():
            required: list[EvaluationPrecondition] = []
            any_groups: dict[str, list[EvaluationPrecondition]] = defaultdict(list)

            for relationship in source_relationships:
                target = self._target_requirement(relationship, rules_by_group)
                if relationship.kind == RelationshipKind.DEPENDENCY_ANY and relationship.group_id:
                    any_groups[relationship.group_id].append(target)
                else:
                    required.append(target)

            required.extend(
                _AnyRequirement(tuple(alternatives)) for alternatives in any_groups.values()
            )
            if required:
                result[source_id] = [_DependencyPrecondition(tuple(required))]

        return result

    @staticmethod
    def _target_requirement(
        relationship: Relationship,
        rules_by_group: Mapping[GroupId, set[RuleId]],
    ) -> EvaluationPrecondition:
        if relationship.target.kind == RelationshipEntityKind.RULE:
            return _RuleRequirement(
                mode=_RequirementMode.ALL,
                rule_ids=frozenset([cast(RuleId, relationship.target.id)]),
            )

        if relationship.target.kind.is_group:
            group_id = cast(GroupId, relationship.target.id)
            mode = (
                _RequirementMode.ANY
                if relationship.target.kind == RelationshipEntityKind.GROUP_ANY
                else _RequirementMode.ALL
            )
            return _RuleRequirement(
                mode=mode,
                rule_ids=frozenset(rules_by_group.get(group_id, set())),
            )

        return _RuleRequirement(mode=_RequirementMode.ALL, rule_ids=frozenset())


class _AwaitPolicyContext(EvaluationPrecondition):
    """Defers a cross-referential rule to the final round.

    Always UNKNOWN: never satisfied early (intermediate rounds skip it) and
    never unsatisfied (the final round includes it). By then, earlier rounds'
    evidence has been observed and fresh discovery admissions are pending in
    the matcher state, so the rule can be judged against its policy context."""

    forces_scheduling = True

    def evaluate(self, evidence: EvaluationEvidence) -> EvaluationPreconditionState:
        del evidence
        return EvaluationPreconditionState.UNKNOWN


class CrossReferenceEvaluationPreconditionProvider(EvaluationPreconditionProvider):
    """Defers rules that reference the agent's other policies.

    A rule whose text refers to the rest of the policy set — implicitly ("a
    request we don't support"), by quantification ("any of our policies"), or
    by definite reference ("the return policy") — cannot be judged in
    isolation. Deferring it to the final round lets its evaluator render the
    rule's policy context (discovered rules plus pending admissions) into the
    prompt. The flag is classified at build time and carried in
    ``rule.metadata["references_other_policies"]``."""

    async def provide(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
    ) -> Mapping[RuleId, Sequence[EvaluationPrecondition]]:
        del context

        # Small landscape: the ranker renders the FULL inventory into the
        # flagged rule's prompt, so waiting for discovery buys nothing — emit
        # no precondition at all (this is what makes a pointless deferral
        # round structurally impossible). Large landscape: the prompt uses the
        # bounded discovered-set context, which is only fresh after discovery
        # reports — defer, forcing phased evaluation if needed.
        if is_small_policy_landscape(len(rules)):
            return {}

        return {
            rule.id: [_AwaitPolicyContext()]
            for rule in rules
            if rule.metadata.get("references_other_policies")
            or metadata.internal_value(rule.metadata, ("references_other_policies",), False)
        }


class EvaluationPlan:
    """Mutable, turn-local execution plan consumed one round at a time.

    A plan tracks two related concepts:

    * raw relevance evidence reported by recall/code/model evaluators;
    * effective evidence after each relevant rule's own preconditions are
      applied.

    The distinction prevents a rule from unlocking its dependents merely
    because one evaluator liked it when that rule's own prerequisite was not
    met. Rules already durably discovered are unconditionally satisfied,
    because their prerequisite was established on an earlier turn.

    Positive evidence is sticky for the lifetime of the plan. Negative evidence
    is conclusive only after every expected evidence source for a rule has
    reported, represented by ``pending_evidence_sources``.
    """

    def __init__(
        self,
        *,
        candidates: Sequence[Rule],
        preconditions: Mapping[RuleId, Sequence[EvaluationPrecondition]],
        pending_evidence_sources: Mapping[RuleId, int],
        initially_satisfied_rule_ids: AbstractSet[RuleId],
        initially_relevant_rule_ids: AbstractSet[RuleId],
        known_rule_ids: AbstractSet[RuleId],
        max_rounds: int,
        enabled: bool,
    ) -> None:
        self._candidates = {rule.id: rule for rule in candidates}
        self._preconditions = {
            rule_id: tuple(conditions) for rule_id, conditions in preconditions.items()
        }
        self._pending_evidence_sources = {
            rule_id: count for rule_id, count in pending_evidence_sources.items() if count > 0
        }
        self._unconditionally_satisfied_rule_ids = set(initially_satisfied_rule_ids)
        self._raw_satisfied_rule_ids = {
            *initially_satisfied_rule_ids,
            *initially_relevant_rule_ids,
        }
        self._known_rule_ids = set(known_rule_ids)
        self._max_rounds = max_rounds
        self._enabled = enabled
        self._evaluated_candidate_ids: set[RuleId] = set()
        self._rounds_started = 0
        self._effective_state_cache: Mapping[RuleId, EvaluationPreconditionState] | None = None

    @property
    def enabled(self) -> bool:
        """Whether this plan is actually splitting work into phases."""

        return self._enabled

    @property
    def has_remaining_candidates(self) -> bool:
        """Whether any candidate has neither run nor been finally skipped."""

        return bool(set(self._candidates) - self._evaluated_candidate_ids)

    @property
    def can_start_round(self) -> bool:
        """Whether another bounded evaluation round may be requested."""

        return self.has_remaining_candidates and self._rounds_started < self._max_rounds

    def next_round(self) -> Sequence[Rule]:
        """Select and reserve candidates for the next evaluation round.

        Intermediate rounds contain only candidates whose preconditions are
        satisfied. The last permitted round is deliberately broader and also
        includes unknown candidates. This fallback handles cycles and chains
        deeper than ``max_rounds`` without silently dropping potentially valid
        work. Candidates known to be unsatisfied remain excluded.

        Calling this method reserves the returned rules, so it must be called
        once per round and followed by one or more :meth:`observe` calls.
        """

        if not self.can_start_round:
            return []

        self._rounds_started += 1
        remaining_ids = set(self._candidates) - self._evaluated_candidate_ids

        if not self._enabled:
            selected_ids = remaining_ids
        else:
            states = self._scheduling_states()
            is_final_round = self._rounds_started == self._max_rounds
            if is_final_round:
                selected_ids = {
                    rule_id
                    for rule_id in remaining_ids
                    if states.get(rule_id) != EvaluationPreconditionState.UNSATISFIED
                }
            else:
                selected_ids = {
                    rule_id
                    for rule_id in remaining_ids
                    if states.get(rule_id) == EvaluationPreconditionState.SATISFIED
                }

        self._evaluated_candidate_ids.update(selected_ids)
        return [rule for rule_id, rule in self._candidates.items() if rule_id in selected_ids]

    def observe(
        self,
        checked_rule_ids: AbstractSet[RuleId],
        relevant_rule_ids: AbstractSet[RuleId],
    ) -> None:
        """Incorporate one evaluator's results after a round.

        A rule can be checked by more than one source, such as recall and rank.
        ``pending_evidence_sources`` records how many such opinions remain.
        Consequently an irrelevant result does not reject a rule while another
        source is outstanding, whereas any relevant result is immediately
        useful and remains so.
        """

        # Evidence is the only input to the effective-state fixed point that
        # changes after plan construction. Round reservation does not affect it,
        # so the snapshot computed by ``next_round`` can be reused by prompt
        # preparation until an observation lands.
        self._effective_state_cache = None
        self._raw_satisfied_rule_ids.update(relevant_rule_ids)

        for rule_id in checked_rule_ids:
            remaining = self._pending_evidence_sources.get(rule_id, 0)
            if remaining <= 1:
                self._pending_evidence_sources.pop(rule_id, None)
            else:
                self._pending_evidence_sources[rule_id] = remaining - 1

    def effective_rule_states(self) -> Mapping[RuleId, EvaluationPreconditionState]:
        """Return the current relevance states after applying each rule's own
        preconditions.

        This is the same evidence snapshot used to decide the next evaluation
        round. Callers may use it to prepare evaluator context, but must not
        treat a raw positive verdict with unmet preconditions as satisfied.
        """
        # Do not expose the cached mutable dictionary itself to callers.
        return dict(self._effective_states())

    def skipped_rules(self) -> Sequence[Rule]:
        """Return candidates left unscheduled after all rounds are exhausted."""

        if self.can_start_round:
            return []
        remaining_ids = set(self._candidates) - self._evaluated_candidate_ids
        return [rule for rule_id, rule in self._candidates.items() if rule_id in remaining_ids]

    def _scheduling_states(self) -> Mapping[RuleId, EvaluationPreconditionState]:
        effective_states = self._effective_states()
        evidence = EvaluationEvidence(
            satisfied_rule_ids={
                rule_id
                for rule_id, state in effective_states.items()
                if state == EvaluationPreconditionState.SATISFIED
            },
            rejected_rule_ids={
                rule_id
                for rule_id, state in effective_states.items()
                if state == EvaluationPreconditionState.UNSATISFIED
            },
            known_rule_ids=self._known_rule_ids,
        )

        states: dict[RuleId, EvaluationPreconditionState] = {}
        for rule_id in self._candidates:
            conditions = self._preconditions.get(rule_id, ())
            if not conditions:
                states[rule_id] = EvaluationPreconditionState.SATISFIED
                continue

            condition_states = [condition.evaluate(evidence) for condition in conditions]
            if EvaluationPreconditionState.UNSATISFIED in condition_states:
                states[rule_id] = EvaluationPreconditionState.UNSATISFIED
            elif all(state == EvaluationPreconditionState.SATISFIED for state in condition_states):
                states[rule_id] = EvaluationPreconditionState.SATISFIED
            else:
                states[rule_id] = EvaluationPreconditionState.UNKNOWN

        return states

    def _effective_states(self) -> Mapping[RuleId, EvaluationPreconditionState]:
        if self._effective_state_cache is not None:
            return self._effective_state_cache

        raw_rejected_ids = {
            rule_id
            for rule_id in self._known_rule_ids
            if rule_id not in self._raw_satisfied_rule_ids
            and self._pending_evidence_sources.get(rule_id, 0) == 0
        }
        states: dict[RuleId, EvaluationPreconditionState] = {
            rule_id: (
                EvaluationPreconditionState.SATISFIED
                if rule_id in self._raw_satisfied_rule_ids
                else EvaluationPreconditionState.UNSATISFIED
                if rule_id in raw_rejected_ids
                else EvaluationPreconditionState.UNKNOWN
            )
            for rule_id in self._known_rule_ids
        }

        for _ in range(len(self._known_rule_ids) + 1):
            changed = False
            evidence = EvaluationEvidence(
                satisfied_rule_ids={
                    rule_id
                    for rule_id, state in states.items()
                    if state == EvaluationPreconditionState.SATISFIED
                },
                rejected_rule_ids={
                    rule_id
                    for rule_id, state in states.items()
                    if state == EvaluationPreconditionState.UNSATISFIED
                },
                known_rule_ids=self._known_rule_ids,
            )

            for rule_id in self._known_rule_ids:
                conditions = self._preconditions.get(rule_id, ())
                if rule_id in self._unconditionally_satisfied_rule_ids:
                    next_state = EvaluationPreconditionState.SATISFIED
                elif rule_id in raw_rejected_ids:
                    next_state = EvaluationPreconditionState.UNSATISFIED
                elif not conditions:
                    next_state = (
                        EvaluationPreconditionState.SATISFIED
                        if rule_id in self._raw_satisfied_rule_ids
                        else EvaluationPreconditionState.UNKNOWN
                    )
                else:
                    condition_states = [condition.evaluate(evidence) for condition in conditions]
                    if EvaluationPreconditionState.UNSATISFIED in condition_states:
                        next_state = EvaluationPreconditionState.UNSATISFIED
                    elif rule_id in self._raw_satisfied_rule_ids and all(
                        state == EvaluationPreconditionState.SATISFIED for state in condition_states
                    ):
                        next_state = EvaluationPreconditionState.SATISFIED
                    else:
                        next_state = EvaluationPreconditionState.UNKNOWN

                if states.get(rule_id) != next_state:
                    states[rule_id] = next_state
                    changed = True

            if not changed:
                break

        self._effective_state_cache = states
        return states


class EvaluationScheduler:
    """Creates bounded evaluation plans from registered prerequisite sources.

    The scheduler has no knowledge of code evaluators, rankers, distillers, or
    recallers. The matcher tells the plan which sources are expected and feeds
    their observations back after each round.
    """

    def __init__(
        self,
        logger: Logger,
        precondition_registry: EvaluationPreconditionRegistry,
        policy: EvaluationSchedulingPolicy,
    ) -> None:
        self._logger = logger
        self._precondition_registry = precondition_registry
        self._policy = policy

    async def create_plan(
        self,
        context: EngineContext,
        *,
        candidates: Sequence[Rule],
        rules: Sequence[Rule],
        pending_evidence_sources: Mapping[RuleId, int],
        initially_satisfied_rule_ids: AbstractSet[RuleId],
        initially_relevant_rule_ids: AbstractSet[RuleId] = frozenset(),
    ) -> EvaluationPlan:
        """Prepare a plan for the current matcher batch.

        ``candidates`` are rules that will actually be evaluated. ``rules`` is
        the complete usable inventory used by providers to resolve references.

        ``pending_evidence_sources`` counts the evaluators expected to report
        on each rule. ``initially_satisfied_rule_ids`` contains durable or
        otherwise authoritative prior matches. ``initially_relevant_rule_ids``
        contains fresh relevance that must still pass its own preconditions.

        Precondition preparation runs whenever there are candidates: a
        forcing precondition can engage phasing even below the candidate
        threshold (and clamps a single-round policy to two rounds), because
        it encodes a correctness requirement, not an optimization. If no
        provider constrains an actual candidate, or if provider preparation
        fails, the returned plan preserves one-pass behavior.
        """

        known_rule_ids = {rule.id for rule in rules}
        enabled = len(candidates) >= self._policy.threshold
        preconditions: dict[RuleId, list[EvaluationPrecondition]] = defaultdict(list)

        if candidates and self._precondition_registry.providers:
            try:
                provided = await safe_gather(
                    *(
                        provider.provide(context, rules)
                        for provider in self._precondition_registry.providers
                    )
                )
                for provider_result in provided:
                    for rule_id, conditions in provider_result.items():
                        preconditions[rule_id].extend(conditions)
            except Exception as exc:
                with self._logger.scope("EvaluationScheduler"):
                    self._logger.warning(
                        f"Evaluation precondition preparation failed; using one pass: {exc}"
                    )
                preconditions.clear()

        constrained = any(rule.id in preconditions for rule in candidates)
        forced = any(
            condition.forces_scheduling
            for rule in candidates
            for condition in preconditions.get(rule.id, ())
        )

        if forced and not enabled and constrained:
            # Phasing engaged by a correctness precondition alone: the
            # candidate count would have kept this one-pass. Logged so
            # unexpected multi-round turns are attributable.
            with self._logger.scope("EvaluationScheduler"):
                self._logger.debug(
                    f"Evaluation scheduling force-enabled by a correctness precondition "
                    f"({len(candidates)} candidate(s), below threshold {self._policy.threshold})"
                )

        enabled = (enabled or forced) and constrained

        max_rounds = self._policy.max_rounds
        if enabled and forced and max_rounds < 2:
            # A forcing precondition is correctness, not optimization: it
            # defers its rule to the FINAL round so round-1 evidence can be
            # absorbed first. With a single-round policy the final round IS
            # round 1 — no deferral at all — so correctness overrides the
            # configured bound here.
            max_rounds = 2
            with self._logger.scope("EvaluationScheduler"):
                self._logger.debug(
                    "Evaluation scheduling round limit clamped to 2 by a correctness "
                    f"precondition (policy max_rounds={self._policy.max_rounds})"
                )

        return EvaluationPlan(
            candidates=candidates,
            preconditions=preconditions,
            pending_evidence_sources=pending_evidence_sources,
            initially_satisfied_rule_ids=initially_satisfied_rule_ids,
            initially_relevant_rule_ids=initially_relevant_rule_ids,
            known_rule_ids=known_rule_ids,
            max_rounds=max_rounds,
            enabled=enabled,
        )
