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

"""The per-rule turn evaluator: scores one rule's relevance per prompt.

Terminology (full definitions in ``matching/common.py``):

- **Policy context**: for a cross-referential rule (one whose text references
  the agent's other policies), the set of other rules rendered into its
  evaluation prompt so its references can be resolved. In a small policy
  landscape (inventory <= MAX_FULL_POLICY_CONTEXT_RULES) this is the FULL
  inventory; in a large one it is the discovered working set plus this turn's
  pending admissions — kept fresh by the evaluation scheduler deferring such
  rules to the final round (see CrossReferenceEvaluationPreconditionProvider).
"""

from enum import Enum
import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence
from typing_extensions import override

from parlant.core.agents import Effort
from parlant.core.common import Weight, DefaultBaseModel, JSONSerializable, to_prompt_json
from parlant.core.engines.compass.tracing import CompassTracer
from parlant.core.engines.compass.prompt_builder import (
    BuiltInSection,
    EventAdaptationFormat,
    PromptBuilder,
    SectionStatus,
)
from parlant.core.engines.compass.matching.common import (
    add_agent_reasoning,
    add_dependency_context,
    add_policy_context,
    add_rule_terms,
    aggregate_generation_info,
)
from parlant.core.engines.compass.matching.rule_evaluation import (
    RuleEvaluation,
    RuleEvaluationResult,
    TurnEvaluator,
)
from parlant.core.engines.compass.context import EngineContext
from parlant.core.rules import Rule, RuleId
from parlant.core.loggers import Logger
from parlant.core.nlp.decision import (
    DecisionMaker,
    Proposition,
    PropositionAnswer,
    PropositionCriteria,
)
from parlant.core.nlp.generation import HEDGE_TIMEOUT_HINT
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.nlp.prompt_routing import PromptConfigurationService
from parlant.core.nlp.model_routing import ModelRouter
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.shots import Shot, ShotCollection
from parlant.core.tracer import Tracer


class PromptSections(str, Enum):
    """The ranker prompt's section names — public API for prompt providers.

    Every module that names prompt sections exposes a ``PromptSections`` enum
    beside the ``add_section`` calls it names; the registry test in
    tests/core/stable/engines/compass/test_prompt_sections.py enforces that
    no compass prompt contains an unregistered name.
    """

    GENERAL_INSTRUCTIONS = "ranker.general-instructions"
    TASK_DESCRIPTION = "ranker.task-description"
    EXAMPLES = "ranker.examples"
    OUTPUT_FORMAT = "ranker.output-format"
    CURRENT_TURN = "ranker.current-turn"
    POLICY_CONTEXT = "ranker.policy-context"
    DEPENDENCY_CONTEXT = "ranker.dependency-context"


def get_dynamic_reasoning_effort_for_matching(context: EngineContext) -> str:
    """Map the context's dynamic effort level to a model ``reasoning_effort`` hint."""
    effort = context.state.dynamic_effort_level

    # Note that low effort has higher reasoning than medium,
    # because the assumption is that medium effort comes
    # with added ARQs in its output anyway.

    match effort:
        case Effort.MIN:
            return "minimal"
        case Effort.LOW:
            return "low"
        case Effort.MEDIUM:
            return "minimal"
        case Effort.HIGH:
            return "low"
        case Effort.MAX:
            return "low"


# The relevance scale, lowest to highest. The prompt renders these as a numbered
# 1-5 list; a DecisionMaker takes them as a Scale's criteria. One source, so the
# scale the model is asked to use cannot drift between the two paths.
RELEVANCE_CRITERIA = (
    "not relevant at all, at no point in the conversation",
    "only very loosely related to the subject of the conversation",
    "partially relevant: was previously relevant but not clearly right now, or only part of the rule matches",
    "requires further checking by the human: a partial match, but one that clearly justifies further inspection",
    "clearly matches",
)


class RuleRankSchema(DefaultBaseModel):
    tldr: str | None = None
    s: int


@dataclass(frozen=True)
class ShotRule:
    """The authored text of a shot's rule (identity-free)."""

    condition: str | None = None
    description: str | None = None


@dataclass
class RuleRankingShot(Shot):
    interaction_events: Sequence[Event]
    # The ranker evaluates a single rule per prompt, so each shot carries one.
    rule: ShotRule
    expected_result: RuleRankSchema


class RuleRanker(TurnEvaluator):
    """First-pass rule filter.

    Before the full rule-matching stages, a smaller language model is used to
    cheaply filter out rules that do not apply for sure. Each rule is
    evaluated in its own prompt; rules that pass this filter go on to the
    later stages of message generation.
    """

    RELEVANCE_SCORE_THRESHOLD = 4
    # The decision path's equivalent: a rule is carried forward when it is at
    # least this likely to be relevant. Deliberately at the midpoint — this is a
    # first-pass filter, where a false negative costs a missed rule and a false
    # positive only costs a later stage some attention.
    RELEVANCE_PROBABILITY_THRESHOLD = 0.5
    RULE_CACHE_BREAKPOINT = "- Rule: ###"
    REASONING_CACHE_BREAKPOINT = "AGENT'S REASONING SO FAR THIS TURN"
    STAGED_EVENTS_CACHE_BREAKPOINT = "STAGED EVENTS"
    CURRENT_TURN_CACHE_BREAKPOINT = "# CURRENT TURN"

    # Message sources counted as "the agent's reply" — the boundary the cached
    # interaction history is truncated at (everything after is this turn's tail).
    _AGENT_MESSAGE_SOURCES = (
        EventSource.AI_AGENT,
        EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT,
        EventSource.HUMAN_AGENT,
    )

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        model_router: ModelRouter,
        prompt_config: PromptConfigurationService,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._model_router = model_router
        self._prompt_config = prompt_config

    async def evaluate(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
    ) -> RuleEvaluationResult:
        if not rules:
            return RuleEvaluationResult([], None)

        with self._logger.scope("RuleRanker"), self._tracer.span("compass.rule_ranker.rank"):
            t_start = time.monotonic()

            # Resolve a decision model only where the effort could use one, so a
            # high-effort turn never pays for a lookup it cannot act on.
            if self._may_use_decisions(context):
                if decision_maker := await self._resolve_decision_maker(context):
                    evaluations, info = await self._decide_rules(context, rules, decision_maker)
                    CompassTracer(self._tracer).rules_ranked(evaluations)
                    return RuleEvaluationResult(evaluations=evaluations, generation_info=info)

            results = await asyncio.gather(*(self._rank_rule(context, rule) for rule in rules))

            t_end = time.monotonic()

            evaluations = [evaluation for evaluation, _ in results]
            CompassTracer(self._tracer).rules_ranked(evaluations)
            return RuleEvaluationResult(
                evaluations=evaluations,
                generation_info=aggregate_generation_info(
                    [info for _, info in results],
                    total_duration=t_end - t_start,
                ),
            )

    @override
    async def evaluates_batch_in_one_request(self, context: EngineContext) -> bool:
        """True exactly when `evaluate` would take the batched decision path.

        Deliberately the same two conditions `evaluate` checks, in the same
        order, so the matcher's strategy choice and the path actually taken
        cannot disagree."""
        return self._may_use_decisions(context) and (
            await self._resolve_decision_maker(context) is not None
        )

    def _may_use_decisions(self, context: EngineContext) -> bool:
        """Whether this turn's effort allows asking its rules as one batched decision.

        Allowed exactly where no per-rule explanation is wanted, since a decision
        model returns no prose. Deriving it from the tldr gate rather than
        repeating an effort list keeps the two from drifting apart."""
        return not self._should_include_tldr(context)

    async def _resolve_decision_maker(self, context: EngineContext) -> DecisionMaker | None:
        """The decision model configured for the ranker's route, if any.

        Configuration decides this, in the same place every other model is chosen:
        `matching.rank[capability=decision]`. With nothing configured the router
        returns None and the per-rule fan-out stands."""
        return await self._model_router.get_decision_maker(
            context=context,
            route="matching.rank",
            qualifiers=context.state.model_route_qualifiers,
        )

    async def _decide_rules(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
        decision_maker: DecisionMaker,
    ) -> tuple[list[RuleEvaluation], GenerationInfo]:
        """Score every rule in the batch with a single decision.

        The conversation and its context become the state, and each rule becomes
        one Scale question over the same relevance criteria the prompt renders.
        Since the state is shared, this costs one round trip for the whole batch
        rather than one per rule."""
        state = self._build_state_prompt(context, await self.shots(), for_decision=True).build()
        questions = {str(rule.id): self._rule_question(context, rule) for rule in rules}

        decision = await decision_maker.decide(state, questions)

        evaluations: list[RuleEvaluation] = []

        for rule in rules:
            answer = decision.answers[str(rule.id)]

            if not isinstance(answer, PropositionAnswer):
                raise TypeError(
                    f"Expected a proposition answer for rule {rule.id}, got {type(answer).__name__}"
                )

            # The probability is already calibrated, so it needs no rescaling — it
            # goes straight onto the evaluation's score and through the threshold.
            probability = answer.probability

            evaluations.append(
                RuleEvaluation(
                    rule=rule,
                    reasoning=(
                        f"This rule was judged {probability:.0%} likely to be relevant "
                        "to your next response."
                    ),
                    is_relevant=probability >= self.RELEVANCE_PROBABILITY_THRESHOLD,
                    score=probability,
                )
            )

        return evaluations, decision.info

    def _rule_question(self, context: EngineContext, rule: Rule) -> Proposition:
        """One rule as a proposition, carrying the same per-rule context the
        fan-out renders into its prompt tail — the rule's terms, its policy
        context, and its dependency context — so a verdict here sees what a
        verdict there would.

        A rule's signals and anti-signals become the true and false criteria.
        They are the same exemplars the recaller trains on, but used differently:
        the recaller can only measure resemblance to them, while a decision model
        can be told to read them as illustrations of the criterion. The framing
        below says exactly that, because exemplars treated as patterns to match
        would just reproduce the recall verdict this stage exists to second-guess."""
        extras = PromptBuilder()

        add_rule_terms(
            extras,
            [
                term
                for term in context.state.terms_by_rule.get(rule.id, [])
                if term not in context.state.glossary_terms
            ],
        )
        add_policy_context(
            extras,
            context,
            rule,
            lambda other: _format_rule(other.title, other.condition, other.content),
            section_name=PromptSections.POLICY_CONTEXT,
        )
        add_dependency_context(
            extras,
            context,
            rule,
            lambda target: _format_rule(target.title, target.condition, target.content),
            section_name=PromptSections.DEPENDENCY_CONTEXT,
        )

        rule_text = _format_rule(rule.title, rule.condition, rule.content)
        rendered_extras = extras.build().strip()

        return Proposition(
            instructions={
                "rule": f"{rule_text}\n\n{rendered_extras}" if rendered_extras else rule_text,
                "question": (
                    "Does the `rule` apply to the current state of the conversation, "
                    "and above all to its most recent part? This is a first-pass "
                    "filter: a later stage makes the final call, so when genuinely "
                    "torn, lean towards yes."
                ),
            },
            criteria=self._rule_criteria(rule),
        )

    def _rule_criteria(self, rule: Rule) -> PropositionCriteria | None:
        """The rule's exemplars as the boundary between yes and no.

        None when the rule carries neither, since the instructions alone are
        enough for most propositions."""
        if not rule.signals and not rule.anti_signals:
            return None

        guidance = (
            "These are illustrations of the kind of situation the rule covers, not "
            "patterns to match. A conversation can fit the rule without resembling "
            "any of them, and can resemble one without fitting the rule — judge the "
            "rule itself."
        )

        return PropositionCriteria(
            true={
                "applies": "The rule governs the conversation as it now stands.",
                "example_messages": list(rule.signals),
                "how_to_read_the_examples": guidance,
            }
            if rule.signals
            else "The rule governs the conversation as it now stands.",
            false={
                "does_not_apply": "The rule does not govern the conversation as it now stands.",
                "example_messages": list(rule.anti_signals),
                "how_to_read_the_examples": guidance,
            }
            if rule.anti_signals
            else "The rule does not govern the conversation as it now stands.",
        )

    @staticmethod
    def _verdict_instruction(for_decision: bool) -> str:
        """How the ranker asks for its verdict, in the terms the model can answer.

        A decision model returns a probability per rule, so the 1-5 scale would be
        an answer format it cannot produce — and describing one answer format while
        asking for another is two questions at once."""
        if for_decision:
            return (
                "For each rule you are asked about, decide whether it applies to the "
                "most recent state of the conversation. When in doubt, lean towards "
                "yes - the human will make the final call."
            )

        criteria = "\n".join(
            f"- {level} - {criterion}"
            for level, criterion in enumerate(RELEVANCE_CRITERIA, start=1)
        )
        return (
            "Output an integer score from 1 to 5 indicating how relevant the rule is "
            f"to the most recent state of the conversation:\n{criteria}\n\n"
            "Scores of 1 and 2 mean the rule will be filtered out for sure; 3 means "
            "maybe; 4 and 5 mean it should continue on to the next stages. When in "
            "doubt, lean towards a higher score - the human will make the final call."
        )

    def _should_include_tldr(self, context: EngineContext) -> bool:
        match context.state.dynamic_effort_level:
            case Effort.MIN:
                return False
            case Effort.LOW:
                return False
            case Effort.MEDIUM:
                return False
            case Effort.HIGH:
                return True
            case Effort.MAX:
                return True

    async def _rank_rule(
        self,
        context: EngineContext,
        rule: Rule,
    ) -> tuple[RuleEvaluation, GenerationInfo]:
        prompt = self._build_prompt(context, rule, shots=await self.shots())
        route = "matching.rank"
        qualifiers = context.state.model_route_qualifiers

        generator = await self._model_router.get_schematic_generator(
            context=context,
            route=route,
            qualifiers=qualifiers,
            schema=RuleRankSchema,
        )
        reasoning_effort = await self._model_router.resolve_schematic_reasoning_effort(
            context=context,
            route=route,
            qualifiers=qualifiers,
        ) or get_dynamic_reasoning_effort_for_matching(context)
        hints: dict[str, Any] = {
            "reasoning_effort": reasoning_effort,
            "cache": {
                "key": self._cache_key(context),
                "breakpoint": self._cache_breakpoint(context),
            },
        }
        hedge_timeout = context.dependencies.optimization_policy.get_hedge_timeout(
            route,
            qualifiers=qualifiers,
        )
        if hedge_timeout is not None:
            hints[HEDGE_TIMEOUT_HINT] = hedge_timeout

        inference = await generator.generate(
            prompt=prompt,
            hints=hints,
        )

        score = inference.content.s

        # A score of 1-2 means "filtered out for sure"; 3+ ("maybe" and up) passes
        # the first-pass filter on to the next stages.
        return (
            RuleEvaluation(
                rule=rule,
                reasoning=inference.content.tldr
                or f"This rule ranked {score * 2} out of 10 in relevance to your next response.",
                is_relevant=score >= self.RELEVANCE_SCORE_THRESHOLD,
                score=float(score) / 5.0,
            ),
            inference.info,
        )

    def _build_reasoning(self, output: RuleRankSchema) -> str:
        if output.tldr:
            match output.s:
                case s if s < 1:
                    return f"NOT relevant. {output.tldr}"
                case 1 | 2:
                    return f"NOT relevant. {output.tldr}"
                case 3:
                    return f"Possibly relevant. {output.tldr}"
                case 4:
                    return f"Quite relevant. {output.tldr}"
                case 5:
                    return f"Very relevant. {output.tldr}"
                case _:
                    return f"Relevance score overflow: {output.s * 2} out of 10. {output.tldr}"
        else:
            return f"This rule ranked {output.s * 2} out of 10 in relevance to your next response."

    def _cache_key(self, context: EngineContext) -> str:
        # Namespace the provider cache per session AND component, so components
        # that cache concurrently (e.g. within a matching batch) never clobber a
        # shared entry.
        return f"{context.session.id}.rule-ranker"

    def _cache_breakpoint(self, context: EngineContext) -> str:
        # The shared prefix always ends right before the per-turn "# CURRENT TURN"
        # section, so the cache boundary is fixed — independent of which per-turn
        # sections (latest message, staged events, reasoning) happen to be present.
        return self.CURRENT_TURN_CACHE_BREAKPOINT

    async def warm_up(self, context: EngineContext) -> GenerationInfo | None:
        """Warm the generator's cache for the ranker's shared prompt prefix.

        `rank` fans out one request per rule concurrently, each repeating the
        shared prefix. The cache hint lets providers that support explicit caching
        cache everything before the first dynamic tail section, so the
        per-rule fan-out sends only the live suffix. Providers without explicit
        caching ignore the hint. The same reasoning_effort is used so the cached
        variant matches. Best-effort: warming failures must not break preparation."""
        with self._logger.scope("RuleRanker"), self._tracer.span("compass.rule_ranker.prefill"):
            try:
                prompt = self._build_prompt(
                    context,
                    rule=self._cache_prefill_rule(),
                    shots=await self.shots(),
                )
                route = "matching.rank"
                qualifiers = context.state.model_route_qualifiers
                generator = await self._model_router.get_schematic_generator(
                    context=context,
                    route=route,
                    qualifiers=qualifiers,
                    schema=RuleRankSchema,
                )
                reasoning_effort = await self._model_router.resolve_schematic_reasoning_effort(
                    context=context,
                    route=route,
                    qualifiers=qualifiers,
                ) or get_dynamic_reasoning_effort_for_matching(context)
                inference = await generator.generate(
                    prompt=prompt,
                    hints={
                        "reasoning_effort": reasoning_effort,
                        "cache": {
                            "key": self._cache_key(context),
                            "breakpoint": self._cache_breakpoint(context),
                        },
                    },
                )
                return inference.info
            except Exception as exc:
                self._logger.warning(f"Rule ranker prefill failed (continuing): {exc}")
                return None

    async def shots(self) -> Sequence[RuleRankingShot]:
        return await shot_collection.list()

    def _cache_prefill_rule(self) -> Rule:
        return Rule(
            id=RuleId("cache-prefill-rule"),
            creation_utc=datetime.now(timezone.utc),
            modified_utc=datetime.now(timezone.utc),
            content="Return a low relevance score.",
            condition="the cache is being warmed before rule ranking",
            enabled=True,
            groups=[],
            metadata={},
            weight=Weight.LOW,
        )

    def _format_shots(self, context: EngineContext, shots: Sequence[RuleRankingShot]) -> str:
        return "\n".join(
            f"Example #{i}: ###\n{self._format_shot(context, shot)}"
            for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, context: EngineContext, shot: RuleRankingShot) -> str:
        def adapt_event(e: Event) -> JSONSerializable:
            source_map: dict[EventSource, str] = {
                EventSource.CUSTOMER: "user",
                EventSource.CUSTOMER_UI: "frontend_application",
                EventSource.HUMAN_AGENT: "human_service_agent",
                EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT: "ai_agent",
                EventSource.AI_AGENT: "ai_agent",
                EventSource.SYSTEM: "system-provided",
            }

            return {
                "event_kind": e.kind.value,
                "event_source": source_map[e.source],
                "data": e.data,
            }

        formatted_shot = ""
        if shot.interaction_events:
            formatted_shot += f"""
- **Interaction Events**:
{to_prompt_json([adapt_event(e) for e in shot.interaction_events])}

"""

        formatted_shot += f"""
- **Rule**:
{_format_rule(None, shot.rule.condition, shot.rule.description)}

"""

        expected_result = shot.expected_result.model_dump(mode="json")
        if not self._should_include_tldr(context) or expected_result.get("tldr") is None:
            expected_result.pop("tldr", None)

        formatted_shot += f"""
- **Expected Result**:
```json
{to_prompt_json(expected_result)}
```
"""

        return formatted_shot

    def _build_prompt(
        self,
        context: EngineContext,
        rule: Rule,
        shots: Sequence[RuleRankingShot],
    ) -> PromptBuilder:
        # Everything that isn't about this particular rule, then the per-rule tail.
        builder = self._build_state_prompt(context, shots)
        return self._build_rule_tail(builder, context, rule)

    def _build_state_prompt(
        self,
        context: EngineContext,
        shots: Sequence[RuleRankingShot],
        *,
        for_decision: bool = False,
    ) -> PromptBuilder:
        """The ranker's view of the turn, with nothing about any one rule in it.

        The fan-out appends one rule to this and asks about it; the batched
        decision path sends it once as the state and asks about every rule.

        Starts from the cross-turn-stable shared prefix, then appends the
        turn-varying tail. That tail opens with a fixed "# CURRENT TURN" marker
        (the cache breakpoint); everything before it is the cacheable prefix."""
        builder = self._build_shared_prompt(context, shots, for_decision=for_decision)

        # The customer's latest message(s) were split out of the cached history (see
        # `_build_shared_prompt`); render them here, in the live suffix, so the cached
        # prefix stays identical from one turn's `prefill` to the next turn's matching.
        _, trailing_events = self._split_interaction_at_last_agent_reply(context.interaction.events)
        builder.add_section(
            name=PromptSections.CURRENT_TURN,
            template="""
# CURRENT TURN
{current_turn_text}
""",
            props={"current_turn_text": self._format_current_turn(builder, trailing_events)},
            status=SectionStatus.ACTIVE,
        )

        # Per-step reasoning goes in the tail (not the cached shared prefix) so the
        # cache stays valid while the matching tracks the agent's evolving reasoning.
        add_agent_reasoning(builder, context.state.reasoning_steps)

        # TODO: It's problematic that tool events aren't on a shared timeline
        # with the reasoning steps. Fix this at some point.
        builder.add_staged_tool_events(context.state.tool_events)

        return builder

    def _build_rule_tail(
        self,
        builder: PromptBuilder,
        context: EngineContext,
        rule: Rule,
    ) -> PromptBuilder:
        """The per-rule sections: the rule itself, plus the context needed to
        read it correctly."""
        # Terms this rule depends on for correct interpretation - rendered in
        # the live tail with the rule, skipping terms already present in the
        # (cached) shared-prefix glossary section.
        add_rule_terms(
            builder,
            [
                term
                for term in context.state.terms_by_rule.get(rule.id, [])
                if term not in context.state.glossary_terms
            ],
        )

        # This rule's condition refers to the agent's other policies, so its
        # policy context is rendered for the verdict to resolve those
        # references against. See matching/common.py: "Policy context".
        # Turn-varying, so it lives in the live tail, never the cached prefix.
        add_policy_context(
            builder,
            context,
            rule,
            lambda other: _format_rule(
                other.title,
                other.condition,
                other.content,
            ),
            section_name=PromptSections.POLICY_CONTEXT,
        )

        # Render the dependency context prepared from scheduler evidence:
        # normally satisfied targets only, or every target as a clearly marked
        # fallback when any remain indeterminate. Per-rule, so it lives in the
        # live tail, never the prefix.
        add_dependency_context(
            builder,
            context,
            rule,
            lambda target: _format_rule(
                target.title,
                target.condition,
                target.content,
            ),
            section_name=PromptSections.DEPENDENCY_CONTEXT,
        )

        builder.add_section(
            name=BuiltInSection.RULES,
            template="""
- Rule: ###
{rule_text}
###
""",
            props={
                "rule_text": _format_rule(
                    rule.title,
                    rule.condition,
                    rule.content,
                ),
            },
            status=SectionStatus.ACTIVE,
        )

        return builder

    def _build_shared_prompt(
        self,
        context: EngineContext,
        shots: Sequence[RuleRankingShot],
        *,
        for_decision: bool = False,
    ) -> PromptBuilder:
        """The cross-turn-stable head of the ranker prompt (instructions, shots,
        agent + customer identity). Excludes turn-varying context, so it stays
        byte-identical across turns — the prefix `prefill` warms."""
        builder = self._prompt_config.create_builder(
            "matching.rank",
            agent_id=context.agent.id,
            qualifiers=context.state.model_route_qualifiers,
        )

        builder.add_section(
            name=PromptSections.GENERAL_INSTRUCTIONS,
            template="""
# GENERAL INSTRUCTIONS

In our system, a conversational multi-turn AI agent's behavior is controlled by a set of rules.
Some rules are condition/action instructions: when the condition applies, the agent is directed to take its associated action.
Other rules are policy-style instructions: they may have no condition or action, and their title and policy text describe the rule, constraint, or information that may govern the conversation.
""",
            props={},
        )
        builder.add_section(
            name=PromptSections.TASK_DESCRIPTION,
            template="""
# Task Description

Act as a first-pass filter to screen out clearly irrelevant rules. You are NOT making the final determination - a human judge will review every rule that isn't clearly irrelevant and decide whether it actually applies.

{verdict_instruction}

You will be given the context of the customer, the interaction so far (including tool calls and results), contextual information (variables) about the customer and the current interaction session, and other information. You shall use all of that information to evaluate the relevance of the rule to the current state of the interaction, but you should focus more on the most recent state of the interaction.

Important considerations:
1. Focus on recency: Evaluate relevance based on the latest part of the conversation, particularly the most recent customer message.
2. Semantic evaluation: Assess the actual meaning of the rule, not just keyword matching.
3. Context matters: Consider the full context and intent behind the user's message, including tool results, context variables, etc'.
4. For condition/action rules, do not rank solely because the action sounds useful; rank by whether the condition applies. The action and details may be used to interpret the condition.
5. For policy-style rules without a condition, rank by whether the title and policy/details text govern the current situation, the customer's request, the agent's next response, or a tool/result currently in play.
6. Match based on entire context: a rule may be matched based on the entire context of the interaction. It may be relevant even if it has nothing to do with the latest customer message.
7. For a condition/action rule whose action the agent has ALREADY taken: the rule still applies if its purpose is still pending on the customer - the agent's question remains unanswered, a required detail was never provided, or the customer pushed back on HOW something is done without abandoning the underlying intent - so score it 4-5. Such a rule stops applying only once the customer has completed their side or clearly moved on from the intent. (This is about the agent's pending ask specifically - it does not narrow condition-only rules that match on something having happened at ANY point in the conversation, per consideration 6.)
""",
            props={"verdict_instruction": self._verdict_instruction(for_decision)},
        )
        builder.add_section(
            name=PromptSections.EXAMPLES,
            template="""
#Examples of Rule Ranking Evaluations:

{formatted_shots}
""",
            props={
                "formatted_shots": self._format_shots(context, shots),
                "shots": shots,
            },
        )

        builder.add_agent_identity(context.agent)
        builder.add_customer_identity(context.customer, context.session)

        # Without a tldr field to capture its reasoning, the model tends to append
        # an unsolicited explanation after the JSON; tell it not to so we don't pay
        # for output tokens we discard. Kept as the final note (recency).
        # (Providers that enforce structured output ignore this harmlessly.)
        json_only_note = (
            ""
            if self._should_include_tldr(context)
            else "\nRespond with ONLY the JSON object above - no explanation, reasoning, or other text."
        )

        # Omitted for a decision model: it returns typed answers rather than
        # filling in a structure, so an output format would be an instruction it
        # structurally cannot follow.
        if not for_decision:
            builder.add_section(
                name=PromptSections.OUTPUT_FORMAT,
                template="""
# OUTPUT FORMAT

- Evaluate the rule by filling in the details in the following structure:
```json
{result_structure_text}
```
{json_only_note}
""",
                props={
                    "result_structure_text": self._format_output(context),
                    "json_only_note": json_only_note,
                },
            )

        builder.add_context_variables(context.state.context_variables)
        builder.add_glossary(list(context.state.glossary_terms))
        if context.state.session_summary:
            builder.add_session_summary(context.state.session_summary)
        # Cache only the history through the agent's last reply; the trailing customer
        # message(s) are rendered in the per-turn tail (see `_build_prompt`), keeping
        # this prefix byte-identical from one turn's `prefill` to the next turn's
        # matching (which only appends the new customer message).
        cached_events, _ = self._split_interaction_at_last_agent_reply(context.interaction.events)
        builder.add_interaction_history(
            cached_events,
            format=EventAdaptationFormat.ROLE_SCRIPT,
        )

        return builder

    def _split_interaction_at_last_agent_reply(
        self,
        events: Sequence[Event],
    ) -> tuple[Sequence[Event], Sequence[Event]]:
        """Split the interaction into the cross-turn-stable head (through the agent's
        last reply) and the per-turn tail (the customer message(s) that arrived after
        it). Caching only the head lets the prefix `prefill` warms at the end of a turn
        be reused by the next turn's matching, which merely appends the new message."""
        cutoff = 0
        for index, event in enumerate(events):
            if event.kind == EventKind.MESSAGE and event.source in self._AGENT_MESSAGE_SOURCES:
                cutoff = index + 1
        return events[:cutoff], events[cutoff:]

    def _format_current_turn(
        self,
        builder: PromptBuilder,
        events: Sequence[Event],
    ) -> str:
        rendered = [
            builder.adapt_event(event, format=EventAdaptationFormat.ROLE_SCRIPT)
            for event in events
            if event.kind != EventKind.STATUS
        ]
        if not rendered:
            return "(No new customer message has arrived yet.)"
        return (
            "The following continues the interaction history above — the customer's "
            "most recent message(s):\n" + "\n".join(rendered)
        )

    def _format_output(self, context: EngineContext) -> str:
        result: dict[str, JSONSerializable] = {}

        if self._should_include_tldr(context):
            result["tldr"] = (
                "<A brief, one-line summary of why the rule is or "
                "isn't relevant to the most recent state of the interaction>"
            )

        result["s"] = (
            "<The integer score from 1 to 5 indicating how relevant the rule is, per the scale described above>"
        )

        return to_prompt_json(result)


def _format_rule(
    title: str | None,
    condition: str | None,
    description: str | None = None,
) -> str:
    title = (title or "").strip()
    condition = (condition or "").strip()
    description = (description or "").strip()

    sections: list[str] = []

    if title:
        sections.append(f"Title: {title}")

    if condition:
        sections.append(f"Condition: {condition}")

    if description:
        label = "Policy" if not condition else "Details"
        sections.append(f"{label}: {description}")

    return "\n\n".join(sections)


def _make_event(e_id: str, source: EventSource, message: str) -> Event:
    return Event(
        id=EventId(e_id),
        source=source,
        kind=EventKind.MESSAGE,
        creation_utc=datetime.now(timezone.utc),
        modified_utc=datetime.now(timezone.utc),
        offset=0,
        trace_id="",
        data={"message": message},
        metadata={},
        deleted=False,
    )


example_1_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "My throat is dry, do you have anything that can help me with that?",
    ),
]

example_1_rule = ShotRule(condition="The customer might be thirsty")

example_1_expected = RuleRankSchema(
    tldr="Dry throat suggests potential thirst",
    s=4,
)


example_2_events = [
    _make_event(
        "21",
        EventSource.CUSTOMER,
        "Can delivery arrive here before 8 PM?",
    ),
]

example_2_rule = ShotRule(condition="The customer asks for late night (10 PM or later) delivery")

example_2_expected = RuleRankSchema(
    tldr="8 PM is not late night (10 PM or later)",
    s=1,
)


example_3_events = [
    _make_event(
        "31",
        EventSource.CUSTOMER,
        "I'll consult with my partner about getting Platinum",
    ),
]

example_3_rule = ShotRule(condition="Expressed unsureness about getting a product")

example_3_expected = RuleRankSchema(
    tldr="The customer needs consultation before getting Platinum, which might be a product, so it's best to forward this to the human evaluator",
    s=4,
)


example_4_events = [
    _make_event(
        "41",
        EventSource.CUSTOMER,
        "I want to renew my gym membership.",
    ),
    _make_event(
        "42",
        EventSource.AI_AGENT,
        "Happy to help! Which branch do you usually train at?",
    ),
    _make_event(
        "43",
        EventSource.CUSTOMER,
        "Can I also freeze it during the summer months?",
    ),
]

example_4_rule = ShotRule(
    condition="the customer wants to renew their membership",
    description="Ask which branch they train at.",
)

example_4_expected = RuleRankSchema(
    tldr="The agent already asked, but the customer hasn't answered — the renewal intent still stands, so the rule still applies",
    s=5,
)


example_5_events = [
    _make_event(
        "51",
        EventSource.CUSTOMER,
        "I need to cancel my flight booking.",
    ),
    _make_event(
        "52",
        EventSource.AI_AGENT,
        "You can cancel it yourself in our app, under My Trips.",
    ),
    _make_event(
        "53",
        EventSource.CUSTOMER,
        "I don't want to use the app. Just cancel it for me here.",
    ),
]

example_5_rule = ShotRule(
    condition="the customer wants to cancel a booking",
    description="Direct them to cancel via the self-service app.",
)

example_5_expected = RuleRankSchema(
    tldr="The customer is pushing back on the channel, not abandoning the cancellation — the rule still applies and the agent should restate the app-only path",
    s=5,
)


_baseline_shots: Sequence[RuleRankingShot] = [
    RuleRankingShot(
        description="",
        interaction_events=example_1_events,
        rule=example_1_rule,
        expected_result=example_1_expected,
    ),
    RuleRankingShot(
        description="",
        interaction_events=example_2_events,
        rule=example_2_rule,
        expected_result=example_2_expected,
    ),
    RuleRankingShot(
        description="",
        interaction_events=example_3_events,
        rule=example_3_rule,
        expected_result=example_3_expected,
    ),
    RuleRankingShot(
        description="",
        interaction_events=example_4_events,
        rule=example_4_rule,
        expected_result=example_4_expected,
    ),
    RuleRankingShot(
        description="",
        interaction_events=example_5_events,
        rule=example_5_rule,
        expected_result=example_5_expected,
    ),
]

shot_collection = ShotCollection[RuleRankingShot](_baseline_shots)
