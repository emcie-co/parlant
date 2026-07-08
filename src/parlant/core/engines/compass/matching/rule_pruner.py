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
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from parlant.core.agents import Effort
from parlant.core.common import JSONSerializable
from parlant.core.engines.alpha.prompt_builder import (
    BuiltInSection,
    EventAdaptationFormat,
    PromptBuilder,
    SectionStatus,
)
from parlant.core.engines.compass.matching.common import aggregate_generation_info
from parlant.core.engines.compass.matching.rule_ranker import (
    RuleRankSchema,
    _format_rule,
)
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.rules import Rule, RuleContent
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.shots import Shot, ShotCollection
from parlant.core.tracer import Tracer


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
            return "minimal"
        case Effort.MAX:
            return "low"


@dataclass(frozen=True)
class PrunedRule:
    rule: Rule
    reasoning: str
    is_still_relevant: bool
    score: float


@dataclass(frozen=True)
class RulePruningResult:
    pruned_rules: Sequence[PrunedRule]
    # Aggregated usage across every per-rule pruning request this call,
    # or None when no requests were sent (or all of them failed).
    generation_info: GenerationInfo | None


@dataclass
class RulePruningShot(Shot):
    interaction_events: Sequence[Event]
    # The pruner evaluates a single rule per prompt, so each shot carries one.
    rule: RuleContent
    expected_result: RuleRankSchema


class RulePruner:
    """Judges whether previously-matched session rules are still relevant.

    Session rules accumulate for the life of a session; once the set exceeds
    its cap, the matcher asks the pruner which members still matter to where the
    conversation stands now (or are likely to matter going forward) and retires the
    stale ones. A structural sibling of :class:`RuleRanker` — one small-model
    call per rule, fanned out concurrently — but retrospective, and run at end
    of turn: the turn is complete, so the FULL interaction history belongs to the
    shared cacheable prefix and the only per-call tail is the rule itself.
    """

    STILL_RELEVANT_SCORE_THRESHOLD = 3
    RULE_CACHE_BREAKPOINT = "- Rule: ###"

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        schematic_generator: SchematicGenerator[RuleRankSchema],
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._schematic_generator = schematic_generator

    async def prune(
        self,
        context: EngineContext,
        rules: Sequence[Rule],
    ) -> RulePruningResult:
        if not rules:
            return RulePruningResult([], None)

        with self._tracer.span("match.rule.prune"):
            t_start = asyncio.get_event_loop().time()
            shots = await self.shots()

            # Warm-first-then-fan-out: the first call alone populates the provider's
            # prefix cache (the shared prefix is identical across the fan-out), so
            # the remaining calls send only their live suffixes. We're off the
            # user-facing path, so the extra serialized round-trip is free cache money.
            first_result = await self._prune_rule(context, rules[0], shots)
            rest_results = await asyncio.gather(
                *(self._prune_rule(context, g, shots) for g in rules[1:])
            )
            results = [first_result, *rest_results]

            t_end = asyncio.get_event_loop().time()

            infos = [info for _, info in results if info is not None]

            return RulePruningResult(
                pruned_rules=[pruned for pruned, _ in results],
                generation_info=aggregate_generation_info(
                    infos,
                    total_duration=t_end - t_start,
                )
                if infos
                else None,
            )

    async def _prune_rule(
        self,
        context: EngineContext,
        rule: Rule,
        shots: Sequence[RulePruningShot],
    ) -> tuple[PrunedRule, GenerationInfo | None]:
        try:
            prompt = self._build_prompt(context, rule, shots)

            inference = await self._schematic_generator.generate(
                prompt=prompt,
                hints={
                    "reasoning_effort": get_dynamic_reasoning_effort_for_matching(context),
                    "cache": {
                        "key": self._cache_key(context),
                        "breakpoint": self._cache_breakpoint(context),
                    },
                },
            )

            score = inference.content.s

            return (
                PrunedRule(
                    rule=rule,
                    reasoning=inference.content.tldr
                    or f"This rule scored {score} out of 5 in continued relevance.",
                    is_still_relevant=score >= self.STILL_RELEVANT_SCORE_THRESHOLD,
                    score=float(score) / 5.0,
                ),
                inference.info,
            )
        except Exception as exc:
            # Losing a session rule to a transient generation error is worse
            # than keeping a stale one for another round: default to keeping it.
            self._logger.warning(f"Session rule pruning call failed (keeping the rule): {exc}")
            return (
                PrunedRule(
                    rule=rule,
                    reasoning="Pruning evaluation failed; keeping the rule.",
                    is_still_relevant=True,
                    score=1.0,
                ),
                None,
            )

    def _should_include_tldr(self, context: EngineContext) -> bool:
        match context.state.dynamic_effort_level:
            case Effort.MIN:
                return False
            case Effort.LOW:
                return False
            case Effort.MEDIUM:
                return True
            case Effort.HIGH:
                return True
            case Effort.MAX:
                return True

    def _cache_key(self, context: EngineContext) -> str:
        # Namespace the provider cache per session AND component, so components
        # that cache concurrently never clobber a shared entry.
        return f"{context.session.id}.rule-pruner"

    def _cache_breakpoint(self, context: EngineContext) -> str:
        # The whole prompt except the rule itself is shared across the
        # fan-out (the turn is complete at pruning time, so even the interaction
        # history is stable), so the cache boundary sits right before it.
        return self.RULE_CACHE_BREAKPOINT

    async def shots(self) -> Sequence[RulePruningShot]:
        return await shot_collection.list()

    def _format_shots(self, context: EngineContext, shots: Sequence[RulePruningShot]) -> str:
        return "\n".join(
            f"Example #{i}: ###\n{self._format_shot(context, shot)}"
            for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, context: EngineContext, shot: RulePruningShot) -> str:
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
{json.dumps([adapt_event(e) for e in shot.interaction_events], indent=2)}

"""

        formatted_shot += f"""
- **Rule**:
{
            _format_rule(
                None,
                shot.rule.condition,
                shot.rule.action,
                shot.rule.description,
            )
        }

"""

        expected_result = shot.expected_result.model_dump(mode="json")
        if not self._should_include_tldr(context) or expected_result.get("tldr") is None:
            expected_result.pop("tldr", None)

        formatted_shot += f"""
- **Expected Result**:
```json
{json.dumps(expected_result, indent=2)}
```
"""

        return formatted_shot

    def _build_prompt(
        self,
        context: EngineContext,
        rule: Rule,
        shots: Sequence[RulePruningShot],
    ) -> PromptBuilder:
        # Everything except the rule itself is the shared, cacheable prefix;
        # the rule is the only per-call tail (see `_cache_breakpoint`).
        builder = self._build_shared_prompt(context, shots)

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
                    rule.content.condition,
                    rule.content.action,
                    rule.content.description,
                ),
            },
            status=SectionStatus.ACTIVE,
        )

        return builder

    def _build_shared_prompt(
        self,
        context: EngineContext,
        shots: Sequence[RulePruningShot],
    ) -> PromptBuilder:
        builder = PromptBuilder()

        builder.add_section(
            name="rule-pruner-general-instructions",
            template="""
# GENERAL INSTRUCTIONS

In our system, a conversational multi-turn AI agent's behavior is controlled by a set of rules.
Some rules are condition/action instructions: when the condition applies, the agent is directed to take its associated action.
Other rules are policy-style instructions: they may have no condition or action, and their title and policy text describe the rule, constraint, or information that may govern the conversation.
Rules that were found relevant at any point during a session are kept in the agent's WORKING SET for that session, so the agent stays aware of them on later turns.
""",
            props={},
        )
        builder.add_section(
            name="rule-pruner-task-description",
            template="""
# Task Description

The rule below was previously judged relevant at some point in this session, and currently sits in the agent's working set. The working set has grown too large, so it is being reviewed. Your task is to decide whether this rule is STILL relevant to where the conversation stands NOW, or likely to matter going forward - or whether it is stale and can be retired from the working set.

Output an integer score from 1 to 5 indicating the rule's continued relevance:
- 1 - stale: tied to a topic that has clearly concluded or been abandoned, and will not matter again in this conversation
- 2 - essentially exhausted: its purpose was fully served and nothing suggests it will come up again
- 3 - uncertain: the topic could plausibly resurface, or it is unclear whether the rule's purpose is done
- 4 - still pertinent: relates to an open thread, a pending commitment, or a standing constraint on the agent's behavior
- 5 - actively governs the current state of the conversation

Scores of 1 and 2 mean the rule will be retired from the working set; 3 and above keep it.

Important considerations:
1. Standing policies and constraints - rules about what the agent must always or never do, security, compliance or privacy rules, tone requirements, and other always-on instructions - remain relevant for the whole conversation unless they are explicitly scoped to a sub-task that has finished. NEVER retire a rule merely because it hasn't been mentioned recently.
2. Weigh the session summary (if provided) together with the full interaction: a topic raised earlier may still be pending even if the last few messages moved elsewhere. A topic is only concluded when it was resolved or clearly abandoned.
3. When in doubt, score higher - keeping a rule in the working set is cheap, while wrongly dropping one can cause the agent to break its instructions later. A human judge will review the retirements.
""",
            props={},
        )
        builder.add_section(
            name="rule-pruner-examples",
            template="""
# Examples of Continued-Relevance Evaluations:

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
        # for output tokens we discard.
        json_only_note = (
            ""
            if self._should_include_tldr(context)
            else "\nRespond with ONLY the JSON object above - no explanation, reasoning, or other text."
        )

        builder.add_section(
            name="rule-pruner-output-format",
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
        builder.add_capabilities_for_rule_matching(context.state.capabilities)
        if context.state.session_summary:
            builder.add_session_summary(context.state.session_summary)
        # The pruner runs at end of turn, when the turn is complete — the FULL
        # interaction history (including the latest exchange) is stable and belongs
        # to the shared prefix, unlike the turn evaluators' per-turn split.
        builder.add_interaction_history(
            context.interaction.events,
            format=EventAdaptationFormat.ROLE_SCRIPT,
        )

        return builder

    def _format_output(self, context: EngineContext) -> str:
        result: dict[str, JSONSerializable] = {}

        if self._should_include_tldr(context):
            result["tldr"] = (
                "<A brief, one-line summary of why the rule is or isn't still "
                "relevant to the conversation going forward>"
            )

        result["s"] = (
            "<The integer score from 1 to 5 indicating the rule's continued relevance, per the scale described above>"
        )

        return json.dumps(result, indent=4)


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


# Shot 1: the rule's topic concluded several exchanges ago and the
# conversation has clearly moved on — stale.
example_1_events = [
    _make_event("11", EventSource.CUSTOMER, "Hi, I need to reset my password."),
    _make_event("21", EventSource.AI_AGENT, "Sure! I've sent a reset link to your email."),
    _make_event("31", EventSource.CUSTOMER, "Got it, that worked. Thanks!"),
    _make_event("41", EventSource.AI_AGENT, "Happy to help! Anything else?"),
    _make_event("51", EventSource.CUSTOMER, "Yes - do you ship internationally?"),
    _make_event("61", EventSource.AI_AGENT, "We ship to over 40 countries."),
    _make_event("71", EventSource.CUSTOMER, "Great, how long does shipping to France take?"),
]

example_1_rule = RuleContent(
    condition="the customer wants to reset their password",
    action="walk them through the password reset flow",
)

example_1_expected = RuleRankSchema(
    tldr=(
        "The password reset was completed and confirmed several exchanges ago, and the "
        "conversation has moved on to shipping - this rule will not matter again."
    ),
    s=1,
)


# Shot 2: the rule governs a request that is still open — the customer paused
# mid-flow, so it remains actively relevant.
example_2_events = [
    _make_event("11", EventSource.CUSTOMER, "I'd like to book a table for Saturday."),
    _make_event("21", EventSource.AI_AGENT, "Great! For how many people?"),
    _make_event("31", EventSource.CUSTOMER, "Four of us."),
    _make_event("41", EventSource.AI_AGENT, "And what time would you like?"),
    _make_event("51", EventSource.CUSTOMER, "Let me check with the others and get back to you."),
    _make_event("61", EventSource.AI_AGENT, "Of course, take your time!"),
]

example_2_rule = RuleContent(
    condition="the customer wants to book a table",
    action="collect the party size, date, and time, then book the table",
)

example_2_expected = RuleRankSchema(
    tldr=(
        "The booking is mid-flow - the time is still missing and the customer said "
        "they'd get back with it - so this rule actively governs the conversation."
    ),
    s=5,
)


# Shot 3: a standing policy that hasn't been mentioned recently. Standing
# constraints stay relevant for the whole conversation — not being mentioned is
# NOT a reason to retire them.
example_3_events = [
    _make_event("11", EventSource.CUSTOMER, "What are your opening hours?"),
    _make_event("21", EventSource.AI_AGENT, "We're open 9am-6pm on weekdays."),
    _make_event("31", EventSource.CUSTOMER, "And where is your nearest branch?"),
    _make_event("41", EventSource.AI_AGENT, "Our nearest branch is on 5th Avenue."),
]

example_3_rule = RuleContent(
    condition="",
    action=None,
    description=(
        "Never disclose customer account details to anyone other than the verified account holder."
    ),
)

example_3_expected = RuleRankSchema(
    tldr=(
        "This is a standing privacy constraint on the agent's behavior; it applies for "
        "the whole conversation regardless of the current topic."
    ),
    s=4,
)


_baseline_shots: Sequence[RulePruningShot] = [
    RulePruningShot(
        description="",
        interaction_events=example_1_events,
        rule=example_1_rule,
        expected_result=example_1_expected,
    ),
    RulePruningShot(
        description="",
        interaction_events=example_2_events,
        rule=example_2_rule,
        expected_result=example_2_expected,
    ),
    RulePruningShot(
        description="",
        interaction_events=example_3_events,
        rule=example_3_rule,
        expected_result=example_3_expected,
    ),
]

shot_collection = ShotCollection[RulePruningShot](_baseline_shots)
