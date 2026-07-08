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

import json
import traceback
from dataclasses import dataclass
from typing import Optional, Sequence

from parlant.core.agents import Agent
from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.rule_matching.generic.common import escape_json_string
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.glossary import Term
from parlant.core.rules import RuleContent
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.indexing.common import EvaluationError, ProgressReport
from parlant.core.shots import Shot, ShotCollection


class RuleSignalProposition(DefaultBaseModel):
    signals: Sequence[str]
    anti_signals: Sequence[str]
    rationale: str


class RuleSignalPropositionSchema(DefaultBaseModel):
    rationale: str
    signals: list[str]
    anti_signals: list[str]


@dataclass
class RuleSignalProposerShot(Shot):
    title: str
    rule: RuleContent
    expected_result: RuleSignalPropositionSchema


class RuleSignalProposer:
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[RuleSignalPropositionSchema],
    ) -> None:
        self._logger = logger
        self._optimization_policy = optimization_policy
        self._schematic_generator = schematic_generator

    async def propose_signals(
        self,
        rule: RuleContent,
        title: str,
        agent: Agent | None,
        glossary_terms: Sequence[Term],
        progress_report: Optional[ProgressReport] = None,
    ) -> RuleSignalProposition:
        if progress_report:
            await progress_report.stretch(1)

        with self._logger.scope("RuleSignalProposer"):
            generation_attempt_temperatures = (
                self._optimization_policy.get_rule_proposition_retry_temperatures(
                    hints={"type": self.__class__.__name__}
                )
            )

            last_generation_exception: Exception | None = None

            for generation_attempt in range(3):
                try:
                    proposition = await self._generate_signals(
                        rule,
                        title,
                        agent,
                        glossary_terms,
                        temperature=generation_attempt_temperatures[generation_attempt],
                    )
                    signals = self._normalize_signals(proposition.signals)
                    anti_signals = self._normalize_signals(proposition.anti_signals)

                    if len(signals) != 10:
                        raise ValueError(
                            f"Expected exactly 10 rule signals, but got {len(signals)}"
                        )

                    if len(anti_signals) != 10:
                        raise ValueError(
                            f"Expected exactly 10 rule anti-signals, but got {len(anti_signals)}"
                        )

                    if progress_report:
                        await progress_report.increment(1)

                    return RuleSignalProposition(
                        signals=signals,
                        anti_signals=anti_signals,
                        rationale=proposition.rationale,
                    )
                except Exception as exc:
                    self._logger.warning(
                        f"RuleSignalProposer attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                    )

                    last_generation_exception = exc

            raise EvaluationError() from last_generation_exception

    def _normalize_signals(self, signals: Sequence[str]) -> Sequence[str]:
        normalized: list[str] = []
        seen: set[str] = set()

        for signal in signals:
            if not (clean := signal.strip()):
                continue

            key = clean.casefold()
            if key in seen:
                continue

            normalized.append(clean)
            seen.add(key)

        return normalized

    async def _build_prompt(
        self,
        rule: RuleContent,
        title: str,
        agent: Agent | None,
        glossary_terms: Sequence[Term],
        shots: Sequence[RuleSignalProposerShot],
    ) -> PromptBuilder:
        builder = PromptBuilder()

        builder.add_section(
            name="rule-signal-proposer-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
In our system, the behavior of a conversational AI agent is guided by "rules".
Each rule has a condition describing when it should apply, and may also have an action and description.

    The Compass engine uses rule signals for semantic recall.
    A signal is a short example user message that should activate the rule.
    An anti-signal is a short example user message that is still plausible for this agent's domain, but should NOT activate this particular rule.
    Signals and anti-signals are embedded separately and used to train a local relevance boundary for the rule.
""",
        )

        builder.add_section(
            name="rule-signal-proposer-task-description",
            template="""
TASK DESCRIPTION
-----------------
    Your task is to suggest positive and negative user messages for the given rule.

    Generate exactly 10 signals and exactly 10 anti-signals.
    Each signal must be phrased as something a user/customer might actually say in a conversation.
    The 10 signals should be distinct and unique from each other while still being clearly relevant to the rule.
    Signals should cover common wording, paraphrases, and important edge cases implied by the rule.

    Each anti-signal must also be phrased as something a user/customer might actually say.
    Anti-signals must stay within the same agent domain, product area, and vocabulary when possible.
    Good anti-signals are near misses: related requests, adjacent intents, or plausible confusions that should activate a different rule or no rule, but not this one.
    Do not use generic unrelated anti-signals just because they are easy to distinguish.

    Prefer concrete natural messages over keywords.
    Do not include messages that would activate a different rule more specifically.
    Do not mention that these are signals, anti-signals, embeddings, or rules inside the message text.
""",
        )

        builder.add_section(
            name="rule-signal-proposer-agent",
            template="""
AGENT
-----------
{agent_text}
""",
            props={"agent_text": self._format_agent(agent)},
        )

        builder.add_section(
            name="rule-signal-proposer-glossary",
            template="""
GLOSSARY
-----------
{glossary_text}
""",
            props={"glossary_text": self._format_glossary(glossary_terms)},
        )

        builder.add_section(
            name="rule-signal-proposer-shots",
            template="""
EXAMPLES
-----------
{shots_text}""",
            props={"shots_text": self._format_shots(shots)},
        )

        builder.add_section(
            name="rule-signal-proposer-rule",
            template="""
RULE
-----------
{rule_text}
""",
            props={"rule_text": self._format_rule(title, rule)},
        )

        builder.add_section(
            name="rule-signal-proposer-output-format",
            template="""OUTPUT FORMAT
-----------
Use the following format:
Expected output (JSON):
```json
{{
	  "rationale": "<str, short explanation of the activation surface you covered>",
	  "signals": [
	    "<str, exactly 10 different example user messages that should activate this rule>"
	  ],
	  "anti_signals": [
	    "<str, exactly 10 different in-domain example user messages that should not activate this rule>"
	  ]
	}}
```
""",
        )

        return builder

    async def _generate_signals(
        self,
        rule: RuleContent,
        title: str,
        agent: Agent | None,
        glossary_terms: Sequence[Term],
        temperature: float,
    ) -> RuleSignalPropositionSchema:
        prompt = await self._build_prompt(
            rule,
            title,
            agent,
            glossary_terms,
            await shot_collection.list(),
        )

        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": temperature},
        )

        self._logger.trace(
            f"RuleSignalProposer response: {response.content.model_dump_json(indent=2)}"
        )

        return response.content

    def _format_agent(self, agent: Agent | None) -> str:
        if agent is None:
            return "No agent context was provided."

        return f"{agent.name}\n\n{agent.description or ''}".strip()

    def _format_glossary(self, terms: Sequence[Term]) -> str:
        if not terms:
            return "No glossary terms were provided."

        def format_term(term: Term) -> str:
            synonyms = f"\nSynonyms: {', '.join(term.synonyms)}" if term.synonyms else ""
            return f"## {term.name}\n{term.description}{synonyms}"

        return "\n\n".join(format_term(term) for term in terms)

    def _format_rule(self, title: str, rule: RuleContent) -> str:
        result = f"# {escape_json_string(title)}"

        if rule.condition and rule.action:
            result += (
                f"\n\n## When {escape_json_string(rule.condition)} "
                f"then {escape_json_string(rule.action)}"
            )
        elif rule.condition:
            result += f"\n\n## Condition: {escape_json_string(rule.condition)}"
        elif rule.action:
            result += f"\n\n## Action: {escape_json_string(rule.action)}"

        if rule.description:
            result += f"\n\n{escape_json_string(rule.description)}"

        return result

    def _format_shots(self, shots: Sequence[RuleSignalProposerShot]) -> str:
        return "\n".join(
            [
                f"""Example {i}: {shot.description}
Rule:
{self._format_rule(shot.title, shot.rule)}

Expected Response:
{json.dumps(shot.expected_result.model_dump(mode="json", exclude_unset=True), indent=2)}
###
"""
                for i, shot in enumerate(shots, start=1)
            ]
        )


example_1_rule = RuleContent(
    condition="The customer wants to report a lost or stolen card",
    action="Help them secure the card and explain replacement options",
)
example_1_shot = RuleSignalProposerShot(
    description="Card-loss rule with several natural phrasings",
    title="Lost or Stolen Card",
    rule=example_1_rule,
    expected_result=RuleSignalPropositionSchema(
        rationale="The signals cover lost cards, stolen cards, and urgent card-security language. The anti-signals stay within banking support while targeting adjacent non-card-loss intents.",
        signals=[
            "I lost my debit card",
            "My card was stolen and I need help",
            "I can't find my credit card, can you block it?",
            "Someone took my card",
            "I need to report a missing card",
            "My wallet is gone and my bank card was inside",
            "Please freeze my card, I misplaced it",
            "I think my card was taken from my bag",
            "I can't locate my card and I'm worried",
            "Can you replace a lost card?",
        ],
        anti_signals=[
            "I forgot my online banking password",
            "Can you increase my credit limit?",
            "I want to dispute a card charge",
            "When will my new card arrive?",
            "Can I change the PIN on my card?",
            "I need to update my billing address",
            "Why was my card declined?",
            "Can you explain this account fee?",
            "I want to open a savings account",
            "Can I add my card to a digital wallet?",
        ],
    ),
)

example_2_rule = RuleContent(
    condition="The customer asks about refund eligibility for a delayed flight",
    action="Review the reservation details and explain eligible refund or compensation options",
)
example_2_shot = RuleSignalProposerShot(
    description="Travel compensation rule where the user may not say refund directly",
    title="Delayed Flight Refund Eligibility",
    rule=example_2_rule,
    expected_result=RuleSignalPropositionSchema(
        rationale="The signals cover direct refund requests, compensation wording, and frustration about delay impact. The anti-signals stay in airline support while targeting non-delay or non-refund needs.",
        signals=[
            "My flight was delayed, can I get a refund?",
            "Do I qualify for compensation because my flight was late?",
            "The delay ruined my plans, what can you do for me?",
            "Am I entitled to anything for a delayed flight?",
            "My plane arrived late and I want to know my options",
            "Can I get money back for the long delay?",
            "The airline delayed us for hours, do I get compensation?",
            "My arrival was much later than scheduled",
            "I missed my connection because the first flight was late",
            "What refund rights do I have after a delayed flight?",
        ],
        anti_signals=[
            "I want to change my flight date",
            "Can I add a checked bag?",
            "I need to pick a seat",
            "My flight was cancelled, what happens now?",
            "Can I upgrade to business class?",
            "I need help checking in",
            "Where is my baggage?",
            "Can I correct the passenger name?",
            "What time does boarding start?",
            "I want to use miles for this booking",
        ],
    ),
)

example_3_rule = RuleContent(
    condition="The customer needs to update the shipping address on an existing order",
    action="Collect the order identifier and new address before updating the order",
)
example_3_shot = RuleSignalProposerShot(
    description="Order-change rule with address-update phrasing",
    title="Update Shipping Address",
    rule=example_3_rule,
    expected_result=RuleSignalPropositionSchema(
        rationale="The signals cover changing, correcting, and redirecting delivery addresses. The anti-signals stay in order support while targeting adjacent non-address intents.",
        signals=[
            "I need to change the delivery address for my order",
            "Can you ship my package somewhere else?",
            "I entered the wrong address at checkout",
            "Please update where my order is being sent",
            "My order is going to the old address",
            "Can I redirect this delivery?",
            "The shipping address on my order is incorrect",
            "I moved and need my package sent to my new place",
            "Please change the destination before it ships",
            "I put the wrong apartment number on my order",
        ],
        anti_signals=[
            "Where is my package right now?",
            "I want to cancel my order",
            "Can I return this item?",
            "My item arrived damaged",
            "Can I change the payment method?",
            "I need a receipt for my order",
            "Can I add another item to the order?",
            "Why was my order delayed?",
            "I want faster shipping",
            "Can you apply a discount code?",
        ],
    ),
)

_baseline_shots: Sequence[RuleSignalProposerShot] = [
    example_1_shot,
    example_2_shot,
    example_3_shot,
]

shot_collection = ShotCollection[RuleSignalProposerShot](_baseline_shots)
