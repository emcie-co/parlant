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

import traceback
from typing import Optional, Sequence

from parlant.core.agents import Agent
from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.glossary import Term
from parlant.core.rules import RuleContent
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.indexing.common import EvaluationError, ProgressReport


class RuleTitleProposition(DefaultBaseModel):
    title: str
    rationale: str


class RuleTitlePropositionSchema(DefaultBaseModel):
    rationale: str
    title: str


class RuleTitleProposer:
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[RuleTitlePropositionSchema],
    ) -> None:
        self._logger = logger
        self._schematic_generator = schematic_generator

    async def propose_title(
        self,
        rule: RuleContent,
        agent: Agent | None,
        glossary_terms: Sequence[Term],
        progress_report: Optional[ProgressReport] = None,
    ) -> RuleTitleProposition:
        if progress_report:
            await progress_report.stretch(1)

        with self._logger.scope("RuleTitleProposer"):
            try:
                proposition = await self._generate_title(
                    rule=rule,
                    agent=agent,
                    glossary_terms=glossary_terms,
                )

                title = self._normalize_title(proposition.title)
                if not title:
                    raise ValueError("Generated title is empty")

                if progress_report:
                    await progress_report.increment(1)

                return RuleTitleProposition(
                    title=title,
                    rationale=proposition.rationale,
                )
            except Exception as exc:
                self._logger.warning(f"RuleTitleProposer failed: {traceback.format_exception(exc)}")
                raise EvaluationError() from exc

    async def _generate_title(
        self,
        rule: RuleContent,
        agent: Agent | None,
        glossary_terms: Sequence[Term],
    ) -> RuleTitlePropositionSchema:
        prompt = await self._build_prompt(
            rule=rule,
            agent=agent,
            glossary_terms=glossary_terms,
        )

        generation = await self._schematic_generator.generate(prompt=prompt)

        return generation.content

    async def _build_prompt(
        self,
        rule: RuleContent,
        agent: Agent | None,
        glossary_terms: Sequence[Term],
    ) -> PromptBuilder:
        builder = PromptBuilder()

        builder.add_section(
            name="rule-title-proposer-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
In our system, the behavior of a conversational AI agent is guided by rules.
Some rules are policy-style rules: they may have only descriptive content and no condition or action.
""",
        )

        builder.add_section(
            name="rule-title-proposer-task-description",
            template="""
TASK DESCRIPTION
-----------------
Generate a concise title for the given rule.

The title must:
- Summarize the rule's subject, not its implementation mechanics.
- Be 2 to 6 words when possible.
- Be specific enough to distinguish this rule from nearby policies.
- Avoid trailing punctuation.
- Avoid mentioning that this is a rule, policy, rule, instruction, or title.
- Avoid inventing scope not supported by the rule text.
""",
        )

        builder.add_section(
            name="rule-title-proposer-agent",
            template="""
AGENT
-----------
{agent_text}
""",
            props={"agent_text": self._format_agent(agent)},
        )

        builder.add_section(
            name="rule-title-proposer-glossary",
            template="""
GLOSSARY
-----------
{glossary_text}
""",
            props={"glossary_text": self._format_glossary(glossary_terms)},
        )

        builder.add_section(
            name="rule-title-proposer-examples",
            template="""
EXAMPLES
-----------
{examples_text}
""",
            props={"examples_text": self._format_examples()},
        )

        builder.add_section(
            name="rule-title-proposer-rule",
            template="""
RULE
-----------
{rule_text}
""",
            props={"rule_text": self._format_rule(rule)},
        )

        builder.add_section(
            name="rule-title-proposer-output-format",
            template="""OUTPUT FORMAT
-----------
Use the following format:
Expected output (JSON):
```json
{{
  "rationale": "<str, short explanation for why the title fits>",
  "title": "<str, concise title>"
}}
```
""",
        )

        return builder

    def _normalize_title(self, title: str) -> str:
        return title.strip().strip("\"'`").rstrip(".:;").strip()

    def _format_agent(self, agent: Agent | None) -> str:
        if not agent:
            return "No agent context provided."

        return "\n".join(
            [
                f"Name: {agent.name}",
                f"Description: {agent.description or 'No description provided.'}",
            ]
        )

    def _format_glossary(self, terms: Sequence[Term]) -> str:
        if not terms:
            return "No glossary terms provided."

        return "\n\n".join(
            f"### {term.name}\n{term.description}" for term in sorted(terms, key=lambda t: t.name)
        )

    def _format_examples(self) -> str:
        return """
Example 1

Rule:
Description: Do not proactively offer compensation unless the customer explicitly asks for it.
Only compensate after confirming the customer's eligibility.

Expected output:
```json
{
  "rationale": "The title names the subject and constraint without adding unsupported details.",
  "title": "Compensation Eligibility"
}
```

Example 2

Rule:
Condition: the customer reports a lost card

Expected output:
```json
{
  "rationale": "The title captures the lost-card handling topic concisely.",
  "title": "Handling Lost Cards"
}
```

Example 3

Rule:
Description: When calculating or comparing prices, balances, times, fees, or quantities, use the calculator tool and communicate the calculation clearly when appropriate.

Expected output:
```json
{
  "rationale": "The title summarizes the calculation-safety requirement without naming implementation details too narrowly.",
  "title": "Handling Calculations Reliably with the Calculation Tool"
}
```
""".strip()

    def _format_rule(self, rule: RuleContent) -> str:
        parts: list[str] = []

        if rule.condition:
            parts.append(f"Condition: {rule.condition}")

        if rule.action:
            parts.append(f"Action: {rule.action}")

        if rule.description:
            parts.append(f"Description: {rule.description}")

        return "\n\n".join(parts) if parts else "No rule content provided."
