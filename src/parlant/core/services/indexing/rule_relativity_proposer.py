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
from typing import Optional
from parlant.core.agents import AgentId
from parlant.core.common import DefaultBaseModel
from parlant.core.rule_representation import escape_json_string
from parlant.core.engines.compass.optimization_policy import CompassOptimizationPolicy
from parlant.core.engines.compass.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.nlp.model_routing import ModelRouter
from parlant.core.services.indexing.common import IndexingError, ProgressReport


class RuleRelativityProposition(DefaultBaseModel):
    references_other_policies: bool


class RuleRelativityPropositionSchema(DefaultBaseModel):
    rationale: str
    references_other_policies: bool


class RuleRelativityProposer:
    """Classifies whether a rule's meaning depends on the agent's other rules.

    A rule is cross-referential when its text contains references whose
    referents live outside the rule itself — in the agent's other policies.
    Such rules cannot be judged in isolation at matching time; the classifier
    flags them so the matcher can evaluate them against their policy context.
    """

    def __init__(
        self,
        logger: Logger,
        optimization_policy: CompassOptimizationPolicy,
        model_router: ModelRouter,
    ) -> None:
        self._logger = logger
        self._optimization_policy = optimization_policy
        self._model_router = model_router

    async def propose_relativity(
        self,
        content: Optional[str],
        condition: Optional[str],
        title: str | None = None,
        agent_id: AgentId | None = None,
        progress_report: Optional[ProgressReport] = None,
    ) -> RuleRelativityProposition:
        if progress_report:
            await progress_report.stretch(1)

        with self._logger.scope("RuleRelativityProposer"):
            generation_attempt_temperatures = (
                self._optimization_policy.get_rule_proposition_retry_temperatures(
                    hints={"type": self.__class__.__name__}
                )
            )

            last_generation_exception: Exception | None = None

            for generation_attempt in range(3):
                try:
                    proposition = await self._generate_relativity(
                        content,
                        condition,
                        title,
                        agent_id,
                        temperature=generation_attempt_temperatures[generation_attempt],
                    )

                    if progress_report:
                        await progress_report.increment(1)

                    return RuleRelativityProposition(
                        references_other_policies=proposition.references_other_policies,
                    )
                except Exception as exc:
                    self._logger.warning(
                        f"RuleRelativityProposer attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                    )

                    last_generation_exception = exc

            raise IndexingError.caused_by(
                "RuleRelativityProposer",
                last_generation_exception,
                attempts=3,
            ) from last_generation_exception

    async def _build_prompt(
        self,
        content: Optional[str],
        condition: Optional[str],
        title: str | None,
    ) -> PromptBuilder:
        builder = PromptBuilder()

        builder.add_section(
            name="rule-relativity-proposer-general-instructions",
            template="""
In our system, the behavior of a conversational AI agent is guided by "rules" (also called policies).
A rule can be stated in either (or a mix) of two forms:
- A condition/action pair: "condition" specifies when the rule applies; "action" is what the agent should do then.
- Free-form policy text: an optional title plus a "description" holding the full policy, procedure, or instructions.

Judge whichever parts are present; a missing part carries no signal either way.

An agent typically has MANY rules. Most rules are self-contained: they can be judged by looking at the conversation alone.
But some rules REFERENCE THE AGENT'S OTHER POLICIES — their meaning depends on what the rest of the rule set says. Such a rule cannot be judged in isolation: deciding whether it applies requires knowing what the agent's other policies cover.

Your task is to evaluate whether the given rule references the agent's other policies, in any of these ways:

1. IMPLICIT COMPLEMENT: the rule negates an undefined coverage set — e.g. "a capability that is not supported", "a request we don't handle", "anything unrelated to what we offer", "an unknown issue". What IS supported/handled/offered is defined by the other policies, not by this rule.
2. QUANTIFIED REFERENCE: the rule quantifies over the policy set — e.g. "doesn't fall under any of our policies", "two or more policies give conflicting guidance", "all applicable policies".
3. DEFINITE REFERENCE: the rule refers to a specific other policy — e.g. "unless the return policy applies", "follow the escalation procedure", "a more specific policy takes precedence", "this other policy".
""",
        )

        builder.add_section(
            name="rule-relativity-proposer-notes",
            template="""
Note that:
    1. A rule that merely mentions business entities (products, orders, accounts) is NOT referencing other policies. The reference must be to the agent's rules/policies/coverage themselves.
    2. Mentioning a business PROCESS the agent performs (e.g. "start the refund flow") is not a policy reference — it names an action, not another rule.
    3. Words like "supported", "covered", "offered", or "known" only make a rule cross-referential when their scope is defined by the rest of the policy set rather than within the rule itself. In particular, a request being "covered" by a product the business sells (an insurance plan, a warranty, travel insurance) is business data, NOT a reference to the agent's policies.
    4. A rule that refers to ITS OWN content is not referencing other policies — e.g. "make sure the rules above apply before calling the API", "do not do this for any reason other than the ones listed above", or "you can modify X, Y, or Z, but nothing else" (the complement is bounded by the rule's own list).
    5. If the rule contains at least one reference to OTHER policies, in any of its parts, classify it as referencing other policies.
""",
        )

        builder.add_section(
            name="rule-relativity-proposer-examples",
            template="""
Examples of rules that REFERENCE other policies:
    - condition: "The customer asks about a capability that is not supported" / action: "Say you cannot help with this"
        ("not supported" is the complement of a coverage set defined by the other policies)
    - condition: "The customer's request doesn't fall under any of our policies" / action: "Say this is beyond what you can help with"
        (quantifies over the whole policy set)
    - condition: "The customer wants to return an item and the return policy does not apply" / action: "Offer store credit"
        (definite reference to a specific other policy)
    - description: "Transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions."
        ("the scope of your actions" is defined by the agent's other policies — an implicit complement, stated as free-form policy text)
Examples of SELF-CONTAINED rules:
    - condition: "The customer wants a refund for a delivered order" / action: "Collect the order number and start the refund flow"
        (judgeable from the conversation alone; "refund flow" names a process, not another policy)
    - condition: "The customer is above 60" / action: "Use simple, non-technical language"
        (fully self-contained style rule)
    - description: "An order can be cancelled only while pending. Confirm the order id and the reason first. Do not accept any reason other than the ones above. The API does not validate these rules, so check them before calling it."
        (every reference — "the ones above", "these rules" — points at the rule's OWN content, not at other policies)
""",
        )

        parts = [
            ("title", title),
            ("condition", condition),
            ("description", content),
        ]
        rule_text = "\n".join(
            f"{name}: {escape_json_string(value)}" for name, value in parts if value
        )

        builder.add_section(
            name="rule-relativity-proposer-rule",
            template="""
Rule
-----------
{rule_text}
""",
            props={"rule_text": rule_text},
        )

        builder.add_section(
            name="rule-relativity-proposer-output-format",
            template="""
Use the following format to evaluate whether the rule references the agent's other policies
Expected output (JSON):
```json
{{
  "rationale": "<str, short explanation of whether the rule references the agent's other policies>",
  "references_other_policies": "<bool>"
}}
```
""",
        )

        return builder

    async def _generate_relativity(
        self,
        content: Optional[str],
        condition: Optional[str],
        title: str | None,
        agent_id: AgentId | None,
        temperature: float,
    ) -> RuleRelativityPropositionSchema:
        prompt = await self._build_prompt(content, condition, title)

        generator = await self._model_router.get_schematic_generator(
            context=None,
            route="indexing",
            schema=RuleRelativityPropositionSchema,
            agent_id=agent_id,
        )
        reasoning_effort = await self._model_router.resolve_schematic_reasoning_effort(
            context=None,
            route="indexing",
            agent_id=agent_id,
        )
        response = await generator.generate(
            prompt=prompt,
            hints={
                "temperature": temperature,
                **({"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}),
            },
        )

        return response.content
