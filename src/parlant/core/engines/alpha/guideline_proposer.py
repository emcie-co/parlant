# Copyright 2024 Emcie Co Ltd.
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
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
import json
import math
import time
from typing import Optional, Sequence

from parlant.core.agents import Agent
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.customers import Customer
from parlant.core.nlp.generation import GenerationInfo, SchematicGenerator
from parlant.core.engines.alpha.guideline_proposition import GuidelineProposition
from parlant.core.engines.alpha.prompt_builder import BuiltInSection, PromptBuilder, SectionStatus
from parlant.core.glossary import Term
from parlant.core.guidelines import Guideline
from parlant.core.sessions import Event
from parlant.core.emissions import EmittedEvent
from parlant.core.common import DefaultBaseModel
from parlant.core.logging import Logger


class GuidelinePropositionSchema(DefaultBaseModel):
    guideline_number: int
    condition: str
    condition_application_rationale: str
    condition_applies: bool
    action: Optional[str] = ""
    guideline_is_continuous: Optional[bool] = False
    guideline_previously_applied_rationale: Optional[str] = ""
    guideline_previously_applied: Optional[bool] = False
    guideline_should_reapply: Optional[bool] = False
    applies_score: int


class GuidelinePropositionsSchema(DefaultBaseModel):
    checks: Sequence[GuidelinePropositionSchema]


@dataclass(frozen=True)
class ConditionApplicabilityEvaluation:
    guideline_number: int
    condition: str
    action: str
    score: int
    condition_application_rationale: str
    guideline_previously_applied_rationale: str
    guideline_should_reapply: bool


@dataclass(frozen=True)
class GuidelinePropositionResult:
    total_duration: float
    batch_count: int
    batch_generations: Sequence[GenerationInfo]
    batches: Sequence[Sequence[GuidelineProposition]]

    @cached_property
    def propositions(self) -> Sequence[GuidelineProposition]:
        return list(chain.from_iterable(self.batches))


class GuidelineProposer:
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[GuidelinePropositionsSchema],
    ) -> None:
        self._logger = logger
        self._schematic_generator = schematic_generator

    async def propose_guidelines(
        self,
        agents: Sequence[Agent],
        customer: Customer,
        guidelines: Sequence[Guideline],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        staged_events: Sequence[EmittedEvent],
    ) -> GuidelinePropositionResult:
        if not guidelines:
            return GuidelinePropositionResult(
                total_duration=0.0, batch_count=0, batch_generations=[], batches=[]
            )

        guidelines_dict = {i: g for i, g in enumerate(guidelines, start=1)}
        t_start = time.time()
        batches = self._create_guideline_batches(
            guidelines_dict,
            batch_size=self._get_optimal_batch_size(guidelines_dict),
        )

        with self._logger.operation(
            f"Guideline proposal ({len(guidelines)} guidelines processed in {len(batches)} batches)"
        ):
            batch_tasks = [
                self._process_guideline_batch(
                    agents,
                    customer,
                    context_variables,
                    interaction_history,
                    staged_events,
                    terms,
                    batch,
                )
                for batch in batches
            ]

            batch_generations, condition_evaluations_batches = zip(
                *(await asyncio.gather(*batch_tasks))
            )

            propositions_batches: list[list[GuidelineProposition]] = []

            for batch in condition_evaluations_batches:
                guideline_propositions = []
                for evaluation in batch:
                    guideline_propositions.append(
                        GuidelineProposition(
                            guideline=guidelines_dict[evaluation.guideline_number],
                            score=evaluation.score,
                            rationale=f'''Condition Application: "{evaluation.condition_application_rationale}"; Guideline Previously Applied: "{evaluation.guideline_previously_applied_rationale}"''',
                            should_reapply=evaluation.guideline_should_reapply,
                        )
                    )
                propositions_batches.append(guideline_propositions)

            t_end = time.time()

            return GuidelinePropositionResult(
                total_duration=t_end - t_start,
                batch_count=len(batches),
                batch_generations=batch_generations,
                batches=propositions_batches,
            )

    def _get_optimal_batch_size(self, guidelines: dict[int, Guideline]) -> int:
        guideline_n = len(guidelines)

        if guideline_n <= 10:
            return 1
        elif guideline_n <= 20:
            return 2
        elif guideline_n <= 30:
            return 3
        else:
            return 5

    def _create_guideline_batches(
        self,
        guidelines_dict: dict[int, Guideline],
        batch_size: int,
    ) -> Sequence[dict[int, Guideline]]:
        batches = []
        guidelines = list(guidelines_dict.items())
        batch_count = math.ceil(len(guidelines_dict) / batch_size)
        for batch_number in range(batch_count):
            start_offset = batch_number * batch_size
            end_offset = start_offset + batch_size
            batch = dict(guidelines[start_offset:end_offset])
            batches.append(batch)

        return batches

    async def _process_guideline_batch(
        self,
        agents: Sequence[Agent],
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        staged_events: Sequence[EmittedEvent],
        terms: Sequence[Term],
        guidelines_dict: dict[int, Guideline],
    ) -> tuple[GenerationInfo, list[ConditionApplicabilityEvaluation]]:
        prompt = self._format_prompt(
            agents,
            customer,
            context_variables=context_variables,
            interaction_history=interaction_history,
            staged_events=staged_events,
            terms=terms,
            guidelines=guidelines_dict,
        )

        with self._logger.operation(
            f"Guideline evaluation batch ({len(guidelines_dict)} guidelines)"
        ):
            propositions_generation_response = await self._schematic_generator.generate(
                prompt=prompt,
                hints={"temperature": 0.3},
            )

        propositions = []

        for proposition in propositions_generation_response.content.checks:
            guideline = guidelines_dict[int(proposition.guideline_number)]

            self._logger.debug(
                f'Guideline evaluation for "when {guideline.content.condition} then {guideline.content.action}":\n'  # noqa
                f'  Score: {proposition.applies_score}/10; Condition rationale: "{proposition.condition_application_rationale}"; continuous: [{proposition.guideline_is_continuous}] ;Re-application rationale: "{proposition.guideline_previously_applied_rationale}"'
            )

            if (proposition.applies_score >= 6) and (
                not proposition.guideline_previously_applied or proposition.guideline_should_reapply
            ):
                propositions.append(
                    ConditionApplicabilityEvaluation(
                        guideline_number=proposition.guideline_number,
                        condition=guidelines_dict[
                            int(proposition.guideline_number)
                        ].content.condition,
                        action=guidelines_dict[int(proposition.guideline_number)].content.action,
                        score=proposition.applies_score,
                        condition_application_rationale=proposition.condition_application_rationale,
                        guideline_previously_applied_rationale=proposition.guideline_previously_applied_rationale
                        or "",
                        guideline_should_reapply=proposition.guideline_should_reapply or False,
                    )
                )

        return propositions_generation_response.info, propositions

    def _format_prompt(
        self,
        agents: Sequence[Agent],
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        staged_events: Sequence[EmittedEvent],
        terms: Sequence[Term],
        guidelines: dict[int, Guideline],
    ) -> str:
        assert len(agents) == 1

        result_structure = [
            {
                "guideline_number": i,
                "condition": g.content.condition,
                "condition_application_rationale": "<Explanation for why the condition is or isn't met>",
                "condition_applies": "<BOOL>",
                "action": g.content.action,
                "guideline_is_continuous": "<BOOL: Optional, only necessary if guideline_previously_applied is true>",
                "guideline_previously_applied_rationale": "<Explanation for whether and how this guideline was previously applied. Optional, necessary only if the condition applied>",
                "guideline_previously_applied": "<BOOL: Optional, whether the condition already applied and the action was already taken>",
                "guideline_should_reapply": "<BOOL: Optional, only necessary if guideline_previously_applied is true>",
                "applies_score": "<Relevance score of the guideline between 1 and 10. A higher score indicates that the guideline should be active>",
            }
            for i, g in guidelines.items()
        ]
        guidelines_text = "\n".join(
            f"{i}) Condition: {g.content.condition}. Action: {g.content.action}"
            for i, g in guidelines.items()
        )

        builder = PromptBuilder()

        builder.add_section(
            f"""
GENERAL INSTRUCTIONS
-----------------
In our system, the behavior of a conversational AI agent is guided by "guidelines". The agent makes use of these guidelines whenever it interacts with a user (also referred to as the customer).
Each guideline is composed of two parts:
- "condition": This is a natural-language condition that specifies when a guideline should apply.
          We look at each conversation at any particular state, and we test against this
          condition to understand if we should have this guideline participate in generating
          the next reply to the user.
- "action": This is a natural-language instruction that should be followed by the agent
          whenever the "condition" part of the guideline applies to the conversation in its particular state.
          Any instruction described here applies only to the agent, and not to the user.


Task Description
----------------
Your task is to evaluate the relevance and applicability of a set of provided 'when' conditions to the most recent state of an interaction between yourself (an AI assistant) and a user.
These conditions, along with the interaction details, will be provided later in this message.
For each condition that is met, determine whether its corresponding action should be taken by the agent or if it has already been addressed previously.


Process Description
-------------------
a. Examine Interaction Events: Review the provided interaction events to discern the most recent state of the interaction between the user and the assistant.
b. Evaluate Conditions: Assess the entire interaction to determine whether each condition is still relevant and directly fulfilled based on the most recent interaction state.
c. Check for Prior Action: Determine whether the condition has already been addressed, i.e., whether it applied in an earlier state and its corresponding action has already been performed.
d. Guideline Application: A guideline should be applied only if:
    (1) Its condition is currently met and its action has not been performed yet, or
    (2) The interaction warrants re-application of its action (e.g., when a recurring condition becomes true again after previously being fulfilled).

For each provided guideline, return:
    (1) Whether its condition is fulfilled.
    (2) Whether its action needs to be applied at this time. See the following section for more details.


Insights Regarding Guideline re-activation
-------------------
A condition typically no longer applies if its corresponding action has already been executed.
However, there are exceptions where re-application is warranted, such as when the condition is re-applied again. For example, a guideline with the condition "the customer is asking a question" should be applied again whenever the customer asks a question.
Additionally, actions that involve continuous behavior (e.g., "do not ask the user for their age", or guidelines involving the language the agent should use) should be re-applied whenever their condition is met. You can mark these types of actions down using "guideline_is_continuous" in your output.

Conversely, actions dictating one-time behavior (e.g., "send the user our address") should be re-applied more conservatively.
Only re-apply these if the condition ceased to be true earlier in the conversation before being fulfilled again in the current context.


Examples of Condition Evaluations:
-------------------
Example #1:
- Interaction Events: ###
[{{"id": "11", "kind": "<message>", "source": "customer",
"data": {{"message": "Can I purchase a subscription to your software?"}}}},
{{"id": "23", "kind": "<message>", "source": "assistant",
"data": {{"message": "Absolutely, I can assist you with that right now."}}}},
{{"id": "34", "kind": "<message>", "source": "customer",
"data": {{"message": "Please proceed with the subscription for the Pro plan."}}}},
{{"id": "56", "kind": "<message>", "source": "assistant",
"data": {{"message": "Your subscription has been successfully activated.
Is there anything else I can help you with?"}}}},
{{"id": "78", "kind": "<message>", "source": "customer",
"data": {{"message": "Yes, can you tell me more about your data security policies?"}}}}]
###
- Guidelines: ###
1) condition: the customer initiates a purchase. action: Open a new cart for the customer
2) condition: the customer asks about data security. action: Refer the customer to our privacy policy page
###
- **Expected Result**:
```json
{{ "checks": [
    {{
        "guideline_number": 1,
        "condition": "the customer initiates a purchase",
        "condition_application_rationale": "The purchase-related guideline was initiated earlier, but is currently irrelevant since the customer completed the purchase and the conversation has moved to a new topic.",
        "condition_applies": false,
        "applies_score": 3
    }},
    {{
        "guideline_number": 2,
        "condition": "the customer asks about data security",
        "condition_applies": true,
        "condition_application_rationale": "The customer specifically inquired about data security policies, making this guideline highly relevant to the ongoing discussion.",
        "action": "Refer the customer to our privacy policy page",
        "guideline_previously_applied_rationale": "This is the first time data security has been mentioned, and the user has not been referred to the privacy policy page yet",
        "guideline_previously_applied": false,
        "guideline_is_continuous": false,
        "guideline_should_reapply": false,
        "applies_score": 9
    }}
]}}
```

###

Example #2:
- Interaction Events: ###
[{{"id": "11", "kind": "<message>", "source": "customer",
"data": {{"message": "I'm looking for a job, what do you have available?"}}}},
{{"id": "23", "kind": "<message>", "source": "assistant",
"data": {{"message": "Hi there! we have plenty of opportunities for you, where are you located?"}}}},
{{"id": "34", "kind": "<message>", "source": "customer",
"data": {{"message": "I'm looking for anything around the bay area"}}}},
{{"id": "56", "kind": "<message>", "source": "assistant",
"data": {{"message": "That's great. We have a number of positions available over there. What kind of role are you interested in?"}}}},
{{"id": "78", "kind": "<message>", "source": "customer",
"data": {{"message": "Anything to do with training and maintaining AI agents"}}}}]

###
- Guidelines: ###
3) condition: the customer indicates that they are looking for a job. action: ask the customer for their location
4) condition: the customer asks about job openings. action: emphasize that we have plenty of positions relevant to the customer
6) condition: discussing job opportunities. action: maintain a positive, assuring tone

###
- **Expected Result**:
```json
{{ "checks": [
    {{
        "guideline_number": 3,
        "condition": "the customer indicates that they are looking for a job.",
        "condition_applies": true,
        "condition_application_rationale": "The current discussion is about the type of job the customer is looking for",
        "action": "ask the customer for their location",
        "guideline_is_continuous": false,
        "guideline_previously_applied_rationale": "The assistant asked for the customer's location earlier in the interaction. There is no need to ask for it again, as it is already known.",
        "guideline_previously_applied": true,
        "guideline_should_reapply": false,
        "applies_score": 3
    }},
    {{
        "guideline_number": 4,
        "condition": "the customer asks about job openings.",
        "condition_applies": true,
        "condition_application_rationale": "the customer asked about job openings, and the discussion still revolves around this request",
        "action": "emphasize that we have plenty of positions relevant to the customer",
        "guideline_is_continuous": false,
        "guideline_previously_applied_rationale": "The assistant already has emphasized that we have open positions. However, since the customer is narrowing down their search, this point should be re-emphasized to clarify that it still holds true.",
        "guideline_previously_applied": true,
        "guideline_should_reapply": true,
        "applies_score": 7
    }},
    {{
        "guideline_number": 6,
        "condition": "discussing job opportunities.",
        "condition_applies": true,
        "condition_application_rationale": "the discussion is about job opportunities that are relevant to the customer, so the condition applies.",
        "action": "maintain a positive, assuring tone",
        "guideline_is_continuous": true,
        "guideline_previously_applied_rationale": "The assistant's tone is positive already. This action describes a continuous action, so the guideline should be re-applied.",
        "guideline_previously_applied": true,
        "guideline_should_reapply": true,
        "applies_score": 9
    }},
    ]
}}
```

Example #3:
- Interaction Events: ###
[{{"id": "11", "kind": "<message>", "source": "customer",
"data": {{"message": "Hi there, what is the S&P500 trading at right now?"}}}},
{{"id": "23", "kind": "<message>", "source": "assistant",
"data": {{"message": "Hello! It's currently priced at just about 6,000$."}}}},
{{"id": "34", "kind": "<message>", "source": "customer",
"data": {{"message": "Better than I hoped. And what's the weather looking like today?"}}}},
{{"id": "56", "kind": "<message>", "source": "assistant",
"data": {{"message": "It's 5 degrees Celsius in London today"}}}},
{{"id": "78", "kind": "<message>", "source": "customer",
"data": {{"message": "Bummer. Does S&P500 still trade at 6,000$ by the way?"}}}}]

###
- Guidelines: ###
1) condition: the customer asks about the value of a stock. action: provide the price using the 'check_stock_price' tool
2) condition: the weather at a certain location is discussed. action: check the weather at that location using the 'check_weather' tool

###
- **Expected Result**:
```json
{{ "checks": [
    {{
        "guideline_number": 1,
        "condition": "the customer asks about the value of a stock.",
        "condition_applies": true,
        "condition_application_rationale": "The customer asked what does the S&P500 trade at",
        "action": "provide the price using the 'check_stock_price' tool",
        "guideline_is_continuous": false,
        "guideline_previously_applied_rationale": "The assistant previously reported about the price of that stock following the customer's question, but since the price might have changed since then, it should be checked again.",
        "guideline_previously_applied": true,
        "guideline_should_reapply": true,
        "applies_score": 9
    }},
    {{
        "guideline_number": 2,
        "condition": "the weather at a certain location is discussed.",
        "condition_applies": false,
        "condition_application_rationale": "while weather was discussed earlier, the conversation have moved on to an entirely different topic (stock prices)",
        "applies_score": 3
    }},
    ]
}}
```
"""  # noqa
        )
        builder.add_agent_identity(agents[0])
        builder.add_context_variables(context_variables)
        builder.add_glossary(terms)
        builder.add_interaction_history(interaction_history)
        builder.add_staged_events(staged_events)
        builder.add_section(
            name=BuiltInSection.GUIDELINES,
            content=f"""
- Guidelines list: ###
{guidelines_text}
###
""",
            status=SectionStatus.ACTIVE,
        )

        builder.add_section(f"""
IMPORTANT: Please note there are exactly {len(guidelines)} guidelines in the list for you to check.

Expected Output
---------------------------
- Specify the applicability of each predicate by filling in the rationale and applied score in the following list:

    ```json
    {{
        "checks":
        {json.dumps(result_structure)}
    }}
    ```""")

        prompt = builder.build()
        with open("guideline proposition prompt.txt", "w") as f:
            f.write(prompt)
        return prompt
