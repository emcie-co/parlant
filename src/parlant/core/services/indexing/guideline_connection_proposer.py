# Copyright 2025 Emcie Co Ltd.
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

from dataclasses import dataclass
from itertools import chain
import json
from typing import Optional, Sequence
from more_itertools import chunked

from parlant.core import async_utils
from parlant.core.agents import Agent
from parlant.core.common import DefaultBaseModel
from parlant.core.entity_cq import EntityQueries
from parlant.core.guidelines import GuidelineContent
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.services.indexing.common import ProgressReport


class GuidelineConnectionPropositionSchema(DefaultBaseModel):
    source_id: int
    target_id: int
    source_when: str
    source_then: str
    target_when: str
    target_when_is_customer_action: bool
    rationale: str
    is_target_when_caused_by_source_then: str
    causation_score: int


class GuidelineConnectionPropositionsSchema(DefaultBaseModel):
    propositions: list[GuidelineConnectionPropositionSchema]


@dataclass(frozen=True)
class GuidelineConnectionProposition:
    source: GuidelineContent
    target: GuidelineContent
    score: int
    rationale: str


class GuidelineConnectionProposer:
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[GuidelineConnectionPropositionsSchema],
        entity_queries: EntityQueries,
    ) -> None:
        self._logger = logger
        self._entity_queries = entity_queries
        self._schematic_generator = schematic_generator
        self._batch_size = 1

    async def propose_connections(
        self,
        agent: Agent,
        introduced_guidelines: Sequence[GuidelineContent],
        existing_guidelines: Sequence[GuidelineContent] = [],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[GuidelineConnectionProposition]:
        if not introduced_guidelines:
            return []

        connection_proposition_tasks = []

        for i, introduced_guideline in enumerate(introduced_guidelines):
            filtered_existing_guidelines = [
                g
                for g in chain(
                    introduced_guidelines[i + 1 :],
                    existing_guidelines,
                )
            ]

            guideline_batches = list(chunked(filtered_existing_guidelines, self._batch_size))

            if progress_report:
                await progress_report.stretch(len(guideline_batches))

            connection_proposition_tasks.extend(
                [
                    self._generate_propositions(agent, introduced_guideline, batch, progress_report)
                    for batch in guideline_batches
                ]
            )

        with self._logger.operation(
            f"Propose guideline connections for {len(connection_proposition_tasks)} "  # noqa
            f"batches (batch size={self._batch_size})",
        ):
            propositions = chain.from_iterable(
                await async_utils.safe_gather(*connection_proposition_tasks)
            )
            return list(propositions)

    async def _build_prompt(
        self,
        agent: Agent,
        evaluated_guideline: GuidelineContent,
        comparison_set: dict[int, GuidelineContent],
    ) -> PromptBuilder:
        builder = PromptBuilder()
        builder.add_agent_identity(agent)
        builder.add_section(
            name="guideline-connection-proposer-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
In our system, the behavior of a conversational AI agent is guided by "guidelines."
These guidelines are used by the agent whenever it responds to a customer in a chat.

Each guideline is composed of two parts:
- "when": A natural-language condition specifying when the guideline applies. After every message that the user (also referred to as the customer) sends, this condition is tested to determine if the guideline should influence the agent's next reply.
- "then": A natural-language instruction that the agent must follow whenever the "when" condition applies to the current state of the conversation. These instructions are directed solely at the agent and do not apply to the customer.

After each message by the customer, a separate component examines which guidelines need to be activated, meaning that they'd affect the agent's response, based on whether their 'when' condition is fulfilled.
Your task is to identify causal connections between guidelines. A causal connection exists when the activation of one guideline directly triggers the conditions of another.

Your role in our system is to determine which guidelines should automatically cause another guideline to activate.
Meaning, to identify cases where the activation of one guideline’s "then" statement directly causes the "when" condition of another guideline to apply.

HOW TO IDENTIFY A CAUSAL CONNECTION
-----------------
Consider two guidelines:
Guideline 1: When <X>, then <Y>.
Guideline 2: When <W>, then <Z>.
We say that guideline 1 forms a "causal connection" (or simply causes) guideline 2 iff applying the "then" of Guideline 1 (<Y>) directly causes the "when" of Guideline 2 (<W>) to hold true.

Additionally, you may assume the following: If <Y> occurs due to Guideline 1, then <X> is currently true. This means both <X> and <Y> can be considered true when assessing whether <W> is true for Guideline 2.

Important clarifications:
1. Assume that the condition of the source guideline is true. Meaning, examine whether the fulfillment of both the 'when' and 'then' statements of the source guideline cause the target's 'when' to be true.

2. Temporal Constraints: If the target guideline’s "when" condition was true in the past or will only become true in the future due to other factors, this is not considered causation. Causation is invalid if the source’s "then" statement can be applied while the target’s "when" condition remains false.

3. Actions do not Apply to the Customer: The action ascribed by the source guideline's 'then' statement cannot directly cause the customer to do something. If the target's 'when' statement describes a user action, mark it as so using the field target_when_is_customer_action, and note that a connection cannot be formed

4. VERY IMPORTANT: Optional or Suggestive Actions: If a guideline has an action that is suggestive ("consider recommending tomatoes"), or should only be followed under certain conditions (I.e. "check if the item is available and if it is suggest it"), assume that action is mandatory when evaluating the guideline.

HOW TO FILL is_target_when_caused_by_source_then
-----------------
The field is_target_when_caused_by_source_then indicates the type of causal relationship between the source guideline's "then" statement and the target guideline's "when" condition.
You must select the most accurate option from the following, listed from the weakest connection to the strongest:
1. "no" - Select this if there is no causal relationship between the source's action and the target's condition.
2. "implies" - Use this when the source's action makes it likely or possible that the target's condition holds true, but does not guarantee it. This is a weaker form of causation, where the action suggests or sets up conditions for the target.
    Example: the action "respond with our opening hours" implies the condition "you are asked about our opening hours"
3. "causes general case" - Choose this when the source's action causes a broad or generic condition, and the target's condition is a specific instance of that broader condition.
    Example: the action "refer the customer to our documentation" causes the general case for the condition "you are referring the customer to our usage documentation"
4. "causes" - Use this for direct and necessary causation, where the source's action guarantees that the target's condition immediately becomes true.
    Example: the action "refer the customer to our documentation" causes the specific case for the condition "referring the customer to our documentation"

Always select the strongest option that applies to the provided guidelines.
E.g., if you could both say that guideline A implies guideline B and that it causes guideline B, choose the latter.

EXAMPLES
-----------------
The following are examples of expected outputs for a given input:
###
Example 1:
Input:

Test guideline: ###
{{"id": 0, "when": "providing the weather update", "then": "try to estimate whether it's likely to rain"}}
###

Causation candidates: ###
{{"id": 1, "when": "the customer asked about the weather", "then": "provide the current weather update"}}
{{"id": 2, "when": "discussing whether an umbrella is needed", "then": "refer the customer to our electronic store"}}
{{"id": 3, "when": "reporting whether it's likely to rain tomorrow", "then": "mention tomorrow's date in your reply"}}
###

Expected Output:

```json
{{
    "propositions": [
        {{
            "source_id": 0,
            "target_id": 1,
            "source_when": "providing the weather update",
            "source_then": "try to estimate whether it's likely to rain",
            "target_when": "the customer asked about the weather",
            "target_when_is_customer_action": true,
            "rationale": "the agent's mentioning the likelihood of rain does not cause the customer ask about the weather retrospectively",
            "is_target_when_caused_by_source_then": "no",
            "causation_score": 3
        }},
        {{
            "source_id": 1,
            "target_id": 0,
            "source_when": "the customer asked about the weather",
            "source_then": "provide the current weather update",
            "target_when": "providing the weather update",
            "target_when_is_customer_action": false,
            "rationale": "the agent's providing a current weather update necessarily causes a weather update to be provided",
            "is_target_when_caused_by_source_then": "causes",
            "causation_score": 10
        }},
        {{
            "source_id": 0,
            "target_id": 2,
            "source_when": "providing the weather update",
            "source_then": "try to estimate whether it's likely to rain",
            "target_when": "discussing whether an umbrella is needed",
            "target_when_is_customer_action": false,
            "rationale": "the agent's mentioning the chances for rain does not retrospectively make the discussion about umbrellas, though it does imply it",
            "is_target_when_caused_by_source_then": "implies",
            "causation_score": 3
        }},
        {{
            "source_id": 2,
            "target_id": 0,
            "source_when": "discussing whether an umbrella is needed",
            "source_then": "refer the customer to our electronic store",
            "target_when": "providing the weather update",
            "target_when_is_customer_action": false,
            "rationale": "the agent's referring to the electronic store does not cause a weather update to be provided",
            "is_target_when_caused_by_source_then": "no",
            "causation_score": 1
        }},
        {{
            "source_id": 0,
            "target_id": 3,
            "source_when": "providing the weather update",
            "source_then": "try to estimate whether it's likely to rain",
            "target_when": "reporting whether it's likely to rain tomorrow",
            "target_when_is_customer_action": false,
            "rationale": "Estimating whether it's likely to rain causes us to report the chances for rain, but since we're not necessarily reporting about tomorrow, it's merely implied.",
            "is_target_when_caused_by_source_then": "implies",
            "causation_score": 5
        }},
        {{
            "source_id": 3,
            "target_id": 0,
            "source_when": "reporting whether it's likely to rain tomorrow",
            "source_then": "mention tomorrow's date in your reply",
            "target_when": "providing the weather update",
            "target_when_is_customer_action": false,
            "rationale": "Mentioning tomorrow's date while reporting weather does not necessarily cause a weather update to be provided.",
            "is_target_when_caused_by_source_then": "no",
            "causation_score": 3
        }}
    ]
}}
```

Example 2
Input:
Test guideline: ###
{{"id": 0, "when": "The customer asks for a book recommendation", "then": "suggest a book"}}
###
Causation candidates:
###
{{"id": 1, "when": "suggesting a book", "then": "mention its availability in the local library"}}
{{"id": 2, "when": "the customer greets you", "then": "greet them back with 'hello'"}}
{{"id": 3, "when": "offering the customer products", "then": "check if the product is available in our store, and only offer it if it is"}}

Expected Output:
```json
{{
    "propositions": [
        {{
            "source_id": 0,
            "target_id": 1,
            "source_when": "The customer asks for a book recommendation",
            "source_then": "suggest a book",
            "target_when": "suggesting a book",
            "target_when_is_customer_action": false,
            "rationale": "the agent's suggesting a book after being asked for book recommendations directly causes the suggestion of a book to be made",
            "is_target_when_caused_by_source_then": "causes",
            "causation_score": 10
        }},
        {{
            "source_id": 1,
            "target_id": 0,
            "source_when": "suggesting a book",
            "source_then": "mention its availability in the local library",
            "target_when": "The customer asks for a book recommendation",
            "target_when_is_customer_action": true,
            "rationale": "the agent's mentioning library availability does not retrospectively make the customer ask for book recommendations",
            "is_target_when_caused_by_source_then": "no",
            "causation_score": 1
        }},
        {{
            "source_id": 0,
            "target_id": 2,
            "source_when": "The customer asks for a book recommendation",
            "source_then": "suggest a book",
            "target_when": "the customer greets you",
            "target_when_is_customer_action": true,
            "rationale": "the agent's suggesting a book does not cause the customer to greet the agent retrospectively",
            "is_target_when_caused_by_source_then": "no",

            "causation_score": 1
        }},
        {{
            "source_id": 2,
            "target_id": 0,
            "source_when": "the customer greets you",
            "source_then": "greet them back with 'hello'",
            "target_when": "The customer asks for a book recommendation",
            "target_when_is_customer_action": true,
            "rationale": "the agent's greeting the customer does not cause them to ask for a book recommendation retrospectively",
            "is_target_when_caused_by_source_then": "no",
            "causation_score": 1
        }},
        {{
            "source_id": 0,
            "target_id": 3,
            "source_when": "The customer asks for a book recommendation",
            "source_then": "suggest a book",
            "target_when": "offering the customer products",
            "target_when_is_customer_action": false,
            "rationale": "the agent's suggesting a book, necessarily causes the offering of a product.",
            "is_target_when_caused_by_source_then": "causes",
            "causation_score": 9
        }},
        {{
            "source_id": 3,
            "target_id": 0,
            "source_when": "suggesting products",
            "source_then": "check if the product is available in our store, and only offer it if it is'",
            "target_when": "The customer asks for a book recommendation",
            "target_when_is_customer_action": true,
            "rationale": "the agent's checking product availability does not cause the customer to ask for book recommendations retrospectively",
            "is_target_when_caused_by_source_then": "no",
            "causation_score": 2
        }}
    ]
}}
```

###
Example 3
Input:
Test guideline: ###
{{"id": 0, "when": "a new topping is suggested", "then": "announce that the suggestion will be forwarded to management for consideration"}}
###
Causation candidates: ###
{{"id": 1, "when": "discussing opening hours", "then": "mention that the store closes early on Sundays"}}
{{"id": 2, "when": "the customer asks for a topping we do not offer", "then": "suggest to add the topping to the menu in the future"}}
{{"id": 3, "when": "forwarding messages to management", "then": "try to forward the message to management via email"}}
{{"id": 4, "when": "forwarding messages to our CEO", "then": "keep the message brief and to the point"}}
Expected Output:
```json
{{
    "propositions": [
        {{
            "source_id": 0,
            "target_id": 1,
            "source_when": "a new topping is suggested",
            "source_then": "announce that the suggestion will be forwarded to management for consideration",
            "target_when": "discussing opening hours",
            "target_when_is_customer_action": false,
            "rationale": "the agent's forwarding something to management has nothing to do with opening hours",
            "is_target_when_caused_by_source_then": "no",
            "causation_score": 1
        }},
        {{
            "source_id": 1,
            "target_id": 0,
            "source_when": "discussing opening hours",
            "source_then": "mention that the store closes early on Sundays",
            "target_when": "a new topping is suggested",
            "target_when_is_customer_action": false,
            "rationale": "the agent's store hours discussion does not cause any new topping suggestion to occur",
            "is_target_when_caused_by_source_then": "no",
            "causation_score": 1
        }},
        {{
            "source_id": 0,
            "target_id": 2,
            "source_when": "a new topping is suggested",
            "source_then": "announce that the suggestion will be forwarded to management for consideration",
            "target_when": "the customer asks for a topping we do not offer",
            "target_when_is_customer_action": true,
            "rationale": "the agent's announcing something does not cause the customer to have retrospectively asked about anything regarding toppings",
            "is_target_when_caused_by_source_then": "no",
            "causation_score": 2
        }},
        {{
            "source_id": 2,
            "target_id": 0,
            "source_when": "the customer asks for a topping we do not offer",
            "source_then": "suggest to add the topping to the menu in the future",
            "target_when": "a new topping is suggested",
            "target_when_is_customer_action": false,
            "rationale": "the agent's suggesting to add the topping to the menu is causing a new topping is being suggested",
            "is_target_when_caused_by_source_then": "causes",
            "causation_score": 9
        }},
        {{
            "source_id": 0,
            "target_id": 3,
            "source_when": "a new topping is suggested",
            "source_then": "announce that the suggestion will be forwarded to management for consideration",
            "target_when": "forwarding messages to management",
            "target_when_is_customer_action": false,
            "rationale": "the agent's' announcement from the source's 'then' should cause a message to be forwarded to management",
            "is_target_when_caused_by_source_then": "causes",
            "causation_score": 8
        }},
        {{
            "source_id": 3,
            "target_id": 0,
            "source_when": "forwarding messages to management",
            "source_then": "try to forward the message to management via email",
            "target_when": "a new topping is suggested",
            "target_when_is_customer_action": false,
            "rationale": "the agent's emailing a message is not necessarily a new topping suggestion",
            "is_target_when_caused_by_source_then": "no",
            "causation_score": 2
        }},
                {{
            "source_id": 0,
            "target_id": 4,
            "source_when": "a new topping is suggested",
            "source_then": "announce that the suggestion will be forwarded to management for consideration",
            "target_when": "forwarding messages to our CEO",
            "target_when_is_customer_action": false,
            "rationale": "announcing that we are forwarding the message to management may imply that it will be forwarded to our CEO, but it doesn't necessarily send it to them specifically",
            "is_target_when_caused_by_source_then": "causes general case",
            "causation_score": 5
        }},
        {{
            "source_id": 4,
            "target_id": 0,
            "source_when": "forwarding messages to our CEO",
            "source_then": "keep the message brief and to the point",
            "target_when": "a new topping is suggested",
            "target_when_is_customer_action": false,
            "rationale": "keeping a message brief does not cause a new topping suggestion",
            "is_target_when_caused_by_source_then": "no",
            "causation_score": 1
        }}
    ]
}}
```

###
Example 4
Input:
Test guideline: ###
{{"id": 0, "when": "a senior customer asks to extend their deal", "then": "Continue with the request if the customer's contract ends in the upcoming year"}}
###
Causation candidates: ###
{{"id": 1, "when": "extending the contract of a senior customer", "then": "Confirm with them that they have read the contract terms"}}
{{"id": 2, "when": "processing a two-way contract extension", "then": "Inform the customer that they might get sent to the G league in the future"}}
Expected Output:
```json
{{
    "propositions": [
        {{
            "source_id": 0,
            "target_id": 1,
            "source_when": "a senior customer asks to extend their deal",
            "source_then": "Continue with the request if the customer's contract ends in the upcoming year",
            "target_when": "extending the contract of a senior customer",
            "target_when_is_customer_action": false,
            "rationale": "continuing to process the request of a senior customer to extend their deal causes the process of extending the contract of a senior customer",
            "is_target_when_caused_by_source_then": "causes",
            "causation_score": 7
        }},
        {{
            "source_id": 1,
            "target_id": 0,
            "source_when": "extending the contract of a senior customer",
            "source_then": "Confirm with them that they have read the contract terms",
            "target_when": "a senior customer asks to extend their deal",
            "target_when_is_customer_action": true,
            "rationale": "an action by the agent can never cause the customer to ask to extend their deal retrospectively",
            "is_target_when_caused_by_source_then": "no",
            "causation_score": 1
        }},
        {{
            "source_id": 0,
            "target_id": 2,
            "source_when": "a senior customer asks to extend their deal",
            "source_then": "Continue with the request if the customer's contract ends in the upcoming year",
            "target_when": "processing a two-way contract extension",
            "target_when_is_customer_action": false,
            "rationale": "while continuing with the request of a senior customer to extend their deal causes a contract extension, it might not be a two-way contract specifically",
            "is_target_when_caused_by_source_then": "causes general case",
            "causation_score": 4
        }},
        {{
            "source_id": 2,
            "target_id": 0,
            "source_when": "processing a two-way contract extension",
            "source_then": "Inform the customer that they might get sent to the G league in the future",
            "target_when": "a senior customer asks to extend their deal",
            "target_when_is_customer_action": true,
            "rationale": "an action by the agent can never cause the customer to ask to extend their deal retrospectively",
            "is_target_when_caused_by_source_then": "no",
            "causation_score": 1
        }}
    ]
}}
```

ADDITIONAL INFORMATION
-----------------
""",
            props={
                "agent": agent,
                "evaluated_guideline": evaluated_guideline,
                "comparison_set": comparison_set,
            },
        )

        # Find and add glossary to prompt
        causation_candidates = "\n\t".join(
            f"{{id: {id}, when: {g.condition}, then: {g.action}}}"
            for id, g in comparison_set.items()
        )
        test_guideline = f"{{id: 0, when: '{evaluated_guideline.condition}', then: '{evaluated_guideline.action}'}}"
        terms = await self._entity_queries.find_glossary_terms_for_context(
            agent_id=agent.id,
            query=test_guideline + causation_candidates,
        )

        builder.add_glossary(terms)
        builder.add_section(
            name="guideline-connection-proposer-guidelines-to-analyze",
            template="""
The guidelines you should analyze for connections are:
Test guideline: ###
{test_guideline}
###

Causation candidates: ###
{causation_candidates}
###""",
            props={
                "test_guideline": test_guideline,
                "causation_candidates": causation_candidates,
            },
        )

        output_propositions_format = "\n".join(
            [
                f"""
        {{
            "source_id": 0,
            "target_id": {id},
            "source_when: {evaluated_guideline.condition},
            "source_then": {evaluated_guideline.action},
            "target_when": {g.condition},
            "target_when_is_customer_action": <BOOL>,
            "rationale": <Explanation for if and how the source's 'then' causes the target's 'when'. The explanation should revolve around the word 'cause' or a conjugation of it>,
            "is_target_when_caused_by_source_then": <str, either 'causes', 'causes general case', 'implies' or 'no'>,
            "causation_score": <Score between 1-10 indicating the strength of the connection>
        }},
        {{
            "source_id": {id},
            "target_id": 0,
            "source_when: {g.condition},
            "source_then": {g.action},
            "target_when": {evaluated_guideline.condition},
            "target_when_is_customer_action": <BOOL>,
            "rationale": <Explanation for if and how the source's 'then' causes the target's 'when'. The explanation should revolve around the word 'cause' or a conjugation of it>,
            "is_target_when_caused_by_source_then": <str, either 'causes', 'causes general case', 'implies' or 'no'>,
            "causation_score": <Score between 1-10 indicating the strength of the connection>
        }},
            """
                for id, g in comparison_set.items()
            ]
        )

        builder.add_section(
            name="guideline-connection-proposer-output-format",
            template="""
OUTPUT FORMAT
-----------------
Please output JSON structured in the following format, which includes two entries for each causation candidate - once with it as the source and once with it as the target:
```json
{{
    "propositions": [
        {output_propositions_format}
    ]
}}
```
""",
            props={
                "output_propositions_format": output_propositions_format,
                "comparison_set": comparison_set,
                "evaluated_guideline": evaluated_guideline,
            },
        )

        return builder

    async def _generate_propositions(
        self,
        agent: Agent,
        guideline_to_test: GuidelineContent,
        guidelines_to_compare: Sequence[GuidelineContent],
        progress_report: Optional[ProgressReport],
    ) -> list[GuidelineConnectionProposition]:
        guidelines_dict = {i: g for i, g in enumerate(guidelines_to_compare, start=1)}
        guidelines_dict[0] = guideline_to_test
        prompt = await self._build_prompt(
            agent,
            guideline_to_test,
            {k: v for k, v in guidelines_dict.items() if k != 0},
        )
        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": 0.0},
        )

        self._logger.debug(
            f"""
----------------------------------------
Connection Propositions Found:
----------------------------------------
{json.dumps([p.model_dump(mode="json") for p in response.content.propositions], indent=2)}
----------------------------------------
"""
        )

        relevant_propositions = [
            GuidelineConnectionProposition(
                source=guidelines_dict[p.source_id],
                target=guidelines_dict[p.target_id],
                score=int(p.causation_score),
                rationale=p.rationale,
            )
            for p in response.content.propositions
            if p.causation_score >= 7
        ]

        if progress_report:
            await progress_report.increment()

        return relevant_propositions
