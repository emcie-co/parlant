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

from dataclasses import dataclass
from itertools import chain
import json
from typing import Optional, Sequence
from more_itertools import chunked

from parlant.core import async_utils
from parlant.core.agents import Agent
from parlant.core.common import DefaultBaseModel
from parlant.core.guideline_connections import ConnectionKind
from parlant.core.guidelines import GuidelineContent
from parlant.core.logging import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.glossary import GlossaryStore
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.services.indexing.common import ProgressReport


class GuidelineConnectionPropositionSchema(DefaultBaseModel):
    source_id: int
    target_id: int
    source_then: str
    target_when: str
    is_target_when_caused_by_source_then: bool
    is_source_then_suggestive_or_optional: bool = False
    target_then: str = ""
    is_target_then_suggestive_or_optional: bool = False
    rationale: str
    causation_score: int


class GuidelineConnectionPropositionsSchema(DefaultBaseModel):
    propositions: list[GuidelineConnectionPropositionSchema]


@dataclass(frozen=True)
class GuidelineConnectionProposition:
    source: GuidelineContent
    target: GuidelineContent
    kind: ConnectionKind
    score: int
    rationale: str


class GuidelineConnectionProposer:
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[GuidelineConnectionPropositionsSchema],
        glossary_store: GlossaryStore,
    ) -> None:
        self._logger = logger
        self._glossary_store = glossary_store
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

    async def _format_connection_propositions(
        self,
        agent: Agent,
        evaluated_guideline: GuidelineContent,
        comparison_set: dict[int, GuidelineContent],
    ) -> str:
        builder = PromptBuilder()
        builder.add_section(
            f"""
GENERAL INSTRUCTIONS
-----------------
In our system, the behavior of a conversational AI agent is guided by "guidelines". The agent makes use of these guidelines whenever it interacts with a customer.

Each guideline is composed of two parts:
- "when": This is a natural-language condition that specifies when a guideline should apply.
          We look at each conversation at any particular state, and we test against this
          condition to understand if we should have this guideline participate in generating
          the next reply to the customer.
- "then": This is a natural-language instruction that should be followed by the agent
          whenever the "when" part of the guideline applies to the conversation in its particular state.
          Any instruction described here applies only to the agent, and not to the customer.
Your task is to identify cases where the activation of one guideline causes another guideline's 'when' condition to apply.

When multiple guidelines are in use, we may encounter the following situation:
Guideline 1: When <X>, then <Y>.
Guideline 2: When <W>, then <Z>.
Situations where applying the "then" of Guideline 1 (<Y>) directly causes the "when" of Guideline 2 (<W>) to hold true, form what we call a "causal connection" or simply "causation" from Guideline 1 to Guideline 2. 


Rephrasing - cau 
This causation can only happen if the agent's action in <Y> directly causes the "when" in Guideline 2 (<W>) to become true.

Important clarification: An action taken by the agent can never cause the customer to do anything. Causation only occurs if applying the source's "then" action directly and immediately causes the "when" of the target guideline to apply. Cases where the source's "then" implies that the target's "when" happened in the past, or will happen in the future, are not considered causation.
As a result of this, if there's any scenario where the source's "then" can be applied while the target's "when" is false - then causation necessarily isn't fulfilled.

When a connection is identified, we wish to know whether it involves actions that the agent must undertake, or if its prescriptions are merely suggestive or optional, resulting in a "suggestive causal connection".
A causal connection is considered suggestive if either of the following conditions is met:
- The source guideline's "then" statement is suggestive or optional.
- The target guideline's "then" statement is suggestive or optional.
For example, a connection is suggestive if it's of the form {{source="When <X> then <Y>", target="When <W> then consider <Z>"}} (where <W> is caused by <Y>), or similarly if {{source="When <X> then only do <Y> under certain conditions", target="When <W> then <Z>"}}.
If both guidelines' "then" statements prescribe mandatory actions that the agent must take, then the connection is not considered suggestive. Conversely, if either "then" statement is optional or suggestive, the connection is considered suggestive.
'Then' statements which prescribe actions which the agent must attempt to take, even if they might fail, are NOT considered suggestive, as they describe an action that's mandatory.

Your task is to:
    1. Evaluate pairs of guidelines and detect which pairs fulfill such causal connections.
    2. Identify whether the causal connections are suggestive. Meaning, if the action described in either guideline's 'then' statement is optional.
    To fulfill the second task, please identify if each 'then' statement in a candidate connection as either suggestive or not.

Please output JSON structured in the following format:

{{
    "propositions": [
        {{
            "source_id": <id of the source guideline>,
            "target_id": <id of the target guideline>,
            "source_then": <The source guideline's 'then'>,
            "target_when": <The target guideline's 'when'>,
            "is_target_when_caused_by_source_then": <BOOL>,
            "is_source_then_suggestive_or_optional": <BOOL>,
            "target_then": <The target guideline's 'then'>,
            "is_target_then_suggestive_or_optional": <BOOL>,
            "rationale": <Explanation for if and how the source's 'then' causes the target's 'when'. The explanation should revolve around the word 'cause' or a conjugation of it>,
            "causation_score": <Score between 1-10 indicating the strength of the connection>
        }},
        ...
    ]
}}

For each causation candidate, you should evaluate two potential propositions: one in which the test guideline is treated as the "source" and the candidate as the "target," and another in which the roles are reversed

The following are examples of expected outputs for a given input:
###
Example 1:
{{
Input:

Test guideline: ###
{{"id": 0, "when": "providing the weather update", "then": "try to estimate whether it's likely to rain"}}
###

Causation candidates: ###
{{"id": 1, "when": "the customer asked about the weather", "then": "provide the current weather update"}}
{{"id": 2, "when": "discussing whether an umbrella is needed", "then": "refer the customer to our electronic store"}}
###

Expected Output:

```json
{{
    "propositions": [
        {{
            "source_id": 0,
            "target_id": 1,
            "source_then": "try to estimate whether it's likely to rain",
            "target_when": "the customer asked about the weather",
            "is_target_when_caused_by_source_then": false,
            "is_source_then_suggestive_or_optional": false,
            "target_then": "provide the current weather update",
            "is_target_then_suggestive_or_optional": false,
            "rationale": "the agent's mentioning the likelihood of rain does not cause the customer ask about the weather retrospectively",
            "causation_score": 3
        }},
        {{
            "source_id": 1,
            "target_id": 0,
            "source_then": "provide the current weather update",
            "target_when": "providing the weather update",
            "is_target_when_caused_by_source_then": true,
            "is_source_then_suggestive_or_optional": false,
            "target_then": "try to estimate whether it's likely to rain",
            "is_target_then_suggestive_or_optional": false,
            "rationale": "the agent's providing a current weather update necessarily causes a weather update to be provided",
            "causation_score": 10
        }},
        {{
            "source_id": 0,
            "target_id": 2,
            "source_then": "try to estimate whether it's likely to rain",
            "target_when": "discussing whether an umbrella is needed",
            "is_target_when_caused_by_source_then": false,
            "is_source_then_suggestive_or_optional": false,
            "target_then": "refer the customer to our electronic store",
            "is_target_then_suggestive_or_optional": false,
            "rationale": "the agent's mentioning the chances for rain does not retrospectively make the discussion about umbrellas",
            "causation_score": 3
        }},
        {{
            "source_id": 2,
            "target_id": 0,
            "source_then": "refer the customer to our electronic store",
            "target_when": "providing the weather update",
            "is_target_when_caused_by_source_then": false,
            "is_source_then_suggestive_or_optional": false,
            "target_then": "try to estimate whether it's likely to rain",
            "is_target_then_suggestive_or_optional": false,
            "rationale": "the agent's referring to the electronic store does not cause a weather update to be provided",
            "causation_score": 1
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
{{"id": 2, "when": "recommending books", "then": "consider highlighting the ones with the best reviews"}}
{{"id": 3, "when": "the customer greets you", "then": "greet them back with 'hello'"}}
{{"id": 4, "when": "suggesting products", "then": "check if the product is available in our store, and only offer it if it is"}}

Expected Output:
```json
{{
    "propositions": [
        {{
            "source_id": 0,
            "target_id": 1,
            "source_then": "suggest a book",
            "target_when": "suggesting a book",
            "is_target_when_caused_by_source_then": true,
            "rationale": "the agent's suggesting a book causes the suggestion of a book",
            "causation_score": 10
        }},
        {{
            "source_id": 1,
            "target_id": 0,
            "source_then": "mention its availability in the local library",
            "target_when": "The customer asks for a book recommendation",
            "is_target_when_caused_by_source_then": false,
            "rationale": "the agent's mentioning library availability does not retrospectively make the customer ask for book recommendations",
            "causation_score": 1
        }},
        {{
            "source_id": 0,
            "target_id": 2,
            "source_then": "suggest a book",
            "target_when": "recommending books",
            "is_target_when_caused_by_source_then": true,
            "is_source_then_suggestive_or_optional": false,
            "target_then": "consider highlighting the ones with the best reviews",
            "is_target_then_suggestive_or_optional": true,
            "rationale": "the agent's applying of 'suggest a book' causes the recommendation of books to occur. The target's then begins with 'consider', making it suggestive.",
            "causation_score": 9
        }},
        {{
            "source_id": 2,
            "target_id": 0,
            "source_then": "consider highlighting the ones with the best reviews",
            "target_when": "The customer asks for a book recommendation",
            "is_target_when_caused_by_source_then": false,
            "is_source_then_suggestive_or_optional": true,
            "target_then": "suggest a book",
            "is_target_then_suggestive_or_optional": false,
            "rationale": "the agent's highlighting reviews does not cause the customer to retrospectively ask for anything",
            "causation_score": 1
        }},
        {{
            "source_id": 0,
            "target_id": 3,
            "source_then": "suggest a book",
            "target_when": "the customer greets you",
            "is_target_when_caused_by_source_then": false,
            "is_source_then_suggestive_or_optional": false,
            "target_then": "greet them back with 'hello'",
            "is_target_then_suggestive_or_optional": false,
            "rationale": "the agent's suggesting a book does not cause the customer to greet the agent retrospectively",
            "causation_score": 1
        }},
        {{
            "source_id": 3,
            "target_id": 0,
            "source_then": "greet them back with 'hello'",
            "target_when": "The customer asks for a book recommendation",
            "is_target_when_caused_by_source_then": false,
            "rationale": "the agent's greeting the customer does not cause them to ask for a book recommendation retrospectively",
            "causation_score": 1
        }},
        {{
            "source_id": 0,
            "target_id": 4,
            "source_then": "suggest a book",
            "target_when": "suggesting products",
            "is_target_when_caused_by_source_then": true,
            "is_source_then_suggestive_or_optional": false,
            "target_then": "check if the product is available in our store, and only offer it if it is'",
            "is_target_then_suggestive_or_optional": true,
            "rationale": "the agent's suggesting a book, necessarily causes the suggestion of a product. The suggestion is optional, making the connection suggestive.",
            "causation_score": 9
        }},
        {{
            "source_id": 4,
            "target_id": 0,
            "source_then": "check if the product is available in our store, and only offer it if it is'",
            "target_when": "The customer asks for a book recommendation",
            "is_target_when_caused_by_source_then": false,
            "is_source_then_suggestive_or_optional": true,
            "target_then": "suggest a book",
            "is_target_then_suggestive_or_optional": false,
            "rationale": "the agent's checking product availability does not cause the customer to ask for book recommendations retrospectively",
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

Expected Output:
```json
{{
    "propositions": [
        {{
            "source_id": 0,
            "target_id": 1,
            "source_then": "announce that the suggestion will be forwarded to management for consideration",
            "target_when": "discussing opening hours",
            "is_target_when_caused_by_source_then": false,
            "rationale": "the agent's forwarding something to management has nothing to do with opening hours",
            "causation_score": 1
        }},
        {{
            "source_id": 1,
            "target_id": 0,
            "source_then": "mention that the store closes early on Sundays",
            "target_when": "a new topping is suggested",
            "is_target_when_caused_by_source_then": false,
            "rationale": "the agent's store hours discussion does not cause any new topping suggestion to occur",
            "causation_score": 1
        }},
        {{
            "source_id": 0,
            "target_id": 2,
            "source_then": "announce that the suggestion will be forwarded to management for consideration",
            "target_when": "the customer asks for a topping we do not offer",
            "is_target_when_caused_by_source_then": false,
            "rationale": "the agent's announcing something does not cause the customer to have retrospectively asked about anything regarding toppings",
            "causation_score": 2
        }},
        {{
            "source_id": 2,
            "target_id": 0,
            "source_then": "suggest to add the topping to the menu in the future",
            "target_when": "a new topping is suggested",
            "is_target_when_caused_by_source_then": true,
            "is_source_then_suggestive_or_optional": false,
            "target_then": "announce that the suggestion will be forwarded to management for consideration",
            "is_target_then_suggestive_or_optional": false,
            "rationale": "the agent's suggesting to add the topping to the menu is causing a new topping is being suggested",
            "causation_score": 9
        }},
        {{
            "source_id": 0,
            "target_id": 3,
            "source_then": "announce that the suggestion will be forwarded to management for consideration",
            "target_when": "forwarding messages to management",
            "is_target_when_caused_by_source_then": true,
            "is_source_then_suggestive_or_optional": false,
            "target_then": "try to forward the message to management via email",
            "is_target_then_suggestive_or_optional": false,
            "rationale": "the agent's' announcement from the source's 'then' should cause a message to be forwarded to management",
            "causation_score": 8
        }},
        {{
            "source_id": 3,
            "target_id": 0,
            "source_then": "try to forward the message to management via email",
            "target_when": "a new topping is suggested",
            "is_target_when_caused_by_source_then": false,
            "is_source_then_suggestive_or_optional": false,
            "target_then": "announce that the suggestion will be forwarded to management for consideration",
            "is_target_then_suggestive_or_optional": false,
            "rationale": "the agent's emailing a message is not necessarily a new topping suggestion",
            "causation_score": 2
        }}
    ]
}}
```

###
Example 4:
Input:

Test guideline: ###
{{"id": 0, "when": "the customer requests a refund", "then": "ask the customer for the date of their purchase"}}
###

Causation candidates: ###
{{"id": 1, "when": "the customer mentions a past purchase", "then": "ask for the order number"}}
###

Expected Output:

```json
{{
    "propositions": [
        {{
            "source_id": 0,
            "target_id": 1,
            "source_then": "ask the customer for the date of their purchase",
            "target_when": "the customer mentions a past purchase",
            "is_target_when_caused_by_source_then": false,
            "is_source_then_suggestive_or_optional": false,
            "target_then": "ask for the order number",
            "is_target_then_suggestive_or_optional": false,
            "rationale": "actions taken by the agent cannot ever cause the customer to do anything",
            "causation_score": 3
        }},
        {{
            "source_id": 1,
            "target_id": 0,
            "source_then": "ask for the order number",
            "target_when": "the customer requests a refund",
            "is_target_when_caused_by_source_then": false,
            "is_source_then_suggestive_or_optional": false,
            "target_then": "ask the customer for the date of their purchase",
            "is_target_then_suggestive_or_optional": false,

            "rationale": "actions taken by the agent cannot ever cause the customer to do anything",
            "causation_score": 3
        }}
    ]
}}
```

###
Example 5:
{{
Input:

Test guideline: ###
{{"id": 0, "when": "mentioning electronic products", "then": "check if the product is available, and inform the customer about its price if it is"}}
###

Causation candidates: ###
{{"id": 1, "when": "the customer complains about their television", "then": "consider suggesting buying a new TV"}}
{{"id": 2, "when": "discussing product prices", "then": "inform the customer about our black friday deals"}}
###

Expected Output:

```json
{{
    "propositions": [
        {{
            "source_id": 0,
            "target_id": 1,
            "source_then": "check if the product is available, and inform the customer about its price if it is",
            "target_when": "the customer complains about their television",
            "is_target_when_caused_by_source_then": false,
            "is_source_then_suggestive_or_optional": true,
            "target_then": "consider suggesting buying a new TV",
            "is_target_then_suggestive_or_optional": true,
            "rationale": "actions taken by the agent cannot ever cause the customer to do anything",
            "causation_score": 1
        }},
        {{
            "source_id": 1,
            "target_id": 0,
            "source_then": "consider suggesting buying a new TV",
            "target_when": "mentioning electronic products",
            "is_target_when_caused_by_source_then": true,
            "is_source_then_suggestive_or_optional": true,
            "target_then": "check if the product is available, and inform the customer about its price if it is",
            "is_target_then_suggestive_or_optional": true,
            "rationale": "suggesting to buy a new TV causes the mentioning of a TV, which is an electronic product",
            "causation_score": 9
        }},
        {{
            "source_id": 0,
            "target_id": 2,
            "source_then": "check if the product is available, and inform the customer about its price if it is",
            "target_when": "discussing product prices",
            "is_target_when_caused_by_source_then": true,
            "is_source_then_suggestive_or_optional": true,
            "target_then": "inform the customer about our black friday deals",
            "is_target_then_suggestive_or_optional": false,
            "rationale": "informing the customer about a price causes the discussion to be about product prices",
            "causation_score": 9
        }},
        {{
            "source_id": 2,
            "target_id": 0,
            "source_then": "inform the customer about our black friday deals",
            "target_when": "mentioning electronic products",
            "is_target_when_caused_by_source_then": false,
            "is_source_then_suggestive_or_optional": false,
            "target_then": "check if the product is available, and inform the customer about its price if it is",
            "is_target_then_suggestive_or_optional": true,
            "rationale": "informing about black friday deals does not necessarily cause the mentioning of an electronic product",
            "causation_score": 2
        }}
    ]
}}
'''
"""  # noqa
        )

        builder.add_agent_identity(agent)
        # Find and add glossary to prompt
        causation_candidates = "\n\t".join(
            f"{{id: {id}, when: {g.condition}, then: {g.action}}}"
            for id, g in comparison_set.items()
        )
        test_guideline = f"{{id: 0, when: '{evaluated_guideline.condition}', then: '{evaluated_guideline.action}'}}"
        terms = await self._glossary_store.find_relevant_terms(
            agent.id,
            query=test_guideline + causation_candidates,
        )
        builder.add_glossary(terms)

        builder.add_section(
            f"""
The guidelines you should analyze for connections are:
Test guideline: ###
{test_guideline}
###

Causation candidates: ###
{causation_candidates}
###"""
        )
        return builder.build()

    async def _generate_propositions(
        self,
        agent: Agent,
        guideline_to_test: GuidelineContent,
        guidelines_to_compare: Sequence[GuidelineContent],
        progress_report: Optional[ProgressReport],
    ) -> list[GuidelineConnectionProposition]:
        guidelines_dict = {i: g for i, g in enumerate(guidelines_to_compare, start=1)}
        guidelines_dict[0] = guideline_to_test
        prompt = await self._format_connection_propositions(
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
                kind={
                    False: ConnectionKind.ENTAILS,
                    True: ConnectionKind.SUGGESTS,
                }[
                    p.is_source_then_suggestive_or_optional
                    or p.is_target_then_suggestive_or_optional
                ],
                score=int(p.causation_score),
                rationale=p.rationale,
            )
            for p in response.content.propositions
            if p.causation_score >= 7
        ]

        if progress_report:
            await progress_report.increment()

        return relevant_propositions
