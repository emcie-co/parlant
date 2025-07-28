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

import asyncio
from datetime import datetime, timezone
from enum import Enum, auto
from itertools import chain
import json
from typing import Optional, Sequence
from more_itertools import chunked
from dataclasses import dataclass

from parlant.core import async_utils
from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.entity_cq import EntityQueries
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.guidelines import GuidelineContent
from parlant.core.loggers import Logger
from parlant.core.agents import Agent
from parlant.core.services.indexing.common import ProgressReport


EVALUATION_BATCH_SIZE = 5
CRITICAL_INCOHERENCE_THRESHOLD = 6
ACTION_CONTRADICTION_SEVERITY_THRESHOLD = 6


class IncoherenceKind(Enum):
    STRICT = auto()
    CONTINGENT = auto()


class ConditionsEntailmentTestSchema(DefaultBaseModel):
    compared_guideline_id: int
    origin_guideline_when: str
    compared_guideline_when: str
    origin_entails_compared_rationale: str
    origin_when_entails_compared_when: bool
    origin_entails_compared_severity: int
    compared_entails_origin_rationale: str
    compared_when_entails_origin_when: bool
    compared_entails_origin_severity: int


class ConditionsEntailmentTestsSchema(DefaultBaseModel):
    condition_entailments: list[ConditionsEntailmentTestSchema]


class ActionsContradictionTestSchema(DefaultBaseModel):
    compared_guideline_id: int
    origin_guideline_then: str
    compared_guideline_then: str
    rationale: str
    thens_contradiction: bool
    severity: int


class ActionsContradictionTestsSchema(DefaultBaseModel):
    action_contradictions: list[ActionsContradictionTestSchema]


@dataclass(frozen=True)
class IncoherenceTest:
    guideline_a: GuidelineContent
    guideline_b: GuidelineContent
    IncoherenceKind: IncoherenceKind
    conditions_entailment_rationale: str
    conditions_entailment_severity: int
    actions_contradiction_rationale: str
    actions_contradiction_severity: int
    creation_utc: datetime


class CoherenceChecker:
    def __init__(
        self,
        logger: Logger,
        conditions_test_schematic_generator: SchematicGenerator[ConditionsEntailmentTestsSchema],
        actions_test_schematic_generator: SchematicGenerator[ActionsContradictionTestsSchema],
        entity_queries: EntityQueries,
    ) -> None:
        self._logger = logger
        self._conditions_entailment_checker = ConditionsEntailmentChecker(
            logger, conditions_test_schematic_generator, entity_queries
        )
        self._actions_contradiction_checker = ActionsContradictionChecker(
            logger, actions_test_schematic_generator, entity_queries
        )

    async def propose_incoherencies(
        self,
        agent: Agent,
        guidelines_to_evaluate: Sequence[GuidelineContent],
        comparison_guidelines: Sequence[GuidelineContent] = [],
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[IncoherenceTest]:
        comparison_guidelines_list = list(comparison_guidelines)
        guidelines_to_evaluate_list = list(guidelines_to_evaluate)
        tasks = []

        for i, guideline_to_evaluate in enumerate(guidelines_to_evaluate):
            filtered_existing_guidelines = [
                g for g in guidelines_to_evaluate_list[i + 1 :] + comparison_guidelines_list
            ]
            guideline_batches = list(chunked(filtered_existing_guidelines, EVALUATION_BATCH_SIZE))
            if progress_report:
                await progress_report.stretch(len(guideline_batches))

            tasks.extend(
                [
                    asyncio.create_task(
                        self._process_proposed_guideline(
                            agent, guideline_to_evaluate, batch, progress_report
                        )
                    )
                    for batch in guideline_batches
                ]
            )
        with self._logger.operation(
            f"Evaluating incoherencies for {len(tasks)} "
            f"batches (batch size={EVALUATION_BATCH_SIZE})",
        ):
            incoherencies = list(chain.from_iterable(await async_utils.safe_gather(*tasks)))

        return incoherencies

    async def _process_proposed_guideline(
        self,
        agent: Agent,
        guideline_to_evaluate: GuidelineContent,
        comparison_guidelines: Sequence[GuidelineContent],
        progress_report: Optional[ProgressReport],
    ) -> Sequence[IncoherenceTest]:
        indexed_comparison_guidelines = {i: c for i, c in enumerate(comparison_guidelines, start=1)}
        (
            conditions_entailment_responses,
            actions_contradiction_responses,
        ) = await async_utils.safe_gather(
            self._conditions_entailment_checker.evaluate(
                agent, guideline_to_evaluate, indexed_comparison_guidelines
            ),
            self._actions_contradiction_checker.evaluate(
                agent, guideline_to_evaluate, indexed_comparison_guidelines
            ),
        )

        incoherencies = []
        for id, g in indexed_comparison_guidelines.items():
            w = [w for w in conditions_entailment_responses if w.compared_guideline_id == id][0]
            t = [t for t in actions_contradiction_responses if t.compared_guideline_id == id][0]
            if t.severity >= ACTION_CONTRADICTION_SEVERITY_THRESHOLD:
                if w.compared_entails_origin_severity > w.origin_entails_compared_severity:
                    entailment_severity = w.compared_entails_origin_severity
                    entailment_rationale = w.compared_entails_origin_rationale
                else:
                    entailment_severity = w.origin_entails_compared_severity
                    entailment_rationale = w.origin_entails_compared_rationale
                incoherencies.append(
                    IncoherenceTest(
                        guideline_a=guideline_to_evaluate,
                        guideline_b=g,
                        IncoherenceKind=IncoherenceKind.STRICT
                        if entailment_severity >= CRITICAL_INCOHERENCE_THRESHOLD
                        else IncoherenceKind.CONTINGENT,
                        conditions_entailment_rationale=entailment_rationale,
                        conditions_entailment_severity=entailment_severity,
                        actions_contradiction_rationale=t.rationale,
                        actions_contradiction_severity=t.severity,
                        creation_utc=datetime.now(timezone.utc),
                    )
                )

        if progress_report:
            await progress_report.increment()

        return incoherencies


class ConditionsEntailmentChecker:
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[ConditionsEntailmentTestsSchema],
        entity_queries: EntityQueries,
    ) -> None:
        self._logger = logger
        self._schematic_generator = schematic_generator
        self._entity_queries = entity_queries

    async def evaluate(
        self,
        agent: Agent,
        guideline_to_evaluate: GuidelineContent,
        indexed_comparison_guidelines: dict[int, GuidelineContent],
    ) -> Sequence[ConditionsEntailmentTestSchema]:
        prompt = await self._build_prompt(
            agent, guideline_to_evaluate, indexed_comparison_guidelines
        )

        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": 0.0},
        )
        self._logger.debug(
            f"""
----------------------------------------
Condition Entailment Test Results:
----------------------------------------
{json.dumps([p.model_dump(mode="json") for p in response.content.condition_entailments], indent=2)}
----------------------------------------
"""
        )

        return response.content.condition_entailments

    async def _build_prompt(
        self,
        agent: Agent,
        guideline_to_evaluate: GuidelineContent,
        indexed_comparison_guidelines: dict[int, GuidelineContent],
    ) -> PromptBuilder:
        builder = PromptBuilder()
        comparison_candidates_text = "\n".join(
            f"""{{"id": {id}, "when": "{g.condition}", "then": "{g.action}"}}"""
            for id, g in indexed_comparison_guidelines.items()
        )
        guideline_to_evaluate_text = f"""{{"when": "{guideline_to_evaluate.condition}", "then": "{guideline_to_evaluate.action}"}}"""

        builder.add_section(
            name="conditions-entailment-checker-general-instructions",
            template="""
In our system, the behavior of a conversational AI agent is guided by "guidelines". The agent makes use of these guidelines whenever it interacts with a customer.

Each guideline is composed of two parts:
- "when": This is a natural-language condition that specifies when a guideline should apply.
          We look at each conversation at any particular state, and we test against this
          condition to understand if we should have this guideline participate in generating
          the next reply to the customer.
- "then": This is a natural-language instruction that should be followed by the agent
          whenever the "when" part of the guideline applies to the conversation in its particular state.
          Any instruction described here applies only to the agent, and not to the customer.


Your task is to evaluate whether pairs of guidelines have entailing 'when' statements.
{formatted_task_description}

To find whether two guidelines have entailing 'when's, independently determine whether the first 'when' entails the second, and vice-versa.
Be forgiving regarding misspellings and grammatical errors.

Please output JSON structured in the following format:
```json
{{
    "condition_entailments": [
        {{
            "compared_guideline_id": <id of the compared guideline>,
            "origin_guideline_when": <The origin guideline's 'when'>,
            "compared_guideline_when": <The compared guideline's 'when'>,
            "origin_entails_compared_rationale": <Explanation for if and how origin_guideline_when entails compared_guideline_when>,
            "origin_when_entails_compared_when": <BOOL of whether origin_guideline_when entails compared_guideline_when>,
            "origin_entails_compared_severity": <Score between 1-10 indicating the strength of the entailment from origin_guideline_when to compared_guideline_when>,
            "compared_entails_origin_rationale": <Explanation for if and how compared_guideline_when entails origin_guideline_when>,
            "compared_when_entails_origin_when": <BOOL of whether compared_guideline_when entails origin_guideline_when>,
            "compared_entails_origin_severity": <Score between 1-10 indicating the strength of the entailment from compared_guideline_when to origin_guideline_when>,

        }},
        ...
    ]
}}
```
The output json should have one such object for each pairing of the origin guideline with one of the compared guidelines.

The following are examples of expected outputs for a given input:
###
Example 1:
###
Input:

Test guideline: ###
{{"when": "a customer orders an electrical appliance", "then": "ship the item immediately"}}
###

Comparison candidates: ###
{{"id": 1, "when": "a customer orders a TV", "then": "wait for the manager's approval before shipping"}}
{{"id": 2, "when": "a customer orders any item", "then": "refer the customer to our electronic store"}}
{{"id": 3, "when": "a customer orders a chair", "then": "reply that the product can only be delivered in-store"}}
{{"id": 4, "when": "a customer asks which discounts we offer on electrical appliances", "then": "reply that we offer free shipping for items over 100$"}}
{{"id": 5, "when": "a customer greets you", "then": "greet them back"}}

###

Expected Output:
```json
{{
    "condition_entailments": [
        {{
            "compared_guideline_id": 1,
            "origin_guideline_when": "a customer orders an electrical appliance",
            "compared_guideline_when": "a customer orders a TV",
            "origin_entails_compared_rationale": "A customer ordering an electrical appliance doesn't necessarily mean they are ordering a TV specifically",
            "origin_when_entails_compared_when": false,
            "origin_entails_compared_severity": 3,
            "compared_entails_origin_rationale": "since TVs are electronic appliances, ordering a TV entails ordering an electrical appliance",
            "compared_when_entails_origin_when": true,
            "compared_entails_origin_severity": 9
        }},
        {{
            "compared_guideline_id": 2,
            "origin_guideline_when": "a customer orders an electrical appliance",
            "compared_guideline_when": "a customer orders any item",
            "origin_entails_compared_rationale": "electrical appliances are items, so ordering an electrical appliance entails ordering an item",
            "origin_when_entails_compared_when": true,
            "origin_entails_compared_severity": 10,
            "compared_entails_origin_rationale": "ordering an electrical appliance doesn't entail ordering any item, since the customer might be ordering a non-electronic item",
            "compared_when_entails_origin_when": false,
            "compared_entails_origin_severity": 2
        }},
        {{
            "compared_guideline_id": 3,
            "origin_guideline_when": "a customer orders an electrical appliance",
            "compared_guideline_when": "a customer orders a chair",
            "origin_entails_compared_rationale": "ordering an electrical appliance doesn't entail ordering a chair, since they are two distinct categories of items",
            "origin_when_entails_compared_when": false,
            "origin_entails_compared_severity": 1,
            "compared_entails_origin_rationale": "chairs are not electrical appliances, so ordering a chair does not entail ordering an electrical appliance",
            "compared_when_entails_origin_when": false,
            "compared_entails_origin_severity": 2
        }},
        {{
            "compared_guideline_id": 4,
            "origin_guideline_when": "a customer orders an electrical appliance",
            "compared_guideline_when": "a customer asks which discounts we offer on electrical appliances",
            "origin_entails_compared_rationale": "an electrical appliance can be ordered without asking for a discount, so it doesn't entail it",
            "origin_when_entails_compared_when": false,
            "origin_entails_compared_severity": 3,
            "compared_entails_origin_rationale": "asking for a discount does not entail the ordering of an electrical appliance, as the discount may apply to another type of item",
            "compared_when_entails_origin_when": false,
            "compared_entails_origin_severity": 2
        }},
        {{
            "compared_guideline_id": 5,
            "origin_guideline_when": "a customer orders an electrical appliance",
            "compared_guideline_when": "a customer greets you",
            "origin_entails_compared_rationale": "ordering an electrical appliance does not entail or mean that the use has greeted the assistant",
            "origin_when_entails_compared_when": false,
            "origin_entails_compared_severity": 1,
            "compared_entails_origin_rationale": "a customer greeting the assistant in no way entails them ordering an electrical appliance",
            "compared_when_entails_origin_when": false,
            "compared_entails_origin_severity": 1
        }}
    ]
}}
```

###
Example 2:
###
Input:

Test guideline: ###
{{"when": "offering products to the customer", "then": "mention the price of the suggested product"}}
###

Comparison candidates: ###
{{"id": 1, "when": "suggesting a TV", "then": "mention the size of the screen"}}
{{"id": 2, "when": "the customer asks for recommendations", "then": "recommend items from the sales department"}}
{{"id": 3, "when": "recommending a TV warranty plan", "then": "encourage the use to get an upgraded warranty"}}
{{"id": 4, "when": "discussing store items", "then": "check the stock for their availability"}}

###

Expected Output:
```json
{{
    "condition_entailments": [
        {{
            "compared_guideline_id": 1,
            "origin_guideline_when": "offering products to the customer",
            "compared_guideline_when": "suggesting a TV",
            "origin_entails_compared_rationale": "offering products does not entail suggesting a tv, as another type of product could be offered",
            "origin_when_entails_compared_when": false,
            "origin_entails_compared_severity": 3,
            "compared_entails_origin_rationale": "by suggesting a TV, a product is offered to the customer",
            "compared_when_entails_origin_when": true,
            "compared_entails_origin_severity": 9
        }},
        {{
            "compared_guideline_id": 2,
            "origin_guideline_when": "offering products to the customer",
            "compared_guideline_when": "the customer asks for recommendations",
            "origin_entails_compared_rationale": "offering products to the customer does not entail them asking for recommendations, since the agent might be offering items for a different reason",
            "origin_when_entails_compared_when": false,
            "origin_entails_compared_severity": 4,
            "compared_entails_origin_rationale": "the customer asking for recommendations does not entail that a product is offered to them. They could be asking out of their own accord",
            "compared_when_entails_origin_when": false,
            "compared_entails_origin_severity": 3
        }},
        {{
            "compared_guideline_id": 3,
            "origin_guideline_when": "offering products to the customer",
            "compared_guideline_when": "recommending a TV warranty plan",
            "origin_entails_compared_rationale": "offering product does not entail recommending a TV warranty, as the product might not be a TV warranty",
            "origin_when_entails_compared_when": false,
            "origin_entails_compared_severity": 3,
            "compared_entails_origin_rationale": "when a TV warranty plan is recommended, a product (the warranty) is offered to the customer, so recommending a TV warranty plan entails offering a product",
            "compared_when_entails_origin_when": true,
            "compared_entails_origin_severity": 8
        }},
        {{
            "compared_guideline_id": 4,
            "origin_guideline_when": "offering products to the customer",
            "compared_guideline_when": "discussing store items",
            "origin_entails_compared_rationale": "discussing store items does not entail offering products, since a different kind of discussion might be occurring",
            "origin_when_entails_compared_when": false,
            "origin_entails_compared_severity": 3,
            "compared_entails_origin_rationale": "offering a product to the customer entails the discussion of a store item, as it's fair to assume that product is a store item",
            "compared_when_entails_origin_when": true,
            "compared_entails_origin_severity": 7
        }}
    ]
}}
```
###
""",
            props={"formatted_task_description": self.get_task_description()},
        )

        builder.add_agent_identity(agent)
        terms = await self._entity_queries.find_glossary_terms_for_context(
            agent_id=agent.id,
            query=guideline_to_evaluate_text + comparison_candidates_text,
        )
        builder.add_glossary(terms)

        builder.add_section(
            name="conditions-entailment-checker-guidelines-to-analyze",
            template="""
The guidelines you should analyze for entailments are:
Origin guideline: ###
{guideline_to_evaluate_text}
###

Comparison candidates: ###
{comparison_candidates_text}
###""",
            props={
                "guideline_to_evaluate_text": guideline_to_evaluate_text,
                "comparison_candidates_text": comparison_candidates_text,
                "guideline_to_evaluate": guideline_to_evaluate,
                "comparison_candidates": indexed_comparison_guidelines,
            },
        )

        return builder

    @staticmethod
    def get_task_description() -> str:
        return """
Two guidelines should be detected as having entailing 'when' statements if and only if one of their 'when' statements being true entails that the other's 'when' statement is also true.
By this, if there is any context in which the 'when' statement of guideline A is false while the 'when' statement of guideline B is true - guideline B can not entail guideline A.
If one 'when' statement being true implies that the other 'when' statement was perhaps true in a past state of the conversation, but strict entailment is not fulfilled - do not consider the 'when' statements as entailing. If one 'when' statement holding true typically means that another 'when' is true, it is not sufficient to be considered entailment."""


class ActionsContradictionChecker:
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[ActionsContradictionTestsSchema],
        entity_queries: EntityQueries,
    ) -> None:
        self._logger = logger
        self._schematic_generator = schematic_generator
        self._entity_queries = entity_queries

    async def evaluate(
        self,
        agent: Agent,
        guideline_to_evaluate: GuidelineContent,
        indexed_comparison_guidelines: dict[int, GuidelineContent],
    ) -> Sequence[ActionsContradictionTestSchema]:
        prompt = await self._build_prompt(
            agent, guideline_to_evaluate, indexed_comparison_guidelines
        )
        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": 0.0},
        )
        self._logger.debug(
            f"""
----------------------------------------
Action Contradiction Test Results:
----------------------------------------
{json.dumps([p.model_dump(mode="json") for p in response.content.action_contradictions], indent=2)}
----------------------------------------
"""
        )

        return response.content.action_contradictions

    async def _build_prompt(
        self,
        agent: Agent,
        guideline_to_evaluate: GuidelineContent,
        indexed_comparison_guidelines: dict[int, GuidelineContent],
    ) -> PromptBuilder:
        builder = PromptBuilder()
        comparison_candidates_text = "\n".join(
            f"""{{"id": {id}, "when": "{g.condition}", "then": "{g.action}"}}"""
            for id, g in indexed_comparison_guidelines.items()
        )
        guideline_to_evaluate_text = f"""{{"when": "{guideline_to_evaluate.condition}", "then": "{guideline_to_evaluate.action}"}}"""

        builder.add_section(
            name="actions-contradiction-checker-general-instructions",
            template="""
In our system, the behavior of a conversational AI agent is guided by "guidelines". The agent makes use of these guidelines whenever it interacts with a customer.

Each guideline is composed of two parts:
- "when": This is a natural-language condition that specifies when a guideline should apply.
          We look at each conversation at any particular state, and we test against this
          condition to understand if we should have this guideline participate in generating
          the next reply to the customer.
- "then": This is a natural-language instruction that should be followed by the agent
          whenever the "when" part of the guideline applies to the conversation in its particular state.
          Any instruction described here applies only to the agent, and not to the customer.

To ensure consistency, it is crucial to avoid scenarios where multiple guidelines with conflicting 'then' statements are applied.
{formatted_task_description}


Be forgiving regarding misspellings and grammatical errors.



Please output JSON structured in the following format:
```json
{{
    "action_contradictions": [
        {{
            "compared_guideline_id": <id of the compared guideline>,
            "origin_guideline_then": <The origin guideline's 'then'>,
            "compared_guideline_then": <The compared guideline's 'then'>,
            "rationale": <Explanation for if and how the 'then' statements contradict each other>,
            "thens_contradiction": <BOOL of whether the two 'then' statements are contradictory>,
            "severity": <Score between 1-10 indicating the strength of the contradiction>
        }},
        ...
    ]
}}
```
The output json should have one such object for each pairing of the origin guideline with one of the compared guidelines.

The following are examples of expected outputs for a given input:
###
Example 1:
###
Input:

Test guideline: ###
{{"when": "a customer orders an electrical appliance", "then": "ship the item immediately"}}
###

Comparison candidates: ###
{{"id": 1, "when": "a customer orders a TV", "then": "wait for the manager's approval before shipping"}}
{{"id": 2, "when": "a customer orders any item", "then": "refer the customer to our electronic store"}}
{{"id": 3, "when": "a customer orders a chair", "then": "reply that the product can only be delivered in-store"}}
{{"id": 4, "when": "a customer asks which discounts we offer on electrical appliances", "then": "reply that we offer free shipping for items over 100$"}}
{{"id": 5, "when": "a customer greets you", "then": "greet them back"}}

###

Expected Output:
```json
{{
    "action_contradictions": [
        {{
            "compared_guideline_id": 1,
            "origin_guideline_then": "ship the item immediately",
            "compared_guideline_then": "wait for the manager's approval before shipping",
            "rationale": "shipping the item immediately contradicts waiting for the manager's approval",
            "thens_contradiction": true,
            "severity": 10
        }},
        {{
            "compared_guideline_id": 2,
            "origin_guideline_then": "ship the item immediately",
            "compared_guideline_then": "refer the customer to our electronic store",
            "rationale": "the agent can both ship the item immediately and refer the customer to the electronic store at the same time, the actions are not contradictory",
            "thens_contradiction": false,
            "severity": 2
        }},
        {{
            "compared_guideline_id": 3,
            "origin_guideline_then": "ship the item immediately",
            "compared_guideline_then": "reply that the product can only be delivered in-store",
            "rationale": "shipping the item immediately contradicts the reply that the product can only be delivered in-store",
            "thens_contradiction": true,
            "severity": 9
        }},
        {{
            "compared_guideline_id": 4,
            "origin_guideline_then": "ship the item immediately",
            "compared_guideline_then": "reply that we offer free shipping for items over 100$",
            "rationale": "replying that we offer free shipping for expensive items does not contradict shipping an item immediately, both actions can be taken simultaneously",
            "thens_contradiction": false,
            "severity": 1
        }},
        {{
            "compared_guideline_id": 5,
            "origin_guideline_then": "ship the item immediately",
            "compared_guideline_then": "greet them back",
            "rationale": "shipping the item immediately can be done while also greeting the customer, both actions can be taken simultaneously",
            "thens_contradiction": false,
            "severity": 1
        }}
    ]
}}
```

###
Example 2:
###
Input:

Test guideline: ###
{{"when": "the customer mentions health issues", "then": "register them to the 5km race"}}
###

Comparison candidates: ###
{{"id": 1, "when": "the customer asks about registering available races", "then": "Reply that you can register them either to the 5km or the 10km race"}}
{{"id": 2, "when": "the customer wishes to register to a race without being verified", "then": "Inform them that they cannot register to races without verification"}}
{{"id": 3, "when": "the customer wants to register races over 10km", "then": "suggest either a half or a full marathon"}}
{{"id": 4, "when": "the customer wants to register to the 10km race", "then": "register them as long as there are available slots"}}
###

Expected Output:
```json
{{
    "action_contradictions": [
        {{
            "compared_guideline_id": 1,
            "origin_guideline_then": "register them to the 5km race",
            "compared_guideline_then": "Reply that you can register them either to the 5km or the 10km race",
            "rationale": "allowing the customer to select from the multiple options for races, while already registering them to the 5km race is contradictory, as it ascribes an action that doesn't align with the agent's response",
            "thens_contradiction": true,
            "severity": 7
        }},
        {{
            "compared_guideline_id": 2,
            "origin_guideline_then": "register them to the 5km race",
            "compared_guideline_then": "Inform them that they cannot register to races without verification",
            "rationale": "Informing the customer that they cannot register to races while registering them to a race is contradictory - the action does not align with the agent's response",
            "thens_contradiction": true,
            "severity": 8
        }},
        {{
            "compared_guideline_id": 3,
            "origin_guideline_then": "register them to the 5km race",
            "compared_guideline_then": "suggest either a half or a full marathon",
            "rationale": "Suggesting a half or a full marathon after the customer asked about over 10km runs, while also registering them to the 5km run, is contradictory.",
            "thens_contradiction": true,
            "severity": 7
        }},

        {{
            "compared_guideline_id": 4,
            "origin_guideline_then": "register them to the 5km race",
            "compared_guideline_then": "register them as long as there are available slots",
            "rationale": "the guidelines dictate registering the customer to two separate races. While this is not inherently contradictory, it can lead to confusing or undefined behavior",
            "thens_contradiction": true,
            "severity": 8
        }}
    ]
}}
```

###""",
            props={"formatted_task_description": self.get_task_description()},
        )
        builder.add_agent_identity(agent)
        terms = await self._entity_queries.find_glossary_terms_for_context(
            agent_id=agent.id,
            query=guideline_to_evaluate_text + comparison_candidates_text,
        )
        builder.add_glossary(terms)

        builder.add_section(
            name="actions-contradiction-checker-guidelines-to-analyze",
            template="""
The guidelines you should analyze for entailments are:
Origin guideline: ###
{guideline_to_evaluate_text}
###

Comparison candidates: ###
{comparison_candidates_text}
###""",
            props={
                "guideline_to_evaluate_text": guideline_to_evaluate_text,
                "comparison_candidates_text": comparison_candidates_text,
                "guideline_to_evaluate": guideline_to_evaluate,
                "comparison_candidates": indexed_comparison_guidelines,
            },
        )
        return builder

    @staticmethod
    def get_task_description() -> str:
        return """
Two 'then' statements are considered contradictory if:

1. Applying both results in an actions which cannot be applied together trivially. This could either describe directly contradictory actions, or actions that interact in an unexpected way.
2. Applying both leads to a confusing or paradoxical response.
3. Applying both would result in the agent taking an action that does not align with the response it should provide to the customer.
While your evaluation should focus on the 'then' statements, remember that each 'then' statement is contextualized by its corresponding 'when' statement. Analyze each 'then' statement within the context provided by its "when" condition. Please be lenient with any misspellings or grammatical errors.
"""
