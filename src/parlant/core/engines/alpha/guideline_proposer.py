import asyncio
from dataclasses import dataclass
from functools import cached_property
from itertools import chain, groupby
import json
import math
import time
from typing import Sequence

from parlant.core.agents import Agent
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.end_users import EndUser, EndUserTag
from parlant.core.nlp.generation import GenerationInfo, SchematicGenerator
from parlant.core.engines.alpha.guideline_proposition import GuidelineProposition
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.glossary import Term
from parlant.core.guidelines import Guideline
from parlant.core.sessions import Event
from parlant.core.emissions import EmittedEvent
from parlant.core.common import DefaultBaseModel
from parlant.core.logging import Logger


class GuidelinePropositionSchema(DefaultBaseModel):
    condition_number: int
    condition: str
    you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction: bool
    is_this_condition_hard_or_tricky_to_confidently_ascertain: bool
    rationale: str
    applies_score: int


class GuidelinePropositionsSchema(DefaultBaseModel):
    checks: Sequence[GuidelinePropositionSchema]


@dataclass(frozen=True)
class ConditionApplicabilityEvaluation:
    condition: str
    score: int
    rationale: str


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
        user_tags_pair: tuple[EndUser, Sequence[EndUserTag]],
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

        t_start = time.time()

        guidelines_grouped_by_condition = {
            condition: list(guidelines)
            for condition, guidelines in groupby(
                sorted(guidelines, key=lambda g: g.content.condition),
                key=lambda g: g.content.condition,
            )
        }

        unique_conditions = list(guidelines_grouped_by_condition.keys())

        batches = self._create_condition_batches(
            unique_conditions,
            batch_size=self._get_optimal_batch_size(unique_conditions),
        )

        with self._logger.operation(
            f"Guideline proposal ({len(guidelines)} guidelines processed in {len(batches)} batches)"
        ):
            batch_tasks = [
                self._process_condition_batch(
                    agents,
                    user_tags_pair,
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
                    guideline_propositions += [
                        GuidelineProposition(
                            guideline=g, score=evaluation.score, rationale=evaluation.rationale
                        )
                        for g in guidelines_grouped_by_condition[evaluation.condition]
                    ]
                propositions_batches.append(guideline_propositions)

            t_end = time.time()

            return GuidelinePropositionResult(
                total_duration=t_end - t_start,
                batch_count=len(batches),
                batch_generations=batch_generations,
                batches=propositions_batches,
            )

    def _get_optimal_batch_size(self, conditions: list[str]) -> int:
        condition_count = len(conditions)

        if condition_count <= 10:
            return 1
        elif condition_count <= 20:
            return 2
        elif condition_count <= 30:
            return 3
        else:
            return 5

    def _create_condition_batches(
        self,
        conditions: Sequence[str],
        batch_size: int,
    ) -> Sequence[Sequence[str]]:
        batches = []
        batch_count = math.ceil(len(conditions) / batch_size)

        for batch_number in range(batch_count):
            start_offset = batch_number * batch_size
            end_offset = start_offset + batch_size
            batch = conditions[start_offset:end_offset]
            batches.append(batch)

        return batches

    async def _process_condition_batch(
        self,
        agents: Sequence[Agent],
        user_tags_pair: tuple[EndUser, Sequence[EndUserTag]],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        staged_events: Sequence[EmittedEvent],
        terms: Sequence[Term],
        batch: Sequence[str],
    ) -> tuple[GenerationInfo, list[ConditionApplicabilityEvaluation]]:
        prompt = self._format_prompt(
            agents,
            user_tags_pair,
            context_variables=context_variables,
            interaction_history=interaction_history,
            staged_events=staged_events,
            terms=terms,
            conditions=batch,
        )

        with self._logger.operation(f"Condition evaluation batch ({len(batch)} conditions)"):
            propositions_generation_response = await self._schematic_generator.generate(
                prompt=prompt,
                hints={"temperature": 0.3},
            )

        propositions = []

        for proposition in propositions_generation_response.content.checks:
            condition = batch[int(proposition.condition_number) - 1]

            self._logger.debug(
                f'Guideline condition evaluation for "{condition}":\n'  # noqa
                f'  Score: {proposition.applies_score}/10; Certain: {not proposition.is_this_condition_hard_or_tricky_to_confidently_ascertain}; Rationale: "{proposition.rationale}"'
            )

            if (proposition.applies_score >= 7) or (
                proposition.applies_score >= 5
                and proposition.is_this_condition_hard_or_tricky_to_confidently_ascertain
            ):
                propositions.append(
                    ConditionApplicabilityEvaluation(
                        condition=batch[int(proposition.condition_number) - 1],
                        score=proposition.applies_score,
                        rationale=proposition.rationale,
                    )
                )

        return propositions_generation_response.info, propositions

    def _format_prompt(
        self,
        agents: Sequence[Agent],
        user_tags_pair: tuple[EndUser, Sequence[EndUserTag]],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        staged_events: Sequence[EmittedEvent],
        terms: Sequence[Term],
        conditions: Sequence[str],
    ) -> str:
        assert len(agents) == 1

        result_structure = [
            {
                "condition_number": i,
                "condition": condition,
                "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": "<BOOL>",
                "is_this_condition_hard_or_tricky_to_confidently_ascertain": "<BOOL>",
                "rationale": "<EXPLANATION WHY THE CONDITION IS RELEVANT OR IRRELEVANT FOR THE "
                "CURRENT STATE OF THE INTERACTION>",
                "applies_score": "<RELEVANCE SCORE>",
            }
            for i, condition in enumerate(conditions, start=1)
        ]

        builder = PromptBuilder()

        builder.add_section(
            """
Task Description
----------------
Your job is to assess the relevance and/or applicability of a few provided conditions
to the last known state of an interaction between yourself, AI assistant, and a user.
The conditions and the interaction will be provided to you later in this message.
"""
        )
        builder.add_section(
            f"""
Process Description
-------------------
a. Examine the provided interaction events to discern the latest state of interaction between the user and the assistant.
b. Evaluate the entire interaction to determine if each condition is still relevant to the most recent interaction state.
c. If the condition has already been addressed, assess its continued applicability.
d. Assign an applicability score to each condition between 1 and 10.
e. IMPORTANT: Note that some conditions are harder to ascertain objectively, especially if they correspond to things relating to emotions or inner thoughts of people. Do not presume to know them for sure, and in such cases prefer to say that you cannot safely presume to ascertain whether they still apply—again, because emotionally-based conditions are hard to ascertain through a textual conversation.

### Examples of Condition Evaluations:

#### Example #1:
- Interaction Events: ###
[{{"id": "11", "kind": "<message>", "source": "user",
"data": {{"message": "Can I purchase a subscription to your software?"}}}},
{{"id": "23", "kind": "<message>", "source": "assistant",
"data": {{"message": "Absolutely, I can assist you with that right now."}}}},
{{"id": "34", "kind": "<message>", "source": "user",
"data": {{"message": "Please proceed with the subscription for the Pro plan."}}}},
{{"id": "56", "kind": "<message>", "source": "assistant",
"data": {{"message": "Your subscription has been successfully activated.
Is there anything else I can help you with?"}}}},
{{"id": "78", "kind": "<message>", "source": "user",
"data": {{"message": "Yes, can you tell me more about your data security policies?"}}}}]
###
- Conditions: ###
1) the client initiates a purchase
2) the client asks about data security
###
- **Expected Result**:
```json
{{ "checks": [
    {{
        "condition_number": "1",
        "condition": "the client initiates a purchase",
        "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": true,
        "is_this_condition_hard_or_tricky_to_confidently_ascertain": true,
        "rationale": "The purchase-related guideline is irrelevant since the client completed the purchase and the conversation has moved to a new topic.",
        "applies_score": 3
    }},
    {{
        "condition_number": "2",
        "condition": "the client asks about data security",
        "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": false,
        "is_this_condition_hard_or_tricky_to_confidently_ascertain": true,
        "rationale": "The client specifically inquired about data security policies, making this guideline highly relevant to the ongoing discussion.",
        "applies_score": 9
    }}
]}}
```

#### Example #2:
[{{"id": "112", "kind": "<message>", "source": "user",
"data": {{"message": "I need to make this quick.
Can you give me a brief overview of your pricing plans?"}}}},
{{"id": "223", "kind": "<message>", "source": "assistant",
"data": {{"message": "Absolutely, I'll keep it concise. We have three main plans: Basic,
Advanced, and Pro. Each offers different features, which I can summarize quickly for you."}}}},
{{"id": "334", "kind": "<message>", "source": "user",
"data": {{"message": "Tell me about the Pro plan."}}}},
###
- Conditions: ###
1) the client indicates they are in a hurry
2) a client inquires about pricing plans
3) a client asks for a summary of the features of the three plans.
###
- **Expected Result**:
```json
{{
    "checks": [
        {{
            "condition_number": "1",
            "condition": "the client indicates they are in a hurry",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": false,
            "is_this_condition_hard_or_tricky_to_confidently_ascertain": true,
            "rationale": "The client initially stated they were in a hurry. This urgency applies throughout the conversation unless stated otherwise.",
            "applies_score": 8
        }},
        {{
            "condition_number": "2",
            "condition": "a client inquires about pricing plans",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": false,
            "is_this_condition_hard_or_tricky_to_confidently_ascertain": true,
            "rationale": "The client inquired about pricing plans, specifically asking for details about the Pro plan.",
            "applies_score": 9
        }},
        {{
            "condition_number": "3",
            "condition": "a client asks for a summary of the features of the three plans.",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": false,
            "rationale": "The plan summarization guideline is irrelevant since the client only asked about the Pro plan.",
            "applies_score": 2
        }},
    ]
}}
```
### Example #3:
- Interaction Events: ###
[{{"id": "13", "kind": "<message>", "source": "user",
"data": {{"message": "Can you recommend a good science fiction movie?"}}}},
{{"id": "14", "kind": "<message>", "source": "assistant",
"data": {{"message": "Sure, I recommend 'Inception'. It's a great science fiction movie."}}}},
{{"id": "15", "kind": "<message>", "source": "user",
"data": {{"message": "Thanks, I'll check it out."}}}}]
###
- Conditions: ###
1) the client asks for a recommendation
2) the client asks about movie genres
###
- **Expected Result**:
```json
{{
    "checks": [
        {{
            "condition_number": "1",
            "condition": "the client asks for a recommendation",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": false,
            "is_this_condition_hard_or_tricky_to_confidently_ascertain": true,
            "rationale": "The client asked for a science fiction movie recommendation and the assistant provided one, making this guideline highly relevant.",
            "applies_score": 9
        }},
        {{
            "condition_number": "2",
            "condition": "the client asks about movie genres",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": true,
            "is_this_condition_hard_or_tricky_to_confidently_ascertain": true,
            "rationale": "The client asked about science fiction movies, but this was already addressed by the assistant.",
            "applies_score": 3
        }}
    ]
}}
```

### Example #4:
- Interaction Events: ###
[{{"id": "54", "kind": "<message>", "source": "user",
"data": {{"message": "Can I add an extra pillow to my bed order?"}}}},
{{"id": "66", "kind": "<message>", "source": "assistant",
"data": {{"message": "An extra pillow has been added to your order."}}}},
{{"id": "72", "kind": "<message>", "source": "user",
"data": {{"message": "Thanks, I'll come to pick up the order. Can you tell me the address?"}}}}]
###
- Conditions: ###
1) the client requests a modification to their order
2) the client asks for the store's location
###
- **Expected Result**:
```json
{{
    "checks": [
        {{
            "condition_number": "1",
            "condition": "the client requests a modification to their order",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": true,
            "is_this_condition_hard_or_tricky_to_confidently_ascertain": true,
            "rationale": "The client requested a modification (an extra pillow) and the assistant confirmed it, making this guideline irrelevant now as it has already been addressed.",
            "applies_score": 3
        }},
        {{
            "condition_number": "2",
            "condition": "the client asks for the store's location",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": false,
            "is_this_condition_hard_or_tricky_to_confidently_ascertain": true,
            "rationale": "The client asked for the store's location, making this guideline highly relevant.",
            "applies_score": 10
        }}
    ]
}}
```

### Example #5:
- Interaction Events: ###
[{{"id": "21", "kind": "<message>", "source": "user",
"data": {{"message": "Can I add an extra charger to my laptop order?"}}}},
{{"id": "34", "kind": "<message>", "source": "assistant",
"data": {{"message": "An extra charger has been added to your order."}}}},
{{"id": "53", "kind": "<message>", "source": "user",
"data": {{"message": "Do you have any external hard drives available?"}}}}]
###
- Conditions: ###
1) the order does not exceed the limit of products
2) the client asks about product availability
###
- **Expected Result**:
```json
{{
    "checks": [
        {{
            "condition_number": "1",
            "condition": "the order does not exceed the limit of products",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": false,
            "rationale": "The client added an extra charger, and the order did not exceed the limit of products, making this guideline relevant.",
            "applies_score": 9
        }},
        {{
            "condition_number": "2",
            "condition": "the client asks about product availability",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": false,
            "rationale": "The client asked about the availability of external hard drives, making this guideline highly relevant as it informs the user if they reach the product limit before adding another item to the cart.",
            "applies_score": 10
        }}
    ]
}}
```

### Example #6:
- Interaction Events: ###
[{{"id": "54", "kind": "<message>", "source": "user",
"data": {{"message": "I disagree with you about this point."}}}},
{{"id": "66", "kind": "<message>", "source": "assistant",
"data": {{"message": "But I fully disproved your thesis!"}}}},
{{"id": "72", "kind": "<message>", "source": "user",
"data": {{"message": "Okay, fine."}}}}]
###
- Conditions: ###
1) the user is currently eating lunch
2) the user agrees with you in the scope of an argument
###
- **Expected Result**:
```json
{{
    "checks": [
        {{
            "condition_number": "1",
            "condition": "the user is currently eating lunch",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": false,
            "is_this_condition_hard_or_tricky_to_confidently_ascertain": false,
            "rationale": "There's nothing to indicate that the user is eating, lunch or otherwise",
            "applies_score": 1
        }},
        {{
            "condition_number": "2",
            "condition": "the user agrees with you in the scope of an argument",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": true,
            "is_this_condition_hard_or_tricky_to_confidently_ascertain": false,
            "rationale": "The user said 'Okay, fine', but it's possible that they are still in disagreement internally",
            "applies_score": 4
        }}
    ]
}}
```
"""  # noqa
        )
        builder.add_agent_identity(agents[0])
        builder.add_interaction_history(interaction_history)
        builder.add_user_name_and_tags(*user_tags_pair)
        builder.add_section(
            f"""
The following is an additional list of staged events that were just added: ###
{staged_events}
###
"""
        )

        builder.add_context_variables(context_variables)
        builder.add_glossary(terms)

        builder.add_guideline_conditions(conditions)
        builder.add_section(f"""
IMPORTANT: Please note there are exactly {len(conditions)} conditions in the list for you to check.

Expected Output
---------------------------
- Specify the applicability of each condition by filling in the rationale and applied score in the following list:

    ```json
    {{
        "checks":
        {json.dumps(result_structure)}
    }}
    ```""")

        prompt = builder.build()

        return prompt
