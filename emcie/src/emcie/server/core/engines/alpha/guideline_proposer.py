import asyncio
from dataclasses import dataclass
from itertools import groupby
import json
import math
from typing import Sequence

from emcie.server.core.agents import Agent
from emcie.server.core.context_variables import ContextVariable, ContextVariableValue
from emcie.server.core.nlp.generation import SchematicGenerator
from emcie.server.core.engines.alpha.guideline_proposition import GuidelineProposition
from emcie.server.core.engines.alpha.prompt_builder import PromptBuilder
from emcie.server.core.glossary import Term
from emcie.server.core.guidelines import Guideline
from emcie.server.core.sessions import Event
from emcie.server.core.emissions import EmittedEvent
from emcie.server.core.common import DefaultBaseModel
from emcie.server.core.logging import Logger


class GuidelinePropositionSchema(DefaultBaseModel):
    predicate_number: int
    predicate: str
    you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction: bool
    is_this_predicate_hard_or_tricky_to_confidently_ascertain: bool
    rationale: str
    applies_score: int


class GuidelinePropositionsSchema(DefaultBaseModel):
    checks: Sequence[GuidelinePropositionSchema]


@dataclass(frozen=True)
class PredicateApplicabilityEvaluation:
    predicate: str
    score: int
    rationale: str


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
        guidelines: Sequence[Guideline],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        staged_events: Sequence[EmittedEvent],
    ) -> Sequence[GuidelineProposition]:
        if not guidelines:
            return []

        guidelines_grouped_by_predicate = {
            predicate: list(guidelines)
            for predicate, guidelines in groupby(
                sorted(guidelines, key=lambda g: g.content.predicate),
                key=lambda g: g.content.predicate,
            )
        }

        unique_predicates = list(guidelines_grouped_by_predicate.keys())

        batches = self._create_predicate_batches(
            unique_predicates,
            batch_size=self._get_optimal_batch_size(unique_predicates),
        )

        with self._logger.operation(
            f"Guideline proposal ({len(guidelines)} guidelines processed in {len(batches)} batches)"
        ):
            batch_tasks = [
                self._process_predicate_batch(
                    agents,
                    context_variables,
                    interaction_history,
                    staged_events,
                    terms,
                    batch,
                )
                for batch in batches
            ]

            predicate_evaluations = sum(await asyncio.gather(*batch_tasks), [])

            guideline_propositions = []

            for evaluation in predicate_evaluations:
                guideline_propositions += [
                    GuidelineProposition(
                        guideline=g, score=evaluation.score, rationale=evaluation.rationale
                    )
                    for g in guidelines_grouped_by_predicate[evaluation.predicate]
                ]

            return guideline_propositions

    def _get_optimal_batch_size(self, predicates: list[str]) -> int:
        predicate_count = len(predicates)

        if predicate_count <= 10:
            return 1
        elif predicate_count <= 20:
            return 2
        elif predicate_count <= 30:
            return 3
        else:
            return 5

    def _create_predicate_batches(
        self,
        predicates: Sequence[str],
        batch_size: int,
    ) -> Sequence[Sequence[str]]:
        batches = []
        batch_count = math.ceil(len(predicates) / batch_size)

        for batch_number in range(batch_count):
            start_offset = batch_number * batch_size
            end_offset = start_offset + batch_size
            batch = predicates[start_offset:end_offset]
            batches.append(batch)

        return batches

    async def _process_predicate_batch(
        self,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        staged_events: Sequence[EmittedEvent],
        terms: Sequence[Term],
        batch: Sequence[str],
    ) -> list[PredicateApplicabilityEvaluation]:
        prompt = self._format_prompt(
            agents,
            context_variables=context_variables,
            interaction_history=interaction_history,
            staged_events=staged_events,
            terms=terms,
            predicates=batch,
        )

        with self._logger.operation(f"Predicate evaluation batch ({len(batch)} predicates)"):
            propositions_json = await self._schematic_generator.generate(
                prompt=prompt,
                hints={"temperature": 0.3},
            )

        propositions = []

        for proposition in propositions_json.content.checks:
            predicate = batch[int(proposition.predicate_number) - 1]

            self._logger.debug(
                f'Guideline predicate evaluation for "{predicate}":\n'  # noqa
                f'  Score: {proposition.applies_score}/10; Certain: {not proposition.is_this_predicate_hard_or_tricky_to_confidently_ascertain}; Rationale: "{proposition.rationale}"'
            )

            if (proposition.applies_score >= 7) or (
                proposition.applies_score >= 5
                and proposition.is_this_predicate_hard_or_tricky_to_confidently_ascertain
            ):
                propositions.append(
                    PredicateApplicabilityEvaluation(
                        predicate=batch[int(proposition.predicate_number) - 1],
                        score=proposition.applies_score,
                        rationale=proposition.rationale,
                    )
                )

        return propositions

    def _format_prompt(
        self,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        staged_events: Sequence[EmittedEvent],
        terms: Sequence[Term],
        predicates: Sequence[str],
    ) -> str:
        assert len(agents) == 1

        result_structure = [
            {
                "predicate_number": i,
                "predicate": predicate,
                "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": "<BOOL>",
                "is_this_predicate_hard_or_tricky_to_confidently_ascertain": "<BOOL>",
                "rationale": "<EXPLANATION WHY THE PREDICATE IS RELEVANT OR IRRELEVANT FOR THE "
                "CURRENT STATE OF THE INTERACTION>",
                "applies_score": "<RELEVANCE SCORE>",
            }
            for i, predicate in enumerate(predicates, start=1)
        ]

        builder = PromptBuilder()

        builder.add_section(
            """
Task Description
----------------
Your job is to assess the relevance and/or applicability of a few provided predicates
to the last known state of an interaction between yourself, AI assistant, and a user.
The predicates and the interaction will be provided to you later in this message.
"""
        )
        builder.add_section(
            f"""
Process Description
-------------------
a. Examine the provided interaction events to discern the latest state of interaction between the user and the assistant.
b. Evaluate the entire interaction to determine if each predicate is still relevant to the most recent interaction state.
c. If the predicate has already been addressed, assess its continued applicability.
d. Assign an applicability score to each predicate between 1 and 10.
e. IMPORTANT: Note that some predicates are harder to ascertain objectively, especially if they correspond to things relating to emotions or inner thoughts of people. Do not presume to know them for sure, and in such cases prefer to say that you cannot safely presume to ascertain whether they still apply—again, because emotionally-based predicates are hard to ascertain through a textual conversation.

### Examples of Predicate Evaluations:

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
- Predicates: ###
1) the client initiates a purchase
2) the client asks about data security
###
- **Expected Result**:
```json
{{ "checks": [
    {{
        "predicate_number": "1",
        "predicate": "the client initiates a purchase",
        "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": true,
        "is_this_predicate_hard_or_tricky_to_confidently_ascertain": true,
        "rationale": "The purchase-related guideline is irrelevant since the client completed the purchase and the conversation has moved to a new topic.",
        "applies_score": 3
    }},
    {{
        "predicate_number": "2",
        "predicate": "the client asks about data security",
        "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": false,
        "is_this_predicate_hard_or_tricky_to_confidently_ascertain": true,
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
- Predicates: ###
1) the client indicates they are in a hurry
2) a client inquires about pricing plans
3) a client asks for a summary of the features of the three plans.
###
- **Expected Result**:
```json
{{
    "checks": [
        {{
            "predicate_number": "1",
            "predicate": "the client indicates they are in a hurry",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": false,
            "is_this_predicate_hard_or_tricky_to_confidently_ascertain": true,
            "rationale": "The client initially stated they were in a hurry. This urgency applies throughout the conversation unless stated otherwise.",
            "applies_score": 8
        }},
        {{
            "predicate_number": "2",
            "predicate": "a client inquires about pricing plans",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": false,
            "is_this_predicate_hard_or_tricky_to_confidently_ascertain": true,
            "rationale": "The client inquired about pricing plans, specifically asking for details about the Pro plan.",
            "applies_score": 9
        }},
        {{
            "predicate_number": "3",
            "predicate": "a client asks for a summary of the features of the three plans.",
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
- Predicates: ###
1) the client asks for a recommendation
2) the client asks about movie genres
###
- **Expected Result**:
```json
{{
    "checks": [
        {{
            "predicate_number": "1",
            "predicate": "the client asks for a recommendation",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": false,
            "is_this_predicate_hard_or_tricky_to_confidently_ascertain": true,
            "rationale": "The client asked for a science fiction movie recommendation and the assistant provided one, making this guideline highly relevant.",
            "applies_score": 9
        }},
        {{
            "predicate_number": "2",
            "predicate": "the client asks about movie genres",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": true,
            "is_this_predicate_hard_or_tricky_to_confidently_ascertain": true,
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
- Predicates: ###
1) the client requests a modification to their order
2) the client asks for the store's location
###
- **Expected Result**:
```json
{{
    "checks": [
        {{
            "predicate_number": "1",
            "predicate": "the client requests a modification to their order",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": true,
            "is_this_predicate_hard_or_tricky_to_confidently_ascertain": true,
            "rationale": "The client requested a modification (an extra pillow) and the assistant confirmed it, making this guideline irrelevant now as it has already been addressed.",
            "applies_score": 3
        }},
        {{
            "predicate_number": "2",
            "predicate": "the client asks for the store's location",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": false,
            "is_this_predicate_hard_or_tricky_to_confidently_ascertain": true,
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
- Predicates: ###
1) the order does not exceed the limit of products
2) the client asks about product availability
###
- **Expected Result**:
```json
{{
    "checks": [
        {{
            "predicate_number": "1",
            "predicate": "the order does not exceed the limit of products",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": false,
            "rationale": "The client added an extra charger, and the order did not exceed the limit of products, making this guideline relevant.",
            "applies_score": 9
        }},
        {{
            "predicate_number": "2",
            "predicate": "the client asks about product availability",
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
- Predicates: ###
1) the user is currently eating lunch
2) the user agrees with you in the scope of an argument
###
- **Expected Result**:
```json
{{
    "checks": [
        {{
            "predicate_number": "1",
            "predicate": "the user is currently eating lunch",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": false,
            "is_this_predicate_hard_or_tricky_to_confidently_ascertain": false,
            "rationale": "There's nothing to indicate that the user is eating, lunch or otherwise",
            "applies_score": 1
        }},
        {{
            "predicate_number": "2",
            "predicate": "the user agrees with you in the scope of an argument",
            "you_the_agent_already_resolved_this_according_to_the_record_of_the_interaction": true,
            "is_this_predicate_hard_or_tricky_to_confidently_ascertain": false,
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

        builder.add_section(
            f"""
The following is an additional list of staged events that were just added: ###
{staged_events}
###
"""
        )

        builder.add_context_variables(context_variables)
        builder.add_glossary(terms)

        builder.add_guideline_predicates(predicates)
        builder.add_section(f"""
IMPORTANT: Please note there are exactly {len(predicates)} predicates in the list for you to check.

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

        return prompt
