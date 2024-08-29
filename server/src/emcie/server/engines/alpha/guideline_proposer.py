import asyncio
import json
import math
import jsonfinder  # type: ignore
from typing import Sequence

from emcie.server.core.agents import Agent
from emcie.server.core.context_variables import ContextVariable, ContextVariableValue
from emcie.server.engines.alpha.guideline_proposition import GuidelineProposition
from emcie.server.engines.alpha.prompt_builder import PromptBuilder
from emcie.server.core.terminology import Term
from emcie.server.engines.alpha.utils import make_llm_client
from emcie.server.core.guidelines import Guideline
from emcie.server.core.sessions import Event
from emcie.server.engines.common import ProducedEvent
from emcie.server.logger import Logger


class GuidelineProposer:
    def __init__(
        self,
        logger: Logger,
    ) -> None:
        self.logger = logger

        self._llm_client = make_llm_client("openai")

    async def propose_guidelines(
        self,
        agents: Sequence[Agent],
        guidelines: Sequence[Guideline],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        staged_events: Sequence[ProducedEvent],
    ) -> Sequence[GuidelineProposition]:
        if not guidelines:
            return []

        batches = self._create_batches(guidelines, batch_size=5)

        with self.logger.operation(f"Total guideline proposal ({len(batches)} batches)"):
            batch_tasks = [
                self._process_batch(
                    agents,
                    context_variables,
                    interaction_history,
                    staged_events,
                    terms,
                    batch,
                )
                for batch in batches
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            propositions = sum(batch_results, [])

        return propositions

    def _create_batches(
        self,
        guidelines: Sequence[Guideline],
        batch_size: int,
    ) -> Sequence[Sequence[Guideline]]:
        batches = []
        batch_count = math.ceil(len(guidelines) / batch_size)

        for batch_number in range(batch_count):
            start_offset = batch_number * batch_size
            end_offset = start_offset + batch_size
            batch = guidelines[start_offset:end_offset]
            batches.append(batch)

        return batches

    async def _process_batch(
        self,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        staged_events: Sequence[ProducedEvent],
        terms: Sequence[Term],
        batch: Sequence[Guideline],
    ) -> list[GuidelineProposition]:
        prompt = self._format_prompt(
            agents,
            context_variables=context_variables,
            interaction_history=interaction_history,
            staged_events=staged_events,
            terms=terms,
            guidelines=batch,
        )

        with self.logger.operation("Guideline batch proposal"):
            llm_response = await self._generate_llm_response(prompt)

        propositions_json = json.loads(llm_response)["checks"]

        propositions = []
        for proposition in propositions_json:
            self.logger.debug(
                f'Guideline proposer result for predicate "{batch[int(proposition["predicate_number"]) - 1].predicate}":\n'  # noqa
                f'    applies_score: {proposition["applies_score"]},\n'
                f'    rationale: "{proposition["rationale"]}"\n'
            )
            if (proposition["applies_score"] >= 7) or (
                proposition["applies_score"] >= 5
                and not proposition[
                    "can_we_safely_presume_to_ascertain_whether_the_predicate_still_applies"
                ]
            ):
                propositions.append(
                    GuidelineProposition(
                        guideline=batch[int(proposition["predicate_number"]) - 1],
                        score=proposition["applies_score"],
                        rationale=proposition["rationale"],
                    )
                )

        return propositions

    def _format_prompt(
        self,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        staged_events: Sequence[ProducedEvent],
        terms: Sequence[Term],
        guidelines: Sequence[Guideline],
    ) -> str:
        assert len(agents) == 1

        result_structure = [
            {
                "predicate_number": i,
                "predicate": "<THE PREDICATE TEXT>",
                "was_already_addressed_or_resolved_according_to_the_record_of_the_interaction": "<BOOL>",
                "can_we_safely_presume_to_ascertain_whether_the_predicate_still_applies": "<BOOL>",
                "rationale": "<EXPLANATION WHY THE PREDICATE IS RELEVANT OR IRRELEVANT FOR THE "
                "CURRENT STATE OF THE INTERACTION>",
                "applies_score": "<RELEVANCE SCORE>",
            }
            for i, g in enumerate(guidelines, start=1)
        ]

        builder = PromptBuilder()

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
        builder.add_terminology(terms)

        builder.add_section(
            """
Task Description
----------------
Your job is to assess the relevance and/or applicability of the provided predicates
to the last known state of an interaction between yourself, AI assistant, and a user.
"""
        )

        builder.add_guideline_predicates(guidelines)

        builder.add_section(
            f"""
IMPORTANT: Please note there are exactly {len(guidelines)} predicates in the list for you to check.

Process Description
-------------------
a. Examine the provided interaction events to discern the latest state of interaction between the user and the assistant.
b. Evaluate the entire interaction to determine if each predicate is still relevant to the most recent interaction state.
c. If the predicate has already been addressed, assess its continued applicability.
d. Assign an applicability score to each predicate between 1 and 10.
e. IMPORTANT: Note that some predicates are harder to ascertain objectively, especially if they correspond to things relating to emotions or inner thoughts of people. Do not presume to know them for sure.

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
        "was_already_addressed_or_resolved_according_to_the_record_of_the_interaction": true,
        "can_we_safely_presume_to_ascertain_whether_the_predicate_still_applies": true,
        "rationale": "The purchase-related guideline is irrelevant since the client completed the purchase and the conversation has moved to a new topic.",
        "applies_score": 3
    }},
    {{
        "predicate_number": "2",
        "predicate": "the client asks about data security",
        "was_already_addressed_or_resolved_according_to_the_record_of_the_interaction": false,
        "can_we_safely_presume_to_ascertain_whether_the_predicate_still_applies": true,
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
            "was_already_addressed_or_resolved_according_to_the_record_of_the_interaction": false,
            "can_we_safely_presume_to_ascertain_whether_the_predicate_still_applies": true,
            "rationale": "The client initially stated they were in a hurry. This urgency applies throughout the conversation unless stated otherwise.",
            "applies_score": 8
        }},
        {{
            "predicate_number": "2",
            "predicate": "a client inquires about pricing plans",
            "was_already_addressed_or_resolved_according_to_the_record_of_the_interaction": false,
            "can_we_safely_presume_to_ascertain_whether_the_predicate_still_applies": true,
            "rationale": "The client inquired about pricing plans, specifically asking for details about the Pro plan.",
            "applies_score": 9
        }},
        {{
            "predicate_number": "3",
            "predicate": "a client asks for a summary of the features of the three plans.",
            "was_already_addressed_or_resolved_according_to_the_record_of_the_interaction": false,
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
            "was_already_addressed_or_resolved_according_to_the_record_of_the_interaction": false,
            "can_we_safely_presume_to_ascertain_whether_the_predicate_still_applies": true,
            "rationale": "The client asked for a science fiction movie recommendation and the assistant provided one, making this guideline highly relevant.",
            "applies_score": 9
        }},
        {{
            "predicate_number": "2",
            "predicate": "the client asks about movie genres",
            "was_already_addressed_or_resolved_according_to_the_record_of_the_interaction": true,
            "can_we_safely_presume_to_ascertain_whether_the_predicate_still_applies": true,
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
            "was_already_addressed_or_resolved_according_to_the_record_of_the_interaction": true,
            "can_we_safely_presume_to_ascertain_whether_the_predicate_still_applies": true,
            "rationale": "The client requested a modification (an extra pillow) and the assistant confirmed it, making this guideline irrelevant now as it has already been addressed.",
            "applies_score": 3
        }},
        {{
            "predicate_number": "2",
            "predicate": "the client asks for the store's location",
            "was_already_addressed_or_resolved_according_to_the_record_of_the_interaction": false,
            "can_we_safely_presume_to_ascertain_whether_the_predicate_still_applies": true,
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
            "was_already_addressed_or_resolved_according_to_the_record_of_the_interaction": false,
            "rationale": "The client added an extra charger, and the order did not exceed the limit of products, making this guideline relevant.",
            "applies_score": 9
        }},
        {{
            "predicate_number": "2",
            "predicate": "the client asks about product availability",
            "was_already_addressed_or_resolved_according_to_the_record_of_the_interaction": false,
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
            "was_already_addressed_or_resolved_according_to_the_record_of_the_interaction": false,
            "can_we_safely_presume_to_ascertain_whether_the_predicate_still_applies": false,
            "rationale": "There's nothing to indicate that the user is eating, lunch or otherwise",
            "applies_score": 1
        }},
        {{
            "predicate_number": "2",
            "predicate": "the user agrees with you in the scope of an argument",
            "was_already_addressed_or_resolved_according_to_the_record_of_the_interaction": true,
            "can_we_safely_presume_to_ascertain_whether_the_predicate_still_applies": false,
            "rationale": "The user said 'Okay, fine', but it's possible that they are still in disagreement internally",
            "applies_score": 4
        }}
    ]
}}
```

Expected Output
---------------------------
- Specify the applicability of each predicate by filling in the rationale and applied score in the following list:

    ```json
    {{
        "checks":
        {result_structure}
    }}
    ```
"""  # noqa
        )

        return builder.build()

    async def _generate_llm_response(self, prompt: str) -> str:
        response = await self._llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        return json.dumps(jsonfinder.only_json(content)[2])  # type: ignore
