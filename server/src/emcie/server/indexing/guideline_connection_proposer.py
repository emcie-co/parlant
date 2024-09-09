import asyncio
from dataclasses import dataclass
from itertools import chain
import json
from typing import Any, Sequence

from more_itertools import chunked
from emcie.server.core.guideline_connections import ConnectionKind
from emcie.server.core.guidelines import GuidelineContent
from emcie.server.engines.alpha.utils import make_llm_client
from emcie.server.logger import Logger


@dataclass(frozen=True)
class GuidelineConnectionProposition:
    source: GuidelineContent
    target: GuidelineContent
    kind: ConnectionKind
    score: int
    rationale: str


class GuidelineConnectionProposer:
    def __init__(self, logger: Logger) -> None:
        self.logger = logger
        self._llm_client = make_llm_client("openai")
        self._batch_size = 5

    async def propose_connections(
        self,
        introduced_guidelines: Sequence[GuidelineContent],
        existing_guidelines: Sequence[GuidelineContent] = [],
    ) -> Sequence[GuidelineConnectionProposition]:
        if not introduced_guidelines:
            return []

        connection_proposition_tasks = []
        connection_kind_classification_tasks = []

        for i, introduced_guideline in enumerate(introduced_guidelines):
            filtered_existing_guidelines = [
                g
                for g in chain(
                    introduced_guidelines[i + 1 :],
                    existing_guidelines,
                )
            ]

            guideline_batches = chunked(filtered_existing_guidelines, self._batch_size)

            connection_proposition_tasks.extend(
                [
                    asyncio.create_task(self._generate_propositions(introduced_guideline, batch))
                    for batch in guideline_batches
                ]
            )

        with self.logger.operation(
            f"Propose guideline connections for {len(connection_proposition_tasks)} "  # noqa
            f"batches (batch size={self._batch_size})",
        ):
            propositions = chain(await asyncio.gather(*connection_proposition_tasks))

        connection_kind_classification_tasks.extend(
            [
                asyncio.create_task(
                    self._classify_connections(
                        batch,
                        introduced_guidelines,
                        existing_guidelines if existing_guidelines else [],
                    )
                )
                for batch in propositions
            ]
        )

        with self.logger.operation(
            f"Determine connections propositions for {len(connection_kind_classification_tasks)} "  # noqa
            f"batches (batch size={self._batch_size})",
        ):
            connections = list(
                chain.from_iterable(await asyncio.gather(*connection_kind_classification_tasks))
            )

        return connections

    def _format_connection_propositions(
        self,
        evaluated_guideline: GuidelineContent,
        comparison_set: Sequence[GuidelineContent],
    ) -> str:
        comparison_set_string = "\n\t".join(
            f"{i}) {{when: {g.predicate}, then: {g.action}}}"
            for i, g in enumerate(comparison_set, start=1)
        )
        evaluated_guideline_string = (
            f"{{when: '{evaluated_guideline.predicate}', then: '{evaluated_guideline.action}'}}"
        )

        return f"""
Input:
- evaluated_guideline: {evaluated_guideline_string}

- comparison_set: ###
{comparison_set_string}
###

Task:
1. Determine if there is a suggestive or entailed connection between the evaluated guideline and each of the guidelines in the comparison set.
2. For example, if guideline A's "then" suggests or entails guideline B's "when", then A is considered the source of B, which here is the target. And vice versa.
3. For each connection found, the output should be JSON structured as follows:

{{
    "propositions": [
        {{
            "source": <The source guideline>,
            "target": <The target guideline>,
            "source_then": <The source guideline's 'then'>,
            "target_when": <The target guideline's 'when'>,
            "rationale": <Explanation for the connection between the source's 'then' and the target's 'when'>,
            "connection_score": <Score between 1-10 indicating the strength of the connection>
        }},
        ...
    ]
}}

IMPORTANT: The evaluated guideline can serve as either the source or the target in the connection.

###Examples:

Example 1:###
{{
    "source": {{"when": "The user asks about the weather", "then": "provide the current weather update"}},
    "target": {{"when": "providing the weather update", "then": "mention the best time to go for a walk"}},
    "rationale": "Mentioning the best time to go for a walk follows logically from providing a weather update.",
    "connection_score": 10
}}
###

Example 2:###
{{
    "source": {{"when": "The user greets you", "then": "Greet them back with 'Hello'"}},
    "target": {{"when": "The user asks for directions", "then": "provide step-by-step directions"}},
    "rationale": "Greeting the user and providing directions are unrelated actions.",
    "connection_score": 2
}}
###

Example 3:###
{{
    "source": {{"when": "The user asks for a book recommendation", "then": "suggest a popular book"}},
    "target": {{"when": "suggesting a book", "then": "mention its availability in the local library"}},
    "rationale": "Mentioning a book's availability in the library is a helpful follow-up to suggesting a book.",
    "connection_score": 9
}}
###

Example 4:###
{{
    "source": {{"when": "The user asks about nearby restaurants", "then": "provide a list of popular restaurants"}},
    "target": {{"when": "listing restaurants", "then": "highlight the one with the best reviews"}},
    "rationale": "Highlighting the best-reviewed restaurant enhances the list of popular restaurants.",
    "connection_score": 9
}}
###

Example 5:###
{{
    "source": {{"when": "The user greets you", "then": "Greet them back with 'Hello'"}},
    "target": {{"when": "The user asks about the weather", "then": "provide the current weather update"}},
    "rationale": "Greeting the user and providing a weather update are unrelated actions.",
    "connection_score": 2
}}
###

Example 6:###
{{
    "source": {{"when": "The user greets you", "then": "Greet them back with 'Hello'"}},
    "target": {{"when": "The user asks for a book recommendation", "then": "suggest a popular book"}},
    "rationale": "Greeting the user and suggesting a book are unrelated actions.",
    "connection_score": 1
}}
###

Example 7:###
{{
    "source": {{"when": "The user greets you", "then": "Greet them back with 'Hello'"}},
    "target": {{"when": "The user mentions being tired", "then": "suggest taking a break"}},
    "rationale": "Greeting the user and suggesting a break are unrelated actions.",
    "connection_score": 1
}}
###

Example 8:###
{{
    "source": {{"when": "The user greets you", "then": "Greet them back with 'Hello'"}},
    "target": {{"when": "The user mentions being new to the area", "then": "offer a local guide"}},
    "rationale": "Greeting the user and offering a local guide are unrelated actions.",
    "connection_score": 2
}}
###

Example 9:###
{{
    "source": {{"when": "The user greets you", "then": "Greet them back with 'Hello'"}},
    "target": {{"when": "The user asks for tech support", "then": "provide the tech support contact"}},
    "rationale": "Greeting the user and providing tech support contact are unrelated actions.",
    "connection_score": 2
}}
###

Example 10:###
{{
    "source": {{"when": "The user inquires about office hours", "then": "tell them the office hours"}},
    "target": {{"when": "mentioning office hours", "then": "suggest the best time to visit for quicker service"}},
    "rationale": "Suggesting the best time to visit follows from mentioning office hours but is not strictly necessary.",
    "connection_score": 8
}}
###

"""  # noqa

    async def _generate_propositions(
        self,
        guideline_to_test: GuidelineContent,
        guidelines_to_compare: Sequence[GuidelineContent],
    ) -> list[dict[str, Any]]:
        prompt = self._format_connection_propositions(guideline_to_test, guidelines_to_compare)
        response = await self._llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""

        all_propositions: list[dict[str, Any]] = json.loads(content)["propositions"]

        self.logger.debug(
            f"""
----------------------------------------
Connection Propositions Found:
----------------------------------------
{json.dumps(all_propositions, indent=2)}
----------------------------------------
"""
        )

        relevant_propositions = [p for p in all_propositions if p["connection_score"] >= 7]

        return relevant_propositions

    def _format_classification_connections(
        self,
        connection_propositions: Sequence[dict[str, Any]],
    ) -> str:
        propositions_output_structure = [
            {
                "source": p["source"],
                "target": p["target"],
                "rationale": "<EXPLANATION FOR WHY THE CONNECTION IS THE KIND CONNECTION DESCRIBED>",  # noqa
                "kind": "<suggests/entails>",
                "score": p["connection_score"],
            }
            for p in connection_propositions
        ]
        return f"""

- Task:
Determine the type of connection (suggests or entails) between the source and target guidelines in each connection proposition.
For each connection found.

- Input: ###
{json.dumps(connection_propositions, indent=2)}
###

- Examples:

Example 1:###
{{
    "source": {{"when": "The user asks about the weather", "then": "provide the current weather update"}},
    "target": {{"when": "providing the weather update", "then": "mention the best time to go for a walk"}},
    "rationale": "Providing a weather update directly entails mentioning the best time to go for a walk.",
    "score": 8,
    "kind": "entails"
}}
###

Example 2:###
{{
    "source": {{"when": "The user inquires about office hours", "then": "tell them the office hours"}},
    "target": {{"when": "mentioning office hours", "then": "suggest the best time to visit for quicker service"}},
    "rationale": "Mentioning office hours suggests the best time to visit for quicker service.",
    "score": 8,
    "kind": "suggests"
}}
###

Example 3:###
{{
    "source": {{"when": "The user asks for a book recommendation", "then": "suggest a popular book"}},
    "target": {{"when": "suggesting a book", "then": "mention its availability in the local library"}},
    "rationale": "Suggesting a book entails mentioning its availability in the local library.",
    "score": 9,
    "kind": "entails"
}}
###

Example 4:###
{{
    "source": {{"when": "The user asks about nearby restaurants", "then": "provide a list of popular restaurants"}},
    "target": {{"when": "listing restaurants", "then": "highlight the one with the best reviews"}},
    "rationale": "Listing restaurants entails highlighting the one with the best reviews.",
    "score": 7,
    "kind": "entails"
}}
###

Example 5:###
{{
    "source": {{"when": "The user mentions being tired", "then": "suggest taking a break"}},
    "target": {{"when": "suggesting a break", "then": "offer a list of relaxing activities"}},
    "rationale": "Suggesting taking a break entails offering a list of relaxing activities.",
    "score": 7,
    "kind": "entails"
}}
###

Example 6:###
{{
    "source": {{"when": "The user asks for tech support", "then": "provide the tech support contact"}},
    "target": {{"when": "providing tech support contact", "then": "offer to connect them with a support agent"}},
    "rationale": "Providing tech support contact entails offering to connect them with a support agent.",
    "score": 10,
    "kind": "entails"
}}
###

Example 7:###
{{
    "source": {{"when": "The user mentions a problem with a product", "then": "offer troubleshooting steps"}},
    "target": {{"when": "offering troubleshooting steps", "then": "suggest contacting support if the problem persists"}},
    "rationale": "Offering troubleshooting steps entails suggesting contacting support if the problem persists.",
    "score": 10,
    "kind": "suggests"
}}
###

Example 8:###
{{
    "source": {{"when": "The user asks about upcoming events", "then": "provide the event schedule"}},
    "target": {{"when": "providing the event schedule", "then": "consider highlighting the most popular events"}},
    "rationale": "Providing the event schedule suggests considering highlighting the most popular events.",
    "score": 10,
    "kind": "suggests"
}}
###

Example 9:###
{{
    "source": {{"when": "The user asks for directions", "then": "provide step-by-step directions"}},
    "target": {{"when": "providing step-by-step directions", "then": "suggest the best route to avoid traffic"}},
    "rationale": "Providing step-by-step directions entails suggesting the best route to avoid traffic.",
    "score": 10,
    "kind": "suggests"
}}
###

Example 10:###
{{
    "source": {{"when": "The user asks about workout routines", "then": "provide a sample workout plan"}},
    "target": {{"when": "providing a workout plan", "then": "suggest healthy diet options"}},
    "rationale": "Providing a workout plan suggests healthy diet options.",
    "score": 8,
    "kind": "suggests"
}}
###

Example 11:###
{{
    "source": {{"when": "The user inquires about pet care tips", "then": "offer basic pet care advice"}},
    "target": {{"when": "offering pet care advice", "then": "consider to recommend local veterinarians"}},
    "rationale": "Offering pet care advice suggests recommending local veterinarians.",
    "score": 8,
    "kind": "suggests"
}}
###

Example 12:###
{{
    "source": {{"when": "The user asks about movie recommendations", "then": "suggest a popular movie"}},
    "target": {{"when": "suggesting a movie", "then": "mention its availability on streaming services"}},
    "rationale": "Suggesting a movie entails mentioning its availability on streaming services.",
    "score": 9,
    "kind": "entails"
}}
###

Example 13:###
{{
    "source": {{"when": "The user asks about cooking recipes", "then": "suggest a simple recipe"}},
    "target": {{"when": "suggesting a recipe", "then": "consider mentioning ingredient substitutes"}},
    "rationale": "Suggesting a recipe suggests mentioning ingredient substitutes.",
    "score": 9,
    "kind": "suggests"
}}
###

Example 14: ###
{{
    "source": {{"when": "The user asks for help with a recipe", "then": "explain the cooking steps in detail"}},
    "target": {{"when": "explaining cooking steps", "then": "mention any special techniques involved"}},
    "rationale": "Explaining the cooking steps in detail entails mentioning any special techniques involved.",
    "score": 9,
    "kind": "entails"
}}
###

Example 15: ###
{{
    "source": {{"when": "The user inquires about data privacy", "then": "provide information on how their data is stored"}},
    "target": {{"when": "providing information on data storage", "then": "mention the security measures in place"}},
    "rationale": "Providing information on how data is stored entails mentioning the security measures in place.",
    "score": 10,
    "kind": "entails"
}}
###

Example 16: ###
{{
    "source": {{"when": "The user asks about product return policies", "then": "explain the steps for returning a product"}},
    "target": {{"when": "explaining return steps", "then": "include details on how to package the return"}},
    "rationale": "Explaining the steps for returning a product entails including details on how to package the return.",
    "score": 8,
    "kind": "entails"
}}
###

Example 17: ###
{{
    "source": {{"when": "The user inquires about account management", "then": "provide instructions for changing account settings"}},
    "target": {{"when": "providing account settings instructions", "then": "mention how to reset passwords"}},
    "rationale": "Providing instructions for changing account settings entails mentioning how to reset passwords.",
    "score": 8,
    "kind": "entails"
}}
###

Example 18: ###
{{
    "source": {{"when": "The user asks about troubleshooting a device", "then": "give them a step-by-step guide"}},
    "target": {{"when": "providing a step-by-step guide", "then": "include a note on common issues and fixes"}},
    "rationale": "Providing a step-by-step guide entails including a note on common issues and fixes.",
    "score": 9,
    "kind": "entails"
}}
###

- Output:
The output should be a JSON object with the following structure, where you need to fill in the rationale and kind: ###
{{"propositions": {json.dumps(propositions_output_structure, indent=2)}}}
###


Note: The evaluated guideline can be either of the kind "entails" or "suggests."
"""  # noqa

    async def _classify_connections(
        self,
        connection_propositions: Sequence[dict[str, Any]],
        introduced_guidelines: Sequence[GuidelineContent],
        existing_guidelines: Sequence[GuidelineContent],
    ) -> Sequence[GuidelineConnectionProposition]:
        prompt = self._format_classification_connections(connection_propositions)
        response = await self._llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""

        connection_list = json.loads(content)["propositions"]

        self.logger.debug(f"Connection Propositions Found: {json.dumps(connection_list, indent=2)}")

        staged_guidelines = {
            f"{s.predicate}_{s.action}".lower(): s
            for s in chain(introduced_guidelines, existing_guidelines)
        }

        propositions = [
            GuidelineConnectionProposition(
                source=staged_guidelines[f'{c["source"]["when"]}_{c["source"]["then"]}'.lower()],
                target=staged_guidelines[f'{c["target"]["when"]}_{c["target"]["then"]}'.lower()],
                kind=ConnectionKind[c["kind"].upper()],
                rationale=c["rationale"],
                score=c["score"],
            )
            for c in connection_list
        ]

        return propositions
