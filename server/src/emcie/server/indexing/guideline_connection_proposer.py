import asyncio
from dataclasses import dataclass
from itertools import chain
import json
from typing import Any, Sequence

from more_itertools import chunked
from emcie.server.core.guideline_connections import ConnectionKind
from emcie.server.core.guidelines import GuidelineContent
from emcie.server.llm.schematic_generators import SchematicGenerator
from emcie.server.logger import Logger
from emcie.server.base_models import DefaultBaseModel


class GuidelineConnectionPropositionSchema(DefaultBaseModel):
    source: dict[str, Any]
    target: dict[str, Any]
    source_then: str
    target_when: str
    is_target_when_implied_by_source_then: bool
    is_target_then_suggestive_or_optional: bool = False
    rationale: str
    implication_score: int


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
        schematic_generator: SchematicGenerator[
            GuidelineConnectionPropositionsSchema
        ],
    ) -> None:
        self.logger = logger
        self._batch_size = 5

        self._schematic_generator = schematic_generator

    async def propose_connections(
        self,
        introduced_guidelines: Sequence[GuidelineContent],
        existing_guidelines: Sequence[GuidelineContent] = [],
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
            propositions = chain.from_iterable(await asyncio.gather(*connection_proposition_tasks))
            return list(propositions)

    def _format_connection_propositions(
        self,
        evaluated_guideline: GuidelineContent,
        comparison_set: Sequence[GuidelineContent],
    ) -> str:
        implication_candidates = "\n\t".join(
            f"{i}) {{when: {g.predicate}, then: {g.action}}}"
            for i, g in enumerate(comparison_set, start=1)
        )
        test_guideline = (
            f"{{when: '{evaluated_guideline.predicate}', then: '{evaluated_guideline.action}'}}"
        )

        return f"""
In our system, the behavior of conversational AI agents is guided by "guidelines".

Each guideline is composed of two parts:
- "when": this is a natural-language predicate that specifies when a guideline should apply.
          we look at each conversation at any particular state, and we test against this
          predicate to understand if we should have this guideline participate in generating
          the next reply to the user.
- "then": this is a natural-language instruction that should be followed whenever the
          "when" part of the guidelines applies to the conversation in its particular state.

Now, if we have many guidelines, it is possible to encounter a situation like this:
Guideline N: When <X>, then <Y>.
Guideline N+1: When <Y>, then <Z>.

Note, in this edge case, that, when testing these two guidelines (N, N+1) against a conversation,
it may happen that <X> applies and therefore Guideline N should apply.
However, the application of Guideline N would also make necessarily <Y> apply.
But just by looking at the conversation itself, before the application of Guideline N,
<Y> may not necessarily apply, which would lead to Guideline N+1 not applying either.
But this is a bug in the expected behavior, because, if the guideline designer said
"When <X> then <Y>" and "When <Y> then <Z>", they clearly want <Z> to happen when <X> applies.

We therefore need to detect these implicit connections between guidelines, and index them as such.

Your job is to detect cases in evaluated guidelines that follow the pattern:
When <X> then <Y>; When <Y> then <Z>.
Meaning, that the "then" part of the first guideline (called the "source" guideline),
by itself implies and makes true the "when" part of the second guideline (called the "target" guideline).
We call this pattern an "implication".
Please note that there is no implication when <Y> is something that *may* happen in the future following <X>.
It is only about whether <Y> in itself is directly and immediately implied by <X>.

There are two types of implication:
- Entailed: This means that the source guideline's "then" necessarily entails
            the target guideline's "when".
- Optional: This is the same as "entailed", except in these cases the target's
            "then" is suggestive instead of necessary. For example, in cases such as
             source="When <X> then <Y>", target="When <Y> then consider <Z>".

You will now receive a test guideline to evaluate against a set of other guidelines which are candidates for
implication. For each candidate, the test guideline may be found as either the source or the target in the
connection, or neither.

Please output JSON structured as per the following examples:

{{
    "propositions": [
        {{
            "source": <The source guideline in its entirety>,
            "target": <The target guideline in its entirety>,
            "source_then": <The source guideline's 'then'>,
            "target_when": <The target guideline's 'when'>,
            "is_target_when_implied_by_source_then": <BOOL>,
            "is_target_then_suggestive_or_optional": <BOOL>,
            "rationale": <Explanation for the implication between the source's 'then' and the target's 'when'>,
            "implication_score": <Score between 1-10 indicating the strength of the connection>
        }},
        ...
    ]
}}

Examples:

Example 1: ###
{{
    "source": {{"when": "The user asks about the weather", "then": "provide the current weather update"}},
    "target": {{"when": "providing the weather update", "then": "mention the best time to go for a walk"}},
    "source_then": "provide the current weather update",
    "target_when": "providing the weather update",
    "is_target_when_implied_by_source_then": true,
    "is_target_then_suggestive_or_optional": false,
    "rationale": "Mentioning the best time to go for a walk follows logically from providing a weather update.",
    "implication_score": 10
}}
###

Example 2: ###
{{
    "source": {{"when": "The user greets you", "then": "Greet them back with 'Hello'"}},
    "target": {{"when": "The user asks for directions", "then": "provide step-by-step directions"}},
    "source_then": "Greet them back with 'Hello'",
    "target_when": "The user asks for directions",
    "is_target_when_implied_by_source_then": false,
    "rationale": "The user's asking for directions is not implied by the agent's greeting the user",
    "implication_score": 2
}}
###

Example 3: ###
{{
    "source": {{"when": "The user asks for a book recommendation", "then": "suggest a book"}},
    "target": {{"when": "suggesting a book", "then": "mention its availability in the local library"}},
    "source_then": "suggest a book",
    "target_when": "suggesting a book",
    "is_target_when_implied_by_source_then": false,
    "rationale": "The source's 'then' directly makes the target's 'when' apply",
    "implication_score": 10
}}
###

Example 4: ###
{{
    "source": {{"when": "The user asks about nearby restaurants", "then": "provide a list of popular restaurants"}},
    "target": {{"when": "listing restaurants", "then": "consider highlighting the one with the best reviews"}},
    "source_then": "provide a list of popular restaurants",
    "target_when": "listing restaurants",
    "is_target_when_implied_by_source_then": true,
    "is_target_then_suggestive_or_optional": true,
    "rationale": "The source's 'then' is a specific case of the target's 'when'",
    "implication_score": 9
}}
###

Example 5: ###
{{
    "source": {{"when": "The user greets you", "then": "Greet them back with 'Hello'"}},
    "target": {{"when": "The user asks about the weather", "then": "provide the current weather update"}},
    "source_then": "Greet them back with 'Hello'",
    "target_when": "The user asks about the weather",
    "is_target_when_implied_by_source_then": false,
    "rationale": "The user's asking about the weather is not implied by the agent's greeting the user",
    "implication_score": 2
}}
###

Example 6: ###
{{
    "source": {{"when": "The user greets you", "then": "Greet them back with 'Hello'"}},
    "target": {{"when": "The user asks for a book recommendation", "then": "suggest a popular book"}},
    "source_then": "Greet them back with 'Hello'",
    "target_when": "The user asks for a book recommendation",
    "is_target_when_implied_by_source_then": false,
    "rationale": "The user's asking for a book recommendation is not implied by the agent's greeting the user",
    "implication_score": 1
}}
###

Example 7: ###
{{
    "source": {{"when": "The user greets you", "then": "Greet them back with 'Hello'"}},
    "target": {{"when": "The user mentions being tired", "then": "suggest taking a break"}},
    "source_then": "Greet them back with 'Hello'",
    "target_when": "The user mentions being tired",
    "is_target_when_implied_by_source_then": false,
    "rationale": "The user's mentioning being tired is not implied by the agent's greeting the user",
    "implication_score": 1
}}
###

Example 8: ###
{{
    "source": {{"when": "The user greets you", "then": "Greet them back with 'Hello'"}},
    "target": {{"when": "The user mentions being new to the area", "then": "offer a local guide"}},
    "source_then": "Greet them back with 'Hello'",
    "target_when": "The user mentions being new to the area",
    "is_target_when_implied_by_source_then": false,
    "rationale": "The user's mentioning being new to the area is not implied by the agent's greeting the user",
    "implication_score": 2
}}
###

Example 9: ###
{{
    "source": {{"when": "The user greets you", "then": "Greet them back with 'Hello'"}},
    "target": {{"when": "The user asks for tech support", "then": "provide the tech support contact"}},
    "source_then": "Greet them back with 'Hello'",
    "target_when": "The user asks for tech support",
    "is_target_when_implied_by_source_then": false,
    "rationale": "The user's asking for tech support is not implied by the agent's greeting the user",
    "implication_score": 2
}}
###

Example 10:###
{{
    "source": {{"when": "The user inquires about office hours", "then": "tell them the office hours"}},
    "target": {{"when": "mentioning office hours", "then": "suggest the best time to visit for quicker service"}},
    "source_then": "tell them the office hours",
    "target_when": "mentioning office hours",
    "is_target_when_implied_by_source_then": true,
    "is_target_then_suggestive_or_optional": false,
    "rationale": "If you tell the user about office hours, then the guideline for what to do when mentioning office hours should also apply",
    "implication_score": 9
}}
###

Example 11: ###
{{
    "source": {{
        "when": "The user asks if Google is a search engine",
        "then": "First double check with Wikipedia, but then answer positively. Explain to them why."
    }},
    "target": {{
        "when": "The user asks you to explain more",
        "then": "Consult Wikipedia with the query 'information about Google'"
    }},
    "source_then": "First double check with Wikipedia, but then answer positively. Explain to them why."
    "target_when": "The user asks you to explain more",
    "is_target_when_implied_by_source_then": false,
    "rationale": "While the user might then ask for more information, thus triggering the target's 'when', this is not certain in advance.",
    "implication_score": 3
}}
###

Input:

Test guideline: ###
{test_guideline}
###

Implication candidates: ###
{implication_candidates}
###


"""  # noqa

    async def _generate_propositions(
        self,
        guideline_to_test: GuidelineContent,
        guidelines_to_compare: Sequence[GuidelineContent],
    ) -> list[GuidelineConnectionProposition]:
        prompt = self._format_connection_propositions(guideline_to_test, guidelines_to_compare)

        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": 0.0},
        )

        self.logger.debug(
            f"""
----------------------------------------
Connection Propositions Found:
----------------------------------------
{json.dumps([p.model_dump() for p in response.content.propositions], indent=2)}
----------------------------------------
"""
        )

        relevant_propositions = [
            GuidelineConnectionProposition(
                source=GuidelineContent(predicate=p.source["when"], action=p.source["then"]),
                target=GuidelineContent(predicate=p.target["when"], action=p.target["then"]),
                kind={
                    False: ConnectionKind.ENTAILS,
                    True: ConnectionKind.SUGGESTS,
                }[p.is_target_then_suggestive_or_optional],
                score=int(p.implication_score),
                rationale=p.rationale,
            )
            for p in response.content.propositions
            if p.implication_score >= 7
        ]

        return relevant_propositions
