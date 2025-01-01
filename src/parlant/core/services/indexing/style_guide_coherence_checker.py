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
from datetime import datetime, timezone
from enum import Enum, auto
from itertools import chain
import json
from typing import Optional, Sequence
from dataclasses import dataclass

from more_itertools import chunked

from parlant.core import async_utils
from parlant.core.agents import Agent
from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.logging import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.glossary import GlossaryStore
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.style_guides import StyleGuideContent

EVALUATION_BATCH_SIZE = 5
CRITICAL_INCOHERENCE_THRESHOLD = 6
BASIC_CONTRADICTION_SEVERITY_THRESHOLD = 5


class IncoherenceKind(Enum):
    STRICT = auto()
    CONTINGENT = auto()


@dataclass(frozen=True)
class IncoherenceTest:
    style_guide_a: StyleGuideContent
    style_guide_b: StyleGuideContent
    incoherence_kind: IncoherenceKind
    principles_rationale: str
    principles_severity: int
    creation_utc: datetime


class StyleGuideContradictionTestSchema(DefaultBaseModel):
    compared_style_guide_id: int
    origin_style_guide_principle: str
    compared_style_guide_principle: str
    rationale: str
    principles_contradiction: bool
    severity: int


class StyleGuideContradictionsSchema(DefaultBaseModel):
    contradictions: list[StyleGuideContradictionTestSchema]


class StyleGuideCoherenceChecker:
    def __init__(
        self,
        logger: Logger,
        principles_test_schematic_generator: SchematicGenerator[StyleGuideContradictionsSchema],
        glossary_store: GlossaryStore,
    ) -> None:
        self._logger = logger
        self._checker = StyleGuideContradictionChecker(
            logger,
            principles_test_schematic_generator,
            glossary_store,
        )

    async def propose_incoherencies(
        self,
        agent: Agent,
        style_guides_to_evaluate: Sequence[StyleGuideContent],
        comparison_style_guides: Sequence[StyleGuideContent] = (),
        progress_report: Optional[ProgressReport] = None,
    ) -> Sequence[IncoherenceTest]:
        comparison_list = list(comparison_style_guides)
        to_evaluate_list = list(style_guides_to_evaluate)
        tasks = []

        for i, sg_to_evaluate in enumerate(to_evaluate_list):
            filtered_comparisons = to_evaluate_list[i + 1 :] + comparison_list
            style_guide_batches = list(chunked(filtered_comparisons, EVALUATION_BATCH_SIZE))
            if progress_report:
                await progress_report.stretch(len(style_guide_batches))

            for batch in style_guide_batches:
                tasks.append(
                    asyncio.create_task(
                        self._evaluate_batch(agent, sg_to_evaluate, batch, progress_report)
                    )
                )

        with self._logger.operation(
            f"Evaluating style guide incoherencies for {len(tasks)} "
            f"batches (batch size={EVALUATION_BATCH_SIZE})",
        ):
            incoherencies = list(chain.from_iterable(await async_utils.safe_gather(*tasks)))

        return incoherencies

    async def _evaluate_batch(
        self,
        agent: Agent,
        sg_to_evaluate: StyleGuideContent,
        batch: Sequence[StyleGuideContent],
        progress_report: Optional[ProgressReport],
    ) -> Sequence[IncoherenceTest]:
        indexed_comps = {i: c for i, c in enumerate(batch, start=1)}

        checker_response = await self._checker.evaluate(
            agent=agent,
            style_guide_to_evaluate=sg_to_evaluate,
            indexed_comparison_guidelines=indexed_comps,
        )

        incoherencies: list[IncoherenceTest] = []

        for contradiction in checker_response:
            csg = indexed_comps[contradiction.compared_style_guide_id]

            if (
                contradiction.principles_contradiction
                and contradiction.severity >= BASIC_CONTRADICTION_SEVERITY_THRESHOLD
            ):
                kind = (
                    IncoherenceKind.STRICT
                    if contradiction.severity >= CRITICAL_INCOHERENCE_THRESHOLD
                    else IncoherenceKind.CONTINGENT
                )

                incoherencies.append(
                    IncoherenceTest(
                        style_guide_a=sg_to_evaluate,
                        style_guide_b=csg,
                        incoherence_kind=kind,
                        principles_rationale=contradiction.rationale,
                        principles_severity=contradiction.severity,
                        creation_utc=datetime.now(timezone.utc),
                    )
                )

        if progress_report:
            await progress_report.increment()

        return incoherencies


class StyleGuideContradictionChecker:
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[StyleGuideContradictionsSchema],
        glossary_store: GlossaryStore,
    ) -> None:
        self._logger = logger
        self._schematic_generator = schematic_generator
        self._glossary_store = glossary_store

    async def evaluate(
        self,
        agent: Agent,
        style_guide_to_evaluate: StyleGuideContent,
        indexed_comparison_guidelines: dict[int, StyleGuideContent],
    ) -> Sequence[StyleGuideContradictionTestSchema]:
        prompt = await self._format_prompt(
            agent, style_guide_to_evaluate, indexed_comparison_guidelines
        )
        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": 0.0},
        )
        self._logger.debug(
            f"""
----------------------------------------
Style Guide Contradiction Test Results:
----------------------------------------
{json.dumps([p.model_dump(mode="json") for p in response.content.contradictions], indent=2)}
----------------------------------------
"""
        )

        return response.content.contradictions

    async def _format_prompt(
        self,
        agent: Agent,
        style_guide_to_evaluate: StyleGuideContent,
        indexed_comparison_style_guides: dict[int, StyleGuideContent],
    ) -> str:
        builder = PromptBuilder()
        comparison_candidates_text = "\n".join(
            f"""{{"id": {id}, "principle": "{s.principle}", "examples": "{json.dumps(s.examples, indent=2)}"}}"""
            for id, s in indexed_comparison_style_guides.items()
        )
        style_guide_to_evaluate_text = f"""{{"principle": "{style_guide_to_evaluate.principle}", "examples": "{json.dumps(style_guide_to_evaluate.examples, indent=2)}"}}"""

        builder.add_section(
            f"""
Style guides define how the conversational AI agent should speak and respond in various situations. Each style guide typically includes:
	1.	A principle, which describes the overarching tone, voice, or rules for communication (e.g., “Use a friendly and casual tone” or “Use only imperial units when discussing measurements”).
	2.	A set of examples illustrating how to apply that principle in before/after scenarios, highlighting which messages are inconsistent (“before”) and how to correct them (“after”).

To ensure consistency, it is crucial to avoid scenarios where multiple style guides with conflicting 'principles' are applied.
{self.get_task_description()}

Be forgiving regarding misspellings and grammatical errors.


Please output JSON structured in the following format:
```json
{{
    "contradictions": [
        {{
            "compared_style_guide_id": <id of the compared style_guide>,
            "origin_style_guide_principle": <The origin style_guide's 'principle'>,
            "compared_style_guide_principle": <The compared style_guide's 'principle'>,
            "rationale": <Explanation for if and how the 'principle' statements contradict each other>,
            "principles_contradiction": <BOOL of whether the two 'principle' statements are contradictory>,
            "severity": <Score between 1-10 indicating the strength of the contradiction>
        }},
        ...
    ]
}}
```
###
"""  # noqa
        )
        builder.add_agent_identity(agent)
        terms = await self._glossary_store.find_relevant_terms(
            agent.id,
            query=style_guide_to_evaluate_text + comparison_candidates_text,
        )
        builder.add_glossary(terms)

        builder.add_section(f"""
The style guides you should analyze are:
Origin style guide: ###
{style_guide_to_evaluate_text}
###

Comparison candidates: ###
{comparison_candidates_text}
###""")
        return builder.build()

    @staticmethod
    def get_task_description() -> str:
        return """
Two 'principle' statements are considered contradictory if:

1. Applying both leads to a confusing or paradoxical response.
2. Applying both would result in the agent responding in a way which does not align with one of the principles.
When multiple style guides are in effect simultaneously, the agent must avoid violating any principles that might conflict.
"""
