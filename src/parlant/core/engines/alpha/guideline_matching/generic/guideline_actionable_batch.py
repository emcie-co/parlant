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

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import traceback
from typing_extensions import override
from parlant.core.common import DefaultBaseModel, JSONSerializable
from parlant.core.engines.alpha.guideline_matching.generic.common import (
    GuidelineInternalRepresentation,
    internal_representation,
)
from parlant.core.engines.alpha.guideline_matching.guideline_match import (
    GuidelineMatch,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingBatch,
    GuidelineMatchingBatchResult,
    GuidelineMatchingContext,
    GuidelineMatchingBatchError,
    GuidelineMatchingStrategy,
)
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.prompt_builder import BuiltInSection, PromptBuilder, SectionStatus
from parlant.core.entity_cq import EntityQueries
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.journeys import Journey
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.shots import Shot, ShotCollection


class GenericActionableBatch(DefaultBaseModel):
    guideline_id: str
    condition: str
    rationale: str
    applies: bool


class GenericActionableGuidelineMatchesSchema(DefaultBaseModel):
    checks: Sequence[GenericActionableBatch]


@dataclass
class GenericActionableGuidelineGuidelineMatchingShot(Shot):
    interaction_events: Sequence[Event]
    guidelines: Sequence[GuidelineContent]
    expected_result: GenericActionableGuidelineMatchesSchema


class GenericActionableGuidelineMatchingBatch(GuidelineMatchingBatch):
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[GenericActionableGuidelineMatchesSchema],
        guidelines: Sequence[Guideline],
        journeys: Sequence[Journey],
        context: GuidelineMatchingContext,
    ) -> None:
        self._logger = logger
        self._optimization_policy = optimization_policy
        self._schematic_generator = schematic_generator
        self._guidelines = {str(i): g for i, g in enumerate(guidelines, start=1)}
        self._journeys = journeys
        self._context = context

    @override
    async def process(self) -> GuidelineMatchingBatchResult:
        with self._logger.operation(f"Batch of {len(self._guidelines)} guidelines"):
            prompt = self._build_prompt(shots=await self.shots())

            try:
                generation_attempt_temperatures = (
                    self._optimization_policy.get_guideline_matching_batch_retry_temperatures(
                        hints={"type": self.__class__.__name__}
                    )
                )

                last_generation_exception: Exception | None = None

                for generation_attempt in range(3):
                    inference = await self._schematic_generator.generate(
                        prompt=prompt,
                        hints={"temperature": generation_attempt_temperatures[generation_attempt]},
                    )

                    if not inference.content.checks:
                        self._logger.warning(
                            "Completion:\nNo checks generated! This shouldn't happen."
                        )
                    else:
                        self._logger.trace(
                            f"Completion:\n{inference.content.model_dump_json(indent=2)}"
                        )

                    matches = []

                    for match in inference.content.checks:
                        if match.applies:
                            self._logger.debug(f"Activated:\n{match.model_dump_json(indent=2)}")

                            matches.append(
                                GuidelineMatch(
                                    guideline=self._guidelines[match.guideline_id],
                                    score=10 if match.applies else 1,
                                    rationale=match.rationale,
                                )
                            )
                        else:
                            self._logger.debug(f"Skipped:\n{match.model_dump_json(indent=2)}")

                    return GuidelineMatchingBatchResult(
                        matches=matches,
                        generation_info=inference.info,
                    )

            except Exception as exc:
                self._logger.warning(
                    f"Attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                )

                last_generation_exception = exc

        raise GuidelineMatchingBatchError() from last_generation_exception

    async def shots(self) -> Sequence[GenericActionableGuidelineGuidelineMatchingShot]:
        return await shot_collection.list()

    def _format_shots(
        self, shots: Sequence[GenericActionableGuidelineGuidelineMatchingShot]
    ) -> str:
        return "\n".join(
            f"Example #{i}: ###\n{self._format_shot(shot)}" for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, shot: GenericActionableGuidelineGuidelineMatchingShot) -> str:
        def adapt_event(e: Event) -> JSONSerializable:
            source_map: dict[EventSource, str] = {
                EventSource.CUSTOMER: "user",
                EventSource.CUSTOMER_UI: "frontend_application",
                EventSource.HUMAN_AGENT: "human_service_agent",
                EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT: "ai_agent",
                EventSource.AI_AGENT: "ai_agent",
                EventSource.SYSTEM: "system-provided",
            }

            return {
                "event_kind": e.kind.value,
                "event_source": source_map[e.source],
                "data": e.data,
            }

        formatted_shot = ""
        if shot.interaction_events:
            formatted_shot += f"""
- **Interaction Events**:
{json.dumps([adapt_event(e) for e in shot.interaction_events], indent=2)}

"""
        if shot.guidelines:
            formatted_guidelines = "\n".join(
                f"{i}) Condition {g.condition}. Action: {g.action}"
                for i, g in enumerate(shot.guidelines, start=1)
            )
            formatted_shot += f"""
- **Guidelines**:
{formatted_guidelines}

"""

        formatted_shot += f"""
- **Expected Result**:
```json
{json.dumps(shot.expected_result.model_dump(mode="json", exclude_unset=True), indent=2)}
```
"""

        return formatted_shot

    def _build_prompt(
        self,
        shots: Sequence[GenericActionableGuidelineGuidelineMatchingShot],
    ) -> PromptBuilder:
        guideline_representations = {
            g.id: internal_representation(g) for g in self._guidelines.values()
        }

        guidelines_text = "\n".join(
            f"{i}) Condition: {guideline_representations[g.id].condition}. Action: {guideline_representations[g.id].action}"
            for i, g in self._guidelines.items()
        )

        builder = PromptBuilder(on_build=lambda prompt: self._logger.trace(f"Prompt:\n{prompt}"))

        builder.add_section(
            name="guideline-not-previously-applied-general-instructions",
            template="""
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
Your task is to evaluate the relevance and applicability of a set of provided 'when' conditions to the most recent state of an interaction between yourself (an AI agent) and a user.
You examine the applicability of each guideline under the assumption that the action was not taken yet during the interaction.

A guideline should be marked as applicable if it is relevant to the latest part of the conversation and in particular the most recent customer message. Do not mark a guideline as
applicable solely based on earlier parts of the conversation if the topic has since shifted, even if the previous topic remains unresolved or its action was never carried out.

If the conversation moves from a broader issue to a related sub-issue (a related detail or follow-up within the same overall issue), you should still consider the guideline as applicable
if it is relevant to the sub-issue, as it is part of the ongoing discussion.
In contrast, if the conversation has clearly moved on to an entirely new topic, previous guidelines should not be marked as applicable.
This ensures that applicability is tied to the current context, but still respects the continuity of a discussion when diving deeper into subtopics.

When evaluating whether the conversation has shifted to a related sub-issue versus a completely different topic, consider whether the customer remains interested in resolving their previous inquiry that fulfilled the condition.
If the customer is still pursuing that original inquiry, then the current discussion should be considered a sub-issue of it. Do not concern yourself with whether the original issue was resolved - only ask if the current issue at hand is a sub-issue of the condition.


The exact format of your response will be provided later in this prompt.

""",
            props={},
        )
        builder.add_section(
            name="guideline-matcher-examples-of-not-previously-applied-evaluations",
            template="""
Examples of Guideline Match Evaluations:
-------------------
{formatted_shots}
""",
            props={
                "formatted_shots": self._format_shots(shots),
                "shots": shots,
            },
        )
        builder.add_agent_identity(self._context.agent)
        builder.add_context_variables(self._context.context_variables)
        builder.add_glossary(self._context.terms)
        builder.add_capabilities_for_guideline_matching(self._context.capabilities)
        builder.add_interaction_history(self._context.interaction_history)
        builder.add_staged_tool_events(self._context.staged_events)
        builder.add_section(
            name=BuiltInSection.GUIDELINES,
            template="""
- Guidelines List: ###
{guidelines_text}
###
""",
            props={"guidelines_text": guidelines_text},
            status=SectionStatus.ACTIVE,
        )

        builder.add_section(
            name="guideline-not-previously-applied-output-format",
            template="""
IMPORTANT: Please note there are exactly {guidelines_len} guidelines in the list for you to check.

OUTPUT FORMAT
-----------------
- Specify the applicability of each guideline by filling in the details in the following list as instructed:
```json
{{
    {result_structure_text}
}}
```
""",
            props={
                "result_structure_text": self._format_of_guideline_check_json_description(
                    guideline_representations=guideline_representations
                ),
                "guidelines_len": len(self._guidelines),
            },
        )

        return builder

    def _format_of_guideline_check_json_description(
        self,
        guideline_representations: dict[GuidelineId, GuidelineInternalRepresentation],
    ) -> str:
        result_structure = [
            {
                "guideline_id": i,
                "condition": guideline_representations[g.id].condition,
                "rationale": "<Explanation for why the condition is or isn't met when focusing on the most recent interaction>",
                "applies": "<BOOL>",
            }
            for i, g in self._guidelines.items()
        ]
        result = {"checks": result_structure}
        return json.dumps(result, indent=4)


class GenericActionableGuidelineMatching(GuidelineMatchingStrategy):
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        entity_queries: EntityQueries,
        schematic_generator: SchematicGenerator[GenericActionableGuidelineMatchesSchema],
    ) -> None:
        self._logger = logger
        self._optimization_policy = optimization_policy
        self._entity_queries = entity_queries
        self._schematic_generator = schematic_generator

    @override
    async def create_matching_batches(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> Sequence[GuidelineMatchingBatch]:
        journeys = (
            self._entity_queries.find_journeys_on_which_this_guideline_depends.get(
                guidelines[0].id, []
            )
            if guidelines
            else []
        )

        batches = []

        guidelines_dict = {g.id: g for g in guidelines}
        batch_size = self._get_optimal_batch_size(guidelines_dict)
        guidelines_list = list(guidelines_dict.items())
        batch_count = math.ceil(len(guidelines_dict) / batch_size)

        for batch_number in range(batch_count):
            start_offset = batch_number * batch_size
            end_offset = start_offset + batch_size
            batch = dict(guidelines_list[start_offset:end_offset])
            batches.append(
                self._create_batch(
                    guidelines=list(batch.values()),
                    journeys=journeys,
                    context=GuidelineMatchingContext(
                        agent=context.agent,
                        session=context.session,
                        customer=context.customer,
                        context_variables=context.context_variables,
                        interaction_history=context.interaction_history,
                        terms=context.terms,
                        capabilities=context.capabilities,
                        staged_events=context.staged_events,
                        active_journeys=journeys,
                        journey_paths=context.journey_paths,
                    ),
                )
            )

        return batches

    def _get_optimal_batch_size(self, guidelines: dict[GuidelineId, Guideline]) -> int:
        guideline_n = len(guidelines)

        if guideline_n <= 10:
            return 1
        elif guideline_n <= 20:
            return 2
        elif guideline_n <= 30:
            return 3
        else:
            return 5

    def _create_batch(
        self,
        guidelines: Sequence[Guideline],
        journeys: Sequence[Journey],
        context: GuidelineMatchingContext,
    ) -> GenericActionableGuidelineMatchingBatch:
        return GenericActionableGuidelineMatchingBatch(
            logger=self._logger,
            optimization_policy=self._optimization_policy,
            schematic_generator=self._schematic_generator,
            guidelines=guidelines,
            journeys=journeys,
            context=context,
        )

    @override
    async def transform_matches(
        self,
        matches: Sequence[GuidelineMatch],
    ) -> Sequence[GuidelineMatch]:
        return matches


def _make_event(e_id: str, source: EventSource, message: str) -> Event:
    return Event(
        id=EventId(e_id),
        source=source,
        kind=EventKind.MESSAGE,
        creation_utc=datetime.now(timezone.utc),
        offset=0,
        correlation_id="",
        data={"message": message},
        deleted=False,
    )


example_1_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Hi, I'm planning a trip to Italy next month. What can I do there?",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "That sounds exciting! I can help you with that. Do you prefer exploring cities or enjoying scenic landscapes?",
    ),
    _make_event(
        "34",
        EventSource.CUSTOMER,
        "Can you help me figure out the best time to visit Rome and what to pack?",
    ),
    _make_event(
        "78",
        EventSource.CUSTOMER,
        "Actually I’m also wondering — do I need any special visas or documents as an American citizen?",
    ),
]

example_1_guidelines = [
    GuidelineContent(
        condition="The customer is looking for flight or accommodation booking assistance",
        action="Provide links or suggestions for flight aggregators and hotel booking platforms.",
    ),
    GuidelineContent(
        condition="The customer ask for activities recommendations",
        action="Guide them in refining their preferences and suggest options that match what they're looking for",
    ),
    GuidelineContent(
        condition="The customer asks for logistical or legal requirements.",
        action="Provide a clear answer or direct them to a trusted official source if uncertain.",
    ),
]

example_1_expected = GenericActionableGuidelineMatchesSchema(
    checks=[
        GenericActionableBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer is looking for flight or accommodation booking assistance",
            rationale="There’s no mention of booking logistics like flights or hotels",
            applies=False,
        ),
        GenericActionableBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer ask for activities recommendations",
            rationale="The customer has moved from seeking activity recommendations to asking about legal requirements. Since they are no longer pursuing their original inquiry about activities, this represents a new topic rather than a sub-issue",
            applies=False,
        ),
        GenericActionableBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer asks for logistical or legal requirements.",
            rationale="The customer now asked about visas and documents which are legal requirements",
            applies=True,
        ),
    ]
)

example_2_events = [
    _make_event(
        "21",
        EventSource.CUSTOMER,
        "Hi, I’m interested in your Python programming course, but I’m not sure if I’m ready for it.",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "Happy to help! Could you share a bit about your background or experience with programming so far?",
    ),
    _make_event(
        "32",
        EventSource.CUSTOMER,
        "I’ve done some HTML and CSS, but never written real code before.",
    ),
    _make_event(
        "48",
        EventSource.AI_AGENT,
        "Thanks for sharing! That gives me a good idea. Our Python course is beginner-friendly, but it does assume you're comfortable with logic and problem solving. Would you like me "
        "to recommend a short prep course first?",
    ),
    _make_event(
        "78",
        EventSource.CUSTOMER,
        "That sounds useful. But I’m also wondering — is the course self-paced? I work full time.",
    ),
]

example_2_guidelines = [
    GuidelineContent(
        condition="The customer mentions a constraint that related to commitment to the course",
        action="Emphasize flexible learning options",
    ),
    GuidelineContent(
        condition="The user expresses hesitation or self-doubt.",
        action="Affirm that it’s okay to be uncertain and provide confidence-building context",
    ),
    GuidelineContent(
        condition="The user asks about certification or course completion benefits.",
        action="Clearly explain what the user receives",
    ),
]

example_2_expected = GenericActionableGuidelineMatchesSchema(
    checks=[
        GenericActionableBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer mentions a constraint that related to commitment to the course",
            rationale="In the most recent message the customer mentions that they work full time which is a constraint",
            applies=True,
        ),
        GenericActionableBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The user expresses hesitation or self-doubt.",
            rationale="In the most recent message the user still sounds hesitating about their fit to the course",
            applies=True,
        ),
        GenericActionableBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The user asks about certification or course completion benefits.",
            rationale="The user didn't ask about certification or course completion benefits",
            applies=False,
        ),
    ]
)


example_3_events = [
    _make_event(
        "21",
        EventSource.CUSTOMER,
        "I'm having trouble logging into my account.",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "I'm sorry to hear that. Can you tell me what happens when you try to log in?",
    ),
    _make_event(
        "27",
        EventSource.CUSTOMER,
        "It says my password is incorrect.",
    ),
    _make_event(
        "48",
        EventSource.AI_AGENT,
        "Have you tried resetting your password?",
    ),
    _make_event(
        "78",
        EventSource.CUSTOMER,
        "Yes, I did, but I can't access my mail to complete the reset.",
    ),
]

example_3_guidelines = [
    GuidelineContent(
        condition="When the user is having a problem with login.",
        action="Help then identify the problem and solve it",
    ),
]

example_3_expected = GenericActionableGuidelineMatchesSchema(
    checks=[
        GenericActionableBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="When the user is having a problem with login.",
            rationale="In the most recent message the customer is still pursuing their login problem, making the mail access problem a sub-issue rather than a new topic",
            applies=True,
        ),
    ]
)


example_4_events = [
    _make_event(
        "21",
        EventSource.CUSTOMER,
        "Hi, I'm thinking about ordering this coat, but I need to know — what's your return policy?",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "You can return items within 30 days either in-store or using our prepaid return label.",
    ),
    _make_event(
        "27",
        EventSource.CUSTOMER,
        "And what happens if I already wore it once?",
    ),
]

example_4_guidelines = [
    GuidelineContent(
        condition="When the customer asks about how to return an item.",
        action="Mention both in-store and delivery service return options.",
    ),
]

example_4_expected = GenericActionableGuidelineMatchesSchema(
    checks=[
        GenericActionableBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="When the customer asks about how to return an item.",
            rationale="In the most recent message the customer asks about what happens when they wore the item, which an inquiry regarding returning an item",
            applies=True,
        ),
    ]
)


_baseline_shots: Sequence[GenericActionableGuidelineGuidelineMatchingShot] = [
    GenericActionableGuidelineGuidelineMatchingShot(
        description="",
        interaction_events=example_1_events,
        guidelines=example_1_guidelines,
        expected_result=example_1_expected,
    ),
    GenericActionableGuidelineGuidelineMatchingShot(
        description="",
        interaction_events=example_2_events,
        guidelines=example_2_guidelines,
        expected_result=example_2_expected,
    ),
    GenericActionableGuidelineGuidelineMatchingShot(
        description="",
        interaction_events=example_3_events,
        guidelines=example_3_guidelines,
        expected_result=example_3_expected,
    ),
    GenericActionableGuidelineGuidelineMatchingShot(
        description="",
        interaction_events=example_4_events,
        guidelines=example_4_guidelines,
        expected_result=example_4_expected,
    ),
]

shot_collection = ShotCollection[GenericActionableGuidelineGuidelineMatchingShot](_baseline_shots)
