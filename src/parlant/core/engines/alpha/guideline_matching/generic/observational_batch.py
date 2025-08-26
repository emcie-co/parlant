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
from parlant.core.engines.alpha.guideline_matching.generic.common import internal_representation
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


class SegmentPreviouslyAppliedActionableRationale(DefaultBaseModel):
    action_segment: str
    rationale: str


class GenericObservationalGuidelineMatchSchema(DefaultBaseModel):
    guideline_id: str
    condition: str
    rationale: str
    applies: bool


class GenericObservationalGuidelineMatchesSchema(DefaultBaseModel):
    checks: Sequence[GenericObservationalGuidelineMatchSchema]


@dataclass
class GenericObservationalGuidelineMatchingShot(Shot):
    interaction_events: Sequence[Event]
    guidelines: Sequence[GuidelineContent]
    expected_result: GenericObservationalGuidelineMatchesSchema


class GenericObservationalGuidelineMatchingBatch(GuidelineMatchingBatch):
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[GenericObservationalGuidelineMatchesSchema],
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

            generation_attempt_temperatures = (
                self._optimization_policy.get_guideline_matching_batch_retry_temperatures(
                    hints={"type": self.__class__.__name__}
                )
            )

            last_generation_exception: Exception | None = None

            for generation_attempt in range(3):
                try:
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
                        if self._match_applies(match):
                            self._logger.debug(f"Activated:\n{match.model_dump_json(indent=2)}")

                            matches.append(
                                GuidelineMatch(
                                    guideline=self._guidelines[match.guideline_id],
                                    score=10 if match.applies else 1,
                                    rationale=f'''Condition Application Rationale: "{match.rationale}"''',
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

    async def shots(self) -> Sequence[GenericObservationalGuidelineMatchingShot]:
        return await shot_collection.list()

    def _match_applies(self, match: GenericObservationalGuidelineMatchSchema) -> bool:
        """This is a separate function to allow overriding in tests and other applications."""
        return match.applies

    def _format_shots(self, shots: Sequence[GenericObservationalGuidelineMatchingShot]) -> str:
        return "\n".join(
            f"Example #{i}: ###\n{self._format_shot(shot)}" for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, shot: GenericObservationalGuidelineMatchingShot) -> str:
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
                f"{i}) {g.condition}" for i, g in enumerate(shot.guidelines, start=1)
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
        shots: Sequence[GenericObservationalGuidelineMatchingShot],
    ) -> PromptBuilder:
        guideline_representations = {
            g.id: internal_representation(g) for g in self._guidelines.values()
        }

        result_structure = [
            {
                "guideline_id": i,
                "condition": guideline_representations[g.id].condition,
                "rationale": "<Explanation for why the condition is or isn't met based on recent interaction>",
                "applies": "<BOOL>",
            }
            for i, g in self._guidelines.items()
        ]
        conditions_text = "\n".join(
            f"{i}) {guideline_representations[g.id].condition}."
            for i, g in self._guidelines.items()
        )

        builder = PromptBuilder(on_build=lambda prompt: self._logger.trace(f"Prompt:\n{prompt}"))

        builder.add_section(
            name="guideline-matcher-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
In our system, the behavior of a conversational AI agent is guided by how the current state of its interaction with a customer (also referred to as "the user") compares to a number of pre-defined conditions:

- "condition": This is a natural-language condition that specifies when a guideline should apply.
          We evaluate each conversation at its current state against these conditions
          to determine which guidelines should inform the agent's next reply.

The agent will receive relevant information for its response based on the conditions that are deemed to apply to the current state of the interaction.

Task Description
----------------
Your task is to evaluate whether each provided condition applies to the current interaction between an AI agent and a user. For each condition, you must determine a binary True/False decision.

Evaluation Criteria:
Evaluate each condition based on its natural meaning and context:

- Current Activity Or State: Conditions about what's happening "now" in the conversation (e.g., "the conversation is about X", "the user asks about Y") apply based on the most recent messages and current topic of discussion.
- Historical Events: Conditions about things that happened during the interaction (e.g., "the user mentioned X", "the customer asked about Y") apply if the event occurred at any point in the conversation.
- Persistent Facts: Conditions about user characteristics or established facts (e.g., "the user is a senior citizen", "the customer has allergies") apply once established, regardless of current discussion topic.

When evaluating current activity or state you should:
- Consider sub issues: Recognize that conversations often evolve naturally within related domains or explore connected subtopics—in these cases, broader thematic conditions may remain applicable.
- Consider topic shifts: When a user previously discussed something that triggered a condition but the conversation has since moved to a different topic or context with no ongoing connection, mark the condition as not applicable.

Key Considerations:
- Use natural language intuition to interpret what each condition is actually asking about.
- Ambiguous phrasing: When a condition's temporal scope is unclear, treat it as a historical event that remains True as long as it was relevant at some point in the interaction.


The exact format of your response will be provided later in this prompt.

""",
            props={},
        )
        builder.add_section(
            name="guideline-matcher-examples-of-condition-evaluations",
            template="""
Examples of Condition Evaluations:
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
- Conditions List: ###
{guidelines_text}
###
""",
            props={"guidelines_text": conditions_text},
            status=SectionStatus.ACTIVE,
        )

        builder.add_section(
            name="guideline-matcher-expected-output",
            template="""
IMPORTANT: Please note there are exactly {guidelines_len} guidelines in the list for you to check.

Expected Output
---------------------------
- Specify the applicability of each guideline by filling in the details in the following list as instructed:

    ```json
    {{
        "checks":
        {result_structure_text}
    }}
    ```""",
            props={
                "result_structure_text": json.dumps(result_structure),
                "result_structure": result_structure,
                "guidelines_len": len(self._guidelines),
            },
        )

        return builder


class ObservationalGuidelineMatching(GuidelineMatchingStrategy):
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        entity_queries: EntityQueries,
        schematic_generator: SchematicGenerator[GenericObservationalGuidelineMatchesSchema],
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
    ) -> GenericObservationalGuidelineMatchingBatch:
        return GenericObservationalGuidelineMatchingBatch(
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
    _make_event("11", EventSource.CUSTOMER, "Can I purchase a subscription to your software?"),
    _make_event("23", EventSource.AI_AGENT, "Absolutely, I can assist you with that right now."),
    _make_event(
        "34", EventSource.CUSTOMER, "Cool, let's go with the subscription for the Pro plan."
    ),
    _make_event(
        "56",
        EventSource.AI_AGENT,
        "Your subscription has been successfully activated. Is there anything else I can help you with?",
    ),
    _make_event(
        "88",
        EventSource.CUSTOMER,
        "Will my son be able to see that I'm subscribed? Or is my data protected?",
    ),
    _make_event(
        "98",
        EventSource.AI_AGENT,
        "If your son is not a member of your same household account, he won't be able to see your subscription. Please refer to our privacy policy page for additional up-to-date information.",
    ),
    _make_event(
        "78",
        EventSource.CUSTOMER,
        "Gotcha, and I imagine that if he does try to add me to the household account he won't be able to see that there already is an account, right?",
    ),
]

example_1_guidelines = [
    GuidelineContent(
        condition="The customer is a senior citizen.",
        action=None,
    ),
    GuidelineContent(
        condition="The customer asks about data security",
        action=None,
    ),
    GuidelineContent(
        condition="Our pro plan is discussed or mentioned",
        action=None,
    ),
]

example_1_expected = GenericObservationalGuidelineMatchesSchema(
    checks=[
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer is a senior citizen",
            rationale="There is no indication regarding the customer's age.",
            applies=False,
        ),
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer asks about data security",
            rationale="The customer asks who can see the account, which is related to data security.",
            applies=True,
        ),
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="our pro plan is discussed or mentioned",
            rationale="Pro plan subscription was discussed and the conversation moved to data security, so it is no longer applicable.",
            applies=False,
        ),
    ]
)

example_2_events = [
    _make_event(
        "11", EventSource.CUSTOMER, "I'm looking for recipe recommendations for a dinner for 5"
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "Sounds good! Are you interested in just entrees or do you need help planning the entire meal and experience?",
    ),
    _make_event(
        "34", EventSource.CUSTOMER, "I have the evening planned, just looking for entrees."
    ),
    _make_event(
        "56",
        EventSource.AI_AGENT,
        "Great. Are there any dietary limitations I should be aware of?",
    ),
    _make_event(
        "88",
        EventSource.CUSTOMER,
        "I have some minor nut allergies",
    ),
    _make_event(
        "98",
        EventSource.AI_AGENT,
        "I see. Should I avoid recipes with all nuts then?",
    ),
    _make_event(
        "78",
        EventSource.CUSTOMER,
        "You can use peanuts. I'm not allergic to those.",
    ),
    _make_event(
        "98",
        EventSource.AI_AGENT,
        "Thanks for clarifying! Are there any particular cuisines or ingredients you'd like to feature in your dinner?",
    ),
    _make_event(
        "78",
        EventSource.CUSTOMER,
        "I'd love something Mediterranean inspired. We all enjoy seafood too if you have any good options.",
    ),
]

example_2_guidelines = [
    GuidelineContent(
        condition="Food allergies are discussed",
        action=None,
    ),
    GuidelineContent(
        condition="The customer is allergic to almonds",
        action=None,
    ),
    GuidelineContent(
        condition="The customer discusses peanut allergies",
        action=None,
    ),
    GuidelineContent(
        condition="The conversation is about recipe recommendations",
        action=None,
    ),
]

example_2_expected = GenericObservationalGuidelineMatchesSchema(
    checks=[
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer discussed about food allergies",
            rationale="Nut allergies were discussed earlier at the conversation",
            applies=True,
        ),
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer is allergic to almonds",
            rationale="While the customer has some nut allergies, we do not know if they are for almonds specifically",
            applies=False,
        ),
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer discusses peanut allergies",
            rationale="Peanut allergies were discussed, but the conversation has moved on from the subject so the it no longer applies.",
            applies=False,
        ),
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer asks about recipe recommendation",
            rationale="The conversation is about preferred foods, which is within the topic of recipe recommendations.",
            applies=True,
        ),
    ]
)

example_3_events = [
    _make_event("11", EventSource.CUSTOMER, "Hi, I'd like to place an order for delivery"),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "Great! I'd be happy to help you with your order. What would you like to order today?",
    ),
    _make_event(
        "34",
        EventSource.CUSTOMER,
        "I'm looking at your pizza menu. Do you have any vegetarian options?",
    ),
    _make_event(
        "56",
        EventSource.AI_AGENT,
        "Absolutely! We have several vegetarian pizzas including Margherita, Veggie Supreme, and Mediterranean. We also have vegan cheese available.",
    ),
    _make_event(
        "88",
        EventSource.CUSTOMER,
        "Perfect! I'll take a large Veggie Supreme pizza.",
    ),
    _make_event(
        "90",
        EventSource.CUSTOMER,
        "Actually, I'm ordering for a party of 6. Do you have any combo deals or discounts for large orders?",
    ),
    _make_event(
        "91",
        EventSource.AI_AGENT,
        "We do! For orders over $50, we offer 15% off. And we have a family deal - 3 large pizzas for $45. Would you like to add more pizzas?",
    ),
    _make_event(
        "92",
        EventSource.CUSTOMER,
        "That family deal sounds great! Can I get two more large pizzas - one pepperoni and one Hawaiian?",
    ),
    _make_event(
        "93",
        EventSource.AI_AGENT,
        "Perfect! So you'll have three large pizzas total with our family deal. Now, what's your delivery address?",
    ),
    _make_event(
        "94",
        EventSource.CUSTOMER,
        "123 Oak Street, apartment 4B. How long will delivery take?",
    ),
]

example_3_guidelines = [
    GuidelineContent(
        condition="the customer requested vegetarian options",
        action=None,
    ),
    GuidelineContent(
        condition="the conversation is about dietary restrictions",
        action=None,
    ),
    GuidelineContent(
        condition="the customer is ordering for multiple people",
        action=None,
    ),
    GuidelineContent(
        condition="discounts are being discussed",
        action=None,
    ),
    GuidelineContent(
        condition="Delivery details are discussed",
        action=None,
    ),
]

example_3_expected = GenericObservationalGuidelineMatchesSchema(
    checks=[
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer requests vegetarian options",
            rationale="The customer asked about vegetarian options earlier in the conversation but now the conversation moved to delivery details.",
            applies=False,
        ),
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the conversation is about dietary restrictions",
            rationale="The conversation has moved from dietary restrictions to delivery details, so it's currently not about dietary restrictions.",
            applies=False,
        ),
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer is ordering for multiple people",
            rationale="The customer mentioned they are ordering for a party of 6 people.",
            applies=True,
        ),
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="discounts are being discussed",
            rationale="Discounts and combo deals were mentioned, but the conversation has moved to delivery logistics.",
            applies=False,
        ),
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="Delivery details are discussed",
            rationale="The most recent messages are about delivery address and timing, which are delivery details.",
            applies=True,
        ),
    ]
)

_baseline_shots: Sequence[GenericObservationalGuidelineMatchingShot] = [
    GenericObservationalGuidelineMatchingShot(
        description="",
        interaction_events=example_1_events,
        guidelines=example_1_guidelines,
        expected_result=example_1_expected,
    ),
    GenericObservationalGuidelineMatchingShot(
        description="",
        interaction_events=example_2_events,
        guidelines=example_2_guidelines,
        expected_result=example_2_expected,
    ),
    GenericObservationalGuidelineMatchingShot(
        description="",
        interaction_events=example_3_events,
        guidelines=example_3_guidelines,
        expected_result=example_3_expected,
    ),
]

shot_collection = ShotCollection[GenericObservationalGuidelineMatchingShot](_baseline_shots)
