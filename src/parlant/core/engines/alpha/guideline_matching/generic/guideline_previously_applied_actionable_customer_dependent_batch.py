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

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import traceback
from typing import Optional, Sequence
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


class GenericPreviouslyAppliedActionableCustomerDependentBatch(DefaultBaseModel):
    guideline_id: str
    condition: str
    action: str
    condition_still_met: bool
    customer_should_reply: Optional[bool] = None
    condition_met_again: Optional[bool] = None
    action_should_reapply: Optional[bool] = None
    action_wasnt_taken: Optional[bool] = None
    tldr: str
    should_apply: bool


class GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema(DefaultBaseModel):
    checks: Sequence[GenericPreviouslyAppliedActionableCustomerDependentBatch]


@dataclass
class GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot(Shot):
    interaction_events: Sequence[Event]
    guidelines: Sequence[GuidelineContent]
    expected_result: GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema


class GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingBatch(
    GuidelineMatchingBatch
):
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[
            GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema
        ],
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
                        if match.should_apply:
                            self._logger.debug(f"Activated:\n{match.model_dump_json(indent=2)}")

                            matches.append(
                                GuidelineMatch(
                                    guideline=self._guidelines[match.guideline_id],
                                    score=10 if match.should_apply else 1,
                                    rationale=match.tldr,
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

    async def shots(
        self,
    ) -> Sequence[GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot]:
        return await shot_collection.list()

    def _format_shots(
        self,
        shots: Sequence[GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot],
    ) -> str:
        return "\n".join(
            f"Example #{i}: ###\n{self._format_shot(shot)}" for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(
        self, shot: GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot
    ) -> str:
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
        shots: Sequence[GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot],
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
            name="guideline-previously-applied-general-instructions",
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

While an action can only instruct the agent to do something, some guidelines may require something from the customer in order to be completed. These are referred to as "customer dependent" guidelines.
For example, the action "get the customer's ID number" requires the agent to ask the customer what's their account number, but the guideline is not fully completed until the user provides it.

Task Description
----------------

Your task is to evaluate whether a set of "customer dependent" guidelines should be applied to the current state of a conversation between an AI agent and a user.

You will be given guidelines where the agent has already performed their part of the action at least once during the interaction. Now you need to determine if each guideline should be reapplied based on the conversation's current state.

A guideline should be applied if either of the following conditions is true:

   1. Incomplete Action: The original condition still holds, the reason that triggered the agent's initial action remains relevant, AND the customer has not yet fulfilled their part of the action. Example: The agent asked for the user's ID, but the user hasn't responded yet, and the conversation is still about accessing their account.
   2. New Context for Same Condition: The condition arises again in a new context, requiring the action to be repeated by both agent and customer. Example: The user switches to asking about a second account, so the agent needs to ask for another ID.

Key Evaluation Rules:

- Avoid Repeating Static Information Requests: Do not reapply guidelines that request static information (ID, name, date of birth) unless there's a genuinely new context. However, if an action combines static and dynamic components (e.g., "ask for name and preferred appointment time"), reapply the guideline when the dynamic component becomes relevant again.

- Focus on Most Recent Context: Base your evaluation primarily on the last user message. A guideline should only be reapplied if its condition is clearly met in that latest message, not based on earlier parts of the conversation.

- Handle Context Shifts: If a user briefly mentions something that would trigger a guideline but then shifts topics within the same message, do NOT consider the condition active.

- Track Resolution Status: If the most recent instance of a condition has been addressed and resolved, there's no need to reapply the guideline. However, if the user is still engaging with an unresolved issue or a new instance arises, reapplication may be appropriate.


""",
            props={},
        )
        builder.add_section(
            name="guideline-matcher-examples-of-previously-applied-evaluations",
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
- Conditions List: ###
{guidelines_text}
###
""",
            props={"guidelines_text": guidelines_text},
            status=SectionStatus.ACTIVE,
        )

        builder.add_section(
            name="guideline-previously-applied-output-format",
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
                    guideline_representations=guideline_representations,
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
                "action": guideline_representations[g.id].action,
                "condition_still_met": "<BOOL, whether the condition that raised the guideline still relevant in the most recent interaction and subject hasn't changed>",
                "customer_should_reply": "<BOOL, include only if condition_still_met=True. whether the customer needs to apply their side of the action>",
                "condition_met_again": "<BOOL, include only if customer_should_reply=False whether the condition is met again in the recent interaction for a new reason and action should be taken again>",
                "action_should_reapply": "<BOOL,  include only if condition_met_again=True. whether the action is not static and should be taken again>",
                "action_wasnt_taken": "<BOOL, include only if action_should_reapply=True, whether the new action wasn't taken yet by the agent or the customer>",
                "tldr": "<str, Explanation for why the guideline should apply in the most recent context>",
                "should_apply": "<BOOL>",
            }
            for i, g in self._guidelines.items()
        ]
        result = {"checks": result_structure}
        return json.dumps(result, indent=4)


class GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatching(
    GuidelineMatchingStrategy
):
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        entity_queries: EntityQueries,
        schematic_generator: SchematicGenerator[
            GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema
        ],
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
    ) -> GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingBatch:
        return GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingBatch(
            logger=self._logger,
            optimization_policy=self._optimization_policy,
            schematic_generator=self._schematic_generator,
            guidelines=guidelines,
            journeys=journeys,
            context=context,
        )


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
        "11", EventSource.CUSTOMER, "I'm planning a trip next month. Any ideas on where to go?"
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "That sounds exciting! What kind of activities do you enjoy — relaxing on the beach, hiking, museums, food tours?",
    ),
    _make_event(
        "44", EventSource.CUSTOMER, "That's a complicated question. I will think and tell you."
    ),
]

example_1_guidelines = [
    GuidelineContent(
        condition="The customer wants recommendations for a trip",
        action="Ask for their preferred activities and recommend accordingly",
    ),
]

example_1_expected = GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema(
    checks=[
        GenericPreviouslyAppliedActionableCustomerDependentBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer wants recommendations for a trip",
            action="Ask for their preferred activities and recommend accordingly",
            condition_still_met=True,
            customer_should_reply=True,
            tldr="The customer should answer what's their preferred activities.",
            should_apply=True,
        ),
    ]
)


example_2_events = [
    _make_event(
        "11", EventSource.CUSTOMER, "I'm planning a trip next month. Any ideas on where to go?"
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "That sounds exciting! What kind of activities do you enjoy — relaxing on the beach, hiking, museums, food tours?",
    ),
    _make_event("25", EventSource.CUSTOMER, "I love hiking and exploring local food scenes."),
]

example_2_guidelines = [
    GuidelineContent(
        condition="The customer wants recommendations for a trip",
        action="Ask for their preferred activities and recommend accordingly",
    ),
]

example_2_expected = GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema(
    checks=[
        GenericPreviouslyAppliedActionableCustomerDependentBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer wants recommendations for a trip",
            action="Ask for their preferred activities and recommend accordingly",
            condition_still_met=True,
            customer_should_reply=False,
            condition_met_again=False,
            tldr="The customer has already answer what's their preferred activities",
            should_apply=False,
        ),
    ]
)

example_3_events = [
    _make_event(
        "11", EventSource.CUSTOMER, "I'm planning a trip next month. Any ideas on where to go?"
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "That sounds exciting! What kind of activities do you enjoy—relaxing on the beach, hiking, museums, food tours?",
    ),
    _make_event("66", EventSource.CUSTOMER, "I love hiking and exploring local food scenes."),
    _make_event(
        "76",
        EventSource.AI_AGENT,
        "Great! You might enjoy a trip to the Pacific Northwest—plenty of trails and great food in Portland and Seattle.",
    ),
    _make_event("89", EventSource.CUSTOMER, "What about a winter trip in Europe?"),
]

example_3_guidelines = [
    GuidelineContent(
        condition="The customer wants recommendations for a trip",
        action="Ask for their preferred activities and recommend accordingly",
    ),
]

example_3_expected = GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema(
    checks=[
        GenericPreviouslyAppliedActionableCustomerDependentBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer wants recommendations for a trip",
            action="Ask for their preferred activities and recommend accordingly",
            condition_still_met=True,
            customer_should_reply=False,
            condition_met_again=True,
            action_should_reapply=True,
            action_wasnt_taken=True,
            tldr="The customer ask about a new trip plan.",
            should_apply=True,
        ),
    ]
)


example_4_events = [
    _make_event(
        "11", EventSource.CUSTOMER, "I'm planning a trip next month. Any ideas on where to go?"
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "That sounds exciting! What kind of activities do you enjoy—relaxing on the beach, hiking, museums, food tours?",
    ),
    _make_event("26", EventSource.CUSTOMER, "I love hiking and exploring local food scenes."),
    _make_event(
        "54",
        EventSource.AI_AGENT,
        "Great! You might enjoy a trip to the Pacific Northwest—plenty of trails and great food in Portland and Seattle.",
    ),
    _make_event("66", EventSource.CUSTOMER, "What about a winter trip in Europe?"),
    _make_event(
        "77",
        EventSource.AI_AGENT,
        "That can be great! What kind of activities would you like to do there?",
    ),
    _make_event("78", EventSource.CUSTOMER, "I will go to France probably"),
]

example_4_guidelines = [
    GuidelineContent(
        condition="The customer wants recommendations for a trip",
        action="Ask for their preferred activities and recommend accordingly",
    ),
]

example_4_expected = GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema(
    checks=[
        GenericPreviouslyAppliedActionableCustomerDependentBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer wants recommendations for a trip",
            action="Ask for their preferred activities and recommend accordingly",
            condition_still_met=True,
            customer_should_reply=True,
            tldr="The customer didn't answer the question.",
            should_apply=True,
        ),
    ]
)

example_5_events = [
    _make_event(
        "11", EventSource.CUSTOMER, "I'm planning a trip next month. Any ideas on where to go?"
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "That sounds exciting! What kind of activities do you enjoy—relaxing on the beach, hiking, museums, food tours?",
    ),
    _make_event("26", EventSource.CUSTOMER, "I love hiking and exploring local food scenes."),
    _make_event(
        "54",
        EventSource.AI_AGENT,
        "Great! You might enjoy a trip to the Pacific Northwest—plenty of trails and great food in Portland and Seattle.",
    ),
    _make_event("66", EventSource.CUSTOMER, "What about a winter trip in Europe?"),
    _make_event(
        "77",
        EventSource.AI_AGENT,
        "That can be great! What kind of activities would you like to do there?",
    ),
    _make_event("78", EventSource.CUSTOMER, "Actually let's stick to the Plan for next month"),
]

example_5_guidelines = [
    GuidelineContent(
        condition="The customer wants recommendations for a trip",
        action="Ask for their preferred activities and recommend accordingly",
    ),
]

example_5_expected = GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema(
    checks=[
        GenericPreviouslyAppliedActionableCustomerDependentBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer wants recommendations for a trip",
            action="Ask for their preferred activities and recommend accordingly",
            condition_still_met=False,
            tldr="The customer regret about the new planning",
            should_apply=False,
        ),
    ]
)


example_6_events = [
    _make_event("11", EventSource.CUSTOMER, "Hi, I need help changing the email on my account."),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "Sure! Could you please provide your account ID so I can verify your identity?",
    ),
    _make_event("26", EventSource.CUSTOMER, "It’s ACC12345."),
    _make_event(
        "54",
        EventSource.AI_AGENT,
        "Thanks! I’ve updated your email.",
    ),
    _make_event("66", EventSource.CUSTOMER, "Also, can you check the last payment on my account?"),
]

example_6_guidelines = [
    GuidelineContent(
        condition="The customer is asking for account-related help",
        action="Ask for their account ID to verify their identity",
    ),
]

example_6_expected = GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema(
    checks=[
        GenericPreviouslyAppliedActionableCustomerDependentBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer is asking for account-related help",
            action="Ask for their account ID to verify their identity",
            condition_still_met=True,
            customer_should_reply=False,
            condition_met_again=True,
            action_should_reapply=False,
            tldr="The customer already provided their account Id",
            should_apply=False,
        ),
    ]
)

_baseline_shots: Sequence[
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot
] = [
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot(
        description="",
        interaction_events=example_1_events,
        guidelines=example_1_guidelines,
        expected_result=example_1_expected,
    ),
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot(
        description="",
        interaction_events=example_2_events,
        guidelines=example_2_guidelines,
        expected_result=example_2_expected,
    ),
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot(
        description="",
        interaction_events=example_3_events,
        guidelines=example_3_guidelines,
        expected_result=example_3_expected,
    ),
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot(
        description="",
        interaction_events=example_4_events,
        guidelines=example_4_guidelines,
        expected_result=example_4_expected,
    ),
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot(
        description="",
        interaction_events=example_5_events,
        guidelines=example_5_guidelines,
        expected_result=example_5_expected,
    ),
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot(
        description="",
        interaction_events=example_6_events,
        guidelines=example_6_guidelines,
        expected_result=example_6_expected,
    ),
]

shot_collection = ShotCollection[
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot
](_baseline_shots)
