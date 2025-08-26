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


class GenericPreviouslyAppliedActionableBatch(DefaultBaseModel):
    guideline_id: str
    condition: str
    action: str
    condition_met_again: bool
    action_wasnt_taken: Optional[bool] = None
    should_reapply: bool


class GenericPreviouslyAppliedActionableGuidelineMatchesSchema(DefaultBaseModel):
    checks: Sequence[GenericPreviouslyAppliedActionableBatch]


@dataclass
class GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot(Shot):
    interaction_events: Sequence[Event]
    guidelines: Sequence[GuidelineContent]
    expected_result: GenericPreviouslyAppliedActionableGuidelineMatchesSchema


class GenericPreviouslyAppliedActionableGuidelineMatchingBatch(GuidelineMatchingBatch):
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[
            GenericPreviouslyAppliedActionableGuidelineMatchesSchema
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
                    self._logger.warning("Completion:\nNo checks generated! This shouldn't happen.")
                else:
                    self._logger.trace(
                        f"Completion:\n{inference.content.model_dump_json(indent=2)}"
                    )

                matches = []

                for match in inference.content.checks:
                    if match.should_reapply:
                        self._logger.debug(f"Activated:\n{match.model_dump_json(indent=2)}")

                        matches.append(
                            GuidelineMatch(
                                guideline=self._guidelines[match.guideline_id],
                                score=10 if match.should_reapply else 1,
                                rationale="",
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
    ) -> Sequence[GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot]:
        return await shot_collection.list()

    def _format_shots(
        self, shots: Sequence[GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot]
    ) -> str:
        return "\n".join(
            f"Example #{i}: ###\n{self._format_shot(shot)}" for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(
        self, shot: GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot
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
        shots: Sequence[GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot],
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


Task Description
----------------
You will be given a set of guidelines, each associated with an action that has already been applied one or more times during the conversation.

In general, a guideline should be reapplied if:
1. The condition is met again for a new reason in the most recent user message, and
2. The associated action has not yet been taken in response to this new occurrence, but still needs to be.

Your task is to determine whether reapplying the action is appropriate, based on whether the guideline’s condition is met again in a way that justifies repeating the action. We will want to repeat the action if the current application refers
 to a new or subtly different context or information
For example, a guideline with the condition “the customer is asking a question” should be reapplied each time the customer asks a new question.
In contrast, guidelines involving one-time behaviors (e.g., “send the user our address”) should be reapplied more conservatively: only if the condition ceased to be true for a while and is now clearly true again in the current context.
For instance, if the customer previously complained about an issue and you already offered compensation, then mentions the same issue again, it is usually not necessary to repeat the compensation offer. However, if the customer raises a new
 issue or clearly indicates a different concern, it may warrant reapplying the guideline.

-- Focusing on the most recent context --
When evaluating whether a guideline should be reapplied, the most recent part of the conversation, specifically the last user message, is what matters. A guideline should only be reapplied if its condition is clearly met again in that latest message.
Always base your decision on the current context to avoid unnecessary repetition and to keep the response aligned with the user’s present needs.
Context May Shift:
    Sometimes, the user may briefly raise an issue that would normally trigger a guideline, but then shift the topic within the same message or shortly after. In such cases, the condition should NOT be considered active, and the guideline should
    not be reapplied.
Conditions Can Arise and Resolve Multiple Times:
    A condition may be met more than once over the course of a conversation and may also be resolved multiple times (the action was taken). If the most recent instance of the condition has already been addressed and resolved, there is no need to
    reapply the guideline. However, if the user is still clearly engaging with the same unresolved issue, or if a new instance of the condition arises, reapplying the guideline may be appropriate.


The conversation and guidelines will follow. Instructions on how to format your response will be provided after that.

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
                "condition_met_again": "<BOOL. Whether the condition met again in a new or subtly different context or information>",
                "action_wasnt_taken": "<BOOL. include only condition_met_again is True if The action wasn't already taken for this new reason>",
                "should_reapply": "<BOOL>",
            }
            for i, g in self._guidelines.items()
        ]
        result = {"checks": result_structure}
        return json.dumps(result, indent=4)


class GenericPreviouslyAppliedActionableGuidelineMatching(GuidelineMatchingStrategy):
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        entity_queries: EntityQueries,
        schematic_generator: SchematicGenerator[
            GenericPreviouslyAppliedActionableGuidelineMatchesSchema
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
    ) -> GenericPreviouslyAppliedActionableGuidelineMatchingBatch:
        return GenericPreviouslyAppliedActionableGuidelineMatchingBatch(
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
        "99",
        EventSource.CUSTOMER,
        "Gotcha, and I imagine that if he does try to add me to the household account he won't be able to see that there already is an account, right?",
    ),
]


example_1_guidelines = [
    GuidelineContent(
        condition="the customer initiates a purchase.",
        action="Open a new cart for the customer",
    ),
    GuidelineContent(
        condition="the customer asks about data security",
        action="Refer the customer to our privacy policy page",
    ),
]

example_1_expected = GenericPreviouslyAppliedActionableGuidelineMatchesSchema(
    checks=[
        GenericPreviouslyAppliedActionableBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer initiates a purchase.",
            action="Open a new cart for the customer",
            condition_met_again=False,
            should_reapply=False,
        ),
        GenericPreviouslyAppliedActionableBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer asks about data security",
            action="Refer the customer to our privacy policy page",
            condition_met_again=True,
            action_wasnt_taken=True,
            should_reapply=True,
        ),
    ]
)


example_2_events = [
    _make_event("11", EventSource.CUSTOMER, "Hi there, what is the S&P500 trading at right now?"),
    _make_event("23", EventSource.AI_AGENT, "Hello! It's currently priced at just about 6,000$."),
    _make_event(
        "34",
        EventSource.CUSTOMER,
        "Better than I hoped. And what's the weather looking like today?",
    ),
    _make_event("56", EventSource.AI_AGENT, "It's 5 degrees Celsius in London today"),
    _make_event(
        "78", EventSource.CUSTOMER, "Bummer. Does S&P500 still trade at 6,000$ by the way?"
    ),
]

example_2_guidelines = [
    GuidelineContent(
        condition="the customer asks about the value of a stock.",
        action="provide the price using the 'check_stock_price' tool",
    ),
    GuidelineContent(
        condition="the weather at a certain location is discussed.",
        action="check the weather at that location using the 'check_weather' tool",
    ),
]


example_2_expected = GenericPreviouslyAppliedActionableGuidelineMatchesSchema(
    checks=[
        GenericPreviouslyAppliedActionableBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer asks about the value of a stock.",
            action="provide the price using the 'check_stock_price' tool",
            condition_met_again=True,
            action_wasnt_taken=True,
            should_reapply=True,
        ),
        GenericPreviouslyAppliedActionableBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the weather at a certain location is discussed.",
            action="check the weather at that location using the 'check_weather' tool",
            condition_met_again=False,
            should_reapply=False,
        ),
    ]
)

example_3_events = [
    _make_event("11", EventSource.CUSTOMER, "Can you tell me my current account balance?"),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "Your current account balance is $75.20. Would you like to hear about your recent payments?",
    ),
    _make_event(
        "34",
        EventSource.CUSTOMER,
        "What’s the amount due on my latest bill?",
    ),
    _make_event(
        "56",
        EventSource.AI_AGENT,
        "Your latest bill is $45.60, due on June 15th",
    ),
    _make_event(
        "88",
        EventSource.CUSTOMER,
        "Have I made any payments this month?",
    ),
    _make_event(
        "98",
        EventSource.AI_AGENT,
        "Yes, you made a payment of $30 on May 5th. Can I help with anything else?",
    ),
    _make_event(
        "99",
        EventSource.CUSTOMER,
        "Yes can you provide me your contact details?",
    ),
]

example_3_guidelines = [
    GuidelineContent(
        condition="The customer asks about their account balance, billing amount, or payment status.",
        action="Provide the current account balance or billing information clearly.",
    ),
]

example_3_expected = GenericPreviouslyAppliedActionableGuidelineMatchesSchema(
    checks=[
        GenericPreviouslyAppliedActionableBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer asks about their account balance, billing amount, or payment status.",
            action="Provide the current account balance or billing information clearly.",
            condition_met_again=False,
            should_reapply=False,
        ),
    ]
)

example_4_events = [
    _make_event("11", EventSource.CUSTOMER, "Hi there, what is the S&P500 trading at right now?"),
    _make_event("23", EventSource.AI_AGENT, "Hello! It's currently priced at just about 6,000$."),
    _make_event(
        "34",
        EventSource.CUSTOMER,
        "Better than I hoped. And what's the weather looking like today?",
    ),
    _make_event("56", EventSource.AI_AGENT, "It's 5 degrees Celsius in London today"),
    _make_event(
        "78", EventSource.CUSTOMER, "Bummer. Does S&P500 still trade at 6,000$ by the way?"
    ),
    _make_event("99", EventSource.AI_AGENT, "I checked that for you and it's still on 6000$!"),
    _make_event("111", EventSource.CUSTOMER, "Cool thanks"),
]

example_4_guidelines = [
    GuidelineContent(
        condition="the customer asks about the value of a stock.",
        action="provide the price using the 'check_stock_price' tool",
    ),
    GuidelineContent(
        condition="the weather at a certain location is discussed.",
        action="check the weather at that location using the 'check_weather' tool",
    ),
]


example_4_expected = GenericPreviouslyAppliedActionableGuidelineMatchesSchema(
    checks=[
        GenericPreviouslyAppliedActionableBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer asks about the value of a stock.",
            action="provide the price using the 'check_stock_price' tool",
            condition_met_again=True,
            action_wasnt_taken=False,
            should_reapply=False,
        ),
        GenericPreviouslyAppliedActionableBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the weather at a certain location is discussed.",
            action="check the weather at that location using the 'check_weather' tool",
            condition_met_again=False,
            should_reapply=False,
        ),
    ]
)

_baseline_shots: Sequence[GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot] = [
    GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot(
        description="",
        interaction_events=example_1_events,
        guidelines=example_1_guidelines,
        expected_result=example_1_expected,
    ),
    GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot(
        description="",
        interaction_events=example_2_events,
        guidelines=example_2_guidelines,
        expected_result=example_2_expected,
    ),
    GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot(
        description="",
        interaction_events=example_3_events,
        guidelines=example_3_guidelines,
        expected_result=example_3_expected,
    ),
    GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot(
        description="",
        interaction_events=example_4_events,
        guidelines=example_4_guidelines,
        expected_result=example_4_expected,
    ),
]

shot_collection = ShotCollection[GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot](
    _baseline_shots
)
