from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from typing import Sequence
from typing_extensions import override
from parlant.core.common import DefaultBaseModel, JSONSerializable
from parlant.core.engines.alpha.guideline_matching.guideline_match import (
    GuidelineMatch,
    PreviouslyAppliedType,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingBatch,
    GuidelineMatchingBatchResult,
    GuidelineMatchingContext,
    GuidelineMatchingStrategy,
)
from parlant.core.engines.alpha.prompt_builder import BuiltInSection, PromptBuilder, SectionStatus
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.shots import Shot, ShotCollection


class GenericPreviouslyAppliedBatch(DefaultBaseModel):
    guideline_id: str
    condition: str
    action: str
    tldr: str
    guideline_should_reapply: bool


class GenericPreviouslyAppliedGuidelineMatchesSchema(DefaultBaseModel):
    checks: Sequence[GenericPreviouslyAppliedBatch]


@dataclass
class GenericPreviouslyAppliedGuidelineGuidelineMatchingShot(Shot):
    interaction_events: Sequence[Event]
    guidelines: Sequence[GuidelineContent]
    expected_result: GenericPreviouslyAppliedGuidelineMatchesSchema


class GenericPreviouslyAppliedGuidelineMatchingBatch(GuidelineMatchingBatch):
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[GenericPreviouslyAppliedGuidelineMatchesSchema],
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> None:
        self._logger = logger
        self._schematic_generator = schematic_generator
        self._guidelines = {g.id: g for g in guidelines}
        self._context = context

    @override
    async def process(self) -> GuidelineMatchingBatchResult:
        prompt = self._build_prompt(shots=await self.shots())

        with self._logger.operation(
            f"GenericPreviouslyAppliedGuidelineMatchingBatch: {len(self._guidelines)} guidelines"
        ):
            inference = await self._schematic_generator.generate(
                prompt=prompt,
                hints={"temperature": 0.15},
            )

        if not inference.content.checks:
            self._logger.warning("Completion:\nNo checks generated! This shouldn't happen.")
        else:
            with open("output_prev_apply_matcher.txt", "a") as f:
                f.write(f"{inference.content.model_dump_json(indent=2)}")
            self._logger.debug(f"Completion:\n{inference.content.model_dump_json(indent=2)}")

        matches = []

        for match in inference.content.checks:
            if match.guideline_should_reapply:
                self._logger.debug(f"Completion::Activated:\n{match.model_dump_json(indent=2)}")

                matches.append(
                    GuidelineMatch(
                        guideline=self._guidelines[GuidelineId(match.guideline_id)],
                        score=10 if match.guideline_should_reapply else 1,
                        rationale=f'''reapply rational: "{match.tldr}"''',
                        guideline_previously_applied=PreviouslyAppliedType.FULLY,
                        guideline_is_continuous=True,
                        should_reapply=match.guideline_should_reapply,
                    )
                )
            else:
                self._logger.debug(f"Completion::Skipped:\n{match.model_dump_json(indent=2)}")

        return GuidelineMatchingBatchResult(
            matches=matches,
            generation_info=inference.info,
        )

    async def shots(self) -> Sequence[GenericPreviouslyAppliedGuidelineGuidelineMatchingShot]:
        return await shot_collection.list()

    def _format_shots(
        self, shots: Sequence[GenericPreviouslyAppliedGuidelineGuidelineMatchingShot]
    ) -> str:
        return "\n".join(
            f"Example #{i}: ###\n{self._format_shot(shot)}" for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, shot: GenericPreviouslyAppliedGuidelineGuidelineMatchingShot) -> str:
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
        shots: Sequence[GenericPreviouslyAppliedGuidelineGuidelineMatchingShot],
    ) -> PromptBuilder:
        guidelines_text = "\n".join(
            f"{i}) Condition: {g.content.condition}. Action: {g.content.action}"
            for i, g in self._guidelines.items()
        )

        builder = PromptBuilder(on_build=lambda prompt: self._logger.debug(f"Prompt:\n{prompt}"))

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
You will be provided with a set of guidelines, each of which has had its action applied one or more times during the conversation.

Your task is to evaluate whether reapplying the action is appropriate, based on whether the guideline’s condition has become true again in a way that justifies reapplying the action after it was previously fulfilled.

For example, a guideline with the condition “the customer is asking a question” should be reapplied each time the customer asks a new question—since this condition can be true multiple times throughout the conversation. 
In contrast, guidelines involving one-time behaviors (e.g., “send the user our address”) should be reapplied more conservatively: only if the condition ceased to be true for a while and is now clearly true again in the current context.

- A guideline should be marked as "should_reapply" if the condition is met again in the MOST RECENT part of the conversation (specifically on the last user message). 
- Do not mark a guideline as applicable solely based on earlier parts of the conversation if the topic has since shifted or resolved. 
- A guideline can be activated and applied multiple times. Pay close attention to when the condition was last met and whether it has become true again since then.

Important clarification on reapplication:
A guideline’s "should_reapply" status depends exclusively on whether the condition holds true for the most recent user message.
Even if the guideline was previously applied multiple times, if the most recent user message does not meet the condition, the guideline should not be reapplied.
This prevents redundant responses and ensures the agent reacts appropriately to the current context rather than lingering past issues.


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
        builder.add_interaction_history(self._context.interaction_history)
        builder.add_staged_events(self._context.staged_events)
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
                "result_structure_text": self._format_of_guideline_check_json_description(),
                "guidelines_len": len(self._guidelines),
            },
        )
        return builder

    def _format_of_guideline_check_json_description(self) -> str:
        result_structure = [
            {
                "guideline_id": g.id,
                "condition": g.content.condition,
                "action": g.content.action,
                "tldr": "<str, Explanation for why the guideline condition is met AGAIN and should reapply when focusing on the MOST RECENT interaction>",
                "guideline_should_reapply": "<BOOL>",
            }
            for g in self._guidelines.values()
        ]
        result = {"checks": result_structure}
        return json.dumps(result, indent=4)


class GenericPreviouslyAppliedGuidelineMatching(GuidelineMatchingStrategy):
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[GenericPreviouslyAppliedGuidelineMatchesSchema],
    ) -> None:
        self._logger = logger
        self._schematic_generator = schematic_generator

    @override
    async def create_batches(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> Sequence[GuidelineMatchingBatch]:
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
                    context=context,
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
        context: GuidelineMatchingContext,
    ) -> GenericPreviouslyAppliedGuidelineMatchingBatch:
        return GenericPreviouslyAppliedGuidelineMatchingBatch(
            logger=self._logger,
            schematic_generator=self._schematic_generator,
            guidelines=guidelines,
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
    _make_event("11", EventSource.CUSTOMER, "Hi, can you tell me what kinds of pizzas you have?"),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "Sure! We offer Margherita, Pepperoni, BBQ Chicken, and Veggie pizzas. Would you like details on any of these?",
    ),
    _make_event(
        "34",
        EventSource.CUSTOMER,
        "Can I customize the toppings on a pizza?",
    ),
    _make_event(
        "56",
        EventSource.AI_AGENT,
        "Currently, customization isn’t supported, but you can choose from our standard menu.",
    ),
    _make_event(
        "88",
        EventSource.CUSTOMER,
        "How long does delivery usually take?",
    ),
    _make_event(
        "98",
        EventSource.AI_AGENT,
        "Delivery usually takes about 30-45 minutes depending on your location.",
    ),
    _make_event(
        "78",
        EventSource.CUSTOMER,
        "Do you accept online payment?",
    ),
]

example_1_guidelines = [
    GuidelineContent(
        condition="The customer asks a question about the service, menu, payment or ordering process",
        action="Politely provide a clear, helpful answer explaining the requested information or options",
    ),
]

example_1_expected = GenericPreviouslyAppliedGuidelineMatchesSchema(
    checks=[
        GenericPreviouslyAppliedBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer asks a question about the service, menu, or ordering process",
            action="Politely provide a clear, helpful answer explaining the requested information or options",
            tldr="The user is asking about payment options which is a part of ordering process.",
            guideline_should_reapply=True,
        ),
    ]
)


example_2_events = [
    _make_event("11", EventSource.CUSTOMER, "The app keeps freezing on my phone."),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "Sorry to hear that! Let’s go through a few troubleshooting steps. First let's try to restart",
    ),
    _make_event(
        "34",
        EventSource.CUSTOMER,
        "Ok how do I do that?",
    ),
    _make_event(
        "56",
        EventSource.AI_AGENT,
        "To restart your phone, press and hold the power button until the restart option appears, then tap 'Restart'. Let me know once you've done that",
    ),
    _make_event(
        "88",
        EventSource.CUSTOMER,
        "Okay, I restarted it. It seems better.",
    ),
    _make_event(
        "98",
        EventSource.AI_AGENT,
        "Great! there is anything else to help you with?",
    ),
    _make_event(
        "78",
        EventSource.CUSTOMER,
        "Actually, it froze just now when I tried to upload a file.",
    ),
    _make_event(
        "98",
        EventSource.AI_AGENT,
        "OK I see, let me check what we can do",
    ),
    _make_event(
        "78",
        EventSource.CUSTOMER,
        "By the way how much a new phone will cost?.",
    ),
]

example_2_guidelines = [
    GuidelineContent(
        condition="The customer is experiencing a technical issue.",
        action="Offer to help troubleshoot the issue.",
    ),
    GuidelineContent(
        condition="The customer reports a technical issue",
        action="Acknowledge the issue and express empathy before proceeding with help.",
    ),
]

example_2_expected = GenericPreviouslyAppliedGuidelineMatchesSchema(
    checks=[
        GenericPreviouslyAppliedBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer is experiencing a technical issue.",
            action="Offer to help troubleshoot the issue.",
            tldr="The customer is facing a new technical issue but the most recent message is about phone price.",
            guideline_should_reapply=False,
        ),
        GenericPreviouslyAppliedBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer reports a technical issue",
            action="Acknowledge the issue and express empathy before proceeding with help.",
            tldr="the most recent message is about phone price.",
            guideline_should_reapply=False,
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
        "78",
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

example_3_expected = GenericPreviouslyAppliedGuidelineMatchesSchema(
    checks=[
        GenericPreviouslyAppliedBatch(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="The customer asks about their account balance, billing amount, or payment status.",
            action="Provide the current account balance or billing information clearly.",
            tldr="The customer last request is not related to balance or payment",
            guideline_should_reapply=False,
        ),
    ]
)

_baseline_shots: Sequence[GenericPreviouslyAppliedGuidelineGuidelineMatchingShot] = [
    GenericPreviouslyAppliedGuidelineGuidelineMatchingShot(
        description="",
        interaction_events=example_1_events,
        guidelines=example_1_guidelines,
        expected_result=example_1_expected,
    ),
    GenericPreviouslyAppliedGuidelineGuidelineMatchingShot(
        description="",
        interaction_events=example_2_events,
        guidelines=example_2_guidelines,
        expected_result=example_2_expected,
    ),
    GenericPreviouslyAppliedGuidelineGuidelineMatchingShot(
        description="",
        interaction_events=example_3_events,
        guidelines=example_3_guidelines,
        expected_result=example_3_expected,
    ),
]

shot_collection = ShotCollection[GenericPreviouslyAppliedGuidelineGuidelineMatchingShot](
    _baseline_shots
)
