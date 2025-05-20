from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
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


class SegmentPreviouslyAppliedRationale(DefaultBaseModel):
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
        schematic_generator: SchematicGenerator[GenericObservationalGuidelineMatchesSchema],
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
            f"GenericGuidelineMatchingBatch: {len(self._guidelines)} guidelines"
        ):
            inference = await self._schematic_generator.generate(
                prompt=prompt,
                hints={"temperature": 0.15},
            )

        if not inference.content.checks:
            self._logger.warning("Completion:\nNo checks generated! This shouldn't happen.")
        else:
            self._logger.debug(f"Completion:\n{inference.content.model_dump_json(indent=2)}")

        matches = []

        for match in inference.content.checks:
            if match.applies:
                self._logger.debug(f"Completion::Activated:\n{match.model_dump_json(indent=2)}")

                matches.append(
                    GuidelineMatch(
                        guideline=self._guidelines[GuidelineId(match.guideline_id)],
                        score=10 if match.applies else 1,
                        rationale=f'''Condition Application: "{match.rationale}"''',
                        guideline_previously_applied=PreviouslyAppliedType("irrelevant"),
                        guideline_is_continuous=True,
                        should_reapply=True,
                    )
                )
            else:
                self._logger.debug(f"Completion::Skipped:\n{match.model_dump_json(indent=2)}")

        return GuidelineMatchingBatchResult(
            matches=matches,
            generation_info=inference.info,
        )

    async def shots(self) -> Sequence[GenericObservationalGuidelineMatchingShot]:
        return await shot_collection.list()

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
        result_structure = [
            {
                "guideline_id": g.id,
                "condition": g.content.condition,
                "rationale": "<Explanation for why the condition is or isn't met>",
                "applies": "<BOOL>",
            }
            for g in self._guidelines.values()
        ]
        conditions_text = "\n".join(
            f"{i}) {g.content.condition}." for i, g in self._guidelines.items()
        )

        builder = PromptBuilder(on_build=lambda prompt: self._logger.debug(f"Prompt:\n{prompt}"))

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

Application Rules:
1. Historical Relevance: Generally, mark a condition as applicable (YES) if it has been satisfied at ANY point during the conversation history, even if not in the most recent messages.

2. Temporal Qualifiers: If a condition contains explicit temporal qualifiers (e.g., "currently discussing," "in the process of," "actively seeking"), evaluate only against the CURRENT state of the conversation.

Example:
- Condition: "the customer is planning a special occasion dinner"
- Mark as YES if: The user mentions celebrating an anniversary, birthday, graduation, or other special event; asks for restaurant recommendations for a "special night"; discusses making reservations for a celebration; or inquires about upscale dining options for an important date.
- Mark as NO if: The user is discussing everyday meal planning, looking for quick casual dining options, or has given no indication of planning any special event-related dining.

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
        builder.add_interaction_history(self._context.interaction_history)
        builder.add_staged_events(self._context.staged_events)
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


class GenericObservationalGuidelineMatching(GuidelineMatchingStrategy):
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[GenericObservationalGuidelineMatchesSchema],
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
    ) -> GenericObservationalGuidelineMatchingBatch:
        return GenericObservationalGuidelineMatchingBatch(
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
        condition="the customer is a senior citizen.",
        action=None,
    ),
    GuidelineContent(
        condition="the customer asks about data security",
        action=None,
    ),
    GuidelineContent(
        condition="our pro plan was discussed or mentioned",
        action=None,
    ),
]

example_1_expected = GenericObservationalGuidelineMatchesSchema(
    checks=[
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer is a senior citizen",
            rationale="there is no indication regarding the customer's age.",
            applies=False,
        ),
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer asks about data security",
            rationale="The customer specifically inquired about data security policies.",
            applies=True,
        ),
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="our pro plan was discussed or mentioned",
            rationale="The customer asked to subscribe to the pro plan",
            applies=True,
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
        condition="food allergies are discussed",
        action=None,
    ),
    GuidelineContent(
        condition="the customer is allergic to almonds",
        action=None,
    ),
    GuidelineContent(
        condition="the conversation is currently about peanut allergies",
        action=None,
    ),
]

example_2_expected = GenericObservationalGuidelineMatchesSchema(
    checks=[
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="food allergies are discussed",
            rationale="nut allergies were discussed earlier in the interaction",
            applies=True,
        ),
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer is allergic to almonds",
            rationale="While the customer has some nut allergies, we do not know if they are for almonds specifically",
            applies=False,
        ),
        GenericObservationalGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the conversation is currently about peanut allergies",
            rationale="peanut allergies were discussed, but the conversation has moved on from the subject",
            applies=False,
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
]

shot_collection = ShotCollection[GenericObservationalGuidelineMatchingShot](_baseline_shots)
