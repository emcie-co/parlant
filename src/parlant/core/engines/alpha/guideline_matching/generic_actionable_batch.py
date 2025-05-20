from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from typing import Optional
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


class GenericActionableGuidelineMatchSchema(DefaultBaseModel):
    guideline_id: str
    condition: str
    condition_application_rationale: str
    condition_applies: bool
    action: Optional[str] = None
    guideline_is_continuous: Optional[bool] = None
    capitalize_exact_words_from_action_in_the_explanations_to_avoid_semantic_pitfalls: Optional[
        bool
    ] = None
    guideline_current_application_refers_to_a_new_or_subtly_different_context_or_information: Optional[
        str
    ] = None
    guideline_previously_applied_rationale: Optional[list[SegmentPreviouslyAppliedRationale]] = None
    guideline_previously_applied: Optional[str] = None
    is_missing_part_cosmetic_or_functional: Optional[str] = None
    guideline_should_reapply: Optional[bool] = None
    applies_score: int


class GenericActionableGuidelineMatchesSchema(DefaultBaseModel):
    checks: Sequence[GenericActionableGuidelineMatchSchema]


@dataclass
class GenericActionableGuidelineMatchingShot(Shot):
    interaction_events: Sequence[Event]
    guidelines: Sequence[GuidelineContent]
    expected_result: GenericActionableGuidelineMatchesSchema


class GenericActionableGuidelineMatchingBatch(GuidelineMatchingBatch):
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[GenericActionableGuidelineMatchesSchema],
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
            if (match.applies_score >= 6) and (
                (match.guideline_previously_applied in [None, "no"])
                or match.guideline_should_reapply
            ):
                self._logger.debug(f"Completion::Activated:\n{match.model_dump_json(indent=2)}")

                matches.append(
                    GuidelineMatch(
                        guideline=self._guidelines[GuidelineId(match.guideline_id)],
                        score=match.applies_score,
                        rationale=f'''Condition Application: "{match.condition_application_rationale}"; Guideline Previously Applied: "{"; ".join(
                            [r.rationale for r in match.guideline_previously_applied_rationale]
                        )
                        if match.guideline_previously_applied_rationale
                        else ""}"''',
                        guideline_previously_applied=PreviouslyAppliedType(
                            match.guideline_previously_applied or "no"
                        ),
                        guideline_is_continuous=match.guideline_is_continuous or False,
                        should_reapply=match.guideline_should_reapply or False,
                    )
                )
            else:
                self._logger.debug(f"Completion::Skipped:\n{match.model_dump_json(indent=2)}")

        return GuidelineMatchingBatchResult(
            matches=matches,
            generation_info=inference.info,
        )

    async def shots(self) -> Sequence[GenericActionableGuidelineMatchingShot]:
        return await shot_collection.list()

    def _format_shots(self, shots: Sequence[GenericActionableGuidelineMatchingShot]) -> str:
        return "\n".join(
            f"Example #{i}: ###\n{self._format_shot(shot)}" for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, shot: GenericActionableGuidelineMatchingShot) -> str:
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
                f"{i}) condition: {g.condition}, action: {g.action}"
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
        shots: Sequence[GenericActionableGuidelineMatchingShot],
    ) -> PromptBuilder:
        result_structure = [
            {
                "guideline_id": g.id,
                "condition": g.content.condition,
                "condition_application_rationale": "<Explanation for why the condition is or isn't met>",
                "condition_applies": "<BOOL>",
                "action": g.content.action,
                "guideline_is_continuous": "<BOOL: Optional, only necessary if guideline_previously_applied is true. Specifies whether the action is taken one-time, or is continuous>",
                "capitalize_exact_words_from_action_in_the_explanations_to_avoid_semantic_pitfalls": True,
                "guideline_previously_applied_rationale": [
                    {
                        "action_segment": "<action_segment_description>",
                        "rationale": "<explanation of whether this action segment was already applied; to avoid pitfalls, try to use the exact same words here as the action segment to determine this. use CAPITALS to highlight the same words in the segment as in your explanation>",
                    }
                ],
                "guideline_current_application_refers_to_a_new_or_subtly_different_context_or_information": "<if the guideline DID previously apply, explain here whether or not it needs to re-apply due to it being applicable to new context or information>",
                "guideline_previously_applied": "<str: either 'no', 'partially' or 'fully' depending on whether and to what degree the action was previously preformed>",
                "is_missing_part_cosmetic_or_functional": "<str: only included if guideline_previously_applied is 'partially'. Value is either 'cosmetic' or 'functional' depending on the nature of the missing segment.",
                "guideline_should_reapply": "<BOOL: Optional, only necessary if guideline_previously_applied is not 'no'>",
                "applies_score": "<Relevance score of the guideline between 1 and 10. A higher score indicates that the guideline should be active>",
            }
            for g in self._guidelines.values()
        ]
        guidelines_text = "\n".join(
            f"{i}) Condition: {g.content.condition}. Action: {g.content.action}"
            for i, g in self._guidelines.items()
        )

        builder = PromptBuilder(on_build=lambda prompt: self._logger.debug(f"Prompt:\n{prompt}"))

        builder.add_section(
            name="guideline-matcher-general-instructions",
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
These conditions, along with the interaction details, will be provided later in this message.
For each condition that is met, determine whether its corresponding action should be taken by the agent or if it has already been addressed previously.


Process Description
-------------------
a. Examine Interaction Events: Review the provided interaction events to discern the most recent state of the interaction between the user and the agent.
b. Evaluate Conditions: Assess the entire interaction to determine whether each condition is still relevant and directly fulfilled based on the most recent interaction state.
c. Check for Prior Action: Determine whether the condition has already been addressed, i.e., whether it applied in an earlier state and its corresponding action has already been performed.
d. Guideline Application: A guideline should be applied only if:
    (1) Its condition is currently met and its action has not been performed yet, or
    (2) The interaction warrants re-application of its action (e.g., when a recurring condition becomes true again after previously being fulfilled).

For each provided guideline, return:
    (1) Whether its condition is fulfilled.
    (2) Whether its action needs to be applied at this time. See the following section for more details.


Insights Regarding Guideline re-activation
-------------------
A condition typically no longer applies if its corresponding action has already been executed.
However, there are exceptions where re-application is warranted, such as when the condition is re-applied again. For example, a guideline with the condition "the customer is asking a question" should be applied again whenever the customer asks a question.
Additionally, actions that involve continuous behavior (e.g., "do not ask the user for their age", or guidelines involving the language the agent should use) should be re-applied whenever their condition is met, even if their action was already taken. Mark these guidelines "guideline_is_continuous" in your output.
If a guideline's condition has multiple requirements, mark it as continuous if at least one of them is continuous. Actions like "tell the customer they are pretty and help them with their order" should be marked as continuous, since 'helping them with their order' is continuous.
Actions that forbid certain behaviors are generally considered continuous, as they must be upheld across multiple messages to ensure consistent adherence.

IMPORTANT: guidelines that only require you to say a specific thing are generally not continuous. Once you said the required thing - the guideline is fulfilled.

Conversely, actions dictating one-time behavior (e.g., "send the user our address") should be re-applied more conservatively.
Only re-apply these if the condition ceased to be true earlier in the conversation before being fulfilled again in the current context.

IMPORTANT: Some guidelines include multiple actions. If only a portion of those actions were fulfilled earlier in the conversation, AND the unfulfilled portions aren't functionally important but more "cosmetic" in nature (e.g. like saying thanks or anything that doesn't influence the direction of, or is important to the interaction) then output "fully" for `guideline_previously_applied`, and treat the guideline as though it has been fully executed.
In such cases, re-apply the guideline only if its condition becomes true again later in the conversation, unless it is marked as continuous.

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
- Guidelines list: ###
{guidelines_text}
###
""",
            props={"guidelines_text": guidelines_text},
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


class GenericActionableGuidelineMatching(GuidelineMatchingStrategy):
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[GenericActionableGuidelineMatchesSchema],
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
    ) -> GenericActionableGuidelineMatchingBatch:
        return GenericActionableGuidelineMatchingBatch(
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
        condition="the customer initiates a purchase.",
        action="Open a new cart for the customer",
    ),
    GuidelineContent(
        condition="the customer asks about data security",
        action="Refer the customer to our privacy policy page",
    ),
    GuidelineContent(
        condition="the customer asked to subscribe to our pro plan",
        action="maintain a helpful tone and thank them for shopping at our store",
    ),
]

example_1_expected = GenericActionableGuidelineMatchesSchema(
    checks=[
        GenericActionableGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer initiates a purchase",
            condition_application_rationale="The purchase-related guideline was initiated earlier, but is currently irrelevant since the customer completed the purchase and the conversation has moved to a new topic.",
            condition_applies=False,
            applies_score=3,
        ),
        GenericActionableGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer asks about data security",
            condition_applies=True,
            condition_application_rationale="The customer specifically inquired about data security policies, making this guideline highly relevant to the ongoing discussion.",
            action="Refer the customer to our privacy policy page",
            capitalize_exact_words_from_action_in_the_explanations_to_avoid_semantic_pitfalls=True,
            guideline_current_application_refers_to_a_new_or_subtly_different_context_or_information="While the guideline previously applied to a *different question*, this is a subtly different question, effectively making it a new question, so the guideline needs to apply again for this new question",
            guideline_previously_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="REFER the customer to our privacy policy page",
                    rationale="While the customer has already asked a question to do with data security, and has been REFERRED to the privacy policy page, they now asked another question, so I should tell them once again to refer to the privacy policy page, perhaps stressing it more this time.",
                )
            ],
            guideline_previously_applied="yes",
            guideline_is_continuous=False,
            guideline_should_reapply=True,
            applies_score=9,
        ),
        GenericActionableGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer asked to subscribe to our pro plan",
            condition_applies=True,
            condition_application_rationale="The customer recently asked to subscribe to the pro plan. The conversation is beginning to drift elsewhere, but still deals with the pro plan",
            action="maintain a helpful tone and thank them for shopping at our store",
            capitalize_exact_words_from_action_in_the_explanations_to_avoid_semantic_pitfalls=True,
            guideline_current_application_refers_to_a_new_or_subtly_different_context_or_information="We're still dealing with the same current need and context",
            guideline_previously_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="MAINTAIN a helpful tone",
                    rationale="a helpful tone was MAINTAINED (i.e. held up)",
                ),
                SegmentPreviouslyAppliedRationale(
                    action_segment="THANK them for shopping at our store",
                    rationale="the agent didn't THANK (i.e. say 'thank you') the customer for shopping at our store, making the guideline partially fulfilled. By this, it should be treated as if it was fully followed",
                ),
            ],
            guideline_previously_applied="partially",
            is_missing_part_cosmetic_or_functional="cosmetic",
            guideline_is_continuous=False,
            guideline_should_reapply=False,
            applies_score=6,
        ),
    ]
)

example_2_events = [
    _make_event("11", EventSource.CUSTOMER, "I'm looking for a job, what do you have available?"),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "Hi there! we have plenty of opportunities for you, where are you located?",
    ),
    _make_event("34", EventSource.CUSTOMER, "I'm looking for anything around the bay area"),
    _make_event(
        "56",
        EventSource.AI_AGENT,
        "That's great. We have a number of positions available over there. What kind of role are you interested in?",
    ),
    _make_event(
        "78", EventSource.CUSTOMER, "Anything to do with training and maintaining AI agents"
    ),
]

example_2_guidelines = [
    GuidelineContent(
        condition="the customer indicates that they are looking for a job.",
        action="ask the customer for their location",
    ),
    GuidelineContent(
        condition="the customer asks about job openings.",
        action="emphasize that we have plenty of positions relevant to the customer, and over 10,000 openings overall",
    ),
    GuidelineContent(
        condition="discussing job opportunities.", action="maintain a positive, assuring tone"
    ),
]

example_2_expected = GenericActionableGuidelineMatchesSchema(
    checks=[
        GenericActionableGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer indicates that they are looking for a job.",
            condition_application_rationale="The current discussion is about the type of job the customer is looking for",
            condition_applies=True,
            action="ask the customer for their location",
            guideline_is_continuous=False,
            capitalize_exact_words_from_action_in_the_explanations_to_avoid_semantic_pitfalls=True,
            guideline_current_application_refers_to_a_new_or_subtly_different_context_or_information="No new context here; the customer's location couldn't have changed so quickly",
            guideline_previously_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="ASK the customer for their location",
                    rationale="The agent ASKED for the customer's location earlier in the interaction. There is no need to ASK for it again, as it is already known.",
                )
            ],
            guideline_previously_applied="fully",
            guideline_should_reapply=False,
            applies_score=3,
        ),
        GenericActionableGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer asks about job openings.",
            condition_applies=True,
            condition_application_rationale="the customer asked about job openings, and the discussion still revolves around this request",
            action="emphasize that we have plenty of positions relevant to the customer, and over 10,000 openings overall",
            guideline_is_continuous=False,
            capitalize_exact_words_from_action_in_the_explanations_to_avoid_semantic_pitfalls=True,
            guideline_current_application_refers_to_a_new_or_subtly_different_context_or_information="No new context here",
            guideline_previously_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="EMPHASIZE we have plenty of relevant positions",
                    rationale="The agent already has EMPHASIZED (i.e. clearly stressed) that we have open positions",
                ),
                SegmentPreviouslyAppliedRationale(
                    action_segment="EMPHASIZE we have over 10,000 openings overall",
                    rationale="The agent neglected to EMPHASIZE (i.e. clearly stressed) that we offer 10k openings overall. The means the guideline partially applies and should be treated as if it was fully applied. However, since the customer is narrowing down their search, this point should be EMPHASIZED again to clarify that it still holds true.",
                ),
            ],
            guideline_previously_applied="partially",
            is_missing_part_cosmetic_or_functional="functional",
            guideline_should_reapply=True,
            applies_score=7,
        ),
        GenericActionableGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="discussing job opportunities.",
            condition_applies=True,
            condition_application_rationale="the discussion is about job opportunities that are relevant to the customer, so the condition applies.",
            action="maintain a positive, assuring tone",
            guideline_is_continuous=True,
            capitalize_exact_words_from_action_in_the_explanations_to_avoid_semantic_pitfalls=True,
            guideline_current_application_refers_to_a_new_or_subtly_different_context_or_information="This is a naturally continuous guideline, so the context is always considered 'new' as long as the condition applies",
            guideline_previously_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="MAINTAIN a positive, assuring tone",
                    rationale="The agent's tone is already MAINTAINED (i.e. held up) as positive. But since this action describes a continuous action, the guideline should be re-applied.",
                ),
            ],
            guideline_previously_applied="fully",
            guideline_should_reapply=True,
            applies_score=9,
        ),
    ]
)


example_3_events = [
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

example_3_guidelines = [
    GuidelineContent(
        condition="the customer asks about the value of a stock.",
        action="provide the price using the 'check_stock_price' tool",
    ),
    GuidelineContent(
        condition="the weather at a certain location is discussed.",
        action="check the weather at that location using the 'check_weather' tool",
    ),
    GuidelineContent(
        condition="the customer asked about the weather.",
        action="provide the customer with the temperature and the chances of precipitation",
    ),
]

example_3_expected = GenericActionableGuidelineMatchesSchema(
    checks=[
        GenericActionableGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer asks about the value of a stock.",
            condition_application_rationale="The customer asked what does the S&P500 trade at",
            condition_applies=True,
            action="provide the price using the 'check_stock_price' tool",
            guideline_is_continuous=False,
            capitalize_exact_words_from_action_in_the_explanations_to_avoid_semantic_pitfalls=True,
            guideline_current_application_refers_to_a_new_or_subtly_different_context_or_information="The agent previously PROVIDED the price, but that was several messages ago. The actual price may have drifted since then.",
            guideline_previously_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="PROVIDE the price using the aforementioned tool",
                    rationale="Several messages ago, the agent previously PROVIDED (i.e. gave or reported) the price of that stock following the customer's question, but since the price might have changed since since those several exchanges between the agent and the customer, it should be checked and PROVIDED again.",
                ),
            ],
            guideline_previously_applied="fully",
            guideline_should_reapply=True,
            applies_score=9,
        ),
        GenericActionableGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the weather at a certain location is discussed.",
            condition_application_rationale="while weather was discussed earlier, the conversation have moved on to an entirely different topic (stock prices)",
            condition_applies=False,
            applies_score=3,
        ),
        GenericActionableGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer asked about the weather.",
            condition_application_rationale="The customer asked about the weather earlier, though the conversation has somewhat moved on to a new topic",
            condition_applies=True,
            action="provide the customer with the temperature and the chances of precipitation",
            guideline_is_continuous=False,
            capitalize_exact_words_from_action_in_the_explanations_to_avoid_semantic_pitfalls=True,
            guideline_current_application_refers_to_a_new_or_subtly_different_context_or_information="No new context; a weather prediction doesn't change so frequently as to require updating within 1 less than hour",
            guideline_previously_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="PROVIDE the temperature",
                    rationale="The action segment was fulfilled by PROVIDING (i.e. giving or reporting) the temperature",
                ),
                SegmentPreviouslyAppliedRationale(
                    action_segment="PROVIDE the changes of precipitation",
                    rationale="The agent did not PROVIDE (i.e. giving or reporting) the chances of precipitation. This means the guideline as a whole was only partially applied.",
                ),
            ],
            guideline_previously_applied="partially",
            is_missing_part_cosmetic_or_functional="functional",
            guideline_should_reapply=True,
            applies_score=6,
        ),
    ]
)

example_4_events = [
    _make_event("11", EventSource.CUSTOMER, "Hey there, I'd like to book an appointment please"),
    _make_event("23", EventSource.AI_AGENT, "Hi, sure thing. With whom and at what time?"),
    _make_event("11", EventSource.CUSTOMER, "Great! With John D. please, thank you."),
]

example_4_guidelines = [
    GuidelineContent(
        condition="the customer wants to book an appointment",
        action="get the name of the person they want to meet and the time they want to meet them",
    ),
]

example_4_expected = GenericActionableGuidelineMatchesSchema(
    checks=[
        GenericActionableGuidelineMatchSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            condition="the customer wants to book an appointment",
            condition_application_rationale="The customer has specifically asked to book an appointment",
            condition_applies=True,
            action="get the name of the person they want to meet and the time they want to meet them",
            guideline_is_continuous=False,
            capitalize_exact_words_from_action_in_the_explanations_to_avoid_semantic_pitfalls=True,
            guideline_current_application_refers_to_a_new_or_subtly_different_context_or_information="No new context",
            guideline_previously_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="GET the name of the person they want to meet",
                    rationale="The action segment was fulfilled by GETTING (i.e. clarifying) the person's name",
                ),
                SegmentPreviouslyAppliedRationale(
                    action_segment="GET at what time they want to meet",
                    rationale="The agent did not yet GET (i.e clarify) the time of the appointment. This means the guideline as a whole was only partially applied.",
                ),
            ],
            guideline_previously_applied="partially",
            is_missing_part_cosmetic_or_functional="functional",
            guideline_should_reapply=True,
            applies_score=8,
        ),
    ]
)


_baseline_shots: Sequence[GenericActionableGuidelineMatchingShot] = [
    GenericActionableGuidelineMatchingShot(
        description="",
        interaction_events=example_1_events,
        guidelines=example_1_guidelines,
        expected_result=example_1_expected,
    ),
    GenericActionableGuidelineMatchingShot(
        description="",
        interaction_events=example_2_events,
        guidelines=example_2_guidelines,
        expected_result=example_2_expected,
    ),
    GenericActionableGuidelineMatchingShot(
        description="",
        interaction_events=example_3_events,
        guidelines=example_3_guidelines,
        expected_result=example_3_expected,
    ),
    GenericActionableGuidelineMatchingShot(
        description="",
        interaction_events=example_4_events,
        guidelines=example_4_guidelines,
        expected_result=example_4_expected,
    ),
]

shot_collection = ShotCollection[GenericActionableGuidelineMatchingShot](_baseline_shots)
