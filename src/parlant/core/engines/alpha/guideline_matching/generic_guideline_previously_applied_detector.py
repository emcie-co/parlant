from dataclasses import dataclass
import json
from itertools import chain
from typing import Mapping, Optional, Sequence
from more_itertools import chunked

from parlant.core import async_utils
from parlant.core.agents import Agent
from parlant.core.common import DefaultBaseModel, JSONSerializable
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.generic_actionable_batch import _make_event
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import GuidelineMatchingContext
from parlant.core.engines.alpha.prompt_builder import BuiltInSection, PromptBuilder
from parlant.core.glossary import Term
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.sessions import Event, EventSource, Session
from parlant.core.shots import Shot, ShotCollection
from parlant.core.tools import ToolId


@dataclass
class GuidelinePreviouslyAppliedDetectionResult:
    previously_applied_guidelines: Sequence[Guideline]
    generation_info: GenerationInfo


class SegmentPreviouslyAppliedRationale(DefaultBaseModel):
    action_segment: str
    action_applied_rationale: str


class GuidelinePreviouslyAppliedDetectionSchema(DefaultBaseModel):
    guideline_id: str
    condition: Optional[str] = None
    action: str
    guideline_applied_rationale: Optional[list[SegmentPreviouslyAppliedRationale]] = None
    guideline_applied_degree: Optional[str] = None
    is_missing_part_functional_or_behavioral_rational: Optional[str] = None
    is_missing_part_functional_or_behavioral: Optional[str] = None
    guideline_applied: bool


class GenericGuidelinePreviouslyAppliedDetectorSchema(DefaultBaseModel):
    checks: Sequence[GuidelinePreviouslyAppliedDetectionSchema]


@dataclass
class GenericGuidelinePreviouslyAppliedDetectorShot(Shot):
    interaction_events: Sequence[Event]
    guidelines: Sequence[GuidelineContent]
    expected_result: GenericGuidelinePreviouslyAppliedDetectorSchema


class GenericGuidelinePreviouslyAppliedDetector:
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[GenericGuidelinePreviouslyAppliedDetectorSchema],
    ) -> None:
        self._logger = logger
        self._schematic_generator = schematic_generator
        self._batch_size = 5

    async def process(
        self,
        agent: Agent,
        session: Session,
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        staged_events: Sequence[EmittedEvent],
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        tool_enabled_guideline_matches: dict[GuidelineMatch, list[ToolId]],
    ) -> GuidelinePreviouslyAppliedDetectionResult:
        context = GuidelineMatchingContext(
            agent=agent,
            session=session,
            customer=customer,
            context_variables=context_variables,
            interaction_history=interaction_history,
            terms=terms,
            staged_events=staged_events,
        )

        all_guidelines = [m.guideline for m in ordinary_guideline_matches] + [
            m.guideline for m in tool_enabled_guideline_matches.keys()
        ]

        guideline_batches = list(chunked(all_guidelines, self._batch_size))

        with self._logger.operation(
            f"GenericGuidelinePreviouslyAppliedDetector: {len(all_guidelines)} guidelines "
            f"in {len(guideline_batches)} batches (batch size={self._batch_size})"
        ):
            batch_tasks = [
                self._process_batch(
                    batch,
                    ordinary_guideline_matches,
                    tool_enabled_guideline_matches,
                    context,
                )
                for batch in guideline_batches
            ]

            batch_results = await async_utils.safe_gather(*batch_tasks)

            all_applied_guidelines = list(
                chain.from_iterable(
                    result.previously_applied_guidelines for result in batch_results
                )
            )

            generation_info = (
                batch_results[-1].generation_info if batch_results else GenerationInfo()
            )

            return GuidelinePreviouslyAppliedDetectionResult(
                previously_applied_guidelines=all_applied_guidelines,
                generation_info=generation_info,
            )

    async def _process_batch(
        self,
        batch: Sequence[Guideline],
        ordinary_guideline_matches: Sequence[GuidelineMatch],
        tool_enabled_guideline_matches: Mapping[GuidelineMatch, Sequence[ToolId]],
        context: GuidelineMatchingContext,
    ) -> GuidelinePreviouslyAppliedDetectionResult:
        batch_guideline_ids = {g.id for g in batch}

        batch_ordinary_guidelines = [
            m.guideline for m in ordinary_guideline_matches if m.guideline.id in batch_guideline_ids
        ]

        batch_tool_match_guidelines = {
            m.guideline: tools
            for m, tools in tool_enabled_guideline_matches.items()
            if m.guideline.id in batch_guideline_ids
        }

        guidelines = {g.id: g for g in batch}

        prompt = self._build_prompt(
            shots=await self.shots(),
            guidelines=guidelines,
            ordinary_guideline=batch_ordinary_guidelines,
            tool_match_guidelines=batch_tool_match_guidelines,
            context=context,
        )

        with self._logger.operation(f"GenericGuidelineMatchingBatch: {len(guidelines)} guidelines"):
            inference = await self._schematic_generator.generate(
                prompt=prompt,
                hints={"temperature": 0.15},
            )
        if not inference.content.checks:
            self._logger.warning("Completion:\nNo checks generated! This shouldn't happen.")
        else:
            with open("output_prev_applied_detector.txt", "a") as f:
                f.write(inference.content.model_dump_json(indent=2))
            self._logger.debug(f"Completion:\n{inference.content.model_dump_json(indent=2)}")

        applied = []

        for check in inference.content.checks:
            if check.guideline_applied:
                self._logger.debug(f"Completion::Activated:\n{check.model_dump_json(indent=2)}")
                applied.append(guidelines[GuidelineId(check.guideline_id)])
            else:
                self._logger.debug(f"Completion::Skipped:\n{check.model_dump_json(indent=2)}")

        return GuidelinePreviouslyAppliedDetectionResult(
            previously_applied_guidelines=applied,
            generation_info=inference.info,
        )

    async def shots(self) -> Sequence[GenericGuidelinePreviouslyAppliedDetectorShot]:
        return await shot_collection.list()

    def _format_shots(self, shots: Sequence[GenericGuidelinePreviouslyAppliedDetectorShot]) -> str:
        return "\n".join(
            f"Example #{i}: ###\n{self._format_shot(shot)}" for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, shot: GenericGuidelinePreviouslyAppliedDetectorShot) -> str:
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
                f"{i}) Condition: {g.condition}, Action: {g.action}"
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

    def _add_guideline_matches_section(
        self,
        ordinary_guidelines: Sequence[Guideline],
        tool_guideline: Mapping[Guideline, Sequence[ToolId]],
    ) -> str:
        i = 1
        ordinary_guidelines_list = ""
        if ordinary_guidelines:
            guidelines: list[str] = []
            for g in ordinary_guidelines:
                guideline = f"{i}) Condition: {g.content.condition}. Action: {g.content.action}"
                i += 1
                guidelines.append(guideline)
            ordinary_guidelines_list = "\n".join(guidelines)

        tools_guidelines_list = ""
        if tool_guideline:
            guidelines = []
            for g, tools in tool_guideline.items():
                guideline = f"{i}) Condition: {g.content.condition}. Action: {g.content.action}"
                i += 1
                guidelines.append(guideline)
                guidelines.append("Associated Tools:")
                for j, id in enumerate(tools, start=1):
                    guidelines.append(f"{j}) {id.service_name}:{id.tool_name}")
            tools_guidelines_list = "\n".join(guidelines)
        guidelines_list = ordinary_guidelines_list + tools_guidelines_list
        return f"""
GUIDELINES
---------------------
Those are the guidelines you need to evaluate if they were applied.
Some guidelines have a tool associated with them. Consider those tools to understand the action that is desired.

Guidelines:
###
{guidelines_list}
###
"""

    def _build_prompt(
        self,
        shots: Sequence[GenericGuidelinePreviouslyAppliedDetectorShot],
        guidelines: dict[GuidelineId, Guideline],
        ordinary_guideline: Sequence[Guideline],
        tool_match_guidelines: Mapping[Guideline, Sequence[ToolId]],
        context: GuidelineMatchingContext,
    ) -> PromptBuilder:
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
Your task is to evaluate whether the action specified by each guideline has now been applied. The guideline you are reviewing has not yet been marked as applied, and you need to determine if the latest agent message in the conversation
satisfies its action so the action can now be considered as applied.

1. Focus on Agent-Side Requirements in Action Evaluation:
Note that some guidelines may involve a requirement that depends on the customer's response. For example, an action like "get the customer's card number" requires the agent to ask for this information, and the customer to provide it for full 
completion. In such cases, you should evaluate only the agent’s part of the action. Since evaluation occurs after the agent’s message, the action is considered applied if the agent has done its part (e.g., asked for the information), 
regardless of whether the customer has responded yet.

2. Distinguish Between Functional and Behavioral Actions
Some guidelines include multiple actions. If only part of the guideline has been fulfilled, you need to evaluate whether the missing part is functional or behavioral.

- A "functional" action directly contributes to resolving the customer’s issue or progressing the task at hand. These actions are core to the outcome of the interaction. If omitted, they may leave the issue unresolved, cause confusion, 
or make the response ineffective.
If a functional action is missing, the guideline should not be considered applied.

- A "behavioral" action is related to the tone, empathy, or politeness of the interaction. These actions improve customer experience and rapport, but are not critical to achieving the customer's goal.
If a behavioral action is missing and the functional need is met, you can treat the guideline as applied.

Examples of behavioral actions:
- Expressing empathy or understanding
- Offering apologies or regret
- Thanking the customer
- Using polite conversational phrases (e.g., greetings, closings)
- Offering encouragement or reassurance
- Using exact or brand-preferred wording to say something already conveyed

Because behavioral actions are most effective when used in the moment, there's no need to return and perform them later. Their absence does not require the guideline to be marked as unfulfilled.
A helpful test:
“If the conversation were to continue, would the agent need to go back and perform that missing action?”
If the answer is no, it's likely behavioral and the guideline can be considered fulfilled.
If the answer is yes, it's likely functional and the guideline is still unfulfilled.

3. Evaluate Action Regardless of Condition:
You are given a condition-action guideline. Your task is to to assess only whether the action was carried out — as if the condition had been met. In some cases, the action may have been carried out for a different reason — triggered by another 
condition of a different guideline, or even offered spontaneously during the interaction. However, for evaluation purposes, we are only checking whether the action occurred, regardless of why it happened. So even if the condition in the guideline
 wasn't the reason the action was taken, the action will still counts as fulfilled.

""",
            props={},
        )
        builder.add_section(
            name="guideline-previously-applied-examples",
            template="""
Examples of ...:
-------------------
{formatted_shots}
""",
            props={
                "formatted_shots": self._format_shots(shots),
                "shots": shots,
            },
        )
        builder.add_agent_identity(context.agent)
        builder.add_context_variables(context.context_variables)
        builder.add_glossary(context.terms)
        builder.add_interaction_history(context.interaction_history)
        builder.add_staged_events(context.staged_events)
        builder.add_section(
            name=BuiltInSection.GUIDELINE_DESCRIPTIONS,
            template=self._add_guideline_matches_section(
                ordinary_guideline,
                tool_match_guidelines,
            ),
            props={
                "ordinary_guidelines": ordinary_guideline,
                "tool_guideline": tool_match_guidelines,
            },
        )

        builder.add_section(
            name="guideline-previously-applied-output-format",
            template="""
IMPORTANT: Please note there are exactly {guidelines_len} guidelines in the list for you to check.

OUTPUT FORMAT
-----------------
- Specify if each guideline was applied by filling in the details in the following list as instructed:
```json
{{
    {result_structure_text}
}}
```
""",
            props={
                "result_structure_text": self._format_of_guideline_check_json_description(
                    guidelines=guidelines
                ),
                "guidelines_len": len(guidelines),
            },
        )
        return builder

    def _format_of_guideline_check_json_description(
        self,
        guidelines: dict[GuidelineId, Guideline],
    ) -> str:
        result_structure = [
            {
                "guideline_id": g.id,
                # "condition": g.content.condition,
                "action": g.content.action,
                "guideline_applied_rationale": [
                    {
                        "action_segment": "<action_segment_description>",
                        "action_applied_rationale": "<explanation of whether this action segment (apart from condition) was applied by the agent; to avoid pitfalls, try to use the exact same words here as the action segment to determine this. use CAPITALS to highlight the same words in the segment as in your explanation>",
                    }
                ],
                "guideline_applied_degree": "<str: either 'no', 'partially' or 'fully' depending on whether and to what degree the action was preformed (apart from condition)>",
                "is_missing_part_functional_or_behavioral_rational": "<str: only included if guideline_applied is 'partially'. short explanation of whether the missing part is functional or behavioral.>",
                "is_missing_part_functional_or_behavioral": "<str: only included if guideline_applied is 'partially'.>",
                "guideline_applied": "<bool>",
            }
            for g in guidelines.values()
        ]
        result = {"checks": result_structure}
        return json.dumps(result, indent=4)


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


example_1_expected = GenericGuidelinePreviouslyAppliedDetectorSchema(
    checks=[
        GuidelinePreviouslyAppliedDetectionSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            # condition="the customer initiates a purchase.",
            action="Open a new cart for the customer",
            guideline_previously_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="OPEN a new cart for the customer",
                    action_applied_rationale="No cart was opened",
                )
            ],
            guideline_applied_degree="no",
            guideline_applied=False,
        ),
        GuidelinePreviouslyAppliedDetectionSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            # condition="the customer asks about data security",
            action="Refer the customer to our privacy policy page",
            guideline_previously_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="REFER the customer to our privacy policy page",
                    action_applied_rationale="The customer has been REFERRED to the privacy policy page.",
                )
            ],
            guideline_applied_degree="fully",
            guideline_applied=True,
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
]

example_2_guidelines = [
    GuidelineContent(
        condition="the customer indicates that they are looking for a job.",
        action="ask the customer for their location and what kind of role they are looking for",
    ),
    GuidelineContent(
        condition="the customer asks about job openings.",
        action="emphasize that we have plenty of positions relevant to the customer, and over 10,000 openings overall",
    ),
]

example_2_expected = GenericGuidelinePreviouslyAppliedDetectorSchema(
    checks=[
        GuidelinePreviouslyAppliedDetectionSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            # condition="the customer indicates that they are looking for a job.",
            action="ask the customer for their location and what kind of role they are looking for",
            guideline_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="ASK the customer for their location",
                    action_applied_rationale="The agent ASKED for the customer's location earlier in the interaction.",
                ),
                SegmentPreviouslyAppliedRationale(
                    action_segment="ASK the customer what kind of role they are looking for",
                    action_applied_rationale="The agent ASKED what kind of role they customer is interested in.",
                ),
            ],
            guideline_applied_degree="fully",
            guideline_applied=True,
        ),
        GuidelinePreviouslyAppliedDetectionSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            # condition="the customer asks about job openings.",
            action="emphasize that we have plenty of positions relevant to the customer, and over 10,000 openings overall",
            guideline_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="EMPHASIZE we have plenty of relevant positions",
                    action_applied_rationale="The agent already has EMPHASIZED (i.e. clearly stressed) that we have open positions",
                ),
                SegmentPreviouslyAppliedRationale(
                    action_segment="EMPHASIZE we have over 10,000 openings overall",
                    action_applied_rationale="The agent neglected to EMPHASIZE (i.e. clearly stressed) that we offer 10k openings overall.",
                ),
            ],
            guideline_applied_degree="partially",
            is_missing_part_functional_or_behavioral_rational="overall intention that there are many open position was made clear so using the exact words is behavioral",
            is_missing_part_functional_or_behavioral="behavioral",
            guideline_applied=True,
        ),
    ]
)


example_3_events = [
    _make_event("11", EventSource.CUSTOMER, "I'm looking for a job, what do you have available?"),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "Hi there! we have plenty of opportunities for you, where are you located?",
    ),
]

example_3_guidelines = [
    GuidelineContent(
        condition="the customer indicates that they are looking for a job.",
        action="ask the customer for their location and what kind of role they are looking for",
    ),
]

example_3_expected = GenericGuidelinePreviouslyAppliedDetectorSchema(
    checks=[
        GuidelinePreviouslyAppliedDetectionSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            # condition="the customer indicates that they are looking for a job.",
            action="ask the customer for their location and what kind of role they are looking for",
            guideline_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="ASK the customer for their location",
                    action_applied_rationale="The agent ASKED for the customer's location earlier in the interaction.",
                ),
                SegmentPreviouslyAppliedRationale(
                    action_segment="ASK the customer what kind of role they are looking for",
                    action_applied_rationale="The agent did not ASK what kind of role the customer is interested in.",
                ),
            ],
            guideline_applied_degree="partially",
            is_missing_part_functional_or_behavioral_rational="Need to ask for the kind of role so can narrow the option and help the customer find the right job fit",
            is_missing_part_functional_or_behavioral="functional",
            guideline_applied=False,
        ),
    ]
)


example_4_events = [
    _make_event("11", EventSource.CUSTOMER, "My screen is frozen and nothing's responding."),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "No problem — I can help reset your password for you. Let me guide you through it.",
    ),
]

example_4_guidelines = [
    GuidelineContent(
        condition="the customer says they forgot their password",
        action="Offer to reset the password and guide them through the process",
    ),
]

example_4_expected = GenericGuidelinePreviouslyAppliedDetectorSchema(
    checks=[
        GuidelinePreviouslyAppliedDetectionSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            # condition="the customer says they forgot their password",
            action="Offer to reset the password.",
            guideline_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="OFFER to reset the password",
                    action_applied_rationale="The agent indeed OFFERED to reset the password.",
                ),
            ],
            guideline_applied_degree="fully",
            guideline_applied=True,
        ),
    ]
)


example_5_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "I've been waiting 40 minutes for my order and it still hasn’t arrived.",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "I'm really sorry about the delay. We’re checking with the delivery partner right now and will update you shortly.",
    ),
]

example_5_guidelines = [
    GuidelineContent(
        condition="there is a problem with the order",
        action="Acknowledge the issue and thank the user for their patience.",
    ),
]

example_5_expected = GenericGuidelinePreviouslyAppliedDetectorSchema(
    checks=[
        GuidelinePreviouslyAppliedDetectionSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            # condition="there is a problem with the order",
            action="Acknowledge the issue and thank the user for their patience.",
            guideline_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="ACKNOWLEDGE the issue",
                    action_applied_rationale="The agent ACKNOWLEDGED the issue by saying they are checking it",
                ),
                SegmentPreviouslyAppliedRationale(
                    action_segment="THANK the user for their patience.",
                    action_applied_rationale="The agent didn't thank the customer for their patient",
                ),
            ],
            guideline_applied_degree="partially",
            is_missing_part_functional_or_behavioral_rational="missing part is about tone and politeness, and doesn’t affect the quality of solving the issue."
            "There’s no need to return and thank the user later in order to complete the response.",
            is_missing_part_functional_or_behavioral="behavioral",
            guideline_applied=True,
        ),
    ]
)


example_6_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "I've been waiting 40 minutes for my order and it still hasn’t arrived.",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "I'm really sorry about the inconvenience. We’re checking with the delivery partner right now and will update you shortly. Any way, let me give you a refund of $20",
    ),
]

example_6_guidelines = [
    GuidelineContent(
        condition="The customer reports that a product arrived damaged",
        action="Offer a $20 refund on the purchase.",
    ),
]

example_6_expected = GenericGuidelinePreviouslyAppliedDetectorSchema(
    checks=[
        GuidelinePreviouslyAppliedDetectionSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            # condition="The customer reports that a product arrived damaged",
            action="Offer a $20 refund on the purchase.",
            guideline_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="OFFER a $20 refund on the purchase.",
                    action_applied_rationale="The agent OFFERED $20 refund for the delay, although not for damaged item.",
                ),
            ],
            guideline_applied_degree="fully",
            guideline_applied=True,
        ),
    ]
)

example_7_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "OK I don't need any other help.",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "Great I was happy to help you, bye bye!",
    ),
]

example_7_guidelines = [
    GuidelineContent(
        condition="The customer said they don't need any other help",
        action="Wish the customer a great day at the end of the interaction by saying goodbye.",
    ),
]

example_7_expected = GenericGuidelinePreviouslyAppliedDetectorSchema(
    checks=[
        GuidelinePreviouslyAppliedDetectionSchema(
            guideline_id=GuidelineId("<example-id-for-few-shots--do-not-use-this-in-output>"),
            # condition="The customer said they don't need any other help",
            action="Wish the customer a great day at the end of the interaction.",
            guideline_applied_rationale=[
                SegmentPreviouslyAppliedRationale(
                    action_segment="Wish the customer a great day",
                    action_applied_rationale="The agent didn't WISH a great day",
                ),
                SegmentPreviouslyAppliedRationale(
                    action_segment="END of the interaction.",
                    action_applied_rationale="The agent END the interaction by saying goodbye.",
                ),
            ],
            guideline_applied_degree="partially",
            is_missing_part_functional_or_behavioral_rational="missing part is about politeness, and doesn’t affect the quality of the interaction",
            is_missing_part_functional_or_behavioral="behavioral",
            guideline_applied=True,
        ),
    ]
)

_baseline_shots: Sequence[GenericGuidelinePreviouslyAppliedDetectorShot] = [
    GenericGuidelinePreviouslyAppliedDetectorShot(
        description="",
        interaction_events=example_1_events,
        guidelines=example_1_guidelines,
        expected_result=example_1_expected,
    ),
    GenericGuidelinePreviouslyAppliedDetectorShot(
        description="",
        interaction_events=example_2_events,
        guidelines=example_2_guidelines,
        expected_result=example_2_expected,
    ),
    GenericGuidelinePreviouslyAppliedDetectorShot(
        description="",
        interaction_events=example_3_events,
        guidelines=example_3_guidelines,
        expected_result=example_3_expected,
    ),
    GenericGuidelinePreviouslyAppliedDetectorShot(
        description="",
        interaction_events=example_4_events,
        guidelines=example_4_guidelines,
        expected_result=example_4_expected,
    ),
    GenericGuidelinePreviouslyAppliedDetectorShot(
        description="",
        interaction_events=example_5_events,
        guidelines=example_5_guidelines,
        expected_result=example_5_expected,
    ),
    GenericGuidelinePreviouslyAppliedDetectorShot(
        description="",
        interaction_events=example_6_events,
        guidelines=example_6_guidelines,
        expected_result=example_6_expected,
    ),
    GenericGuidelinePreviouslyAppliedDetectorShot(
        description="",
        interaction_events=example_7_events,
        guidelines=example_7_guidelines,
        expected_result=example_7_expected,
    ),
]

shot_collection = ShotCollection[GenericGuidelinePreviouslyAppliedDetectorShot](_baseline_shots)
