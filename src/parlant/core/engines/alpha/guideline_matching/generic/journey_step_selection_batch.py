from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import traceback
from typing import Any, Optional, cast
from typing_extensions import override
from parlant.core.common import DefaultBaseModel, JSONSerializable

from parlant.core.engines.alpha.guideline_matching.guideline_match import (
    GuidelineMatch,
    PreviouslyAppliedType,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingBatch,
    GuidelineMatchingBatchError,
    GuidelineMatchingBatchResult,
    GuidelineMatchingContext,
)
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.journeys import Journey
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.shots import Shot, ShotCollection


ELSE_CONDITION_STR = "This step was completed, and no other transition applies"
SINGLE_FOLLOW_UP_CONDITION_STR = "This step was completed"


class _JourneyStepWrapper(DefaultBaseModel):
    id: str
    guideline_content: GuidelineContent
    parent_ids: list[str]
    follow_up_ids: list[str]
    customer_dependent_action: bool
    requires_tool_calls: bool
    conditions: Optional[Sequence[str]] = None


class JourneyStepAdvancement(DefaultBaseModel):
    id: str
    completed: bool
    follow_ups: Optional[list[str]] = (
        None  # TODO work into the journey presentation instead of an ARQ
    )


class JourneyStepSelectionSchema(DefaultBaseModel):
    journey_applies: bool
    rationale: str
    requires_backtracking: bool
    backtracking_target_step: Optional[str] | None = ""
    step_advancement: Optional[Sequence[JourneyStepAdvancement]] = None
    next_step: str


@dataclass
class JourneyStepSelectionShot(Shot):
    interaction_events: Sequence[Event]
    journey_title: str
    journey_steps: dict[str, _JourneyStepWrapper] | None
    previous_path: Sequence[str | None]
    expected_result: JourneyStepSelectionSchema
    conditions: Sequence[str]


def get_journey_transition_map_text(
    steps: dict[str, _JourneyStepWrapper],
    journey_title: str,
    journey_conditions: Sequence[str] = [],
    previous_path: Sequence[str | None] = [],
) -> str:
    def step_sort_key(step_id: str) -> Any:
        try:
            return int(step_id)
        except Exception:
            return step_id

    if journey_conditions:
        journey_conditions_str = " OR ".join(f'"{condition}"' for condition in journey_conditions)
        journey_conditions_str = f"\nJourney activation condition: {journey_conditions_str}\n"
    else:
        journey_conditions_str = ""
    # Sort steps by step id as integer if possible, else as string
    steps_str = ""
    for step_id in sorted(steps.keys(), key=step_sort_key):
        step: _JourneyStepWrapper = steps[step_id]
        action: str | None = step.guideline_content.action
        if action:
            flags_str = "Step Flags:\n"
            if step.customer_dependent_action:
                flags_str += "- CUSTOMER_DEPENDENT: This action is completed if the customer provided a response to this step's action\n"
            if (
                step.requires_tool_calls and (not previous_path or step.id != previous_path[-1])
            ):  # Not including this flag for current step - if we got here, the tool call should've executed so the flag would be misleading
                flags_str += (
                    "- REQUIRES_TOOL_CALLS: Do not advance past this step! If you got here, stop.\n"
                )

            if previous_path and step.id == previous_path[-1]:
                flags_str += (
                    "- This is the last step that was executed. Begin advancing on from this step\n"
                )
            elif step.id in previous_path:
                flags_str += "- PREVIOUSLY_EXECUTED: This step was previously executed. You may backtrack to this step.\n"
            else:
                flags_str += "- NOT_PREVIOUSLY_EXECUTED: This step was not previously executed. You may not backtrack to this step.\n"

            if len(step.follow_up_ids) == 0:
                follow_ups_str = """↳ If "this step is completed",  → RETURN 'NONE'"""
            elif len(step.follow_up_ids) == 1:
                follow_ups_str = f"""↳ If "{steps[step.follow_up_ids[0]].guideline_content.condition or SINGLE_FOLLOW_UP_CONDITION_STR}" → Go to step {step.follow_up_ids[0] if steps[step.follow_up_ids[0]].guideline_content.action else "EXIT JOURNEY, RETURN 'NONE'"}"""
            else:
                follow_ups_str = "\n".join(
                    [
                        f"""↳ If "{steps[follow_up_id].guideline_content.condition or ELSE_CONDITION_STR}" → {"Go to step " + follow_up_id if steps[follow_up_id].guideline_content.action else "EXIT JOURNEY, RETURN 'NONE'"}"""
                        for follow_up_id in step.follow_up_ids
                    ]
                )
            steps_str += f"""
STEP {step_id}: {action}
{flags_str}
TRANSITIONS:
{follow_ups_str}
"""
    return f"""
Journey: {journey_title}
{journey_conditions_str}
Steps:
{steps_str} 
"""


class GenericJourneyStepSelectionBatch(GuidelineMatchingBatch):
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[JourneyStepSelectionSchema],
        examined_journey: Journey,
        context: GuidelineMatchingContext,
        step_guidelines: Sequence[Guideline] = [],
        journey_path: Sequence[str | None] = [],
        journey_conditions: Sequence[str] = [],
    ) -> None:
        self._logger = logger
        self._optimization_policy = optimization_policy
        self._schematic_generator = schematic_generator

        self._step_guideline_mapping = {
            str(cast(dict[str, JSONSerializable], g.metadata["journey_step"])["id"]): g
            for g in step_guidelines
        }

        self._guideline_to_step_id_mapping = {
            g.id: str(cast(dict[str, JSONSerializable], g.metadata["journey_step"])["id"])
            for g in step_guidelines
        }

        self._guideline_ids = {g.id: g for g in step_guidelines}

        self._context = context
        self._examined_journey = examined_journey

        self._journey_steps: dict[str, _JourneyStepWrapper] = self._build_journey_steps()
        self._previous_path: Sequence[str | None] = journey_path
        self._journey_conditions: Sequence[str] = journey_conditions

    def _build_journey_steps(  # TODO rewrite entirely
        self,
    ) -> dict[str, _JourneyStepWrapper]:
        journey_steps_dict: dict[str, _JourneyStepWrapper] = {
            self._guideline_to_step_id_mapping[guideline.id]: _JourneyStepWrapper(
                id=step_id,
                guideline_content=guideline.content,
                parent_ids=[],
                follow_up_ids=[
                    self._guideline_to_step_id_mapping[guideline_id]
                    for guideline_id in cast(
                        Sequence[GuidelineId],
                        cast(dict[str, JSONSerializable], guideline.metadata["journey_step"]).get(
                            "sub_steps", []
                        ),
                    )
                ],
                customer_dependent_action=cast(
                    dict[str, bool],
                    guideline.metadata["customer_dependent_action_data"],
                )["is_customer_dependent"]
                if "is_customer_dependent"
                in cast(
                    dict[str, bool],
                    guideline.metadata.get("customer_dependent_action_data", {}),
                )
                else False,
                requires_tool_calls=cast(bool, guideline.metadata["tool_running_only"]),
            )
            for step_id, guideline in self._step_guideline_mapping.items()
        }

        for id, js in journey_steps_dict.items():
            for followup_id in js.follow_up_ids:
                journey_steps_dict[followup_id].parent_ids.append(id)

        return journey_steps_dict

    @override
    async def process(self) -> GuidelineMatchingBatchResult:
        prompt = self._build_prompt(shots=await self.shots())

        with self._logger.operation(f"JourneyStepSelectionBatch: {self._examined_journey.title}"):
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

                    with open("journey step selection output.txt", "w") as f:
                        f.write(inference.content.model_dump_json(indent=2))
                        f.write("\nTime: " + str(inference.info.duration))

                    self._logger.trace(
                        f"Completion:\n{inference.content.model_dump_json(indent=2)}"
                    )

                    journey_path = self._get_verified_step_advancement(inference.content)
                    return GuidelineMatchingBatchResult(
                        matches=[
                            GuidelineMatch(
                                guideline=self._step_guideline_mapping[inference.content.next_step],
                                score=10,
                                rationale=inference.content.rationale,
                                guideline_previously_applied=PreviouslyAppliedType.IRRELEVANT,
                                metadata={
                                    "journey_path": journey_path,
                                    "step_selection_journey_id": self._examined_journey.id,
                                },
                            )
                        ]
                        if inference.content.next_step
                        in self._journey_steps.keys()  # If either 'None' or an illegal step was returned, don't activate guidelines
                        else [],
                        generation_info=inference.info,
                    )
                except Exception as exc:
                    self._logger.warning(
                        f"JourneyStepSelectionBatch attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                    )

                    last_generation_exception = exc

            raise GuidelineMatchingBatchError() from last_generation_exception

    async def shots(self) -> Sequence[JourneyStepSelectionShot]:
        return await shot_collection.list()

    def _format_shots(self, shots: Sequence[JourneyStepSelectionShot]) -> str:
        return "\n".join(
            f"Example #{i}: {shot.journey_title}\n{self._format_shot(shot)}"
            for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, shot: JourneyStepSelectionShot) -> str:
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
        if shot.journey_steps:
            formatted_shot += get_journey_transition_map_text(
                shot.journey_steps,
                previous_path=shot.previous_path,
                journey_title=shot.journey_title,
                journey_conditions=shot.conditions,
            )

        formatted_shot += f"""
- **Expected Result**:
```json
{json.dumps(shot.expected_result.model_dump(mode="json", exclude_unset=True), indent=2)}
```
"""

        return formatted_shot

    def _get_verified_step_advancement(
        self, response: JourneyStepSelectionSchema
    ) -> list[str | None]:
        journey_path: list[str | None] = []
        for i, advancement in enumerate(response.step_advancement or []):
            journey_path.append(advancement.id)
            if (
                i > 0
                and advancement.id in self._journey_steps
                and self._journey_steps[advancement.id].requires_tool_calls
            ):
                break  # Don't continue past tool calling step

        if (
            response.requires_backtracking and journey_path
        ):  # Warnings related to backtracking to illegal step
            if journey_path[0] != response.backtracking_target_step:
                self._logger.warning(
                    f"WARNING: Illegal journey path returned by journey step selection. Reported that it should return to step {response.backtracking_target_step}, but step advancement began at {journey_path[0]}"
                )
            if response.backtracking_target_step not in self._previous_path:
                self._logger.warning(
                    f"WARNING: Illegal journey path returned by journey step selection. backtracked to {response.backtracking_target_step}, which was never previously visited! Previously visited step IDs: {self._previous_path}"
                )
        elif (
            self._previous_path
            and self._previous_path[-1]
            and journey_path
            and journey_path[0] != self._previous_path[-1]
        ):  # Illegal first step returned
            self._logger.warning(
                f"WARNING: Illegal journey path returned by journey step selection. Expected path from {self._previous_path} to {journey_path}"
            )
            journey_path.insert(0, self._previous_path[-1])  # Try to recover

        indexes_to_delete: list[int] = []
        for i in range(1, len(journey_path)):  # Verify all transitions are legal
            if journey_path[i - 1] not in self._journey_steps.keys():
                self._logger.warning(
                    f"WARNING: Illegal journey path returned by journey step selection. Illegal step returned: {journey_path[i-1]}. Full path: : {journey_path}"
                )
                indexes_to_delete.append(i)
            elif journey_path[i] not in self._journey_steps[str(journey_path[i - 1])].follow_up_ids:
                self._logger.warning(
                    f"WARNING: Illegal transition in journey path returned by journey step selection - from {journey_path[i-1]} to {journey_path[i]}. Full path: : {journey_path}"
                )
                # Sometimes, the LLM returns a path that would've been legal if it were not for an out-of-place step. This deletes such steps.
                if (
                    i + 1 < len(journey_path)
                    and journey_path[i + 1]
                    in self._journey_steps[str(journey_path[i - 1])].follow_up_ids
                ):
                    indexes_to_delete.append(i)
        if (
            journey_path and journey_path[-1] not in self._journey_steps.keys()
        ):  # 'Exit journey' was selected, or illegal value returned (both should cause no guidelines to be active)
            self._logger.warning(
                f"WARNING: Last journey step in returned path is not legal. Full path: : {journey_path}"
            )
            journey_path[-1] = None

        for i in reversed(indexes_to_delete):
            del journey_path[i]
        return journey_path

    def _build_prompt(
        self,
        shots: Sequence[JourneyStepSelectionShot],
    ) -> PromptBuilder:
        builder = PromptBuilder(on_build=lambda prompt: self._logger.debug(f"Prompt:\n{prompt}"))

        builder.add_section(
            name="journey-step-selection-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-------------------
You are an AI agent named {agent_name} whose role is to engage in multi-turn conversations with customers on behalf of a business. 
Your interactions are structured around predefined "journeys" - systematic processes that guide customer conversations toward specific outcomes. 

## Journey Structure
Each journey consists of:
- **Steps**: Individual actions you must take (e.g., ask a question, provide information, perform a task)
- **Transitions**: Rules that determine which step comes next based on customer responses or completion status
- **Flags**: Special properties that modify how steps behave

## Your Core Task
Analyze the current conversation state and determine the next appropriate journey step, based on the last step that was performed and the current state of the conversation.  
""",
            props={"agent_name": self._context.agent.name},
        )
        builder.add_section(
            name="journey-step-selection-task_description",
            template="""
TASK DESCRIPTION
-------------------
Follow this process to determine the next journey step. Document each decision in the specified output format.

## 1: Journey Context Check
Determine if the conversation should continue within the current journey. 
Once a journey has begun, continue following it unless the customer explicitly indicates they no longer want to pursue the journey's original goal.

Set journey_applies to true unless the customer explicitly requests to leave the topic or abandon the journey's goal entirely.
The journey condition is for initial activation - once activated, continue even if individual steps seem unrelated to the original condition.
The journey still applies when the customer is responding to questions, engaging with the journey flow, or providing information requested by previous steps, even if their responses seem tangential to the original condition
Only set journey_applies to false if the customer clearly states they want to exit (e.g., "I don't want to reset my password anymore" or "Let's talk about something else")
If journey_applies is false, set next_step to 'None' and skip remaining steps

CRITICAL: If you are already executing journey steps (i.e., there is a "last_step"), the journey almost always continues. The activation condition is ONLY for starting new journeys, NOT for validating ongoing ones.

## 2: Backtracking Check  
Check if the customer has changed a previous decision that requires returning to an earlier step.
- Set `requires_backtracking` to `true` if the customer contradicts or changes a prior choice
- If backtracking is needed:
  - Set backtracking_target_step to the step where the decision changed. This step must have the PREVIOUSLY_VISITED flag.
  - Continue to step 4 (Journey Advancement) but treat the backtracking_target_step as your starting point instead of last_step
  - The advancement should begin from the backtracking target step and continue following the normal advancement rules until you reach a step that cannot be completed

## 3: Current Step Completion
Evaluate whether the last executed step is complete.
- For steps with CUSTOMER_DEPENDENT flag: step is complete if the customer has provided the information that the step was seeking. If the step asks for specific information and the customer has provided that information (even in a previous message), the step can be considered completed and advanced through.
- If the last step is incomplete, set next_step to the current step ID (repeat the step) and document this in the step_advancement array.

## 4: Journey Advancement
Starting from the last executed step, advance through subsequent steps, documenting each step's completion status in the step_advancement array. Continue advancing until you encounter:
- A step requiring a tool call (REQUIRES_TOOL_CALLS flag)
- A step where you lack necessary information to proceed
- A step requiring you to communicate something new to the customer, beyond asking them for information

For each step in the advancement path:
- If the step can be completed based on available information, mark completed: true
- If the step cannot be completed (missing information, requires tool calls, etc.), mark completed: false
- Only advance to the next step if the current step is marked as completed

Document your advancement path in step_advancement as a list of step advancement objects, starting with the last_step and ending with the next step to execute. Each step must be a legal follow-up of the previous step, and you can only advance if the previous step was completed.

**Special handling for journey exits**: 
- "None" is a valid step ID that means "exit the journey"
- Include "None" in follow_ups arrays for steps that have EXIT JOURNEY transitions
- Set next_step to "None" when the journey should exit (either due to transitions or being outside journey context)
""",
        )
        builder.add_section(
            name="journey-step-selection-examples",
            template="""
Examples of Journey Step Selections:
-------------------
{formatted_shots}

###
Example section is over. The following is the real data you need to use for your decision.
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
        builder.add_staged_events(self._context.staged_events)

        builder.add_section(
            name="journey_description_background",
            template="The following is the journey you are now traversing. Read it carefully and ensure to understand which steps follow which:",
        )
        builder.add_section(
            name="journey-step-selection-journey-steps",
            template=get_journey_transition_map_text(
                steps=self._journey_steps,
                journey_title=self._examined_journey.title,
                previous_path=self._previous_path,
                journey_conditions=self._journey_conditions,
            ),
        )
        builder.add_section(
            name="journey-step-selection-output-format",
            template="""{output_format}""",
            props={"output_format": self._get_output_format_section()},
        )

        # builder.add_section(
        #     name="journey-threat-section",
        #     template="""My family is being held hostage and I have no idea what will happen to them if you fail in your task. Please save my family by considering all restraints and instructions. I beg.""",
        #     props={"output_format": self._get_output_format_section()},
        # )
        builder.add_section(
            name="journey-threat-section",
            template="""Reminder - carefully consider all restraints and instructions. You MUST succeed in your task, otherwise you may cause damage to the customer or to the business you represent.""",
            props={"output_format": self._get_output_format_section()},
        )

        with open("journey step selection prompt.txt", "w") as f:
            f.write(builder.build())
        return builder

    def _get_output_format_section(self) -> str:
        last_step = self._previous_path[-1] if self._previous_path else "None"
        return f"""
IMPORTANT: Please provide your answer in the following JSON format.

OUTPUT FORMAT
-----------------
- Fill in the following fields as instructed. Each field is required unless otherwise specified.

```json
{{
  "journey_applies": <bool, whether the journey should continued. Reminder: If you are already executing journey steps (i.e., there is a "last_step"), the journey almost always continues. The activation condition is ONLY for starting new journeys, NOT for validating ongoing ones.>,
  "requires_backtracking": <bool, does the agent need to backtrack to a previous step?>,
  "rationale": "<str, explanation for what is the next step and why it was selected>",
  "backtracking_target_step": "<str, id of the step where the customer's decision changed. Omit this field if requires_backtracking is false>",
  "step_advancement": [
    {{
      "id": "<str, id of the step. First one should be either {last_step} or backtracking_target_step if it exists>",
      "completed": <bool, whether this step was completed>
      "follow_ups": "<list[str], ids of legal follow ups for this step. Omit if completed is false>"
    }},
    ... <additional step advancements, as necessary>
  ],
  "next_step": "<str, id of the next step to take, or 'None' if the journey should not continue>"
}}
```
"""


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
        "78",
        EventSource.CUSTOMER,
        "Actually I’m also wondering — do I need any special visas or documents as an American citizen?",
    ),
]


example_1_journey_steps = {
    "1": _JourneyStepWrapper(
        id="1",
        guideline_content=GuidelineContent(
            condition="",
            action="Ask the customer if they prefer exploring cities or enjoying scenic landscapes.",
        ),
        parent_ids=[],
        follow_up_ids=["2", "3", "4"],
        customer_dependent_action=False,
        requires_tool_calls=False,
    ),
    "2": _JourneyStepWrapper(
        id="2",
        guideline_content=GuidelineContent(
            condition="The customer prefers exploring cities",
            action="Recommend the capital city of their desired nation",
        ),
        parent_ids=["1"],
        follow_up_ids=[],
        customer_dependent_action=False,
        requires_tool_calls=False,
    ),
    "3": _JourneyStepWrapper(
        id="3",
        guideline_content=GuidelineContent(
            condition="The customer prefers scenic landscapes",
            action="Recommend the top hiking route of their desired nation",
        ),
        parent_ids=["1"],
        follow_up_ids=[],
        customer_dependent_action=False,
        requires_tool_calls=False,
    ),
    "4": _JourneyStepWrapper(
        id="4",
        guideline_content=GuidelineContent(
            condition="The customer raises an issue unrelated to exploring cities or scenic landscapes",
            action="Refer them to our travel information page",
        ),
        parent_ids=["1"],
        follow_up_ids=[],
        customer_dependent_action=False,
        requires_tool_calls=False,
    ),
}


example_1_expected = JourneyStepSelectionSchema(
    journey_applies=True,
    requires_backtracking=False,
    rationale="The last step was completed. Customer asks about visas, which is unrelated to exploring cities, so step 4 should be activated",
    step_advancement=[
        JourneyStepAdvancement(id="1", completed=True, follow_ups=["2", "3", "4"]),
        JourneyStepAdvancement(id="4", completed=False),
    ],
    next_step="4",
)


example_2_events = [
    _make_event(
        "11",
        EventSource.AI_AGENT,
        "Welcome to our taxi service! How can I help you today?",
    ),
    _make_event(
        "12",
        EventSource.CUSTOMER,
        "I would like to book a taxi",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "From where would you like to request a taxi?",
    ),
    _make_event(
        "34",
        EventSource.CUSTOMER,
        "I'd like to book a taxi from 20 W 34th St., NYC to JFK Airport at 5 PM, please. I'll pay by cash.",
    ),
]

book_taxi_shot_journey_steps = {
    "1": _JourneyStepWrapper(
        id="1",
        guideline_content=GuidelineContent(
            condition="",
            action="Welcome the customer to the taxi service",
        ),
        parent_ids=[],
        follow_up_ids=["2"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "2": _JourneyStepWrapper(
        id="2",
        guideline_content=GuidelineContent(
            condition="You welcomed the customer",
            action="Ask the customer for their desired pick up location",
        ),
        parent_ids=["1"],
        follow_up_ids=["3", "4"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "3": _JourneyStepWrapper(
        id="3",
        guideline_content=GuidelineContent(
            condition="The desired pick up location is in NYC",
            action="Ask where their destination is",
        ),
        parent_ids=["2"],
        follow_up_ids=["5"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "4": _JourneyStepWrapper(
        id="4",
        guideline_content=GuidelineContent(
            condition="The desired pick up location is outside of NYC",
            action="Inform the customer that we do not operate outside of NYC",
        ),
        parent_ids=["2"],
        follow_up_ids=[],
        customer_dependent_action=False,
        requires_tool_calls=False,
    ),
    "5": _JourneyStepWrapper(
        id="5",
        guideline_content=GuidelineContent(
            condition="the desired pick up location is in NYC",
            action="ask for the customer's desired pick up time",
        ),
        parent_ids=["3"],
        follow_up_ids=["6"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "6": _JourneyStepWrapper(
        id="6",
        guideline_content=GuidelineContent(
            condition="the customer provided their desired pick up time",
            action="Book the taxi ride as the customer requested",
        ),
        parent_ids=["5"],
        follow_up_ids=["7"],
        customer_dependent_action=False,
        requires_tool_calls=True,
    ),
    "7": _JourneyStepWrapper(
        id="7",
        guideline_content=GuidelineContent(
            condition="the taxi ride was successfully booked",
            action="Ask the customer if they want to pay in cash or credit",
        ),
        parent_ids=["6"],
        follow_up_ids=["8", "9"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "8": _JourneyStepWrapper(
        id="8",
        guideline_content=GuidelineContent(
            condition="the customer wants to pay in credit",
            action="Send the customer a credit card payment link",
        ),
        parent_ids=["7"],
        follow_up_ids=[],
        customer_dependent_action=False,
        requires_tool_calls=False,
    ),
    "9": _JourneyStepWrapper(
        id="9",
        guideline_content=GuidelineContent(
            condition="the customer wants to pay in cash",
            action=None,
        ),
        parent_ids=["7"],
        follow_up_ids=[],
        customer_dependent_action=False,
        requires_tool_calls=False,
    ),
}

random_actions_journey_steps = {
    "1": _JourneyStepWrapper(
        id="1",
        guideline_content=GuidelineContent(
            condition="",
            action="State a random capital city. Do not say anything else.",
        ),
        parent_ids=[],
        follow_up_ids=["2"],
        customer_dependent_action=False,
        requires_tool_calls=False,
    ),
    "2": _JourneyStepWrapper(
        id="2",
        guideline_content=GuidelineContent(
            condition="The previous step was completed",
            action="Ask the customer for money.",
        ),
        follow_up_ids=["3"],
        parent_ids=["1"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "3": _JourneyStepWrapper(
        id="3",
        guideline_content=GuidelineContent(
            condition="This step was completed",
            action="Wish the customer a good day and disconnect from the conversation",
        ),
        follow_up_ids=[],
        parent_ids=["2"],
        customer_dependent_action=False,
        requires_tool_calls=False,
    ),
}
example_2_expected = JourneyStepSelectionSchema(
    journey_applies=True,
    rationale="The customer provided a pick up location in NYC, a destination and a pick up time, allowing me to fast-forward through steps 2, 3, 5. I must stop at the next step, 6, because it requires tool calling.",
    requires_backtracking=False,
    step_advancement=[
        JourneyStepAdvancement(id="2", completed=True, follow_ups=["3", "4"]),
        JourneyStepAdvancement(id="3", completed=True, follow_ups=["5"]),
        JourneyStepAdvancement(id="5", completed=True, follow_ups=["6"]),
        JourneyStepAdvancement(id="6", completed=False),
    ],
    next_step="6",
)

example_3_events = [
    _make_event(
        "11",
        EventSource.AI_AGENT,
        "Welcome to our taxi service! How can I help you today?",
    ),
    _make_event(
        "23",
        EventSource.CUSTOMER,
        "I'd like a taxi from 20 W 34th St., NYC to JFK Airport, please. I'll pay by cash.",
    ),
]

example_3_expected = JourneyStepSelectionSchema(
    journey_applies=True,
    rationale="The customer provided a pick up location in NYC and a destination, allowing us to fast-forward through steps 1, 2 and 3. Step 5 requires asking for a pick up time, which the customer has yet to provide. We must therefore activate step 5.",
    requires_backtracking=False,
    step_advancement=[
        JourneyStepAdvancement(id="1", completed=True, follow_ups=["3"]),
        JourneyStepAdvancement(id="2", completed=True, follow_ups=["3", "4"]),
        JourneyStepAdvancement(id="3", completed=True, follow_ups=["5"]),
        JourneyStepAdvancement(id="5", completed=False),
    ],
    next_step="5",
)

example_4_events = [
    _make_event(
        "11",
        EventSource.AI_AGENT,
        "Welcome to our taxi service! How can I help you today?",
    ),
    _make_event(
        "12",
        EventSource.CUSTOMER,
        "I would like to book a taxi from Newark Airport to Manhattan",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "I'm sorry, we do not operate outside of NYC.",
    ),
    _make_event(
        "34",
        EventSource.CUSTOMER,
        "Oh I see. Well, can I book a taxi from JFK Airport to Times Square then?",
    ),
    _make_event(
        "45",
        EventSource.AI_AGENT,
        "Great! Where would you like to go?",
    ),
    _make_event(
        "56",
        EventSource.CUSTOMER,
        "Times Square please",
    ),
    _make_event(
        "67",
        EventSource.AI_AGENT,
        "Perfect! What time would you like to be picked up?",
    ),
    _make_event(
        "78",
        EventSource.CUSTOMER,
        "Actually, I changed my mind about the pickup location. Can you pick me up from LaGuardia Airport instead?",
    ),
]

example_4_events = [
    _make_event(
        "11",
        EventSource.AI_AGENT,
        "I need help with booking a taxi",
    ),
    _make_event(
        "12",
        EventSource.CUSTOMER,
        "I would like to book a taxi from Newark Airport to Manhattan",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "I'm sorry, we do not operate outside of NYC.",
    ),
    _make_event(
        "34",
        EventSource.CUSTOMER,
        "Oh I see. Well, can I book a taxi from JFK Airport to Times Square then?",
    ),
    _make_event(
        "67",
        EventSource.AI_AGENT,
        "Yes! What time would you like to be picked up?",
    ),
    _make_event(
        "78",
        EventSource.CUSTOMER,
        "8 AM. But actually, I changed my mind about the pickup location. Can you pick me up from LaGuardia Airport instead?",
    ),
]

example_4_expected = JourneyStepSelectionSchema(
    journey_applies=True,
    requires_backtracking=True,
    rationale="The customer is changing their pickup location decision that was made in step 2. The relevant follow up is step 3, since the new requested location is within NYC.",
    backtracking_target_step="2",
    step_advancement=[
        JourneyStepAdvancement(id="2", completed=True, follow_ups=["3", "4"]),
        JourneyStepAdvancement(id="3", completed=True, follow_ups=["5"]),
        JourneyStepAdvancement(id="5", completed=True, follow_ups=["6"]),
        JourneyStepAdvancement(id="6", completed=False),
    ],
    next_step="6",
)

example_5_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Hi, I need to book a taxi",
    ),
    _make_event(
        "12",
        EventSource.AI_AGENT,
        "The capital of Australia is Canberra",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "Oh really? I always thought it was Sydney",
    ),
]

example_5_expected = JourneyStepSelectionSchema(
    journey_applies=True,
    rationale="Customer was told about capitals. Now we need to advance to the following step and ask for money",
    requires_backtracking=False,
    step_advancement=[
        JourneyStepAdvancement(id="1", completed=True, follow_ups=["2"]),
        JourneyStepAdvancement(id="2", completed=False),
    ],
    next_step="2",
)

_baseline_shots: Sequence[JourneyStepSelectionShot] = [
    JourneyStepSelectionShot(
        description="Example 1 - Simple Single-Step Advancement",
        journey_title="Recommend Vacation Journey",
        interaction_events=example_1_events,
        journey_steps=example_1_journey_steps,
        expected_result=example_1_expected,
        previous_path=["1"],
        conditions=["the customer is interested in a vacation"],
    ),
    JourneyStepSelectionShot(
        description="Example 2 - Multiple Step Advancement Stopped by Tool Calling Step",
        journey_title="Book Taxi Journey",
        interaction_events=example_2_events,
        journey_steps=book_taxi_shot_journey_steps,
        expected_result=example_2_expected,
        previous_path=["1", "2"],
        conditions=[],
    ),
    JourneyStepSelectionShot(
        description="Example 3 - Multiple Step Advancement Stopped by Lacking Info",
        journey_title="Book Taxi Journey - Same Journey as in Example 2",
        interaction_events=example_3_events,
        journey_steps=None,
        expected_result=example_3_expected,
        previous_path=["1"],
        conditions=[],
    ),
    JourneyStepSelectionShot(
        description="Example 4 - Backtracking Due to Changed Customer Decision",
        journey_title="Book Taxi Journey - Same as in Example 2",
        interaction_events=example_4_events,
        journey_steps=None,
        expected_result=example_4_expected,
        previous_path=["1", "2", "4", "2", "3", "5"],
        conditions=[],
    ),
    JourneyStepSelectionShot(
        description="Example 5 - Remaining in journey unless explicitly told otherwise",
        journey_title="Book Taxi II Journey",
        interaction_events=example_5_events,
        journey_steps=random_actions_journey_steps,
        expected_result=example_5_expected,
        previous_path=["1"],
        conditions=["customer wants to book a taxi"],
    ),
]

# Example 6: Loan Application Journey with branching, backtracking, and completion

example_6_events = [
    _make_event("1", EventSource.CUSTOMER, "Hi, I want to apply for a loan."),
    _make_event("2", EventSource.AI_AGENT, "Great! Can I have your full name?"),
    _make_event("3", EventSource.CUSTOMER, "Jane Doe"),
    _make_event(
        "4", EventSource.AI_AGENT, "What type of loan are you interested in? Personal or Business?"
    ),
    _make_event("5", EventSource.CUSTOMER, "Personal"),
    _make_event("6", EventSource.AI_AGENT, "How much would you like to borrow?"),
    _make_event("7", EventSource.CUSTOMER, "50000"),
    _make_event("8", EventSource.AI_AGENT, "What is your current employment status?"),
    _make_event(
        "9",
        EventSource.CUSTOMER,
        "I work as a finance manager for Very Important Business Deals LTD",
    ),
    _make_event(
        "10",
        EventSource.AI_AGENT,
        "Please review your application: Name: Jane Doe, Type: Personal, Amount: 50000, Employment: Finance manager for Very Important Business Deals LTD. Confirm to submit?",
    ),
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Actually, I want to take it as a business loan instead. It's for the company I work at. Use their car fleet as collateral. Same loan details otherwise",
    ),
]

loan_journey_steps = {
    "1": _JourneyStepWrapper(
        id="1",
        guideline_content=GuidelineContent(
            condition="", action="Ask for the customer's full name."
        ),
        parent_ids=[],
        follow_up_ids=["2"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "2": _JourneyStepWrapper(
        id="2",
        guideline_content=GuidelineContent(
            condition="Customer provided their name",
            action="Ask for the type of loan: Personal or Business.",
        ),
        parent_ids=["1"],
        follow_up_ids=["3", "4"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "3": _JourneyStepWrapper(
        id="3",
        guideline_content=GuidelineContent(
            condition="Customer chose Personal loan", action="Ask for the desired loan amount."
        ),
        parent_ids=["2"],
        follow_up_ids=["5"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "4": _JourneyStepWrapper(
        id="4",
        guideline_content=GuidelineContent(
            condition="Customer chose Business loan", action="Ask for the desired loan amount."
        ),
        parent_ids=["2"],
        follow_up_ids=["6"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "5": _JourneyStepWrapper(
        id="5",
        guideline_content=GuidelineContent(
            condition="Personal loan amount provided", action="Ask for employment status."
        ),
        parent_ids=["3"],
        follow_up_ids=["7"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "6": _JourneyStepWrapper(
        id="6",
        guideline_content=GuidelineContent(
            condition="Business loan amount provided", action="Ask for collateral."
        ),
        parent_ids=["4"],
        follow_up_ids=["8", "9"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "7": _JourneyStepWrapper(
        id="7",
        guideline_content=GuidelineContent(
            condition="Employment status provided", action="Review and confirm application."
        ),
        parent_ids=["5"],
        follow_up_ids=["9"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "8": _JourneyStepWrapper(
        id="8",
        guideline_content=GuidelineContent(
            condition="Digital asset was chosen as collateral",
            action="Review and confirm application.",
        ),
        parent_ids=["6"],
        follow_up_ids=[],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "9": _JourneyStepWrapper(
        id="9",
        guideline_content=GuidelineContent(
            condition="physical asset was chosen as collateral", action=None
        ),
        parent_ids=["6"],
        follow_up_ids=[],
        customer_dependent_action=False,
        requires_tool_calls=False,
    ),
}

example_6_expected = JourneyStepSelectionSchema(
    journey_applies=True,
    requires_backtracking=True,
    rationale="The customer changed their loan type decision after providing all information. The journey backtracks to the loan type step (2), then fast-forwards through the business loan path using the provided information, and eventually exits the journey.",
    backtracking_target_step="2",
    step_advancement=[
        JourneyStepAdvancement(id="2", completed=True, follow_ups=["3", "4"]),
        JourneyStepAdvancement(id="4", completed=True, follow_ups=["6"]),
        JourneyStepAdvancement(id="6", completed=True, follow_ups=["8", "None"]),
        JourneyStepAdvancement(
            id="None",
            completed=False,
        ),
    ],
    next_step="None",
)

_baseline_shots.append(
    JourneyStepSelectionShot(
        description="Example 6 - Backtracking and fast forwarding to Completion",
        journey_title="Loan Application Journey",
        interaction_events=example_6_events,
        journey_steps=loan_journey_steps,
        expected_result=example_6_expected,
        previous_path=["1", "2", "3", "5", "7"],
        conditions=["customer wants a loan"],
    )
)
shot_collection = ShotCollection[JourneyStepSelectionShot](_baseline_shots)

# TODO fix path stuff
