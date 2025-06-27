from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any, Optional, cast
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
)
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.journeys import Journey
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.shots import Shot, ShotCollection


class _JourneyStepWrapper(DefaultBaseModel):
    id: str
    guideline_content: GuidelineContent
    parent_ids: list[str]
    follow_up_ids: list[str]
    customer_dependent_action: bool
    requires_tool_calls: bool


class JourneyStepSelectionSchema(DefaultBaseModel):
    last_customer_message: str
    journey_applies: bool
    last_current_step: str
    rationale: str
    requires_backtracking: bool
    backtracking_target_step: Optional[str] | None = ""
    last_current_step_completed: Optional[bool] | None = None
    step_advance: Optional[Sequence[str | None]] = []
    next_step: str


@dataclass
class JourneyStepSelectionShot(Shot):
    interaction_events: Sequence[Event]
    journey_title: str
    journey_steps: dict[str, _JourneyStepWrapper] | None
    previous_path: Sequence[str | None]
    expected_result: JourneyStepSelectionSchema


class GenericJourneyStepSelectionBatch(GuidelineMatchingBatch):
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[JourneyStepSelectionSchema],
        examined_journey: Journey,  # NOTE THAT JOURNEY STEPS MUST NOT CHANGE MID-SESSION
        context: GuidelineMatchingContext,
        guidelines: Sequence[Guideline] = [],
        journey_path: Sequence[str | None] = [],
    ) -> None:
        self._logger = logger
        self._schematic_generator = schematic_generator

        self._guidelines = {str(i): g for i, g in enumerate(guidelines, start=1)}
        self._guideline_ids = {g.id: g for g in guidelines}

        self._context = context

        self._examined_journey = examined_journey
        self._guideline_id_to_journey_step_id = {
            id: str(i) for i, id in enumerate(examined_journey.steps, start=1)
        }

        self._journey_steps: dict[str, _JourneyStepWrapper] = self._build_journey_steps()
        self._previous_path: Sequence[str | None] = journey_path

    def _build_journey_steps(
        self,
    ) -> dict[
        str, _JourneyStepWrapper
    ]:  # TODO add comment with what should happen in the path validation, Dor / Kfir will validate
        journey_steps = self._examined_journey.steps
        journey_steps_dict: dict[str, _JourneyStepWrapper] = {
            self._guideline_id_to_journey_step_id[step_guideline_id]: _JourneyStepWrapper(
                id=self._guideline_id_to_journey_step_id[step_guideline_id],
                guideline_content=self._guideline_ids[step_guideline_id].content,
                parent_ids=[],
                follow_up_ids=[
                    self._guideline_id_to_journey_step_id[guideline_id]
                    for guideline_id in cast(
                        Sequence[GuidelineId],
                        self._guideline_ids[step_guideline_id].metadata.get("sub_steps", []),
                    )
                ],
                customer_dependent_action=cast(
                    dict[str, bool],
                    self._guideline_ids[step_guideline_id].metadata[
                        "customer_dependent_action_data"
                    ],
                )["is_customer_dependent"]
                if "is_customer_dependent"
                in cast(
                    dict[str, bool],
                    self._guideline_ids[step_guideline_id].metadata.get(
                        "customer_dependent_action_data", dict()
                    ),
                )
                else False,
                requires_tool_calls=cast(
                    bool, self._guideline_ids[step_guideline_id].metadata["tool_running_only"]
                ),
            )  # TODO make less ugly
            for step_guideline_id in journey_steps
        }

        for id, js in journey_steps_dict.items():
            for followup_id in js.follow_up_ids:
                journey_steps_dict[followup_id].parent_ids.append(id)

        return journey_steps_dict

    @override
    async def process(self) -> GuidelineMatchingBatchResult:
        prompt = self._build_prompt(shots=await self.shots())

        with self._logger.operation(f"JourneyStepSelectionBatch: {self._examined_journey.title}"):
            inference = await self._schematic_generator.generate(
                prompt=prompt,
                hints={"temperature": 0.15},
            )
        with open("journey step selection output.txt", "w") as f:  # TODO delete
            f.write(inference.content.model_dump_json(indent=2))

        self._logger.debug(f"Completion:\n{inference.content.model_dump_json(indent=2)}")

        if inference.content.requires_backtracking:
            journey_path: list[str | None] = [inference.content.next_step]
        else:
            journey_path = cast(list[str | None], inference.content.step_advance)
            if (
                self._previous_path
                and not self._previous_path[-1]
                and journey_path[0] != self._previous_path[-1]
            ):
                self._logger.debug(
                    f"WARNING: Illegal journey path returned by journey step selection. Expected path from {self._previous_path} to {journey_path}"
                )
                journey_path = [
                    inference.content.next_step
                ]  # If path is illegal, return only the next step
            for i in range(1, len(journey_path)):
                if journey_path[i - 1] not in self._journey_steps.keys():
                    self._logger.debug(
                        f"WARNING: Illegal journey path returned by journey step selection. Illegal step returned: {journey_path[i]}. Full path: : {journey_path}"
                    )
                elif (
                    journey_path[i]
                    not in self._journey_steps[str(journey_path[i - 1])].follow_up_ids
                ):
                    self._logger.debug(
                        f"WARNING: Illegal transition in journey path returned by journey step selection - from {journey_path[i-1]} to {journey_path[i]}. Full path: : {journey_path}"
                    )
                    journey_path = [inference.content.next_step]

            if (
                journey_path
                and journey_path[-1] not in self._journey_steps.keys()
                and inference.content.next_step is not None
            ):  # 'Exit journey' was selected, or illegal value returned (both cause no guidelines to be active)
                self._logger.debug(
                    f"WARNING: Last journey step in returned path is not legal. Full path: : {journey_path}"
                )
                journey_path[-1] = None

        return GuidelineMatchingBatchResult(
            matches=[
                GuidelineMatch(
                    guideline=self._guidelines[inference.content.next_step],
                    score=10,
                    rationale=inference.content.rationale,
                    guideline_previously_applied=PreviouslyAppliedType.IRRELEVANT,
                    metadata={"journey_path": journey_path},
                )
            ]
            if inference.content.next_step
            in self._journey_steps.keys()  # If either 'None' or an illegal step was returned, don't activate guidelines
            else [],
            generation_info=inference.info,
        )

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
        formatted_shot += f"The steps that were visited in past messages in this example are {shot.previous_path}."
        if shot.journey_steps:
            formatted_shot += self._get_journey_steps_section(
                shot.journey_steps, shot.journey_title
            )

        formatted_shot += f"""
- **Expected Result**:
```json
{json.dumps(shot.expected_result.model_dump(mode="json", exclude_unset=True), indent=2)}
```
"""

        return formatted_shot

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
Determine if the conversation remains within the journey scope.
- Set `journey_applies` to `true` unless the customer explicitly requests to leave the topic or changes the subject completely
- If `journey_applies` is `false`, set `next_step` to `'None'` and skip remaining steps

## 2: Backtracking Check  
Check if the customer has changed a previous decision that requires returning to an earlier step.
- Set `requires_backtracking` to `true` if the customer contradicts or changes a prior choice
- If backtracking is needed:
  - Set `backtracking_target_step` to the step where the decision changed
  - Set `next_step` to the appropriate follow-up step based on the customer's new choice (e.g., if they change their delivery address, don't re-ask for the address - proceed to the next step that handles the new address)

## 3: Current Step Completion
Evaluate whether the last executed step is complete.
- Set `last_current_step_completed` to `true` if the agent performed the required action
- For steps with `CUSTOMER_DEPENDENT` flag: step is complete only if both agent acted AND customer responded appropriately. These are usually questions that the customer must answer for the step to be considered completed.
- If incomplete, set `next_step` to the current step ID (repeat the step) and return the current step ID as the sole member of 'step_advance'.

## 4: Journey Advancement
If the current step is complete, advance through subsequent steps until you encounter:
- A step requiring a tool call (`REQUIRES_TOOL_CALLS` flag)
- A step where you lack necessary information to proceed
- A step requiring you to communicate something new to the customer, beyond asking them for information

Document your advancement path in `step_advance` as a list of step IDs, starting with last_current_step and ending with the next step to execute.
""",
        )
        builder.add_section(
            name="journey-step-selection-examples",
            template="""
Examples of Journey Step Selections:
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
        builder.add_staged_events(self._context.staged_events)
        builder.add_section(
            name="journey-step-selection-previous_path",
            template=self._get_previous_path_section(self._previous_path),
        )
        builder.add_section(
            name="journey-step-selection-journey-steps",
            template=self._get_journey_steps_section(
                self._journey_steps, self._examined_journey.title
            ),
        )
        builder.add_section(
            name="journey-step-selection-output-format",
            template=self._get_output_format_section(),
        )

        with open("journey step selection prompt.txt", "w") as f:  # TODO delete
            f.write(builder.build())
        return builder

    def _get_output_format_section(self) -> str:
        return """
IMPORTANT: Please provide your answer in the following JSON format.

OUTPUT FORMAT
-----------------
- Fill in the following fields as instructed. Each field is required unless otherwise specified.

```json
{{
  "last_customer_message": "<str, the most recent message from the customer>",
  "journey_applies": <bool, whether the journey should be continued>,
  "last_current_step": "<str, the id of the last current step>",
  "rationale": "<str, explanation for what is the next step and why it was selected>",
  "requires_backtracking": <bool, does the agent need to backtrack to a previous step?>,
  "backtracking_target_step": "<str, id of the step to backtrack to. Omit this field if requires_backtracking is false>",
  "last_current_step_completed": <bool or null, whether the last current step was completed. Should be omitted if either requires_backtracking or requires_fast_forwarding is true>,
  "step_advance": <list of step ids (str) to advance through, beginning in last_current_step and ending in next_step>, 
  "next_step": "<str, id of the next step to take, or 'None' if the journey should not continue>"
}}
```
"""

    def _get_journey_steps_section(
        self, steps: dict[str, _JourneyStepWrapper], journey_title: str
    ) -> str:
        def step_sort_key(step_id: str) -> Any:
            try:
                return int(step_id)
            except Exception:
                return step_id

        # Sort steps by step id as integer if possible, else as string
        steps_str = ""
        for step_id in sorted(steps.keys(), key=step_sort_key):
            step: _JourneyStepWrapper = steps[step_id]
            action: str | None = step.guideline_content.action
            flags_str = ""
            if action:
                if step.customer_dependent_action or step.requires_tool_calls:
                    flags_str += "Step Flags:\n"
                    if step.customer_dependent_action:
                        flags_str += (
                            "- CUSTOMER_DEPENDENT: Requires customer action to be completed\n"
                        )
                    if (
                        step.requires_tool_calls
                        and (not self._previous_path or step.id != self._previous_path[-1])
                    ):  # Not including this flag for current step - if we got here, the tool call should've executed so the flag would be misleading
                        flags_str += "- REQUIRES_TOOL_CALLS: Do not advance past this step\n"
                    if self._previous_path and step.id == self._previous_path[-1]:
                        flags_str += "- This is the last step that was executed. Begin advancing on from this step\n"
                if step.follow_up_ids:
                    follow_ups_str = "\n".join(
                        [
                            f"""↳ If "{steps[follow_up_id].guideline_content.condition}" → Go to step {follow_up_id if steps[follow_up_id].guideline_content.action else "EXIT JOURNEY, RETURN 'NONE'"}"""
                            for follow_up_id in step.follow_up_ids
                        ]
                    )
                else:
                    follow_ups_str = "↳ IF this step is completed,  → RETURN 'NONE'"
                steps_str += f"""
STEP {step_id}: {action}
{flags_str}
TRANSITIONS:
{follow_ups_str}
"""
        # TODO consider adding the trigger string here
        return f"""
Journey: {journey_title}

Steps:
{steps_str} 
"""

    def _get_previous_path_section(self, previous_path: Sequence[str | None]) -> str:
        if not previous_path or all([p is None for p in previous_path]):
            return "The journey has just began. No previous steps have been performed. Begin at step 1."
        return f"""
The steps that were visited in past messages of these conversation, in chronological order, are: {previous_path}. 
You may only backtrack to one of these steps.
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
    last_current_step="1",
    last_customer_message="Actually I’m also wondering — do I need any special visas or documents as an American citizen?",
    journey_applies=True,
    rationale="The last step was completed. Customer asks about visas, which is unrelated to exploring cities, so step 4 should be activated",
    requires_backtracking=False,
    last_current_step_completed=True,
    step_advance=["1", "4"],
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
        follow_up_ids=["3"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "2": _JourneyStepWrapper(
        id="2",
        guideline_content=GuidelineContent(
            condition="You welcomed the customer",
            action="Ask the customer where their desired pick up location",
        ),
        parent_ids=[],
        follow_up_ids=["3"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "3": _JourneyStepWrapper(
        id="3",
        guideline_content=GuidelineContent(
            condition="The desired pick up location is in of NYC",
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
            condition="the customer provided their destination",
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

example_2_expected = JourneyStepSelectionSchema(
    last_current_step="2",
    last_customer_message="I'd like a taxi from 20 W 34th St., NYC to JFK Airport, at 5 PM please. I'll pay by cash.",
    journey_applies=True,
    rationale="The customer provided a pick up location in NYC, a destination and a pick up time, allowing me to fast-forward through steps 2, 3, 5. I must stop at the next step, 6, because it requires tool calling.",
    requires_backtracking=False,
    last_current_step_completed=True,
    step_advance=["2", "3", "5", "6"],
    next_step="6",
)

example_3_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Welcome to our taxi service! How can I help you today?",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "I'd like a taxi from 20 W 34th St., NYC to JFK Airport, please. I'll pay by cash.",
    ),
]

example_3_expected = JourneyStepSelectionSchema(
    last_current_step="1",
    last_customer_message="I'd like a taxi from 20 W 34th St., NYC to JFK Airport, please. I'll pay by cash.",
    journey_applies=True,
    rationale="The customer provided a pick up location in NYC and a destination, allowing us to fast-forward through steps 1, 2 and 3. Step 5 requires asking for a pick up time, which the customer has yet to provide. We must therefore activate step 5.",
    requires_backtracking=False,
    last_current_step_completed=True,
    step_advance=["1", "2", "3", "5"],
    next_step="5",
)

# TODO add flag for previously visited steps
# TODO add few-shots
# 4. Backtracking
# 5. Step needs to be repeated?
# 6. journey completed?

_baseline_shots: Sequence[JourneyStepSelectionShot] = [
    JourneyStepSelectionShot(
        description="Example 1 - Simple Single-Step Advancement",
        journey_title="Recommend Vacation Journey",
        interaction_events=example_1_events,
        journey_steps=example_1_journey_steps,
        expected_result=example_1_expected,
        previous_path=["1"],
    ),
    JourneyStepSelectionShot(
        description="Example 2 - Multiple Step Advancement Stopped by Tool Calling Step",
        journey_title="Book Taxi Journey",
        interaction_events=example_2_events,
        journey_steps=book_taxi_shot_journey_steps,
        expected_result=example_2_expected,
        previous_path=["1", "2"],
    ),
    JourneyStepSelectionShot(
        description="Example 3 - Multiple Step Advancement Stopped by Lacking Info",
        journey_title="Book Taxi Journey - Same Journey as Previous Example",
        interaction_events=example_3_events,
        journey_steps=None,
        expected_result=example_3_expected,
        previous_path=["1"],
    ),
]

shot_collection = ShotCollection[JourneyStepSelectionShot](_baseline_shots)
