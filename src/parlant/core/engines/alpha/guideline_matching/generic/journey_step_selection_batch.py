from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
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
)
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.guidelines import Guideline, GuidelineContent
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


class JourneyStepSelectionSchema(DefaultBaseModel):
    last_customer_message: str
    journey_applies: bool
    last_current_step: str
    rationale: str
    requires_backtracking: bool
    backtracking_target_step: Optional[str] | None = (
        ""  # TODO consider adding extra arq for its parent
    )
    last_current_step_completed: Optional[bool] | None = None
    step_advance: Sequence[str]
    next_step: str


@dataclass
class JourneyStepSelectionShot(Shot):
    interaction_events: Sequence[Event]
    journey_steps: dict[str, _JourneyStepWrapper]
    expected_result: JourneyStepSelectionSchema


class GenericJourneyStepSelectionBatch(GuidelineMatchingBatch):
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[JourneyStepSelectionSchema],
        examined_journey: Journey,
        context: GuidelineMatchingContext,
        guidelines: Sequence[Guideline] = [],
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
        self._last_step_id = 0  # TODO change to initial step

    def _build_journey_steps(self) -> dict[str, _JourneyStepWrapper]:
        journey_steps = self._examined_journey.steps

        journey_steps_dict: dict[str, _JourneyStepWrapper] = {
            self._guideline_id_to_journey_step_id[step_guideline_id]: _JourneyStepWrapper(
                id=self._guideline_id_to_journey_step_id[step_guideline_id],
                guideline_content=self._guideline_ids[step_guideline_id].content,
                parent_ids=[],
                follow_up_ids=[
                    self._guideline_id_to_journey_step_id[guideline_id]
                    for guideline_id in self._guideline_ids[step_guideline_id].sub_steps
                ],
            )
            for step_guideline_id in journey_steps
        }

        for id, js in journey_steps_dict.items():
            for followup_id in js.follow_up_ids:
                journey_steps_dict[followup_id].parent_ids.append(id)

        return journey_steps_dict

    @override
    async def process(self) -> GuidelineMatchingBatchResult:
        prompt = self._build_prompt(shots=await self.shots())

        with self._logger.operation(f"JourneyStepSelectionBacth: {self._examined_journey.title}"):
            inference = await self._schematic_generator.generate(
                prompt=prompt,
                hints={"temperature": 0.15},
            )

        self._logger.debug(f"Completion:\n{inference.content.model_dump_json(indent=2)}")

        return GuidelineMatchingBatchResult(
            matches=[
                GuidelineMatch(
                    guideline=self._guidelines[inference.content.next_step],
                    score=10,
                    rationale=inference.content.rationale,
                    guideline_previously_applied=PreviouslyAppliedType.IRRELEVANT,
                )
            ]
            if inference.content.next_step
            else [],
            generation_info=inference.info,
        )

    async def shots(self) -> Sequence[JourneyStepSelectionShot]:
        return await shot_collection.list()

    def _format_shots(self, shots: Sequence[JourneyStepSelectionShot]) -> str:
        return "\n".join(
            f"Example #{i}: ###\n{self._format_shot(shot)}" for i, shot in enumerate(shots, start=1)
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
        # TODO add journey steps to shot

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
You are an AI agent named {agent_name}. Your role is to chat with customers in a multi-turn conversation on behalf of the business you represent.
Your behavior is guided by "journeys" - predefined processes from the business you represent that guide user interactions.  
Journeys are defined as a collection of steps, each dictating an action that the agent should take.
After the performance of each step, the journey advances to the next step according to pre-defined rules, that will be provided to you later in this prompt.

Your task is to determine which journey step should apply next, based on the last step that was performed and the current state of the conversation. 
""",
            props={"agent_name": self._context.agent.name},
        )
        builder.add_section(  # TODO add note about customer dependent actions
            name="journey-step-selection-task_description",
            template="""
TASK DESCRIPTION
-------------------
Apply the following process to determine which journey step should apply next. Document your decisions in the output, according to a format that will be provided to you later in this prompt.

1. Context check: Determine if we're still within the journey context and document your decision under the "journey_applies" key.
 Unless the customer explicitly requests to leave the subject of the journey, journey_applies should be true.

2. Backtrack check: Check if we need to return to an earlier step (e.g., customer changes a decision they took in a previous step). 
 Backtracking is required if the customer changes a decision that was made in a previous step. 
 If backtracking is required, document your decision under the "requires_backtracking" key. 
 If it is required, apply the following process to determine which step to return to:
    a. Identify the step transition in which the customer made a decision that requires backtracking.
    b. Within that transition, identify which step should be taken next according to the customer's latest decision.
 If backtracking is required, you may skip the rest of the process and return to the step that requires backtracking.

 3. Last Step Completion check: Determine if the current step is complete, meaning that the agent already took the action it ascribes.
 Document your decision under the "last_current_step_completed" key.
 If it is not complete, it must be repeated - return the id of the current step.
 If it is complete, keep advancing in the journey, step by step, until you encounter either:
  i. A transition which you do not have enough information to perform. The step just before that transition should be the next step to be taken.  
  ii. You encounter a step that has the REQUIRES_TOOOL_CALLS flag. If this occurs, stop and return the id of that step.
 You must document each step that you advance through, one step id at a time, under the "step_advance" key.  



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
        builder.add_staged_events(
            self._context.staged_events
        )  # TODO add section about previous path
        builder.add_section(
            name="journey-step-selection-journey-steps",
            template=self._get_journey_steps_section(self._journey_steps),
        )
        builder.add_section(
            name="journey-step-selection-output-format",
            template=self._get_output_format_section(),
        )

        with open("journey step selection prompt.txt", "w") as f:
            f.write(builder.build())
        return builder

    def _get_output_format_section(self) -> str:
        return """
IMPORTANT: Please provide your answer in the following JSON format.

OUTPUT FORMAT
-----------------
- Fill in the following fields as instructed. Each field is required unless otherwise specified.

```json
{
  "last_customer_message": "<str, the most recent message from the customer>",
  "journey_applies: <bool, whether the journey should be continued>,
  "last_current_step": "<str, the id of the last current step>",
  "rationale": "<str, explanation for why the next step was selected>",
  "requires_backtracking": <bool, does the agent need to backtrack to a previous step?>,
  "backtracking_target_step": "<str, id of the step to backtrack to. Should be omitted if requires_backtracking is false>", ↓ 
  "requires_fast_forwarding": <bool, does the agent need to fast-forward to a future step?>,
  "fast_forward_path": <list of step ids to fast-forward through. Should be omitted if requires_fast_forwarding is false> 
  "last_current_step_completed": <bool or null, whether the last current step was completed. Should be omitted if either requires_backtracking or requires_fast_forwarding is true>,
  "next_step": "<str, id of the next step to take>"
}
```
"""

    def _get_journey_steps_section(
        self, steps: dict[str, _JourneyStepWrapper]
    ) -> str:  # TODO add REQUIRES_TOOOL_CALLS flag
        def step_sort_key(step_id):
            try:
                return int(step_id)
            except Exception:
                return step_id

        journey = self._examined_journey
        trigger_str = " OR ".join(journey.conditions)

        # Sort steps by step id as integer if possible, else as string
        steps_str = ""
        for step_id in sorted(steps.keys(), key=step_sort_key):
            step = steps[step_id]
            action = step.guideline_content.action
            if action:
                if step.follow_up_ids:
                    follow_ups_str = "\n".join(
                        [
                            f"""↳ If "{steps[follow_up_id].guideline_content.condition}" → STEP {follow_up_id if steps[follow_up_id].guideline_content.action else "EXIT JOURNEY, RETURN 'NONE'"}"""
                            for follow_up_id in step.follow_up_ids
                        ]
                    )
                else:
                    follow_ups_str = ["↳ IF this step is completed, RETURN 'NONE'"]
                steps_str += f"""
STEP {step_id}: {action}
{follow_ups_str}
"""

        return f"""
Journey: {journey.title}
Trigger: {trigger_str}

Steps:
{steps_str} 
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


example_1_journey_steps = {
    "1": _JourneyStepWrapper(
        id="1",
        guideline_content=GuidelineContent(
            condition="The customer is looking for flight or accommodation booking assistance",
            action="Provide links or suggestions for flight aggregators and hotel booking platforms.",
        ),
        parent_ids=[],
        follow_up_ids=["2"],
    ),
    "2": _JourneyStepWrapper(
        id="2",
        guideline_content=GuidelineContent(
            condition="The customer ask for activities recommendations",
            action="Guide them in refining their preferences and suggest options that match what they're looking for",
        ),
        parent_ids=["1"],
        follow_up_ids=["3"],
    ),
}

example_1_expected = JourneyStepSelectionSchema(
    last_current_step="1",
    last_customer_message="I'm looking for a flight to New York",
    journey_applies=True,
    rationale="The customer is looking for flight or accommodation booking assistance",
    requires_backtracking=False,
    backtracking_target_step=None,
    last_current_step_completed=False,
    step_advance=["1", "2"],
    next_step="2",
)

# TODO add few-shots
# One step simple advancement
# Backtracking
# Step needs to be repeated
# Multiple steps advancement - stopped by lacking info
# journey no longer applies
# Multiple steps advancement - stopped by requires toool calls
# journey completed

_baseline_shots: Sequence[JourneyStepSelectionShot] = [
    JourneyStepSelectionShot(
        description="Example 1",
        interaction_events=example_1_events,
        journey_steps=example_1_journey_steps,
        expected_result=example_1_expected,
    ),
]

shot_collection = ShotCollection[JourneyStepSelectionShot](_baseline_shots)
