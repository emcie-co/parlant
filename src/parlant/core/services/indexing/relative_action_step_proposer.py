from dataclasses import dataclass
import json
from typing import Optional, Sequence, cast
from parlant.core.common import DefaultBaseModel, JSONSerializable
from parlant.core.engines.alpha.guideline_matching.generic.journey_step_selection_batch import (
    _JourneyStepWrapper,
    get_journey_transition_map_text,
    build_journey_steps,
)
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.guidelines import Guideline, GuidelineContent
from parlant.core.journeys import Journey
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.shots import Shot, ShotCollection


class RewrittenActionResult(DefaultBaseModel):
    id: str
    rewritten_actions: str


class RelativeActionStepProposition(DefaultBaseModel):
    actions: Sequence[RewrittenActionResult]


class RelativeActionStepBatch(DefaultBaseModel):
    id: str
    condition: str
    action: str
    needs_rewrite_rational: str
    needs_rewrite: bool
    former_reference: Optional[str] = None
    rewritten_action: Optional[str] = None


class RelativeActionStepSchema(DefaultBaseModel):
    actions: Sequence[RelativeActionStepBatch]


@dataclass
class RelativeActionStepShot(Shot):
    journey_title: str
    journey_steps: dict[str, _JourneyStepWrapper]
    expected_result: RelativeActionStepSchema


class RelativeActionStepProposer:
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[RelativeActionStepSchema],
        service_registry: ServiceRegistry,
    ) -> None:
        self._logger = logger
        self._schematic_generator = schematic_generator
        self._service_registry = service_registry

    async def propose_relative_action_step(
        self,
        examined_journey: Journey,
        step_guidelines: Sequence[Guideline] = [],
        progress_report: Optional[ProgressReport] = None,
    ) -> RelativeActionStepProposition:
        if progress_report:
            await progress_report.stretch(1)

        with self._logger.scope("RelativeActionStepProposer"):
            result = await self._generate_relative_action_step_proposer(
                examined_journey,
                step_guidelines,
            )

        if progress_report:
            await progress_report.increment(1)

        rewritten_actions = []
        for a in result.actions:
            if a.needs_rewrite:
                rewritten_actions.append(
                    RewrittenActionResult(
                        id=a.id,
                        rewritten_actions=a.rewritten_action,
                    )
                )
        return RelativeActionStepProposition(actions=rewritten_actions)

    def get_journey_text(
        self,
        examined_journey: Journey,
        step_guidelines: Sequence[Guideline],
    ) -> str:
        step_guideline_mapping = {
            str(cast(dict[str, JSONSerializable], g.metadata["journey_step"])["id"]): g
            for g in step_guidelines
        }

        guideline_to_step_id_mapping = {
            g.id: str(cast(dict[str, JSONSerializable], g.metadata["journey_step"])["id"])
            for g in step_guidelines
        }
        journey_steps: dict[str, _JourneyStepWrapper] = build_journey_steps(
            guideline_to_step_id_mapping, step_guideline_mapping
        )
        return get_journey_transition_map_text(
            journey_steps,
            examined_journey.title,
        )

    async def _build_prompt(
        self,
        examined_journey: Journey,
        step_guidelines: Sequence[Guideline],
        shots: Sequence[RelativeActionStepShot],
    ) -> PromptBuilder:
        builder = PromptBuilder()

        builder.add_section(
            name="relative-action-step-proposer-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
In our system, the behavior of a conversational AI agent is structured around predefined "journeys" - structured workflows that guide customer interactions toward specific outcomes.

## Journey Structure
Each journey consists of:
- **Steps**: Individual actions that the agent must execute (e.g., ask a question, provide information, perform a task)
- **Transitions**: Rules that determine which step comes next based on customer responses or completion status

A pre-run evaluator analyzes journeys and outputs two key components:

Condition: The rule / circumstance that triggers the action
Action: What the agent should do

These condition-action pairs are then sent to an agent for execution. However, many actions are written with implicit dependencies on earlier journey context, making them unclear when viewed in isolation.

""",
        )

        builder.add_section(
            name="relative-action-step-proposer-task-description",
            template="""
TASK DESCRIPTION
-----------------
Your task is to evaluate whether actions are self-contained and comprehensible without additional context.

You will be asked to:
1. Determine if the action description is sufficiently clear on its own:
    - Can an agent understand exactly what to do based solely on the condition and action?
    - Does the action rely on unstated context from previous journey steps?

2. Rewriting (when needed): If an action lacks clarity, rewrite it to be completely self-contained
    - Include all necessary context within the action description
    - Ensure the agent can execute the action without referring to the broader journey
    - Maintain the original intent without elaborating beyond what is explicitly provided

A well-written action should:

- Clearly specify what the agent needs to do
- Contain all necessary information for execution
- Maintain the original functional intent without adding any information not explicitly stated in the journey

""",
        )
        builder.add_section(
            name="relative-action-step-proposer-shots",
            template="""
EXAMPLES
-----------
{shots_text}
""",
            props={"shots_text": self._format_shots(shots)},
        )

        builder.add_section(
            name="relative-action-step-proposer-journey-steps",
            template=self.get_journey_text(
                examined_journey,
                step_guidelines,
            ),
        )

        builder.add_section(
            name="relative-action-step-proposer-output-format",
            template="""
OUTPUT FORMAT
-----------
Use the following format to evaluate whether the guideline has a customer dependent action:
Expected output (JSON):
```json
{{
    {result_structure_text}
}}
```
""",
            props={"result_structure_text": self._format_text(step_guidelines)},
        )
        with open("prompt_relative_action_proposer.txt", "w") as f:
            f.write(builder.build())
        return builder

    def _format_text(
        self,
        step_guidelines: Sequence[Guideline],
    ) -> str:
        result_structure = [
            {
                "id": str(cast(dict[str, JSONSerializable], g.metadata["journey_step"])["id"]),
                "condition": g.content.condition,
                "action": g.content.action,
                "needs_rewrite_rational": "<Brief explanation of why the action does or does not need rewriting. Is it refer to something that is not mentioned in the current step>",
                "needs_rewrite": "<BOOL>",
                "former_reference": "<information from previous steps that the definition is referring to>",
                "rewritten_action": "<str. Full, self-contained version of the action - include only if requires_rewrite is True>",
            }
            for g in step_guidelines
        ]
        result = {"actions": result_structure}
        return json.dumps(result, indent=4)

    async def _generate_relative_action_step_proposer(
        self,
        examined_journey: Journey,
        step_guidelines: Sequence[Guideline] = [],
    ) -> RelativeActionStepSchema:
        prompt = await self._build_prompt(examined_journey, step_guidelines, _baseline_shots)

        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": 0.0},
        )
        with open("output_relative_action", "w") as f:
            f.write(response.content.model_dump_json(indent=2))
        return response.content

    def _format_shots(
        self,
        shots: Sequence[RelativeActionStepShot],
    ) -> str:
        return "\n".join(
            f"""
Example #{i}: ###
{self._format_shot(shot)}
###
"""
            for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(
        self,
        shot: RelativeActionStepShot,
    ) -> str:
        formatted_shot = ""
        formatted_shot += f"""
- **Context**:
{shot.description}
"""
        journey_text = get_journey_transition_map_text(shot.journey_steps, shot.journey_title)
        formatted_shot += f"""
- **Journey**:
    {journey_text}
"""
        formatted_shot += f"""
- **Expected Result**:
```json
{json.dumps(shot.expected_result.model_dump(mode="json", exclude_unset=True), indent=2)}
```"""
        return formatted_shot


book_hotel_shot_journey_steps = {
    "1": _JourneyStepWrapper(
        id="1",
        guideline_content=GuidelineContent(
            condition="",
            action="Ask the customer which hotel they would like to book.",
        ),
        parent_ids=[],
        follow_up_ids=["2"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "2": _JourneyStepWrapper(
        id="2",
        guideline_content=GuidelineContent(
            condition="The customer has specified the hotel name",
            action="Ask the customer how many guests will be staying.",
        ),
        parent_ids=["1"],
        follow_up_ids=["3"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "3": _JourneyStepWrapper(
        id="3",
        guideline_content=GuidelineContent(
            condition="The customer has specified the number of guests.",
            action="Ask the customer for the check-in and check-out dates.",
        ),
        parent_ids=["2"],
        follow_up_ids=["4"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "4": _JourneyStepWrapper(
        id="4",
        guideline_content=GuidelineContent(
            condition="he customer has provided check-in and check-out dates",
            action="Make sure it's available",
        ),
        parent_ids=["3"],
        follow_up_ids=["5", "6"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "5": _JourneyStepWrapper(
        id="5",
        guideline_content=GuidelineContent(
            condition="The availability check passed",
            action="Book it.",
        ),
        parent_ids=["4"],
        follow_up_ids=["7"],
        customer_dependent_action=False,
        requires_tool_calls=False,
    ),
    "6": _JourneyStepWrapper(
        id="6",
        guideline_content=GuidelineContent(
            condition="The availability check failed",
            action="Explain it to the user",
        ),
        parent_ids=["4"],
        follow_up_ids=[],
        customer_dependent_action=False,
        requires_tool_calls=False,
    ),
    "7": _JourneyStepWrapper(
        id="7",
        guideline_content=GuidelineContent(
            condition="The hotel booking was successful",
            action="Ask the customer to provide their email address to send the booking confirmation.",
        ),
        parent_ids=["5"],
        follow_up_ids=["8", "9"],
        customer_dependent_action=True,
        requires_tool_calls=False,
    ),
    "8": _JourneyStepWrapper(
        id="8",
        guideline_content=GuidelineContent(
            condition="The customer has provided a valid email address.",
            action="send it to them",
        ),
        parent_ids=["7"],
        follow_up_ids=["10"],
        customer_dependent_action=False,
        requires_tool_calls=True,
    ),
    "9": _JourneyStepWrapper(
        id="9",
        guideline_content=GuidelineContent(
            condition="The customer has provided an invalid email address.",
            action="Inform the customer that the email address is invalid and ask for a valid one.",
        ),
        parent_ids=["7"],
        follow_up_ids=["7"],
        customer_dependent_action=False,
        requires_tool_calls=True,
    ),
    "10": _JourneyStepWrapper(
        id="10",
        guideline_content=GuidelineContent(
            condition="The booking confirmation was sent successfully.",
            action="Ask the customer if there is anything else you can help with.",
        ),
        parent_ids=["6"],
        follow_up_ids=[],
        customer_dependent_action=False,
        requires_tool_calls=True,
    ),
}

example_1_shot = RelativeActionStepShot(
    description=" ",
    journey_title="",
    journey_steps=book_hotel_shot_journey_steps,
    expected_result=RelativeActionStepSchema(
        actions=[
            RelativeActionStepBatch(
                id="1",
                condition=book_hotel_shot_journey_steps["1"].guideline_content.condition,
                action=book_hotel_shot_journey_steps["1"].guideline_content.action,
                needs_rewrite_rational="The action is self contained",
                needs_rewrite=False,
            ),
            RelativeActionStepBatch(
                id="2",
                condition=book_hotel_shot_journey_steps["2"].guideline_content.condition,
                action=book_hotel_shot_journey_steps["2"].guideline_content.action,
                needs_rewrite_rational="The action is self contained",
                needs_rewrite=False,
            ),
            RelativeActionStepBatch(
                id="3",
                condition=book_hotel_shot_journey_steps["3"].guideline_content.condition,
                action=book_hotel_shot_journey_steps["3"].guideline_content.action,
                needs_rewrite_rational="The action is self contained",
                needs_rewrite=False,
            ),
            RelativeActionStepBatch(
                id="4",
                condition=book_hotel_shot_journey_steps["4"].guideline_content.condition,
                action=book_hotel_shot_journey_steps["4"].guideline_content.action,
                needs_rewrite_rational="The action does not specify what availability to check",
                needs_rewrite=True,
                former_reference="The availability refers to rooms matching the provided hotel, dates and number of guests.",
                rewritten_action="Make sure there is an available room in the asked hotel for the specified dates and number of guests.",
            ),
            RelativeActionStepBatch(
                id="5",
                condition=book_hotel_shot_journey_steps["5"].guideline_content.condition,
                action=book_hotel_shot_journey_steps["5"].guideline_content.action,
                needs_rewrite_rational="Need to explain what to book",
                needs_rewrite=True,
                former_reference="the booking refers to a hotel selected earlier",
                rewritten_action="Book the selected hotel for the specified dates and number of guests.",
            ),
            RelativeActionStepBatch(
                id="6",
                condition=book_hotel_shot_journey_steps["6"].guideline_content.condition,
                action=book_hotel_shot_journey_steps["6"].guideline_content.action,
                needs_rewrite_rational="I'ts clear that need to explain that the availability check failed, given the condition",
                needs_rewrite=False,
            ),
            RelativeActionStepBatch(
                id="7",
                condition=book_hotel_shot_journey_steps["7"].guideline_content.condition,
                action=book_hotel_shot_journey_steps["7"].guideline_content.action,
                needs_rewrite_rational="no need",
                needs_rewrite=False,
            ),
            RelativeActionStepBatch(
                id="8",
                condition=book_hotel_shot_journey_steps["8"].guideline_content.condition,
                action=book_hotel_shot_journey_steps["8"].guideline_content.action,
                needs_rewrite_rational="Need to clarify what to send",
                needs_rewrite=True,
                former_reference="previous steps says that need the mail address to send the booking confirmation.",
                rewritten_action="Send them the confirmation detailed of the booked hotel",
            ),
            RelativeActionStepBatch(
                id="9",
                condition=book_hotel_shot_journey_steps["9"].guideline_content.condition,
                action=book_hotel_shot_journey_steps["9"].guideline_content.action,
                needs_rewrite_rational="no need",
                needs_rewrite=False,
            ),
            RelativeActionStepBatch(
                id="10",
                condition=book_hotel_shot_journey_steps["10"].guideline_content.condition,
                action=book_hotel_shot_journey_steps["10"].guideline_content.action,
                needs_rewrite_rational="no need",
                needs_rewrite=False,
            ),
        ]
    ),
)

_baseline_shots: Sequence[RelativeActionStepShot] = [
    example_1_shot,
]

shot_collection = ShotCollection[RelativeActionStepShot](_baseline_shots)
