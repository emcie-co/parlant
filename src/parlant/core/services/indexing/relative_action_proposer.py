from dataclasses import dataclass
import json
import traceback
from typing import Optional, Sequence
from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.guideline_matching.generic.journey_node_selection_batch import (
    _JourneyEdge,
    _JourneyNode,
    JourneyNodeKind,
    build_node_wrappers,
    get_journey_transition_map_text,
)
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.guidelines import Guideline
from parlant.core.journeys import Journey
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.indexing.common import EvaluationError, ProgressReport
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.shots import Shot, ShotCollection


class RewrittenActionResult(DefaultBaseModel):
    index: str
    rewritten_actions: str


class RelativeActionProposition(DefaultBaseModel):
    actions: Sequence[RewrittenActionResult]


class RelativeActionBatch(DefaultBaseModel):
    index: str
    conditions: str
    action: str
    needs_rewrite_rational: str
    needs_rewrite: bool
    former_reference: Optional[str] = None
    rewritten_action: Optional[str] = None


class RelativeActionSchema(DefaultBaseModel):
    actions: Sequence[RelativeActionBatch]


@dataclass
class RelativeActionShot(Shot):
    journey_title: str
    journey_steps: dict[str, _JourneyNode]
    expected_result: RelativeActionSchema


class RelativeActionProposer:
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[RelativeActionSchema],
        service_registry: ServiceRegistry,
    ) -> None:
        self._logger = logger
        self._optimization_policy = optimization_policy

        self._schematic_generator = schematic_generator
        self._service_registry = service_registry

    async def propose_relative_action(
        self,
        examined_journey: Journey,
        step_guidelines: Sequence[Guideline] = [],
        journey_conditions: Sequence[Guideline] = [],
        progress_report: Optional[ProgressReport] = None,
    ) -> RelativeActionProposition:
        if progress_report:
            await progress_report.stretch(1)

        to_eval = [g for g in step_guidelines if g.content.action]

        if not to_eval:
            return RelativeActionProposition(actions=[])

        with self._logger.scope("RelativeActionProposer"):
            generation_attempt_temperatures = (
                self._optimization_policy.get_guideline_proposition_retry_temperatures(
                    hints={"type": self.__class__.__name__}
                )
            )

            last_generation_exception: Exception | None = None

            for generation_attempt in range(3):
                try:
                    result = await self._generate_relative_action_step_proposer(
                        examined_journey,
                        step_guidelines,
                        journey_conditions,
                        temperature=generation_attempt_temperatures[generation_attempt],
                    )

                    if progress_report:
                        await progress_report.increment(1)

                    rewritten_actions = []
                    for a in result.actions:
                        if a.needs_rewrite:
                            rewritten_actions.append(
                                RewrittenActionResult(
                                    index=a.index,
                                    rewritten_actions=a.rewritten_action,
                                )
                            )
                    return RelativeActionProposition(actions=rewritten_actions)
                except Exception as exc:
                    self._logger.warning(
                        f"RelativeActionProposer attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                    )

                    last_generation_exception = exc

            raise EvaluationError() from last_generation_exception

    def get_journey_text(
        self,
        examined_journey: Journey,
        step_guidelines: Sequence[Guideline],
        journey_conditions: Sequence[Guideline],
    ) -> str:
        node_wrappers: dict[str, _JourneyNode] = build_node_wrappers(step_guidelines)
        return get_journey_transition_map_text(
            nodes=node_wrappers,
            journey_title=examined_journey.title,
            journey_conditions=journey_conditions,
        )

    async def _build_prompt(
        self,
        examined_journey: Journey,
        step_guidelines: Sequence[Guideline],
        journey_conditions: Sequence[Guideline],
        shots: Sequence[RelativeActionShot],
    ) -> PromptBuilder:
        builder = PromptBuilder()

        builder.add_section(
            name="relative-action-proposer-general-instructions",
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
            name="relative-action-proposer-task-description",
            template="""
TASK DESCRIPTION
-----------------
Your task is to evaluate whether actions are self-contained and comprehensible without additional context.

You will be asked to:
1. Determine if the action description is sufficiently clear on its own:
    - Can an agent understand exactly what to do based solely on the condition and action?
    - Does the action rely on unstated context from previous journey steps?
    - Are there ambiguous references like "it", "that", that are unclear given only the condition?

2. Rewriting (when needed): If an action lacks clarity, rewrite it to be completely self-contained
    - Include all necessary context within the action description
    - Replace ambiguous pronouns and references with specific nouns from the journey context
    - Ensure the agent can execute the action without referring to the broader journey
    - Maintain the original intent without elaborating beyond what is explicitly provided

Common issues requiring clarification: Pronouns like "it", "that" when their referent is unclear from the condition alone
Standard, unambiguous pronouns (don't need clarification): We are in the context of customer service. In this context "they/them" referring to "the customer" and is completely standard and unambiguous.
No need to rewrite such actions (needs_rewrite is False)

""",
        )
        builder.add_section(
            name="relative-action-proposer-shots",
            template="""
EXAMPLES
-----------
{shots_text}
""",
            props={"shots_text": self._format_shots(shots)},
        )

        builder.add_section(
            name="relative-action-proposer-journey-steps",
            template=self.get_journey_text(
                examined_journey,
                step_guidelines,
                journey_conditions,
            ),
        )

        builder.add_section(
            name="relative-action-proposer-output-format",
            template="""
OUTPUT FORMAT
-----------
Use the following format to evaluate whether the action is relative and need rewriting:
Expected output (JSON):
```json
{{
    {result_structure_text}
}}
```
""",
            props={"result_structure_text": self._format_text(step_guidelines)},
        )
        return builder

    def _format_text(
        self,
        step_guidelines: Sequence[Guideline],
    ) -> str:
        node_wrappers: dict[str, _JourneyNode] = build_node_wrappers(step_guidelines)
        to_eval = {idx: node for idx, node in node_wrappers.items() if node.action}
        result_structure = [
            {
                "index": idx,
                "conditions": str([edge.condition for edge in node.incoming_edges]),
                "action": node.action,
                "needs_rewrite_rational": "<Brief explanation of is it refer to something that is not mentioned in the current step>",
                "needs_rewrite": "<BOOL>",
                "former_reference": "<information from previous steps that the definition is referring to>",
                "rewritten_action": "<str. Full, self-contained version of the action - include only if requires_rewrite is True>",
            }
            for idx, node in to_eval.items()
            if node.action
        ]
        result = {"actions": result_structure}
        return json.dumps(result, indent=4)

    async def _generate_relative_action_step_proposer(
        self,
        examined_journey: Journey,
        step_guidelines: Sequence[Guideline],
        journey_conditions: Sequence[Guideline],
        temperature: float,
    ) -> RelativeActionSchema:
        prompt = await self._build_prompt(
            examined_journey,
            step_guidelines,
            journey_conditions,
            _baseline_shots,
        )

        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": temperature},
        )

        return response.content

    def _format_shots(
        self,
        shots: Sequence[RelativeActionShot],
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
        shot: RelativeActionShot,
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
    "1": _JourneyNode(
        id="1",
        action="Ask the customer which hotel they would like to book.",
        incoming_edges=[],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The customer has specified the hotel name",
                source_node_index="1",
                target_node_index="2",
            )
        ],
        customer_dependent_action=True,
        kind=JourneyNodeKind.CHAT,
    ),
    "2": _JourneyNode(
        id="2",
        action="Ask them how many guests will be staying.",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The customer has specified the hotel name",
                source_node_index="1",
                target_node_index="2",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The customer has specified the number of guests.",
                source_node_index="2",
                target_node_index="3",
            )
        ],
        customer_dependent_action=True,
        kind=JourneyNodeKind.CHAT,
    ),
    "3": _JourneyNode(
        id="3",
        action="Ask the customer for the check-in and check-out dates.",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The customer has specified the number of guests.",
                source_node_index="2",
                target_node_index="3",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The customer has provided check-in and check-out dates.",
                source_node_index="3",
                target_node_index="4",
            )
        ],
        customer_dependent_action=True,
        kind=JourneyNodeKind.CHAT,
    ),
    "4": _JourneyNode(
        id="4",
        action="Make sure it's available",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The customer has provided check-in and check-out dates.",
                source_node_index="3",
                target_node_index="4",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The availability check passed",
                source_node_index="4",
                target_node_index="5",
            ),
            _JourneyEdge(
                target_guideline=None,
                condition="The availability check failed",
                source_node_index="4",
                target_node_index="6",
            ),
        ],
        customer_dependent_action=True,
        kind=JourneyNodeKind.CHAT,
    ),
    "5": _JourneyNode(
        id="5",
        action="Book it.",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The availability check passed",
                source_node_index="4",
                target_node_index="5",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The hotel booking was successful",
                source_node_index="5",
                target_node_index="7",
            )
        ],
        customer_dependent_action=False,
        kind=JourneyNodeKind.CHAT,
    ),
    "6": _JourneyNode(
        id="6",
        action="Explain it to the user",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The availability check failed",
                source_node_index="4",
                target_node_index="6",
            ),
        ],
        outgoing_edges=[],
        customer_dependent_action=False,
        kind=JourneyNodeKind.CHAT,
    ),
    "7": _JourneyNode(
        id="7",
        action="Ask the customer to provide their email address to send the booking confirmation.",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The hotel booking was successful",
                source_node_index="5",
                target_node_index="7",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The customer has provided a valid email address",
                source_node_index="7",
                target_node_index="8",
            ),
            _JourneyEdge(
                target_guideline=None,
                condition="The customer has provided an invalid email address",
                source_node_index="7",
                target_node_index="9",
            ),
        ],
        customer_dependent_action=True,
        kind=JourneyNodeKind.CHAT,
    ),
    "8": _JourneyNode(
        id="8",
        action="send it to them",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The customer has provided a valid email address",
                source_node_index="7",
                target_node_index="8",
            ),
            _JourneyEdge(
                target_guideline=None,
                condition="The customer has provided a valid email address",
                source_node_index="9",
                target_node_index="8",
            ),
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The booking confirmation was sent successfully",
                source_node_index="8",
                target_node_index="10",
            )
        ],
        customer_dependent_action=False,
        kind=JourneyNodeKind.TOOL,
    ),
    "9": _JourneyNode(
        id="9",
        action="Inform them that the email address is invalid and ask for a valid one.",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The customer has provided an invalid email address",
                source_node_index="7",
                target_node_index="9",
            ),
            _JourneyEdge(
                target_guideline=None,
                condition="The customer has provided an invalid email address",
                source_node_index="9",
                target_node_index="9",
            ),
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The customer has provided an invalid email address",
                source_node_index="9",
                target_node_index="9",
            ),
            _JourneyEdge(
                target_guideline=None,
                condition="The customer has provided a valid email address",
                source_node_index="9",
                target_node_index="8",
            ),
        ],
        customer_dependent_action=False,
        kind=JourneyNodeKind.TOOL,
    ),
    "10": _JourneyNode(
        id="10",
        action="Ask the customer if there is anything else you can help with.",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The booking confirmation was sent successfully",
                source_node_index="8",
                target_node_index="10",
            )
        ],
        outgoing_edges=[],
        customer_dependent_action=False,
        kind=JourneyNodeKind.TOOL,
    ),
}

example_1_shot = RelativeActionShot(
    description=" ",
    journey_title="",
    journey_steps=book_hotel_shot_journey_steps,
    expected_result=RelativeActionSchema(
        actions=[
            RelativeActionBatch(
                index="1",
                conditions=str(
                    [edge.condition for edge in book_hotel_shot_journey_steps["1"].incoming_edges]
                ),
                action=book_hotel_shot_journey_steps["1"].action or "",
                needs_rewrite_rational="The action is self-contained and clearly specifies what to ask the customer.",
                needs_rewrite=False,
            ),
            RelativeActionBatch(
                index="2",
                conditions=str(
                    [edge.condition for edge in book_hotel_shot_journey_steps["2"].incoming_edges]
                ),
                action=book_hotel_shot_journey_steps["2"].action or "",
                needs_rewrite_rational="The action is self-contained. 'them' refers to the customer so it's not ambiguous and no need to rewrite.",
                needs_rewrite=False,
            ),
            RelativeActionBatch(
                index="3",
                conditions=str(
                    [edge.condition for edge in book_hotel_shot_journey_steps["3"].incoming_edges]
                ),
                action=book_hotel_shot_journey_steps["3"].action or "",
                needs_rewrite_rational="The action is self-contained and clearly specifies what to ask the customer.",
                needs_rewrite=False,
            ),
            RelativeActionBatch(
                index="4",
                conditions=str(
                    [edge.condition for edge in book_hotel_shot_journey_steps["4"].incoming_edges]
                ),
                action=book_hotel_shot_journey_steps["4"].action or "",
                needs_rewrite_rational="The action does not specify what availability to check based on the condition alone.",
                needs_rewrite=True,
                former_reference="The availability refers to hotel rooms matching the specified hotel, dates, and number of guests from previous steps.",
                rewritten_action="Make sure there is an available room in the specified hotel for the provided dates and number of guests.",
            ),
            RelativeActionBatch(
                index="5",
                conditions=str(
                    [edge.condition for edge in book_hotel_shot_journey_steps["5"].incoming_edges]
                ),
                action=book_hotel_shot_journey_steps["5"].action or "",
                needs_rewrite_rational="The action does not specify what to book based on the condition alone.",
                needs_rewrite=True,
                former_reference="The booking refers to the hotel reservation with the specified details from previous steps.",
                rewritten_action="Book the hotel for the specified dates and number of guests.",
            ),
            RelativeActionBatch(
                index="6",
                conditions=str(
                    [edge.condition for edge in book_hotel_shot_journey_steps["6"].incoming_edges]
                ),
                action=book_hotel_shot_journey_steps["6"].action or "",
                needs_rewrite_rational="'it' refers to the fact that the availability check failed. I'ts clear that need to explain that the availability check failed, given the condition",
                needs_rewrite=False,
            ),
            RelativeActionBatch(
                index="7",
                conditions=str(
                    [edge.condition for edge in book_hotel_shot_journey_steps["7"].incoming_edges]
                ),
                action=book_hotel_shot_journey_steps["7"].action or "",
                needs_rewrite_rational="The action is self-contained and clearly specifies what to ask the customer and why.",
                needs_rewrite=False,
            ),
            RelativeActionBatch(
                index="8",
                conditions=str(
                    [edge.condition for edge in book_hotel_shot_journey_steps["8"].incoming_edges]
                ),
                action=book_hotel_shot_journey_steps["8"].action or "",
                needs_rewrite_rational="The action does not specify what to send based on the condition alone.",
                needs_rewrite=True,
                former_reference="Previous step mentions asking for email address to send booking confirmation.",
                rewritten_action="Send them the booking confirmation.",
            ),
            RelativeActionBatch(
                index="9",
                conditions=str(
                    [edge.condition for edge in book_hotel_shot_journey_steps["9"].incoming_edges]
                ),
                action=book_hotel_shot_journey_steps["9"].action or "",
                needs_rewrite_rational="The action is self-contained. 'them' refers to the customer so it's not ambiguous and no need to rewrite",
                needs_rewrite=False,
            ),
            RelativeActionBatch(
                index="10",
                conditions=str(
                    [edge.condition for edge in book_hotel_shot_journey_steps["10"].incoming_edges]
                ),
                action=book_hotel_shot_journey_steps["10"].action or "",
                needs_rewrite_rational="The action is self-contained and clearly specifies what to ask the customer.",
                needs_rewrite=False,
            ),
        ]
    ),
)

_baseline_shots: Sequence[RelativeActionShot] = [
    example_1_shot,
]

shot_collection = ShotCollection[RelativeActionShot](_baseline_shots)
