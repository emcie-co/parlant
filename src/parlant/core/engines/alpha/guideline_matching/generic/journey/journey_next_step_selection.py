from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import traceback
from typing import Any, Optional, Sequence, cast
from parlant.core.common import Criticality, DefaultBaseModel, JSONSerializable
from parlant.core.engines.alpha.guideline_matching.generic.common import internal_representation
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingBatchError,
    GuidelineMatchingBatchResult,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matching_context import (
    GuidelineMatchingContext,
)
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId, GuidelineStore
from parlant.core.journeys import Journey
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.shots import Shot, ShotCollection

PRE_ROOT_INDEX = "0"
ROOT_INDEX = "1"

FORK_NODE_ACTION_STR = (
    "No action necessary - always advance to the next step based on the relevant transition"
)

EXIT_NODE_ACTION = "Exit the journey"


class JourneyNodeKind(Enum):
    FORK = "fork"
    CHAT = "chat"
    TOOL = "tool"
    NA = "NA"


class JourneyNextStepSelectionSchema(DefaultBaseModel):
    journey_continues: bool
    current_step_completed_rationale: str
    current_step_completed: bool
    next_step_rationale: str
    applied_condition_id: str


@dataclass
class _JourneyNode:
    id: str
    action: str
    kind: JourneyNodeKind
    customer_dependent_action: bool
    customer_action_description: Optional[str] = None
    agent_dependent_action: Optional[bool] = None
    agent_action_description: Optional[str] = None
    guideline: Guideline | None = None


@dataclass
class _JourneyEdge:
    condition: str
    target_node_action: str | None


@dataclass
class JourneyNextStepSelectionShot(Shot):
    interaction_events: Sequence[Event]
    journey_title: str
    conditions: Sequence[str]
    current_node: _JourneyNode
    follow_up_conditions: dict[str, _JourneyEdge]
    expected_result: JourneyNextStepSelectionSchema


class JourneyNextStepSelection:
    def __init__(
        self,
        logger: Logger,
        guideline_store: GuidelineStore,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[JourneyNextStepSelectionSchema],
        examined_journey: Journey,
        context: GuidelineMatchingContext,
        node_guidelines: Sequence[Guideline] = [],
        journey_path: Sequence[str | None] = [],
        journey_conditions: Sequence[Guideline] = [],
    ) -> None:
        self._logger = logger

        self._guideline_store = guideline_store

        self._optimization_policy = optimization_policy
        self._schematic_generator = schematic_generator

        self._context = context
        self._examined_journey = examined_journey
        self._journey_conditions = journey_conditions

        self._guideline_id_to_guideline: dict[GuidelineId, Guideline] = {
            g.id: g for g in node_guidelines
        }
        self._guideline_id_to_node_index: dict[GuidelineId, str] = {
            g.id: self._get_guideline_node_index(g) for g in node_guidelines
        }
        self._node_index_to_guideline_id: dict[str, GuidelineId] = {
            self._guideline_id_to_node_index[id]: id for id in self._guideline_id_to_node_index
        }

        (
            self._current_node,
            self._follow_up_conditions,
            self._condition_to_path,
            self._previous_path,
            self._reset_journey,
        ) = self.build_node_wrappers(journey_path)

    def _get_guideline_node_index(self, guideline: Guideline) -> str:
        return str(
            cast(dict[str, JSONSerializable], guideline.metadata["journey_node"]).get(
                "index", "-1"
            ),
        )

    def build_node_wrappers(
        self,
        previous_path: Sequence[str | None],
    ) -> tuple[
        _JourneyNode, dict[str, _JourneyEdge], dict[str, Sequence[str]], Sequence[str | None], bool
    ]:
        def _get_reachable_follow_ups(
            guideline_id: GuidelineId,
            guideline_id_to_guideline: dict[GuidelineId, Guideline],
        ) -> list[dict[str, JSONSerializable]]:
            guideline = guideline_id_to_guideline[guideline_id]
            return cast(
                list[dict[str, JSONSerializable]],
                cast(dict[str, JSONSerializable], guideline.metadata["journey_node"]).get(
                    "reachable_follow_ups", []
                ),
            )

        def _create_node(
            guideline_id: GuidelineId,
        ) -> _JourneyNode:
            guideline = self._guideline_id_to_guideline[guideline_id]

            kind = JourneyNodeKind(
                cast(dict[str, Any], guideline.metadata.get("journey_node", {})).get("kind", "NA")
            )
            customer_dependent_action = cast(
                dict[str, bool], guideline.metadata.get("customer_dependent_action_data", {})
            ).get("is_customer_dependent", False)

            if kind == JourneyNodeKind.FORK:
                action: str = FORK_NODE_ACTION_STR
            elif not internal_representation(guideline).action:
                action = FORK_NODE_ACTION_STR
            else:
                action = cast(str, internal_representation(guideline).action)

            node = _JourneyNode(
                id=self._get_guideline_node_index(guideline),
                action=action,
                kind=kind,
                customer_dependent_action=customer_dependent_action,
                customer_action_description=cast(
                    dict[str, str | None],
                    guideline.metadata.get("customer_dependent_action_data", {}),
                ).get("customer_action", None),
                agent_dependent_action=cast(
                    dict[str, bool], guideline.metadata.get("customer_dependent_action_data", {})
                ).get(
                    "is_agent_dependent",
                    not customer_dependent_action and kind == JourneyNodeKind.CHAT,
                ),
                agent_action_description=cast(
                    dict[str, str | None],
                    guideline.metadata.get("customer_dependent_action_data", {}),
                ).get("agent_action", None),
                guideline=guideline,
            )
            return node

        reset_journey = False
        if not previous_path:
            root_g = self._guideline_id_to_guideline[self._node_index_to_guideline_id[ROOT_INDEX]]
            follow_ups = cast(
                dict[str, Sequence[GuidelineId]], root_g.metadata.get("journey_node", {})
            ).get("follow_ups", [])
            if (not root_g.content.action) and len(follow_ups) == 1:
                # Root has a single follow up, so that follow up is first to be executed
                current_g_id = follow_ups[0]
            else:
                current_g_id = root_g.id
        else:
            if previous_path[-1]:
                current_g_id = self._node_index_to_guideline_id[previous_path[-1]]
            else:
                current_g_id = self._node_index_to_guideline_id[ROOT_INDEX]
                reset_journey = True
                previous_path = []

        current_node = _create_node(current_g_id)

        follow_up_conditions: dict[str, _JourneyEdge] = {}

        reachable_follow_ups = _get_reachable_follow_ups(
            current_g_id, self._guideline_id_to_guideline
        )

        condition_to_path: dict[str, Sequence[str]] = {}

        for i, follow_up in enumerate(reachable_follow_ups, start=1):
            journey_node_path: Sequence[str] = cast(Sequence[str], follow_up["path"])
            transition_condition: str = cast(str, follow_up["condition"])

            target_node_action = None

            if journey_node_path[-1] in self._node_index_to_guideline_id:
                target_g = self._guideline_id_to_guideline[
                    self._node_index_to_guideline_id[journey_node_path[-1]]
                ]
                target_node_action = internal_representation(target_g).action

            follow_up_conditions[str(i)] = _JourneyEdge(
                condition=transition_condition,
                target_node_action=target_node_action,
            )
            condition_to_path[str(i)] = journey_node_path

        return current_node, follow_up_conditions, condition_to_path, previous_path, reset_journey

    async def process(self) -> GuidelineMatchingBatchResult:
        prompt = self._build_prompt(shots=await self.shots())

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
                    hints={
                        "temperature": generation_attempt_temperatures[generation_attempt],
                    },
                )
                self._logger.trace(f"Completion:\n{inference.content.model_dump_json(indent=2)}")
                with open("dumps/journey/journey next step/output.txt", "w") as f:
                    f.write(inference.content.model_dump_json(indent=2))
                if inference.content.applied_condition_id:
                    if inference.content.applied_condition_id == "None":
                        # Exit journey
                        journey_path = list(self._previous_path) + [None]
                        return GuidelineMatchingBatchResult(
                            matches=[
                                GuidelineMatch(
                                    guideline=self._guideline_id_to_guideline[
                                        self._node_index_to_guideline_id[ROOT_INDEX]
                                    ],
                                    score=10,
                                    rationale=f"Root guideline was selected indicating should exit the journey, the rational for this choice: {inference.content.next_step_rationale}",
                                    metadata={
                                        "journey_path": journey_path,
                                        "step_selection_journey_id": self._examined_journey.id,
                                    },
                                )
                            ],
                            generation_info=inference.info,
                        )
                    elif inference.content.applied_condition_id == "0":
                        # Stay in the same node
                        matched_guideline = self._guideline_id_to_guideline[
                            self._node_index_to_guideline_id[self._current_node.id]
                        ]
                        return GuidelineMatchingBatchResult(
                            matches=[
                                GuidelineMatch(
                                    guideline=matched_guideline,
                                    score=10,
                                    rationale=f"This guideline was selected as part of a 'journey' - a sequence of actions that are performed in order. Use this rationale to better understand how the conversation got to its current point. The rationale for choosing this specific step in the journey was: {inference.content.next_step_rationale}",
                                    metadata={
                                        "journey_path": self._previous_path
                                        if self._previous_path
                                        else [self._current_node.id],
                                        "step_selection_journey_id": self._examined_journey.id,
                                    },
                                )
                            ],
                            generation_info=inference.info,
                        )
                    else:
                        condition_id = inference.content.applied_condition_id
                        if condition_id in self._condition_to_path:
                            next_path = self._condition_to_path[condition_id]
                            next_node = next_path[-1]
                            # Journey has finished
                            if (
                                next_node == "None"
                                or self._guideline_id_to_guideline[
                                    self._node_index_to_guideline_id[next_node]
                                ].content.action
                                is None
                            ):
                                journey_path = list(self._previous_path) + [None]

                                return GuidelineMatchingBatchResult(
                                    matches=[
                                        GuidelineMatch(
                                            guideline=self._guideline_id_to_guideline[
                                                self._node_index_to_guideline_id[ROOT_INDEX]
                                            ],
                                            score=10,
                                            rationale=f"Root guideline was selected indicating should exit the journey, the rational for this choice: {inference.content.next_step_rationale}",
                                            metadata={
                                                "journey_path": journey_path,
                                                "step_selection_journey_id": self._examined_journey.id,
                                            },
                                        )
                                    ],
                                    generation_info=inference.info,
                                )
                            else:
                                if not self._previous_path:
                                    # we started from the root and root was completed, so include it in journey path
                                    journey_path = cast(
                                        list[str | None], [self._current_node.id] + list(next_path)
                                    )
                                else:
                                    journey_path = list(self._previous_path) + list(next_path)
                                matched_guideline = self._guideline_id_to_guideline[
                                    self._node_index_to_guideline_id[next_node]
                                ]
                                return GuidelineMatchingBatchResult(
                                    matches=[
                                        GuidelineMatch(
                                            guideline=matched_guideline,
                                            score=10,
                                            rationale=f"This guideline was selected as part of a 'journey' - a sequence of actions that are performed in order. Use this rationale to better understand how the conversation got to its current point. The rationale for choosing this specific step in the journey was: {inference.content.next_step_rationale}",
                                            metadata={
                                                "journey_path": journey_path,
                                                "step_selection_journey_id": self._examined_journey.id,
                                            },
                                        )
                                    ],
                                    generation_info=inference.info,
                                )
                        else:  # condition index invalid
                            return GuidelineMatchingBatchResult(
                                matches=[],
                                generation_info=inference.info,
                            )
            except Exception as exc:
                self._logger.warning(
                    f"Attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                )

                last_generation_exception = exc

        raise GuidelineMatchingBatchError() from last_generation_exception

    def get_journey_transition_map_text(
        self,
        current_node: _JourneyNode,
        follow_up_conditions: dict[str, _JourneyEdge],
        journey_title: str,
        journey_description: str = "",
        journey_conditions: Sequence[Guideline] = [],
    ) -> str:
        if journey_description:
            journey_description_str = f"\nJourney Description: {journey_description}"
        else:
            journey_description_str = ""
        if journey_conditions:
            journey_conditions_str = " OR ".join(
                f'"{g.content.condition}"' for g in journey_conditions
            )
            journey_conditions_str = f"\nJourney activation condition: {journey_conditions_str}"
        else:
            journey_conditions_str = ""
        journey_restart = ""
        if self._reset_journey:
            journey_restart = """
Important:
This journey has been restarted after a previous execution. 
Carefully determine what information from the previous execution can still be assumed valid and what needs to be asked again. 
When in doubt, prefer to re-verify previous decisions unless it's clear they haven't changed"""

        flags_str = "Step Flags:\n"

        # Customer / Agent dependent flags
        if current_node.customer_dependent_action:
            if current_node.customer_action_description:
                flags_str += f'- CUSTOMER DEPENDENT: This action requires an action from the customer to be considered complete. It is completed if the following action was completed: "{current_node.customer_action_description}" \n'
            else:
                flags_str += "- CUSTOMER DEPENDENT: This action requires an action from the customer to be considered complete. Mark it as complete if the customer answered the question in the action, if there is one.\n"
        if current_node.kind == JourneyNodeKind.CHAT and current_node.agent_dependent_action:
            flags_str += "- REQUIRES AGENT ACTION: This step may require the agent to say something for it to be completed. Only advance through it if the agent performed the described action."
        elif current_node.kind == JourneyNodeKind.FORK:
            flags_str += "- This step serves as a transition and does not require evaluating completion; it only needs to determine the appropriate next transition.\n"
        elif current_node.kind == JourneyNodeKind.TOOL:
            flags_str += "- TOOL EXECUTION: This step is considered complete as long as the tool has been executed.\n"

        current_node_description = f"""
{current_node.action}
{flags_str}
"""

        follow_ups_nodes_description = ""
        for id, e in follow_up_conditions.items():
            # target_node_action = e.target_node_action if e.target_node_action else EXIT_NODE_ACTION
            follow_ups_nodes_description += f"""
Condition ({id}): {e.condition}
"""

        transition_description = f"""

Journey: {journey_title}
{journey_conditions_str}{journey_description_str}

CURRENT STEP - 
{current_node_description}

POSSIBLE TRANSITIONS - 
If the journey is not applicable anymore, return None as next step. 

If the current step hasn't completed, return '0' as the condition id.

In any other case, return next step from the following possible transitions:
{follow_ups_nodes_description} 
{journey_restart}
"""
        return transition_description

    def _get_output_format_section(self) -> str:
        return """
IMPORTANT: Please provide your answer in the following JSON format.

OUTPUT FORMAT
-----------------
- Fill in the following fields as instructed. Each field is required unless otherwise specified.

```json
{
"journey_continues": <bool, whether the journey should continued. Reminder: If you are already executing journey steps (i.e., there is a "last_step"), the journey almost always continues. The activation condition is ONLY for starting new journeys, NOT for validating ongoing ones.>,
"current_step_completed_rationale": "<str, short explanation of whether current step completed>",
"current_step_completed": <bool, whether the current step completed.>,
"next_step_rationale": "<str, explanation for which condition best fits and why. Consider all the information provided in CURRENT and EARLIER messages>",
"applied_condition_id": "<str, id of the applied condition, '0' if current step hasn't completed or 'None' if the journey should not continue>"
}
```
"""

    async def shots(self) -> Sequence[JourneyNextStepSelectionShot]:
        return await shot_collection.list()

    def _format_shots(self, shots: Sequence[JourneyNextStepSelectionShot]) -> str:
        return "\n".join(
            f"Example #{i}: {shot.journey_title}\n{self._format_shot(shot)}"
            for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, shot: JourneyNextStepSelectionShot) -> str:
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
        formatted_shot += self.get_journey_transition_map_text(
            current_node=shot.current_node,
            follow_up_conditions=shot.follow_up_conditions,
            journey_title=shot.journey_title,
            journey_conditions=[
                Guideline(
                    id=GuidelineId(f"c-{i}"),
                    creation_utc=datetime.now(timezone.utc),
                    metadata={"journey_node": {"journey_id": "journey"}},
                    content=GuidelineContent(
                        condition=c,
                        action=None,
                    ),
                    enabled=False,
                    criticality=Criticality.HIGH,
                    tags=[],
                )
                for i, c in enumerate(shot.conditions)
            ],
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
        shots: Sequence[JourneyNextStepSelectionShot],
    ) -> PromptBuilder:
        builder = PromptBuilder(on_build=lambda prompt: self._logger.trace(f"Prompt:\n{prompt}"))

        builder.add_agent_identity(self._context.agent)

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
Analyze the current conversation state and determine the next appropriate journey step, by evaluating which condition holds based on the last step that was performed and the current state of the conversation.
    """,
            props={"agent_name": self._context.agent.name},
        )
        builder.add_section(
            name="journey-next-step-selection-task-description",
            template="""
TASK DESCRIPTION
-------------------
## 1: Journey Context Check
Determine if the conversation should continue within the current journey.
Once a journey has begun, continue following it unless the customer explicitly indicates they no longer want to pursue the journey's original goal.
**Important**: The activation condition only starts a journey. It does NOT need to remain true for the journey to continue.
If the journey should end, set `applied_condition_id` to `"None"`.

## 2: Current Step Completion
Evaluate whether the last executed step is complete:
    - For CUSTOMER_DEPENDENT steps: The step is completed if customer has provided the required information. It can be either after being asked or proactively in earlier messages.  
    If the customer provided the information, set current_step_completed to 'true'.
    If not, set completed to 'current_step_completed' as 'false' and applied_condition_id as '0'. 

    - For REQUIRES AGENT ACTION steps: The step is completed if the agent has performed the required communication or action. 
    If so, set current_step_completed to 'true'. 
    If not, set 'current_step_completed' as 'false' and applied_condition_id as '0'. 

    - For TOOL EXECUTION steps: The tool was executed, and its result will appear as a staged event. Evaluate which condition applies for the next transition based on the tool result..
        Note that the tool execution is the final action in the interaction, meaning all message exchanges occurred beforehand. Make sure to consider this order in your evaluation.

## 3: Journey Advancement
If the journey continues AND the current step is complete, choose the next step by evaluating which condition best fits.    
The condition contains one or more sub-conditions that must all be evaluated and met for the condition to be considered the best match.

Select the condition ID that best matches:
    - Consider all the condition parts in your evaluation.
    - Only ONE transition condition should be the best fit
    - Return its ID as `applied_condition_id`

**How to determine if condition / sub condition is fulfilled if the action is CUSTOMER DEPENDENT:**
The action is fulfilled if the customer has provided the required information. It can be either after being asked or proactively in earlier messages. 
That means, the agent does not need to ask for something for the action to be fulfilled.
Note that the customer may provide multiple details at once (in one message), and you should consider all of them to identify the most relevant condition.
Also, note that the customer may provide some of the answers in previous messages, consider those answers too. 
The answers may not arrive in the order we expect. An answer for a later step may have been provided in earlier messages. As long as we have the required 
information, the condition is considered met.

**Handling partial condition matches**
Conditions may contain multiple sub-conditions (e.g., "customer provided X AND agent did Y AND customer hasn't provided Z")
If ALL information has been provided (for example also Z) and no condition is fully satisfied, select the condition with the MOST satisfied sub-conditions 
This represents the path closest to completion, even if technically the condition isn't met

Important - You tend to ignore customer action completions that were provided in previous messages. It's important to notice ALL customer messages
    history in details and evaluate which information was already provided. Please correct yourself in the future.

You will be given a description of the current step that need to execute, and the conditions of the following transitions later in this prompt.
    """,
        )
        builder.add_section(
            name="journey-next-step-selection-examples",
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

        builder.add_customer_identity(self._context.customer, self._context.session)
        builder.add_context_variables(self._context.context_variables)
        builder.add_glossary(self._context.terms)
        builder.add_capabilities_for_guideline_matching(self._context.capabilities)
        builder.add_interaction_history(self._context.interaction_history)
        builder.add_staged_tool_events(self._context.staged_events)

        builder.add_section(
            name="journey-next-step-selection-journey-steps",
            template=self.get_journey_transition_map_text(
                current_node=self._current_node,
                follow_up_conditions=self._follow_up_conditions,
                journey_title=self._examined_journey.title,
                journey_description=self._examined_journey.description,
                journey_conditions=self._journey_conditions,
            ),
        )
        builder.add_section(
            name="journey-next-step-selection-output-format",
            template="""{output_format}""",
            props={"output_format": self._get_output_format_section()},
        )

        builder.add_section(
            name="journey-general_reminder-section",
            template="""Reminder - carefully consider all restraints and instructions. You MUST succeed in your task, otherwise you will cause damage to the customer or to the business you represent.""",
        )

        return builder


def _make_event(e_id: str, source: EventSource, message: str) -> Event:
    return Event(
        id=EventId(e_id),
        source=source,
        kind=EventKind.MESSAGE,
        creation_utc=datetime.now(timezone.utc),
        offset=0,
        trace_id="",
        data={"message": message},
        deleted=False,
        metadata={},
    )


# Example 1: Step not yet complete

example_1_events = [
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
        "I'd like to book a taxi to JFK Airport at 5 PM, please. I'll pay by cash.",
    ),
]

example_1_current_node = _JourneyNode(
    id="",
    kind=JourneyNodeKind.CHAT,
    action="Ask the customer for their desired pick up location",
    customer_dependent_action=True,
    customer_action_description="The customer provided their desired pick up location",
)

example_1_follow_up_nodes = {
    "1": _JourneyEdge(
        condition="The customer's desired pick up location is in NYC and customer hasn't provided their destination location yet",
        target_node_action="Ask where their destination is",
    ),
    "2": _JourneyEdge(
        condition="The customer's desired pick up location is outside of NYC and the agent hasn't informed the customer that we do not operate outside of NYC",
        target_node_action="Inform the customer that we do not operate outside of NYC",
    ),
    "3": _JourneyEdge(
        condition="The customer's desired pick up location is in NYC and they provided their destination location but hasn't provided the pickup time yet",
        target_node_action="Ask for the customer's desired pick up time",
    ),
    "4": _JourneyEdge(
        condition="The customer's desired pick up location is in NYC and and they provided their destination location and pickup time but the agent hasn't booked the taxi ride yet",
        target_node_action="Book the taxi ride as the customer requested",
    ),
    "5": _JourneyEdge(
        condition="The customer's desired pick up location is in NYC and they provided their destination location and pickup time and the agent booked the taxi ride "
        "but the agent hasn't ask the customer if they want to pay in cash or credit",
        target_node_action="Ask the customer if they want to pay in cash or credit",
    ),
}


example_1_expected = JourneyNextStepSelectionSchema(
    journey_continues=True,
    current_step_completed_rationale="The customer has NOT provided the pickup location, which is what the current step asks for. The current step is therefore incomplete",
    current_step_completed=False,
    next_step_rationale="Current step hasn't completed so applied condition is '0",
    applied_condition_id="0",
)


example_2_events = [
    _make_event(
        "11",
        EventSource.AI_AGENT,
        "Welcome to our taxi service! How can I help you today?",
    ),
    _make_event(
        "23",
        EventSource.CUSTOMER,
        "I'd like a taxi from 20 W 34th St., NYC at 6 AM, please. I'll pay by cash.",
    ),
]


example_2_current_node = _JourneyNode(
    id="",
    kind=JourneyNodeKind.CHAT,
    action="Welcome the customer to the taxi service",
    customer_dependent_action=False,
)

example_2_follow_up_nodes = {
    "1": _JourneyEdge(
        condition="The customer did not provide their desired pick up location.",
        target_node_action="Ask the customer for their desired pick up location",
    ),
    "2": _JourneyEdge(
        condition="The customer provided their desired pick up location and the location is in NYC, and the customer hasn't provided their destination location yet",
        target_node_action="Ask where their destination is",
    ),
    "3": _JourneyEdge(
        condition="The customer provided their desired pick up location and the location is outside of NYC and the agent hasn't informed the customer that we do not operate outside of NYC",
        target_node_action="Inform the customer that we do not operate outside of NYC",
    ),
    "4": _JourneyEdge(
        condition="The customer provided their desired pick up location which is in NYC and also provided their destination location but hasn't provided the pickup time yet",
        target_node_action="Ask for the customer's desired pick up time",
    ),
    "5": _JourneyEdge(
        condition="The customer provided their desired pick up location which is in NYC and also provided their destination location and the pickup time and the agent hasn't booked the taxi ride yet",
        target_node_action="Book the taxi ride as the customer requested",
    ),
    "6": _JourneyEdge(
        condition="The customer provided their desired pick up location which is in NYC and also provided their destination location and the pickup time and the agent booked the taxi ride "
        "and the agent hasn't ask the customer if they want to pay in cash or credit",
        target_node_action="Ask the customer if they want to pay in cash or credit",
    ),
}
example_2_expected = JourneyNextStepSelectionSchema(
    journey_continues=True,
    current_step_completed_rationale="The agent welcomed the customer, so current step completed.",
    current_step_completed=True,
    next_step_rationale="The customer provided a pick up location in NYC and a pick up time, but has not provided a destination, so condition 2 best holds.",
    applied_condition_id="2",
)

example_3_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Hello, I'm Helen Jay, I'd like to take a loan for 10,000$ and put stocks as collateral, is that possible?",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "Sure, I can help you with that. What type of loan are you interested in?",
    ),
    _make_event(
        "34",
        EventSource.CUSTOMER,
        "What do you mean?",
    ),
    _make_event(
        "45",
        EventSource.AI_AGENT,
        "Are you interested in a business or a personal loan?",
    ),
    _make_event(
        "56",
        EventSource.CUSTOMER,
        "Does it matter?",
    ),
    _make_event(
        "67",
        EventSource.AI_AGENT,
        "We need to know this information to proceed with the loan application, as the two loan types have different requirements.",
    ),
    _make_event(
        "78",
        EventSource.CUSTOMER,
        "Ok, let me check for a sec",
    ),
    _make_event(
        "89",
        EventSource.AI_AGENT,
        "Sure, take your time",
    ),
    _make_event(
        "78",
        EventSource.CUSTOMER,
        "It's a loan for my restaurant",
    ),
]


example_3_current_node = _JourneyNode(
    id="",
    kind=JourneyNodeKind.CHAT,
    action="Ask for the type of loan: Personal or Business.",
    customer_dependent_action=True,
    customer_action_description="the customer specified which type of loan they'd like to take",
)

example_3_follow_up_nodes = {
    "1": _JourneyEdge(
        condition="Customer chose personal loan and the customer has not provided their desired loan amount",
        target_node_action="Ask for the desired loan amount.",
    ),
    "2": _JourneyEdge(
        condition="Customer chose business loan and the customer has not provided their desired loan amount",
        target_node_action="Ask for the desired loan amount.",
    ),
    "3": _JourneyEdge(
        condition="Customer chose personal loan and the customer provided their desired loan amount and hasn't provided their employment status",
        target_node_action="Ask for employment status.",
    ),
    "4": _JourneyEdge(
        condition="Customer chose business loan and the customer provided their desired loan amount but hasn't provided the collateral",
        target_node_action="Ask for collateral.",
    ),
    "5": _JourneyEdge(
        condition="Customer chose personal loan and the customer provided their desired loan amount provided the employment status but agent has not yet confirmed the application",
        target_node_action="Review and confirm application.",
    ),
    "6": _JourneyEdge(
        condition="Customer chose business loan and the customer provided their desired loan amount and provided the collateral which is a digital asset but agent has not yet confirmed the application",
        target_node_action="Review and confirm application.",
    ),
}


example_3_expected = JourneyNextStepSelectionSchema(
    journey_continues=True,
    current_step_completed_rationale="The customer wants a loan for their restaurant, making it a business loan. So current step completed",
    current_step_completed=True,
    next_step_rationale="The customer has already specified in previous messages the amount of the loan and stocks as collateral which are digital. "
    "The agent hasn't reviewed and confirmed the application so condition 6 is most appropriate",
    applied_condition_id="6",
)


example_4_events = [
    _make_event(
        "11",
        EventSource.AI_AGENT,
        "Welcome to our taxi service! How can I help you today?",
    ),
    _make_event(
        "23",
        EventSource.CUSTOMER,
        "I'd like a taxi from 20 W 34th St., NYC to the Plaza Hotel at 6 AM, please. I'll pay by cash.",
    ),
]


example_4_current_node = _JourneyNode(
    id="",
    kind=JourneyNodeKind.CHAT,
    action="Welcome the customer to the taxi service",
    customer_dependent_action=False,
)

example_4_follow_up_nodes = {
    "1": _JourneyEdge(
        condition="The customer did not provide their desired pick up location.",
        target_node_action="Ask the customer for their desired pick up location",
    ),
    "2": _JourneyEdge(
        condition="The customer provided their desired pick up location and the location is in NYC, and the customer hasn't provided their destination location yet",
        target_node_action="Ask where their destination is",
    ),
    "3": _JourneyEdge(
        condition="The customer provided their desired pick up location and the location is outside of NYC and the agent hasn't informed the customer that we do not operate outside of NYC",
        target_node_action="Inform the customer that we do not operate outside of NYC",
    ),
    "4": _JourneyEdge(
        condition="The customer provided their desired pick up location which is in NYC and also provided their destination location but hasn't provided the pickup time yet",
        target_node_action="Ask for the customer's desired pick up time",
    ),
}

example_4_expected = JourneyNextStepSelectionSchema(
    journey_continues=True,
    current_step_completed_rationale="The agent welcomed the customer, so current step completed.",
    current_step_completed=True,
    next_step_rationale="The customer provided pickup location (NYC), destination (Plaza Hotel), time (6 AM), and payment method (cash). All conditions have some unsatisfied parts but condition 4 has the most satisfied sub-conditions",
    applied_condition_id="4",
)


# Example 5: Customer provided information proactively across multiple messages
example_5_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Hi, I need a loan. I need 50,000.",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "I'd be happy to help you with a loan. To get started, what type of loan are you interested in - personal or business?",
    ),
    _make_event(
        "34",
        EventSource.CUSTOMER,
        "I'ts for me. I'm unemployed right now so it can help me.",
    ),
]


example_5_current_node = _JourneyNode(
    id="",
    kind=JourneyNodeKind.CHAT,
    action="Ask for the type of loan: Personal or Business.",
    customer_dependent_action=True,
    customer_action_description="The customer specified if the loan is personal or business",
)

example_5_follow_up_nodes = {
    "1": _JourneyEdge(
        condition="Customer chose personal loan and the customer has not provided their desired loan amount",
        target_node_action="Ask for the desired loan amount.",
    ),
    "2": _JourneyEdge(
        condition="Customer chose business loan and the customer has not provided their desired loan amount",
        target_node_action="Ask for the desired loan amount.",
    ),
    "3": _JourneyEdge(
        condition="Customer chose personal loan and the customer provided their desired loan amount and hasn't provided their employment status",
        target_node_action="Ask for employment status.",
    ),
    "4": _JourneyEdge(
        condition="Customer chose business loan and the customer provided their desired loan amount but hasn't provided the collateral",
        target_node_action="Ask for collateral.",
    ),
    "5": _JourneyEdge(
        condition="Customer chose personal loan and the customer provided their desired loan amount provided the employment status but agent has not yet confirmed the application",
        target_node_action="Review and confirm application.",
    ),
    "6": _JourneyEdge(
        condition="Customer chose business loan and the customer provided their desired loan amount and provided the collateral which is a digital asset but agent has not yet confirmed the application",
        target_node_action="Review and confirm application.",
    ),
}


example_5_expected = JourneyNextStepSelectionSchema(
    journey_continues=True,
    current_step_completed_rationale="The customer said the loan is for them because they unemployed, so it's personal loan.",
    current_step_completed=True,
    next_step_rationale="The customer has already mentioned in initial message that they need 50,000, so they provided the amount in earlier messages and it considered complete."
    " Also, they provided the employment status by saying they unemployed. The agent has not confirmed the application so condition 5 fits.",
    applied_condition_id="5",
)


_baseline_shots: Sequence[JourneyNextStepSelectionShot] = [
    JourneyNextStepSelectionShot(
        description="Example 1 - Stay on current step",
        interaction_events=example_1_events,
        journey_title="Book Taxi Journey",
        conditions=["The customer wants to book a taxi"],
        follow_up_conditions=example_1_follow_up_nodes,
        current_node=example_1_current_node,
        expected_result=example_1_expected,
    ),
    JourneyNextStepSelectionShot(
        description="Example 2 - Information provided not on journey step order",
        interaction_events=example_2_events,
        journey_title="Book Taxi Journey",
        conditions=["The customer wants to book a taxi"],
        follow_up_conditions=example_2_follow_up_nodes,
        current_node=example_2_current_node,
        expected_result=example_2_expected,
    ),
    JourneyNextStepSelectionShot(
        description="Example 3 -  Information provided earlier in the conversation",
        interaction_events=example_3_events,
        journey_title="Loan Journey",
        conditions=["The customer wants a loan"],
        follow_up_conditions=example_3_follow_up_nodes,
        current_node=example_3_current_node,
        expected_result=example_3_expected,
    ),
    JourneyNextStepSelectionShot(
        description="Example 4 - All required information is provided; select the best matching condition",
        interaction_events=example_4_events,
        journey_title="Book Taxi Journey",
        conditions=["The customer wants to book a taxi"],
        follow_up_conditions=example_4_follow_up_nodes,
        current_node=example_4_current_node,
        expected_result=example_4_expected,
    ),
    JourneyNextStepSelectionShot(
        description="Example 5 -  Information provided in current and earlier messages",
        interaction_events=example_5_events,
        journey_title="Loan Journey",
        conditions=["The customer wants a loan"],
        follow_up_conditions=example_5_follow_up_nodes,
        current_node=example_5_current_node,
        expected_result=example_5_expected,
    ),
]


shot_collection = ShotCollection[JourneyNextStepSelectionShot](_baseline_shots)
