from dataclasses import dataclass
from datetime import datetime, timezone
import json
import traceback
from typing import Any, Sequence, cast
from parlant.core.common import Criticality, DefaultBaseModel, JSONSerializable
from parlant.core.engines.alpha.guideline_matching.generic.common import internal_representation
from parlant.core.engines.alpha.guideline_matching.generic.journey.journey_backtrack_node_selection import (
    DEFAULT_ROOT_ACTION,
    ELSE_CONDITION_STR,
    PRE_ROOT_INDEX,
    ROOT_INDEX,
    SINGLE_FOLLOW_UP_CONDITION_STR,
    _JourneyEdge,
    _JourneyNode,
    JourneyNodeKind,
    get_pruned_nodes,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingBatchError,
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
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.shots import Shot, ShotCollection

FORK_NODE_ACTION_STR = "No action to perform in this node"
EXIT_JOURNEY_INSTRUCTION = "There are no further transitions."


class JourneyBacktrackCheckSchema(DefaultBaseModel):
    rationale: str | None = None
    requires_backtracking: bool
    backtrack_to_same_journey_process: bool | None = None


@dataclass
class JourneyBacktrackCheckShot(Shot):
    interaction_events: Sequence[Event]
    journey_title: str
    journey_nodes: dict[str, _JourneyNode] | None
    previous_path: Sequence[str | None]
    expected_result: JourneyBacktrackCheckSchema
    conditions: Sequence[str]


class BacktrackCheckResult(DefaultBaseModel):
    requires_backtracking: bool
    backtrack_to_same_journey_process: bool
    generation_info: GenerationInfo


class JourneyBacktrackCheck:
    def __init__(
        self,
        logger: Logger,
        guideline_store: GuidelineStore,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[JourneyBacktrackCheckSchema],
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
        self._node_wrappers: dict[str, _JourneyNode] = self._build_node_wrappers(node_guidelines)
        self._context = context
        self._examined_journey = examined_journey
        self._previous_path: Sequence[str | None] = journey_path
        self._journey_conditions = journey_conditions

    def _build_node_wrappers(self, guidelines: Sequence[Guideline]) -> dict[str, _JourneyNode]:
        def _get_guideline_node_index(guideline: Guideline) -> str:
            return str(
                cast(dict[str, JSONSerializable], guideline.metadata["journey_node"]).get(
                    "index", "-1"
                ),
            )

        guideline_id_to_guideline: dict[GuidelineId, Guideline] = {g.id: g for g in guidelines}
        guideline_id_to_node_index: dict[GuidelineId, str] = {
            g.id: _get_guideline_node_index(g) for g in guidelines
        }
        node_wrappers: dict[str, _JourneyNode] = {}

        # Build nodes
        for g in guidelines:
            node_index: str = guideline_id_to_node_index[g.id]
            if node_index not in node_wrappers:
                kind = JourneyNodeKind(
                    cast(dict[str, Any], g.metadata.get("journey_node", {})).get("kind", "NA")
                )
                customer_dependent_action = cast(
                    dict[str, bool], g.metadata.get("customer_dependent_action_data", {})
                ).get("is_customer_dependent", False)
                node_wrappers[node_index] = _JourneyNode(
                    id=_get_guideline_node_index(g),
                    action=FORK_NODE_ACTION_STR
                    if kind == JourneyNodeKind.FORK
                    else internal_representation(g).action,
                    incoming_edges=[],
                    outgoing_edges=[],
                    kind=kind,
                    customer_dependent_action=customer_dependent_action,
                    customer_action_description=cast(
                        dict[str, str | None], g.metadata.get("customer_dependent_action_data", {})
                    ).get("customer_action", None),
                    agent_dependent_action=cast(
                        dict[str, bool], g.metadata.get("customer_dependent_action_data", {})
                    ).get(
                        "is_agent_dependent",
                        not customer_dependent_action and kind == JourneyNodeKind.CHAT,
                    ),
                    agent_action_description=cast(
                        dict[str, str | None], g.metadata.get("customer_dependent_action_data", {})
                    ).get("agent_action", None),
                )

        # Build edges
        registered_edges: set[tuple[str, str]] = set()
        for g in guidelines:
            source_node_index: str = guideline_id_to_node_index[g.id]
            for followup_id in cast(
                dict[str, Sequence[GuidelineId]], g.metadata.get("journey_node", {})
            ).get("follow_ups", []):
                followup_node_index: str = guideline_id_to_node_index[GuidelineId(followup_id)]
                followup_guideline = next((g for g in guidelines if g.id == followup_id), None)
                if (
                    followup_guideline
                    and (source_node_index, followup_node_index) not in registered_edges
                ):
                    edge = _JourneyEdge(
                        target_guideline=guideline_id_to_guideline[followup_id],
                        condition=guideline_id_to_guideline[followup_id].content.condition,
                        source_node_index=source_node_index,
                        target_node_index=followup_node_index,
                    )
                    node_wrappers[source_node_index].outgoing_edges.append(edge)
                    node_wrappers[followup_node_index].incoming_edges.append(edge)
                    registered_edges.add((source_node_index, followup_node_index))
        if (
            ROOT_INDEX in node_wrappers
            and node_wrappers[ROOT_INDEX].action
            and len(node_wrappers[ROOT_INDEX].incoming_edges) == 0
        ):
            node_wrappers[ROOT_INDEX].incoming_edges.append(
                _JourneyEdge(
                    target_guideline=next(
                        g for g in guidelines if _get_guideline_node_index(g) == ROOT_INDEX
                    ),
                    condition=None,
                    source_node_index=PRE_ROOT_INDEX,
                    target_node_index=ROOT_INDEX,
                )
            )

        return node_wrappers

    def _get_journey_transition_map_text(
        self,
        nodes: dict[str, _JourneyNode],
        journey_title: str,
        journey_description: str = "",
        journey_conditions: Sequence[Guideline] = [],
        previous_path: Sequence[str | None] = [],
        print_customer_action_description: bool = False,
        to_prune: bool = False,
        max_depth: int = 5,
    ) -> str:
        def node_sort_key(node_index: str) -> Any:
            try:
                return int(node_index)
            except Exception:
                return node_index

        def get_node_transition_text(node: _JourneyNode) -> str:
            result = ""
            if len(node.outgoing_edges) == 0:
                result = f"""↳ If "this step is completed",  → {EXIT_JOURNEY_INSTRUCTION}"""
            elif len(node.outgoing_edges) == 1:
                if not (
                    to_prune
                    and nodes[node.outgoing_edges[0].target_node_index].action
                    and node.outgoing_edges[0].target_node_index in nodes
                    and node.outgoing_edges[0].target_node_index not in unpruned_nodes
                ):
                    followup_instruction = (
                        f"Go to step {node.outgoing_edges[0].target_node_index}"
                        if (
                            node.outgoing_edges[0].target_node_index in nodes
                            and nodes[node.outgoing_edges[0].target_node_index].action
                        )
                        else EXIT_JOURNEY_INSTRUCTION
                    )
                    result = f"""↳ If "{node.outgoing_edges[0].condition or SINGLE_FOLLOW_UP_CONDITION_STR}" → {followup_instruction}"""
            else:
                if not (
                    to_prune
                    and any(
                        e.target_node_index not in unpruned_nodes
                        for e in node.outgoing_edges
                        if nodes[node.outgoing_edges[0].target_node_index].action
                    )
                ):
                    result = "\n".join(
                        [
                            f"""↳ If "{e.condition or ELSE_CONDITION_STR}" → {
                                f"Go to step {e.target_node_index}"
                                if e.target_node_index in nodes
                                and nodes[e.target_node_index].action
                                else EXIT_JOURNEY_INSTRUCTION
                            }"""
                            for e in node.outgoing_edges
                        ]
                    )
                else:
                    result = EXIT_JOURNEY_INSTRUCTION
            return result

        unpruned_nodes = (
            get_pruned_nodes(
                nodes,
                previous_path,
                max_depth,
            )
            if to_prune
            else nodes
        )

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

        last_executed_node_id = next(
            (node_id for node_id in reversed(previous_path) if node_id is not None), None
        )
        nodes_str = ""
        displayed_node_action = ""
        for node_index in sorted(unpruned_nodes.keys(), key=node_sort_key):
            node: _JourneyNode = nodes[node_index]
            print_node = True
            flags_str = "Step Flags:\n"
            if node.id == ROOT_INDEX:
                if (
                    node.action and node.action != DEFAULT_ROOT_ACTION
                ):  # Root with real action, so we must print it
                    displayed_node_action = node.action
                elif (
                    len(node.outgoing_edges) > 1
                ):  # Root has no real action but has multiple followups, so should be printed
                    displayed_node_action = FORK_NODE_ACTION_STR
                else:  # Root has no action and a single follow up, so that follow up is first to be executed
                    print_node = False

            # Node kind flags
            if node.kind in {JourneyNodeKind.CHAT, JourneyNodeKind.NA} and node.action is None:
                print_node = False
            elif node.kind == JourneyNodeKind.FORK:
                displayed_node_action = FORK_NODE_ACTION_STR
            else:
                displayed_node_action = cast(str, node.action)

            # Previously executed-related flags
            if node.id == last_executed_node_id:
                flags_str += "- This is the current step that should be executed."
            elif node.id in previous_path:
                flags_str += "- PREVIOUSLY EXECUTED: This step was previously executed. May need to backtrack to this step.\n"
            elif node.id != ROOT_INDEX:
                flags_str += "- NOT PREVIOUSLY EXECUTED: This step was not previously executed. We can not backtrack to this step.\n"
            if print_node:
                nodes_str += f"""
    STEP {node_index}: {displayed_node_action}
    {flags_str}
    TRANSITIONS:
    {get_node_transition_text(node)}
    """
        return f"""
    Journey: {journey_title}
    {journey_conditions_str}{journey_description_str}

    Steps:
    {nodes_str}
    """

    async def process(self) -> BacktrackCheckResult:
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
                    hints={"temperature": generation_attempt_temperatures[generation_attempt]},
                )

                self._logger.trace(f"Completion:\n{inference.content.model_dump_json(indent=2)}")

                if not inference.content.requires_backtracking:
                    return BacktrackCheckResult(
                        requires_backtracking=inference.content.requires_backtracking,
                        backtrack_to_same_journey_process=False,
                        generation_info=inference.info,
                    )
                else:
                    return BacktrackCheckResult(
                        requires_backtracking=inference.content.requires_backtracking,
                        backtrack_to_same_journey_process=inference.content.backtrack_to_same_journey_process,
                        generation_info=inference.info,
                    )

            except Exception as exc:
                self._logger.warning(
                    f"Attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                )

                last_generation_exception = exc

        raise GuidelineMatchingBatchError() from last_generation_exception

    def _build_prompt(
        self,
        shots: Sequence[JourneyBacktrackCheckShot],
    ) -> PromptBuilder:
        builder = PromptBuilder(on_build=lambda prompt: self._logger.trace(f"Prompt:\n{prompt}"))

        builder.add_agent_identity(self._context.agent)

        builder.add_section(
            name="journey-backtrack-check-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-------------------
In our system, the behavior of a conversational AI agent is structured around predefined "journeys" - structured workflows that guide customer interactions toward specific outcomes.

## Journey Structure
Each journey consists of:
- **Steps**: Individual actions that the agent must execute (e.g., ask a question, provide information, perform a task)
- **Transitions**: Rules that determine which step comes next based on customer responses or completion status
    """,
            props={"agent_name": self._context.agent.name},
        )
        builder.add_section(
            name="journey-backtrack-check-task-description",
            template="""
TASK DESCRIPTION
-------------------
Analyze the current conversation state and determine if need to backtrack to a journey step that was already executed.

Backtracking scenarios:
    - The customer has changed a previous decision, which requires returning to an earlier step. This means retaking a step that was already visited and modifying the actions taken there.
    - The customer wants to perform the same journey process again but for a different purpose. In this case, backtrack to the beginning and re-perform the journey.
    - The customer wants to resume to the journey process that was stopped midway. In this case, continue the journey from the last executed step.

- If returning to a previous step (or restarting the journey from the beginning) is needed, set `requires_backtracking` to `true`.
    - Only steps marked with PREVIOUSLY EXECUTED flags are eligible for backtracking
- If backtracking is needed, specify the reason:
        Set 'backtrack_to_same_journey_process' to 'true' if need to revisit the journey for the same reason - changing previous decisions or resuming the journey after exiting it, for the same purpose as before.
        Set 'backtrack_to_same_journey_process' to 'false' if the journey is being revisited for a new purpose.

Example: If the journey represents a process for purchasing an item and the customer wants to change the quantity they previously requested, this is the same journey execution and the same purpose.
If, however, the customer wants to purchase a different item, the journey should restart from the beginning, which is considered a new purpose.

Exit the journey: 
If the journey needs to be exited because it was completed or the customer requests to leave the process, then backtracking is not required ('requires_backtracking' = False).
Exiting a journey does not involve backtracking to the beginning.
""",
        )
        builder.add_section(
            name="journey-backtrack-check-examples",
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
            name="journey-backtrack-check-journey-steps",
            template=self._get_journey_transition_map_text(
                nodes=self._node_wrappers,
                journey_title=self._examined_journey.title,
                previous_path=self._previous_path,
                journey_conditions=self._journey_conditions,
                journey_description=self._examined_journey.description,
                print_customer_action_description=True,
                to_prune=True,
            ),
        )
        builder.add_section(
            name="journey-backtrack-check-output-format",
            template="""{output_format}""",
            props={"output_format": self._get_output_format_section()},
        )

        return builder

    def _get_output_format_section(self) -> str:
        return """
IMPORTANT: Please provide your answer in the following JSON format.

OUTPUT FORMAT
-----------------
- Fill in the following fields as instructed. Each field is required unless otherwise specified.

```json
{
    "rationale": "<str, explanation for whether need to perform backtrack and why>",
    "requires_backtracking": <bool, does the agent need to backtrack to a previous step?>,
    "backtrack_to_same_journey_process": "<bool, include only if requires_backtracking is true, whether need to return to the same journey process>",
}
```
"""

    async def shots(self) -> Sequence[JourneyBacktrackCheckShot]:
        return await shot_collection.list()

    def _format_shots(self, shots: Sequence[JourneyBacktrackCheckShot]) -> str:
        return "\n".join(
            f"Example #{i}: {shot.journey_title}\n{self._format_shot(shot)}"
            for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, shot: JourneyBacktrackCheckShot) -> str:
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
        if shot.journey_nodes:
            formatted_shot += self._get_journey_transition_map_text(
                shot.journey_nodes,
                previous_path=shot.previous_path,
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
                print_customer_action_description=True,
            )

        formatted_shot += f"""
- **Expected Result**:
```json
{json.dumps(shot.expected_result.model_dump(mode="json", exclude_unset=True), indent=2)}
```
"""
        return formatted_shot


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


book_taxi_shot_journey_nodes = {
    "1": _JourneyNode(
        id="1",
        action="Welcome the customer to the taxi service",
        incoming_edges=[],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="You welcomed the customer",
                source_node_index="1",
                target_node_index="2",
            )
        ],
        customer_dependent_action=False,
        kind=JourneyNodeKind.CHAT,
    ),
    "2": _JourneyNode(
        id="2",
        action="Ask the customer for their desired pick up location",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="You welcomed the customer",
                source_node_index="1",
                target_node_index="2",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The desired pick up location is in NYC",
                source_node_index="2",
                target_node_index="3",
            ),
            _JourneyEdge(
                target_guideline=None,
                condition="The desired pick up location is outside of NYC",
                source_node_index="2",
                target_node_index="4",
            ),
        ],
        customer_dependent_action=True,
        customer_action_description="the customer provided their pick up location",
        kind=JourneyNodeKind.CHAT,
    ),
    "3": _JourneyNode(
        id="3",
        action="Ask where their destination is",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The desired pick up location is in NYC",
                source_node_index="2",
                target_node_index="3",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition=None,
                source_node_index="3",
                target_node_index="5",
            )
        ],
        customer_dependent_action=True,
        customer_action_description="the customer provided their desired destination",
        kind=JourneyNodeKind.CHAT,
    ),
    "4": _JourneyNode(
        id="4",
        action="Inform the customer that we do not operate outside of NYC",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The desired pick up location is outside of NYC",
                source_node_index="2",
                target_node_index="4",
            )
        ],
        outgoing_edges=[],
        customer_dependent_action=False,
        kind=JourneyNodeKind.CHAT,
    ),
    "5": _JourneyNode(
        id="5",
        action="ask for the customer's desired pick up time",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition=None,
                source_node_index="3",
                target_node_index="5",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="the customer provided their desired pick up time",
                source_node_index="5",
                target_node_index="6",
            )
        ],
        customer_dependent_action=True,
        customer_action_description="the customer provided their desired pick up time",
        kind=JourneyNodeKind.CHAT,
    ),
    "6": _JourneyNode(
        id="6",
        action="Book the taxi ride as the customer requested",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="the customer provided their desired pick up time",
                source_node_index="5",
                target_node_index="6",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="the taxi ride was successfully booked",
                source_node_index="6",
                target_node_index="7",
            )
        ],
        customer_dependent_action=False,
        kind=JourneyNodeKind.TOOL,
    ),
    "7": _JourneyNode(
        id="7",
        action="Ask the customer if they want to pay in cash or credit",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="the taxi ride was successfully booked",
                source_node_index="6",
                target_node_index="7",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="the customer wants to pay in credit",
                source_node_index="7",
                target_node_index="8",
            ),
            _JourneyEdge(
                target_guideline=None,
                condition="the customer wants to pay in cash",
                source_node_index="7",
                target_node_index="9",
            ),
        ],
        customer_dependent_action=True,
        customer_action_description="the customer specified which payment method they'd like to use'",
        kind=JourneyNodeKind.CHAT,
    ),
    "8": _JourneyNode(
        id="8",
        action="Send the customer a credit card payment link",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="the customer wants to pay in credit",
                source_node_index="7",
                target_node_index="8",
            )
        ],
        outgoing_edges=[],
        customer_dependent_action=False,
        kind=JourneyNodeKind.CHAT,
    ),
    "9": _JourneyNode(
        id="9",
        action=None,
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="the customer wants to pay in cash",
                source_node_index="7",
                target_node_index="9",
            )
        ],
        outgoing_edges=[],
        customer_dependent_action=False,
        kind=JourneyNodeKind.CHAT,
    ),
}


example_1_events = [
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


expected_output_1 = JourneyBacktrackCheckSchema(
    rationale="The customer is changing their mind about their pickup location which is a step that was previously visited in current journey process",
    requires_backtracking=True,
    backtrack_to_same_journey_process=True,
)


example_2_events = [
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
        "8 AM.",
    ),
    _make_event(
        "89",
        EventSource.CUSTOMER,
        "Actually Let's make it 7.",
    ),
]


expected_output_2 = JourneyBacktrackCheckSchema(
    rationale="The customer is changing their mind about their answer to the current journey step, so no backtrack needed.",
    requires_backtracking=False,
)


example_3_events = [
    _make_event(
        "2",
        EventSource.CUSTOMER,
        "Hi, I need a taxi from Manhattan to JFK Airport",
    ),
    _make_event(
        "3",
        EventSource.AI_AGENT,
        "Great! You'd like to go from Manhattan to JFK Airport. What time would you like to be picked up?",
    ),
    _make_event(
        "4",
        EventSource.CUSTOMER,
        "Tomorrow at 2 PM please",
    ),
    _make_event(
        "5",
        EventSource.AI_AGENT,
        "Perfect! I've booked your taxi ride from Manhattan to JFK Airport for tomorrow at 2 PM. Would you like to pay with cash or credit?",
    ),
    _make_event(
        "6",
        EventSource.CUSTOMER,
        "Credit card please",
    ),
    _make_event(
        "7",
        EventSource.AI_AGENT,
        "Excellent! I'm sending you a payment link now for your credit card. Is there anything else I can help you with?",
    ),
    _make_event(
        "8",
        EventSource.CUSTOMER,
        "Great!",
    ),
    _make_event(
        "9",
        EventSource.AI_AGENT,
        "Is there anything else I can do for you?",
    ),
    _make_event(
        "8",
        EventSource.CUSTOMER,
        "I need to book another taxi for my son",
    ),
]

expected_output_3 = JourneyBacktrackCheckSchema(
    rationale="The customer wants to order another taxi, so need to restart the journey from the beginning",
    requires_backtracking=True,
    backtrack_to_same_journey_process=False,
)


example_4_events = [
    _make_event(
        "2",
        EventSource.CUSTOMER,
        "Hi, I need a taxi from Manhattan to JFK Airport",
    ),
    _make_event(
        "3",
        EventSource.AI_AGENT,
        "Great! You'd like to go from Manhattan to JFK Airport. What time would you like to be picked up?",
    ),
    _make_event(
        "4",
        EventSource.CUSTOMER,
        "Actually, I don't need this taxi anymore, sorry. But can you help me check if I had any rides with you last month? I want to know how much they cost me",
    ),
]

expected_output_4 = JourneyBacktrackCheckSchema(
    rationale="The customer changed their mind and no longer wants to book a taxi, so the journey needs to be exited. No backtracking is needed.",
    requires_backtracking=False,
)

_baseline_shots: Sequence[JourneyBacktrackCheckShot] = [
    JourneyBacktrackCheckShot(
        description="Example 1 - Backtrack to current journey",
        interaction_events=example_1_events,
        journey_title="Book Taxi Journey",
        journey_nodes=book_taxi_shot_journey_nodes,
        previous_path=["1", "2", "4", "2", "3", "5"],
        expected_result=expected_output_1,
        conditions=[],
    ),
    JourneyBacktrackCheckShot(
        description="Example 2 - No backtracking",
        interaction_events=example_2_events,
        journey_title="Book Taxi Journey",
        journey_nodes=book_taxi_shot_journey_nodes,
        previous_path=["1", "2", "4", "2", "3", "5"],
        expected_result=expected_output_2,
        conditions=[],
    ),
    JourneyBacktrackCheckShot(
        description="Example 3 - Backtrack to a new journey process",
        interaction_events=example_3_events,
        journey_title="Book Taxi Journey",
        journey_nodes=book_taxi_shot_journey_nodes,
        previous_path=["1", "2", "3", "5", "7", "8", "None"],
        expected_result=expected_output_3,
        conditions=[],
    ),
    JourneyBacktrackCheckShot(
        description="Example 4 - Exiting the journey",
        interaction_events=example_4_events,
        journey_title="Book Taxi Journey",
        journey_nodes=book_taxi_shot_journey_nodes,
        previous_path=["1", "2"],
        expected_result=expected_output_4,
        conditions=[],
    ),
]


shot_collection = ShotCollection[JourneyBacktrackCheckShot](_baseline_shots)
