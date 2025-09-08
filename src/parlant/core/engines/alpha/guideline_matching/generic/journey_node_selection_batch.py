from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import traceback
from typing import Any, Optional, cast
from typing_extensions import override
from parlant.core import async_utils
from parlant.core.common import DefaultBaseModel, JSONSerializable

from parlant.core.engines.alpha.guideline_matching.generic.common import internal_representation
from parlant.core.engines.alpha.guideline_matching.guideline_match import (
    GuidelineMatch,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingBatch,
    GuidelineMatchingBatchError,
    GuidelineMatchingBatchResult,
    GuidelineMatchingContext,
)
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId, GuidelineStore
from parlant.core.journeys import Journey
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.shots import Shot, ShotCollection

PRE_ROOT_INDEX = "0"
ROOT_INDEX = "1"

DEFAULT_ROOT_ACTION = (
    "<<JOURNEY ROOT: start the journey at the appropriate step based on the context>>"
)
BEGIN_JOURNEY_AT_ACTIONLESS_ROOT_FLAG_TEXT = "- BEGIN HERE: Begin the journey advancement at this step. Advance to the next node based on the relevant transition."
BEGIN_JOURNEY_AT_ROOT_WITH_ACTION_FLAG_TEXT = "- BEGIN HERE: Begin the journey advancement at this step. Advance onward if this step was already completed."
EXIT_JOURNEY_INSTRUCTION = "RETURN 'NONE'"
ELSE_CONDITION_STR = "This step was completed, and no other transition applies"
SINGLE_FOLLOW_UP_CONDITION_STR = "This step was completed"
FORK_NODE_ACTION_STR = (
    "No action necessary - always advance to the next step based on the relevant transition"
)
LAST_PRESENTED_NODE_INSTRUCTION = "Do not advance past this step. If you got here - mark this step as incomplete and return it as next_step"


class JourneyNodeKind(Enum):
    FORK = "fork"
    CHAT = "chat"
    TOOL = "tool"
    NA = "NA"


class StepCompletionStatus(Enum):
    COMPLETED = "completed"
    NEEDS_CUSTOMER_INPUT = "needs_customer_input"
    NEEDS_AGENT_ACTION = "needs_agent_action"
    NEEDS_TOOL_CALL = "needs_tool_call"


@dataclass
class _JourneyEdge:
    target_guideline: Guideline | None
    condition: str | None
    source_node_index: str
    target_node_index: str


@dataclass
class _JourneyNode:  # Refactor after node type is implemented
    id: str
    action: str | None
    incoming_edges: list[_JourneyEdge]
    outgoing_edges: list[_JourneyEdge]
    kind: JourneyNodeKind
    customer_dependent_action: bool
    customer_action_description: Optional[str] = None


class JourneyNodeAdvancement(DefaultBaseModel):
    id: str
    completed: StepCompletionStatus
    follow_ups: Optional[list[str]] = None


class JourneyNodeSelectionSchema(DefaultBaseModel):
    rationale: str | None = None
    journey_applies: bool | None = None
    requires_backtracking: bool | None = None
    backtracking_target_step: str | None = None
    step_advancement: Sequence[JourneyNodeAdvancement] | None = None
    next_step: str | None = None


@dataclass
class JourneyNodeSelectionShot(Shot):
    interaction_events: Sequence[Event]
    journey_title: str
    journey_nodes: dict[str, _JourneyNode] | None
    previous_path: Sequence[str | None]
    expected_result: JourneyNodeSelectionSchema
    conditions: Sequence[str]


def get_pruned_nodes(
    nodes: dict[str, _JourneyNode],
    previous_path: Sequence[str | None],
    max_depth: int,
) -> dict[str, _JourneyNode]:
    # TODO can be implemented in cleaner fashion if we maintain a dictionary of the distance of each node from the previous path / current node
    # If we encounter any trouble with pruning - we should implement it as such
    if previous_path and set(previous_path) != set([None]):
        nodes_to_traverse = set(previous_path)
    else:
        nodes_to_traverse = set("1")

    visited: set[str | None] = set()
    result: set[str | None] = set()

    queue: deque[tuple[str | None, int]] = deque()

    for node in nodes_to_traverse:
        visited = set()
        queue.append((node, 0))
        while queue:
            current, depth = queue.popleft()
            if not current:
                continue

            if depth > max_depth or current in visited:
                continue

            visited.add(current)
            result.add(current)

            # If node run tools, no need to show the steps further.
            if nodes[current].kind == JourneyNodeKind.TOOL and (
                not previous_path or current not in previous_path
            ):
                continue

            for edge in nodes[current].outgoing_edges:
                neighbor = edge.target_node_index
                queue.append((neighbor, depth + 1))

    pruned_nodes = {idx: nodes[idx] for idx in result if idx}
    if not pruned_nodes:  # Recover in case some unexpected error caused all nodes to be pruned
        return get_pruned_nodes(nodes, [], max_depth)
    return pruned_nodes


def build_node_wrappers(guidelines: Sequence[Guideline]) -> dict[str, _JourneyNode]:
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
            node_wrappers[node_index] = _JourneyNode(
                id=_get_guideline_node_index(g),
                action=FORK_NODE_ACTION_STR
                if kind == JourneyNodeKind.FORK
                else internal_representation(g).action,
                incoming_edges=[],
                outgoing_edges=[],
                kind=kind,
                customer_dependent_action=cast(
                    dict[str, bool], g.metadata.get("customer_dependent_action_data", {})
                ).get("is_customer_dependent", False),
                customer_action_description=cast(
                    dict[str, str | None], g.metadata.get("customer_dependent_action_data", {})
                ).get("customer_action", None),
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


def get_journey_transition_map_text(
    nodes: dict[str, _JourneyNode],
    journey_title: str,
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
            if (
                to_prune
                and nodes[node.outgoing_edges[0].target_node_index].action
                and node.outgoing_edges[0].target_node_index in nodes
                and node.outgoing_edges[0].target_node_index not in unpruned_nodes
            ):
                result = LAST_PRESENTED_NODE_INSTRUCTION
            else:
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
            if to_prune and any(
                e.target_node_index not in unpruned_nodes
                for e in node.outgoing_edges
                if nodes[node.outgoing_edges[0].target_node_index].action
            ):
                result = LAST_PRESENTED_NODE_INSTRUCTION
            else:
                result = "\n".join(
                    [
                        f"""↳ If "{e.condition or ELSE_CONDITION_STR}" → {
                            f"Go to step {e.target_node_index}"
                            if e.target_node_index in nodes and nodes[e.target_node_index].action
                            else EXIT_JOURNEY_INSTRUCTION
                        }"""
                        for e in node.outgoing_edges
                    ]
                )
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

    if journey_conditions:
        journey_conditions_str = " OR ".join(f'"{g.content.condition}"' for g in journey_conditions)
        journey_conditions_str = f"\nJourney activation condition: {journey_conditions_str}\n"
    else:
        journey_conditions_str = ""

    last_executed_node_id = next(
        (node_id for node_id in reversed(previous_path) if node_id is not None), None
    )
    first_node_to_execute: str | None = None
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
                if not previous_path or set(previous_path) == set([None]):
                    flags_str += BEGIN_JOURNEY_AT_ROOT_WITH_ACTION_FLAG_TEXT + "\n"
                displayed_node_action = node.action
            elif (
                len(node.outgoing_edges) > 1
            ):  # Root has no real action but has multiple followups, so should be printed
                if not previous_path or set(previous_path) == set([None]):
                    flags_str += BEGIN_JOURNEY_AT_ACTIONLESS_ROOT_FLAG_TEXT + "\n"
                displayed_node_action = FORK_NODE_ACTION_STR
            else:  # Root has no action and a single follow up, so that follow up is first to be executed
                print_node = False
                if not previous_path or set(previous_path) == set([None]):
                    first_node_to_execute = node.outgoing_edges[0].target_node_index
        # Customer / Agent dependent flags
        if node.customer_dependent_action:
            if print_customer_action_description and node.customer_action_description:
                flags_str += f'- CUSTOMER DEPENDENT: This action requires an action from the customer to be considered complete. It is completed if the following action was completed: "{node.customer_action_description}" \n'
            else:
                flags_str += "- CUSTOMER DEPENDENT: This action requires an action from the customer to be considered complete. Mark it as complete if the customer answered the question in the action, if there is one.\n"
        elif node.kind == JourneyNodeKind.CHAT:
            flags_str += "- REQUIRES AGENT ACTION: This step may require the agent to say something for it to be completed. Only advance through it if the agent performed the described action.\n"

        # Node kind flags
        if node.kind in {JourneyNodeKind.CHAT, JourneyNodeKind.NA} and node.action is None:
            print_node = False
        elif node.kind == JourneyNodeKind.FORK:
            displayed_node_action = FORK_NODE_ACTION_STR
            flags_str += "- NEVER OUTPUT THIS STEP AS NEXT_STEP: This step is transitional and should never be returned as the next_step. Always advance onwards from it.\n"
        else:
            displayed_node_action = cast(str, node.action)
        if node.kind == JourneyNodeKind.TOOL and node.id != last_executed_node_id:
            flags_str += (
                "- REQUIRES TOOL CALLS: Do not advance past this step! If you got here, stop.\n"
            )

        # Previously executed-related flags
        if node.id == first_node_to_execute:
            flags_str += BEGIN_JOURNEY_AT_ROOT_WITH_ACTION_FLAG_TEXT
        elif node.id == last_executed_node_id:
            flags_str += (
                "- This is the last step that was executed. Begin advancing on from this step\n"
            )
        elif node.id in previous_path:
            flags_str += "- PREVIOUSLY EXECUTED: This step was previously executed. You may backtrack to this step.\n"
        elif node.id != ROOT_INDEX:
            flags_str += "- NOT PREVIOUSLY EXECUTED: This step was not previously executed. You may not backtrack to this step.\n"
        if print_node:
            nodes_str += f"""
STEP {node_index}: {displayed_node_action}
{flags_str}
TRANSITIONS:
{get_node_transition_text(node)}
"""
    return f"""
Journey: {journey_title}
{journey_conditions_str}
Steps:
{nodes_str}
"""


class GenericJourneyNodeSelectionBatch(GuidelineMatchingBatch):
    def __init__(
        self,
        logger: Logger,
        guideline_store: GuidelineStore,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[JourneyNodeSelectionSchema],
        examined_journey: Journey,
        context: GuidelineMatchingContext,
        node_guidelines: Sequence[Guideline] = [],
        journey_path: Sequence[str | None] = [],
    ) -> None:
        self._logger = logger

        self._guideline_store = guideline_store

        self._optimization_policy = optimization_policy
        self._schematic_generator = schematic_generator
        self._node_wrappers: dict[str, _JourneyNode] = build_node_wrappers(node_guidelines)
        self._context = context
        self._examined_journey = examined_journey
        self._previous_path: Sequence[str | None] = journey_path

    def auto_return_match(self) -> GuidelineMatchingBatchResult | None:
        if self._previous_path and self._previous_path[-1] in self._node_wrappers:
            last_visited_node = self._node_wrappers[self._previous_path[-1]]
            if (
                last_visited_node.kind == JourneyNodeKind.TOOL
                and len(last_visited_node.outgoing_edges) == 1
            ):
                generation_info = GenerationInfo(
                    schema_name="No inference performed",
                    model="No inference performed",
                    duration=0.0,
                    usage=UsageInfo(
                        input_tokens=0,
                        output_tokens=0,
                        extra={},
                    ),
                )
                if last_visited_node.outgoing_edges[0].target_guideline:
                    return GuidelineMatchingBatchResult(
                        matches=[
                            GuidelineMatch(
                                guideline=last_visited_node.outgoing_edges[0].target_guideline,
                                score=10,
                                rationale="This guideline was selected as part of a 'journey' - a sequence of actions that are performed in order. It was automatically selected as the only viable follow up for the last step that was executed",
                                metadata={
                                    "journey_path": [
                                        last_visited_node.id,
                                        last_visited_node.outgoing_edges[0].target_node_index,
                                    ],
                                    "step_selection_journey_id": self._examined_journey.id,
                                },
                            )
                        ],
                        generation_info=generation_info,
                    )
                else:
                    return GuidelineMatchingBatchResult(matches=[], generation_info=generation_info)
        return None

    @override
    async def process(self) -> GuidelineMatchingBatchResult:
        automatic_match = self.auto_return_match()
        if automatic_match:
            return automatic_match  # TODO fix this

        journey_conditions = list(
            await async_utils.safe_gather(
                *[
                    self._guideline_store.read_guideline(c)
                    for c in self._examined_journey.conditions
                ]
            )
        )

        with self._logger.operation(self._examined_journey.title):
            prompt = self._build_prompt(journey_conditions, shots=await self.shots())

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

                    self._logger.trace(
                        f"Completion:\n{inference.content.model_dump_json(indent=2)}"
                    )

                    journey_path = self._get_verified_node_advancement(inference.content)

                    # Get correct guideline to return based on the transition into next_step  TODO consider surrounding with try catch specifically
                    matched_guideline: Guideline | None = None
                    if inference.content.next_step in self._node_wrappers:
                        if len(journey_path) > 1 and [
                            e
                            for e in self._node_wrappers[inference.content.next_step].incoming_edges
                            if e.source_node_index == journey_path[-2]
                        ]:
                            matched_guideline = next(
                                e
                                for e in self._node_wrappers[
                                    inference.content.next_step
                                ].incoming_edges
                                if e.source_node_index == journey_path[-2]
                            ).target_guideline
                        else:
                            matched_guideline = (
                                self._node_wrappers[inference.content.next_step]
                                .incoming_edges[0]
                                .target_guideline
                            )
                    # TODO how do we return the path if we're exiting the journey?
                    return GuidelineMatchingBatchResult(
                        matches=[
                            GuidelineMatch(
                                guideline=matched_guideline,
                                score=10,
                                rationale=f"This guideline was selected as part of a 'journey' - a sequence of actions that are performed in order. Use this rationale to better understand how the conversation got to its current point. The rationale for choosing this specific step in the journey was: {inference.content.rationale}",
                                metadata={
                                    "journey_path": journey_path,
                                    "step_selection_journey_id": self._examined_journey.id,
                                },
                            )
                        ]
                        if matched_guideline  # If either 'None' or an illegal step was returned, don't activate guidelines
                        else [],
                        generation_info=inference.info,
                    )
                except Exception as exc:
                    self._logger.warning(
                        f"Attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                    )

                    last_generation_exception = exc

            raise GuidelineMatchingBatchError() from last_generation_exception

    async def shots(self) -> Sequence[JourneyNodeSelectionShot]:
        return await shot_collection.list()

    def _format_shots(self, shots: Sequence[JourneyNodeSelectionShot]) -> str:
        return "\n".join(
            f"Example #{i}: {shot.journey_title}\n{self._format_shot(shot)}"
            for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, shot: JourneyNodeSelectionShot) -> str:
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
            formatted_shot += get_journey_transition_map_text(
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

    def _get_verified_node_advancement(
        self, response: JourneyNodeSelectionSchema
    ) -> list[str | None]:
        def add_and_remove_list_values(
            list_to_alter: list[Any],
            indexes_to_add: Sequence[tuple[int, Any]],
            indexes_to_delete: Sequence[int],
        ) -> list[Any]:
            result = list_to_alter.copy()

            for i in reversed(indexes_to_delete):
                del result[i]

            for original_i, value in indexes_to_add:
                deletions_before = sum(1 for del_i in indexes_to_delete if del_i < original_i)
                additions_before = sum(1 for add_i, _ in indexes_to_add if add_i < original_i)
                adjusted_i = original_i - deletions_before + additions_before
                result.insert(adjusted_i, value)

            return result

        journey_path: list[str | None] = []
        for i, advancement in enumerate(response.step_advancement or []):
            journey_path.append(advancement.id)
            if (
                i > 0
                and advancement.id in self._node_wrappers
                and self._node_wrappers[advancement.id].kind == JourneyNodeKind.TOOL
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
        indexes_to_add: list[tuple[int, str]] = []
        for i in range(1, len(journey_path)):  # Verify all transitions are legal
            if journey_path[i - 1] not in self._node_wrappers:
                self._logger.warning(
                    f"WARNING: Illegal journey path returned by journey step selection. Illegal step returned: {journey_path[i - 1]}. Full path: : {journey_path}"
                )
                indexes_to_delete.append(i)
            elif journey_path[i] not in [
                e.target_node_index
                for e in self._node_wrappers[cast(str, journey_path[i - 1])].outgoing_edges
            ]:
                self._logger.warning(
                    f"WARNING: Illegal transition in journey path returned by journey step selection - from {journey_path[i - 1]} to {journey_path[i]}. Full path: : {journey_path}"
                )
                # Sometimes, the LLM returns a path that would've been legal if it were not for an out-of-place step. This deletes such steps.
                if i + 1 < len(journey_path) and journey_path[i + 1] in [
                    e.target_node_index
                    for e in self._node_wrappers[str(journey_path[i - 1])].outgoing_edges
                ]:
                    indexes_to_delete.append(i)
                else:
                    # In other cases, it skips a node that would make the path valid. We want to identify and add the missing node
                    previous_node_follow_ups = set(
                        e.target_node_index
                        for e in self._node_wrappers[cast(str, journey_path[i - 1])].outgoing_edges
                        if e.source_node_index == journey_path[i - 1]
                    )
                    if journey_path[i] in self._node_wrappers:
                        current_node_origins = set(
                            e.source_node_index
                            for e in self._node_wrappers[cast(str, journey_path[i])].incoming_edges
                        )

                        possible_connector_nodes: list[str] = list(
                            previous_node_follow_ups.intersection(current_node_origins)
                        )
                    else:
                        possible_connector_nodes = list(previous_node_follow_ups)
                    if len(possible_connector_nodes) == 1:
                        indexes_to_add.append((i, possible_connector_nodes[0]))
        journey_path = add_and_remove_list_values(journey_path, indexes_to_add, indexes_to_delete)

        if (
            journey_path and journey_path[-1] not in self._node_wrappers
        ):  # 'Exit journey' was selected, or illegal value returned (both should cause no guidelines to be active)
            journey_path[-1] = None

        return journey_path

    def _build_prompt(
        self,
        journey_conditions: Sequence[Guideline],
        shots: Sequence[JourneyNodeSelectionShot],
    ) -> PromptBuilder:
        builder = PromptBuilder(on_build=lambda prompt: self._logger.trace(f"Prompt:\n{prompt}"))

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
Evaluate whether the last executed step is complete:
- For CUSTOMER_DEPENDENT steps: Customer has provided the required information (either after being asked or proactively in earlier messages. If so, set completed to 'completed'.
 If not, set completed to 'needs_customer_input' and do not advance past this step.
- For REQUIRES AGENT ACTION steps: The agent has performed the required communication or action. If so, set completed to 'completed'. If not, set completed to 'needs_agent_action'
and do not advance past this step.
- For REQUIRES_TOOL_CALLS steps: The step requires a tool call to execute for it to be completed. If you begin your advancement at this step, mark it as complete if the tool executed, and move onwards. Otherwise, always set completed to false and return it as next_step.
- If the last step is incomplete, set next_step to the current step ID (repeat the step) and document this in the step_advancement array.

## 4: Journey Advancement
Starting from the last executed step, advance through subsequent steps, documenting each step's completion status in the step_advancement array.
Only advance to the next step if the current step is marked as completed.
At each completed step, carefully evaluate the follow-up steps from the 'transitions' section, and advance only to the step whose condition is satisfied.
Base advancement decisions strictly on these transitions and their conditions — never jump to a step whose condition was not met, even if you believe it should logically be executed next.
Pleasing the customer is not a valid reason to violate the transitions - always traverse to the next step according to its conditions.

Document your advancement path in step_advancement as a list of step advancement objects, starting with the last_step and ending with the next step to execute. Each step must be a legal
follow-up of the previous step, and you can only advance if the previous step was completed.

Continue advancing until you encounter:
- A step requiring a tool call (REQUIRES_TOOL_CALLS flag)
- A step where you lack necessary information to proceed
- A step requiring you to communicate something new to the customer, beyond asking them for information (REQUIRES AGENT ACTION flag)

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
        builder.add_staged_tool_events(self._context.staged_events)

        builder.add_section(
            name="journey_description_background",
            template="The following is the journey you are now traversing. Read it carefully and ensure to understand which steps follow which:",
        )
        builder.add_section(
            name="journey-step-selection-journey-steps",
            template=get_journey_transition_map_text(
                nodes=self._node_wrappers,
                journey_title=self._examined_journey.title,
                previous_path=self._previous_path,
                journey_conditions=journey_conditions,
                print_customer_action_description=True,
                to_prune=True,
            ),
        )
        builder.add_section(
            name="journey-step-selection-output-format",
            template="""{output_format}""",
            props={"output_format": self._get_output_format_section()},
        )

        builder.add_section(
            name="journey-general_reminder-section",
            template="""Reminder - carefully consider all restraints and instructions. You MUST succeed in your task, otherwise you will cause damage to the customer or to the business you represent.""",
        )
        return builder

    def _get_output_format_section(self) -> str:
        last_node = self._previous_path[-1] if self._previous_path else "None"
        return f"""
IMPORTANT: Please provide your answer in the following JSON format.

OUTPUT FORMAT
-----------------
- Fill in the following fields as instructed. Each field is required unless otherwise specified.

```json
{{
  "rationale": "<str, explanation for what is the next step and why it was selected>",
  "journey_applies": <bool, whether the journey should continued. Reminder: If you are already executing journey steps (i.e., there is a "last_step"), the journey almost always continues. The activation condition is ONLY for starting new journeys, NOT for validating ongoing ones.>,
  "requires_backtracking": <bool, does the agent need to backtrack to a previous step?>,
  "backtracking_target_step": "<str, id of the step where the customer's decision changed. Omit this field if requires_backtracking is false>",
  "step_advancement": [
    {{
        "id": "<str, id of the step. First one should be either {last_node} or backtracking_target_step if it exists>",
        "completed": <str, either 'completed' or 'needs_customer_input' or 'needs_agent_action' or 'needs_tool_call'>,
        "follow_ups": "<list[str], ids of legal follow ups for this step. Omit if completed is not 'completed'>"
    }},
    ... <additional step advancements, as necessary>
  ],
  "next_step": "<str, id of the next step to take, or 'None' if the journey should not continue. Must be equal to the last step in step_advancement>"
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


example_1_journey_nodes = {
    "1": _JourneyNode(
        id="1",
        kind=JourneyNodeKind.CHAT,
        action="Ask the customer if they prefer exploring cities or enjoying scenic landscapes.",
        incoming_edges=[],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,  # Would need actual guidelines
                condition="The customer prefers exploring cities",
                source_node_index="1",
                target_node_index="2",
            ),
            _JourneyEdge(
                target_guideline=None,  # Would need actual guidelines
                condition="The customer prefers scenic landscapes",
                source_node_index="1",
                target_node_index="3",
            ),
            _JourneyEdge(
                target_guideline=None,  # Would need actual guidelines
                condition="The customer raises an issue unrelated to exploring cities or scenic landscapes",
                source_node_index="1",
                target_node_index="4",
            ),
        ],
        customer_dependent_action=True,
        customer_action_description="the customer responded regarding their preference between exploring cities and scenic landscapes",
    ),
    "2": _JourneyNode(
        id="2",
        kind=JourneyNodeKind.CHAT,
        action="Recommend the capital city of their desired nation",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,  # Would need actual guidelines
                condition="The customer prefers exploring cities",
                source_node_index="1",
                target_node_index="2",
            )
        ],
        outgoing_edges=[],
        customer_dependent_action=False,
    ),
    "3": _JourneyNode(
        id="3",
        kind=JourneyNodeKind.CHAT,
        action="Recommend the top hiking route of their desired nation",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,  # Would need actual guidelines
                condition="The customer prefers scenic landscapes",
                source_node_index="1",
                target_node_index="3",
            )
        ],
        outgoing_edges=[],
        customer_dependent_action=False,
    ),
    "4": _JourneyNode(
        id="4",
        action="Refer them to our travel information page",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,  # Would need actual guidelines
                condition="The customer raises an issue unrelated to exploring cities or scenic landscapes",
                source_node_index="1",
                target_node_index="4",
            )
        ],
        outgoing_edges=[],
        customer_dependent_action=False,
        kind=JourneyNodeKind.CHAT,
    ),
}


example_1_expected = JourneyNodeSelectionSchema(
    journey_applies=True,
    requires_backtracking=False,
    rationale="The last step was completed. Customer asks about visas, which is unrelated to exploring cities, so step 4 should be activated",
    step_advancement=[
        JourneyNodeAdvancement(
            id="1", completed=StepCompletionStatus.COMPLETED, follow_ups=["2", "3", "4"]
        ),
        JourneyNodeAdvancement(id="4", completed=StepCompletionStatus.NEEDS_AGENT_ACTION),
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
                condition="the desired pick up location is in NYC",
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

random_actions_journey_nodes = {
    "1": _JourneyNode(
        id="1",
        action="State a random capital city. Do not say anything else.",
        incoming_edges=[],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The previous step was completed",
                source_node_index="1",
                target_node_index="2",
            )
        ],
        customer_dependent_action=False,
        kind=JourneyNodeKind.CHAT,
    ),
    "2": _JourneyNode(
        id="2",
        action="Ask the customer for money.",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="The previous step was completed",
                source_node_index="1",
                target_node_index="2",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="This step was completed",
                source_node_index="2",
                target_node_index="3",
            )
        ],
        customer_dependent_action=True,
        kind=JourneyNodeKind.CHAT,
        customer_action_description="the customer directly responded to the agent's request for money",
    ),
    "3": _JourneyNode(
        id="3",
        action="Tell the customer goodbye and disconnect from the conversation",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="This step was completed",
                source_node_index="2",
                target_node_index="3",
            )
        ],
        outgoing_edges=[],
        customer_dependent_action=False,
        kind=JourneyNodeKind.CHAT,
    ),
}

example_2_expected = JourneyNodeSelectionSchema(
    journey_applies=True,
    rationale="The customer provided a pick up location in NYC, a destination and a pick up time, allowing me to fast-forward through steps 2, 3, 5. I must stop at the next step, 6, because it requires tool calling.",
    requires_backtracking=False,
    step_advancement=[
        JourneyNodeAdvancement(
            id="2", completed=StepCompletionStatus.COMPLETED, follow_ups=["3", "4"]
        ),
        JourneyNodeAdvancement(id="3", completed=StepCompletionStatus.COMPLETED, follow_ups=["5"]),
        JourneyNodeAdvancement(id="5", completed=StepCompletionStatus.COMPLETED, follow_ups=["6"]),
        JourneyNodeAdvancement(id="6", completed=StepCompletionStatus.NEEDS_TOOL_CALL),
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

example_3_expected = JourneyNodeSelectionSchema(
    journey_applies=True,
    rationale="The customer provided a pick up location in NYC and a destination, allowing us to fast-forward through steps 1, 2 and 3. Step 5 requires asking for a pick up time, which the customer has yet to provide. We must therefore activate step 5.",
    requires_backtracking=False,
    step_advancement=[
        JourneyNodeAdvancement(id="1", completed=StepCompletionStatus.COMPLETED, follow_ups=["3"]),
        JourneyNodeAdvancement(
            id="2", completed=StepCompletionStatus.COMPLETED, follow_ups=["3", "4"]
        ),
        JourneyNodeAdvancement(id="3", completed=StepCompletionStatus.COMPLETED, follow_ups=["5"]),
        JourneyNodeAdvancement(id="5", completed=StepCompletionStatus.NEEDS_CUSTOMER_INPUT),
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

example_4_expected = JourneyNodeSelectionSchema(
    journey_applies=True,
    requires_backtracking=True,
    rationale="The customer is changing their pickup location decision that was made in step 2. The relevant follow up is step 3, since the new requested location is within NYC.",
    backtracking_target_step="2",
    step_advancement=[
        JourneyNodeAdvancement(
            id="2", completed=StepCompletionStatus.COMPLETED, follow_ups=["3", "4"]
        ),
        JourneyNodeAdvancement(
            id="3",
            completed=StepCompletionStatus.COMPLETED,
            follow_ups=["5"],
        ),
        JourneyNodeAdvancement(
            id="5",
            completed=StepCompletionStatus.COMPLETED,
            follow_ups=["6"],
        ),
        JourneyNodeAdvancement(id="6", completed=StepCompletionStatus.NEEDS_TOOL_CALL),
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
        EventSource.CUSTOMER,
        "Oh really? I always thought it was Sydney",
    ),
]

example_5_expected = JourneyNodeSelectionSchema(
    journey_applies=True,
    rationale="Customer was told about capitals. Now we need to advance to the following step and ask for money",
    requires_backtracking=False,
    step_advancement=[
        JourneyNodeAdvancement(id="1", completed=StepCompletionStatus.COMPLETED, follow_ups=["2"]),
        JourneyNodeAdvancement(id="2", completed=StepCompletionStatus.NEEDS_CUSTOMER_INPUT),
    ],
    next_step="2",
)


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

loan_journey_nodes = {
    "1": _JourneyNode(
        id="1",
        action="Ask for the customer's full name.",
        incoming_edges=[],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="Customer provided their name",
                source_node_index="1",
                target_node_index="2",
            )
        ],
        customer_dependent_action=True,
        kind=JourneyNodeKind.CHAT,
        customer_action_description="the customer provided their full name",
    ),
    "2": _JourneyNode(
        id="2",
        action="Ask for the type of loan: Personal or Business.",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="Customer provided their name",
                source_node_index="1",
                target_node_index="2",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="Customer chose Personal loan",
                source_node_index="2",
                target_node_index="3",
            ),
            _JourneyEdge(
                target_guideline=None,
                condition="Customer chose Business loan",
                source_node_index="2",
                target_node_index="4",
            ),
        ],
        customer_dependent_action=True,
        kind=JourneyNodeKind.CHAT,
        customer_action_description="the customer specified which type of loan they'd like to take",
    ),
    "3": _JourneyNode(
        id="3",
        action="Ask for the desired loan amount.",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="Customer chose Personal loan",
                source_node_index="2",
                target_node_index="3",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="Personal loan amount provided",
                source_node_index="3",
                target_node_index="5",
            )
        ],
        customer_dependent_action=True,
        kind=JourneyNodeKind.CHAT,
        customer_action_description="the customer provided the desired loan amount",
    ),
    "4": _JourneyNode(
        id="4",
        action="Ask for the desired loan amount.",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="Customer chose Business loan",
                source_node_index="2",
                target_node_index="4",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="Business loan amount provided",
                source_node_index="4",
                target_node_index="6",
            )
        ],
        customer_dependent_action=True,
        customer_action_description="the customer provided the desired loan amount",
        kind=JourneyNodeKind.CHAT,
    ),
    "5": _JourneyNode(
        id="5",
        action="Ask for employment status.",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="Personal loan amount provided",
                source_node_index="3",
                target_node_index="5",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="Employment status provided",
                source_node_index="5",
                target_node_index="7",
            )
        ],
        customer_dependent_action=True,
        customer_action_description="the customer specified their employment status",
        kind=JourneyNodeKind.CHAT,
    ),
    "6": _JourneyNode(
        id="6",
        action="Ask for collateral.",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="Business loan amount provided",
                source_node_index="4",
                target_node_index="6",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="Digital asset was chosen as collateral",
                source_node_index="6",
                target_node_index="8",
            ),
            _JourneyEdge(
                target_guideline=None,
                condition="physical asset was chosen as collateral",
                source_node_index="6",
                target_node_index="9",
            ),
        ],
        customer_dependent_action=True,
        customer_action_description="the customer provided their collateral",
        kind=JourneyNodeKind.CHAT,
    ),
    "7": _JourneyNode(
        id="7",
        action="Review and confirm application.",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="Employment status provided",
                source_node_index="5",
                target_node_index="7",
            )
        ],
        outgoing_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="This step was completed",
                source_node_index="7",
                target_node_index="9",
            )
        ],
        customer_dependent_action=True,
        customer_action_description="the customer confirmed the application and its details",
        kind=JourneyNodeKind.CHAT,
    ),
    "8": _JourneyNode(
        id="8",
        action="Review and confirm application.",
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="Digital asset was chosen as collateral",
                source_node_index="6",
                target_node_index="8",
            )
        ],
        outgoing_edges=[],
        customer_dependent_action=True,
        customer_action_description="the customer confirmed the application and its details",
        kind=JourneyNodeKind.CHAT,
    ),
    "9": _JourneyNode(
        id="9",
        action=None,
        incoming_edges=[
            _JourneyEdge(
                target_guideline=None,
                condition="physical asset was chosen as collateral",
                source_node_index="6",
                target_node_index="9",
            ),
            _JourneyEdge(
                target_guideline=None,
                condition="This step was completed",
                source_node_index="7",
                target_node_index="9",
            ),
        ],
        outgoing_edges=[],
        customer_dependent_action=False,
        kind=JourneyNodeKind.CHAT,
    ),
}

example_6_expected = JourneyNodeSelectionSchema(
    journey_applies=True,
    requires_backtracking=True,
    rationale="The customer changed their loan type decision after providing all information. The journey backtracks to the loan type step (2), then fast-forwards through the business loan path using the provided information, and eventually exits the journey.",
    backtracking_target_step="2",
    step_advancement=[
        JourneyNodeAdvancement(
            id="2", completed=StepCompletionStatus.COMPLETED, follow_ups=["3", "4"]
        ),
        JourneyNodeAdvancement(
            id="4",
            completed=StepCompletionStatus.COMPLETED,
            follow_ups=["6"],
        ),
        JourneyNodeAdvancement(
            id="6",
            completed=StepCompletionStatus.COMPLETED,
            follow_ups=["8", "None"],
        ),
    ],
    next_step="None",
)

# Example 7: Loan Application Journey where relevant answers were provided earlier in the conversation

example_7_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Hello, I'd like to take a loan for 10,000$ and put stocks as collateral, is that possible?",
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

example_7_expected = JourneyNodeSelectionSchema(
    journey_applies=True,
    requires_backtracking=False,
    rationale="The customer wants a loan for their restaurant, making it a business loan. We can proceed through steps 4 and 6, since the customer already specified their desired loan amount and the collateral for the loan. This brings us to step 8, which was not completed yet.",
    step_advancement=[
        JourneyNodeAdvancement(
            id="2", completed=StepCompletionStatus.COMPLETED, follow_ups=["3", "4"]
        ),
        JourneyNodeAdvancement(id="4", completed=StepCompletionStatus.COMPLETED, follow_ups=["6"]),
        JourneyNodeAdvancement(
            id="6", completed=StepCompletionStatus.COMPLETED, follow_ups=["8", "None"]
        ),
        JourneyNodeAdvancement(
            id="8",
            completed=StepCompletionStatus.NEEDS_CUSTOMER_INPUT,
        ),
    ],
    next_step="8",
)

_baseline_shots: Sequence[JourneyNodeSelectionShot] = [
    JourneyNodeSelectionShot(
        description="Example 1 - Simple Single-Step Advancement",
        journey_title="Recommend Vacation Journey",
        interaction_events=example_1_events,
        journey_nodes=example_1_journey_nodes,
        expected_result=example_1_expected,
        previous_path=["1"],
        conditions=["the customer is interested in a vacation"],
    ),
    JourneyNodeSelectionShot(
        description="Example 2 - Multiple Step Advancement Stopped by Tool Calling Step",
        journey_title="Book Taxi Journey",
        interaction_events=example_2_events,
        journey_nodes=book_taxi_shot_journey_nodes,
        expected_result=example_2_expected,
        previous_path=["1", "2"],
        conditions=[],
    ),
    JourneyNodeSelectionShot(
        description="Example 3 - Multiple Step Advancement Stopped by Lacking Info",
        journey_title="Book Taxi Journey - Same Journey as in Example 2",
        interaction_events=example_3_events,
        journey_nodes=None,
        expected_result=example_3_expected,
        previous_path=["1"],
        conditions=[],
    ),
    JourneyNodeSelectionShot(
        description="Example 4 - Backtracking Due to Changed Customer Decision",
        journey_title="Book Taxi Journey - Same as in Example 2",
        interaction_events=example_4_events,
        journey_nodes=None,
        expected_result=example_4_expected,
        previous_path=["1", "2", "4", "2", "3", "5"],
        conditions=[],
    ),
    JourneyNodeSelectionShot(
        description="Example 5 - Remaining in journey unless explicitly told otherwise",
        journey_title="Book Taxi II Journey",
        interaction_events=example_5_events,
        journey_nodes=random_actions_journey_nodes,
        expected_result=example_5_expected,
        previous_path=["1"],
        conditions=["customer wants to book a taxi"],
    ),
    JourneyNodeSelectionShot(
        description="Example 6 - Backtracking and fast forwarding to Completion",
        journey_title="Loan Application Journey",
        interaction_events=example_6_events,
        journey_nodes=loan_journey_nodes,
        expected_result=example_6_expected,
        previous_path=["1", "2", "3", "5", "7"],
        conditions=["customer wants a loan"],
    ),
    JourneyNodeSelectionShot(
        description="Example 7 - fast forwarding due to information provided earlier in the conversation",
        journey_title="Loan Application Journey - Same as in Example 6",
        interaction_events=example_7_events,
        journey_nodes=loan_journey_nodes,
        expected_result=example_7_expected,
        previous_path=["1", "2"],
        conditions=["customer wants a loan"],
    ),
]


shot_collection = ShotCollection[JourneyNodeSelectionShot](_baseline_shots)
