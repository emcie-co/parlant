import copy
from dataclasses import dataclass, field
from enum import Enum
import json
import traceback
from typing import Any, List, Optional, Sequence, Set, Tuple, cast
from parlant.core.common import DefaultBaseModel, JSONSerializable
from parlant.core.engines.alpha.guideline_matching.generic.common import internal_representation
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.guidelines import Guideline, GuidelineId

from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator

from parlant.core.services.indexing.common import EvaluationError, ProgressReport
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.shots import Shot, ShotCollection


PRE_ROOT_INDEX = "0"
ROOT_INDEX = "1"
REMINDER_OF_ACTION_TYPE_CURRENT = "Reminder: when stating whether step_action has been completed, consider the rules for CUSTOMER DEPENDENT ACTION - CUSTOMER'S perspective or REQUIRES AGENT ACTION - AGENT'S perspective"
REMINDER_OF_ACTION_TYPE_CHILD = "Reminder: when stating whether child_action has been completed, consider the rules for CUSTOMER DEPENDENT ACTION - CUSTOMER'S perspective or REQUIRES AGENT ACTION - AGENT'S perspective"
REMINDER_OF_ACTION_TYPE_NOT_CHILD = "Reminder: when stating whether child_action has not been completed, consider the rules for CUSTOMER DEPENDENT ACTION - CUSTOMER'S perspective or REQUIRES AGENT ACTION - AGENT'S perspective"

REMINDER_OPTIONS = "Reminder: when stating an action completion consider Condition Clarity and Specificity, include all options in conditions"


class JourneyNodeKind(Enum):
    FORK = "fork"
    CHAT = "chat"
    TOOL = "tool"
    NA = "NA"


@dataclass
class _JourneyEdge:
    condition: str | None
    source_node_index: str
    target_node_index: str


@dataclass
class _ReachableFollowUps:
    condition: str
    path: list[str]


@dataclass
class _JourneyNode:  # Refactor after node type is implemented
    id: str
    action: str | None
    incoming_edges: list[_JourneyEdge]
    outgoing_edges: list[_JourneyEdge]
    kind: JourneyNodeKind
    customer_dependent_action: bool
    customer_action_description: Optional[str] = None
    agent_dependent_action: Optional[bool] = None
    agent_action_description: Optional[str] = None
    reachable_follow_ups: Sequence[_ReachableFollowUps] = field(default_factory=list)


@dataclass
class _ChildInfo:
    action: str | None
    edge_condition: str | None
    id_to_reachable_follow_ups: dict[str, _ReachableFollowUps] = field(default_factory=dict)
    customer_action_description: Optional[str] = None
    agent_action_description: Optional[str] = None


class PathCondition(DefaultBaseModel):
    id: str
    path_condition: str
    condition_to_child_then_to_path: str


class ChildEvaluation(DefaultBaseModel):
    child_id: str
    child_action: str
    condition_to_child: str
    condition_to_child_and_stop: str
    conditions_to_child_and_forward: Optional[list[PathCondition]] = None


class ReachableNodesEvaluationSchema(DefaultBaseModel):
    step_action: str
    step_action_completed: str
    children_conditions: Optional[Sequence[ChildEvaluation]] = None


class ReachableNodesEvaluation(DefaultBaseModel):
    node_to_reachable_follow_ups: dict[str, Sequence[tuple[str, Sequence[str]]]]


@dataclass
class JourneyReachableNodesEvaluationShot(Shot):
    node: _JourneyNode
    children_info: dict[str, _ChildInfo]
    expected_result: ReachableNodesEvaluationSchema


class JourneyReachableNodesEvaluator:
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[ReachableNodesEvaluationSchema],
        service_registry: ServiceRegistry,
    ) -> None:
        self._logger = logger
        self._optimization_policy = optimization_policy

        self._schematic_generator = schematic_generator
        self._service_registry = service_registry

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
            node_index: str = _get_guideline_node_index(g)
            if node_index not in node_wrappers:
                kind = JourneyNodeKind(
                    cast(dict[str, Any], g.metadata.get("journey_node", {})).get("kind", "NA")
                )
                customer_dependent_action = cast(
                    dict[str, bool], g.metadata.get("customer_dependent_action_data", {})
                ).get("is_customer_dependent", False)
                node_wrappers[node_index] = _JourneyNode(
                    id=node_index,
                    action=internal_representation(g).action,
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
                    condition=None,
                    source_node_index=PRE_ROOT_INDEX,
                    target_node_index=ROOT_INDEX,
                )
            )

        return node_wrappers

    def _get_dfs_ordering(self, graph: dict[str, _JourneyNode]) -> List[str]:
        # Use to standardize the cycles in dfs order, to later break by duplicate the first node
        dfs_order: List[str] = []

        visited: Set[str] = set()

        def dfs_ordering(node: str) -> None:
            visited.add(node)
            dfs_order.append(node)
            for e in graph[node].outgoing_edges:
                neighbor = e.target_node_index
                if neighbor not in visited:
                    dfs_ordering(neighbor)

        for node in graph:
            if node not in visited:
                dfs_ordering(node)

        return dfs_order

    def _find_cycles(self, graph: dict[str, _JourneyNode]) -> list[list[str]]:
        dfs_order = self._get_dfs_ordering(graph)

        dfs_index: dict[str, int] = {node: i for i, node in enumerate(dfs_order)}

        cycles: set[Tuple[str, ...]] = set()

        def canonicalize(path: list[str]) -> Tuple[str, ...]:
            min_idx = min(range(len(path)), key=lambda i: dfs_index[path[i]])

            rotated = tuple(path[min_idx:] + path[:min_idx])
            return rotated

        def dfs(start: str, node: str, visited: set[str], stack: list[str]) -> None:
            for e in graph[node].outgoing_edges:
                nxt = e.target_node_index
                if nxt == start:
                    # Found cycle
                    cycle = canonicalize(stack.copy())
                    cycles.add(cycle)
                elif nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
                    dfs(start, nxt, visited, stack)
                    stack.pop()
                    visited.remove(nxt)

        for start in graph:
            dfs(start, start, {start}, [start])

        return [list(c) for c in cycles]

    def _break_cycles(
        self, cycles: list[list[str]], graph: dict[str, _JourneyNode]
    ) -> tuple[dict[str, _JourneyNode], dict[str, str]]:
        new_graph = copy.deepcopy(graph)

        duplicate_to_orig_id: dict[str, str] = {}

        def break_cycle(cycle: list[str]) -> None:
            # For example if we have 1->2->1 it will become 1->2->1_duplicate
            start: _JourneyNode = graph[cycle[0]]
            end: _JourneyNode = graph[cycle[-1]]

            edge = None
            for e in end.outgoing_edges:
                if e.target_node_index == start.id:
                    edge = e
                    break

            dup_id = f"{start.id}_{list(duplicate_to_orig_id.values()).count(start.id) + 1}"
            new_edge = _JourneyEdge(
                condition=edge.condition if edge else None,
                source_node_index=end.id,
                target_node_index=dup_id,
            )
            dup_start = _JourneyNode(
                id=dup_id,
                action=start.action,
                incoming_edges=[new_edge],
                outgoing_edges=[],  # TODO if the node is fork we may want to duplicate also the following nodes
                kind=start.kind,
                customer_dependent_action=start.customer_dependent_action,
                customer_action_description=start.customer_action_description,
                agent_dependent_action=start.agent_dependent_action,
                agent_action_description=start.agent_action_description,
            )
            new_graph[dup_id] = dup_start

            # update start's incoming
            incoming = [e for e in start.incoming_edges if (e.source_node_index != end.id)]
            new_graph[cycle[0]].incoming_edges = incoming

            # update end's outgoing
            outgoing = [e for e in end.outgoing_edges if (e.target_node_index != start.id)] + [
                new_edge
            ]
            new_graph[cycle[-1]].outgoing_edges = outgoing

            duplicate_to_orig_id[dup_id] = start.id

        for c in cycles:
            break_cycle(c)

        return new_graph, duplicate_to_orig_id

    def _topological_sort(self, graph: dict[str, _JourneyNode]) -> List[str]:
        visited: set[str] = set()
        order: list[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            for e in graph[node].outgoing_edges:
                neighbor = e.target_node_index
                if neighbor not in visited:
                    dfs(neighbor)
            order.append(node)

        if PRE_ROOT_INDEX in graph:
            dfs(PRE_ROOT_INDEX)
        else:
            dfs(ROOT_INDEX)

        for node in graph:
            if node not in visited:
                dfs(node)

        return order

    async def evaluate_reachable_follow_ups(
        self,
        node_guidelines: Sequence[Guideline] = [],
        progress_report: Optional[ProgressReport] = None,
        max_depth: int = 3,
        max_transitions: int = 10,
    ) -> ReachableNodesEvaluation:
        if progress_report:
            await progress_report.stretch(1)

        # Want to run the evaluation in topological order, so first need to find cycles and remove them by duplicate nodes
        graph: dict[str, _JourneyNode] = self._build_node_wrappers(guidelines=node_guidelines)

        cycles = self._find_cycles(graph)
        new_graph, duplicate_to_orig_id = self._break_cycles(cycles, graph)

        order = self._topological_sort(new_graph)

        node_to_reachable_follow_ups = {}
        for node_idx in order:
            children_info: dict[str, _ChildInfo] = {}
            node = new_graph[node_idx]
            if not node.action and not node.outgoing_edges:
                continue
            for e in node.outgoing_edges:
                child_idx = e.target_node_index
                id = 1
                if (
                    not new_graph[child_idx].action
                    and len(node.outgoing_edges) == 1
                    and not e.condition
                    and not new_graph[node.outgoing_edges[0].target_node_index].outgoing_edges
                ):
                    # only one child which is a terminal node (no action and no outgoing edges) with no condition to it
                    break
                truncated_follow_ups: dict[str, _ReachableFollowUps] = {}
                # truncate paths that starts with tool node / agent action node
                if (
                    new_graph[child_idx].kind != JourneyNodeKind.TOOL
                    and not new_graph[child_idx].agent_dependent_action
                ):
                    for r in new_graph[child_idx].reachable_follow_ups:
                        # We don't want paths that exceed depth, but if they end with fork we will allow extra edge.
                        if len(r.path) + 1 <= max_depth or (
                            len(r.path) > 1 and new_graph[r.path[-2]].kind == JourneyNodeKind.FORK
                        ):
                            truncated_follow_ups[str(id)] = _ReachableFollowUps(
                                condition=r.condition,
                                path=r.path,
                            )
                            id += 1
                children_info[child_idx] = _ChildInfo(
                    action=new_graph[child_idx].action,
                    customer_action_description=new_graph[child_idx].customer_action_description,
                    agent_action_description=new_graph[child_idx].agent_action_description,
                    edge_condition=e.condition,
                    id_to_reachable_follow_ups=truncated_follow_ups,
                )

            reachable_follow_ups = await self.do_node_evaluation(
                new_graph,
                node_idx,
                children_info,
            )

            if progress_report:
                await progress_report.increment(1)

            result: list[tuple[str, Sequence[str]]] = []
            for r in reachable_follow_ups:
                path = [duplicate_to_orig_id.get(id, id) for id in r.path]
                result.append((r.condition, path))

            if node_idx not in duplicate_to_orig_id:
                node_to_reachable_follow_ups[node_idx] = result

        return ReachableNodesEvaluation(node_to_reachable_follow_ups=node_to_reachable_follow_ups)

    def get_children_info_description(
        self,
        node: _JourneyNode,
        children_info: dict[str, _ChildInfo],
    ) -> str:
        desc = ""

        if node.action:
            desc += f"""
Current node action:
    {node.action} """
        else:
            desc += """
    There is no action to take in this node"""
        if node.customer_dependent_action:
            desc += """
- CUSTOMER DEPENDENT: This action requires an action from the customer to be considered complete. Mark it as complete if the customer answered the question in the action, if there is one."""
            if node.customer_action_description:
                desc += f"""
- The action is completed if: {node.customer_action_description}"""
        elif node.agent_dependent_action:
            desc += """
- REQUIRES AGENT ACTION: This step requires from the agent to say something for it to be completed."""
            if node.agent_action_description:
                desc += f"""
- The action is completed if: {node.agent_action_description}"""

        if children_info:
            for id, info in children_info.items():
                desc += f"""
    Child id: {id}"""

                if info.action:
                    desc += f"""
        Action of child ({id}): 
        {info.action}"""
                    if info.customer_action_description:
                        desc += f"""
        - CUSTOMER DEPENDENT: This action requires an action from the customer to be considered complete. The action is completed if: {info.customer_action_description}"""
                    if info.agent_action_description:
                        desc += f"""
        - REQUIRES AGENT ACTION: This step requires from the agent to say something for it to be completed. The action is completed if: {info.agent_action_description}"""
                else:
                    desc += """
        There is no action to take in this child"""

                if info.edge_condition:
                    desc += f"""
        Condition of the transition to child ({id}):
        {info.edge_condition}"""
                if info.id_to_reachable_follow_ups:
                    desc += """
            The conditions that describe the possible paths from child onward:"""
                    for path_id, r in info.id_to_reachable_follow_ups.items():
                        desc += f"""
                - Condition ({path_id}) : {r.condition}"""
        else:
            desc += """
    This step has no children"""

        return desc

    async def shots(self) -> Sequence[JourneyReachableNodesEvaluationShot]:
        return await shot_collection.list()

    def _format_shots(self, shots: Sequence[JourneyReachableNodesEvaluationShot]) -> str:
        return "\n".join(
            f"Example #{i}\n{self._format_shot(shot)}" for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, shot: JourneyReachableNodesEvaluationShot) -> str:
        formatted_shot = ""

        formatted_shot += self.get_children_info_description(
            node=shot.node,
            children_info=shot.children_info,
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
        node: _JourneyNode,
        children_info: dict[str, _ChildInfo],
        shots: Sequence[JourneyReachableNodesEvaluationShot],
    ) -> PromptBuilder:
        builder = PromptBuilder()

        builder.add_section(
            name="journey-reachable-nodes-evaluation-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
In our system, the behavior of a conversational AI agent is structured around predefined "journeys" - structured workflows that guide customer interactions toward specific outcomes.

## Journey Structure
Each journey consists of:
- **Steps**: Individual actions that the agent must execute (e.g., ask a question, provide information, perform a task)
- **Transitions**: Rules that determine which step comes next based on customer responses or completion status
""",
        )

        builder.add_section(
            name="journey-reachable-nodes-evaluation-task-description",
            template="""
TASK DESCRIPTION
-----------------
You will be given a journey step and information about each of it's outgoing directed steps (children), and your task is to write the condition that describes the transition to each child.
The information you will have for each of the children steps is:
1. The condition of the transition from current step to each child (if exists).
2. The conditions that describe the transitions from them onward. 

The rule for creating the conditions for the given node is as follows:

1. **condition_to_child_and_stop**: The transition condition to reach the child is satisfied AND the child's action (child_action) has NOT been completed yet

2. **condition_to_child_then_to_path**: For each possible path forward from the child, combine:
   - child_action - The child's action was completed 
   - condition_to_child - The transition condition to reach the child (if doesn't exist, see "Condition to child" to more details)
   - path_condition - The path condition from the child onward  
* Do not include that current step condition was completed.
* Note that condition_to_child_then_to_path may be long, it's ok! It's important to include all condition parts to get well defined transitions.

If the current node has no children:
    - **step_action_completed ** - The condition that the current step action was completed
    - No children_conditions array is needed

Condition to child:
If a node has one child and the condition_to_child is unspecified, there is no condition to include.
If a node has a child whose condition_to_child is unspecified, while other children do have specific conditions, then in the field "condition_to_child" of the unspecified child you must 
state the complementary condition.

So eventually we will get all possible options to continue from the current node.

**Action completion:**
You will be asked to phrase conditions stating whether an action was or wasn't completed. Pay close attention to the following rules based on action type:

CUSTOMER DEPENDENT ACTION:
For actions requiring customer responses (e.g., "Ask the customer which type of pizza they want"), the action is completed when the customer provided the requested information - whether the agent explicitly requested it OR the customer volunteered it unprompted.

Always phrase completion from the CUSTOMER'S perspective, not the agent's.
- CORRECT: "The customer chose which type of pizza they want"
- WRONG: "The agent asked the customer which type of pizza they want"

The action is complete when the INFORMATION EXISTS, regardless of whether the agent asked for it.

REQUIRES AGENT ACTION:
For actions requiring the agent to communicate something, describe completion based on whether the agent fulfilled their responsibility.
- CORRECT: "The agent informed the customer that..."
- WRONG: "The customer was informed that..."


**IMPORTANT: Specify the options and details**
Conditions must be self-contained and understandable without additional context. Anyone reading the condition should be able to evaluate it against a conversation transcript without needing to reference the action or step definitions.
Therefor, when actions present specific options (e.g., "Ask if they want Margherita, Pepperoni, or Vegan"), conditions MUST specify all those options:
- CORRECT: "The customer chose which type of pizza they want - Margherita, Pepperoni, or Vegan", or "The customer hasn't chose the type of pizza they want yet (Margherita, Pepperoni, or Vegan)"
- WRONG: "The customer chose which type of pizza they want". Or  "The customer hasn't chose the type of pizza they want yet" - Without listing the option.
That's important so it will be clear that if customer said "I want 3 margarita", they completed the step. 

Notes:
- If condition contains multiple statements where one implies the other, include only the more specific one. For example "The customer specified the type of pizza they want and it is Vegan" could become "The customer wants Vegan pizza".
""",
        )
        builder.add_section(
            name="journey-reachable-nodes-evaluation-examples",
            template="""
Examples of reachable nodes evaluation:
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

        builder.add_section(
            name="journey-reachable-nodes-evaluation-node-and-children-description",
            template=self.get_children_info_description(
                node=node,
                children_info=children_info,
            ),
        )
        builder.add_section(
            name="journey-reachable-nodes-evaluation-output-format",
            template="""{output_format}""",
            props={"output_format": self._get_output_format_section(node, children_info)},
        )

        return builder

    def _sort_by_transition_condition(
        self,
        children_info: dict[str, _ChildInfo],
    ) -> list[str]:
        # If we have more than one child and there is a child with no condition in the transition, we want to present the
        # children with the condition first to help the model infer the complementary condition

        return sorted(children_info.keys(), key=lambda k: children_info[k].edge_condition is None)

    def _get_output_format_section(
        self,
        node: _JourneyNode,
        children_info: dict[str, _ChildInfo],
    ) -> str:
        def _get_children_condition() -> str:
            children_conditions = ""
            sorted_ids = self._sort_by_transition_condition(children_info)
            for id in sorted_ids:
                info = children_info[id]
                child_desc = f"""
            "child_id": "{id}",
            "child_action": "{info.action if info.action else "There is no action to perform in this child step"}",
            "condition_to_child": "{info.edge_condition if info.edge_condition else "<str.There is no condition associated with the transition to this child, if there are other children state here the complementary condition of ALL children>"}",
            "condition_to_child_and_stop": {f"<str, condition_to_child (if exists) AND that child_action hasn't completed (if exists).{REMINDER_OF_ACTION_TYPE_NOT_CHILD}. {REMINDER_OPTIONS}>" if info.action or info.edge_condition else ""},"""

                conditions_to_child_and_forward = ""
                for path_id, r in info.id_to_reachable_follow_ups.items():
                    conditions_to_child_and_forward += f"""
                {{
                    "id": "{path_id}",
                    "path_condition": "{r.condition}",
                    "condition_to_child_then_to_path": "<str, child_action completed (if exists) AND condition_to_child (if exists) AND path_condition. {REMINDER_OF_ACTION_TYPE_CHILD}. {REMINDER_OPTIONS}>",
                }},"""
                if conditions_to_child_and_forward:
                    child_desc += f"""
            "conditions_to_child_and_forward": [{conditions_to_child_and_forward}
            ]"""
                children_conditions += f"""
        {{{child_desc}
        }},"""
            return (
                f"""
    "children_conditions": [{children_conditions}
    ]"""
                if children_conditions
                else ""
            )

        return f"""
IMPORTANT: Please provide your answer in the following JSON format.

OUTPUT FORMAT
-----------------
- Fill in the following fields as instructed. Each field is required unless otherwise specified.

```json
{{
    "step_action": "{node.action if node.action else ""}",
    "step_action_completed": "{f"<str, condition that says that step_action completed, if exists. {REMINDER_OF_ACTION_TYPE_CURRENT}. {REMINDER_OPTIONS}>" if node.action else ""}",{_get_children_condition()}
}}
```
"""

    async def do_node_evaluation(
        self,
        new_graph: dict[str, _JourneyNode],
        node_idx: str,
        children_info: dict[str, _ChildInfo],
    ) -> Sequence[_ReachableFollowUps]:
        node = new_graph[node_idx]

        prompt = self._build_prompt(node, children_info, _baseline_shots)

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

                reachable_follow_ups = []

                if not children_info:
                    reachable_follow_ups.append(
                        _ReachableFollowUps(
                            condition=inference.content.step_action_completed, path=["None"]
                        )
                    )
                elif inference.content.children_conditions:
                    for c in inference.content.children_conditions:
                        # Condition of the path that ends with child
                        if not new_graph[c.child_id].kind == JourneyNodeKind.FORK:
                            if (
                                not children_info[c.child_id].action
                                and not new_graph[c.child_id].kind == JourneyNodeKind.FORK
                            ):
                                path = ["None"]
                            else:
                                path = [c.child_id]
                            reachable_follow_ups.append(
                                _ReachableFollowUps(
                                    condition=c.condition_to_child_and_stop,
                                    path=path,
                                )
                            )

                        # Conditions of the paths to child and forward
                        if c.conditions_to_child_and_forward:
                            for p in c.conditions_to_child_and_forward:
                                path = (
                                    children_info[c.child_id].id_to_reachable_follow_ups[p.id].path
                                )
                                path.insert(0, c.child_id)
                                reachable_follow_ups.append(
                                    _ReachableFollowUps(
                                        condition=p.condition_to_child_then_to_path,
                                        path=path,
                                    )
                                )
                # update field in graph node for parents evaluations
                new_graph[node_idx].reachable_follow_ups = reachable_follow_ups
                return reachable_follow_ups

            except Exception as exc:
                self._logger.warning(
                    f"Attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                )

                last_generation_exception = exc

        raise EvaluationError() from last_generation_exception


node_example_1 = _JourneyNode(
    id="2",
    action="Ask the customer for their desired pick up location",
    incoming_edges=[
        _JourneyEdge(
            condition="",
            source_node_index="1",
            target_node_index="2",
        )
    ],
    outgoing_edges=[
        _JourneyEdge(
            condition="The desired pick up location is in NYC",
            source_node_index="2",
            target_node_index="3",
        ),
        _JourneyEdge(
            condition="The desired pick up location is outside of NYC",
            source_node_index="2",
            target_node_index="4",
        ),
    ],
    kind=JourneyNodeKind.CHAT,
    customer_dependent_action=True,
    customer_action_description="the customer provided their desired pick up location",
    reachable_follow_ups=[  # This is the expected result
        _ReachableFollowUps(
            condition="The customer's desired pick up location is in NYC and customer hasn't provided their destination location yet",
            path=["3"],
        ),
        _ReachableFollowUps(
            condition="The customer's desired pick up location is outside of NYC and the agent hasn't informed the customer that we do not operate outside of NYC",
            path=["4"],
        ),
        _ReachableFollowUps(
            condition="The customer's desired pick up location is in NYC and they provided their destination location but hasn't provided the pickup time yet",
            path=["3", "5"],
        ),
        _ReachableFollowUps(
            condition="The customer's desired pick up location is in NYC and and they provided their destination location and pickup time but the agent hasn't booked the taxi ride yet",
            path=["3", "5", "6"],
        ),
    ],
)
children_info_example_1 = {
    "3": _ChildInfo(
        action="Ask where their destination is",
        edge_condition="The desired pick up location is in NYC",
        id_to_reachable_follow_ups={
            "1": _ReachableFollowUps(
                condition="The customer provided their destination location but hasn't provided the pickup time yet",
                path=["5"],
            ),
            "2": _ReachableFollowUps(
                condition="they provided their destination location and pickup time but the agent hasn't booked the taxi ride yet",
                path=["5", "6"],
            ),
        },
    ),
    "4": _ChildInfo(
        action="Inform the customer that we do not operate outside of NYC",
        edge_condition="The desired pick up location is outside of NYC",
        id_to_reachable_follow_ups={
            "1": _ReachableFollowUps(
                condition="The agent informed the customer that we do not operate outside of NYC",
                path=["None"],
            ),
        },
    ),
}

expected_result_example_1 = ReachableNodesEvaluationSchema(
    step_action=node_example_1.action,
    step_action_completed="The customer provided their desired pick up location",
    children_conditions=[
        ChildEvaluation(
            child_id="3",
            child_action=children_info_example_1["3"].action,
            condition_to_child=children_info_example_1["3"].edge_condition,
            condition_to_child_and_stop="The customer's desired pick up location is in NYC and customer hasn't provided their destination location yet",
            conditions_to_child_and_forward=[
                PathCondition(
                    id="1",
                    path_condition=children_info_example_1["3"]
                    .id_to_reachable_follow_ups["1"]
                    .condition,
                    condition_to_child_then_to_path="The customer's desired pick up location is in NYC and they provided their destination location but hasn't provided the pickup time yet",
                ),
                PathCondition(
                    id="2",
                    path_condition=children_info_example_1["3"]
                    .id_to_reachable_follow_ups["2"]
                    .condition,
                    condition_to_child_then_to_path="The customer's desired pick up location is in NYC and they provided their destination location and pickup time but the agent hasn't booked the taxi ride yets",
                ),
            ],
        ),
        ChildEvaluation(
            child_id="4",
            child_action=children_info_example_1["4"].action,
            condition_to_child=children_info_example_1["4"].edge_condition,
            condition_to_child_and_stop="The desired pick up location is outside of NYC and the agent informed the customer that we do not operate outside of NYC",
            conditions_to_child_and_forward=[
                PathCondition(
                    id="1",
                    path_condition=children_info_example_1["4"]
                    .id_to_reachable_follow_ups["1"]
                    .condition,
                    condition_to_child_then_to_path="The customer's desired pick up location is outside of NYC and the agent hasn't informed the customer that we do not operate outside of NYC",
                ),
            ],
        ),
    ],
)

node_example_2 = _JourneyNode(
    id="5",
    action="Ask the customer what's their shipping address",
    incoming_edges=[
        _JourneyEdge(
            condition="The customer provided the amount of items",
            source_node_index="4",
            target_node_index="5",
        )
    ],
    outgoing_edges=[
        _JourneyEdge(
            condition="",
            source_node_index="5",
            target_node_index="6",
        ),
    ],
    kind=JourneyNodeKind.CHAT,
    customer_dependent_action=True,
    customer_action_description="The customer provided their shipping address",
    reachable_follow_ups=[  # This is the expected result
        _ReachableFollowUps(
            condition="The customer hasn't chosen the delivery speed they prefer: Standard (5-7 days), Express (2-3 days), or Overnight",
            path=["6"],
        ),
        _ReachableFollowUps(
            condition="The customer chose the delivery speed (Standard, Express or Overnight) but hasn't provided the payment method (cash or credit)",
            path=["6", "7"],
        ),
        _ReachableFollowUps(
            condition="The customer chose the delivery speed (Standard, Express or Overnight) and provided the payment method (cash or credit) but the agent hasn't confirmed the order yet",
            path=["6", "7", "8"],
        ),
    ],
)

children_info_example_2 = {
    "6": _ChildInfo(
        action="Ask the customer which delivery speed they prefer: Standard (5-7 days), Express (2-3 days), or Overnight",
        edge_condition="",
        id_to_reachable_follow_ups={
            "1": _ReachableFollowUps(
                condition="The customer hasn't chosen their payment method - cash or credit",
                path=["7"],
            ),
            "2": _ReachableFollowUps(
                condition="The customer chose their payment method - cash or credit, and the agent hasn't confirmed the order yet",
                path=["7", "8"],
            ),
        },
    ),
}

expected_result_example_2 = ReachableNodesEvaluationSchema(
    step_action=node_example_2.action,
    step_action_completed="The customer provided the shipping address",
    children_conditions=[
        ChildEvaluation(
            child_id="6",
            child_action=children_info_example_2["6"].action,
            condition_to_child=children_info_example_2["6"].edge_condition,
            condition_to_child_and_stop="The customer hasn't chosen the delivery speed they prefer: Standard (5-7 days), Express (2-3 days), or Overnight",
            conditions_to_child_and_forward=[
                PathCondition(
                    id="1",
                    path_condition=children_info_example_2["6"]
                    .id_to_reachable_follow_ups["1"]
                    .condition,
                    condition_to_child_then_to_path="The customer chose the delivery speed (Standard, Express or Overnight) but hasn't provided the payment method (cash or credit)",
                ),
                PathCondition(
                    id="2",
                    path_condition=children_info_example_2["6"]
                    .id_to_reachable_follow_ups["2"]
                    .condition,
                    condition_to_child_then_to_path="The customer chose the delivery speed (Standard, Express or Overnight) and provided the payment method (cash or credit) but the agent hasn't confirmed the order yet",
                ),
            ],
        ),
    ],
)

_baseline_shots: Sequence[JourneyReachableNodesEvaluationShot] = [
    JourneyReachableNodesEvaluationShot(
        description="",
        node=node_example_1,
        children_info=children_info_example_1,
        expected_result=expected_result_example_1,
    ),
    JourneyReachableNodesEvaluationShot(
        description="Elaborate the options and details in the condition",
        node=node_example_2,
        children_info=children_info_example_2,
        expected_result=expected_result_example_2,
    ),
]

shot_collection = ShotCollection[JourneyReachableNodesEvaluationShot](_baseline_shots)
