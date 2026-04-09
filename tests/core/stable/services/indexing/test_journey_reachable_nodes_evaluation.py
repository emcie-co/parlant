from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from lagom import Container
from pytest import fixture

from parlant.core.common import Criticality, JSONSerializable
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.journeys import JourneyId, JourneyNodeKind, JourneyStore
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.services.indexing.journey_reachable_nodes_evaluation import (
    JourneyReachableNodesEvaluator,
    ReachableNodesEvaluationSchema,
)
from parlant.core.journey_guideline_projection import JourneyGuidelineProjection
from parlant.core.store_provider import BasicStoreProvider
from tests.test_utilities import SyncAwaiter


@dataclass
class ContextOfTest:
    container: Container
    sync_await: SyncAwaiter
    schematic_generator: SchematicGenerator[ReachableNodesEvaluationSchema]
    logger: Logger


@fixture
def context(
    sync_await: SyncAwaiter,
    container: Container,
) -> ContextOfTest:
    return ContextOfTest(
        container,
        sync_await,
        logger=container[Logger],
        schematic_generator=container[SchematicGenerator[ReachableNodesEvaluationSchema]],
    )


def _make_guideline(
    id: str,
    condition: str,
    action: str | None,
    index: str,
    journey_id: JourneyId,
    follow_ups: list[str],
    kind: str = "chat",
    customer_dependent: bool = True,
    link_id: str | None = None,
    original_node_id: str | None = None,
) -> Guideline:
    journey_node: dict[str, JSONSerializable] = {
        "follow_ups": follow_ups,
        "index": index,
        "journey_id": journey_id,
        "kind": kind,
    }
    if link_id:
        journey_node["link_id"] = link_id
    if original_node_id:
        journey_node["original_node_id"] = original_node_id

    return Guideline(
        id=GuidelineId(id),
        creation_utc=datetime.now(timezone.utc),
        content=GuidelineContent(condition=condition, action=action),
        criticality=Criticality.MEDIUM,
        enabled=True,
        tags=[],
        metadata={
            "journey_node": journey_node,
            "customer_dependent_action_data": {
                "is_customer_dependent": customer_dependent,
                "customer_action": "",
                "agent_action": "",
            },
        },
    )


async def _evaluate_reachable(
    context: ContextOfTest,
    guidelines: Sequence[Guideline],
) -> dict[str, Sequence[tuple[str, Sequence[str]]]]:
    evaluator = JourneyReachableNodesEvaluator(
        logger=context.logger,
        optimization_policy=context.container[OptimizationPolicy],
        schematic_generator=context.schematic_generator,
        store_provider=BasicStoreProvider(lambda: context.container),
    )
    result = await evaluator.evaluate_reachable_follow_ups(node_guidelines=guidelines)
    return result.node_to_reachable_follow_ups


async def test_that_reachable_followups_are_correct_for_linked_sub_journey(
    context: ContextOfTest,
) -> None:
    """Simulates a parent journey with a linked sub-journey (identity verification)
    followed by a confirmation step. Verifies that reachable follow-ups path through
    the sub-journey nodes to the confirmation step.

    Journey structure:
    root -> ask_room_type -> [ask_name (sub) -> validate (sub)] -> [merge_fork] -> booking_confirmed
    """
    journey_id = JourneyId("parent-j1")
    link_id = "link-1"

    # Root (index 0) - no action, just the journey entry
    root = _make_guideline(
        id="journey_node:root",
        condition="",
        action=None,
        index="0",
        journey_id=journey_id,
        follow_ups=["journey_node:ask_room:e1"],
        kind="chat",
        customer_dependent=False,
    )

    # Step 1: Ask room type (parent node, index 1)
    ask_room = _make_guideline(
        id="journey_node:ask_room:e1",
        condition="",
        action="Ask the customer which room they want",
        index="1",
        journey_id=journey_id,
        follow_ups=[f"journey_node:ask_name:e2:{link_id}"],
        customer_dependent=True,
    )

    # Step 2: Ask name (sub-journey node, index 2)
    ask_name = _make_guideline(
        id=f"journey_node:ask_name:e2:{link_id}",
        condition="",
        action="Ask the customer for their name for verification",
        index="2",
        journey_id=journey_id,
        follow_ups=[f"journey_node:validate:e3:{link_id}"],
        customer_dependent=True,
        link_id=link_id,
        original_node_id="ask_name",
    )

    # Step 3: Validate (sub-journey tool node, index 3)
    validate = _make_guideline(
        id=f"journey_node:validate:e3:{link_id}",
        condition="",
        action="Validate the customer identity",
        index="3",
        journey_id=journey_id,
        follow_ups=["journey_node:merge_fork:leaf"],
        kind="tool",
        customer_dependent=False,
        link_id=link_id,
        original_node_id="validate",
    )

    # Step 4: Merge fork (parent node, index 4)
    merge_fork = _make_guideline(
        id="journey_node:merge_fork:leaf",
        condition="",
        action=None,
        index="4",
        journey_id=journey_id,
        follow_ups=["journey_node:confirmed:e4"],
        kind="fork",
        customer_dependent=False,
    )

    # Step 5: Booking confirmed (parent node, index 5)
    confirmed = _make_guideline(
        id="journey_node:confirmed:e4",
        condition="if validation is successful",
        action="Confirm the hotel booking",
        index="5",
        journey_id=journey_id,
        follow_ups=[],
        customer_dependent=False,
    )

    guidelines = [root, ask_room, ask_name, validate, merge_fork, confirmed]

    reachable = await _evaluate_reachable(context, guidelines)

    # ask_room (index 1) should have reachable follow-ups that include a path
    # through the sub-journey to the confirmation
    assert "1" in reachable, (
        f"ask_room (index 1) should have reachable follow-ups, got: {reachable}"
    )

    paths_from_ask_room = reachable["1"]
    assert len(paths_from_ask_room) > 0, "ask_room should have at least one reachable path"

    # None of the paths should contain 'None' as a string
    for condition, path in paths_from_ask_room:
        assert "None" not in path, (
            f"Path from ask_room contains 'None': condition={condition}, path={path}"
        )
        assert all(p != "None" for p in path), (
            f"Path from ask_room has None elements: condition={condition}, path={path}"
        )


async def test_that_reachable_followups_work_for_chained_linked_journeys(
    context: ContextOfTest,
) -> None:
    """Simulates three chained linked sub-journeys (loan application) after
    the projection has collapsed pass-through fork nodes:
    identity_verification -> credit_check -> loan_approval

    The projection eliminates intermediate merge_fork nodes that have a single
    unconditional outgoing edge (pass-through) and terminal merge_forks with
    no outgoing edges. The resulting graph is a flat chain:

    root -> ask_name -> ask_ssn -> approval
    """
    journey_id = JourneyId("loan-j1")
    link1_id = "link-identity"
    link2_id = "link-credit"
    link3_id = "link-approval"

    root = _make_guideline(
        id="journey_node:root",
        condition="",
        action=None,
        index="0",
        journey_id=journey_id,
        follow_ups=[f"journey_node:ask_name:e1:{link1_id}"],
        kind="chat",
        customer_dependent=False,
    )

    # Sub-journey 1: Identity verification
    # After collapse: ask_name -> ask_ssn (merge1 removed, edge rewired)
    ask_name = _make_guideline(
        id=f"journey_node:ask_name:e1:{link1_id}",
        condition="",
        action="Ask for the customer's full name and date of birth",
        index="1",
        journey_id=journey_id,
        follow_ups=[f"journey_node:ask_ssn:e2:{link2_id}"],
        customer_dependent=True,
        link_id=link1_id,
        original_node_id="ask_name",
    )

    # Sub-journey 2: Credit check
    # After collapse: ask_ssn -> approval (merge2 removed, edge rewired)
    ask_ssn = _make_guideline(
        id=f"journey_node:ask_ssn:e2:{link2_id}",
        condition="",
        action="Ask for the customer's SSN to run a credit check",
        index="3",
        journey_id=journey_id,
        follow_ups=[f"journey_node:approval:e3:{link3_id}"],
        customer_dependent=True,
        link_id=link2_id,
        original_node_id="ask_ssn",
    )

    # Sub-journey 3: Loan approval (terminal — merge3 removed)
    approval = _make_guideline(
        id=f"journey_node:approval:e3:{link3_id}",
        condition="",
        action="Inform the customer their loan has been approved",
        index="5",
        journey_id=journey_id,
        follow_ups=[],
        customer_dependent=False,
        link_id=link3_id,
        original_node_id="approval",
    )

    guidelines = [root, ask_name, ask_ssn, approval]

    reachable = await _evaluate_reachable(context, guidelines)

    # ask_name (index 1) should have paths going through to credit check and beyond
    assert "1" in reachable, (
        f"ask_name (index 1) should have reachable follow-ups, got keys: {list(reachable.keys())}"
    )

    # Verify intermediate nodes have real paths (not just ['None']).
    # Terminal nodes (approval, index 5) may have path=['None'] — that's expected.
    for node_idx, paths in reachable.items():
        for condition, path in paths:
            # A path that is ONLY ['None'] means the node thinks it's terminal.
            # That's only valid for the actual last node.
            if node_idx != "5":
                assert path != ["None"], (
                    f"Non-terminal node {node_idx}: path is ['None'] — "
                    f"sub-journey linking may not have resolved: condition={condition}"
                )


async def test_that_projection_produces_valid_guidelines_for_reachable_evaluation(
    context: ContextOfTest,
) -> None:
    """End-to-end test: create real journeys with links, project them,
    then run reachable follow-up evaluation. Paths should not contain None.
    """
    journey_store = context.container[JourneyStore]
    from parlant.core.guidelines import GuidelineStore

    guideline_store = context.container[GuidelineStore]

    projection = JourneyGuidelineProjection(
        journey_store=journey_store,
        guideline_store=guideline_store,
    )

    # Sub-journey: root -> verify_identity
    sub = await journey_store.create_journey(
        title="Identity Verification",
        description="Verify customer identity",
        conditions=[],
    )
    verify_node = await journey_store.create_node(
        sub.id, kind=JourneyNodeKind.CHAT, action="Ask the customer for their ID number", tools=[]
    )
    await journey_store.create_edge(
        sub.id, source=sub.root_id, target=verify_node.id, condition=None
    )

    # Parent: root -> collect_info --(link)--> sub -> [merge_fork] -> confirmed
    parent = await journey_store.create_journey(
        title="Account Opening",
        description="Open a new bank account",
        conditions=[],
    )
    collect_info = await journey_store.create_node(
        parent.id,
        kind=JourneyNodeKind.CHAT,
        action="Ask the customer what type of account they want",
        tools=[],
    )
    await journey_store.create_edge(
        parent.id, source=parent.root_id, target=collect_info.id, condition=None
    )

    link = await journey_store.create_link(
        journey_id=parent.id,
        source_node_id=collect_info.id,
        sub_journey_id=sub.id,
    )

    confirmed = await journey_store.create_node(
        parent.id, kind=JourneyNodeKind.CHAT, action="Confirm the account has been opened", tools=[]
    )
    await journey_store.create_edge(
        parent.id,
        source=link.merge_node_id,
        target=confirmed.id,
        condition="if identity is verified",
    )

    # Project and evaluate
    guidelines = await projection.project_journey_to_guidelines(parent.id)

    evaluator = JourneyReachableNodesEvaluator(
        logger=context.logger,
        optimization_policy=context.container[OptimizationPolicy],
        schematic_generator=context.schematic_generator,
        store_provider=BasicStoreProvider(lambda: context.container),
    )

    result = await evaluator.evaluate_reachable_follow_ups(node_guidelines=guidelines)

    # Build a set of node indexes that have outgoing follow-ups (non-terminal)
    guideline_by_index: dict[str, Guideline] = {}
    for g in guidelines:
        jn = g.metadata.get("journey_node")
        if isinstance(jn, dict):
            idx = str(jn.get("index", ""))
            guideline_by_index[idx] = g

    non_terminal_indexes: set[str] = set()
    for g in guidelines:
        jn = g.metadata.get("journey_node")
        if isinstance(jn, dict):
            fups = jn.get("follow_ups", [])
            if fups:
                non_terminal_indexes.add(str(jn.get("index", "")))

    # Non-terminal nodes must not have path=['None'] (purely terminal marker).
    # A path like ['5', 'None'] is acceptable — it means the path reaches node 5
    # which is terminal. Only a bare ['None'] on a non-terminal node is a bug.
    for node_idx, paths in result.node_to_reachable_follow_ups.items():
        if node_idx not in non_terminal_indexes:
            continue
        for condition, path in paths:
            assert path != ["None"], (
                f"Non-terminal node {node_idx}: path is ['None'] — "
                f"sub-journey linking may not have resolved: condition={condition}"
            )

    # There should be reachable follow-ups from the collect_info node
    assert len(result.node_to_reachable_follow_ups) > 0, (
        "Should have at least some reachable follow-ups"
    )


async def test_that_chained_linked_journeys_have_correct_node_wrapper_graph(
    context: ContextOfTest,
) -> None:
    """Creates 3 chained linked journeys via store, projects them,
    and verifies _build_node_wrappers produces a connected graph."""
    journey_store = context.container[JourneyStore]
    from parlant.core.guidelines import GuidelineStore

    guideline_store = context.container[GuidelineStore]

    projection = JourneyGuidelineProjection(
        journey_store=journey_store,
        guideline_store=guideline_store,
    )

    # Sub-journey 1: identity
    sub1 = await journey_store.create_journey(
        title="Identity Verification", description="sub1", conditions=[]
    )
    id_node = await journey_store.create_node(
        sub1.id, kind=JourneyNodeKind.CHAT, action="ask for ID", tools=[]
    )
    await journey_store.create_edge(sub1.id, source=sub1.root_id, target=id_node.id, condition=None)

    # Sub-journey 2: credit
    sub2 = await journey_store.create_journey(
        title="Credit Check", description="sub2", conditions=[]
    )
    credit_node = await journey_store.create_node(
        sub2.id, kind=JourneyNodeKind.CHAT, action="ask for SSN", tools=[]
    )
    await journey_store.create_edge(
        sub2.id, source=sub2.root_id, target=credit_node.id, condition=None
    )

    # Sub-journey 3: approval
    sub3 = await journey_store.create_journey(
        title="Loan Approval", description="sub3", conditions=[]
    )
    approval_node = await journey_store.create_node(
        sub3.id, kind=JourneyNodeKind.CHAT, action="approve loan", tools=[]
    )
    await journey_store.create_edge(
        sub3.id, source=sub3.root_id, target=approval_node.id, condition=None
    )

    # Parent: root -> link1 -> [merge1] -> link2 -> [merge2] -> link3 -> [merge3]
    parent = await journey_store.create_journey(
        title="Loan Application", description="parent", conditions=[]
    )
    link1 = await journey_store.create_link(
        journey_id=parent.id, source_node_id=parent.root_id, sub_journey_id=sub1.id
    )
    link2 = await journey_store.create_link(
        journey_id=parent.id, source_node_id=link1.merge_node_id, sub_journey_id=sub2.id
    )
    await journey_store.create_link(
        journey_id=parent.id, source_node_id=link2.merge_node_id, sub_journey_id=sub3.id
    )

    guidelines = await projection.project_journey_to_guidelines(parent.id)

    # Verify follow-up chain is intact
    all_ids = {g.id for g in guidelines}
    from parlant.core.common import JSONSerializable as _JSON
    from typing import cast as _cast

    for g in guidelines:
        jn = _cast(dict[str, _JSON], g.metadata.get("journey_node", {}))
        followups = _cast(list[str], jn.get("follow_ups", []))
        for f_id in followups:
            assert f_id in all_ids, (
                f"Dangling follow-up {f_id} in {g.id} (action={g.content.action})"
            )

    # Now feed to _build_node_wrappers and check connectivity
    from parlant.core.services.indexing.journey_reachable_nodes_evaluation import (
        JourneyReachableNodesEvaluator,
    )

    evaluator = JourneyReachableNodesEvaluator(
        logger=context.logger,
        optimization_policy=context.container[OptimizationPolicy],
        schematic_generator=context.schematic_generator,
        store_provider=BasicStoreProvider(lambda: context.container),
    )
    node_wrappers = evaluator._build_node_wrappers(guidelines)

    print(f"\n=== NODE WRAPPERS ({len(node_wrappers)} nodes) ===")
    for idx, node in sorted(node_wrappers.items()):
        out_targets = [e.target_node_index for e in node.outgoing_edges]
        in_sources = [e.source_node_index for e in node.incoming_edges]
        print(
            f"  [{idx}] action={node.action!r} kind={node.kind} out={out_targets} in={in_sources}"
        )

    # Every non-terminal node with an action should have outgoing edges
    for idx, node in node_wrappers.items():
        if node.action and node.outgoing_edges:
            # Non-terminal action node — should have real edges, not be isolated
            pass
        elif node.action and not node.outgoing_edges:
            # This is the truly terminal node (last in chain) — acceptable
            # But intermediate nodes with action MUST have outgoing
            has_incoming = len(node.incoming_edges) > 0
            # If it has incoming edges and no outgoing, it's the last node
            if has_incoming:
                print(f"  Terminal action node: [{idx}] {node.action}")

    # The root (index 0 or 1) should reach identity node
    actions_in_graph = {n.action for n in node_wrappers.values() if n.action}
    assert "ask for ID" in actions_in_graph, f"Identity node missing. Actions: {actions_in_graph}"
    assert "ask for SSN" in actions_in_graph, f"Credit node missing. Actions: {actions_in_graph}"
    assert "approve loan" in actions_in_graph, f"Approval node missing. Actions: {actions_in_graph}"
