from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from lagom import Container
from pytest import fixture

from parlant.core.common import Criticality, JSONSerializable
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.journeys import JourneyId, JourneyStore
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


async def test_that_reachable_followups_work_for_concatenated_linked_journeys(
    context: ContextOfTest,
) -> None:
    """Simulates three concatenated sub-journeys (loan application):
    identity_verification -> credit_check -> loan_approval

    Verifies that reachable follow-ups traverse through all linked sub-journeys
    without producing None paths.

    Journey structure:
    root -> [identity_ask_name (sub1)] -> [merge1] -> [credit_ask_ssn (sub2)] -> [merge2] -> [approval (sub3)] -> [merge3]
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
    ask_name = _make_guideline(
        id=f"journey_node:ask_name:e1:{link1_id}",
        condition="",
        action="Ask for the customer's full name and date of birth",
        index="1",
        journey_id=journey_id,
        follow_ups=["journey_node:merge1:leaf1"],
        customer_dependent=True,
        link_id=link1_id,
        original_node_id="ask_name",
    )

    merge1 = _make_guideline(
        id="journey_node:merge1:leaf1",
        condition="",
        action=None,
        index="2",
        journey_id=journey_id,
        follow_ups=[f"journey_node:ask_ssn:e2:{link2_id}"],
        kind="fork",
        customer_dependent=False,
    )

    # Sub-journey 2: Credit check
    ask_ssn = _make_guideline(
        id=f"journey_node:ask_ssn:e2:{link2_id}",
        condition="",
        action="Ask for the customer's SSN to run a credit check",
        index="3",
        journey_id=journey_id,
        follow_ups=["journey_node:merge2:leaf2"],
        customer_dependent=True,
        link_id=link2_id,
        original_node_id="ask_ssn",
    )

    merge2 = _make_guideline(
        id="journey_node:merge2:leaf2",
        condition="",
        action=None,
        index="4",
        journey_id=journey_id,
        follow_ups=[f"journey_node:approval:e3:{link3_id}"],
        kind="fork",
        customer_dependent=False,
    )

    # Sub-journey 3: Loan approval
    approval = _make_guideline(
        id=f"journey_node:approval:e3:{link3_id}",
        condition="",
        action="Inform the customer their loan has been approved",
        index="5",
        journey_id=journey_id,
        follow_ups=["journey_node:merge3:leaf3"],
        customer_dependent=False,
        link_id=link3_id,
        original_node_id="approval",
    )

    merge3 = _make_guideline(
        id="journey_node:merge3:leaf3",
        condition="",
        action=None,
        index="6",
        journey_id=journey_id,
        follow_ups=[],
        kind="fork",
        customer_dependent=False,
    )

    guidelines = [root, ask_name, merge1, ask_ssn, merge2, approval, merge3]

    reachable = await _evaluate_reachable(context, guidelines)

    # ask_name (index 1) should have paths going through to credit check and beyond
    assert "1" in reachable, (
        f"ask_name (index 1) should have reachable follow-ups, got keys: {list(reachable.keys())}"
    )

    for node_idx, paths in reachable.items():
        for condition, path in paths:
            assert "None" not in path, (
                f"Node {node_idx}: path contains 'None': condition={condition}, path={path}"
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
        sub.id, action="Ask the customer for their ID number", tools=[]
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
        parent.id, action="Ask the customer what type of account they want", tools=[]
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
        parent.id, action="Confirm the account has been opened", tools=[]
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

    # Verify no None paths
    for node_idx, paths in result.node_to_reachable_follow_ups.items():
        for condition, path in paths:
            assert "None" not in [str(p) for p in path], (
                f"Node {node_idx}: path contains None: condition={condition}, path={path}"
            )

    # There should be reachable follow-ups from the collect_info node
    assert len(result.node_to_reachable_follow_ups) > 0, (
        "Should have at least some reachable follow-ups"
    )
