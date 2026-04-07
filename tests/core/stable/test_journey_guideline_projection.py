from typing import Sequence, cast
from lagom import Container

from parlant.core.common import JSONSerializable
from parlant.core.guidelines import Guideline, GuidelineStore
from parlant.core.journey_guideline_projection import (
    JourneyGuidelineProjection,
    extract_link_id_from_journey_node_guideline_id,
    extract_node_id_from_journey_node_guideline_id,
)
from parlant.core.journeys import JourneyStore


async def test_that_projection_yields_followup_for_existing_guideline(container: Container) -> None:
    journey_store = container[JourneyStore]
    guideline_store = container[GuidelineStore]

    projection = JourneyGuidelineProjection(
        journey_store=journey_store,
        guideline_store=guideline_store,
    )

    journey = await journey_store.create_journey(
        title="Broken Follow-up Journey",
        description="Test bug with dangling follow_up",
        triggers=[],
    )

    node_a = await journey_store.create_node(
        journey.id,
        action="ask_name",
        tools=[],
    )

    node_b = await journey_store.create_node(
        journey.id,
        action="ask_email",
        tools=[],
    )

    _ = await journey_store.create_edge(
        journey.id,
        source=node_a.id,
        target=node_b.id,
        condition="got_name",
    )

    guidelines = await projection.project_journey_to_guidelines(journey.id)

    all_ids = {g.id for g in guidelines}

    for g in guidelines:
        followups = cast(dict[str, JSONSerializable], g.metadata.get("journey_node", {})).get(
            "follow_ups", []
        )
        for f_id in cast(list[str], followups):
            assert f_id in all_ids, (
                f"Bug: follow-up ID {f_id} listed in {g.id} but no guideline was created for it"
            )


def _get_actions(guidelines: Sequence[Guideline]) -> set[str | None]:
    return {g.content.action for g in guidelines}


async def test_that_projection_resolves_multi_step_sub_journey_with_tool(
    container: Container,
) -> None:
    """Mimics the SDK validation linking test structure to debug projection."""
    journey_store = container[JourneyStore]
    guideline_store = container[GuidelineStore]

    projection = JourneyGuidelineProjection(
        journey_store=journey_store,
        guideline_store=guideline_store,
    )

    # Sub-journey: root -> ask_name -> validate_tool (leaf)
    sub = await journey_store.create_journey(
        title="Validation Sub",
        description="sub",
        conditions=[],
    )
    ask_name = await journey_store.create_node(sub.id, action="ask for name", tools=[])
    validate_tool = await journey_store.create_node(sub.id, action="validate", tools=[])
    await journey_store.create_edge(sub.id, source=sub.root_id, target=ask_name.id, condition=None)
    await journey_store.create_edge(
        sub.id, source=ask_name.id, target=validate_tool.id, condition=None
    )

    # Parent: root -> room_type --(link)--> sub --> [merge_fork] -> confirmed / denied
    parent = await journey_store.create_journey(
        title="Parent",
        description="parent",
        conditions=[],
    )
    room_type = await journey_store.create_node(parent.id, action="ask room type", tools=[])
    await journey_store.create_edge(
        parent.id, source=parent.root_id, target=room_type.id, condition=None
    )

    link = await journey_store.create_link(
        journey_id=parent.id,
        source_node_id=room_type.id,
        sub_journey_id=sub.id,
    )

    confirmed = await journey_store.create_node(parent.id, action="booking confirmed", tools=[])
    denied = await journey_store.create_node(parent.id, action="booking denied", tools=[])
    await journey_store.create_edge(
        parent.id,
        source=link.merge_node_id,
        target=confirmed.id,
        condition="if validation successful",
    )
    await journey_store.create_edge(
        parent.id,
        source=link.merge_node_id,
        target=denied.id,
        condition="if validation failed",
    )

    guidelines = await projection.project_journey_to_guidelines(parent.id)

    actions = _get_actions(guidelines)
    # All nodes should be represented
    assert "ask room type" in actions
    assert "ask for name" in actions
    assert "validate" in actions
    assert "booking confirmed" in actions
    assert "booking denied" in actions

    # Follow-up chain integrity
    all_ids = {g.id for g in guidelines}
    for g in guidelines:
        followups = cast(dict[str, JSONSerializable], g.metadata.get("journey_node", {})).get(
            "follow_ups", []
        )
        for f_id in cast(list[str], followups):
            assert f_id in all_ids, (
                f"Follow-up ID {f_id} listed in {g.id} but no guideline was created for it"
            )

    # Verify the chain: room_type -> ask_name -> validate -> merge_fork -> confirmed/denied
    room_g = next(g for g in guidelines if g.content.action == "ask room type")
    room_followups = cast(
        list[str],
        cast(dict[str, JSONSerializable], room_g.metadata["journey_node"])["follow_ups"],
    )
    assert len(room_followups) > 0, "room_type should have follow-ups to sub-journey"

    # ask_name should have follow-up to validate
    name_g = [g for g in guidelines if g.content.action == "ask for name"]
    assert len(name_g) >= 1
    name_followups = cast(
        list[str],
        cast(dict[str, JSONSerializable], name_g[0].metadata["journey_node"])["follow_ups"],
    )
    assert len(name_followups) > 0, "ask_name should have follow-up to validate"

    # validate should have follow-up to merge_fork
    val_g = [g for g in guidelines if g.content.action == "validate"]
    assert len(val_g) >= 1
    val_followups = cast(
        list[str],
        cast(dict[str, JSONSerializable], val_g[0].metadata["journey_node"])["follow_ups"],
    )
    assert len(val_followups) > 0, "validate should have follow-up to merge_fork"

    # merge_fork should have follow-ups to confirmed and denied
    # The merge_fork has action=None
    fork_guidelines = [
        g
        for g in guidelines
        if g.content.action is None
        and cast(dict[str, JSONSerializable], g.metadata.get("journey_node", {})).get("kind")
        == "fork"
    ]
    # There could be multiple guidelines for the fork (one per incoming edge)
    fork_followup_actions: set[str | None] = set()
    for fg in fork_guidelines:
        fups = cast(
            list[str],
            cast(dict[str, JSONSerializable], fg.metadata["journey_node"])["follow_ups"],
        )
        for fup_id in fups:
            fup_g = next((g for g in guidelines if g.id == fup_id), None)
            if fup_g:
                fork_followup_actions.add(fup_g.content.action)
    assert "booking confirmed" in fork_followup_actions, (
        f"merge_fork should have follow-up to 'booking confirmed', got {fork_followup_actions}"
    )
    assert "booking denied" in fork_followup_actions, (
        f"merge_fork should have follow-up to 'booking denied', got {fork_followup_actions}"
    )


async def test_that_projection_resolves_sub_journey_link(container: Container) -> None:
    journey_store = container[JourneyStore]
    guideline_store = container[GuidelineStore]

    projection = JourneyGuidelineProjection(
        journey_store=journey_store,
        guideline_store=guideline_store,
    )

    # Create sub-journey: root -> sub_node
    sub = await journey_store.create_journey(
        title="Sub Journey",
        description="sub",
        conditions=[],
    )
    sub_node = await journey_store.create_node(sub.id, action="do sub thing", tools=[])
    await journey_store.create_edge(sub.id, source=sub.root_id, target=sub_node.id, condition=None)

    # Create parent journey: root -> source_node --(link)--> sub journey -> merge_node
    parent = await journey_store.create_journey(
        title="Parent Journey",
        description="parent",
        conditions=[],
    )
    source_node = await journey_store.create_node(parent.id, action="ask something", tools=[])
    await journey_store.create_edge(
        parent.id, source=parent.root_id, target=source_node.id, condition=None
    )

    await journey_store.create_link(
        journey_id=parent.id,
        source_node_id=source_node.id,
        sub_journey_id=sub.id,
        condition="if user wants sub",
    )

    guidelines = await projection.project_journey_to_guidelines(parent.id)

    for g in guidelines:
        jn = cast(dict[str, JSONSerializable], g.metadata.get("journey_node", {}))
        print(
            f"  {g.id} | cond={g.content.condition!r} | action={g.content.action!r} | followups={jn.get('follow_ups', [])}"
        )

    actions = _get_actions(guidelines)

    # Should include: root action, source_node action, sub_node action, merge fork (None action)
    assert "ask something" in actions
    assert "do sub thing" in actions

    # Verify follow-up chain integrity
    all_ids = {g.id for g in guidelines}
    for g in guidelines:
        followups = cast(dict[str, JSONSerializable], g.metadata.get("journey_node", {})).get(
            "follow_ups", []
        )
        for f_id in cast(list[str], followups):
            assert f_id in all_ids, (
                f"Follow-up ID {f_id} listed in {g.id} but no guideline was created for it"
            )

    # Source node should have follow-up to sub-journey node (via the link condition)
    source_guidelines = [g for g in guidelines if g.content.action == "ask something"]
    assert len(source_guidelines) >= 1
    source_g = source_guidelines[0]
    source_followups = cast(
        list[str],
        cast(dict[str, JSONSerializable], source_g.metadata["journey_node"])["follow_ups"],
    )
    assert len(source_followups) > 0, "Source node should have follow-ups to sub-journey nodes"


async def test_that_projection_includes_link_id_in_guideline_ids_for_linked_nodes(
    container: Container,
) -> None:
    journey_store = container[JourneyStore]
    guideline_store = container[GuidelineStore]

    projection = JourneyGuidelineProjection(
        journey_store=journey_store,
        guideline_store=guideline_store,
    )

    # Create sub-journey: root -> sub_node -> END
    sub = await journey_store.create_journey(
        title="Sub Journey",
        description="sub",
        conditions=[],
    )
    sub_node = await journey_store.create_node(sub.id, action="sub action", tools=[])
    await journey_store.create_edge(sub.id, source=sub.root_id, target=sub_node.id, condition=None)

    # Create parent: root -> source_node --(link)--> sub
    parent = await journey_store.create_journey(
        title="Parent Journey",
        description="parent",
        conditions=[],
    )
    source_node = await journey_store.create_node(parent.id, action="parent action", tools=[])
    await journey_store.create_edge(
        parent.id, source=parent.root_id, target=source_node.id, condition=None
    )

    link = await journey_store.create_link(
        journey_id=parent.id,
        source_node_id=source_node.id,
        sub_journey_id=sub.id,
        condition="wants sub",
    )

    guidelines = await projection.project_journey_to_guidelines(parent.id)

    # Sub-journey node guidelines should have link_id in their ID
    sub_guidelines = [g for g in guidelines if g.content.action == "sub action"]
    assert len(sub_guidelines) >= 1

    for g in sub_guidelines:
        extracted_link_id = extract_link_id_from_journey_node_guideline_id(g.id)
        assert extracted_link_id == link.id, (
            f"Expected link_id={link.id} in guideline ID {g.id}, got {extracted_link_id}"
        )

        # Original node_id should be extractable (not namespaced)
        extracted_node_id = extract_node_id_from_journey_node_guideline_id(g.id)
        assert extracted_node_id == sub_node.id, (
            f"Expected node_id={sub_node.id} in guideline ID {g.id}, got {extracted_node_id}"
        )

    # Parent-only node guidelines should NOT have link_id
    parent_guidelines = [g for g in guidelines if g.content.action == "parent action"]
    assert len(parent_guidelines) >= 1
    for g in parent_guidelines:
        extracted_link_id = extract_link_id_from_journey_node_guideline_id(g.id)
        assert extracted_link_id is None, (
            f"Parent guideline {g.id} should not have link_id, got {extracted_link_id}"
        )


async def test_that_same_sub_journey_linked_twice_produces_no_collisions(
    container: Container,
) -> None:
    journey_store = container[JourneyStore]
    guideline_store = container[GuidelineStore]

    projection = JourneyGuidelineProjection(
        journey_store=journey_store,
        guideline_store=guideline_store,
    )

    # Create sub-journey: root -> sub_node -> END
    sub = await journey_store.create_journey(
        title="Shared Sub Journey",
        description="reusable sub",
        conditions=[],
    )
    sub_node = await journey_store.create_node(sub.id, action="shared action", tools=[])
    await journey_store.create_edge(sub.id, source=sub.root_id, target=sub_node.id, condition=None)

    # Create parent: root -> node_a --(link1)--> sub
    #                root -> node_b --(link2)--> sub (same sub-journey)
    parent = await journey_store.create_journey(
        title="Parent with double link",
        description="parent",
        conditions=[],
    )
    node_a = await journey_store.create_node(parent.id, action="path A", tools=[])
    node_b = await journey_store.create_node(parent.id, action="path B", tools=[])
    await journey_store.create_edge(
        parent.id, source=parent.root_id, target=node_a.id, condition="go A"
    )
    await journey_store.create_edge(
        parent.id, source=parent.root_id, target=node_b.id, condition="go B"
    )

    link_a = await journey_store.create_link(
        journey_id=parent.id,
        source_node_id=node_a.id,
        sub_journey_id=sub.id,
        condition="enter sub from A",
    )
    link_b = await journey_store.create_link(
        journey_id=parent.id,
        source_node_id=node_b.id,
        sub_journey_id=sub.id,
        condition="enter sub from B",
    )

    guidelines = await projection.project_journey_to_guidelines(parent.id)

    # Should have two separate guidelines for the shared sub_node (one per link)
    shared_guidelines = [g for g in guidelines if g.content.action == "shared action"]
    assert len(shared_guidelines) == 2, (
        f"Expected 2 guidelines for shared sub-node (one per link), got {len(shared_guidelines)}"
    )

    # They should have different link_ids
    link_ids = {extract_link_id_from_journey_node_guideline_id(g.id) for g in shared_guidelines}
    assert link_ids == {link_a.id, link_b.id}, (
        f"Expected link IDs {{{link_a.id}, {link_b.id}}}, got {link_ids}"
    )

    # All guideline IDs should be unique (no collisions)
    all_ids = [g.id for g in guidelines]
    assert len(all_ids) == len(set(all_ids)), "Guideline IDs are not unique — collision detected"

    # Follow-up chain integrity
    all_id_set = set(all_ids)
    for g in guidelines:
        followups = cast(dict[str, JSONSerializable], g.metadata.get("journey_node", {})).get(
            "follow_ups", []
        )
        for f_id in cast(list[str], followups):
            assert f_id in all_id_set, (
                f"Follow-up ID {f_id} listed in {g.id} but no guideline was created for it"
            )


async def test_that_journey_metadata_can_be_set_and_read(container: Container) -> None:
    journey_store = container[JourneyStore]

    journey = await journey_store.create_journey(
        title="Metadata Test Journey",
        description="test",
        conditions=[],
    )

    # Journey should start with empty metadata
    assert journey.metadata == {}

    # Update metadata
    updated = await journey_store.update_journey(
        journey_id=journey.id,
        params={"metadata": {"follow_ups": ["node_1", "node_2"], "active_state": "node_1"}},
    )

    assert updated.metadata["follow_ups"] == ["node_1", "node_2"]
    assert updated.metadata["active_state"] == "node_1"

    # Read back
    read_journey = await journey_store.read_journey(journey.id)
    assert read_journey.metadata == updated.metadata


async def test_that_projection_collapses_pass_through_forks_for_concatenated_journeys(
    container: Container,
) -> None:
    """Three sub-journeys chained: identity -> credit -> approval.
    Pass-through merge_forks should be collapsed so the resulting guidelines
    form a flat chain without fork nodes in between."""
    journey_store = container[JourneyStore]
    guideline_store = container[GuidelineStore]

    projection = JourneyGuidelineProjection(
        journey_store=journey_store,
        guideline_store=guideline_store,
    )

    # Sub-journey 1: identity verification
    sub1 = await journey_store.create_journey(
        title="Identity Verification", description="sub1", conditions=[]
    )
    id_node = await journey_store.create_node(sub1.id, action="ask for ID", tools=[])
    await journey_store.create_edge(sub1.id, source=sub1.root_id, target=id_node.id, condition=None)

    # Sub-journey 2: credit check
    sub2 = await journey_store.create_journey(
        title="Credit Check", description="sub2", conditions=[]
    )
    credit_node = await journey_store.create_node(sub2.id, action="ask for SSN", tools=[])
    await journey_store.create_edge(
        sub2.id, source=sub2.root_id, target=credit_node.id, condition=None
    )

    # Sub-journey 3: approval
    sub3 = await journey_store.create_journey(
        title="Loan Approval", description="sub3", conditions=[]
    )
    approval_node = await journey_store.create_node(sub3.id, action="approve loan", tools=[])
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

    actions = _get_actions(guidelines)
    assert "ask for ID" in actions, f"Missing identity node. Actions: {actions}"
    assert "ask for SSN" in actions, f"Missing credit node. Actions: {actions}"
    assert "approve loan" in actions, f"Missing approval node. Actions: {actions}"

    # No fork nodes should remain — all pass-through/terminal forks collapsed
    for g in guidelines:
        jn = cast(dict[str, JSONSerializable], g.metadata.get("journey_node", {}))
        assert jn.get("kind") != "fork", (
            f"Fork node should have been collapsed: {g.id} action={g.content.action}"
        )

    # Follow-up chain integrity
    all_ids = {g.id for g in guidelines}
    for g in guidelines:
        followups = cast(dict[str, JSONSerializable], g.metadata.get("journey_node", {})).get(
            "follow_ups", []
        )
        for f_id in cast(list[str], followups):
            assert f_id in all_ids, (
                f"Follow-up ID {f_id} listed in {g.id} but no guideline exists for it"
            )

    # The chain should be: root -> ask_for_ID -> ask_for_SSN -> approve_loan
    id_g = [g for g in guidelines if g.content.action == "ask for ID"]
    assert len(id_g) == 1
    id_followups = cast(
        list[str],
        cast(dict[str, JSONSerializable], id_g[0].metadata["journey_node"])["follow_ups"],
    )
    assert len(id_followups) > 0, "ask_for_ID should have follow-ups to credit check"

    # The follow-up should point to ask_for_SSN (not a fork)
    ssn_g = [g for g in guidelines if g.content.action == "ask for SSN"]
    assert len(ssn_g) == 1
    assert ssn_g[0].id in id_followups, (
        f"ask_for_ID follow-ups {id_followups} should include ask_for_SSN {ssn_g[0].id}"
    )
