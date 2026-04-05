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
