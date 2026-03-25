from typing import Sequence, cast
from lagom import Container

from parlant.core.common import JSONSerializable
from parlant.core.guidelines import Guideline, GuidelineStore
from parlant.core.journey_guideline_projection import JourneyGuidelineProjection
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
        conditions=[],
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
