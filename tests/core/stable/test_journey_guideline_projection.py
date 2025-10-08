from typing import cast
from lagom import Container

from parlant.core.common import JSONSerializable
from parlant.core.guidelines import GuidelineStore
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
