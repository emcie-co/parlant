from dataclasses import dataclass
from typing import Sequence

from parlant.core.agents import CompositionMode
from parlant.core.guidelines import Guideline, GuidelineId, GuidelineStore
from parlant.core.loggers import Logger
from parlant.core.journeys import (
    JourneyEdge,
    JourneyId,
    JourneyNode,
    JourneyStore,
    Journey,
    JourneyUpdateParams,
)
from parlant.core.tags import Tag, TagId


@dataclass(frozen=True)
class JourneyGraph:
    journey: Journey
    nodes: Sequence[JourneyNode]
    edges: Sequence[JourneyEdge]


@dataclass(frozen=True)
class JourneyConditionUpdateParams:
    add: Sequence[GuidelineId] | None
    remove: Sequence[GuidelineId] | None


@dataclass(frozen=True)
class JourneyTagUpdateParams:
    add: Sequence[TagId] | None = None
    remove: Sequence[TagId] | None = None


class JourneyModule:
    def __init__(
        self,
        logger: Logger,
        journey_store: JourneyStore,
        guideline_store: GuidelineStore,
    ):
        self._logger = logger
        self._journey_store = journey_store
        self._guideline_store = guideline_store

    async def create(
        self,
        title: str,
        description: str,
        conditions: Sequence[str],
        tags: Sequence[TagId] | None,
        id: JourneyId | None = None,
        composition_mode: CompositionMode | None = None,
    ) -> tuple[Journey, Sequence[Guideline]]:
        guidelines = [
            await self._guideline_store.create_guideline(
                condition=condition,
                action=None,
                tags=[],
            )
            for condition in conditions
        ]

        journey = await self._journey_store.create_journey(
            title=title,
            description=description,
            conditions=[g.id for g in guidelines],
            tags=tags,
            id=id,
            composition_mode=composition_mode,
        )

        for guideline in guidelines:
            await self._guideline_store.upsert_tag(
                guideline_id=guideline.id,
                tag_id=Tag.for_journey_id(journey.id),
            )

        return journey, guidelines

    async def read(self, journey_id: JourneyId) -> JourneyGraph:
        journey = await self._journey_store.read_journey(journey_id=journey_id)
        nodes = await self._journey_store.list_nodes(journey_id=journey.id)
        edges = await self._journey_store.list_edges(journey_id=journey.id)

        return JourneyGraph(journey=journey, nodes=nodes, edges=edges)

    async def find(self, tag_id: TagId | None) -> Sequence[Journey]:
        if tag_id:
            journeys = await self._journey_store.list_journeys(
                tags=[tag_id],
            )
        else:
            journeys = await self._journey_store.list_journeys()

        return journeys

    async def update(
        self,
        journey_id: JourneyId,
        title: str | None,
        description: str | None,
        conditions: JourneyConditionUpdateParams | None,
        tags: JourneyTagUpdateParams | None,
        composition_mode: CompositionMode | None = None,
    ) -> Journey:
        journey = await self._journey_store.read_journey(journey_id=journey_id)

        update_params: JourneyUpdateParams = {}
        if title:
            update_params["title"] = title
        if description:
            update_params["description"] = description
        if composition_mode is not None:
            update_params["composition_mode"] = composition_mode

        if update_params:
            journey = await self._journey_store.update_journey(
                journey_id=journey_id,
                params=update_params,
            )

        if conditions:
            if conditions.add:
                for condition in conditions.add:
                    await self._journey_store.add_condition(
                        journey_id=journey_id,
                        condition=condition,
                    )

                    guideline = await self._guideline_store.read_guideline(guideline_id=condition)

                    await self._guideline_store.upsert_tag(
                        guideline_id=condition,
                        tag_id=Tag.for_journey_id(journey_id),
                    )

            if conditions.remove:
                for condition in conditions.remove:
                    await self._journey_store.remove_condition(
                        journey_id=journey_id,
                        condition=condition,
                    )

                    guideline = await self._guideline_store.read_guideline(guideline_id=condition)

                    if guideline.tags == [Tag.for_journey_id(journey_id)]:
                        await self._guideline_store.delete_guideline(guideline_id=condition)
                    else:
                        await self._guideline_store.remove_tag(
                            guideline_id=condition,
                            tag_id=Tag.for_journey_id(journey_id),
                        )

        if tags:
            if tags.add:
                for tag in tags.add:
                    await self._journey_store.upsert_tag(journey_id=journey_id, tag_id=tag)

            if tags.remove:
                for tag in tags.remove:
                    await self._journey_store.remove_tag(journey_id=journey_id, tag_id=tag)

        journey = await self._journey_store.read_journey(journey_id=journey_id)

        return journey

    async def delete(self, journey_id: JourneyId) -> None:
        journey = await self._journey_store.read_journey(journey_id=journey_id)

        await self._journey_store.delete_journey(journey_id=journey_id)

        for condition in journey.conditions:
            if not await self._journey_store.list_journeys(condition=condition):
                await self._guideline_store.delete_guideline(guideline_id=condition)
            else:
                guideline = await self._guideline_store.read_guideline(guideline_id=condition)

                if guideline.tags == [Tag.for_journey_id(journey_id)]:
                    await self._guideline_store.delete_guideline(guideline_id=condition)
                else:
                    await self._guideline_store.remove_tag(
                        guideline_id=condition,
                        tag_id=Tag.for_journey_id(journey_id),
                    )
