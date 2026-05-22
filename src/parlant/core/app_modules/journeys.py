from dataclasses import dataclass
from typing import Sequence, Set

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
class JourneyTriggerUpdateParams:
    add: Sequence[GuidelineId] | None
    remove: Sequence[GuidelineId] | None


@dataclass(frozen=True)
class JourneyTagUpdateParams:
    add: Sequence[TagId] | None = None
    remove: Sequence[TagId] | None = None


@dataclass(frozen=True)
class JourneyLabelsUpdateParams:
    upsert: Set[str] | None = None
    remove: Set[str] | None = None


@dataclass(frozen=True)
class JourneyNodeLabelsUpdateParams:
    upsert: Set[str] | None = None
    remove: Set[str] | None = None


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
        triggers: Sequence[str],
        tags: Sequence[TagId] | None,
        id: JourneyId | None = None,
        composition_mode: CompositionMode | None = None,
        labels: Set[str] | None = None,
        priority: int = 0,
    ) -> tuple[Journey, Sequence[Guideline]]:
        guidelines = [
            await self._guideline_store.create_guideline(
                condition=trigger,
                action=None,
                tags=[],
            )
            for trigger in triggers
        ]

        journey = await self._journey_store.create_journey(
            title=title,
            description=description,
            triggers=[g.id for g in guidelines],
            tags=tags,
            id=id,
            composition_mode=composition_mode,
            labels=labels,
            priority=priority,
        )

        for guideline in guidelines:
            await self._guideline_store.upsert_tag(
                guideline_id=guideline.id,
                tag_id=Tag.for_journey_id(journey.id).id,
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
        triggers: JourneyTriggerUpdateParams | None,
        tags: JourneyTagUpdateParams | None,
        composition_mode: CompositionMode | None = None,
        labels: JourneyLabelsUpdateParams | None = None,
        priority: int | None = None,
    ) -> Journey:
        journey = await self._journey_store.read_journey(journey_id=journey_id)

        update_params: JourneyUpdateParams = {}
        if title:
            update_params["title"] = title
        if description:
            update_params["description"] = description
        if composition_mode is not None:
            update_params["composition_mode"] = composition_mode
        if priority is not None:
            update_params["priority"] = priority

        if update_params:
            journey = await self._journey_store.update_journey(
                journey_id=journey_id,
                params=update_params,
            )

        if triggers:
            if triggers.add:
                for trigger in triggers.add:
                    await self._journey_store.add_trigger(
                        journey_id=journey_id,
                        trigger=trigger,
                    )

                    guideline = await self._guideline_store.read_guideline(guideline_id=trigger)

                    await self._guideline_store.upsert_tag(
                        guideline_id=trigger,
                        tag_id=Tag.for_journey_id(journey_id).id,
                    )

            if triggers.remove:
                for trigger in triggers.remove:
                    await self._journey_store.remove_trigger(
                        journey_id=journey_id,
                        trigger=trigger,
                    )

                    guideline = await self._guideline_store.read_guideline(guideline_id=trigger)

                    if guideline.tags == [Tag.for_journey_id(journey_id).id]:
                        await self._guideline_store.delete_guideline(guideline_id=trigger)
                    else:
                        await self._guideline_store.remove_tag(
                            guideline_id=trigger,
                            tag_id=Tag.for_journey_id(journey_id).id,
                        )

        if tags:
            if tags.add:
                for tag in tags.add:
                    await self._journey_store.upsert_tag(journey_id=journey_id, tag_id=tag)

            if tags.remove:
                for tag in tags.remove:
                    await self._journey_store.remove_tag(journey_id=journey_id, tag_id=tag)

        if labels:
            if labels.upsert:
                await self._journey_store.upsert_journey_labels(
                    journey_id=journey_id,
                    labels=labels.upsert,
                )

            if labels.remove:
                await self._journey_store.remove_journey_labels(
                    journey_id=journey_id,
                    labels=labels.remove,
                )

        journey = await self._journey_store.read_journey(journey_id=journey_id)

        return journey

    async def delete(self, journey_id: JourneyId) -> None:
        journey = await self._journey_store.read_journey(journey_id=journey_id)

        await self._journey_store.delete_journey(journey_id=journey_id)

        for trigger in journey.triggers:
            if not await self._journey_store.list_journeys(trigger=trigger):
                await self._guideline_store.delete_guideline(guideline_id=trigger)
            else:
                guideline = await self._guideline_store.read_guideline(guideline_id=trigger)

                if guideline.tags == [Tag.for_journey_id(journey_id).id]:
                    await self._guideline_store.delete_guideline(guideline_id=trigger)
                else:
                    await self._guideline_store.remove_tag(
                        guideline_id=trigger,
                        tag_id=Tag.for_journey_id(journey_id).id,
                    )
