from dataclasses import dataclass
from typing import Mapping, Sequence, Set

from parlant.core.agents import CompositionMode
from parlant.core.app_modules.request_context import RequestContext
from parlant.core.rules import Rule, RuleId, RuleStore
from parlant.core.loggers import Logger
from parlant.core.common import JSONSerializable
from parlant.core.journeys import (
    JourneyEdge,
    JourneyId,
    JourneyNode,
    JourneyStore,
    Journey,
    JourneyUpdateParams,
)
from parlant.core.groups import GroupIds, GroupId
from parlant.core.store_provider import StoreProviderHints, StoreProvider


@dataclass(frozen=True)
class JourneyGraph:
    journey: Journey
    nodes: Sequence[JourneyNode]
    edges: Sequence[JourneyEdge]


@dataclass(frozen=True)
class JourneyTriggerUpdateParams:
    add: Sequence[RuleId] | None
    remove: Sequence[RuleId] | None


@dataclass(frozen=True)
class JourneyGroupUpdateParams:
    add: Sequence[GroupId] | None = None
    remove: Sequence[GroupId] | None = None


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
        request_context: RequestContext,
        logger: Logger,
        store_provider: StoreProvider,
    ):
        self._request_context = request_context
        self._logger = logger
        self._store_provider = store_provider

    @property
    def _journey_store(self) -> JourneyStore:
        return self._store_provider.get_store(
            JourneyStore,
            StoreProviderHints(
                call_site="app",
                origin=self._request_context.get_origin(),
            ),
        )

    @property
    def _rule_store(self) -> RuleStore:
        return self._store_provider.get_store(
            RuleStore,
            StoreProviderHints(
                call_site="app",
                origin=self._request_context.get_origin(),
            ),
        )

    async def create(
        self,
        title: str,
        description: str,
        triggers: Sequence[str],
        groups: Sequence[GroupId] | None,
        id: JourneyId | None = None,
        composition_mode: CompositionMode | None = None,
        labels: Set[str] | None = None,
        priority: int = 0,
    ) -> tuple[Journey, Sequence[Rule]]:
        rules = [
            await self._rule_store.create_rule(
                condition=trigger,
                action=None,
                groups=[],
            )
            for trigger in triggers
        ]

        journey = await self._journey_store.create_journey(
            title=title,
            description=description,
            triggers=[g.id for g in rules],
            groups=groups,
            id=id,
            composition_mode=composition_mode,
            labels=labels,
            priority=priority,
        )

        for rule in rules:
            await self._rule_store.upsert_group(
                rule_id=rule.id,
                group_id=GroupIds.for_journey_id(journey.id),
            )

        return journey, rules

    async def read(self, journey_id: JourneyId) -> JourneyGraph:
        journey = await self._journey_store.read_journey(journey_id=journey_id)
        nodes = await self._journey_store.list_nodes(journey_id=journey.id)
        edges = await self._journey_store.list_edges(journey_id=journey.id)

        return JourneyGraph(journey=journey, nodes=nodes, edges=edges)

    async def find(self, group_id: GroupId | None) -> Sequence[Journey]:
        if group_id:
            journeys = await self._journey_store.list_journeys(
                groups=[group_id],
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
        groups: JourneyGroupUpdateParams | None,
        composition_mode: CompositionMode | None = None,
        labels: JourneyLabelsUpdateParams | None = None,
        priority: int | None = None,
        metadata: Mapping[str, JSONSerializable] | None = None,
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
        if metadata is not None:
            update_params["metadata"] = metadata

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

                    rule = await self._rule_store.read_rule(rule_id=trigger)

                    await self._rule_store.upsert_group(
                        rule_id=trigger,
                        group_id=GroupIds.for_journey_id(journey_id),
                    )

            if triggers.remove:
                for trigger in triggers.remove:
                    await self._journey_store.remove_trigger(
                        journey_id=journey_id,
                        trigger=trigger,
                    )

                    rule = await self._rule_store.read_rule(rule_id=trigger)

                    if rule.groups == [GroupIds.for_journey_id(journey_id)]:
                        await self._rule_store.delete_rule(rule_id=trigger)
                    else:
                        await self._rule_store.remove_group(
                            rule_id=trigger,
                            group_id=GroupIds.for_journey_id(journey_id),
                        )

        if groups:
            if groups.add:
                for group in groups.add:
                    await self._journey_store.upsert_group(journey_id=journey_id, group_id=group)

            if groups.remove:
                for group in groups.remove:
                    await self._journey_store.remove_group(journey_id=journey_id, group_id=group)

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
                await self._rule_store.delete_rule(rule_id=trigger)
            else:
                rule = await self._rule_store.read_rule(rule_id=trigger)

                if rule.groups == [GroupIds.for_journey_id(journey_id)]:
                    await self._rule_store.delete_rule(rule_id=trigger)
                else:
                    await self._rule_store.remove_group(
                        rule_id=trigger,
                        group_id=GroupIds.for_journey_id(journey_id),
                    )
