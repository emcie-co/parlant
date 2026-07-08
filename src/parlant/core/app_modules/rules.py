from dataclasses import dataclass
from itertools import chain
from typing import Mapping, Optional, Sequence, Set, cast

from parlant.core.agents import AgentId, AgentStore, CompositionMode, Effort
from parlant.core.common import Weight, ItemNotFoundError, JSONSerializable, UniqueId
from parlant.core.rule_tool_associations import (
    RuleToolAssociation,
    RuleToolAssociationStore,
)
from parlant.core.journeys import JourneyId, JourneyStore
from parlant.core.loggers import Logger
from parlant.core.rules import RuleId, RuleStore, Rule, RuleUpdateParams
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipId,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.groups import Group, GroupIds, GroupId, GroupStore
from parlant.core.tools import Tool, ToolId
from parlant.core.store_provider import StoreProviderHints, StoreProvider
from parlant.core.app_modules.request_context import RequestContext


@dataclass(frozen=True)
class RuleMetadataUpdateParams:
    set: Mapping[str, JSONSerializable] | None = None
    unset: Sequence[str] | None = None


@dataclass(frozen=True)
class RuleGroupsUpdateParams:
    add: Sequence[GroupId] | None = None
    remove: Sequence[GroupId] | None = None


@dataclass(frozen=True)
class RuleToolAssociationUpdateParams:
    add: Sequence[ToolId] | None = None
    remove: Sequence[ToolId] | None = None


@dataclass(frozen=True)
class RuleLabelsUpdateParams:
    upsert: Set[str] | None = None
    remove: Set[str] | None = None


@dataclass
class RuleRelationship:
    id: RelationshipId
    source: Rule | Group | Tool
    source_type: RelationshipEntityKind
    target: Rule | Group | Tool
    target_type: RelationshipEntityKind
    kind: RelationshipKind
    group_id: Optional[str] = None


class RuleModule:
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
    def _rule_store(self) -> RuleStore:
        return self._store_provider.get_store(
            RuleStore,
            StoreProviderHints(
                call_site="app",
                origin=self._request_context.get_origin(),
            ),
        )

    @property
    def _group_store(self) -> GroupStore:
        return self._store_provider.get_store(
            GroupStore,
            StoreProviderHints(
                call_site="app",
                origin=self._request_context.get_origin(),
            ),
        )

    @property
    def _agent_store(self) -> AgentStore:
        return self._store_provider.get_store(
            AgentStore,
            StoreProviderHints(
                call_site="app",
                origin=self._request_context.get_origin(),
            ),
        )

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
    def _relationship_store(self) -> RelationshipStore:
        return self._store_provider.get_store(
            RelationshipStore,
            StoreProviderHints(
                call_site="app",
                origin=self._request_context.get_origin(),
            ),
        )

    @property
    def _rule_tool_association_store(self) -> RuleToolAssociationStore:
        return self._store_provider.get_store(
            RuleToolAssociationStore,
            StoreProviderHints(
                call_site="app",
                origin=self._request_context.get_origin(),
            ),
        )

    @property
    def _service_registry(self) -> ServiceRegistry:
        return self._store_provider.get_store(
            ServiceRegistry,
            StoreProviderHints(
                call_site="app",
                origin=self._request_context.get_origin(),
            ),
        )

    async def _ensure_tag(self, group_id: GroupId) -> None:
        if agent_id := GroupIds.extract_agent_id(group_id):
            _ = await self._agent_store.read_agent(agent_id=AgentId(agent_id))
        elif journey_id := GroupIds.extract_journey_id(group_id):
            _ = await self._journey_store.read_journey(journey_id=JourneyId(journey_id))
        else:
            _ = await self._group_store.read_group(group_id=group_id)

    async def create(
        self,
        condition: str,
        action: str | None,
        description: str | None,
        title: str | None,
        criticality: Weight | None,
        metadata: Mapping[str, JSONSerializable] | None,
        enabled: bool | None,
        groups: Sequence[GroupId] | None,
        id: RuleId | None = None,
        composition_mode: CompositionMode | None = None,
        effort: Effort | None = None,
        track: bool = True,
        labels: Set[str] | None = None,
        priority: int = 0,
        signals: Sequence[str] = [],
        anti_signals: Sequence[str] = [],
    ) -> Rule:
        if groups:
            for group_id in groups:
                await self._ensure_tag(group_id)

            groups = list(set(groups))

        rule = await self._rule_store.create_rule(
            condition=condition,
            action=action,
            description=description,
            title=title,
            weight=criticality,
            metadata=metadata or {},
            enabled=enabled if enabled is not None else True,
            groups=groups,
            id=id,
            composition_mode=composition_mode,
            effort_lift=effort,
            track=track,
            labels=labels,
            priority=priority,
            signals=signals,
            anti_signals=anti_signals,
        )

        return rule

    async def read(self, rule_id: RuleId) -> Rule:
        rule = await self._rule_store.read_rule(rule_id=rule_id)
        return rule

    async def find(
        self,
        group_id: GroupId | None,
    ) -> Sequence[Rule]:
        if group_id:
            rules = await self._rule_store.list_rules(
                groups=[group_id],
            )
        else:
            rules = await self._rule_store.list_rules()

        return rules

    async def update(
        self,
        rule_id: RuleId,
        condition: str | None,
        action: str | None,
        description: str | None,
        title: str | None,
        criticality: Weight | None,
        tool_associations: RuleToolAssociationUpdateParams | None,
        enabled: bool | None,
        groups: RuleGroupsUpdateParams | None,
        metadata: RuleMetadataUpdateParams | None,
        composition_mode: CompositionMode | None = None,
        effort: Effort | None = None,
        labels: RuleLabelsUpdateParams | None = None,
        priority: int | None = None,
        signals: Sequence[str] | None = None,
        anti_signals: Sequence[str] | None = None,
    ) -> Rule:
        _ = await self._rule_store.read_rule(rule_id=rule_id)

        if (
            condition
            or action
            or description is not None
            or title is not None
            or criticality is not None
            or enabled is not None
            or composition_mode is not None
            or effort is not None
            or priority is not None
            or signals is not None
            or anti_signals is not None
        ):
            update_params: RuleUpdateParams = {}
            if condition:
                update_params["condition"] = condition
            if action:
                update_params["action"] = action
            if description is not None:
                update_params["description"] = description
            if title is not None:
                update_params["title"] = title
            if criticality is not None:
                update_params["criticality"] = criticality
            if enabled is not None:
                update_params["enabled"] = enabled
            if composition_mode is not None:
                update_params["composition_mode"] = composition_mode
            if effort is not None:
                update_params["effort"] = effort
            if priority is not None:
                update_params["priority"] = priority
            if signals is not None:
                update_params["signals"] = signals
            if anti_signals is not None:
                update_params["anti_signals"] = anti_signals

            await self._rule_store.update_rule(
                rule_id=rule_id,
                params=RuleUpdateParams(**update_params),
            )

        if metadata:
            if metadata.set:
                for key, value in metadata.set.items():
                    await self._rule_store.set_metadata(
                        rule_id=rule_id,
                        key=key,
                        value=value,
                    )

            if metadata.unset:
                for key in metadata.unset:
                    await self._rule_store.unset_metadata(
                        rule_id=rule_id,
                        key=key,
                    )

        if tool_associations and tool_associations.add:
            for tool_id in tool_associations.add:
                service_name = tool_id.service_name
                tool_name = tool_id.tool_name

                try:
                    service = await self._service_registry.read_tool_service(service_name)
                    _ = await service.read_tool(tool_name)
                except ItemNotFoundError:
                    raise ItemNotFoundError(
                        UniqueId(tool_name),
                        f"Tool not found (service='{service_name}', tool='{tool_name}')",
                    )

                await self._rule_tool_association_store.create_association(
                    rule_id=rule_id,
                    tool_id=ToolId(service_name=service_name, tool_name=tool_name),
                )

        if tool_associations and tool_associations.remove:
            associations = await self._rule_tool_association_store.list_associations()

            for tool_id in tool_associations.remove:
                if association := next(
                    (
                        assoc
                        for assoc in associations
                        if assoc.tool_id.service_name == tool_id.service_name
                        and assoc.tool_id.tool_name == tool_id.tool_name
                        and assoc.rule_id == rule_id
                    ),
                    None,
                ):
                    await self._rule_tool_association_store.delete_association(association.id)
                else:
                    raise ItemNotFoundError(
                        UniqueId(tool_name),
                        f"Tool association not found for service '{tool_id.service_name}' and tool '{tool_id.tool_name}'",
                    )

        if groups:
            if groups.add:
                for group_id in groups.add:
                    await self._ensure_tag(group_id)

                    await self._rule_store.upsert_group(
                        rule_id=rule_id,
                        group_id=group_id,
                    )

            if groups.remove:
                for group_id in groups.remove:
                    await self._rule_store.remove_group(
                        rule_id=rule_id,
                        group_id=group_id,
                    )

        if labels:
            if labels.upsert:
                await self._rule_store.upsert_labels(
                    rule_id=rule_id,
                    labels=labels.upsert,
                )

            if labels.remove:
                await self._rule_store.remove_labels(
                    rule_id=rule_id,
                    labels=labels.remove,
                )

        rule = await self._rule_store.read_rule(rule_id=rule_id)

        return rule

    async def delete(self, rule_id: RuleId) -> None:
        rule = await self._rule_store.read_rule(rule_id=rule_id)

        for r, _ in await self.find_relationships(
            rule_id=rule_id,
            include_indirect=False,
        ):
            related_rule = r.target if cast(Rule | Group, r.source).id == rule_id else r.source
            if (
                isinstance(related_rule, Rule)
                and related_rule.groups
                and not any(t in related_rule.groups for t in rule.groups)
            ):
                await self._relationship_store.delete_relationship(r.id)

        for associastion in await self._rule_tool_association_store.list_associations():
            if associastion.rule_id == rule_id:
                await self._rule_tool_association_store.delete_association(associastion.id)

        journeys = await self._journey_store.list_journeys()
        for journey in journeys:
            for trigger in journey.triggers:
                if trigger == rule_id:
                    await self._journey_store.remove_trigger(
                        journey_id=journey.id,
                        trigger=trigger,
                    )

        await self._rule_store.delete_rule(rule_id=rule_id)

    async def _get_rule_relationships_by_kind(
        self,
        entity_id: RuleId | GroupId,
        kind: RelationshipKind,
        include_indirect: bool = True,
    ) -> Sequence[tuple[RuleRelationship, bool]]:
        async def _get_entity(
            entity_id: RuleId | GroupId,
            entity_type: RelationshipEntityKind,
        ) -> Rule | Group:
            if entity_type == RelationshipEntityKind.RULE:
                return await self._rule_store.read_rule(rule_id=cast(RuleId, entity_id))
            elif entity_type.is_group:
                return await self._group_store.read_group(group_id=cast(GroupId, entity_id))
            else:
                raise ValueError(f"Unsupported entity type: {entity_type}")

        relationships = []

        for r in chain(
            await self._relationship_store.list_relationships(
                kind=kind,
                indirect=include_indirect,
                source_id=entity_id,
            ),
            await self._relationship_store.list_relationships(
                kind=kind,
                indirect=include_indirect,
                target_id=entity_id,
            ),
        ):
            assert r.source.kind == RelationshipEntityKind.RULE or r.source.kind.is_group
            assert r.target.kind == RelationshipEntityKind.RULE or r.target.kind.is_group
            assert type(r.kind) is RelationshipKind

            relationships.append(
                RuleRelationship(
                    id=r.id,
                    source=await _get_entity(cast(RuleId | GroupId, r.source.id), r.source.kind),
                    source_type=r.source.kind,
                    target=await _get_entity(cast(RuleId | GroupId, r.target.id), r.target.kind),
                    target_type=r.target.kind,
                    kind=r.kind,
                    group_id=r.group_id,
                )
            )

        return [
            (
                r,
                entity_id not in [cast(Rule | Group, r.source).id, cast(Rule | Group, r.target).id],
            )
            for r in relationships
        ]

    async def find_relationships(
        self,
        rule_id: RuleId,
        include_indirect: bool = True,
    ) -> Sequence[tuple[RuleRelationship, bool]]:
        return list(
            chain.from_iterable(
                [
                    await self._get_rule_relationships_by_kind(
                        entity_id=rule_id,
                        kind=kind,
                        include_indirect=include_indirect,
                    )
                    for kind in list(RelationshipKind)
                ]
            )
        )

    async def find_tool_associations(
        self,
        rule_id: RuleId,
    ) -> Sequence[RuleToolAssociation]:
        associations = await self._rule_tool_association_store.list_associations()
        return [a for a in associations if a.rule_id == rule_id]
