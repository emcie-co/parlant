from dataclasses import dataclass
from itertools import chain
from typing import Mapping, Sequence, cast

from parlant.core.agents import AgentId, AgentStore, CompositionMode
from parlant.core.common import Criticality, ItemNotFoundError, JSONSerializable, UniqueId
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociation,
    GuidelineToolAssociationStore,
)
from parlant.core.journeys import JourneyId, JourneyStore
from parlant.core.loggers import Logger
from parlant.core.guidelines import GuidelineId, GuidelineStore, Guideline, GuidelineUpdateParams
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipId,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tags import Tag, TagId, TagStore
from parlant.core.tools import Tool, ToolId


@dataclass(frozen=True)
class GuidelineMetadataUpdateParams:
    set: Mapping[str, JSONSerializable] | None = None
    unset: Sequence[str] | None = None


@dataclass(frozen=True)
class GuidelineTagsUpdateParams:
    add: Sequence[TagId] | None = None
    remove: Sequence[TagId] | None = None


@dataclass(frozen=True)
class GuidelineToolAssociationUpdateParams:
    add: Sequence[ToolId] | None = None
    remove: Sequence[ToolId] | None = None


@dataclass
class GuidelineRelationship:
    id: RelationshipId
    source: Guideline | Tag | Tool
    source_type: RelationshipEntityKind
    target: Guideline | Tag | Tool
    target_type: RelationshipEntityKind
    kind: RelationshipKind


class GuidelineModule:
    def __init__(
        self,
        logger: Logger,
        guideline_store: GuidelineStore,
        tag_store: TagStore,
        agent_store: AgentStore,
        journey_store: JourneyStore,
        relationship_store: RelationshipStore,
        guideline_tool_association_store: GuidelineToolAssociationStore,
        service_registry: ServiceRegistry,
    ):
        self._logger = logger
        self._guideline_store = guideline_store
        self._tag_store = tag_store
        self._agent_store = agent_store
        self._journey_store = journey_store
        self._relationship_store = relationship_store
        self._guideline_tool_association_store = guideline_tool_association_store
        self._service_registry = service_registry

    async def _ensure_tag(self, tag_id: TagId) -> None:
        if agent_id := Tag.extract_agent_id(tag_id):
            _ = await self._agent_store.read_agent(agent_id=AgentId(agent_id))
        elif journey_id := Tag.extract_journey_id(tag_id):
            _ = await self._journey_store.read_journey(journey_id=JourneyId(journey_id))
        else:
            _ = await self._tag_store.read_tag(tag_id=tag_id)

    async def create(
        self,
        condition: str,
        action: str | None,
        description: str | None,
        criticality: Criticality | None,
        metadata: Mapping[str, JSONSerializable] | None,
        enabled: bool | None,
        tags: Sequence[TagId] | None,
        id: GuidelineId | None = None,
        composition_mode: CompositionMode | None = None,
    ) -> Guideline:
        if tags:
            for tag_id in tags:
                await self._ensure_tag(tag_id)

            tags = list(set(tags))

        guideline = await self._guideline_store.create_guideline(
            condition=condition,
            action=action,
            description=description,
            criticality=criticality,
            metadata=metadata or {},
            enabled=enabled or True,
            tags=tags,
            id=id,
            composition_mode=composition_mode,
        )

        return guideline

    async def read(self, guideline_id: GuidelineId) -> Guideline:
        guideline = await self._guideline_store.read_guideline(guideline_id=guideline_id)
        return guideline

    async def find(
        self,
        tag_id: TagId | None,
    ) -> Sequence[Guideline]:
        if tag_id:
            guidelines = await self._guideline_store.list_guidelines(
                tags=[tag_id],
            )
        else:
            guidelines = await self._guideline_store.list_guidelines()

        return guidelines

    async def update(
        self,
        guideline_id: GuidelineId,
        condition: str | None,
        action: str | None,
        description: str | None,
        criticality: Criticality | None,
        tool_associations: GuidelineToolAssociationUpdateParams | None,
        enabled: bool | None,
        tags: GuidelineTagsUpdateParams | None,
        metadata: GuidelineMetadataUpdateParams | None,
        composition_mode: CompositionMode | None = None,
    ) -> Guideline:
        _ = await self._guideline_store.read_guideline(guideline_id=guideline_id)

        if (
            condition
            or action
            or description is not None
            or criticality is not None
            or enabled is not None
            or composition_mode is not None
        ):
            update_params: GuidelineUpdateParams = {}
            if condition:
                update_params["condition"] = condition
            if action:
                update_params["action"] = action
            if description is not None:
                update_params["description"] = description
            if criticality is not None:
                update_params["criticality"] = criticality
            if enabled is not None:
                update_params["enabled"] = enabled
            if composition_mode is not None:
                update_params["composition_mode"] = composition_mode

            await self._guideline_store.update_guideline(
                guideline_id=guideline_id,
                params=GuidelineUpdateParams(**update_params),
            )

        if metadata:
            if metadata.set:
                for key, value in metadata.set.items():
                    await self._guideline_store.set_metadata(
                        guideline_id=guideline_id,
                        key=key,
                        value=value,
                    )

            if metadata.unset:
                for key in metadata.unset:
                    await self._guideline_store.unset_metadata(
                        guideline_id=guideline_id,
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

                await self._guideline_tool_association_store.create_association(
                    guideline_id=guideline_id,
                    tool_id=ToolId(service_name=service_name, tool_name=tool_name),
                )

        if tool_associations and tool_associations.remove:
            associations = await self._guideline_tool_association_store.list_associations()

            for tool_id in tool_associations.remove:
                if association := next(
                    (
                        assoc
                        for assoc in associations
                        if assoc.tool_id.service_name == tool_id.service_name
                        and assoc.tool_id.tool_name == tool_id.tool_name
                        and assoc.guideline_id == guideline_id
                    ),
                    None,
                ):
                    await self._guideline_tool_association_store.delete_association(association.id)
                else:
                    raise ItemNotFoundError(
                        UniqueId(tool_name),
                        f"Tool association not found for service '{tool_id.service_name}' and tool '{tool_id.tool_name}'",
                    )

        if tags:
            if tags.add:
                for tag_id in tags.add:
                    await self._ensure_tag(tag_id)

                    await self._guideline_store.upsert_tag(
                        guideline_id=guideline_id,
                        tag_id=tag_id,
                    )

            if tags.remove:
                for tag_id in tags.remove:
                    await self._guideline_store.remove_tag(
                        guideline_id=guideline_id,
                        tag_id=tag_id,
                    )

        guideline = await self._guideline_store.read_guideline(guideline_id=guideline_id)

        return guideline

    async def delete(self, guideline_id: GuidelineId) -> None:
        guideline = await self._guideline_store.read_guideline(guideline_id=guideline_id)

        for r, _ in await self.find_relationships(
            guideline_id=guideline_id,
            include_indirect=False,
        ):
            related_guideline = (
                r.target if cast(Guideline | Tag, r.source).id == guideline_id else r.source
            )
            if (
                isinstance(related_guideline, Guideline)
                and related_guideline.tags
                and not any(t in related_guideline.tags for t in guideline.tags)
            ):
                await self._relationship_store.delete_relationship(r.id)

        for associastion in await self._guideline_tool_association_store.list_associations():
            if associastion.guideline_id == guideline_id:
                await self._guideline_tool_association_store.delete_association(associastion.id)

        journeys = await self._journey_store.list_journeys()
        for journey in journeys:
            for condition in journey.conditions:
                if condition == guideline_id:
                    await self._journey_store.remove_condition(
                        journey_id=journey.id,
                        condition=condition,
                    )

        await self._guideline_store.delete_guideline(guideline_id=guideline_id)

    async def _get_guideline_relationships_by_kind(
        self,
        entity_id: GuidelineId | TagId,
        kind: RelationshipKind,
        include_indirect: bool = True,
    ) -> Sequence[tuple[GuidelineRelationship, bool]]:
        async def _get_entity(
            entity_id: GuidelineId | TagId,
            entity_type: RelationshipEntityKind,
        ) -> Guideline | Tag:
            if entity_type == RelationshipEntityKind.GUIDELINE:
                return await self._guideline_store.read_guideline(
                    guideline_id=cast(GuidelineId, entity_id)
                )
            elif entity_type == RelationshipEntityKind.TAG:
                return await self._tag_store.read_tag(tag_id=cast(TagId, entity_id))
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
            assert r.source.kind in (RelationshipEntityKind.GUIDELINE, RelationshipEntityKind.TAG)
            assert r.target.kind in (RelationshipEntityKind.GUIDELINE, RelationshipEntityKind.TAG)
            assert type(r.kind) is RelationshipKind

            relationships.append(
                GuidelineRelationship(
                    id=r.id,
                    source=await _get_entity(cast(GuidelineId | TagId, r.source.id), r.source.kind),
                    source_type=r.source.kind,
                    target=await _get_entity(cast(GuidelineId | TagId, r.target.id), r.target.kind),
                    target_type=r.target.kind,
                    kind=r.kind,
                )
            )

        return [
            (
                r,
                entity_id
                not in [cast(Guideline | Tag, r.source).id, cast(Guideline | Tag, r.target).id],
            )
            for r in relationships
        ]

    async def find_relationships(
        self,
        guideline_id: GuidelineId,
        include_indirect: bool = True,
    ) -> Sequence[tuple[GuidelineRelationship, bool]]:
        return list(
            chain.from_iterable(
                [
                    await self._get_guideline_relationships_by_kind(
                        entity_id=guideline_id,
                        kind=kind,
                        include_indirect=include_indirect,
                    )
                    for kind in list(RelationshipKind)
                ]
            )
        )

    async def find_tool_associations(
        self,
        guideline_id: GuidelineId,
    ) -> Sequence[GuidelineToolAssociation]:
        associations = await self._guideline_tool_association_store.list_associations()
        return [a for a in associations if a.guideline_id == guideline_id]
