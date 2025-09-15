from dataclasses import dataclass
from itertools import chain
from typing import Sequence, cast

from parlant.core.agents import AgentId, AgentStore
from parlant.core.guidelines import Guideline, GuidelineId, GuidelineStore
from parlant.core.journeys import JourneyId, JourneyNodeId, JourneyStore
from parlant.core.loggers import Logger
from parlant.core.relationships import (
    RelationshipEntity,
    RelationshipEntityId,
    RelationshipEntityKind,
    RelationshipId,
    RelationshipKind,
    RelationshipStore,
    Relationship,
)
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tags import Tag, TagId, TagStore
from parlant.core.tools import Tool, ToolId


@dataclass(frozen=True)
class RelationshipModel:
    id: RelationshipId
    source_guideline: Guideline | None
    source_tag: Tag | None
    target_guideline: Guideline | None
    target_tag: Tag | None
    source_tool: Tool | None
    target_tool: Tool | None
    kind: RelationshipKind


class RelationshipModule:
    def __init__(
        self,
        logger: Logger,
        relationship_store: RelationshipStore,
        tag_store: TagStore,
        guideline_store: GuidelineStore,
        service_registry: ServiceRegistry,
        agent_store: AgentStore,
        journey_store: JourneyStore,
    ):
        self._logger = logger
        self._relationship_store = relationship_store
        self._tag_store = tag_store
        self._guideline_store = guideline_store
        self._service_registry = service_registry
        self._agent_store = agent_store
        self._journey_store = journey_store

    async def _entity_id_to_tag(
        self,
        tag_id: RelationshipEntityId | TagId | GuidelineId | ToolId,
    ) -> Tag:
        tag_id = cast(TagId, tag_id)

        if agent_id := Tag.extract_agent_id(tag_id):
            agent = await self._agent_store.read_agent(agent_id=cast(AgentId, agent_id))
            return Tag(
                id=tag_id,
                name=agent.name,
                creation_utc=agent.creation_utc,
            )
        elif journey_id := Tag.extract_journey_id(tag_id):
            journey = await self._journey_store.read_journey(journey_id=cast(JourneyId, journey_id))
            return Tag(
                id=tag_id,
                name=journey.title,
                creation_utc=journey.creation_utc,
            )
        elif journey_node_id := Tag.extract_journey_node_id(tag_id):
            journey_node = await self._journey_store.read_node(
                node_id=cast(JourneyNodeId, journey_node_id)
            )
            return Tag(
                id=tag_id,
                name=str(journey_node.action),
                creation_utc=journey_node.creation_utc,
            )
        else:
            return await self._tag_store.read_tag(tag_id=tag_id)

    async def _relationship_to_model(
        self,
        relationship: Relationship,
    ) -> RelationshipModel:
        source_guideline = (
            await self._guideline_store.read_guideline(
                guideline_id=cast(GuidelineId, relationship.source.id)
            )
            if relationship.source.kind == RelationshipEntityKind.GUIDELINE
            else None
        )

        source_tag = (
            await self._entity_id_to_tag(
                relationship.source.id,
            )
            if relationship.source.kind == RelationshipEntityKind.TAG
            else None
        )

        target_guideline = (
            await self._guideline_store.read_guideline(
                guideline_id=cast(GuidelineId, relationship.target.id)
            )
            if relationship.target.kind == RelationshipEntityKind.GUIDELINE
            else None
        )

        target_tag = (
            await self._entity_id_to_tag(
                relationship.target.id,
            )
            if relationship.target.kind == RelationshipEntityKind.TAG
            else None
        )

        source_tool = (
            await (
                await self._service_registry.read_tool_service(
                    name=cast(ToolId, relationship.source.id).service_name
                )
            ).read_tool(name=cast(ToolId, relationship.source.id).tool_name)
            if relationship.source.kind == RelationshipEntityKind.TOOL
            else None
        )

        target_tool = (
            await (
                await self._service_registry.read_tool_service(
                    name=cast(ToolId, relationship.target.id).service_name
                )
            ).read_tool(name=cast(ToolId, relationship.target.id).tool_name)
            if relationship.target.kind == RelationshipEntityKind.TOOL
            else None
        )

        return RelationshipModel(
            id=relationship.id,
            source_guideline=source_guideline
            if relationship.source.kind == RelationshipEntityKind.GUIDELINE
            else None,
            source_tag=source_tag
            if relationship.source.kind == RelationshipEntityKind.TAG
            else None,
            target_guideline=target_guideline
            if relationship.target.kind == RelationshipEntityKind.GUIDELINE
            else None,
            target_tag=target_tag
            if relationship.target.kind == RelationshipEntityKind.TAG
            else None,
            source_tool=source_tool
            if relationship.source.kind == RelationshipEntityKind.TOOL
            else None,
            target_tool=target_tool
            if relationship.target.kind == RelationshipEntityKind.TOOL
            else None,
            kind=relationship.kind,
        )

    def _get_relationship_entity(
        self,
        guideline_id: GuidelineId | None,
        tag_id: TagId | None,
        tool_id: ToolId | None,
    ) -> RelationshipEntity:
        if guideline_id:
            return RelationshipEntity(id=guideline_id, kind=RelationshipEntityKind.GUIDELINE)
        elif tag_id:
            return RelationshipEntity(id=tag_id, kind=RelationshipEntityKind.TAG)
        elif tool_id:
            return RelationshipEntity(id=tool_id, kind=RelationshipEntityKind.TOOL)
        else:
            raise ValueError("No entity provided")

    async def create(
        self,
        source_guideline: GuidelineId | None,
        source_tag: TagId | None,
        source_tool: ToolId | None,
        target_guideline: GuidelineId | None,
        target_tag: TagId | None,
        target_tool: ToolId | None,
        kind: RelationshipKind,
    ) -> RelationshipModel:
        source: RelationshipEntity
        target: RelationshipEntity

        if source_guideline:
            await self._guideline_store.read_guideline(guideline_id=source_guideline)
            source = RelationshipEntity(id=source_guideline, kind=RelationshipEntityKind.GUIDELINE)
        elif source_tag:
            await self._entity_id_to_tag(
                source_tag,
            )
            source = RelationshipEntity(id=source_tag, kind=RelationshipEntityKind.TAG)
        elif source_tool:
            service = await self._service_registry.read_tool_service(name=source_tool.service_name)
            _ = await service.read_tool(name=source_tool.tool_name)
            source = RelationshipEntity(id=source_tool, kind=RelationshipEntityKind.TOOL)

        if target_guideline:
            await self._guideline_store.read_guideline(guideline_id=target_guideline)
            target = RelationshipEntity(id=target_guideline, kind=RelationshipEntityKind.GUIDELINE)
        elif target_tag:
            await self._entity_id_to_tag(
                target_tag,
            )
            target = RelationshipEntity(id=target_tag, kind=RelationshipEntityKind.TAG)
        elif target_tool:
            service = await self._service_registry.read_tool_service(name=target_tool.service_name)
            _ = await service.read_tool(name=target_tool.tool_name)
            target = RelationshipEntity(id=target_tool, kind=RelationshipEntityKind.TOOL)

        relationship = await self._relationship_store.create_relationship(
            source=source,
            target=target,
            kind=kind,
        )

        return await self._relationship_to_model(relationship=relationship)

    async def read(self, relationship_id: RelationshipId) -> RelationshipModel:
        relationship = await self._relationship_store.read_relationship(
            relationship_id=relationship_id
        )

        return await self._relationship_to_model(relationship=relationship)

    async def find(
        self,
        kind: RelationshipKind | None,
        indirect: bool,
        guideline_id: GuidelineId | None,
        tag_id: TagId | None,
        tool_id: ToolId | None,
    ) -> Sequence[RelationshipModel]:
        if not guideline_id and not tag_id and not tool_id:
            relationships = await self._relationship_store.list_relationships(
                kind=kind if kind else None,
                indirect=indirect,
            )

            return [
                await self._relationship_to_model(relationship=relationship)
                for relationship in relationships
            ]

        entity_id: GuidelineId | TagId | ToolId
        if guideline_id:
            await self._guideline_store.read_guideline(guideline_id=guideline_id)
            entity_id = guideline_id
        elif tag_id:
            await self._entity_id_to_tag(
                tag_id,
            )
            entity_id = tag_id
        elif tool_id:
            service = await self._service_registry.read_tool_service(name=tool_id.service_name)
            _ = await service.read_tool(name=tool_id.tool_name)
            entity_id = tool_id
        else:
            raise ValueError("Invalid entity ID")

        source_relationships = await self._relationship_store.list_relationships(
            kind=kind if kind else None,
            source_id=entity_id,
            indirect=indirect,
        )

        target_relationships = await self._relationship_store.list_relationships(
            kind=kind if kind else None,
            target_id=entity_id,
            indirect=indirect,
        )

        relationships = list(chain(source_relationships, target_relationships))

        return [
            await self._relationship_to_model(relationship=relationship)
            for relationship in relationships
        ]

    async def delete(self, relationship_id: RelationshipId) -> None:
        await self._relationship_store.delete_relationship(relationship_id=relationship_id)
