from dataclasses import dataclass
from typing import Sequence

from parlant.core.agents import AgentId, AgentStore
from parlant.core.common import JSONSerializable
from parlant.core.loggers import Logger
from parlant.core.context_variables import (
    ContextVariableId,
    ContextVariableStore,
    ContextVariable,
    ContextVariableUpdateParams,
    ContextVariableValue,
)
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tags import Tag, TagId, TagStore
from parlant.core.tools import ToolId


@dataclass(frozen=True)
class ContextVariableTagsUpdateParams:
    add: Sequence[TagId] | None = None
    remove: Sequence[TagId] | None = None


class ContextVariableModule:
    def __init__(
        self,
        logger: Logger,
        context_variable_store: ContextVariableStore,
        service_registry: ServiceRegistry,
        agent_store: AgentStore,
        tag_store: TagStore,
    ) -> None:
        self._logger = logger
        self._variable_store = context_variable_store
        self._service_registry = service_registry
        self._agent_store = agent_store
        self._tag_store = tag_store

    async def create(
        self,
        name: str,
        description: str | None,
        tool_id: ToolId | None,
        freshness_rules: str | None,
        tags: Sequence[TagId] | None,
    ) -> ContextVariable:
        if tool_id:
            service = await self._service_registry.read_tool_service(tool_id.service_name)
            _ = await service.read_tool(tool_id.tool_name)

        if tags:
            for tag_id in tags:
                if agent_id := Tag.extract_agent_id(tag_id):
                    _ = await self._agent_store.read_agent(agent_id=AgentId(agent_id))
                else:
                    _ = await self._tag_store.read_tag(tag_id=tag_id)

            tags = list(set(tags))

        variable = await self._variable_store.create_variable(
            name=name,
            description=description,
            tool_id=ToolId(tool_id.service_name, tool_id.tool_name) if tool_id else None,
            freshness_rules=freshness_rules,
            tags=tags,
        )
        return variable

    async def read(self, variable_id: ContextVariableId) -> ContextVariable:
        variable = await self._variable_store.read_variable(variable_id=variable_id)
        return variable

    async def find(self, tag_id: TagId | None) -> Sequence[ContextVariable]:
        if tag_id:
            variables = await self._variable_store.list_variables(
                tags=[tag_id],
            )
        else:
            variables = await self._variable_store.list_variables()

        return variables

    async def update(
        self,
        variable_id: ContextVariableId,
        name: str | None,
        description: str | None,
        tool_id: ToolId | None,
        freshness_rules: str | None,
        tags: ContextVariableTagsUpdateParams | None,
    ) -> ContextVariable:
        if name or description or tool_id or freshness_rules:
            update_params: ContextVariableUpdateParams = {}
            if name:
                update_params["name"] = name
            if description:
                update_params["description"] = description
            if tool_id:
                update_params["tool_id"] = tool_id
            if freshness_rules:
                update_params["freshness_rules"] = freshness_rules

            await self._variable_store.update_variable(
                variable_id=variable_id,
                params=update_params,
            )

        if tags:
            if tags.add:
                for tag_id in tags.add:
                    if agent_id := Tag.extract_agent_id(tag_id):
                        _ = await self._agent_store.read_agent(agent_id=AgentId(agent_id))
                    else:
                        _ = await self._tag_store.read_tag(tag_id=tag_id)
                    await self._variable_store.add_variable_tag(variable_id, tag_id)

            if tags.remove:
                for tag_id in tags.remove:
                    await self._variable_store.remove_variable_tag(variable_id, tag_id)

        updated_variable = await self._variable_store.read_variable(variable_id=variable_id)

        return updated_variable

    async def delete_many(self, tag_id: TagId | None) -> None:
        if tag_id:
            variables = await self._variable_store.list_variables(
                tags=[tag_id],
            )
            for v in variables:
                updated_variable = await self._variable_store.remove_variable_tag(
                    variable_id=v.id,
                    tag_id=tag_id,
                )
                if not updated_variable.tags:
                    await self._variable_store.delete_variable(variable_id=v.id)

        else:
            variables = await self._variable_store.list_variables()
            for v in variables:
                await self._variable_store.delete_variable(variable_id=v.id)

    async def delete(self, variable_id: ContextVariableId) -> None:
        await self._variable_store.delete_variable(variable_id=variable_id)

    async def read_value(
        self,
        variable_id: ContextVariableId,
        key: str,
    ) -> ContextVariableValue | None:
        _ = await self._variable_store.read_variable(variable_id=variable_id)

        value = await self._variable_store.read_value(variable_id=variable_id, key=key)
        return value

    async def find_values(
        self,
        variable_id: ContextVariableId,
    ) -> Sequence[tuple[str, ContextVariableValue]]:
        key_value_pairs = await self._variable_store.list_values(variable_id=variable_id)
        return key_value_pairs

    async def update_value(
        self,
        variable_id: ContextVariableId,
        key: str,
        data: JSONSerializable,
    ) -> ContextVariableValue:
        _ = await self._variable_store.read_variable(variable_id=variable_id)

        updated_value = await self._variable_store.update_value(
            variable_id=variable_id,
            key=key,
            data=data,
        )
        return updated_value

    async def delete_value(
        self,
        variable_id: ContextVariableId,
        key: str,
    ) -> None:
        await self._variable_store.delete_value(
            variable_id=variable_id,
            key=key,
        )
