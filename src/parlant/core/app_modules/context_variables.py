from dataclasses import dataclass
from typing import Sequence

from parlant.core.agents import AgentId, AgentStore
from parlant.core.app_modules.request_context import RequestContext
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
from parlant.core.groups import GroupIds, GroupId, GroupStore
from parlant.core.tools import ToolId
from parlant.core.store_provider import StoreProviderHints, StoreProvider


@dataclass(frozen=True)
class ContextVariableTagsUpdateParams:
    add: Sequence[GroupId] | None = None
    remove: Sequence[GroupId] | None = None


class ContextVariableModule:
    def __init__(
        self,
        request_context: RequestContext,
        logger: Logger,
        store_provider: StoreProvider,
    ) -> None:
        self._request_context = request_context
        self._logger = logger
        self._store_provider = store_provider

    @property
    def _variable_store(self) -> ContextVariableStore:
        return self._store_provider.get_store(
            ContextVariableStore,
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
    def _group_store(self) -> GroupStore:
        return self._store_provider.get_store(
            GroupStore,
            StoreProviderHints(
                call_site="app",
                origin=self._request_context.get_origin(),
            ),
        )

    async def create(
        self,
        name: str,
        description: str | None,
        tool_id: ToolId | None,
        freshness_rules: str | None,
        groups: Sequence[GroupId] | None,
    ) -> ContextVariable:
        if tool_id:
            service = await self._service_registry.read_tool_service(tool_id.service_name)
            _ = await service.read_tool(tool_id.tool_name)

        if groups:
            for group_id in groups:
                if agent_id := GroupIds.extract_agent_id(group_id):
                    _ = await self._agent_store.read_agent(agent_id=AgentId(agent_id))
                else:
                    _ = await self._group_store.read_group(group_id=group_id)

            groups = list(set(groups))

        variable = await self._variable_store.create_variable(
            name=name,
            description=description,
            tool_id=ToolId(tool_id.service_name, tool_id.tool_name) if tool_id else None,
            freshness_rules=freshness_rules,
            groups=groups,
        )
        return variable

    async def read(self, variable_id: ContextVariableId) -> ContextVariable:
        variable = await self._variable_store.read_variable(variable_id=variable_id)
        return variable

    async def find(self, group_id: GroupId | None) -> Sequence[ContextVariable]:
        if group_id:
            variables = await self._variable_store.list_variables(
                groups=[group_id],
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
        groups: ContextVariableTagsUpdateParams | None,
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

        if groups:
            if groups.add:
                for group_id in groups.add:
                    if agent_id := GroupIds.extract_agent_id(group_id):
                        _ = await self._agent_store.read_agent(agent_id=AgentId(agent_id))
                    else:
                        _ = await self._group_store.read_group(group_id=group_id)
                    await self._variable_store.add_variable_tag(variable_id, group_id)

            if groups.remove:
                for group_id in groups.remove:
                    await self._variable_store.remove_variable_tag(variable_id, group_id)

        updated_variable = await self._variable_store.read_variable(variable_id=variable_id)

        return updated_variable

    async def delete_many(self, group_id: GroupId | None) -> None:
        if group_id:
            variables = await self._variable_store.list_variables(
                groups=[group_id],
            )
            for v in variables:
                updated_variable = await self._variable_store.remove_variable_tag(
                    variable_id=v.id,
                    group_id=group_id,
                )
                if not updated_variable.groups:
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
