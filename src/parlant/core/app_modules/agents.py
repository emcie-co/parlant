from dataclasses import dataclass
from typing import Sequence

from parlant.core.app_modules.request_context import RequestContext
from parlant.core.loggers import Logger
from parlant.core.agents import (
    AgentId,
    AgentStore,
    Agent,
    AgentUpdateParams,
    CompositionMode,
    Effort,
    MessageOutputMode,
)
from parlant.core.groups import GroupId, GroupStore
from parlant.core.store_provider import StoreProviderHints, StoreProvider


@dataclass(frozen=True)
class AgentGroupUpdateParamsModel:
    add: list[GroupId] | None = None
    remove: list[GroupId] | None = None


class AgentModule:
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

    async def _ensure_tag(self, group_id: GroupId) -> None:
        await self._group_store.read_group(group_id)

    async def create(
        self,
        name: str,
        description: str | None,
        max_engine_iterations: int | None,
        composition_mode: CompositionMode | None,
        message_output_mode: MessageOutputMode | None,
        effort: Effort | None,
        groups: list[GroupId] | None,
        id: AgentId | None = None,
    ) -> Agent:
        if groups:
            for group_id in groups:
                await self._ensure_tag(group_id)

            groups = list(set(groups))

        agent = await self._agent_store.create_agent(
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations,
            composition_mode=composition_mode,
            message_output_mode=message_output_mode,
            effort=effort,
            groups=groups,
            id=id,
        )
        return agent

    async def read(self, agent_id: AgentId) -> Agent:
        agent = await self._agent_store.read_agent(agent_id=agent_id)
        return agent

    async def find(self) -> Sequence[Agent]:
        agents = await self._agent_store.list_agents()
        return agents

    async def update(
        self,
        agent_id: AgentId,
        name: str | None,
        description: str | None,
        max_engine_iterations: int | None,
        composition_mode: CompositionMode | None,
        message_output_mode: MessageOutputMode | None,
        effort: Effort | None,
        groups: AgentGroupUpdateParamsModel | None,
    ) -> Agent:
        update_params: AgentUpdateParams = {}

        if name:
            update_params["name"] = name

        if description:
            update_params["description"] = description

        if max_engine_iterations:
            update_params["max_engine_iterations"] = max_engine_iterations

        if composition_mode:
            update_params["composition_mode"] = composition_mode

        if message_output_mode:
            update_params["message_output_mode"] = message_output_mode

        if effort:
            update_params["effort"] = effort

        await self._agent_store.update_agent(agent_id=agent_id, params=update_params)

        if groups:
            if groups.add:
                for group_id in groups.add:
                    await self._ensure_tag(group_id)

                    await self._agent_store.upsert_group(
                        agent_id=agent_id,
                        group_id=group_id,
                    )

            if groups.remove:
                for group_id in groups.remove:
                    await self._agent_store.remove_group(
                        agent_id=agent_id,
                        group_id=group_id,
                    )

        agent = await self._agent_store.read_agent(agent_id)

        return agent

    async def delete(self, agent_id: AgentId) -> None:
        await self._agent_store.delete_agent(agent_id=agent_id)
