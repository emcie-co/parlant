from dataclasses import dataclass
from typing import Sequence

from parlant.core.loggers import Logger
from parlant.core.agents import AgentId, AgentStore, Agent, AgentUpdateParams, CompositionMode
from parlant.core.tags import TagId, TagStore


@dataclass(frozen=True)
class AgentTagUpdateParamsModel:
    add: list[TagId] | None = None
    remove: list[TagId] | None = None


class AgentModule:
    def __init__(
        self,
        logger: Logger,
        agent_store: AgentStore,
        tag_store: TagStore,
    ):
        self._logger = logger
        self._agent_store = agent_store
        self._tag_store = tag_store

    async def _ensure_tag(self, tag_id: TagId) -> None:
        await self._tag_store.read_tag(tag_id)

    async def create(
        self,
        name: str,
        description: str | None,
        max_engine_iterations: int | None,
        composition_mode: CompositionMode | None,
        tags: list[TagId] | None,
        id: AgentId | None = None,
    ) -> Agent:
        if tags:
            for tag_id in tags:
                await self._ensure_tag(tag_id)

            tags = list(set(tags))

        agent = await self._agent_store.create_agent(
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations,
            composition_mode=composition_mode,
            tags=tags,
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
        tags: AgentTagUpdateParamsModel | None,
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

        await self._agent_store.update_agent(agent_id=agent_id, params=update_params)

        if tags:
            if tags.add:
                for tag_id in tags.add:
                    await self._ensure_tag(tag_id)

                    await self._agent_store.upsert_tag(
                        agent_id=agent_id,
                        tag_id=tag_id,
                    )

            if tags.remove:
                for tag_id in tags.remove:
                    await self._agent_store.remove_tag(
                        agent_id=agent_id,
                        tag_id=tag_id,
                    )

        agent = await self._agent_store.read_agent(agent_id)

        return agent

    async def delete(self, agent_id: AgentId) -> None:
        await self._agent_store.delete_agent(agent_id=agent_id)
