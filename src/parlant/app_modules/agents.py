from dataclasses import dataclass
from typing import Sequence
from lagom import Container

from parlant.core.loggers import Logger
from parlant.core.agents import AgentId, AgentStore, Agent, AgentUpdateParams, CompositionMode
from parlant.core.tags import TagId, TagStore


@dataclass(frozen=True)
class AgentTagUpdateParamsModule:
    add: list[TagId] | None = None
    remove: list[TagId] | None = None


@dataclass(frozen=True)
class AgentUpdateParamsModule:
    name: str | None = None
    description: str | None = None
    max_engine_iterations: int | None = None
    composition_mode: CompositionMode | None = None
    tags: AgentTagUpdateParamsModule | None = None


class AgentModule:
    def __init__(
        self,
        container: Container,
    ):
        self._logger = container[Logger]
        self._agent_store = container[AgentStore]
        self._tag_store = container[TagStore]

    async def _ensure_tag(self, tag_id: TagId) -> None:
        await self._tag_store.read_tag(tag_id)

    async def create(
        self,
        name: str,
        description: str | None,
        max_engine_iterations: int | None,
        composition_mode: CompositionMode | None,
        tags: list[TagId] | None,
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
        )
        return agent

    async def read(self, agent_id: AgentId) -> Agent:
        agent = await self._agent_store.read_agent(agent_id=agent_id)
        return agent

    async def find(self) -> Sequence[Agent]:
        agents = await self._agent_store.list_agents()
        return agents

    async def update(self, agent_id: AgentId, params: AgentUpdateParamsModule) -> Agent:
        update_params: AgentUpdateParams = {}

        if params.name:
            update_params["name"] = params.name

        if params.description:
            update_params["description"] = params.description

        if params.max_engine_iterations:
            update_params["max_engine_iterations"] = params.max_engine_iterations

        if params.composition_mode:
            update_params["composition_mode"] = params.composition_mode

        await self._agent_store.update_agent(agent_id=agent_id, params=update_params)

        if params.tags:
            if params.tags.add:
                for tag_id in params.tags.add:
                    await self._ensure_tag(tag_id)

                    await self._agent_store.upsert_tag(
                        agent_id=agent_id,
                        tag_id=tag_id,
                    )

            if params.tags.remove:
                for tag_id in params.tags.remove:
                    await self._agent_store.remove_tag(
                        agent_id=agent_id,
                        tag_id=tag_id,
                    )

        agent = await self._agent_store.read_agent(agent_id)

        return agent

    async def delete(self, agent_id: AgentId) -> None:
        await self._agent_store.delete_agent(agent_id=agent_id)
