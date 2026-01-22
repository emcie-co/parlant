from dataclasses import dataclass
from typing import Any, Sequence

from parlant.core.loggers import Logger
from parlant.core.agents import AgentId, AgentStore, Agent, AgentUpdateParams, CompositionMode
from parlant.core.playbooks import PlaybookId, PlaybookStore, DisabledRuleRef
from parlant.core.tags import TagId, TagStore

# Sentinel value to indicate "not provided" (distinct from None which means "clear")
_NOT_PROVIDED: Any = object()


@dataclass(frozen=True)
class AgentTagUpdateParamsModel:
    add: list[TagId] | None = None
    remove: list[TagId] | None = None


@dataclass(frozen=True)
class AgentDisabledRulesUpdateParamsModel:
    add: list[DisabledRuleRef] | None = None
    remove: list[DisabledRuleRef] | None = None


class AgentModule:
    def __init__(
        self,
        logger: Logger,
        agent_store: AgentStore,
        tag_store: TagStore,
        playbook_store: PlaybookStore,
    ):
        self._logger = logger
        self._agent_store = agent_store
        self._tag_store = tag_store
        self._playbook_store = playbook_store

    async def _ensure_tag(self, tag_id: TagId) -> None:
        await self._tag_store.read_tag(tag_id)

    async def _ensure_playbook(self, playbook_id: PlaybookId) -> None:
        await self._playbook_store.read_playbook(playbook_id)

    async def create(
        self,
        name: str,
        description: str | None,
        max_engine_iterations: int | None,
        composition_mode: CompositionMode | None,
        tags: list[TagId] | None,
        id: AgentId | None = None,
        playbook_id: PlaybookId | None = None,
    ) -> Agent:
        if tags:
            for tag_id in tags:
                await self._ensure_tag(tag_id)

            tags = list(set(tags))

        if playbook_id:
            await self._ensure_playbook(playbook_id)

        agent = await self._agent_store.create_agent(
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations,
            composition_mode=composition_mode,
            tags=tags,
            id=id,
            playbook_id=playbook_id,
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
        playbook_id: PlaybookId | None | Any = _NOT_PROVIDED,
        disabled_rules: AgentDisabledRulesUpdateParamsModel | None = None,
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

        if playbook_id is not _NOT_PROVIDED:
            if playbook_id:
                await self._ensure_playbook(playbook_id)
            update_params["playbook_id"] = playbook_id

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

        if disabled_rules:
            if disabled_rules.add:
                for rule_ref in disabled_rules.add:
                    await self._agent_store.add_disabled_rule(
                        agent_id=agent_id,
                        rule_ref=rule_ref,
                    )

            if disabled_rules.remove:
                for rule_ref in disabled_rules.remove:
                    await self._agent_store.remove_disabled_rule(
                        agent_id=agent_id,
                        rule_ref=rule_ref,
                    )

        agent = await self._agent_store.read_agent(agent_id)

        return agent

    async def delete(self, agent_id: AgentId) -> None:
        await self._agent_store.delete_agent(agent_id=agent_id)
