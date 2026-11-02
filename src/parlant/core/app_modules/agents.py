from dataclasses import dataclass
from typing import Mapping, Sequence

from parlant.core.app_modules.request_context import RequestContext
from parlant.core.loggers import Logger
from parlant.core.nlp.moderation import NoModeration
from parlant.core.nlp.model_routing import ModelRouter
from parlant.core.nlp.model_routing import (
    ModelConfigurationService,
    ModelRouteUpdates,
    ModelRouteValue,
    ModelScope,
)
from parlant.core.agents import (
    Moderation,
    AgentId,
    AgentStore,
    Agent,
    AgentUpdateParams,
    Effort,
    MessageOutputMode,
)
from parlant.core.groups import GroupId, GroupStore
from parlant.core.store_provider import StoreProviderHints, StoreProvider


@dataclass(frozen=True)
class AgentGroupUpdateParamsModel:
    add: list[GroupId] | None = None
    remove: list[GroupId] | None = None


async def validate_required_moderation(
    model_router: ModelRouter,
    agent: Agent,
) -> None:
    """Enforce moderation=ON: both moderation routes must be configured.

    Routing a checkpoint to the "null" provider counts - the route is then
    genuinely configured, and the checkpoint is deliberately waived. Called on
    agent creation/update and for every agent at server startup.
    """
    if agent.moderation is not Moderation.ON:
        return

    for route in ("moderation.customer", "moderation.agent"):
        try:
            await model_router.get_moderation_service(
                context=None,
                agent_id=agent.id,
                route=route,
            )
        except Exception as exc:
            raise ValueError(
                f"Agent '{agent.name}' has moderation=on, but no model is configured "
                f"for the '{route}' route. Route it to a moderation model, or to the "
                f"'null' provider to deliberately waive this checkpoint."
            ) from exc


async def validate_moderation_configuration(
    model_router: ModelRouter,
    agent: Agent,
    logger: Logger,
) -> None:
    """Full moderation-configuration check for an agent.

    Enforces moderation=ON route requirements (raising), and warns when
    agent-side moderation is configured but the agent's output mode isn't
    BLOCK: only the blocking loop runs the agent-message gate, so streamed
    or voice output is delivered without it.
    """
    await validate_required_moderation(model_router, agent)

    if agent.message_output_mode is MessageOutputMode.BLOCK:
        return
    if agent.moderation is Moderation.OFF:
        return

    try:
        service = await model_router.get_moderation_service(
            context=None,
            agent_id=agent.id,
            route="moderation.agent",
        )
    except Exception:
        return  # nothing configured agent-side; nothing is being bypassed

    if isinstance(service, NoModeration):
        return  # deliberately waived via the null provider

    logger.warning(
        f"Agent '{agent.name}' has agent-message moderation configured but uses "
        f"'{agent.message_output_mode.value}' output mode. Agent-side moderation "
        "only gates block output — streamed messages are delivered without the "
        "check. Set message output mode to block, or waive the 'moderation.agent' "
        "checkpoint by routing it to the 'null' provider."
    )


class AgentModule:
    def __init__(
        self,
        request_context: RequestContext,
        logger: Logger,
        store_provider: StoreProvider,
        model_configuration: ModelConfigurationService,
        model_router: ModelRouter,
    ):
        self._request_context = request_context
        self._logger = logger
        self._store_provider = store_provider
        self._model_configuration = model_configuration
        self._model_router = model_router

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
        instructions: str | None,
        max_steps: int | None,
        message_output_mode: MessageOutputMode | None,
        effort: Effort | None,
        groups: list[GroupId] | None,
        id: AgentId | None = None,
        models: ModelRouteUpdates | None = None,
        moderation: Moderation | None = None,
        message_review: bool = False,
        tool_review: bool = False,
    ) -> Agent:
        if groups:
            for group_id in groups:
                await self._ensure_tag(group_id)

            groups = list(set(groups))

        agent = await self._agent_store.create_agent(
            name=name,
            instructions=instructions,
            max_steps=max_steps,
            message_output_mode=message_output_mode,
            effort=effort,
            groups=groups,
            id=id,
            moderation=moderation,
            message_review=message_review,
            tool_review=tool_review,
        )
        try:
            if models:
                await self._model_configuration.configure(ModelScope.agent(agent.id), models)
            await validate_moderation_configuration(self._model_router, agent, self._logger)
        except Exception:
            await self._agent_store.delete_agent(agent.id)
            raise
        return agent

    async def read_models(self, agent_id: AgentId) -> Mapping[str, ModelRouteValue]:
        return await self._model_configuration.read(ModelScope.agent(agent_id))

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
        instructions: str | None,
        max_steps: int | None,
        message_output_mode: MessageOutputMode | None,
        effort: Effort | None,
        groups: AgentGroupUpdateParamsModel | None,
        models: ModelRouteUpdates | None = None,
        moderation: Moderation | None = None,
        message_review: bool | None = None,
        tool_review: bool | None = None,
    ) -> Agent:
        await self._agent_store.read_agent(agent_id)
        update_params: AgentUpdateParams = {}

        if name:
            update_params["name"] = name

        if instructions:
            update_params["instructions"] = instructions

        if max_steps:
            update_params["max_steps"] = max_steps

        if message_output_mode:
            update_params["message_output_mode"] = message_output_mode

        if effort:
            update_params["effort"] = effort

        if moderation is not None:
            update_params["moderation"] = moderation

        if message_review is not None:
            update_params["message_review"] = message_review

        if tool_review is not None:
            update_params["tool_review"] = tool_review

        if models is not None:
            await self._model_configuration.configure(ModelScope.agent(agent_id), models)

        await self._agent_store.update_agent(agent_id=agent_id, params=update_params)

        await validate_moderation_configuration(
            self._model_router,
            await self._agent_store.read_agent(agent_id),
            self._logger,
        )

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
        routes = await self._model_configuration.read(ModelScope.agent(agent_id))
        if routes:
            await self._model_configuration.configure(
                ModelScope.agent(agent_id),
                {route: None for route in routes},
            )
        await self._agent_store.delete_agent(agent_id=agent_id)
