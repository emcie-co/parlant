from dataclasses import dataclass
from typing import Sequence

from parlant.core.agents import AgentId, AgentStore
from parlant.core.app_modules.request_context import RequestContext
from parlant.core.journeys import JourneyId, JourneyStore
from parlant.core.loggers import Logger
from parlant.core.capabilities import (
    CapabilityId,
    CapabilityStore,
    Capability,
    CapabilityUpdateParams,
)
from parlant.core.groups import GroupIds, GroupId, GroupStore
from parlant.core.store_provider import StoreProviderHints, StoreProvider


@dataclass(frozen=True)
class CapabilityGroupUpdateParamsModel:
    add: Sequence[GroupId] | None = None
    remove: Sequence[GroupId] | None = None


class CapabilityModule:
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
    def _capability_store(self) -> CapabilityStore:
        return self._store_provider.get_store(
            CapabilityStore,
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
    def _group_store(self) -> GroupStore:
        return self._store_provider.get_store(
            GroupStore,
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
        title: str,
        description: str,
        signals: Sequence[str],
        groups: Sequence[GroupId] | None,
    ) -> Capability:
        if groups:
            for group_id in groups:
                await self._ensure_tag(group_id=group_id)

        capability = await self._capability_store.create_capability(
            title=title,
            description=description,
            signals=signals,
            groups=groups if groups else None,
        )

        return capability

    async def read(self, capability_id: CapabilityId) -> Capability:
        capability = await self._capability_store.read_capability(capability_id=capability_id)
        return capability

    async def find(self, group_id: GroupId | None) -> Sequence[Capability]:
        if group_id:
            capabilities = await self._capability_store.list_capabilities(
                groups=[group_id],
            )
        else:
            capabilities = await self._capability_store.list_capabilities()

        return capabilities

    async def update(
        self,
        capability_id: CapabilityId,
        title: str | None,
        description: str | None,
        signals: Sequence[str] | None,
        groups: CapabilityGroupUpdateParamsModel | None,
    ) -> Capability:
        update_params: CapabilityUpdateParams = {}
        if title:
            update_params["title"] = title
        if description:
            update_params["description"] = description
        if signals:
            update_params["signals"] = signals

        if update_params:
            capability = await self._capability_store.update_capability(
                capability_id=capability_id,
                params=update_params,
            )

        else:
            capability = await self._capability_store.read_capability(capability_id=capability_id)

        if groups:
            if groups.add:
                for group_id in groups.add:
                    await self._ensure_tag(group_id)

                    await self._capability_store.upsert_group(
                        capability_id=capability_id, group_id=group_id
                    )

            if groups.remove:
                for group_id in groups.remove:
                    await self._capability_store.remove_group(
                        capability_id=capability_id, group_id=group_id
                    )

        capability = await self._capability_store.read_capability(capability_id=capability_id)

        return capability

    async def delete(self, capability_id: CapabilityId) -> None:
        await self._capability_store.delete_capability(capability_id=capability_id)
