from dataclasses import dataclass
from typing import Sequence

from parlant.core.agents import AgentId, AgentStore
from parlant.core.app_modules.application_context import ApplicationContext
from parlant.core.journeys import JourneyId, JourneyStore
from parlant.core.loggers import Logger
from parlant.core.capabilities import (
    CapabilityId,
    CapabilityStore,
    Capability,
    CapabilityUpdateParams,
)
from parlant.core.tags import Tag, TagId, TagStore
from parlant.core.store_provider import StoreProviderHints, StoreProvider


@dataclass(frozen=True)
class CapabilityTagUpdateParamsModel:
    add: Sequence[TagId] | None = None
    remove: Sequence[TagId] | None = None


class CapabilityModule:
    def __init__(
        self,
        application_context: ApplicationContext,
        logger: Logger,
        store_provider: StoreProvider,
    ):
        self._application_context = application_context
        self._logger = logger
        self._store_provider = store_provider

    @property
    def _capability_store(self) -> CapabilityStore:
        return self._store_provider.get_store(
            CapabilityStore,
            StoreProviderHints(
                call_site="app",
                origin=self._application_context.get_origin(),
            ),
        )

    @property
    def _agent_store(self) -> AgentStore:
        return self._store_provider.get_store(
            AgentStore,
            StoreProviderHints(
                call_site="app",
                origin=self._application_context.get_origin(),
            ),
        )

    @property
    def _journey_store(self) -> JourneyStore:
        return self._store_provider.get_store(
            JourneyStore,
            StoreProviderHints(
                call_site="app",
                origin=self._application_context.get_origin(),
            ),
        )

    @property
    def _tag_store(self) -> TagStore:
        return self._store_provider.get_store(
            TagStore,
            StoreProviderHints(
                call_site="app",
                origin=self._application_context.get_origin(),
            ),
        )

    async def _ensure_tag(self, tag_id: TagId) -> None:
        if agent_id := Tag.extract_agent_id(tag_id):
            _ = await self._agent_store.read_agent(agent_id=AgentId(agent_id))
        elif journey_id := Tag.extract_journey_id(tag_id):
            _ = await self._journey_store.read_journey(journey_id=JourneyId(journey_id))
        else:
            _ = await self._tag_store.read_tag(tag_id=tag_id)

    async def create(
        self,
        title: str,
        description: str,
        signals: Sequence[str],
        tags: Sequence[TagId] | None,
    ) -> Capability:
        if tags:
            for tag_id in tags:
                await self._ensure_tag(tag_id=tag_id)

        capability = await self._capability_store.create_capability(
            title=title,
            description=description,
            signals=signals,
            tags=tags if tags else None,
        )

        return capability

    async def read(self, capability_id: CapabilityId) -> Capability:
        capability = await self._capability_store.read_capability(capability_id=capability_id)
        return capability

    async def find(self, tag_id: TagId | None) -> Sequence[Capability]:
        if tag_id:
            capabilities = await self._capability_store.list_capabilities(
                tags=[tag_id],
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
        tags: CapabilityTagUpdateParamsModel | None,
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

        if tags:
            if tags.add:
                for tag_id in tags.add:
                    await self._ensure_tag(tag_id)

                    await self._capability_store.upsert_tag(
                        capability_id=capability_id, tag_id=tag_id
                    )

            if tags.remove:
                for tag_id in tags.remove:
                    await self._capability_store.remove_tag(
                        capability_id=capability_id, tag_id=tag_id
                    )

        capability = await self._capability_store.read_capability(capability_id=capability_id)

        return capability

    async def delete(self, capability_id: CapabilityId) -> None:
        await self._capability_store.delete_capability(capability_id=capability_id)
