from dataclasses import dataclass
from typing import Sequence, Mapping

from parlant.core.agents import AgentId, AgentStore
from parlant.core.app_modules.request_context import RequestContext
from parlant.core.common import JSONSerializable
from parlant.core.canned_responses import (
    CannedResponse,
    CannedResponseField,
    CannedResponseId,
    CannedResponseStore,
    CannedResponseUpdateParams,
)
from parlant.core.journeys import JourneyId, JourneyStore
from parlant.core.loggers import Logger
from parlant.core.groups import GroupIds, GroupId, GroupStore
from parlant.core.store_provider import StoreProviderHints, StoreProvider


@dataclass(frozen=True)
class CannedResponseGroupUpdateParamsModel:
    add: Sequence[GroupId] | None = None
    remove: Sequence[GroupId] | None = None


@dataclass(frozen=True)
class CannedResponseMetadataUpdateParamsModel:
    set: Mapping[str, JSONSerializable] | None = None
    unset: Sequence[str] | None = None


class CannedResponseModule:
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
    def _canrep_store(self) -> CannedResponseStore:
        return self._store_provider.get_store(
            CannedResponseStore,
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
        value: str,
        fields: Sequence[CannedResponseField],
        signals: Sequence[str] | None,
        groups: Sequence[GroupId] | None,
        metadata: Mapping[str, JSONSerializable] | None = None,
        field_dependencies: Sequence[str] | None = None,
    ) -> CannedResponse:
        if groups:
            for group_id in groups:
                await self._ensure_tag(group_id=group_id)

        canrep = await self._canrep_store.create_canned_response(
            value=value,
            fields=fields,
            signals=signals,
            groups=groups if groups else None,
            metadata=metadata or {},
            field_dependencies=field_dependencies,
        )

        return canrep

    async def read(self, canned_response_id: CannedResponseId) -> CannedResponse:
        canrep = await self._canrep_store.read_canned_response(
            canned_response_id=canned_response_id
        )
        return canrep

    async def find(self, groups: Sequence[GroupId] | None) -> Sequence[CannedResponse]:
        if groups:
            canreps = await self._canrep_store.list_canned_responses(groups=groups)
        else:
            canreps = await self._canrep_store.list_canned_responses()

        return canreps

    async def update(
        self,
        canned_response_id: CannedResponseId,
        value: str | None,
        fields: Sequence[CannedResponseField],
        groups: CannedResponseGroupUpdateParamsModel | None,
        metadata: CannedResponseMetadataUpdateParamsModel | None = None,
    ) -> CannedResponse:
        update_params: CannedResponseUpdateParams = {}
        needs_update = False

        if value:
            update_params["value"] = value
            update_params["fields"] = fields
            needs_update = True

        if metadata:
            # Get current canned response to merge metadata
            current_canrep = await self._canrep_store.read_canned_response(canned_response_id)
            current_metadata = dict(current_canrep.metadata) if current_canrep.metadata else {}

            # Apply set operations
            if metadata.set:
                current_metadata.update(metadata.set)

            # Apply unset operations
            if metadata.unset:
                for key in metadata.unset:
                    current_metadata.pop(key, None)

            update_params["metadata"] = current_metadata
            needs_update = True

        if needs_update:
            await self._canrep_store.update_canned_response(canned_response_id, update_params)

        if groups:
            if groups.add:
                for group_id in groups.add:
                    await self._ensure_tag(group_id=group_id)
                    await self._canrep_store.upsert_group(canned_response_id, group_id)
            if groups.remove:
                for group_id in groups.remove:
                    await self._canrep_store.remove_group(canned_response_id, group_id)

        updated_canrep = await self._canrep_store.read_canned_response(canned_response_id)

        return updated_canrep

    async def delete(self, canned_response_id: CannedResponseId) -> None:
        await self._canrep_store.delete_canned_response(canned_response_id=canned_response_id)
