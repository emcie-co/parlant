from dataclasses import dataclass
from typing import Sequence, Mapping

from parlant.core.agents import AgentId, AgentStore
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
from parlant.core.tags import Tag, TagId, TagStore


@dataclass(frozen=True)
class CannedResponseTagUpdateParamsModel:
    add: Sequence[TagId] | None = None
    remove: Sequence[TagId] | None = None


@dataclass(frozen=True)
class CannedResponseMetadataUpdateParamsModel:
    set: Mapping[str, JSONSerializable] | None = None
    unset: Sequence[str] | None = None


class CannedResponseModule:
    def __init__(
        self,
        logger: Logger,
        canned_response_store: CannedResponseStore,
        agent_store: AgentStore,
        journey_store: JourneyStore,
        tag_store: TagStore,
    ):
        self._logger = logger
        self._canrep_store = canned_response_store
        self._agent_store = agent_store
        self._journey_store = journey_store
        self._tag_store = tag_store

    async def _ensure_tag(self, tag_id: TagId) -> None:
        if agent_id := Tag.extract_agent_id(tag_id):
            _ = await self._agent_store.read_agent(agent_id=AgentId(agent_id))
        elif journey_id := Tag.extract_journey_id(tag_id):
            _ = await self._journey_store.read_journey(journey_id=JourneyId(journey_id))
        else:
            _ = await self._tag_store.read_tag(tag_id=tag_id)

    async def create(
        self,
        value: str,
        fields: Sequence[CannedResponseField],
        signals: Sequence[str] | None,
        tags: Sequence[TagId] | None,
        metadata: Mapping[str, JSONSerializable] | None = None,
    ) -> CannedResponse:
        if tags:
            for tag_id in tags:
                await self._ensure_tag(tag_id=tag_id)

        canrep = await self._canrep_store.create_canned_response(
            value=value,
            fields=fields,
            signals=signals,
            tags=tags if tags else None,
            metadata=metadata or {},
        )

        return canrep

    async def read(self, canned_response_id: CannedResponseId) -> CannedResponse:
        canrep = await self._canrep_store.read_canned_response(
            canned_response_id=canned_response_id
        )
        return canrep

    async def find(self, tags: Sequence[TagId] | None) -> Sequence[CannedResponse]:
        if tags:
            canreps = await self._canrep_store.list_canned_responses(tags=tags)
        else:
            canreps = await self._canrep_store.list_canned_responses()

        return canreps

    async def update(
        self,
        canned_response_id: CannedResponseId,
        value: str | None,
        fields: Sequence[CannedResponseField],
        tags: CannedResponseTagUpdateParamsModel | None,
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

        if tags:
            if tags.add:
                for tag_id in tags.add:
                    await self._ensure_tag(tag_id=tag_id)
                    await self._canrep_store.upsert_tag(canned_response_id, tag_id)
            if tags.remove:
                for tag_id in tags.remove:
                    await self._canrep_store.remove_tag(canned_response_id, tag_id)

        updated_canrep = await self._canrep_store.read_canned_response(canned_response_id)

        return updated_canrep

    async def delete(self, canned_response_id: CannedResponseId) -> None:
        await self._canrep_store.delete_canned_response(canned_response_id=canned_response_id)
