from dataclasses import dataclass
from typing import Sequence

from parlant.core.agents import AgentId, AgentStore
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
    ) -> CannedResponse:
        if tags:
            for tag_id in tags:
                await self._ensure_tag(tag_id=tag_id)

        canrep = await self._canrep_store.create_canned_response(
            value=value,
            fields=fields,
            signals=signals,
            tags=tags if tags else None,
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
    ) -> CannedResponse:
        if value:
            update_params: CannedResponseUpdateParams = {
                "value": value,
                "fields": fields,
            }

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
