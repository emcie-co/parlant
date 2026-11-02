from dataclasses import dataclass
from typing import Sequence

from parlant.core.agents import AgentId, AgentStore
from parlant.core.app_modules.request_context import RequestContext
from parlant.core.loggers import Logger
from parlant.core.glossary import TermId, GlossaryStore, Term, TermUpdateParams
from parlant.core.groups import GroupIds, GroupId, GroupStore
from parlant.core.store_provider import StoreProviderHints, StoreProvider


@dataclass(frozen=True)
class TermGroupsUpdateParamsModel:
    add: Sequence[GroupId] | None = None
    remove: Sequence[GroupId] | None = None


class GlossaryModule:
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
    def _glossary_store(self) -> GlossaryStore:
        return self._store_provider.get_store(
            GlossaryStore,
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
    def _group_store(self) -> GroupStore:
        return self._store_provider.get_store(
            GroupStore,
            StoreProviderHints(
                call_site="app",
                origin=self._request_context.get_origin(),
            ),
        )

    async def _ensure_tag(self, group: GroupId) -> None:
        if agent_id := GroupIds.extract_agent_id(group):
            _ = await self._agent_store.read_agent(agent_id=AgentId(agent_id))
        else:
            _ = await self._group_store.read_group(group_id=group)

    async def create(
        self,
        name: str,
        description: str,
        synonyms: Sequence[str],
        groups: Sequence[GroupId] | None,
        id: TermId | None = None,
    ) -> Term:
        if groups:
            for group_id in groups:
                await self._ensure_tag(group_id)

            groups = list(set(groups))

        term = await self._glossary_store.create_term(
            name=name,
            description=description,
            synonyms=synonyms,
            groups=groups or None,
            id=id,
        )

        return term

    async def read(self, term_id: TermId) -> Term:
        term = await self._glossary_store.read_term(term_id=term_id)
        return term

    async def find(self, group_id: GroupId | None) -> Sequence[Term]:
        if group_id:
            terms = await self._glossary_store.list_terms(groups=[group_id])
        else:
            terms = await self._glossary_store.list_terms()

        return terms

    async def update(
        self,
        term_id: TermId,
        name: str | None,
        description: str | None,
        synonyms: Sequence[str] | None,
        groups: TermGroupsUpdateParamsModel | None,
    ) -> Term:
        if groups:
            if groups.add:
                for group_id in groups.add:
                    await self._ensure_tag(group_id)
                    await self._glossary_store.upsert_group(
                        term_id=term_id,
                        group_id=group_id,
                    )

            if groups.remove:
                for group_id in groups.remove:
                    await self._glossary_store.remove_group(
                        term_id=term_id,
                        group_id=group_id,
                    )

        params: TermUpdateParams = {}
        if name:
            params["name"] = name
        if description:
            params["description"] = description
        if synonyms:
            params["synonyms"] = synonyms

        term = await self._glossary_store.update_term(
            term_id=term_id,
            params=params,
        )

        return term

    async def delete(self, term_id: TermId) -> None:
        await self._glossary_store.delete_term(term_id=term_id)
