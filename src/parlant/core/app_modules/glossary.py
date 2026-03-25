from dataclasses import dataclass
from typing import Sequence

from parlant.core.agents import AgentId, AgentStore
from parlant.core.loggers import Logger
from parlant.core.glossary import TermId, GlossaryStore, Term, TermUpdateParams
from parlant.core.tags import Tag, TagId, TagStore
from parlant.core.store_provider import APP_CALL_SITE, StoreProvider


@dataclass(frozen=True)
class TermTagsUpdateParamsModel:
    add: Sequence[TagId] | None = None
    remove: Sequence[TagId] | None = None


class GlossaryModule:
    def __init__(
        self,
        logger: Logger,
        store_provider: StoreProvider,
    ):
        self._logger = logger
        self._store_provider = store_provider

    @property
    def _glossary_store(self) -> GlossaryStore:
        return self._store_provider.get_store(GlossaryStore, APP_CALL_SITE)

    @property
    def _agent_store(self) -> AgentStore:
        return self._store_provider.get_store(AgentStore, APP_CALL_SITE)

    @property
    def _tag_store(self) -> TagStore:
        return self._store_provider.get_store(TagStore, APP_CALL_SITE)

    async def _ensure_tag(self, tag: TagId) -> None:
        if agent_id := Tag.extract_agent_id(tag):
            _ = await self._agent_store.read_agent(agent_id=AgentId(agent_id))
        else:
            _ = await self._tag_store.read_tag(tag_id=tag)

    async def create(
        self,
        name: str,
        description: str,
        synonyms: Sequence[str],
        tags: Sequence[TagId] | None,
        id: TermId | None = None,
    ) -> Term:
        if tags:
            for tag_id in tags:
                await self._ensure_tag(tag_id)

            tags = list(set(tags))

        term = await self._glossary_store.create_term(
            name=name,
            description=description,
            synonyms=synonyms,
            tags=tags or None,
            id=id,
        )

        return term

    async def read(self, term_id: TermId) -> Term:
        term = await self._glossary_store.read_term(term_id=term_id)
        return term

    async def find(self, tag_id: TagId | None) -> Sequence[Term]:
        if tag_id:
            terms = await self._glossary_store.list_terms(tags=[tag_id])
        else:
            terms = await self._glossary_store.list_terms()

        return terms

    async def update(
        self,
        term_id: TermId,
        name: str | None,
        description: str | None,
        synonyms: Sequence[str] | None,
        tags: TermTagsUpdateParamsModel | None,
    ) -> Term:
        if tags:
            if tags.add:
                for tag_id in tags.add:
                    await self._ensure_tag(tag_id)
                    await self._glossary_store.upsert_tag(
                        term_id=term_id,
                        tag_id=tag_id,
                    )

            if tags.remove:
                for tag_id in tags.remove:
                    await self._glossary_store.remove_tag(
                        term_id=term_id,
                        tag_id=tag_id,
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
