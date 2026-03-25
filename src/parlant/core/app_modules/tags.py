from typing import Optional, Sequence

from parlant.core.loggers import Logger
from parlant.core.tags import TagId, TagStore, Tag, TagUpdateParams
from parlant.core.store_provider import APP_CALL_SITE, StoreProvider


class TagModule:
    def __init__(
        self,
        logger: Logger,
        store_provider: StoreProvider,
    ):
        self._logger = logger
        self._store_provider = store_provider

    @property
    def _tag_store(self) -> TagStore:
        return self._store_provider.get_store(TagStore, APP_CALL_SITE)

    async def create(self, name: str) -> Tag:
        tag = await self._tag_store.create_tag(name=name)
        return tag

    async def read(self, tag_id: TagId) -> Tag:
        tag = await self._tag_store.read_tag(tag_id=tag_id)
        return tag

    async def find(self, name: Optional[str] = None) -> Sequence[Tag]:
        tags = await self._tag_store.list_tags(name=name)
        return tags

    async def update(self, tag_id: TagId, params: TagUpdateParams) -> Tag:
        tag = await self._tag_store.update_tag(tag_id=tag_id, params=params)
        return tag

    async def delete(self, tag_id: TagId) -> None:
        await self._tag_store.delete_tag(tag_id=tag_id)
