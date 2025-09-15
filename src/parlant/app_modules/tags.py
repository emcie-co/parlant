from typing import Sequence

from parlant.core.loggers import Logger
from parlant.core.tags import TagId, TagStore, Tag, TagUpdateParams


class TagModule:
    def __init__(
        self,
        logger: Logger,
        tag_store: TagStore,
    ):
        self._logger = logger
        self._tag_store = tag_store

    async def create(self, name: str) -> Tag:
        tag = await self._tag_store.create_tag(name=name)
        return tag

    async def read(self, tag_id: TagId) -> Tag:
        tag = await self._tag_store.read_tag(tag_id=tag_id)
        return tag

    async def find(self) -> Sequence[Tag]:
        tags = await self._tag_store.list_tags()
        return tags

    async def update(self, tag_id: TagId, params: TagUpdateParams) -> Tag:
        tag = await self._tag_store.update_tag(tag_id=tag_id, params=params)
        return tag

    async def delete(self, tag_id: TagId) -> None:
        await self._tag_store.delete_tag(tag_id=tag_id)
