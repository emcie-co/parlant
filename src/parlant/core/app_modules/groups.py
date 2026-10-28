from typing import Optional, Sequence

from parlant.core.app_modules.request_context import RequestContext
from parlant.core.loggers import Logger
from parlant.core.groups import GroupId, GroupStore, Group, GroupUpdateParams
from parlant.core.store_provider import StoreProviderHints, StoreProvider


class GroupModule:
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
    def _group_store(self) -> GroupStore:
        return self._store_provider.get_store(
            GroupStore,
            StoreProviderHints(
                call_site="app",
                origin=self._request_context.get_origin(),
            ),
        )

    async def create(self, name: str) -> Group:
        group = await self._group_store.create_group(name=name)
        return group

    async def read(self, group_id: GroupId) -> Group:
        group = await self._group_store.read_group(group_id=group_id)
        return group

    async def find(self, name: Optional[str] = None) -> Sequence[Group]:
        groups = await self._group_store.list_groups(name=name)
        return groups

    async def update(self, group_id: GroupId, params: GroupUpdateParams) -> Group:
        group = await self._group_store.update_group(group_id=group_id, params=params)
        return group

    async def delete(self, group_id: GroupId) -> None:
        await self._group_store.delete_group(group_id=group_id)
