from dataclasses import dataclass
from typing import Mapping, Sequence

from parlant.core.agents import AgentId, AgentStore
from parlant.core.app_modules.request_context import RequestContext
from parlant.core.loggers import Logger
from parlant.core.customers import CustomerId, CustomerStore, Customer, CustomerListing
from parlant.core.persistence.common import Cursor, SortDirection
from parlant.core.groups import GroupIds, GroupId, GroupStore
from parlant.core.store_provider import StoreProviderHints, StoreProvider


@dataclass(frozen=True)
class CustomerListingModel:
    """Paginated result model for customers at the application layer"""

    items: Sequence[Customer]
    total_count: int
    has_more: bool
    next_cursor: Cursor | None = None


@dataclass(frozen=True)
class CustomerMetadataUpdateParams:
    set: Mapping[str, str] | None = None
    unset: Sequence[str] | None = None


@dataclass(frozen=True)
class CustomerGroupUpdateParams:
    add: Sequence[GroupId] | None = None
    remove: Sequence[GroupId] | None = None


class CustomerModule:
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
    def _customer_store(self) -> CustomerStore:
        return self._store_provider.get_store(
            CustomerStore,
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

    async def _ensure_tag(self, group_id: GroupId) -> None:
        if agent_id := GroupIds.extract_agent_id(group_id):
            _ = await self._agent_store.read_agent(agent_id=AgentId(agent_id))
        else:
            _ = await self._group_store.read_group(group_id=group_id)

    async def create(
        self,
        name: str,
        extra: Mapping[str, str],
        groups: Sequence[GroupId] | None,
        id: CustomerId | None = None,
    ) -> Customer:
        if groups:
            for group_id in groups:
                await self._ensure_tag(group_id)

            groups = list(set(groups))

        customer = await self._customer_store.create_customer(
            name=name,
            extra=extra,
            groups=groups or [],
            id=id,
        )
        return customer

    async def read(self, customer_id: CustomerId) -> Customer:
        customer = await self._customer_store.read_customer(customer_id=customer_id)
        return customer

    async def find(
        self,
        limit: int | None = None,
        cursor: Cursor | None = None,
        sort_direction: SortDirection | None = None,
    ) -> CustomerListing:
        result = await self._customer_store.list_customers(
            limit=limit,
            cursor=cursor,
            sort_direction=sort_direction,
        )
        return result

    async def update(
        self,
        customer_id: CustomerId,
        name: str | None,
        metadata: CustomerMetadataUpdateParams | None,
        groups: CustomerGroupUpdateParams | None,
    ) -> Customer:
        if name:
            _ = await self._customer_store.update_customer(
                customer_id=customer_id,
                params={"name": name},
            )

        if metadata:
            if metadata.set:
                await self._customer_store.upsert_extra(customer_id, metadata.set)
            if metadata.unset:
                await self._customer_store.remove_extra(customer_id, metadata.unset)

        if groups:
            if groups.add:
                for group_id in groups.add:
                    await self._ensure_tag(group_id)
                    await self._customer_store.upsert_group(customer_id, group_id)
            if groups.remove:
                for group_id in groups.remove:
                    await self._customer_store.remove_group(customer_id, group_id)

        customer = await self.read(customer_id)
        return customer

    async def delete(self, customer_id: CustomerId) -> None:
        await self._customer_store.delete_customer(customer_id=customer_id)
