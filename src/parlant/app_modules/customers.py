from dataclasses import dataclass
from typing import Mapping, Sequence
from lagom import Container

from parlant.core.agents import AgentId, AgentStore
from parlant.core.loggers import Logger
from parlant.core.customers import CustomerId, CustomerStore, Customer
from parlant.core.tags import Tag, TagId, TagStore


@dataclass(frozen=True)
class CustomerMetadataUpdateParams:
    set: Mapping[str, str] | None = None
    unset: Sequence[str] | None = None


@dataclass(frozen=True)
class CustomerTagUpdateParams:
    add: Sequence[TagId] | None = None
    remove: Sequence[TagId] | None = None


class CustomerModule:
    def __init__(
        self,
        container: Container,
    ):
        self._logger = container[Logger]
        self._customer_store = container[CustomerStore]
        self._agent_store = container[AgentStore]
        self._tag_store = container[TagStore]

    async def _ensure_tag(self, tag_id: TagId) -> None:
        if agent_id := Tag.extract_agent_id(tag_id):
            _ = await self._agent_store.read_agent(agent_id=AgentId(agent_id))
        else:
            _ = await self._tag_store.read_tag(tag_id=tag_id)

    async def create(
        self,
        name: str,
        extra: Mapping[str, str],
        tags: Sequence[TagId] | None,
    ) -> Customer:
        if tags:
            for tag_id in tags:
                await self._ensure_tag(tag_id)

            tags = list(set(tags))

        customer = await self._customer_store.create_customer(
            name=name,
            extra=extra,
            tags=tags or [],
        )
        return customer

    async def read(self, customer_id: CustomerId) -> Customer:
        customer = await self._customer_store.read_customer(customer_id=customer_id)
        return customer

    async def find(self) -> Sequence[Customer]:
        customers = await self._customer_store.list_customers()
        return customers

    async def update(
        self,
        customer_id: CustomerId,
        name: str | None,
        metadata: CustomerMetadataUpdateParams | None,
        tags: CustomerTagUpdateParams | None,
    ) -> Customer:
        if name:
            _ = await self._customer_store.update_customer(
                customer_id=customer_id,
                params={"name": name},
            )

        if metadata:
            if metadata.set:
                await self._customer_store.add_extra(customer_id, metadata.set)
            if metadata.unset:
                await self._customer_store.remove_extra(customer_id, metadata.unset)

        if tags:
            if tags.add:
                for tag_id in tags.add:
                    await self._ensure_tag(tag_id)
                    await self._customer_store.upsert_tag(customer_id, tag_id)
            if tags.remove:
                for tag_id in tags.remove:
                    await self._customer_store.remove_tag(customer_id, tag_id)

        customer = await self.read(customer_id)
        return customer

    async def delete(self, customer_id: CustomerId) -> None:
        await self._customer_store.delete_customer(customer_id=customer_id)
