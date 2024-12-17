# Copyright 2024 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, NewType, Optional, Sequence
from typing_extensions import override, TypedDict, Self

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.tags import TagId
from parlant.core.common import ItemNotFoundError, UniqueId, Version, generate_id
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import DocumentDatabase, DocumentCollection

CustomerId = NewType("CustomerId", str)


@dataclass(frozen=True)
class Customer:
    id: CustomerId
    creation_utc: datetime
    name: str
    extra: Mapping[str, str]
    tags: Sequence[TagId]


class CustomerUpdateParams(TypedDict, total=False):
    name: str


class CustomerStore(ABC):
    GUEST_ID = CustomerId("guest")

    @abstractmethod
    async def create_customer(
        self,
        name: str,
        extra: Mapping[str, str] = {},
        creation_utc: Optional[datetime] = None,
    ) -> Customer: ...

    @abstractmethod
    async def read_customer(
        self,
        customer_id: CustomerId,
    ) -> Customer: ...

    @abstractmethod
    async def update_customer(
        self,
        customer_id: CustomerId,
        params: CustomerUpdateParams,
    ) -> Customer: ...

    @abstractmethod
    async def delete_customer(
        self,
        customer_id: CustomerId,
    ) -> None: ...

    @abstractmethod
    async def list_customers(
        self,
    ) -> Sequence[Customer]: ...

    @abstractmethod
    async def add_tag(
        self,
        customer_id: CustomerId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> Customer: ...

    @abstractmethod
    async def remove_tag(
        self,
        customer_id: CustomerId,
        tag_id: TagId,
    ) -> Customer: ...

    @abstractmethod
    async def add_extra(
        self,
        customer_id: CustomerId,
        extra: Mapping[str, str],
    ) -> Customer: ...

    @abstractmethod
    async def remove_extra(
        self,
        customer_id: CustomerId,
        keys: Sequence[str],
    ) -> Customer: ...


class _CustomerDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    name: str
    extra: Mapping[str, str]


class _CustomerTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    customer_id: CustomerId
    tag_id: TagId


class CustomerDocumentStore(CustomerStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        database: DocumentDatabase,
    ) -> None:
        self._database = database
        self._customers_collection: DocumentCollection[_CustomerDocument]
        self._customer_tag_association_collection: DocumentCollection[
            _CustomerTagAssociationDocument
        ]

        self._lock = ReaderWriterLock()

    async def __aenter__(self) -> Self:
        self._customers_collection = await self._database.get_or_create_collection(
            name="customers",
            schema=_CustomerDocument,
        )
        self._customer_tag_association_collection = await self._database.get_or_create_collection(
            name="customer_tag_associations",
            schema=_CustomerTagAssociationDocument,
        )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    def _serialize_customer(self, customer: Customer) -> _CustomerDocument:
        return _CustomerDocument(
            id=ObjectId(customer.id),
            version=self.VERSION.to_string(),
            creation_utc=customer.creation_utc.isoformat(),
            name=customer.name,
            extra=customer.extra,
        )

    async def _deserialize_customer(self, customer_document: _CustomerDocument) -> Customer:
        tags = [
            doc["tag_id"]
            for doc in await self._customer_tag_association_collection.find(
                {"customer_id": {"$eq": customer_document["id"]}}
            )
        ]

        return Customer(
            id=CustomerId(customer_document["id"]),
            creation_utc=datetime.fromisoformat(customer_document["creation_utc"]),
            name=customer_document["name"],
            extra=customer_document["extra"],
            tags=tags,
        )

    @override
    async def create_customer(
        self,
        name: str,
        extra: Mapping[str, str] = {},
        creation_utc: Optional[datetime] = None,
    ) -> Customer:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            customer = Customer(
                id=CustomerId(generate_id()),
                name=name,
                extra=extra,
                creation_utc=creation_utc,
                tags=[],
            )

            await self._customers_collection.insert_one(
                document=self._serialize_customer(customer=customer)
            )

        return customer

    @override
    async def read_customer(
        self,
        customer_id: CustomerId,
    ) -> Customer:
        async with self._lock.reader_lock:
            if customer_id == CustomerStore.GUEST_ID:
                return Customer(
                    id=CustomerStore.GUEST_ID,
                    name="<guest>",
                    creation_utc=datetime.now(timezone.utc),
                    extra={},
                    tags=[],
                )

            customer_document = await self._customers_collection.find_one(
                filters={"id": {"$eq": customer_id}}
            )

        if not customer_document:
            raise ItemNotFoundError(item_id=UniqueId(customer_id))

        return await self._deserialize_customer(customer_document)

    @override
    async def update_customer(
        self,
        customer_id: CustomerId,
        params: CustomerUpdateParams,
    ) -> Customer:
        async with self._lock.writer_lock:
            customer_document = await self._customers_collection.find_one(
                filters={"id": {"$eq": customer_id}}
            )

            if not customer_document:
                raise ItemNotFoundError(item_id=UniqueId(customer_id))

            result = await self._customers_collection.update_one(
                filters={"id": {"$eq": customer_id}},
                params={"name": params["name"]},
            )

        assert result.updated_document

        return await self._deserialize_customer(customer_document=result.updated_document)

    async def list_customers(
        self,
    ) -> Sequence[Customer]:
        async with self._lock.reader_lock:
            return [await self.read_customer(CustomerStore.GUEST_ID)] + [
                await self._deserialize_customer(e)
                for e in await self._customers_collection.find({})
            ]

    @override
    async def delete_customer(
        self,
        customer_id: CustomerId,
    ) -> None:
        if customer_id == CustomerStore.GUEST_ID:
            raise ValueError("Removing the guest customer is not allowed")

        async with self._lock.writer_lock:
            result = await self._customers_collection.delete_one({"id": {"$eq": customer_id}})

        if result.deleted_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(customer_id))

    @override
    async def add_tag(
        self,
        customer_id: CustomerId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> Customer:
        async with self._lock.writer_lock:
            customer = await self.read_customer(customer_id)

            if tag_id in customer.tags:
                return customer

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_document: _CustomerTagAssociationDocument = {
                "id": ObjectId(generate_id()),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "customer_id": customer_id,
                "tag_id": tag_id,
            }

            _ = await self._customer_tag_association_collection.insert_one(
                document=association_document
            )

            customer_document = await self._customers_collection.find_one(
                {"id": {"$eq": customer_id}}
            )

        if not customer_document:
            raise ItemNotFoundError(item_id=UniqueId(customer_id))

        return await self._deserialize_customer(customer_document=customer_document)

    @override
    async def remove_tag(
        self,
        customer_id: CustomerId,
        tag_id: TagId,
    ) -> Customer:
        async with self._lock.writer_lock:
            delete_result = await self._customer_tag_association_collection.delete_one(
                {
                    "customer_id": {"$eq": customer_id},
                    "tag_id": {"$eq": tag_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(tag_id))

            customer_document = await self._customers_collection.find_one(
                {"id": {"$eq": customer_id}}
            )

        if not customer_document:
            raise ItemNotFoundError(item_id=UniqueId(customer_id))

        return await self._deserialize_customer(customer_document=customer_document)

    @override
    async def add_extra(
        self,
        customer_id: CustomerId,
        extra: Mapping[str, str],
    ) -> Customer:
        async with self._lock.writer_lock:
            customer_document = await self._customers_collection.find_one(
                {"id": {"$eq": customer_id}}
            )

            if not customer_document:
                raise ItemNotFoundError(item_id=UniqueId(customer_id))

            updated_extra = {**customer_document["extra"], **extra}

            result = await self._customers_collection.update_one(
                filters={"id": {"$eq": customer_id}},
                params={"extra": updated_extra},
            )

        assert result.updated_document

        return await self._deserialize_customer(customer_document=result.updated_document)

    @override
    async def remove_extra(
        self,
        customer_id: CustomerId,
        keys: Sequence[str],
    ) -> Customer:
        async with self._lock.writer_lock:
            customer_document = await self._customers_collection.find_one(
                {"id": {"$eq": customer_id}}
            )

            if not customer_document:
                raise ItemNotFoundError(item_id=UniqueId(customer_id))

            updated_extra = {k: v for k, v in customer_document["extra"].items() if k not in keys}

            result = await self._customers_collection.update_one(
                filters={"id": {"$eq": customer_id}},
                params={"extra": updated_extra},
            )

        assert result.updated_document

        return await self._deserialize_customer(customer_document=result.updated_document)
