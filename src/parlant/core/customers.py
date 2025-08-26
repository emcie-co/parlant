# Copyright 2025 Emcie Co Ltd.
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
from typing import Mapping, NewType, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper
from parlant.core.tags import TagId
from parlant.core.common import ItemNotFoundError, UniqueId, Version, IdGenerator, md5_checksum
from parlant.core.persistence.common import ObjectId, Where
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
)

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
        tags: Optional[Sequence[TagId]] = None,
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
        tags: Optional[Sequence[TagId]] = None,
    ) -> Sequence[Customer]: ...

    @abstractmethod
    async def upsert_tag(
        self,
        customer_id: CustomerId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_tag(
        self,
        customer_id: CustomerId,
        tag_id: TagId,
    ) -> None: ...

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
        id_generator: IdGenerator,
        database: DocumentDatabase,
        allow_migration: bool = False,
    ) -> None:
        self._id_generator = id_generator

        self._database = database
        self._customers_collection: DocumentCollection[_CustomerDocument]
        self._tag_association_collection: DocumentCollection[_CustomerTagAssociationDocument]

        self._allow_migration = allow_migration

        self._lock = ReaderWriterLock()

    async def _document_loader(self, doc: BaseDocument) -> Optional[_CustomerDocument]:
        if doc["version"] == "0.1.0":
            return cast(_CustomerDocument, doc)

        return None

    async def _association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[_CustomerTagAssociationDocument]:
        if doc["version"] == "0.1.0":
            doc = cast(_CustomerTagAssociationDocument, doc)
            return _CustomerTagAssociationDocument(
                id=doc["id"],
                version=Version.String("0.2.0"),
                creation_utc=doc["creation_utc"],
                customer_id=doc["customer_id"],
                tag_id=doc["tag_id"],
            )

        if doc["version"] == "0.2.0":
            return cast(_CustomerTagAssociationDocument, doc)

        return None

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
        ):
            self._customers_collection = await self._database.get_or_create_collection(
                name="customers",
                schema=_CustomerDocument,
                document_loader=self._document_loader,
            )

            self._tag_association_collection = await self._database.get_or_create_collection(
                name="customer_tag_associations",
                schema=_CustomerTagAssociationDocument,
                document_loader=self._association_document_loader,
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
            for doc in await self._tag_association_collection.find(
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
        tags: Optional[Sequence[TagId]] = None,
    ) -> Customer:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            customer_checksum = md5_checksum(f"{name}{extra}{tags}")

            customer = Customer(
                id=CustomerId(self._id_generator.generate(customer_checksum)),
                name=name,
                extra=extra,
                creation_utc=creation_utc,
                tags=tags or [],
            )

            await self._customers_collection.insert_one(
                document=self._serialize_customer(customer=customer)
            )

            for tag_id in tags or []:
                tag_checksum = md5_checksum(f"{customer.id}{tag_id}")

                await self._tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(self._id_generator.generate(tag_checksum)),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "customer_id": customer.id,
                        "tag_id": tag_id,
                    }
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
                    name="Guest",
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
        tags: Optional[Sequence[TagId]] = None,
    ) -> Sequence[Customer]:
        filters: Where = {}

        async with self._lock.reader_lock:
            if tags is not None:
                if len(tags) == 0:
                    customer_ids = {
                        doc["customer_id"]
                        for doc in await self._tag_association_collection.find(filters={})
                    }
                    filters = (
                        {"$and": [{"id": {"$ne": id}} for id in customer_ids]}
                        if customer_ids
                        else {}
                    )
                else:
                    tag_filters: Where = {"$or": [{"tag_id": {"$eq": tag}} for tag in tags]}
                    tag_associations = await self._tag_association_collection.find(
                        filters=tag_filters
                    )
                    customer_ids = {assoc["customer_id"] for assoc in tag_associations}

                    if not customer_ids:
                        return [await self.read_customer(CustomerStore.GUEST_ID)]

                    filters = {"$or": [{"id": {"$eq": id}} for id in customer_ids]}

            return [await self.read_customer(CustomerStore.GUEST_ID)] + [
                await self._deserialize_customer(c)
                for c in await self._customers_collection.find(filters=filters)
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
    async def upsert_tag(
        self,
        customer_id: CustomerId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            customer = await self.read_customer(customer_id)

            if tag_id in customer.tags:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_checksum = md5_checksum(f"{customer_id}{tag_id}")

            association_document: _CustomerTagAssociationDocument = {
                "id": ObjectId(self._id_generator.generate(association_checksum)),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "customer_id": customer_id,
                "tag_id": tag_id,
            }

            _ = await self._tag_association_collection.insert_one(document=association_document)

            customer_document = await self._customers_collection.find_one(
                {"id": {"$eq": customer_id}}
            )

        if not customer_document:
            raise ItemNotFoundError(item_id=UniqueId(customer_id))

        return True

    @override
    async def remove_tag(
        self,
        customer_id: CustomerId,
        tag_id: TagId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._tag_association_collection.delete_one(
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

        return None

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
