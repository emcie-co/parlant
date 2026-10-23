# Copyright 2026 Emcie Co Ltd.
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
from enum import Enum
from datetime import datetime, timezone
from itertools import chain
from typing import Iterator, Mapping, NewType, Optional, Sequence
from typing_extensions import override, TypedDict, Self

from parlant.core.async_utils import ReaderWriterLock, safe_gather
from parlant.core.persistence.document_database_helper import (
    DocumentMigrationHelper,
    DocumentStoreMigrationHelper,
)
from parlant.core.groups import GroupId
from parlant.core.common import (
    ItemNotFoundError,
    try_or_none,
    UniqueId,
    Version,
    IdGenerator,
    xxh3_checksum,
)
from parlant.core.persistence.common import Cursor, ObjectId, SortDirection, Where
from parlant.core.persistence.document_database import (
    BaseDocument,
    CollectionIndex,
    DocumentCollection,
    DocumentDatabase,
    group_ids_by_entity,
)

CustomerId = NewType("CustomerId", str)


class CustomerKind(Enum):
    """Whether a customer stands for a real person or for a test persona."""

    REAL = "real"
    """An actual customer. The default, and the reading of any absent value."""

    PERSONA = "persona"
    """A stand-in used for testing, which no end user can authenticate as."""


_GUEST_CUSTOMER_TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Customer:
    id: CustomerId
    creation_utc: datetime
    modified_utc: datetime
    name: str
    metadata: Mapping[str, str]
    groups: Sequence[GroupId]
    kind: CustomerKind = CustomerKind.REAL
    """Defaulted so that a customer built without stating a kind is a real one.

    Every path that omits it — an older document, a caller written before the
    field existed — resolves to the answer that keeps a test run away from it.
    """


@dataclass(frozen=True)
class CustomerListing:
    items: Sequence[Customer]
    total_count: int
    has_more: bool
    next_cursor: Optional[Cursor] = None

    def __iter__(self) -> Iterator[Customer]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


class CustomerUpdateParams(TypedDict, total=False):
    name: str


class CustomerStore(ABC):
    GUEST_ID = CustomerId("guest")

    @abstractmethod
    async def create_customer(
        self,
        name: str,
        metadata: Mapping[str, str] = {},
        creation_utc: Optional[datetime] = None,
        groups: Optional[Sequence[GroupId]] = None,
        id: Optional[CustomerId] = None,
        kind: CustomerKind = CustomerKind.REAL,
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
        groups: Optional[Sequence[GroupId]] = None,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
        min_modified_utc: Optional[datetime] = None,
        name_prefix: Optional[str] = None,
        kind: Optional[CustomerKind] = None,
    ) -> CustomerListing: ...

    @abstractmethod
    async def upsert_group(
        self,
        customer_id: CustomerId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_group(
        self,
        customer_id: CustomerId,
        group_id: GroupId,
    ) -> None: ...

    @abstractmethod
    async def upsert_metadata(
        self,
        customer_id: CustomerId,
        metadata: Mapping[str, str],
    ) -> Customer: ...

    @abstractmethod
    async def remove_metadata(
        self,
        customer_id: CustomerId,
        keys: Sequence[str],
    ) -> Customer: ...


class _CustomerDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    modified_utc: str
    name: str
    name_lower: str
    metadata: Mapping[str, str]
    kind: str


class _CustomerGroupAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    customer_id: CustomerId
    group_id: GroupId


def _name_prefix_filter(name_prefix: Optional[str]) -> Optional[Where]:
    """A case-insensitive prefix match on the normalized name, expressed as a
    range over ``name_lower`` so every backend can serve it with existing
    operators (and Mongo with an index)."""
    if name_prefix is None:
        return None
    prefix = name_prefix.lower()
    return {
        "$and": [
            {"name_lower": {"$gte": prefix}},
            {"name_lower": {"$lt": prefix + "\uffff"}},
        ]
    }


class CustomerDocumentStore(CustomerStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        database: DocumentDatabase,
        allow_migration: bool = False,
        collections_prefix: str | None = None,
    ) -> None:
        self._id_generator = id_generator

        self._database = database
        self._customers_collection: DocumentCollection[_CustomerDocument]
        self._tag_association_collection: DocumentCollection[_CustomerGroupAssociationDocument]

        self._allow_migration = allow_migration
        self._collections_prefix = collections_prefix

        self._lock = ReaderWriterLock()

    async def _document_loader(self, doc: BaseDocument) -> Optional[_CustomerDocument]:
        # Fresh data baseline: no historical versions. Register converters here
        # when the document schema next changes.
        loaded = await DocumentMigrationHelper[_CustomerDocument](self, {}).migrate(doc)

        if loaded is not None and "kind" not in loaded:
            # A document stored before customers had a kind. Filling it in here
            # rather than at every read means a query may filter on the field
            # without having to account for its absence: `matches_filters`
            # subscripts the candidate, so a document still missing the field
            # would raise rather than simply fail to match. Both file-backed
            # adapters persist what this returns, so the fill-in sticks.
            loaded["kind"] = CustomerKind.REAL.value

        return loaded

    async def _association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[_CustomerGroupAssociationDocument]:
        # Fresh data baseline: no historical versions. Register converters here
        # when the document schema next changes.
        return await DocumentMigrationHelper[_CustomerGroupAssociationDocument](self, {}).migrate(
            doc
        )

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
            collections_prefix=self._collections_prefix,
        ):
            self._customers_collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_customers"
                if self._collections_prefix
                else "customers",
                schema=_CustomerDocument,
                document_loader=self._document_loader,
            )

            self._tag_association_collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_customer_tag_associations"
                if self._collections_prefix
                else "customer_tag_associations",
                schema=_CustomerGroupAssociationDocument,
                document_loader=self._association_document_loader,
            )
            await self._customers_collection.ensure_indexes(
                [
                    CollectionIndex(fields=(("id", SortDirection.ASC),)),
                    CollectionIndex(fields=(("name_lower", SortDirection.ASC),)),
                    CollectionIndex(fields=(("kind", SortDirection.ASC),)),
                    CollectionIndex(
                        fields=(
                            ("modified_utc", SortDirection.ASC),
                            ("id", SortDirection.ASC),
                        )
                    ),
                ]
            )
            await self._tag_association_collection.ensure_indexes(
                [
                    CollectionIndex(fields=(("customer_id", SortDirection.ASC),)),
                    CollectionIndex(fields=(("group_id", SortDirection.ASC),)),
                    CollectionIndex(
                        fields=(
                            ("customer_id", SortDirection.ASC),
                            ("group_id", SortDirection.ASC),
                        )
                    ),
                ]
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
            modified_utc=customer.modified_utc.isoformat(),
            name=customer.name,
            name_lower=customer.name.lower(),
            metadata=customer.metadata,
            kind=customer.kind.value,
        )

    async def _deserialize_customer(self, customer_document: _CustomerDocument) -> Customer:
        groups = [
            doc["group_id"]
            for doc in await self._tag_association_collection.find(
                {"customer_id": {"$eq": customer_document["id"]}}
            )
        ]

        return self._to_customer(customer_document, groups)

    async def _deserialize_many(
        self,
        documents: Sequence[_CustomerDocument],
    ) -> Sequence[Customer]:
        """Reconstruct a whole page of entities with one association query."""
        group_ids = await group_ids_by_entity(
            self._tag_association_collection,
            entity_field="customer_id",
            value_field="group_id",
            entity_ids=[d["id"] for d in documents],
        )

        return [self._to_customer(d, group_ids[d["id"]]) for d in documents]

    def _to_customer(
        self,
        customer_document: _CustomerDocument,
        group_ids: Sequence[str],
    ) -> Customer:
        return Customer(
            id=CustomerId(customer_document["id"]),
            creation_utc=datetime.fromisoformat(customer_document["creation_utc"]),
            modified_utc=datetime.fromisoformat(customer_document["modified_utc"]),
            name=customer_document["name"],
            metadata=customer_document["metadata"],
            groups=[GroupId(group_id) for group_id in group_ids],
            # The loader fills this in for stored documents, so the fallback is
            # for a document that reached the collection without passing
            # through it. Reading an absent kind as real is the safe direction.
            kind=CustomerKind(customer_document.get("kind", CustomerKind.REAL.value)),
        )

    @override
    async def create_customer(
        self,
        name: str,
        metadata: Mapping[str, str] = {},
        creation_utc: Optional[datetime] = None,
        groups: Optional[Sequence[GroupId]] = None,
        id: Optional[CustomerId] = None,
        kind: CustomerKind = CustomerKind.REAL,
    ) -> Customer:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            # Use provided ID or generate one
            if id is not None:
                customer_id = id

                # Check if customer with this ID already exists
                existing = await self._customers_collection.find_one(
                    filters={"id": {"$eq": customer_id}}
                )
                if existing:
                    raise ValueError(f"Customer with id '{customer_id}' already exists")
            else:
                customer_checksum = xxh3_checksum(f"{name}{metadata}{groups}")
                customer_id = CustomerId(self._id_generator.generate(customer_checksum))

            customer = Customer(
                id=customer_id,
                name=name,
                metadata=metadata,
                creation_utc=creation_utc,
                modified_utc=creation_utc,
                groups=groups or [],
                kind=kind,
            )

            await self._customers_collection.insert_one(
                document=self._serialize_customer(customer=customer)
            )

            for group_id in groups or []:
                tag_checksum = xxh3_checksum(f"{customer.id}{group_id}")

                await self._tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(self._id_generator.generate(tag_checksum)),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "customer_id": customer.id,
                        "group_id": group_id,
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
                    creation_utc=_GUEST_CUSTOMER_TIME,
                    modified_utc=_GUEST_CUSTOMER_TIME,
                    metadata={},
                    groups=[],
                    # Nobody in particular, but not a persona: guest sessions
                    # carry production traffic, so a test run must not be
                    # handed one for want of a named customer.
                    kind=CustomerKind.REAL,
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

            if customer_document["name"] == params["name"]:
                return await self._deserialize_customer(customer_document=customer_document)

            result = await self._customers_collection.update_one(
                filters={"id": {"$eq": customer_id}},
                params={
                    "name": params["name"],
                    "name_lower": params["name"].lower(),
                    "modified_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

        assert result.updated_document

        return await self._deserialize_customer(customer_document=result.updated_document)

    async def list_customers(
        self,
        groups: Optional[Sequence[GroupId]] = None,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
        min_modified_utc: Optional[datetime] = None,
        name_prefix: Optional[str] = None,
        kind: Optional[CustomerKind] = None,
    ) -> CustomerListing:
        if min_modified_utc is not None:
            return await self._list_customers_by_modified_utc(
                groups=groups,
                limit=limit,
                cursor=cursor,
                sort_direction=sort_direction,
                min_modified_utc=min_modified_utc,
                name_prefix=name_prefix,
                kind=kind,
            )

        filters: Where = {}

        async with self._lock.reader_lock:
            if groups is not None:
                if len(groups) == 0:
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
                    tag_filters: Where = {"$or": [{"group_id": {"$eq": group}} for group in groups]}
                    tag_associations = await self._tag_association_collection.find(
                        filters=tag_filters
                    )
                    customer_ids = {assoc["customer_id"] for assoc in tag_associations}

                    if not customer_ids:
                        guest = await self.read_customer(CustomerStore.GUEST_ID)
                        return CustomerListing(
                            items=[guest],
                            total_count=1,
                            has_more=False,
                            next_cursor=None,
                        )

                    filters = {"$or": [{"id": {"$eq": id}} for id in customer_ids]}

            if (name_filter := _name_prefix_filter(name_prefix)) is not None:
                filters = {"$and": [filters, name_filter]} if filters else name_filter

            if kind is not None:
                kind_filter: Where = {"kind": {"$eq": kind.value}}
                filters = {"$and": [filters, kind_filter]} if filters else kind_filter

            # The guest is a real customer, so it belongs in an unfiltered
            # listing and in a listing of real customers, but never in a
            # listing of personas.
            include_guest = (
                groups is None and name_prefix is None and kind in (None, CustomerKind.REAL)
            )

            # Use the document collection's find method with pagination
            result = await self._customers_collection.find(
                filters=filters,
                limit=limit - 1
                if limit is not None and cursor is None and include_guest
                else limit,  # Adjust only the first unfiltered page for the guest.
                cursor=cursor,
                sort_direction=sort_direction,
            )

            # Convert documents to Customer objects
            customers = list(await self._deserialize_many(list(result.items)))

            # Handle guest customer and total count
            if include_guest:
                # Always include guest in total count
                total_count = result.total_count + 1

                if cursor is None:
                    # First page: add guest customer at the beginning
                    guest = await self.read_customer(CustomerStore.GUEST_ID)
                    customers = [guest] + customers
            else:
                # Filtered by groups or name: no guest customer
                total_count = result.total_count

            has_more = result.has_more

            return CustomerListing(
                items=customers,
                total_count=total_count,
                has_more=has_more,
                next_cursor=result.next_cursor,
            )

    async def _list_customers_by_modified_utc(
        self,
        groups: Optional[Sequence[GroupId]],
        limit: Optional[int],
        cursor: Optional[Cursor],
        sort_direction: Optional[SortDirection],
        min_modified_utc: datetime,
        name_prefix: Optional[str] = None,
        kind: Optional[CustomerKind] = None,
    ) -> CustomerListing:
        """Incremental customer traversal keyed on (modified_utc, id).

        This is a separate cursor mode from the store's creation-time cursor:
        pagination survives updates that land between pages because the cursor
        position is a coordinate in the (modified_utc, id) sort order, not a
        fixed offset, and it is recomputed fresh from the live data on every
        call.

        `min_modified_utc` is compared lexicographically against ISO-8601
        strings that are always serialized with a "+00:00" UTC offset, so the
        checkpoint is normalized to UTC before filtering: naive datetimes are
        treated as already being in UTC (tzinfo is attached, not converted),
        and timezone-aware datetimes are converted to UTC via `astimezone`.

        Known parity quirk (inherited from the creation-time path): on an
        unfiltered first page (`groups=None`, `cursor=None`) with `limit=1`,
        the single slot is consumed by the guest customer, so `has_more` is
        `True` but `next_cursor` is `None` even though a real customer is
        pending. Incremental consumers must use `limit >= 2`.
        """
        if min_modified_utc.tzinfo is None:
            min_modified_utc = min_modified_utc.replace(tzinfo=timezone.utc)
        else:
            min_modified_utc = min_modified_utc.astimezone(timezone.utc)

        filters: Where = {"modified_utc": {"$gte": min_modified_utc.isoformat()}}

        async with self._lock.reader_lock:
            if groups is not None:
                if len(groups) == 0:
                    customer_ids = {
                        doc["customer_id"]
                        for doc in await self._tag_association_collection.find(filters={})
                    }
                    group_filter: Where = (
                        {"$and": [{"id": {"$ne": id}} for id in customer_ids]}
                        if customer_ids
                        else {}
                    )
                else:
                    tag_filters: Where = {"$or": [{"group_id": {"$eq": group}} for group in groups]}
                    tag_associations = await self._tag_association_collection.find(
                        filters=tag_filters
                    )
                    customer_ids = {assoc["customer_id"] for assoc in tag_associations}

                    if not customer_ids:
                        return CustomerListing(
                            items=[],
                            total_count=0,
                            has_more=False,
                            next_cursor=None,
                        )

                    group_filter = {"$or": [{"id": {"$eq": id}} for id in customer_ids]}

                if group_filter:
                    filters = {"$and": [filters, group_filter]}

            if (name_filter := _name_prefix_filter(name_prefix)) is not None:
                filters = {"$and": [filters, name_filter]}

            if kind is not None:
                filters = {"$and": [filters, {"kind": {"$eq": kind.value}}]}

            all_matching = await self._customers_collection.find(filters=filters)

            direction = sort_direction or SortDirection.ASC

            sorted_docs = sorted(
                all_matching.items,
                key=lambda doc: (doc["modified_utc"], doc["id"]),
                reverse=direction == SortDirection.DESC,
            )

            if cursor is not None:
                cursor_key = (str(cursor.sort_value), str(cursor.id))

                if direction == SortDirection.DESC:
                    sorted_docs = [
                        doc for doc in sorted_docs if (doc["modified_utc"], doc["id"]) < cursor_key
                    ]
                else:
                    sorted_docs = [
                        doc for doc in sorted_docs if (doc["modified_utc"], doc["id"]) > cursor_key
                    ]

            # The guest is only ever injected on the very first, unfiltered page,
            # and it must never consume or advance the durable (modified_utc, id)
            # cursor over real customer documents.
            unfiltered = (
                groups is None and name_prefix is None and kind in (None, CustomerKind.REAL)
            )
            include_guest = unfiltered and cursor is None
            effective_limit = limit - 1 if limit is not None and include_guest else limit

            has_more = False
            if effective_limit is not None and len(sorted_docs) > effective_limit:
                has_more = True
                page_docs = sorted_docs[:effective_limit]
            else:
                page_docs = sorted_docs

            next_cursor = None
            if has_more and page_docs:
                last_doc = page_docs[-1]
                next_cursor = Cursor(
                    sort_value=str(last_doc["modified_utc"]),
                    id=ObjectId(str(last_doc["id"])),
                )

            customers = list(await self._deserialize_many(list(page_docs)))

            if unfiltered:
                total_count = all_matching.total_count + 1

                if include_guest:
                    guest = await self.read_customer(CustomerStore.GUEST_ID)
                    customers = [guest] + customers
            else:
                total_count = all_matching.total_count

            return CustomerListing(
                items=customers,
                total_count=total_count,
                has_more=has_more,
                next_cursor=next_cursor,
            )

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
    async def upsert_group(
        self,
        customer_id: CustomerId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            customer = await self.read_customer(customer_id)

            if group_id in customer.groups:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_checksum = xxh3_checksum(f"{customer_id}{group_id}")

            association_document: _CustomerGroupAssociationDocument = {
                "id": ObjectId(self._id_generator.generate(association_checksum)),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "customer_id": customer_id,
                "group_id": group_id,
            }

            _ = await self._tag_association_collection.insert_one(document=association_document)

            customer_document = await self._customers_collection.find_one(
                {"id": {"$eq": customer_id}}
            )

            if not customer_document:
                raise ItemNotFoundError(item_id=UniqueId(customer_id))

            result = await self._customers_collection.update_one(
                filters={"id": {"$eq": customer_id}},
                params={"modified_utc": datetime.now(timezone.utc).isoformat()},
            )

        assert result.updated_document

        return True

    @override
    async def remove_group(
        self,
        customer_id: CustomerId,
        group_id: GroupId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._tag_association_collection.delete_one(
                {
                    "customer_id": {"$eq": customer_id},
                    "group_id": {"$eq": group_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(group_id))

            customer_document = await self._customers_collection.find_one(
                {"id": {"$eq": customer_id}}
            )

            if not customer_document:
                raise ItemNotFoundError(item_id=UniqueId(customer_id))

            result = await self._customers_collection.update_one(
                filters={"id": {"$eq": customer_id}},
                params={"modified_utc": datetime.now(timezone.utc).isoformat()},
            )

        assert result.updated_document

        return None

    @override
    async def upsert_metadata(
        self,
        customer_id: CustomerId,
        metadata: Mapping[str, str],
    ) -> Customer:
        async with self._lock.writer_lock:
            customer_document = await self._customers_collection.find_one(
                {"id": {"$eq": customer_id}}
            )

            if not customer_document:
                raise ItemNotFoundError(item_id=UniqueId(customer_id))

            updated_metadata = {**customer_document["metadata"], **metadata}

            if updated_metadata == customer_document["metadata"]:
                return await self._deserialize_customer(customer_document=customer_document)

            result = await self._customers_collection.update_one(
                filters={"id": {"$eq": customer_id}},
                params={
                    "metadata": updated_metadata,
                    "modified_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

        assert result.updated_document

        return await self._deserialize_customer(customer_document=result.updated_document)

    @override
    async def remove_metadata(
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

            updated_metadata = {
                k: v for k, v in customer_document["metadata"].items() if k not in keys
            }

            if updated_metadata == customer_document["metadata"]:
                return await self._deserialize_customer(customer_document=customer_document)

            result = await self._customers_collection.update_one(
                filters={"id": {"$eq": customer_id}},
                params={
                    "metadata": updated_metadata,
                    "modified_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

        assert result.updated_document

        return await self._deserialize_customer(customer_document=result.updated_document)


class CompositeCustomerStore(CustomerStore):
    def __init__(
        self,
        writable_store: CustomerStore,
        readable_stores: Sequence[CustomerStore],
    ) -> None:
        self._writable_store = writable_store
        self._readable_stores = readable_stores
        self._all_stores: Sequence[CustomerStore] = [writable_store, *readable_stores]

    @override
    async def create_customer(
        self,
        name: str,
        metadata: Mapping[str, str] = {},
        creation_utc: Optional[datetime] = None,
        groups: Optional[Sequence[GroupId]] = None,
        id: Optional[CustomerId] = None,
        kind: CustomerKind = CustomerKind.REAL,
    ) -> Customer:
        return await self._writable_store.create_customer(
            name=name,
            metadata=metadata,
            creation_utc=creation_utc,
            groups=groups,
            id=id,
            kind=kind,
        )

    @override
    async def read_customer(
        self,
        customer_id: CustomerId,
    ) -> Customer:
        results = await safe_gather(
            *[try_or_none(store.read_customer(customer_id)) for store in self._all_stores]
        )
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise ItemNotFoundError(item_id=UniqueId(customer_id))
        return result

    @override
    async def update_customer(
        self,
        customer_id: CustomerId,
        params: CustomerUpdateParams,
    ) -> Customer:
        return await self._writable_store.update_customer(customer_id, params)

    @override
    async def delete_customer(
        self,
        customer_id: CustomerId,
    ) -> None:
        return await self._writable_store.delete_customer(customer_id)

    @override
    async def list_customers(
        self,
        groups: Optional[Sequence[GroupId]] = None,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
        min_modified_utc: Optional[datetime] = None,
        name_prefix: Optional[str] = None,
        kind: Optional[CustomerKind] = None,
    ) -> CustomerListing:
        results = await safe_gather(
            *[
                store.list_customers(
                    groups=groups,
                    limit=limit,
                    cursor=cursor,
                    sort_direction=sort_direction,
                    min_modified_utc=min_modified_utc,
                    name_prefix=name_prefix,
                    kind=kind,
                )
                for store in self._all_stores
            ]
        )
        all_items = list(chain.from_iterable(r.items for r in results))

        # Every backing store synthesizes its own guest customer: it lists it on
        # the first unfiltered page and counts it on every unfiltered page. The
        # merged listing keeps the first one and discounts the rest.
        if groups is None and name_prefix is None:
            first_guest = next(
                (i for i, c in enumerate(all_items) if c.id == CustomerStore.GUEST_ID), None
            )
            all_items = [
                customer
                for index, customer in enumerate(all_items)
                if customer.id != CustomerStore.GUEST_ID or index == first_guest
            ]
            duplicate_guests = len(self._all_stores) - 1
        else:
            duplicate_guests = 0

        continuations = []
        for result in results:
            if result.has_more:
                if result.next_cursor is None or result.next_cursor == cursor:
                    raise ValueError(
                        "Customer store returned more results without an advancing cursor"
                    )
                continuations.append(result.next_cursor)

        next_cursor = None
        if continuations:
            descending = sort_direction == SortDirection.DESC

            def cursor_key(position: Cursor) -> tuple[str, str]:
                return (position.sort_value, str(position.id))

            # Every store receives the same keyset cursor. Only emit the prefix
            # all stores have covered, so advancing it cannot skip rows from a
            # slower store. Rows beyond this boundary are fetched on the next page.
            next_cursor = (max if descending else min)(continuations, key=cursor_key)
            boundary = cursor_key(next_cursor)

            def covered(customer: Customer) -> bool:
                if customer.id == CustomerStore.GUEST_ID:
                    return True
                timestamp = customer.modified_utc if min_modified_utc else customer.creation_utc
                # Backing stores order serialized timestamps, including offsets.
                key = (timestamp.isoformat(), str(customer.id))
                return key >= boundary if descending else key <= boundary

            all_items = [customer for customer in all_items if covered(customer)]

        return CustomerListing(
            items=all_items,
            total_count=sum(r.total_count for r in results) - duplicate_guests,
            has_more=any(r.has_more for r in results),
            next_cursor=next_cursor,
        )

    @override
    async def upsert_group(
        self,
        customer_id: CustomerId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        return await self._writable_store.upsert_group(customer_id, group_id, creation_utc)

    @override
    async def remove_group(
        self,
        customer_id: CustomerId,
        group_id: GroupId,
    ) -> None:
        return await self._writable_store.remove_group(customer_id, group_id)

    @override
    async def upsert_metadata(
        self,
        customer_id: CustomerId,
        metadata: Mapping[str, str],
    ) -> Customer:
        return await self._writable_store.upsert_metadata(customer_id, metadata)

    @override
    async def remove_metadata(
        self,
        customer_id: CustomerId,
        keys: Sequence[str],
    ) -> Customer:
        return await self._writable_store.remove_metadata(customer_id, keys)
