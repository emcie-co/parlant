"""Regression for the customer pages consumed by dashboard import."""

import pytest

from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.common import IdGenerator
from parlant.core.customers import CustomerDocumentStore, CustomerId, CustomerStore


@pytest.mark.asyncio
async def test_unfiltered_customer_pages_keep_the_limit_after_the_guest_page() -> None:
    database = TransientDocumentDatabase()
    async with CustomerDocumentStore(IdGenerator(), database) as store:
        created = [await store.create_customer(name=f"Customer {n}") for n in range(8)]
        cursor = None
        seen: list[CustomerId] = []
        sizes = []
        while True:
            page = await store.list_customers(limit=3, cursor=cursor)
            assert len(page.items) <= 3
            assert page.total_count == 9
            sizes.append(len(page.items))
            seen.extend(customer.id for customer in page.items)
            if not page.has_more:
                break
            assert page.next_cursor is not None
            assert page.next_cursor != cursor
            cursor = page.next_cursor
        assert sizes == [3, 3, 3]
        assert len(seen) == len(set(seen)) == 9
        assert set(seen) == {CustomerStore.GUEST_ID, *(c.id for c in created)}


@pytest.mark.parametrize("direction", ["asc", "desc"])
@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize("writable_count", [0, 4])
async def test_composite_customer_pages_advance_without_skipping_either_store(
    direction: str, incremental: bool, writable_count: int
) -> None:
    from collections import Counter
    from datetime import datetime, timedelta, timezone

    from parlant.core.customers import CompositeCustomerStore
    from parlant.core.persistence.common import SortDirection

    async with (
        CustomerDocumentStore(IdGenerator(), TransientDocumentDatabase()) as writable,
        CustomerDocumentStore(IdGenerator(), TransientDocumentDatabase()) as readable,
    ):
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        expected = []
        for store, count, offset in [(writable, writable_count, 0), (readable, 9, 1)]:
            for n in range(count):
                customer = await store.create_customer(
                    name=f"Customer {offset}-{n}",
                    creation_utc=start + timedelta(days=2 * n + offset),
                )
                expected.append(customer.id)
                if incremental and n == 0:
                    await store.upsert_metadata(customer.id, {"updated": "yes"})
        composite = CompositeCustomerStore(writable, [readable])
        cursor = None
        seen: list[CustomerId] = []
        for _ in range(20):
            page = await composite.list_customers(
                limit=3,
                cursor=cursor,
                sort_direction=SortDirection.ASC if direction == "asc" else SortDirection.DESC,
                min_modified_utc=start if incremental else None,
            )
            seen.extend(c.id for c in page.items if c.id != CustomerStore.GUEST_ID)
            assert page.total_count == len(expected) + 1
            if not page.has_more:
                break
            assert page.next_cursor is not None
            assert page.next_cursor != cursor
            cursor = page.next_cursor
        else:
            pytest.fail("composite listing did not drain")
        assert Counter(seen) == Counter(expected)


@pytest.mark.parametrize("direction", ["asc", "desc"])
async def test_composite_cursor_uses_the_document_stores_serialized_timestamp_order(
    direction: str,
) -> None:
    from datetime import datetime

    from parlant.core.customers import CompositeCustomerStore, CustomerId
    from parlant.core.persistence.common import SortDirection

    async with CustomerDocumentStore(IdGenerator(), TransientDocumentDatabase()) as store:
        for name, timestamp in [
            ("a", "2020-01-01T10:00:00+00:00"),
            ("b", "2020-01-01T11:00:00+02:00"),
            ("c", "2020-01-01T12:00:00+00:00"),
            ("d", "2020-01-01T13:00:00"),
        ]:
            await store.create_customer(
                name=name, id=CustomerId(name), creation_utc=datetime.fromisoformat(timestamp)
            )
        composite = CompositeCustomerStore(store, [])
        seen: list[CustomerId] = []
        cursor = None
        for _ in range(5):
            page = await composite.list_customers(
                limit=3,
                cursor=cursor,
                sort_direction=SortDirection.ASC if direction == "asc" else SortDirection.DESC,
            )
            seen.extend(c.id for c in page if c.id != CustomerStore.GUEST_ID)
            if not page.has_more:
                break
            assert page.next_cursor is not None
            cursor = page.next_cursor
        assert seen == (["a", "b", "c", "d"] if direction == "asc" else ["d", "c", "b", "a"])


@pytest.mark.parametrize("limit", [None, 2])
async def test_that_a_composite_store_lists_the_guest_customer_once(limit: int | None) -> None:
    from parlant.core.customers import CompositeCustomerStore

    async with (
        CustomerDocumentStore(IdGenerator(), TransientDocumentDatabase()) as writable,
        CustomerDocumentStore(IdGenerator(), TransientDocumentDatabase()) as readable,
    ):
        for store in (writable, readable):
            for n in range(2):
                await store.create_customer(name=f"Customer {n}")
        composite = CompositeCustomerStore(writable, [readable])
        seen: list[CustomerId] = []
        cursor = None
        for _ in range(10):
            page = await composite.list_customers(limit=limit, cursor=cursor)
            assert page.total_count == 5
            seen.extend(c.id for c in page.items)
            if not page.has_more:
                break
            cursor = page.next_cursor
        else:
            pytest.fail("composite listing did not drain")
        assert seen.count(CustomerStore.GUEST_ID) == 1
        assert seen[0] == CustomerStore.GUEST_ID
        assert len(seen) == 5
