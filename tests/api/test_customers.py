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

from datetime import datetime, timedelta, timezone

import dateutil.parser
from fastapi import status
import httpx
from parlant.core.container import Container
from pytest import raises

from parlant.core.application import Application
from parlant.core.common import ItemNotFoundError
from parlant.core.customers import CustomerId, CustomerKind, CustomerStore
from parlant.core.groups import GroupStore
from parlant.core.persistence.common import Cursor, SortDirection


async def test_that_a_customer_can_be_created(
    async_client: httpx.AsyncClient,
) -> None:
    name = "John Doe"
    metadata = {"email": "john@gmail.com"}

    response = await async_client.post(
        "/customers",
        json={
            "name": name,
            "metadata": metadata,
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    customer = response.json()
    assert customer["name"] == name
    assert customer["metadata"] == metadata
    assert "id" in customer
    assert "creation_utc" in customer


async def test_that_a_customer_can_be_created_with_tags(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    group_store = container[GroupStore]
    group1 = await group_store.create_group("group1")
    group2 = await group_store.create_group("group2")

    response = await async_client.post(
        "/customers",
        json={
            "name": "John Doe",
            "groups": [group1.id, group1.id, group2.id],
        },
    )
    assert response.status_code == status.HTTP_201_CREATED

    customer_dto = (
        (await async_client.get(f"/customers/{response.json()['id']}")).raise_for_status().json()
    )

    assert len(customer_dto["groups"]) == 2
    assert set(customer_dto["groups"]) == {group1.id, group2.id}


async def test_that_a_customer_can_be_read(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    name = "Menachem Brich"
    metadata = {"id": str(102938485)}

    customer = await customer_store.create_customer(name, metadata)

    read_response = await async_client.get(f"/customers/{customer.id}")
    assert read_response.status_code == status.HTTP_200_OK

    data = read_response.json()
    assert data["id"] == customer.id
    assert data["name"] == name
    assert data["metadata"] == metadata
    assert dateutil.parser.parse(data["creation_utc"]) == customer.creation_utc


async def test_that_all_customers_including_guests_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    first_name = "YamChuk"
    first_metadata = {"address": "Hawaii"}

    second_name = "DorZo"
    second_metadata = {"address": "Alaska"}

    await customer_store.create_customer(
        name=first_name,
        metadata=first_metadata,
    )

    await customer_store.create_customer(
        name=second_name,
        metadata=second_metadata,
    )

    customers = (await async_client.get("/customers")).raise_for_status().json()

    assert len(customers) == 3
    assert any(
        first_name == customer["name"] and first_metadata == customer["metadata"]
        for customer in customers
    )
    assert any(
        second_name == customer["name"] and second_metadata == customer["metadata"]
        for customer in customers
    )
    assert any("Guest" == customer["name"] for customer in customers)


async def test_that_a_customer_can_be_updated_with_a_new_name(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    name = "Original Name"
    metadata = {"role": "customer"}

    customer = await customer_store.create_customer(name=name, metadata=metadata)

    new_name = "Updated Name"

    customer_dto = (
        (
            await async_client.patch(
                f"/customers/{customer.id}",
                json={
                    "name": new_name,
                },
            )
        )
        .raise_for_status()
        .json()
    )

    assert customer_dto["name"] == new_name
    assert customer_dto["metadata"] == metadata


async def test_that_a_customer_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    name = "Original Name"

    customer = await customer_store.create_customer(name=name)

    delete_response = await async_client.delete(f"/customers/{customer.id}")
    assert delete_response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await customer_store.read_customer(customer.id)


async def test_that_a_tag_can_be_added(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    customer_store = container[CustomerStore]
    group_store = container[GroupStore]

    group = await group_store.create_group(name="VIP")

    name = "Grouped Customer"

    customer = await customer_store.create_customer(name=name)

    update_response = await async_client.patch(
        f"/customers/{customer.id}",
        json={
            "groups": {"add": [group.id]},
        },
    )
    assert update_response.status_code == status.HTTP_200_OK

    updated_customer = await customer_store.read_customer(customer.id)
    assert group.id in updated_customer.groups


async def test_that_a_tag_can_be_removed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    customer_store = container[CustomerStore]
    group_store = container[GroupStore]

    group = await group_store.create_group(name="VIP")

    name = "Grouped Customer"

    customer = await customer_store.create_customer(name=name)

    await customer_store.upsert_group(customer_id=customer.id, group_id=group.id)

    update_response = await async_client.patch(
        f"/customers/{customer.id}",
        json={
            "groups": {"remove": [group.id]},
        },
    )
    assert update_response.status_code == status.HTTP_200_OK

    updated_customer = await customer_store.read_customer(customer.id)
    assert group.id not in updated_customer.groups


async def test_that_metadata_can_be_set(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    customer_store = container[CustomerStore]
    name = "Customer with metadatas"

    customer = await customer_store.create_customer(name=name)

    new_metadata = {"department": "sales"}

    update_response = await async_client.patch(
        f"/customers/{customer.id}",
        json={
            "metadata": {"set": new_metadata},
        },
    )
    assert update_response.status_code == status.HTTP_200_OK

    updated_customer = await customer_store.read_customer(customer.id)
    assert updated_customer.metadata.get("department") == "sales"


async def test_that_metadata_can_be_unset(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    customer_store = container[CustomerStore]
    name = "Customer with metadatas"

    customer = await customer_store.create_customer(name=name, metadata={"department": "sales"})

    update_response = await async_client.patch(
        f"/customers/{customer.id}",
        json={
            "metadata": {"unset": ["department"]},
        },
    )
    assert update_response.status_code == status.HTTP_200_OK

    updated_customer = await customer_store.read_customer(customer.id)
    assert "department" not in updated_customer.metadata


async def test_that_adding_nonexistent_group_to_customer_returns_404(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    customer = await customer_store.create_customer("test_customer")

    response = await async_client.patch(
        f"/customers/{customer.id}",
        json={"groups": {"add": ["nonexistent_group"]}},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_a_customer_can_be_created_with_custom_id(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    custom_id = "my_custom_customer_id"
    name = "Custom ID Customer"
    metadata = {"source": "api_test"}

    response = await async_client.post(
        "/customers",
        json={
            "name": name,
            "metadata": metadata,
            "id": custom_id,
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    customer = response.json()
    assert customer["id"] == custom_id
    assert customer["name"] == name
    assert customer["metadata"] == metadata


async def test_that_multiple_customers_can_be_created_with_different_custom_ids(
    async_client: httpx.AsyncClient,
) -> None:
    # Create first customer with custom ID
    response1 = await async_client.post(
        "/customers",
        json={
            "name": "First Customer",
            "id": "customer_1",
        },
    )
    assert response1.status_code == status.HTTP_201_CREATED
    assert response1.json()["id"] == "customer_1"

    # Create second customer with different custom ID
    response2 = await async_client.post(
        "/customers",
        json={
            "name": "Second Customer",
            "id": "customer_2",
        },
    )
    assert response2.status_code == status.HTTP_201_CREATED
    assert response2.json()["id"] == "customer_2"


async def test_that_creating_customer_with_duplicate_custom_id_fails(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    custom_id = CustomerId("duplicate_customer_id")

    # Create first customer with custom ID
    response1 = await async_client.post(
        "/customers",
        json={
            "name": "First Customer",
            "id": custom_id,
        },
    )
    assert response1.status_code == status.HTTP_201_CREATED
    assert response1.json()["id"] == custom_id

    # Try to create second customer with same ID at the store level - should fail
    customer_store = container[CustomerStore]
    with raises(ValueError, match="already exists"):
        await customer_store.create_customer(
            name="Second Customer",
            id=custom_id,
        )


async def test_that_list_customers_can_be_paginated(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    # Create several customers to test pagination
    customers = []
    for i in range(5):
        customer = await customer_store.create_customer(
            name=f"Customer_{i}",
            metadata={"order": str(i)},
        )
        customers.append(customer)

    # Test first page with limit
    response = await async_client.get("/customers?limit=3&sort=asc")
    assert response.status_code == status.HTTP_200_OK

    first_page = response.json()
    assert len(first_page["items"]) == 3
    assert first_page["total_count"] == 6  # 5 created + 1 guest
    assert first_page["has_more"] is True
    assert first_page["next_cursor"] is not None
    # Test second page using cursor
    next_cursor = first_page["next_cursor"]
    response = await async_client.get(f"/customers?limit=3&cursor={next_cursor}&sort=asc")
    assert response.status_code == status.HTTP_200_OK

    second_page = response.json()
    assert len(second_page["items"]) == 3
    # Note: total_count behavior on subsequent pages may differ from first page
    assert second_page["has_more"] is False
    assert second_page["next_cursor"] is None
    # Test descending sort
    response = await async_client.get("/customers?limit=2&sort=desc")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert len(data["items"]) == 2
    assert data["has_more"] is True


async def test_that_list_customers_pagination_with_invalid_cursor(
    async_client: httpx.AsyncClient,
) -> None:
    # Test with invalid cursor
    response = await async_client.get("/customers?cursor=invalid_cursor")
    assert response.status_code == status.HTTP_200_OK

    # Should return results as if no cursor was provided, which in our case one customer (the guest)
    data = response.json()
    assert len(data) == 1


async def test_that_customer_modified_utc_is_initialized_from_creation_time(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    creation_utc = datetime(2024, 1, 1, tzinfo=timezone.utc)

    customer = await customer_store.create_customer(
        name="Time Customer",
        creation_utc=creation_utc,
    )

    assert customer.modified_utc == customer.creation_utc


async def test_that_customer_name_update_advances_modified_utc(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    creation_utc = datetime(2024, 1, 1, tzinfo=timezone.utc)

    customer = await customer_store.create_customer(
        name="Original Name",
        creation_utc=creation_utc,
    )

    updated = await customer_store.update_customer(customer.id, {"name": "Updated"})

    assert updated.modified_utc > customer.modified_utc

    no_op = await customer_store.update_customer(customer.id, {"name": "Updated"})

    assert no_op.modified_utc == updated.modified_utc


async def test_that_customer_extra_updates_advance_modified_utc(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    creation_utc = datetime(2024, 1, 1, tzinfo=timezone.utc)

    customer = await customer_store.create_customer(
        name="Extra Customer",
        creation_utc=creation_utc,
    )

    updated = await customer_store.upsert_metadata(customer.id, {"department": "sales"})

    assert updated.modified_utc > customer.modified_utc

    no_op = await customer_store.upsert_metadata(customer.id, {"department": "sales"})

    assert no_op.modified_utc == updated.modified_utc


async def test_that_removing_absent_extra_keys_does_not_advance_modified_utc(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    creation_utc = datetime(2024, 1, 1, tzinfo=timezone.utc)

    customer = await customer_store.create_customer(
        name="Extra Customer",
        creation_utc=creation_utc,
        metadata={"department": "sales"},
    )

    no_op = await customer_store.remove_metadata(customer.id, ["nonexistent_key"])

    assert no_op.modified_utc == customer.modified_utc


async def test_that_customer_group_updates_advance_modified_utc(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]
    group_store = container[GroupStore]

    group = await group_store.create_group(name="VIP Group")

    creation_utc = datetime(2024, 1, 1, tzinfo=timezone.utc)

    customer = await customer_store.create_customer(
        name="Grouped Customer",
        creation_utc=creation_utc,
    )

    added = await customer_store.upsert_group(customer_id=customer.id, group_id=group.id)
    assert added is True

    after_add = await customer_store.read_customer(customer.id)
    assert after_add.modified_utc > customer.modified_utc

    no_op = await customer_store.upsert_group(customer_id=customer.id, group_id=group.id)
    assert no_op is False

    after_noop = await customer_store.read_customer(customer.id)
    assert after_noop.modified_utc == after_add.modified_utc


async def test_that_stable_customer_listing_paginates_customers_with_identical_modified_utc(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    tie_utc = datetime(2024, 6, 1, tzinfo=timezone.utc)

    first = await customer_store.create_customer(
        name="Tie A",
        creation_utc=tie_utc,
        id=CustomerId("tie-customer-a"),
    )
    second = await customer_store.create_customer(
        name="Tie B",
        creation_utc=tie_utc,
        id=CustomerId("tie-customer-b"),
    )

    assert first.modified_utc == second.modified_utc

    page = await customer_store.list_customers(
        limit=1,
        min_modified_utc=tie_utc,
        groups=[],
        sort_direction=SortDirection.ASC,
    )
    assert page.next_cursor is not None

    next_page = await customer_store.list_customers(
        limit=1,
        cursor=page.next_cursor,
        min_modified_utc=tie_utc,
        groups=[],
        sort_direction=SortDirection.ASC,
    )

    drained_ids = [customer.id for customer in [*page.items, *next_page.items]]
    assert drained_ids == [first.id, second.id]
    assert next_page.has_more is False
    assert next_page.next_cursor is None


async def test_that_stable_customer_listing_advances_past_customers_updated_between_pages(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    checkpoint = datetime(2020, 1, 1, tzinfo=timezone.utc)

    customer_a = await customer_store.create_customer(
        name="A",
        creation_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        id=CustomerId("drain-customer-a"),
    )
    customer_b = await customer_store.create_customer(
        name="B",
        creation_utc=datetime(2020, 1, 2, tzinfo=timezone.utc),
        id=CustomerId("drain-customer-b"),
    )
    customer_c = await customer_store.create_customer(
        name="C",
        creation_utc=datetime(2020, 1, 3, tzinfo=timezone.utc),
        id=CustomerId("drain-customer-c"),
    )

    page1 = await customer_store.list_customers(
        limit=1,
        min_modified_utc=checkpoint,
        groups=[],
        sort_direction=SortDirection.ASC,
    )
    assert [customer.id for customer in page1.items] == [customer_a.id]
    assert page1.next_cursor is not None

    # customer_a is updated between pages, advancing its modified_utc past b and c.
    await customer_store.update_customer(customer_a.id, {"name": "A Updated"})

    drained_ids: list[CustomerId] = [customer.id for customer in page1.items]
    cursor: Cursor | None = page1.next_cursor

    for _ in range(10):
        page = await customer_store.list_customers(
            limit=1,
            cursor=cursor,
            min_modified_utc=checkpoint,
            groups=[],
            sort_direction=SortDirection.ASC,
        )
        drained_ids.extend(customer.id for customer in page.items)
        cursor = page.next_cursor

        if not page.has_more:
            break

    # b and c were never modified, so each must appear exactly once.
    assert drained_ids.count(customer_b.id) == 1
    assert drained_ids.count(customer_c.id) == 1
    # a was modified mid-drain, so it may legitimately be seen again at its new position.
    assert drained_ids.count(customer_a.id) >= 1
    assert drained_ids[-1] == customer_a.id
    assert set(drained_ids) == {customer_a.id, customer_b.id, customer_c.id}
    assert cursor is None


async def test_that_min_modified_utc_filters_out_customers_modified_before_the_checkpoint(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    checkpoint = datetime(2022, 1, 1, tzinfo=timezone.utc)

    customer_before = await customer_store.create_customer(
        name="Before",
        creation_utc=datetime(2021, 12, 31, tzinfo=timezone.utc),
        id=CustomerId("checkpoint-customer-before"),
    )
    customer_at = await customer_store.create_customer(
        name="At",
        creation_utc=checkpoint,
        id=CustomerId("checkpoint-customer-at"),
    )
    customer_after = await customer_store.create_customer(
        name="After",
        creation_utc=datetime(2022, 1, 2, tzinfo=timezone.utc),
        id=CustomerId("checkpoint-customer-after"),
    )

    result = await customer_store.list_customers(
        limit=10,
        min_modified_utc=checkpoint,
        groups=[],
    )

    ids = {customer.id for customer in result.items}
    assert customer_before.id not in ids
    assert ids == {customer_at.id, customer_after.id}


async def test_that_stable_customer_listing_supports_ascending_and_descending_traversal(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    checkpoint = datetime(2019, 1, 1, tzinfo=timezone.utc)

    customer_1 = await customer_store.create_customer(
        name="One",
        creation_utc=datetime(2019, 1, 1, tzinfo=timezone.utc),
        id=CustomerId("direction-customer-1"),
    )
    customer_2 = await customer_store.create_customer(
        name="Two",
        creation_utc=datetime(2019, 1, 2, tzinfo=timezone.utc),
        id=CustomerId("direction-customer-2"),
    )
    customer_3 = await customer_store.create_customer(
        name="Three",
        creation_utc=datetime(2019, 1, 3, tzinfo=timezone.utc),
        id=CustomerId("direction-customer-3"),
    )

    ascending = await customer_store.list_customers(
        limit=10,
        min_modified_utc=checkpoint,
        groups=[],
        sort_direction=SortDirection.ASC,
    )
    assert [customer.id for customer in ascending.items] == [
        customer_1.id,
        customer_2.id,
        customer_3.id,
    ]

    descending = await customer_store.list_customers(
        limit=10,
        min_modified_utc=checkpoint,
        groups=[],
        sort_direction=SortDirection.DESC,
    )
    assert [customer.id for customer in descending.items] == [
        customer_3.id,
        customer_2.id,
        customer_1.id,
    ]

    # Confirm paginated descending traversal also honors the reversed tiebreak.
    drained_ids: list[CustomerId] = []
    cursor = None
    for _ in range(10):
        page = await customer_store.list_customers(
            limit=1,
            cursor=cursor,
            min_modified_utc=checkpoint,
            groups=[],
            sort_direction=SortDirection.DESC,
        )
        drained_ids.extend(customer.id for customer in page.items)
        cursor = page.next_cursor

        if not page.has_more:
            break

    assert drained_ids == [customer_3.id, customer_2.id, customer_1.id]


async def test_that_stable_customer_listing_preserves_the_guest_only_on_the_first_page(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    checkpoint = datetime(2022, 1, 1, tzinfo=timezone.utc)

    excluded = await customer_store.create_customer(
        name="Excluded",
        creation_utc=datetime(2021, 1, 1, tzinfo=timezone.utc),
        id=CustomerId("guest-page-customer-excluded"),
    )
    customer_a = await customer_store.create_customer(
        name="A",
        creation_utc=datetime(2022, 1, 2, tzinfo=timezone.utc),
        id=CustomerId("guest-page-customer-a"),
    )
    customer_b = await customer_store.create_customer(
        name="B",
        creation_utc=datetime(2022, 1, 3, tzinfo=timezone.utc),
        id=CustomerId("guest-page-customer-b"),
    )

    page1 = await customer_store.list_customers(
        limit=2,
        min_modified_utc=checkpoint,
        sort_direction=SortDirection.ASC,
    )
    assert [customer.id for customer in page1.items] == [
        CustomerStore.GUEST_ID,
        customer_a.id,
    ]
    assert page1.has_more is True
    assert page1.next_cursor is not None

    page2 = await customer_store.list_customers(
        limit=2,
        cursor=page1.next_cursor,
        min_modified_utc=checkpoint,
        sort_direction=SortDirection.ASC,
    )
    assert [customer.id for customer in page2.items] == [customer_b.id]
    assert page2.has_more is False
    assert page2.next_cursor is None
    assert excluded.id not in (
        [customer.id for customer in page1.items] + [customer.id for customer in page2.items]
    )


async def test_that_customer_limit_above_one_hundred_raises_value_error(
    container: Container,
) -> None:
    application = container[Application]

    with raises(ValueError, match="limit"):
        await application.customers.find(limit=101)


async def test_that_customer_limit_below_one_raises_value_error(
    container: Container,
) -> None:
    application = container[Application]

    with raises(ValueError, match="limit"):
        await application.customers.find(limit=0)


async def test_that_min_modified_utc_checkpoint_normalizes_naive_and_offset_timezones_to_utc(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    # Stored instant: 2022-01-01T03:00:00+00:00.
    stored_instant_utc = datetime(2022, 1, 1, 3, 0, 0, tzinfo=timezone.utc)

    customer = await customer_store.create_customer(
        name="Checkpoint Customer",
        creation_utc=stored_instant_utc,
        id=CustomerId("checkpoint-tz-customer"),
    )

    # A checkpoint expressed with a +03:00 offset that names the exact same
    # instant as stored_instant_utc (03:00 UTC == 06:00 at UTC+3). Without
    # normalizing to UTC first, a naive lexicographic string comparison of
    # "...T06:00:00+03:00" against "...T03:00:00+00:00" would wrongly treat
    # the checkpoint as later than the stored instant and exclude it.
    offset_checkpoint = datetime(2022, 1, 1, 6, 0, 0, tzinfo=timezone(timedelta(hours=3)))
    assert offset_checkpoint.astimezone(timezone.utc) == stored_instant_utc

    offset_result = await customer_store.list_customers(
        limit=10,
        min_modified_utc=offset_checkpoint,
        groups=[],
    )
    assert customer.id in {c.id for c in offset_result.items}

    # A naive checkpoint equal to the same instant (naive is treated as UTC)
    # must behave identically to passing the aware UTC form directly.
    naive_checkpoint = datetime(2022, 1, 1, 3, 0, 0)
    assert naive_checkpoint.tzinfo is None

    naive_result = await customer_store.list_customers(
        limit=10,
        min_modified_utc=naive_checkpoint,
        groups=[],
    )
    aware_result = await customer_store.list_customers(
        limit=10,
        min_modified_utc=stored_instant_utc,
        groups=[],
    )

    assert {c.id for c in naive_result.items} == {c.id for c in aware_result.items}
    assert customer.id in {c.id for c in naive_result.items}


async def test_that_min_modified_utc_listing_respects_group_filters(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]
    group_store = container[GroupStore]

    group = await group_store.create_group(name="VIP Group")
    other_group = await group_store.create_group(name="Other Group")

    checkpoint = datetime(2022, 1, 1, tzinfo=timezone.utc)

    grouped_customer = await customer_store.create_customer(
        name="Grouped",
        creation_utc=datetime(2022, 1, 2, tzinfo=timezone.utc),
        id=CustomerId("group-filter-customer-grouped"),
        groups=[group.id],
    )
    await customer_store.create_customer(
        name="Ungrouped",
        creation_utc=datetime(2022, 1, 3, tzinfo=timezone.utc),
        id=CustomerId("group-filter-customer-ungrouped"),
    )

    # Inclusion branch: filtering by the customer's own group returns exactly
    # that customer, not the guest and not the ungrouped customer.
    matching = await customer_store.list_customers(
        limit=10,
        min_modified_utc=checkpoint,
        groups=[group.id],
    )
    assert [c.id for c in matching.items] == [grouped_customer.id]
    assert matching.total_count == 1
    assert matching.has_more is False

    # No-match branch: filtering by a group nobody belongs to returns an
    # empty listing, not a fallback to the guest customer.
    non_matching = await customer_store.list_customers(
        limit=10,
        min_modified_utc=checkpoint,
        groups=[other_group.id],
    )
    assert non_matching.items == []
    assert non_matching.total_count == 0
    assert non_matching.has_more is False


async def test_that_customers_can_be_searched_by_name_prefix(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    jane = await customer_store.create_customer(name="Jane Doe")
    janet = await customer_store.create_customer(name="Janet Smith")
    await customer_store.create_customer(name="Bob Stone")

    result = await customer_store.list_customers(name_prefix="jan")

    assert {customer.id for customer in result.items} == {jane.id, janet.id}
    assert result.total_count == 2


async def test_that_name_prefix_search_is_case_insensitive(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    jane = await customer_store.create_customer(name="jane doe")
    await customer_store.create_customer(name="Bob Stone")

    result = await customer_store.list_customers(name_prefix="JAN")

    assert [customer.id for customer in result.items] == [jane.id]


async def test_that_a_name_prefix_search_with_no_matches_returns_an_empty_listing(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    await customer_store.create_customer(name="Jane Doe")

    result = await customer_store.list_customers(name_prefix="zzz")

    assert result.items == []
    assert result.total_count == 0
    assert result.has_more is False


async def test_that_a_name_prefix_search_does_not_inject_the_guest_customer(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    await customer_store.create_customer(name="Guenther Ott")

    result = await customer_store.list_customers(name_prefix="gue")

    assert [customer.name for customer in result.items] == ["Guenther Ott"]
    assert all(customer.id != CustomerStore.GUEST_ID for customer in result.items)


async def test_that_a_name_prefix_search_can_be_paginated(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    for i in range(5):
        await customer_store.create_customer(name=f"Prefixed {i}")
    await customer_store.create_customer(name="Unrelated")

    first_page = await customer_store.list_customers(
        name_prefix="prefixed",
        limit=3,
        sort_direction=SortDirection.ASC,
    )

    assert len(first_page.items) == 3
    assert first_page.total_count == 5
    assert first_page.has_more is True
    assert first_page.next_cursor is not None

    second_page = await customer_store.list_customers(
        name_prefix="prefixed",
        limit=3,
        cursor=first_page.next_cursor,
        sort_direction=SortDirection.ASC,
    )

    assert len(second_page.items) == 2
    assert second_page.has_more is False

    all_names = [c.name for c in [*first_page.items, *second_page.items]]
    assert sorted(all_names) == [f"Prefixed {i}" for i in range(5)]


async def test_that_customers_can_be_filtered_by_name_prefix_via_the_api(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    await customer_store.create_customer(name="Alice Green")
    await customer_store.create_customer(name="Alicia Stone")
    await customer_store.create_customer(name="Bob Brown")

    response = await async_client.get("/customers?name_prefix=ali")
    assert response.status_code == status.HTTP_200_OK

    names = {customer["name"] for customer in response.json()}
    assert names == {"Alice Green", "Alicia Stone"}

    paginated_response = await async_client.get("/customers?name_prefix=ali&limit=1&sort=asc")
    assert paginated_response.status_code == status.HTTP_200_OK

    page = paginated_response.json()
    assert len(page["items"]) == 1
    assert page["total_count"] == 2
    assert page["has_more"] is True


async def test_that_a_persona_can_be_created_through_the_api(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/customers",
        json={
            "name": "Test Persona",
            "kind": "persona",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["kind"] == "persona"


async def test_that_a_customer_created_through_the_api_is_real_by_default(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post("/customers", json={"name": "Jane Doe"})

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["kind"] == "real"


async def test_that_a_customers_kind_cannot_be_patched(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    """The update endpoint carries no kind field, so a kind sent to it does nothing.

    No DTO in this API forbids extra fields, so the request is accepted and the
    unknown key is dropped. What this test pins is the part that matters for
    safety: whatever the caller sends, the customer stays real. A persona
    cannot be conjured out of a real customer through the update path.

    The weaker half of this is the status code. Rejecting the request outright
    would tell an integrator that their intent did not land, but it would make
    this one endpoint stricter than every other one in the API.
    """
    customer = await container[CustomerStore].create_customer(name="Real Customer")

    await async_client.patch(
        f"/customers/{customer.id}",
        json={"name": "Renamed", "kind": "persona"},
    )

    updated = await container[CustomerStore].read_customer(customer.id)

    assert updated.name == "Renamed"
    assert updated.kind is CustomerKind.REAL


async def test_that_customers_can_be_listed_by_kind(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    real = await customer_store.create_customer(name="Real Customer")
    persona = await customer_store.create_customer(name="Persona", kind=CustomerKind.PERSONA)

    response = await async_client.get("/customers", params={"kind": "persona"})

    assert response.status_code == status.HTTP_200_OK

    listed = [customer["id"] for customer in response.json()]

    assert persona.id in listed
    assert real.id not in listed
    assert CustomerStore.GUEST_ID not in listed
