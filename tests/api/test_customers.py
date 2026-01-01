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

import dateutil.parser
from fastapi import status
import httpx
from lagom import Container
from pytest import raises

from parlant.core.common import ItemNotFoundError
from parlant.core.customers import CustomerId, CustomerStore
from parlant.core.tags import TagStore


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
    tag_store = container[TagStore]
    tag1 = await tag_store.create_tag("tag1")
    tag2 = await tag_store.create_tag("tag2")

    response = await async_client.post(
        "/customers",
        json={
            "name": "John Doe",
            "tags": [tag1.id, tag1.id, tag2.id],
        },
    )
    assert response.status_code == status.HTTP_201_CREATED

    customer_dto = (
        (await async_client.get(f"/customers/{response.json()['id']}")).raise_for_status().json()
    )

    assert len(customer_dto["tags"]) == 2
    assert set(customer_dto["tags"]) == {tag1.id, tag2.id}


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
        extra=first_metadata,
    )

    await customer_store.create_customer(
        name=second_name,
        extra=second_metadata,
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

    customer = await customer_store.create_customer(name=name, extra=metadata)

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
    tag_store = container[TagStore]

    tag = await tag_store.create_tag(name="VIP")

    name = "Tagged Customer"

    customer = await customer_store.create_customer(name=name)

    update_response = await async_client.patch(
        f"/customers/{customer.id}",
        json={
            "tags": {"add": [tag.id]},
        },
    )
    assert update_response.status_code == status.HTTP_200_OK

    updated_customer = await customer_store.read_customer(customer.id)
    assert tag.id in updated_customer.tags


async def test_that_a_tag_can_be_removed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    customer_store = container[CustomerStore]
    tag_store = container[TagStore]

    tag = await tag_store.create_tag(name="VIP")

    name = "Tagged Customer"

    customer = await customer_store.create_customer(name=name)

    await customer_store.upsert_tag(customer_id=customer.id, tag_id=tag.id)

    update_response = await async_client.patch(
        f"/customers/{customer.id}",
        json={
            "tags": {"remove": [tag.id]},
        },
    )
    assert update_response.status_code == status.HTTP_200_OK

    updated_customer = await customer_store.read_customer(customer.id)
    assert tag.id not in updated_customer.tags


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
    assert updated_customer.extra.get("department") == "sales"


async def test_that_metadata_can_be_unset(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    customer_store = container[CustomerStore]
    name = "Customer with metadatas"

    customer = await customer_store.create_customer(name=name, extra={"department": "sales"})

    update_response = await async_client.patch(
        f"/customers/{customer.id}",
        json={
            "metadata": {"unset": ["department"]},
        },
    )
    assert update_response.status_code == status.HTTP_200_OK

    updated_customer = await customer_store.read_customer(customer.id)
    assert "department" not in updated_customer.extra


async def test_that_adding_nonexistent_tag_to_customer_returns_404(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    customer = await customer_store.create_customer("test_customer")

    response = await async_client.patch(
        f"/customers/{customer.id}",
        json={"tags": {"add": ["nonexistent_tag"]}},
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
            extra={"order": str(i)},
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

    # Should return results as if no cursor was provided
    data = response.json()
    assert "items" in data
    assert data["total_count"] >= 1  # At least guest customer
