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

from fastapi import status
import httpx
from lagom import Container
from pytest import raises

from parlant.core.common import ItemNotFoundError
from parlant.core.groups import GroupStore


async def test_that_a_tag_can_be_created(
    async_client: httpx.AsyncClient,
) -> None:
    name = "VIP"

    response = await async_client.post(
        "/groups",
        json={
            "name": name,
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    group = response.json()

    assert group["name"] == name
    assert "id" in group


async def test_that_a_tag_can_be_read(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    group_store = container[GroupStore]

    name = "VIP"

    group = await group_store.create_group(name)

    read_response = await async_client.get(f"/groups/{group.id}")
    assert read_response.status_code == status.HTTP_200_OK

    data = read_response.json()
    assert data["id"] == group.id
    assert data["name"] == name


async def test_that_tags_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    group_store = container[GroupStore]

    first_name = "VIP"
    second_name = "Female"

    _ = await group_store.create_group(first_name)
    _ = await group_store.create_group(second_name)

    groups = (await async_client.get("/groups")).raise_for_status().json()

    assert len(groups) == 2
    assert any(first_name == group["name"] for group in groups)
    assert any(second_name == group["name"] for group in groups)


async def test_that_tags_can_be_listed_filtered_by_name(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    group_store = container[GroupStore]

    _ = await group_store.create_group("VIP")
    _ = await group_store.create_group("Female")

    groups = (await async_client.get("/groups", params={"name": "VIP"})).raise_for_status().json()

    assert len(groups) == 1
    assert groups[0]["name"] == "VIP"


async def test_that_tags_filtered_by_nonexistent_name_returns_empty_list(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    group_store = container[GroupStore]

    _ = await group_store.create_group("VIP")

    groups = (
        (await async_client.get("/groups", params={"name": "nonexistent"}))
        .raise_for_status()
        .json()
    )

    assert groups == []


async def test_that_creating_a_tag_with_duplicate_name_raises_error(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    group_store = container[GroupStore]

    _ = await group_store.create_group("VIP")

    with raises(ValueError, match="already exists"):
        await group_store.create_group("VIP")


async def test_that_a_tag_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    group_store = container[GroupStore]

    old_name = "VIP"

    group = await group_store.create_group(old_name)

    new_name = "Alpha"
    updated_tag_dto = (
        (
            await async_client.patch(
                f"/groups/{group.id}",
                json={
                    "name": new_name,
                },
            )
        )
        .raise_for_status()
        .json()
    )

    assert updated_tag_dto["id"] == group.id
    assert updated_tag_dto["name"] == new_name


async def test_that_a_tag_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    group_store = container[GroupStore]

    name = "VIP"

    group = await group_store.create_group(name)

    await async_client.delete(f"/groups/{group.id}")

    with raises(ItemNotFoundError):
        _ = await group_store.read_group(group.id)
