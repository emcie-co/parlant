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
from parlant.core.tags import TagStore


async def test_that_a_tag_can_be_created(
    async_client: httpx.AsyncClient,
) -> None:
    name = "VIP"

    response = await async_client.post(
        "/tags",
        json={
            "name": name,
        },
    )

    assert response.status_code == status.HTTP_201_CREATED
    tag = response.json()

    assert tag["name"] == name
    assert "id" in tag


async def test_that_a_tag_can_be_read(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]

    name = "VIP"

    tag = await tag_store.create_tag(name)

    read_response = await async_client.get(f"/tags/{tag.id}")
    assert read_response.status_code == status.HTTP_200_OK

    data = read_response.json()
    assert data["id"] == tag.id
    assert data["name"] == name


async def test_that_tags_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]

    first_name = "VIP"
    second_name = "Female"

    _ = await tag_store.create_tag(first_name)
    _ = await tag_store.create_tag(second_name)

    tags = (await async_client.get("/tags")).raise_for_status().json()

    assert len(tags) == 2
    assert any(first_name == tag["name"] for tag in tags)
    assert any(second_name == tag["name"] for tag in tags)


async def test_that_a_tag_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]

    old_name = "VIP"

    tag = await tag_store.create_tag(old_name)

    new_name = "Alpha"
    updated_tag_dto = (
        (
            await async_client.patch(
                f"/tags/{tag.id}",
                json={
                    "name": new_name,
                },
            )
        )
        .raise_for_status()
        .json()
    )

    assert updated_tag_dto["id"] == tag.id
    assert updated_tag_dto["name"] == new_name


async def test_that_a_tag_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    tag_store = container[TagStore]

    name = "VIP"

    tag = await tag_store.create_tag(name)

    await async_client.delete(f"/tags/{tag.id}")

    with raises(ItemNotFoundError):
        _ = await tag_store.read_tag(tag.id)
