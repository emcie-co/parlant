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

from parlant.core.container import Container
from parlant.core.groups import GroupStore


async def test_that_groups_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    group_store = container[GroupStore]
    vip = await group_store.create_group(name="VIP")
    beta = await group_store.create_group(name="Beta")

    groups = (await async_client.get("/groups")).raise_for_status().json()

    assert {g["id"] for g in groups} == {vip.id, beta.id}
    assert {g["name"] for g in groups} == {"VIP", "Beta"}


async def test_that_groups_can_be_listed_filtered_by_name(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    group_store = container[GroupStore]
    vip = await group_store.create_group(name="VIP")
    await group_store.create_group(name="Beta")

    groups = (await async_client.get("/groups?name=VIP")).raise_for_status().json()

    assert [g["id"] for g in groups] == [vip.id]


async def test_that_groups_filtered_by_a_nonexistent_name_returns_an_empty_list(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    await container[GroupStore].create_group(name="VIP")

    groups = (await async_client.get("/groups?name=NoSuchGroup")).raise_for_status().json()

    assert groups == []


async def test_that_group_write_endpoints_do_not_exist(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    group = await container[GroupStore].create_group(name="VIP")

    assert (
        await async_client.post("/groups", json={"name": "New Group"})
    ).status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    # With the write and read-single routes gone, nothing is mounted at
    # /groups/{id} at all, so any method there is an unmatched path.
    assert (
        await async_client.patch(f"/groups/{group.id}", json={"name": "Renamed"})
    ).status_code == status.HTTP_404_NOT_FOUND

    assert (
        await async_client.delete(f"/groups/{group.id}")
    ).status_code == status.HTTP_404_NOT_FOUND

    assert (await async_client.get(f"/groups/{group.id}")).status_code == status.HTTP_404_NOT_FOUND
