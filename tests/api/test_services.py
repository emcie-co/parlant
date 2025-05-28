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

import json
import os
import tempfile
from fastapi import status
import httpx
from lagom import Container

from parlant.core.services.tools.plugins import tool
from parlant.core.tools import ToolResult, ToolContext
from parlant.core.services.tools.service_registry import ServiceRegistry

from tests.test_utilities import run_openapi_server, run_service_server


async def test_that_sdk_service_is_created(
    async_client: httpx.AsyncClient,
) -> None:
    content = (
        (
            await async_client.put(
                "/services/my_sdk_service",
                json={
                    "kind": "sdk",
                    "sdk": {
                        "url": "https://example.com/sdk",
                    },
                },
            )
        )
        .raise_for_status()
        .json()
    )

    assert content["name"] == "my_sdk_service"
    assert content["kind"] == "sdk"
    assert content["url"] == "https://example.com/sdk"


async def test_that_sdk_service_fails_to_create_due_to_url_not_starting_with_http_or_https(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.put(
        "/services/my_sdk_service",
        json={
            "kind": "sdk",
            "sdk": {
                "url": "example.com/sdk",
            },
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert response.json()["detail"] == "Service URL is missing schema (http:// or https://)"


async def test_that_openapi_service_is_created_with_url_source(
    async_client: httpx.AsyncClient,
) -> None:
    async with run_openapi_server() as server_info:
        url = f"{server_info.url}:{server_info.port}"
        source = f"{url}/openapi.json"

        response = await async_client.put(
            "/services/my_openapi_service",
            json={
                "kind": "openapi",
                "openapi": {
                    "url": url,
                    "source": source,
                },
            },
        )
        response.raise_for_status()
        content = response.json()

        assert content["name"] == "my_openapi_service"
        assert content["kind"] == "openapi"
        assert content["url"] == url


async def test_that_openapi_service_is_created_with_file_source(
    async_client: httpx.AsyncClient,
) -> None:
    openapi_json = {
        "openapi": "3.0.0",
        "info": {"title": "TestAPI", "version": "1.0.0"},
        "paths": {
            "/hello": {
                "get": {
                    "summary": "Say Hello",
                    "operationId": "print_hello__get",
                    "responses": {
                        "200": {
                            "description": "Successful Response",
                            "content": {"application/json": {"schema": {"type": "string"}}},
                        }
                    },
                }
            }
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_file:
        json.dump(openapi_json, tmp_file)
        source = tmp_file.name

    response = await async_client.put(
        "/services/my_openapi_file_service",
        json={
            "kind": "openapi",
            "openapi": {
                "url": "http://localhost",
                "source": source,
            },
        },
    )
    response.raise_for_status()
    content = response.json()

    assert content["name"] == "my_openapi_file_service"
    assert content["kind"] == "openapi"
    assert content["url"] == "http://localhost"

    os.remove(source)


async def test_that_sdk_service_is_created_and_deleted(
    async_client: httpx.AsyncClient,
) -> None:
    _ = (
        (
            await async_client.put(
                "/services/my_sdk_service",
                json={
                    "kind": "sdk",
                    "sdk": {
                        "url": "https://example.com/sdk",
                    },
                },
            )
        )
        .raise_for_status()
        .json()
    )

    await async_client.delete("/services/my_sdk_service")

    response = await async_client.get("/services/my_sdk_service")
    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_openapi_service_is_created_and_deleted(
    async_client: httpx.AsyncClient,
) -> None:
    async with run_openapi_server() as server_info:
        url = f"{server_info.url}:{server_info.port}"
        source = f"{url}/openapi.json"

        _ = (
            await async_client.put(
                "/services/my_openapi_service",
                json={
                    "kind": "openapi",
                    "openapi": {
                        "url": url,
                        "source": source,
                    },
                },
            )
        ).raise_for_status()

    await async_client.delete("/services/my_openapi_service")

    response = await async_client.get("/services/my_sdk_service")
    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_services_can_be_listed(
    async_client: httpx.AsyncClient,
) -> None:
    assert (await async_client.get("/services")).raise_for_status().json() == []

    _ = (
        (
            await async_client.put(
                "/services/my_sdk_service",
                json={
                    "kind": "sdk",
                    "sdk": {
                        "url": "https://example.com/sdk",
                    },
                },
            )
        )
        .raise_for_status()
        .json()
    )

    async with run_openapi_server() as server_info:
        url = f"{server_info.url}:{server_info.port}"
        source = f"{url}/openapi.json"
        response = await async_client.put(
            "/services/my_openapi_service",
            json={
                "kind": "openapi",
                "openapi": {
                    "url": url,
                    "source": source,
                },
            },
        )
        response.raise_for_status()

    services = (await async_client.get("/services")).raise_for_status().json()

    assert len(services) == 2

    sdk_service = next((p for p in services if p["name"] == "my_sdk_service"), None)
    assert sdk_service is not None
    assert sdk_service["kind"] == "sdk"
    assert sdk_service["url"] == "https://example.com/sdk"

    openapi_service = next((p for p in services if p["name"] == "my_openapi_service"), None)
    assert openapi_service is not None
    assert openapi_service["kind"] == "openapi"
    assert openapi_service["url"] == url


async def test_that_reading_an_existing_openapi_service_returns_its_metadata_and_tools(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    service_registry = container[ServiceRegistry]
    async with run_openapi_server() as server_info:
        url = f"{server_info.url}:{server_info.port}"
        source = f"{url}/openapi.json"
        await service_registry.update_tool_service(
            name="my_openapi_service",
            kind="openapi",
            url=url,
            source=source,
        )

    service_data = (
        (await async_client.get("/services/my_openapi_service")).raise_for_status().json()
    )

    assert service_data["name"] == "my_openapi_service"
    assert service_data["kind"] == "openapi"
    assert service_data["url"] == url

    tools = service_data["tools"]
    assert len(tools) > 0

    for t in tools:
        assert "name" in t
        assert "description" in t


async def test_that_reading_an_existing_sdk_service_returns_its_metadata_and_tools(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    @tool
    def my_tool(context: ToolContext, arg_1: int, arg_2: int) -> ToolResult:
        return ToolResult(arg_1 + arg_2)

    @tool
    async def my_async_tool(context: ToolContext, message: str) -> ToolResult:
        return ToolResult(f"Echo: {message}")

    service_registry = container[ServiceRegistry]

    async with run_service_server([my_tool, my_async_tool]) as server:
        await service_registry.update_tool_service(
            name="my_sdk_service",
            kind="sdk",
            url=server.url,
        )

        response = await async_client.get("/services/my_sdk_service")
        response.raise_for_status()
        service_data = response.json()

        assert service_data["name"] == "my_sdk_service"
        assert service_data["kind"] == "sdk"
        assert service_data["url"] == server.url

        tools_list = service_data["tools"]
        assert len(tools_list) == 2

        assert any(
            t["name"] == my_tool.tool.name and t["description"] == my_tool.tool.description
            for t in tools_list
        )
        assert any(
            t["name"] == my_async_tool.tool.name
            and t["description"] == my_async_tool.tool.description
            for t in tools_list
        )
