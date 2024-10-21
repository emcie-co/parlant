from fastapi import status
from fastapi.testclient import TestClient
import httpx
from lagom import Container

from emcie.server.core.services.tools.service_registry import ServiceRegistry
from tests.core.services.tools.test_openapi import (
    OPENAPI_SERVER_URL,
    get_openapi_spec,
    rng_app,
    run_openapi_server,
)

from emcie.common.plugin import tool
from emcie.common.tools import ToolResult, ToolContext

from tests.core.services.tools.test_plugin_client import run_service_server


async def test_that_sdk_service_is_created(
    client: TestClient,
) -> None:
    content = (
        client.put(
            "/services/my_sdk_service",
            json={
                "kind": "sdk",
                "url": "https://example.com/sdk",
            },
        )
        .raise_for_status()
        .json()
    )

    assert content["name"] == "my_sdk_service"


async def test_that_openapi_service_is_created(
    client: TestClient,
) -> None:
    async with run_openapi_server(rng_app()):
        openapi_json = await get_openapi_spec(OPENAPI_SERVER_URL)

    content = (
        client.put(
            "/services/my_openapi_service",
            json={
                "kind": "openapi",
                "url": OPENAPI_SERVER_URL,
                "openapi_json": openapi_json,
            },
        )
        .raise_for_status()
        .json()
    )

    assert content["name"] == "my_openapi_service"


def test_that_sdk_service_is_created_and_deleted(
    client: TestClient,
) -> None:
    _ = (
        client.put(
            "/services/my_sdk_service",
            json={
                "kind": "sdk",
                "url": "https://example.com/sdk",
            },
        )
        .raise_for_status()
        .json()
    )

    response = client.delete("/services/my_sdk_service")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == "my_sdk_service"

    response = client.delete("/services/my_sdk_service")
    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_openapi_service_is_created_and_deleted(
    client: TestClient,
) -> None:
    async with run_openapi_server(rng_app()):
        openapi_json = await get_openapi_spec(OPENAPI_SERVER_URL)

        _ = (
            client.put(
                "/services/my_openapi_service",
                json={
                    "kind": "openapi",
                    "url": OPENAPI_SERVER_URL,
                    "openapi_json": openapi_json,
                },
            )
            .raise_for_status()
            .json()
        )

        response = client.delete("/services/my_openapi_service")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "my_openapi_service"

        response = client.delete("/services/my_openapi_service")
        assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_services_are_listed_correctly(
    client: TestClient,
) -> None:
    response = client.get("/services/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["services"] == []

    _ = (
        client.put(
            "/services/my_sdk_service",
            json={
                "kind": "sdk",
                "url": "https://example.com/sdk",
            },
        )
        .raise_for_status()
        .json()
    )

    async with run_openapi_server(rng_app()):
        openapi_json = await get_openapi_spec(OPENAPI_SERVER_URL)
        _ = (
            client.put(
                "/services/my_openapi_service",
                json={
                    "kind": "openapi",
                    "url": OPENAPI_SERVER_URL,
                    "openapi_json": openapi_json,
                },
            )
            .raise_for_status()
            .json()
        )

    response = client.get("/services/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    services = data["services"]

    assert len(services) == 2

    sdk_service = next((p for p in services if p["name"] == "my_sdk_service"), None)
    assert sdk_service is not None
    assert sdk_service["kind"] == "sdk"
    assert sdk_service["url"] == "https://example.com/sdk"

    openapi_service = next((p for p in services if p["name"] == "my_openapi_service"), None)
    assert openapi_service is not None
    assert openapi_service["kind"] == "openapi"
    assert openapi_service["url"] == OPENAPI_SERVER_URL


async def test_that_reading_an_existing_openapi_service_returns_its_metadata_and_tools(
    client: TestClient,
    container: Container,
) -> None:
    service_registry = container[ServiceRegistry]

    async with run_openapi_server(rng_app()):
        openapi_json = await get_openapi_spec(OPENAPI_SERVER_URL)
        await service_registry.update_tool_service(
            name="my_openapi_service",
            kind="openapi",
            url=OPENAPI_SERVER_URL,
            openapi_json=openapi_json,
        )

        service_data = client.get("/services/my_openapi_service").raise_for_status().json()

        assert service_data["name"] == "my_openapi_service"
        assert service_data["kind"] == "openapi"
        assert service_data["url"] == OPENAPI_SERVER_URL

        tools = service_data["tools"]
        assert len(tools) > 0

        for tool in tools:
            assert "id" in tool
            assert "name" in tool
            assert "description" in tool


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

        for tool_data in tools_list:
            if tool_data["id"] == my_tool.tool.id:
                assert tool_data["name"] == my_tool.tool.name
                assert tool_data["description"] == my_tool.tool.description
            elif tool_data["id"] == my_async_tool.tool.id:
                assert tool_data["name"] == my_async_tool.tool.name
                assert tool_data["description"] == my_async_tool.tool.description
