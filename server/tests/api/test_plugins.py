from fastapi import status
from fastapi.testclient import TestClient

from tests.core.services.tools.test_openapi import (
    OPENAPI_SERVER_URL,
    get_openapi_spec,
    rng_app,
    run_openapi_server,
)


async def test_that_sdk_plugin_is_created(
    client: TestClient,
) -> None:
    content = (
        client.put(
            "/plugins/my_sdk_plugin",
            json={
                "kind": "sdk",
                "url": "https://example.com/sdk",
            },
        )
        .raise_for_status()
        .json()
    )

    assert content["name"] == "my_sdk_plugin"


async def test_that_openapi_plugin_is_created(
    client: TestClient,
) -> None:
    async with run_openapi_server(rng_app()):
        openapi_json = await get_openapi_spec(OPENAPI_SERVER_URL)

    content = (
        client.put(
            "/plugins/my_openapi_plugin",
            json={
                "kind": "openapi",
                "url": OPENAPI_SERVER_URL,
                "openapi_json": openapi_json,
            },
        )
        .raise_for_status()
        .json()
    )

    assert content["name"] == "my_openapi_plugin"
