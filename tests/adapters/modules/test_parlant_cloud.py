from starlette.requests import Request

from parlant.adapters.modules.parlant_cloud import (
    PLATFORM_SECRET_HEADER,
    ParlantCloudAuthorizationPolicy,
)
from parlant.api.authorization import Operation


def _request(headers: dict[str, str] | None = None) -> Request:
    raw_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/chat/",
            "headers": raw_headers,
            "client": ("127.0.0.1", 12345),
        }
    )


async def test_parlant_cloud_auth_allows_integrated_ui_without_platform_secret() -> None:
    policy = ParlantCloudAuthorizationPolicy(platform_secret="secret")

    assert await policy.check_permission(_request(), Operation.ACCESS_INTEGRATED_UI)


async def test_parlant_cloud_auth_keeps_api_docs_platform_only() -> None:
    policy = ParlantCloudAuthorizationPolicy(platform_secret="secret")

    assert not await policy.check_permission(_request(), Operation.ACCESS_API_DOCS)


async def test_parlant_cloud_auth_allows_trusted_platform_requests() -> None:
    policy = ParlantCloudAuthorizationPolicy(platform_secret="secret")

    assert await policy.check_permission(
        _request({PLATFORM_SECRET_HEADER: "secret"}),
        Operation.CREATE_AGENT,
    )
