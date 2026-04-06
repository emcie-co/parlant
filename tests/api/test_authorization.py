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

import pytest
from fastapi import Request
from limits import RateLimitItemPerMinute

from parlant.api.authorization import (
    AuthorizationException,
    Operation,
    BasicRateLimiter,
)


def make_request(
    *,
    path: str = "/",
    x_forwarded_for: str | None = "203.0.113.10",
    client_host: str | None = "127.0.0.1",
) -> Request:
    headers = []

    if x_forwarded_for is not None:
        headers.append((b"x-forwarded-for", x_forwarded_for.encode("latin-1")))

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": headers,
        "client": (client_host, 12345) if client_host is not None else None,
        "query_string": b"",
        "http_version": "1.1",
        "scheme": "http",
        "server": ("testserver", 80),
    }

    return Request(scope)


async def test_that_a_configured_operation_is_limited_per_minute() -> None:
    limiter = BasicRateLimiter(
        rate_limit_item_per_operation={
            Operation.LIST_EVENTS: RateLimitItemPerMinute(2),
        }
    )

    request = make_request()

    assert await limiter.check(request, Operation.LIST_EVENTS) is True
    assert await limiter.check(request, Operation.LIST_EVENTS) is True
    assert await limiter.check(request, Operation.LIST_EVENTS) is False


async def test_that_limits_are_isolated_per_operation_bucket() -> None:
    limiter = BasicRateLimiter(
        rate_limit_item_per_operation={
            Operation.LIST_EVENTS: RateLimitItemPerMinute(1),
        }
    )

    request = make_request()

    assert await limiter.check(request, Operation.LIST_EVENTS) is True
    assert await limiter.check(request, Operation.LIST_EVENTS) is False


async def test_that_limits_are_isolated_per_client_ip() -> None:
    limiter = BasicRateLimiter(
        rate_limit_item_per_operation={
            Operation.LIST_EVENTS: RateLimitItemPerMinute(1),
        }
    )

    req_ip1 = make_request(x_forwarded_for="198.51.100.7")
    req_ip2 = make_request(x_forwarded_for="198.51.100.8")

    assert await limiter.check(req_ip1, Operation.LIST_EVENTS) is True
    assert await limiter.check(req_ip2, Operation.LIST_EVENTS) is True

    assert await limiter.check(req_ip1, Operation.LIST_EVENTS) is False


async def test_that_x_forwarded_for_overrides_request_client_host_for_ip_selection() -> None:
    limiter = BasicRateLimiter(
        rate_limit_item_per_operation={
            Operation.LIST_EVENTS: RateLimitItemPerMinute(1),
        }
    )

    req_a = make_request(x_forwarded_for="1.1.1.1", client_host="10.0.0.5")
    req_b = make_request(x_forwarded_for="1.1.1.2", client_host="10.0.0.5")

    assert await limiter.check(req_a, Operation.LIST_EVENTS) is True
    assert await limiter.check(req_b, Operation.LIST_EVENTS) is True
    assert await limiter.check(req_a, Operation.LIST_EVENTS) is False


async def test_that_missing_client_ip_raises_authorization_exception() -> None:
    limiter = BasicRateLimiter(
        rate_limit_item_per_operation={
            Operation.LIST_EVENTS: RateLimitItemPerMinute(1),
        }
    )
    request = make_request(x_forwarded_for=None, client_host=None)

    with pytest.raises(AuthorizationException):
        await limiter.check(request, Operation.LIST_EVENTS)
