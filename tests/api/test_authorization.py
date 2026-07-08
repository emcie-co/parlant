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

from datetime import datetime, timezone
from typing import Mapping

import pytest
from fastapi import Request
from lagom import Container
from limits import RateLimitItemPerMinute

from parlant.api.authentication import (
    AdminPrincipal,
    AnonymousPrincipal,
    AuthenticationError,
    CustomerPrincipal,
    GuestPrincipal,
    TokenAuthenticator,
)
from parlant.api.authorization import (
    AuthorizationException,
    CompositeAuthorizationPolicy,
    DefaultPermissionGate,
    DevelopmentAuthorizationPolicy,
    GUEST_SESSION_METADATA_KEY,
    Operation,
    OwnershipAuthorizer,
    BasicRateLimiter,
)
from parlant.core.agents import AgentId
from parlant.core.common import JSONSerializable
from parlant.core.customers import CustomerId, CustomerStore
from parlant.core.sessions import Session, SessionId


def make_request(
    *,
    path: str = "/",
    x_forwarded_for: str | None = "203.0.113.10",
    client_host: str | None = "127.0.0.1",
    extra_headers: dict[str, str] | None = None,
) -> Request:
    headers = []

    if x_forwarded_for is not None:
        headers.append((b"x-forwarded-for", x_forwarded_for.encode("latin-1")))
    for name, value in (extra_headers or {}).items():
        headers.append((name.encode("latin-1"), value.encode("latin-1")))

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


async def test_that_authorization_exception_redacts_sensitive_headers() -> None:
    request = make_request(
        extra_headers={
            "authorization": "Bearer secret-token",
            "cookie": "access_token=secret-cookie",
            "x-parlant-cloud-project-token": "raw-project-token-value",
            "host": "localhost:2222",
        }
    )

    exception = AuthorizationException(request, Operation.STREAM_LOGS)
    message = str(exception)

    assert "secret-token" not in message
    assert "secret-cookie" not in message
    assert "raw-project-token-value" not in message
    assert "'host': 'localhost:2222'" in message
    assert "'authorization': '<redacted>'" in message
    assert "'cookie': '<redacted>'" in message
    assert "'x-parlant-cloud-project-token': '<redacted>'" in message


###############################################################################
# Permission gate
###############################################################################


ADMIN_OPERATIONS = [
    Operation.CREATE_AGENT,
    Operation.UPDATE_AGENT,
    Operation.DELETE_AGENT,
    Operation.CREATE_CUSTOMER,
    Operation.DELETE_CUSTOMER,
    Operation.CREATE_JOURNEY,
    Operation.CREATE_RULE,
    Operation.UPDATE_SERVICE,
    Operation.DELETE_SESSION,
    Operation.DELETE_SESSIONS,
    Operation.UPDATE_SESSION,
]


async def test_that_an_anonymous_principal_may_only_create_guest_sessions_and_read_agents() -> None:
    gate = DefaultPermissionGate()
    principal = AnonymousPrincipal()

    assert await gate.check(principal, Operation.CREATE_GUEST_SESSION) is True
    assert await gate.check(principal, Operation.READ_AGENT) is True

    assert await gate.check(principal, Operation.LIST_EVENTS) is False
    assert await gate.check(principal, Operation.READ_SESSION) is False
    assert await gate.check(principal, Operation.CREATE_CUSTOMER_SESSION) is False
    assert await gate.check(principal, Operation.CREATE_CUSTOMER_EVENT) is False

    for operation in ADMIN_OPERATIONS:
        assert await gate.check(principal, operation) is False


async def test_that_a_guest_principal_is_allowed_only_conversation_operations() -> None:
    gate = DefaultPermissionGate()
    principal = GuestPrincipal(guest_instance_id="guest-instance-1")

    assert await gate.check(principal, Operation.CREATE_GUEST_SESSION) is True
    assert await gate.check(principal, Operation.READ_AGENT) is True
    assert await gate.check(principal, Operation.READ_SESSION) is True
    assert await gate.check(principal, Operation.LIST_EVENTS) is True
    assert await gate.check(principal, Operation.READ_EVENT) is True
    assert await gate.check(principal, Operation.CREATE_CUSTOMER_EVENT) is True
    assert await gate.check(principal, Operation.CREATE_STATUS_EVENT) is True

    assert await gate.check(principal, Operation.CREATE_CUSTOMER_SESSION) is False
    assert await gate.check(principal, Operation.LIST_SESSIONS) is False

    for operation in ADMIN_OPERATIONS:
        assert await gate.check(principal, operation) is False


async def test_that_a_customer_principal_may_use_customer_session_operations_but_not_admin_operations() -> (
    None
):
    gate = DefaultPermissionGate()
    principal = CustomerPrincipal(customer_id=CustomerId("cust_42"))

    assert await gate.check(principal, Operation.CREATE_CUSTOMER_SESSION) is True
    assert await gate.check(principal, Operation.READ_AGENT) is True
    assert await gate.check(principal, Operation.READ_SESSION) is True
    assert await gate.check(principal, Operation.LIST_SESSIONS) is True
    assert await gate.check(principal, Operation.LIST_EVENTS) is True
    assert await gate.check(principal, Operation.READ_EVENT) is True
    assert await gate.check(principal, Operation.CREATE_CUSTOMER_EVENT) is True
    assert await gate.check(principal, Operation.CREATE_STATUS_EVENT) is True

    assert await gate.check(principal, Operation.CREATE_GUEST_SESSION) is False

    for operation in ADMIN_OPERATIONS:
        assert await gate.check(principal, operation) is False


async def test_that_an_admin_principal_is_allowed_every_operation() -> None:
    gate = DefaultPermissionGate()
    principal = AdminPrincipal()

    for operation in Operation:
        assert await gate.check(principal, operation) is True


###############################################################################
# Ownership authorizer
###############################################################################


def make_session(
    customer_id: str = "cust_1",
    metadata: Mapping[str, JSONSerializable] | None = None,
) -> Session:
    now = datetime.now(timezone.utc)

    return Session(
        id=SessionId("test-session"),
        creation_utc=now,
        modified_utc=now,
        customer_id=CustomerId(customer_id),
        agent_id=AgentId("test-agent"),
        mode="auto",
        title=None,
        consumption_offsets={},
        agent_states=[],
        metadata=metadata or {},
        labels=set(),
    )


def make_guest_session(guest_instance_id: str) -> Session:
    return make_session(
        customer_id=CustomerStore.GUEST_ID,
        metadata={GUEST_SESSION_METADATA_KEY: guest_instance_id},
    )


async def test_that_a_guest_can_access_a_session_it_created() -> None:
    authorizer = OwnershipAuthorizer()

    principal = GuestPrincipal(guest_instance_id="guest-instance-1")
    session = make_guest_session("guest-instance-1")

    assert await authorizer.check_session_access(principal, session) is True


async def test_that_a_guest_cannot_access_a_session_created_by_another_guest() -> None:
    authorizer = OwnershipAuthorizer()

    principal = GuestPrincipal(guest_instance_id="guest-instance-1")
    session = make_guest_session("guest-instance-2")

    assert await authorizer.check_session_access(principal, session) is False


async def test_that_a_guest_cannot_access_a_customer_session() -> None:
    authorizer = OwnershipAuthorizer()

    principal = GuestPrincipal(guest_instance_id="guest-instance-1")
    session = make_session(customer_id="cust_42")

    assert await authorizer.check_session_access(principal, session) is False


async def test_that_a_customer_can_access_its_own_session() -> None:
    authorizer = OwnershipAuthorizer()

    principal = CustomerPrincipal(customer_id=CustomerId("cust_42"))
    session = make_session(customer_id="cust_42")

    assert await authorizer.check_session_access(principal, session) is True


async def test_that_a_customer_cannot_access_another_customers_session() -> None:
    authorizer = OwnershipAuthorizer()

    principal = CustomerPrincipal(customer_id=CustomerId("cust_42"))
    session = make_session(customer_id="cust_43")

    assert await authorizer.check_session_access(principal, session) is False


async def test_that_a_customer_cannot_access_a_guest_session() -> None:
    authorizer = OwnershipAuthorizer()

    principal = CustomerPrincipal(customer_id=CustomerId("cust_42"))
    session = make_guest_session("guest-instance-1")

    assert await authorizer.check_session_access(principal, session) is False


async def test_that_an_admin_can_access_any_session() -> None:
    authorizer = OwnershipAuthorizer()

    principal = AdminPrincipal()

    assert await authorizer.check_session_access(principal, make_session("cust_42")) is True
    assert (
        await authorizer.check_session_access(principal, make_guest_session("guest-instance-1"))
        is True
    )


async def test_that_an_anonymous_principal_cannot_access_any_session() -> None:
    authorizer = OwnershipAuthorizer()

    principal = AnonymousPrincipal()

    assert await authorizer.check_session_access(principal, make_session("cust_42")) is False
    assert (
        await authorizer.check_session_access(principal, make_guest_session("guest-instance-1"))
        is False
    )


###############################################################################
# Composite policy
###############################################################################


ADMIN_API_KEY = "test-admin-api-key"
CUSTOMER_JWT_SECRET = "integrator-shared-secret-0123456789abcdef"


def make_composite_policy(
    customer_store: CustomerStore | None = None,
    auto_provision_customers: bool = False,
) -> CompositeAuthorizationPolicy:
    return CompositeAuthorizationPolicy(
        authenticator=TokenAuthenticator(
            guest_token_secret="test-guest-secret-0123456789abcdef",
            admin_api_key=ADMIN_API_KEY,
            customer_jwt_secret=CUSTOMER_JWT_SECRET,
        ),
        customer_store=customer_store,
        auto_provision_customers=auto_provision_customers,
    )


def mint_customer_jwt(customer_id: str) -> str:
    import time

    import jwt

    return jwt.encode(
        {"sub": customer_id, "exp": int(time.time()) + 300},
        CUSTOMER_JWT_SECRET,
        algorithm="HS256",
    )


async def test_that_the_composite_policy_allows_an_admin_to_perform_admin_operations() -> None:
    policy = make_composite_policy()

    request = make_request(extra_headers={"authorization": f"Bearer {ADMIN_API_KEY}"})

    assert await policy.check_permission(request, Operation.CREATE_AGENT) is True


async def test_that_the_composite_policy_denies_admin_operations_to_anonymous_callers() -> None:
    policy = make_composite_policy()

    request = make_request()

    with pytest.raises(AuthorizationException):
        await policy.authorize(request, Operation.CREATE_AGENT)


async def test_that_the_composite_policy_allows_anonymous_guest_session_creation() -> None:
    policy = make_composite_policy()

    request = make_request()

    assert await policy.check_permission(request, Operation.CREATE_GUEST_SESSION) is True


async def test_that_the_composite_policy_rejects_an_invalid_bearer_token() -> None:
    policy = make_composite_policy()

    request = make_request(extra_headers={"authorization": "Bearer complete-garbage"})

    with pytest.raises(AuthenticationError):
        await policy.authorize(request, Operation.LIST_EVENTS)


async def test_that_the_composite_policy_caches_the_resolved_principal_on_the_request() -> None:
    policy = make_composite_policy()

    request = make_request(extra_headers={"authorization": f"Bearer {ADMIN_API_KEY}"})

    first = await policy.resolve_principal(request)
    second = await policy.resolve_principal(request)

    assert first is second


async def test_that_the_composite_policy_authorizes_session_access_by_ownership() -> None:
    policy = make_composite_policy()

    token = policy.issue_guest_token("guest-instance-1")
    request = make_request(extra_headers={"authorization": f"Bearer {token}"})

    own_session = make_guest_session("guest-instance-1")
    foreign_session = make_guest_session("guest-instance-2")

    await policy.authorize_session_access(request, own_session)

    with pytest.raises(AuthorizationException):
        await policy.authorize_session_access(request, foreign_session)


async def test_that_the_composite_policy_rejects_an_unknown_customer_when_auto_provisioning_is_disabled(
    container: Container,
) -> None:
    policy = make_composite_policy(
        customer_store=container[CustomerStore],
        auto_provision_customers=False,
    )

    token = mint_customer_jwt("cust_unknown")
    request = make_request(extra_headers={"authorization": f"Bearer {token}"})

    with pytest.raises(AuthenticationError):
        await policy.resolve_principal(request)


async def test_that_the_composite_policy_auto_provisions_an_unknown_customer_when_enabled(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    policy = make_composite_policy(
        customer_store=customer_store,
        auto_provision_customers=True,
    )

    token = mint_customer_jwt("cust_new_here")
    request = make_request(extra_headers={"authorization": f"Bearer {token}"})

    principal = await policy.resolve_principal(request)

    assert isinstance(principal, CustomerPrincipal)
    assert principal.customer_id == CustomerId("cust_new_here")

    customer = await customer_store.read_customer(CustomerId("cust_new_here"))
    assert customer.id == CustomerId("cust_new_here")


async def test_that_the_composite_policy_resolves_a_known_customer_without_provisioning(
    container: Container,
) -> None:
    customer_store = container[CustomerStore]

    customer = await customer_store.create_customer(
        name="Known Customer", id=CustomerId("cust_known")
    )

    policy = make_composite_policy(
        customer_store=customer_store,
        auto_provision_customers=False,
    )

    token = mint_customer_jwt("cust_known")
    request = make_request(extra_headers={"authorization": f"Bearer {token}"})

    principal = await policy.resolve_principal(request)

    assert isinstance(principal, CustomerPrincipal)
    assert principal.customer_id == customer.id


async def test_that_the_development_policy_remains_fully_permissive() -> None:
    policy = DevelopmentAuthorizationPolicy()

    request = make_request()

    for operation in Operation:
        assert await policy.check_permission(request, operation) is True

    assert await policy.resolve_principal(request) is None

    await policy.authorize_session_access(request, make_session("cust_42"))
    await policy.authorize_session_access(request, make_guest_session("guest-instance-1"))
