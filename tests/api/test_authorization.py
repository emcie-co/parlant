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

import hmac
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping

import pytest
from fastapi import FastAPI, Request, WebSocket
from parlant.core.container import Container
from limits import RateLimitItemPerMinute

from parlant.api.authentication import (
    AdminPrincipal,
    AnonymousPrincipal,
    AuthenticationError,
    CustomerPrincipal,
    GuestPrincipal,
    Principal,
    TokenAuthenticator,
)
from parlant.api.authorization import (
    AuthorizationException,
    CompositeAuthorizationPolicy,
    DefaultPermissionGate,
    DevelopmentAuthorizationPolicy,
    GUEST_SESSION_METADATA_KEY,
    HeaderApiKeyAuthorizationPolicy,
    Operation,
    OwnershipAuthorizer,
    BasicRateLimiter,
)
from parlant.core import metadata as metadata_utils
from parlant.core.agents import AgentId
from parlant.core.common import JSONSerializable
from parlant.core.customers import CustomerId, CustomerKind, CustomerStore
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

    req_ip1 = make_request(x_forwarded_for=None, client_host="198.51.100.7")
    req_ip2 = make_request(x_forwarded_for=None, client_host="198.51.100.8")

    assert await limiter.check(req_ip1, Operation.LIST_EVENTS) is True
    assert await limiter.check(req_ip2, Operation.LIST_EVENTS) is True

    assert await limiter.check(req_ip1, Operation.LIST_EVENTS) is False


async def test_that_x_forwarded_for_is_honored_when_the_peer_is_a_trusted_proxy() -> None:
    limiter = BasicRateLimiter(
        rate_limit_item_per_operation={
            Operation.LIST_EVENTS: RateLimitItemPerMinute(1),
        },
        trusted_proxies=frozenset({"10.0.0.5"}),
    )

    # The peer is the configured proxy, so the forwarded client IP is trusted
    # and distinguishes the two callers.
    req_a = make_request(x_forwarded_for="1.1.1.1", client_host="10.0.0.5")
    req_b = make_request(x_forwarded_for="1.1.1.2", client_host="10.0.0.5")

    assert await limiter.check(req_a, Operation.LIST_EVENTS) is True
    assert await limiter.check(req_b, Operation.LIST_EVENTS) is True
    assert await limiter.check(req_a, Operation.LIST_EVENTS) is False


async def test_that_x_forwarded_for_is_ignored_when_the_peer_is_not_a_trusted_proxy() -> None:
    limiter = BasicRateLimiter(
        rate_limit_item_per_operation={
            Operation.LIST_EVENTS: RateLimitItemPerMinute(1),
        }
    )

    # No trusted proxies configured (the default): a direct client cannot lift
    # its own rate limit by rotating X-Forwarded-For, because the peer address
    # is used and both requests share a bucket.
    req_a = make_request(x_forwarded_for="1.1.1.1", client_host="10.0.0.5")
    req_b = make_request(x_forwarded_for="1.1.1.2", client_host="10.0.0.5")

    assert await limiter.check(req_a, Operation.LIST_EVENTS) is True
    assert await limiter.check(req_b, Operation.LIST_EVENTS) is False


async def test_that_prepending_forwarded_for_does_not_change_the_rate_limit_identity() -> None:
    limiter = BasicRateLimiter(
        rate_limit_item_per_operation={
            Operation.LIST_EVENTS: RateLimitItemPerMinute(1),
        },
        trusted_proxies=frozenset({"10.0.0.5"}),
    )

    # The same client behind the same trusted proxy, prepending a different
    # forged address each time: it must stay in one bucket.
    req_a = make_request(x_forwarded_for="1.1.1.1, 203.0.113.7", client_host="10.0.0.5")
    req_b = make_request(x_forwarded_for="1.1.1.2, 203.0.113.7", client_host="10.0.0.5")

    assert await limiter.check(req_a, Operation.LIST_EVENTS) is True
    assert await limiter.check(req_b, Operation.LIST_EVENTS) is False


async def test_that_rate_limit_trusted_proxies_may_be_given_as_cidr_ranges() -> None:
    limiter = BasicRateLimiter(
        rate_limit_item_per_operation={
            Operation.LIST_EVENTS: RateLimitItemPerMinute(1),
        },
        trusted_proxies=frozenset({"10.0.0.0/8"}),
    )

    req_a = make_request(x_forwarded_for="203.0.113.7", client_host="10.20.30.40")
    req_b = make_request(x_forwarded_for="203.0.113.8", client_host="10.20.30.40")

    assert await limiter.check(req_a, Operation.LIST_EVENTS) is True
    assert await limiter.check(req_b, Operation.LIST_EVENTS) is True


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
            "x-parlant-project-token": "raw-project-token-value",
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
    assert "'x-parlant-project-token': '<redacted>'" in message


###############################################################################
# Permission gate
###############################################################################


ADMIN_OPERATIONS = [
    Operation.CREATE_CUSTOMER,
    Operation.UPDATE_CUSTOMER,
    Operation.DELETE_CUSTOMER,
    Operation.LIST_CUSTOMERS,
    Operation.LIST_GROUPS,
    Operation.DELETE_SESSION,
    Operation.DELETE_SESSIONS,
    Operation.UPDATE_SESSION,
    Operation.STREAM_LOGS,
    # Reading an event's internal reasoning (the "thinking" status payload) and
    # a tool call's raw result data is operator-only: a conversation-scoped
    # caller may list/read events, but those deep fields are stubbed out for it.
    Operation.READ_EVENT_REASONING,
    Operation.READ_TOOL_RESULT_DATA,
    # The Parlant-internal event metadata (__parlant__ / failure tracebacks) is
    # operator-only too.
    Operation.READ_INTERNAL_EVENT_METADATA,
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


async def test_that_reading_event_reasoning_and_tool_result_data_is_admin_only() -> None:
    gate = DefaultPermissionGate()

    guarded = [
        Operation.READ_EVENT_REASONING,
        Operation.READ_TOOL_RESULT_DATA,
        Operation.READ_INTERNAL_EVENT_METADATA,
    ]

    non_admins: list[Principal] = [
        AnonymousPrincipal(),
        GuestPrincipal(guest_instance_id="guest-instance-1"),
        CustomerPrincipal(customer_id=CustomerId("cust_42")),
    ]

    for operation in guarded:
        # An admin (operator) sees the deep fields.
        assert await gate.check(AdminPrincipal(), operation) is True

        # Everyone else is denied the operation, so the field is stubbed out for
        # them even though they may still list/read the event itself.
        for principal in non_admins:
            assert await gate.check(principal, operation) is False


async def test_that_the_development_policy_exposes_event_reasoning_and_tool_result_data() -> None:
    # In development (the default when no auth is configured), everything is
    # permitted — so events are returned in full, never redacted. This keeps
    # local debugging transparent and guards against a dev-policy change silently
    # starting to hide reasoning or tool data.
    policy = DevelopmentAuthorizationPolicy()
    request = make_request()

    assert await policy.check_permission(request, Operation.READ_EVENT_REASONING) is True
    assert await policy.check_permission(request, Operation.READ_TOOL_RESULT_DATA) is True


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
        agent_states=[],
        metadata=metadata or {},
        labels=set(),
    )


def make_guest_session(guest_instance_id: str) -> Session:
    # Mirror the production stamp: the guest instance id lives in the
    # session's INTERNAL metadata under ("auth", ...), not as a public key.
    return make_session(
        customer_id=CustomerStore.GUEST_ID,
        metadata=metadata_utils.add_internal_metadata(
            {}, {"auth": {GUEST_SESSION_METADATA_KEY: guest_instance_id}}
        ),
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

    assert await policy.check_permission(request, Operation.CREATE_CUSTOMER) is True


async def test_that_the_composite_policy_denies_admin_operations_to_anonymous_callers() -> None:
    policy = make_composite_policy()

    request = make_request()

    with pytest.raises(AuthorizationException):
        await policy.authorize(request, Operation.CREATE_CUSTOMER)


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


###############################################################################
# Strict header API key policy
###############################################################################


PROJECT_TOKEN_HEADER = "X-Parlant-Project-Token"
PROJECT_TOKEN = "test-project-token"


def make_request_from_headers(headers: list[tuple[bytes, bytes]]) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "query_string": b"",
        "http_version": "1.1",
        "scheme": "http",
        "server": ("testserver", 80),
    }

    return Request(scope)


def make_websocket_from_headers(
    headers: list[tuple[bytes, bytes]], query_string: str = ""
) -> WebSocket:
    scope = {
        "type": "websocket",
        "path": "/",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "query_string": query_string.encode("latin-1"),
        "scheme": "ws",
        "server": ("testserver", 80),
    }

    async def receive() -> dict[str, Any]:
        return {"type": "websocket.connect"}

    async def send(message: MutableMapping[str, Any]) -> None:
        pass

    return WebSocket(scope, receive=receive, send=send)


def make_header_api_key_policy(
    *,
    api_key: str = PROJECT_TOKEN,
    header_name: str = PROJECT_TOKEN_HEADER,
) -> HeaderApiKeyAuthorizationPolicy:
    return HeaderApiKeyAuthorizationPolicy(api_key=api_key, header_name=header_name)


def project_token_header(value: str = PROJECT_TOKEN) -> tuple[bytes, bytes]:
    return (PROJECT_TOKEN_HEADER.lower().encode("ascii"), value.encode("utf-8"))


@pytest.mark.parametrize("api_key", ["", " ", "\t\n"])
def test_that_the_header_api_key_policy_rejects_a_blank_api_key(api_key: str) -> None:
    with pytest.raises(ValueError):
        make_header_api_key_policy(api_key=api_key)


@pytest.mark.parametrize(
    "header_name",
    [
        "",
        "X Project Token",
        "X-Parlant-Töken",
        "x" * 129,
    ],
)
def test_that_the_header_api_key_policy_rejects_an_invalid_header_name(header_name: str) -> None:
    with pytest.raises(ValueError, match="Invalid API credential header name"):
        make_header_api_key_policy(header_name=header_name)


def test_that_the_header_api_key_policy_has_a_stable_name() -> None:
    assert make_header_api_key_policy().name == "header-api-key"


def test_that_the_header_api_key_policy_repr_does_not_expose_the_secret() -> None:
    assert PROJECT_TOKEN not in repr(make_header_api_key_policy())


async def test_that_the_header_api_key_policy_allows_every_operation_with_exactly_one_matching_header() -> (
    None
):
    policy = make_header_api_key_policy()
    request = make_request_from_headers([project_token_header()])

    for operation in Operation:
        assert await policy.check_permission(request, operation) is True


@pytest.mark.parametrize(
    ("headers", "credential_values"),
    [
        pytest.param([], [], id="missing"),
        pytest.param(
            [project_token_header("wrong-project-token")],
            ["wrong-project-token"],
            id="wrong",
        ),
        pytest.param(
            [project_token_header(), project_token_header("duplicate-project-token")],
            ["test-project-token", "duplicate-project-token"],
            id="duplicate",
        ),
    ],
)
async def test_that_the_header_api_key_policy_rejects_invalid_http_credentials_without_leaking_them(
    headers: list[tuple[bytes, bytes]], credential_values: list[str]
) -> None:
    policy = make_header_api_key_policy()

    with pytest.raises(AuthenticationError) as exc_info:
        await policy.check_permission(
            make_request_from_headers(headers), Operation.ACCESS_INTEGRATED_UI
        )

    assert str(exc_info.value) == "Invalid or missing API credential."
    for credential_value in credential_values:
        assert credential_value not in str(exc_info.value)


async def test_that_the_header_api_key_policy_resolves_an_admin_principal() -> None:
    policy = make_header_api_key_policy()
    request = make_request_from_headers([project_token_header()])

    assert isinstance(await policy.resolve_principal(request), AdminPrincipal)


async def test_that_the_header_api_key_policy_does_not_bypass_authentication_for_localhost() -> (
    None
):
    policy = make_header_api_key_policy()
    request = make_request(
        x_forwarded_for=None,
        client_host="127.0.0.1",
        extra_headers={"host": "localhost:8800"},
    )

    with pytest.raises(AuthenticationError, match="^Invalid or missing API credential\\.$"):
        await policy.check_permission(request, Operation.ACCESS_API_DOCS)


async def test_that_the_header_api_key_policy_uses_constant_time_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The spy on hmac.compare_digest is the only way to observe that credential
    # comparison routes through a constant-time primitive.
    comparisons: list[tuple[bytes, bytes]] = []

    def compare_digest(candidate: bytes, expected: bytes) -> bool:
        comparisons.append((candidate, expected))
        return True

    monkeypatch.setattr(hmac, "compare_digest", compare_digest)

    policy = make_header_api_key_policy()
    request = make_request_from_headers([project_token_header()])

    assert await policy.check_permission(request, Operation.READ_AGENT) is True
    assert comparisons == [(PROJECT_TOKEN.encode("utf-8"), PROJECT_TOKEN.encode("utf-8"))]


async def test_that_the_header_api_key_policy_accepts_a_websocket_with_the_exact_header() -> None:
    policy = make_header_api_key_policy()
    websocket = make_websocket_from_headers([project_token_header()])

    assert await policy.check_websocket_permission(websocket, Operation.STREAM_LOGS) is True


@pytest.mark.parametrize(
    ("headers", "credential_values"),
    [
        pytest.param([], [], id="missing"),
        pytest.param(
            [project_token_header("wrong-websocket-token")],
            ["wrong-websocket-token"],
            id="wrong",
        ),
        pytest.param(
            [project_token_header(), project_token_header("duplicate-websocket-token")],
            ["test-project-token", "duplicate-websocket-token"],
            id="duplicate",
        ),
    ],
)
async def test_that_the_header_api_key_policy_rejects_invalid_websocket_credentials_without_leaking_them(
    headers: list[tuple[bytes, bytes]], credential_values: list[str]
) -> None:
    policy = make_header_api_key_policy()

    with pytest.raises(AuthenticationError) as exc_info:
        await policy.check_websocket_permission(
            make_websocket_from_headers(headers), Operation.STREAM_LOGS
        )

    assert str(exc_info.value) == "Invalid or missing API credential."
    for credential_value in credential_values:
        assert credential_value not in str(exc_info.value)


async def test_that_the_header_api_key_policy_rejects_a_websocket_query_string_token() -> None:
    policy = make_header_api_key_policy()
    websocket = make_websocket_from_headers([], query_string=f"token={PROJECT_TOKEN}")

    with pytest.raises(AuthenticationError) as exc_info:
        await policy.check_websocket_permission(websocket, Operation.STREAM_LOGS)

    assert str(exc_info.value) == "Invalid or missing API credential."
    assert PROJECT_TOKEN not in str(exc_info.value)


async def test_that_the_header_api_key_policy_inherits_configuration_without_adding_cors() -> None:
    policy = make_header_api_key_policy()
    app = FastAPI()

    assert await policy.configure_app(app) is app


async def test_that_conversation_principals_may_attach_files_by_default() -> None:
    gate = DefaultPermissionGate()

    assert (
        await gate.check(AdminPrincipal(), Operation.CREATE_CUSTOMER_EVENT_WITH_ATTACHMENTS) is True
    )
    assert (
        await gate.check(
            CustomerPrincipal(customer_id=CustomerId("cust_42")),
            Operation.CREATE_CUSTOMER_EVENT_WITH_ATTACHMENTS,
        )
        is True
    )
    assert (
        await gate.check(
            GuestPrincipal(guest_instance_id="guest-instance-1"),
            Operation.CREATE_CUSTOMER_EVENT_WITH_ATTACHMENTS,
        )
        is True
    )
    assert (
        await gate.check(AnonymousPrincipal(), Operation.CREATE_CUSTOMER_EVENT_WITH_ATTACHMENTS)
        is False
    )


class _FakeWebSocket:
    """The minimal surface `resolve_websocket` reads: headers and query params."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers
        self.query_params: dict[str, str] = {}


async def test_that_the_composite_policy_restricts_group_listing_to_admins() -> None:
    policy = make_composite_policy()

    admin_request = make_request(extra_headers={"authorization": f"Bearer {ADMIN_API_KEY}"})
    assert await policy.check_permission(admin_request, Operation.LIST_GROUPS) is True

    customer_request = make_request(
        extra_headers={"authorization": f"Bearer {mint_customer_jwt('cust_1')}"}
    )
    assert await policy.check_permission(customer_request, Operation.LIST_GROUPS) is False

    with pytest.raises(AuthorizationException):
        await policy.authorize(make_request(), Operation.LIST_GROUPS)


async def test_that_the_composite_policy_restricts_log_streaming_to_admins() -> None:
    policy = make_composite_policy()

    admin_ws = _FakeWebSocket({"authorization": f"Bearer {ADMIN_API_KEY}"})
    assert await policy.check_websocket_permission(admin_ws, Operation.STREAM_LOGS) is True  # type: ignore[arg-type]

    customer_ws = _FakeWebSocket({"authorization": f"Bearer {mint_customer_jwt('cust_1')}"})
    assert await policy.check_websocket_permission(customer_ws, Operation.STREAM_LOGS) is False  # type: ignore[arg-type]

    anonymous_ws = _FakeWebSocket({})
    assert await policy.check_websocket_permission(anonymous_ws, Operation.STREAM_LOGS) is False  # type: ignore[arg-type]


async def test_that_a_customer_token_for_a_persona_is_rejected(
    container: Container,
) -> None:
    """A persona is not an identity anybody can authenticate as.

    Resolution reads the token's subject as a customer id and accepts whatever
    it finds, so a persona sharing an id with a real person's token would
    quietly become that person's identity: their live session would attach to
    the persona and no error would be raised anywhere. Refusing the token is
    what keeps a persona reachable only by an admin.
    """
    customer_store = container[CustomerStore]

    await customer_store.create_customer(
        name="Test Persona",
        id=CustomerId("cust_persona"),
        kind=CustomerKind.PERSONA,
    )

    policy = make_composite_policy(
        customer_store=customer_store,
        auto_provision_customers=False,
    )

    token = mint_customer_jwt("cust_persona")
    request = make_request(extra_headers={"authorization": f"Bearer {token}"})

    with pytest.raises(AuthenticationError):
        await policy.resolve_principal(request)


async def test_that_an_auto_provisioned_customer_is_real(
    container: Container,
) -> None:
    """Provisioning must never mint a persona.

    A customer that appears because somebody authenticated is by definition a
    real one, and the store's default is what guarantees it. Pinning it here
    means a later change to the provisioning call cannot silently widen what
    a test run is allowed to touch.
    """
    customer_store = container[CustomerStore]

    policy = make_composite_policy(
        customer_store=customer_store,
        auto_provision_customers=True,
    )

    token = mint_customer_jwt("cust_provisioned")
    request = make_request(extra_headers={"authorization": f"Bearer {token}"})

    await policy.resolve_principal(request)

    customer = await customer_store.read_customer(CustomerId("cust_provisioned"))

    assert customer.kind is CustomerKind.REAL


async def test_that_the_persona_session_operation_is_denied_to_every_non_admin_principal() -> None:
    """The named capability is admin-only until somebody is granted it.

    This is the permission that a future test runner can be given without the
    admin key. Today nobody but an admin holds it, and the gate is where that
    is decided.
    """
    gate = DefaultPermissionGate()

    assert await gate.check(AdminPrincipal(), Operation.CREATE_PERSONA_SESSION) is True

    for principal in (
        CustomerPrincipal(customer_id=CustomerId("cust_42")),
        GuestPrincipal(guest_instance_id="guest-instance-1"),
        AnonymousPrincipal(),
    ):
        assert await gate.check(principal, Operation.CREATE_PERSONA_SESSION) is False
