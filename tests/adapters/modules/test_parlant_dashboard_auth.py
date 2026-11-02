# Copyright 2026 Parlant (Emcie Co Ltd.)
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

"""The Dashboard auth policy is a project-token trust wrapper: a valid
X-Parlant-Project-Token short-circuits every check to admin semantics, and
everything else delegates to an inner policy — by default the OSS composite
policy (guest tokens, customer JWTs, session ownership) built from the
standard PARLANT_* environment variables."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping

import jwt
import pytest
from typing_extensions import override

from parlant.adapters.modules.parlant_dashboard.auth import (
    ParlantDashboardAuthorizationPolicy,
    build_default_inner_policy,
)
from parlant.adapters.modules.parlant_dashboard.config import PROJECT_TOKEN_HEADER
from parlant.api.authentication import AdminPrincipal, GuestPrincipal, TokenAuthenticator
from parlant.api.authorization import (
    GUEST_SESSION_METADATA_PATH,
    AuthorizationPolicy,
    CompositeAuthorizationPolicy,
    Operation,
)
from parlant.core import metadata
from parlant.core.agents import AgentId
from parlant.core.customers import CustomerStore
from parlant.core.sessions import Session, SessionId


PROJECT_TOKEN = "project-token"
GUEST_SECRET = "guest-secret-0123456789abcdef"


@dataclass
class FakeWebSocket:
    headers: Mapping[str, str]
    query_params: Mapping[str, str] = field(default_factory=dict)


@dataclass
class FakeRequest:
    headers: Mapping[str, str]
    state: Any = field(default_factory=SimpleNamespace)


class _SpyInner(AuthorizationPolicy):
    """Records delegation and answers with fixed verdicts."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    @override
    def name(self) -> str:
        return "spy"

    @override
    async def check_permission(self, request: Any, operation: Operation) -> bool:
        self.calls.append(f"check_permission:{operation.value}")
        return False

    @override
    async def check_rate_limit(self, request: Any, operation: Operation) -> bool:
        self.calls.append("check_rate_limit")
        return False

    @override
    async def check_websocket_permission(self, websocket: Any, operation: Operation) -> bool:
        self.calls.append("check_websocket_permission")
        return False

    @override
    async def check_session_access(self, request: Any, session: Session) -> bool:
        self.calls.append("check_session_access")
        return False

    @override
    def issue_guest_token(self, guest_instance_id: str) -> str | None:
        self.calls.append("issue_guest_token")
        return "spy-guest-token"


def _composite_inner() -> CompositeAuthorizationPolicy:
    return CompositeAuthorizationPolicy(
        authenticator=TokenAuthenticator(
            guest_token_secret=GUEST_SECRET,
            admin_api_key="inner-admin-key",
        ),
    )


def _wrapped(inner: AuthorizationPolicy) -> ParlantDashboardAuthorizationPolicy:
    return ParlantDashboardAuthorizationPolicy(project_token=PROJECT_TOKEN, inner=inner)


def _guest_session(guest_instance_id: str) -> Session:
    now = datetime.now(timezone.utc)
    return Session(
        id=SessionId("session-1"),
        creation_utc=now,
        modified_utc=now,
        customer_id=CustomerStore.GUEST_ID,
        agent_id=AgentId("agent-1"),
        mode="auto",
        title=None,
        agent_states=[],
        metadata=metadata.set_internal_value({}, GUEST_SESSION_METADATA_PATH, guest_instance_id),
    )


# ─────────────────────────────── trusted tier ───────────────────────────────


async def test_that_the_project_token_grants_every_api_operation() -> None:
    policy = _wrapped(_SpyInner())
    request = FakeRequest(headers={PROJECT_TOKEN_HEADER: PROJECT_TOKEN})

    for operation in (Operation.LIST_AGENTS, Operation.READ_HEALTH_REPORT):
        assert await policy.check_permission(request, operation) is True  # type: ignore[arg-type]
    assert await policy.check_rate_limit(request, Operation.LIST_AGENTS) is True  # type: ignore[arg-type]
    assert (
        await policy.check_session_access(request, _guest_session("someone"))  # type: ignore[arg-type]
        is True
    )


async def test_that_the_project_token_can_stream_logs() -> None:
    policy = _wrapped(_SpyInner())
    websocket = FakeWebSocket(headers={PROJECT_TOKEN_HEADER: PROJECT_TOKEN})

    assert (
        await policy.check_websocket_permission(websocket, Operation.STREAM_LOGS)  # type: ignore[arg-type]
        is True
    )


async def test_that_the_project_token_resolves_an_admin_principal() -> None:
    policy = _wrapped(_SpyInner())
    request = FakeRequest(headers={PROJECT_TOKEN_HEADER: PROJECT_TOKEN})

    assert isinstance(await policy.resolve_principal(request), AdminPrincipal)  # type: ignore[arg-type]


async def test_that_a_localhost_host_header_grants_no_trust() -> None:
    # The Host header is client-controlled; it must never be a trust signal.
    inner = _SpyInner()
    policy = _wrapped(inner)

    for host in ("localhost:2222", "127.0.0.1:2222"):
        request = FakeRequest(headers={"host": host})
        assert (
            await policy.check_permission(request, Operation.LIST_AGENTS)  # type: ignore[arg-type]
            is False
        )
    assert inner.calls  # denied by delegation, not by the wrapper itself


# ────────────────────────── delegation to the inner ─────────────────────────


async def test_that_tokenless_requests_delegate_every_check_to_the_inner_policy() -> None:
    inner = _SpyInner()
    policy = _wrapped(inner)
    request = FakeRequest(headers={})

    assert await policy.check_permission(request, Operation.READ_AGENT) is False  # type: ignore[arg-type]
    assert await policy.check_rate_limit(request, Operation.READ_AGENT) is False  # type: ignore[arg-type]
    assert await policy.check_session_access(request, _guest_session("g")) is False  # type: ignore[arg-type]
    assert (
        await policy.check_websocket_permission(
            FakeWebSocket(headers={}),  # type: ignore[arg-type]
            Operation.STREAM_LOGS,
        )
        is False
    )
    assert policy.issue_guest_token("instance-1") == "spy-guest-token"

    assert inner.calls == [
        "check_permission:read_agent",
        "check_rate_limit",
        "check_session_access",
        "check_websocket_permission",
        "issue_guest_token",
    ]


async def test_that_a_wrong_project_token_falls_through_to_the_inner_policy() -> None:
    inner = _SpyInner()
    policy = _wrapped(inner)
    request = FakeRequest(headers={PROJECT_TOKEN_HEADER: "wrong-token"})

    assert (
        await policy.check_permission(request, Operation.LIST_AGENTS)  # type: ignore[arg-type]
        is False
    )
    assert inner.calls == ["check_permission:list_agents"]


# ─────────────────── untrusted tier = real guest/customer auth ──────────────


async def test_that_anonymous_callers_can_start_a_guest_conversation_but_not_read_sessions() -> (
    None
):
    policy = _wrapped(_composite_inner())
    request = FakeRequest(headers={})

    assert await policy.check_permission(request, Operation.READ_AGENT) is True  # type: ignore[arg-type]
    assert (
        await policy.check_permission(request, Operation.CREATE_GUEST_SESSION)  # type: ignore[arg-type]
        is True
    )
    assert await policy.check_permission(request, Operation.READ_SESSION) is False  # type: ignore[arg-type]
    assert await policy.check_permission(request, Operation.LIST_AGENTS) is False  # type: ignore[arg-type]


async def test_that_a_guest_token_round_trip_grants_access_to_the_owned_session_only() -> None:
    policy = _wrapped(_composite_inner())

    guest_token = policy.issue_guest_token("instance-1")
    assert guest_token is not None

    request = FakeRequest(headers={"authorization": f"Bearer {guest_token}"})

    assert await policy.check_permission(request, Operation.READ_SESSION) is True  # type: ignore[arg-type]
    assert (
        await policy.check_session_access(request, _guest_session("instance-1"))  # type: ignore[arg-type]
        is True
    )

    other_request = FakeRequest(headers={"authorization": f"Bearer {guest_token}"})
    assert (
        await policy.check_session_access(other_request, _guest_session("someone-else"))  # type: ignore[arg-type]
        is False
    )


# ───────────────────────────── default inner policy ─────────────────────────


async def test_that_the_default_inner_policy_is_the_composite_with_guest_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PARLANT_ADMIN_API_KEY", raising=False)
    monkeypatch.delenv("PARLANT_CUSTOMER_JWT_SECRET", raising=False)
    monkeypatch.delenv("PARLANT_CUSTOMER_JWKS_URL", raising=False)

    inner = build_default_inner_policy(customer_store=None)

    assert isinstance(inner, CompositeAuthorizationPolicy)
    assert inner.issue_guest_token("instance-1")


async def test_that_the_default_inner_policy_takes_the_maximum_guest_identity_age_from_the_environment_and_defaults_to_28_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    day = 24 * 60 * 60
    now = int(time.time())
    monkeypatch.setenv("PARLANT_GUEST_TOKEN_SECRET", GUEST_SECRET)

    async def renewed_expiry(identity_started: int) -> int:
        aging_token = jwt.encode(
            {
                "sub": "instance-1",
                "typ": "parlant:guest",
                "iat": now - 4 * day,
                "identity_iat": identity_started,
                "exp": now + 3 * day,
            },
            GUEST_SECRET,
            algorithm="HS256",
        )
        authenticator = build_default_inner_policy(customer_store=None).authenticator
        assert isinstance(authenticator, TokenAuthenticator)
        principal = await authenticator.resolve(
            FakeRequest(headers={"authorization": f"Bearer {aging_token}"})  # type: ignore[arg-type]
        )
        assert isinstance(principal, GuestPrincipal)
        renewed = authenticator.renew_guest_token(principal)
        assert renewed is not None
        return int(jwt.decode(renewed, GUEST_SECRET, algorithms=["HS256"])["exp"])

    monkeypatch.setenv("PARLANT_GUEST_IDENTITY_MAX_AGE_DAYS", "10")
    assert await renewed_expiry(now - 6 * day) == now - 6 * day + 10 * day

    monkeypatch.delenv("PARLANT_GUEST_IDENTITY_MAX_AGE_DAYS")
    assert await renewed_expiry(now - 22 * day) == now - 22 * day + 28 * day


async def test_that_the_default_inner_policy_rejects_conflicting_customer_jwt_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARLANT_CUSTOMER_JWT_SECRET", "secret")
    monkeypatch.setenv("PARLANT_CUSTOMER_JWKS_URL", "https://idp.example.com/jwks.json")

    with pytest.raises(ValueError):
        build_default_inner_policy(customer_store=None)
