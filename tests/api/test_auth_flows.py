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

import time
from typing import Any, AsyncIterator

import httpx
import jwt
from fastapi import status
from typing_extensions import override
from parlant.core.container import Container
from pytest import fixture, mark

from parlant.api.app import create_api_app
from parlant.api.authentication import Principal, TokenAuthenticator
from fastapi import Request

from parlant.api.authorization import (
    AuthorizationPolicy,
    CompositeAuthorizationPolicy,
    DevelopmentAuthorizationPolicy,
    HeaderApiKeyAuthorizationPolicy,
    Operation,
    PermissionGate,
)
from parlant.core.agents import AgentId, AgentStore
from parlant.core.customers import CustomerId, CustomerKind, CustomerStore
from parlant.core.sessions import Session

from tests.test_utilities import create_agent


GUEST_TOKEN_HEADER = "X-Parlant-Guest-Token"
ADMIN_API_KEY = "test-admin-api-key"
CUSTOMER_JWT_SECRET = "integrator-shared-secret-0123456789abcdef"
PROJECT_TOKEN_HEADER = "X-Parlant-Project-Token"
PROJECT_TOKEN = "strict-project-token"


def mint_customer_jwt(customer_id: str, expires_in_seconds: int = 300) -> str:
    return jwt.encode(
        {"sub": customer_id, "exp": int(time.time()) + expires_in_seconds},
        CUSTOMER_JWT_SECRET,
        algorithm="HS256",
    )


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def make_composite_policy(
    container: Container,
    auto_provision_customers: bool = False,
) -> CompositeAuthorizationPolicy:
    return CompositeAuthorizationPolicy(
        authenticator=TokenAuthenticator(
            guest_token_secret="test-guest-secret-0123456789abcdef",
            admin_api_key=ADMIN_API_KEY,
            customer_jwt_secret=CUSTOMER_JWT_SECRET,
        ),
        customer_store=container[CustomerStore],
        auto_provision_customers=auto_provision_customers,
    )


@fixture
async def agent_id(container: Container) -> AgentId:
    agent = await create_agent(container, name="auth-flow-test-agent")
    return agent.id


@fixture
async def secure_client(container: Container) -> AsyncIterator[httpx.AsyncClient]:
    secure_container = container.clone()
    secure_container[AuthorizationPolicy] = make_composite_policy(container)

    app = await create_api_app(secure_container)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


@fixture
async def strict_client(container: Container) -> AsyncIterator[httpx.AsyncClient]:
    strict_container = container.clone()
    strict_container[AuthorizationPolicy] = HeaderApiKeyAuthorizationPolicy(
        api_key=PROJECT_TOKEN,
        header_name=PROJECT_TOKEN_HEADER,
    )

    app = await create_api_app(strict_container)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


@fixture
async def auto_provision_client(container: Container) -> AsyncIterator[httpx.AsyncClient]:
    secure_container = container.clone()
    secure_container[AuthorizationPolicy] = make_composite_policy(
        container, auto_provision_customers=True
    )

    app = await create_api_app(secure_container)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


@fixture
async def customer_alpha(container: Container) -> CustomerId:
    customer = await container[CustomerStore].create_customer(
        name="Customer Alpha", id=CustomerId("cust_alpha")
    )
    return customer.id


@fixture
async def customer_beta(container: Container) -> CustomerId:
    customer = await container[CustomerStore].create_customer(
        name="Customer Beta", id=CustomerId("cust_beta")
    )
    return customer.id


async def create_guest_session(
    client: httpx.AsyncClient,
    agent_id: AgentId,
    token: str | None = None,
) -> tuple[str, str]:
    """Creates a guest session, returning (session_id, guest_token)."""
    response = await client.post(
        "/sessions",
        json={"agent_id": agent_id},
        headers=bearer(token) if token else {},
    )

    assert response.status_code == status.HTTP_201_CREATED

    guest_token = token or response.headers.get(GUEST_TOKEN_HEADER)
    assert guest_token

    return response.json()["id"], guest_token


async def create_customer_session(
    client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> str:
    response = await client.post(
        "/sessions",
        json={"agent_id": agent_id},
        headers=bearer(mint_customer_jwt(customer_id)),
    )

    assert response.status_code == status.HTTP_201_CREATED

    return str(response.json()["id"])


###############################################################################
# Strict header API key flows
###############################################################################


async def test_that_the_strict_policy_rejects_an_agent_read_without_a_project_token(
    strict_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    response = await strict_client.get(f"/agents/{agent_id}")

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


async def test_that_the_strict_policy_rejects_an_agent_read_with_the_wrong_project_token(
    strict_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    response = await strict_client.get(
        f"/agents/{agent_id}",
        headers={PROJECT_TOKEN_HEADER: "wrong-project-token"},
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


async def test_that_the_strict_policy_rejects_duplicate_project_token_headers(
    strict_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    response = await strict_client.get(
        f"/agents/{agent_id}",
        headers=[
            (PROJECT_TOKEN_HEADER, PROJECT_TOKEN),
            (PROJECT_TOKEN_HEADER, PROJECT_TOKEN),
        ],
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == {"detail": "Invalid or missing API credential."}
    assert PROJECT_TOKEN not in response.text


async def test_that_the_strict_policy_failure_response_is_generic_and_does_not_leak_credentials(
    strict_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    wrong_credential = "supplied-wrong-project-token"
    response = await strict_client.get(
        f"/agents/{agent_id}",
        headers={PROJECT_TOKEN_HEADER: wrong_credential},
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == {"detail": "Invalid or missing API credential."}
    assert PROJECT_TOKEN not in response.text
    assert wrong_credential not in response.text


async def test_that_the_strict_policy_allows_an_agent_read_with_one_exact_project_token(
    strict_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    response = await strict_client.get(
        f"/agents/{agent_id}",
        headers={PROJECT_TOKEN_HEADER: PROJECT_TOKEN},
    )

    assert response.status_code == status.HTTP_200_OK


async def test_that_the_strict_policy_allows_a_session_mutation_with_one_exact_project_token(
    strict_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    response = await strict_client.post(
        "/sessions",
        json={"agent_id": agent_id},
        headers={PROJECT_TOKEN_HEADER: PROJECT_TOKEN},
    )

    assert response.status_code == status.HTTP_201_CREATED


###############################################################################
# Guest flows
###############################################################################


async def test_that_creating_a_session_without_credentials_returns_a_guest_session_and_token(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    response = await secure_client.post("/sessions", json={"agent_id": agent_id})

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["customer_id"] == "guest"
    assert response.headers.get(GUEST_TOKEN_HEADER)


async def test_that_a_guest_token_grants_reading_and_event_listing_on_its_own_session(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    session_id, guest_token = await create_guest_session(secure_client, agent_id)

    read_response = await secure_client.get(f"/sessions/{session_id}", headers=bearer(guest_token))
    assert read_response.status_code == status.HTTP_200_OK

    events_response = await secure_client.get(
        f"/sessions/{session_id}/events",
        params={"wait_for_data": 0},
        headers=bearer(guest_token),
    )
    assert events_response.status_code == status.HTTP_200_OK


async def test_that_a_guest_token_is_rejected_for_another_guests_session(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    session_a, _token_a = await create_guest_session(secure_client, agent_id)
    _session_b, token_b = await create_guest_session(secure_client, agent_id)

    read_response = await secure_client.get(f"/sessions/{session_a}", headers=bearer(token_b))
    assert read_response.status_code == status.HTTP_403_FORBIDDEN

    events_response = await secure_client.get(
        f"/sessions/{session_a}/events",
        params={"wait_for_data": 0},
        headers=bearer(token_b),
    )
    assert events_response.status_code == status.HTTP_403_FORBIDDEN


async def test_that_a_guest_token_cannot_access_a_customer_session(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_alpha: CustomerId,
) -> None:
    customer_session = await create_customer_session(secure_client, agent_id, customer_alpha)
    _guest_session, guest_token = await create_guest_session(secure_client, agent_id)

    events_response = await secure_client.get(
        f"/sessions/{customer_session}/events",
        params={"wait_for_data": 0},
        headers=bearer(guest_token),
    )
    assert events_response.status_code == status.HTTP_403_FORBIDDEN


async def test_that_a_guest_cannot_create_a_session_for_a_named_customer(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_alpha: CustomerId,
) -> None:
    _session, guest_token = await create_guest_session(secure_client, agent_id)

    response = await secure_client.post(
        "/sessions",
        json={"agent_id": agent_id, "customer_id": customer_alpha},
        headers=bearer(guest_token),
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN


async def test_that_a_returning_guest_token_owns_all_the_sessions_it_created(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    session_a, guest_token = await create_guest_session(secure_client, agent_id)

    second_response = await secure_client.post(
        "/sessions",
        json={"agent_id": agent_id},
        headers=bearer(guest_token),
    )
    assert second_response.status_code == status.HTTP_201_CREATED
    assert second_response.headers.get(GUEST_TOKEN_HEADER) is None
    session_b = second_response.json()["id"]

    for session_id in (session_a, session_b):
        events_response = await secure_client.get(
            f"/sessions/{session_id}/events",
            params={"wait_for_data": 0},
            headers=bearer(guest_token),
        )
        assert events_response.status_code == status.HTTP_200_OK


async def test_that_a_guest_can_post_a_message_into_its_own_session_but_not_a_foreign_one(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    session_a, token_a = await create_guest_session(secure_client, agent_id)
    _session_b, token_b = await create_guest_session(secure_client, agent_id)

    own_response = await secure_client.post(
        f"/sessions/{session_a}/events",
        json={"kind": "message", "source": "customer", "message": "hello"},
        headers=bearer(token_a),
    )
    assert own_response.status_code == status.HTTP_201_CREATED

    foreign_response = await secure_client.post(
        f"/sessions/{session_a}/events",
        json={"kind": "message", "source": "customer", "message": "sneaky"},
        headers=bearer(token_b),
    )
    assert foreign_response.status_code == status.HTTP_403_FORBIDDEN


###############################################################################
# Customer flows
###############################################################################


async def test_that_a_customer_jwt_creates_a_session_bound_to_the_token_customer(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_alpha: CustomerId,
) -> None:
    response = await secure_client.post(
        "/sessions",
        json={"agent_id": agent_id},
        headers=bearer(mint_customer_jwt(customer_alpha)),
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["customer_id"] == customer_alpha


async def test_that_a_body_customer_id_mismatching_the_token_is_rejected(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_alpha: CustomerId,
    customer_beta: CustomerId,
) -> None:
    response = await secure_client.post(
        "/sessions",
        json={"agent_id": agent_id, "customer_id": customer_beta},
        headers=bearer(mint_customer_jwt(customer_alpha)),
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN


async def test_that_a_customer_jwt_grants_event_listing_only_on_its_own_sessions(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_alpha: CustomerId,
    customer_beta: CustomerId,
) -> None:
    own_session = await create_customer_session(secure_client, agent_id, customer_alpha)
    foreign_session = await create_customer_session(secure_client, agent_id, customer_beta)

    own_response = await secure_client.get(
        f"/sessions/{own_session}/events",
        params={"wait_for_data": 0},
        headers=bearer(mint_customer_jwt(customer_alpha)),
    )
    assert own_response.status_code == status.HTTP_200_OK

    foreign_response = await secure_client.get(
        f"/sessions/{foreign_session}/events",
        params={"wait_for_data": 0},
        headers=bearer(mint_customer_jwt(customer_alpha)),
    )
    assert foreign_response.status_code == status.HTTP_403_FORBIDDEN


async def test_that_a_customer_jwt_cannot_post_a_message_into_another_customers_session(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_alpha: CustomerId,
    customer_beta: CustomerId,
) -> None:
    foreign_session = await create_customer_session(secure_client, agent_id, customer_beta)

    response = await secure_client.post(
        f"/sessions/{foreign_session}/events",
        json={"kind": "message", "source": "customer", "message": "sneaky"},
        headers=bearer(mint_customer_jwt(customer_alpha)),
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN


async def test_that_a_customer_can_post_a_message_into_its_own_session(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_alpha: CustomerId,
) -> None:
    session_id = await create_customer_session(secure_client, agent_id, customer_alpha)

    response = await secure_client.post(
        f"/sessions/{session_id}/events",
        json={"kind": "message", "source": "customer", "message": "hello there"},
        headers=bearer(mint_customer_jwt(customer_alpha)),
    )

    assert response.status_code == status.HTTP_201_CREATED


async def test_that_a_customer_sees_only_its_own_sessions_when_listing(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_alpha: CustomerId,
    customer_beta: CustomerId,
) -> None:
    own_session = await create_customer_session(secure_client, agent_id, customer_alpha)
    foreign_session = await create_customer_session(secure_client, agent_id, customer_beta)

    response = await secure_client.get(
        "/sessions",
        headers=bearer(mint_customer_jwt(customer_alpha)),
    )

    assert response.status_code == status.HTTP_200_OK

    session_ids = [s["id"] for s in response.json()]
    assert own_session in session_ids
    assert foreign_session not in session_ids


async def test_that_a_customer_cannot_explicitly_list_another_customers_sessions(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_alpha: CustomerId,
    customer_beta: CustomerId,
) -> None:
    await create_customer_session(secure_client, agent_id, customer_beta)

    response = await secure_client.get(
        "/sessions",
        params={"customer_id": customer_beta},
        headers=bearer(mint_customer_jwt(customer_alpha)),
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN


async def test_that_an_unknown_customer_in_a_valid_jwt_is_rejected_when_auto_provisioning_is_disabled(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    response = await secure_client.post(
        "/sessions",
        json={"agent_id": agent_id},
        headers=bearer(mint_customer_jwt("cust_never_seen")),
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


async def test_that_an_unknown_customer_in_a_valid_jwt_is_auto_provisioned_when_enabled(
    auto_provision_client: httpx.AsyncClient,
    agent_id: AgentId,
    container: Container,
) -> None:
    response = await auto_provision_client.post(
        "/sessions",
        json={"agent_id": agent_id},
        headers=bearer(mint_customer_jwt("cust_auto_provisioned")),
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["customer_id"] == "cust_auto_provisioned"

    customer = await container[CustomerStore].read_customer(CustomerId("cust_auto_provisioned"))
    assert customer.id == CustomerId("cust_auto_provisioned")


###############################################################################
# Admin flows
###############################################################################


async def test_that_an_admin_key_grants_full_api_access_including_entity_creation(
    secure_client: httpx.AsyncClient,
) -> None:
    create_customer_response = await secure_client.post(
        "/customers",
        json={"name": "admin-created-customer"},
        headers=bearer(ADMIN_API_KEY),
    )
    assert create_customer_response.status_code == status.HTTP_201_CREATED


async def test_that_an_admin_can_create_a_session_for_any_customer(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_alpha: CustomerId,
) -> None:
    response = await secure_client.post(
        "/sessions",
        json={"agent_id": agent_id, "customer_id": customer_alpha},
        headers=bearer(ADMIN_API_KEY),
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["customer_id"] == customer_alpha


async def test_that_an_admin_can_access_any_session(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_alpha: CustomerId,
) -> None:
    customer_session = await create_customer_session(secure_client, agent_id, customer_alpha)
    guest_session, _guest_token = await create_guest_session(secure_client, agent_id)

    for session_id in (customer_session, guest_session):
        response = await secure_client.get(
            f"/sessions/{session_id}/events",
            params={"wait_for_data": 0},
            headers=bearer(ADMIN_API_KEY),
        )
        assert response.status_code == status.HTTP_200_OK


###############################################################################
# Anonymous / invalid credentials
###############################################################################


async def test_that_a_request_without_credentials_cannot_access_a_session(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    session_id, _guest_token = await create_guest_session(secure_client, agent_id)

    read_response = await secure_client.get(f"/sessions/{session_id}")
    assert read_response.status_code == status.HTTP_403_FORBIDDEN

    events_response = await secure_client.get(
        f"/sessions/{session_id}/events",
        params={"wait_for_data": 0},
    )
    assert events_response.status_code == status.HTTP_403_FORBIDDEN


async def test_that_a_request_without_credentials_cannot_create_entities(
    secure_client: httpx.AsyncClient,
) -> None:
    response = await secure_client.post("/customers", json={"name": "sneaky-customer"})

    assert response.status_code == status.HTTP_403_FORBIDDEN


async def test_that_an_invalid_bearer_token_is_rejected_with_401(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    session_id, _guest_token = await create_guest_session(secure_client, agent_id)

    response = await secure_client.get(
        f"/sessions/{session_id}/events",
        params={"wait_for_data": 0},
        headers=bearer("complete-garbage"),
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


async def test_that_an_expired_customer_jwt_is_rejected_with_401(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_alpha: CustomerId,
) -> None:
    response = await secure_client.post(
        "/sessions",
        json={"agent_id": agent_id},
        headers=bearer(mint_customer_jwt(customer_alpha, expires_in_seconds=-10)),
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


async def test_that_only_an_administrator_can_access_diagnostic_reports(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_alpha: CustomerId,
) -> None:
    _session_id, guest_token = await create_guest_session(secure_client, agent_id)

    for path in ("/debug/reports", "/debug/reports/nonexistent"):
        guest_response = await secure_client.get(path, headers=bearer(guest_token))
        customer_response = await secure_client.get(
            path,
            headers=bearer(mint_customer_jwt(customer_alpha)),
        )
        admin_response = await secure_client.get(path, headers=bearer(ADMIN_API_KEY))

        assert guest_response.status_code == status.HTTP_403_FORBIDDEN
        assert customer_response.status_code == status.HTTP_403_FORBIDDEN
        expected_admin_status = (
            status.HTTP_200_OK if path == "/debug/reports" else status.HTTP_404_NOT_FOUND
        )
        assert admin_response.status_code == expected_admin_status


###############################################################################
# Development-policy regression
###############################################################################


async def test_that_the_development_policy_keeps_full_open_access(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_alpha: CustomerId,
) -> None:
    create_response = await async_client.post(
        "/sessions",
        json={"agent_id": agent_id, "customer_id": customer_alpha},
    )
    assert create_response.status_code == status.HTTP_201_CREATED

    session_id = create_response.json()["id"]

    events_response = await async_client.get(
        f"/sessions/{session_id}/events",
        params={"wait_for_data": 0},
    )
    assert events_response.status_code == status.HTTP_200_OK

    customers_response = await async_client.post("/customers", json={"name": "dev-customer"})
    assert customers_response.status_code == status.HTTP_201_CREATED


###############################################################################
# Attachment permission flows
###############################################################################


class _NoAttachmentsPolicy(DevelopmentAuthorizationPolicy):
    async def check_permission(self, request: Request, operation: Operation) -> bool:
        if operation is Operation.CREATE_CUSTOMER_EVENT_WITH_ATTACHMENTS:
            return False
        return await super().check_permission(request, operation)


@fixture
async def no_attachments_client(container: Container) -> AsyncIterator[httpx.AsyncClient]:
    restricted_container = container.clone()
    restricted_container[AuthorizationPolicy] = _NoAttachmentsPolicy()

    app = await create_api_app(restricted_container)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


def _attachment_message_body(with_attachment: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "kind": "message",
        "source": "customer",
        "message": "Here is my receipt.",
    }
    if with_attachment:
        body["attachments"] = [
            {"url": "https://files.example.com/receipt.pdf", "media_type": "application/pdf"}
        ]
    return body


async def test_that_a_policy_may_reject_customer_messages_with_attachments(
    no_attachments_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    session_id = (
        (await no_attachments_client.post("/sessions", json={"agent_id": agent_id}))
        .raise_for_status()
        .json()["id"]
    )

    denied = await no_attachments_client.post(
        f"/sessions/{session_id}/events",
        json=_attachment_message_body(with_attachment=True),
    )
    assert denied.status_code == status.HTTP_403_FORBIDDEN
    assert "attach" in denied.json()["detail"].lower()

    allowed = await no_attachments_client.post(
        f"/sessions/{session_id}/events",
        json=_attachment_message_body(with_attachment=False),
    )
    assert allowed.status_code == status.HTTP_201_CREATED


async def test_that_the_production_policy_allows_customer_attachments_by_default(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
    container: Container,
) -> None:
    customer = await container[CustomerStore].create_customer(name="Attaching Customer")
    session_id = await create_customer_session(secure_client, agent_id, customer.id)

    response = await secure_client.post(
        f"/sessions/{session_id}/events",
        json=_attachment_message_body(with_attachment=True),
        headers=bearer(mint_customer_jwt(customer.id)),
    )

    assert response.status_code == status.HTTP_201_CREATED


###############################################################################
# Custom event ownership
###############################################################################


class _OwnSessionsOnlyPolicy(DevelopmentAuthorizationPolicy):
    """Grants every operation, but only on sessions whose title is 'mine'."""

    async def check_session_access(self, request: Request, session: Session) -> bool:
        return session.title == "mine"


@fixture
async def own_sessions_only_client(container: Container) -> AsyncIterator[httpx.AsyncClient]:
    restricted_container = container.clone()
    restricted_container[AuthorizationPolicy] = _OwnSessionsOnlyPolicy()

    app = await create_api_app(restricted_container)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


async def test_that_custom_events_can_only_be_added_to_sessions_the_caller_may_access(
    own_sessions_only_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    async def create_session(title: str) -> str:
        response = await own_sessions_only_client.post(
            "/sessions", json={"agent_id": agent_id, "title": title}
        )
        return str(response.raise_for_status().json()["id"])

    page_view = {
        "kind": "custom",
        "source": "customer_ui",
        "data": {"type": "page_view", "page": "/pricing"},
    }

    allowed = await own_sessions_only_client.post(
        f"/sessions/{await create_session('mine')}/events", json=page_view
    )
    assert allowed.status_code == status.HTTP_201_CREATED

    denied = await own_sessions_only_client.post(
        f"/sessions/{await create_session('theirs')}/events", json=page_view
    )
    assert denied.status_code == status.HTTP_403_FORBIDDEN


###############################################################################
# /healthz body gating
###############################################################################


async def test_that_healthz_shows_only_status_without_the_read_health_report_permission(
    secure_client: httpx.AsyncClient,
) -> None:
    response = await secure_client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "checks" not in body
    assert "diagnostic_reports" not in body


async def test_that_an_admin_sees_the_full_healthz_body(
    secure_client: httpx.AsyncClient,
) -> None:
    response = await secure_client.get("/healthz", headers=bearer(ADMIN_API_KEY))

    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "checks" in body


async def test_that_a_forbidden_response_is_generic_and_does_not_echo_request_headers(
    secure_client: httpx.AsyncClient,
) -> None:
    # An anonymous caller hitting an admin-only route must not get their own
    # (proxy-injectable) request headers reflected back in the error body.
    marker = "10.1.2.3-internal-proxy-marker"
    response = await secure_client.get(
        "/groups",
        headers={"X-Forwarded-For": marker, "X-Internal-User": marker},
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert response.json() == {"detail": "Forbidden"}
    assert marker not in response.text


async def test_that_an_internal_error_response_does_not_expose_the_exception(
    container: Container,
    agent_id: AgentId,
) -> None:
    # A malformed enum value on an admin-readable list route raises a ValueError
    # deep in the handler; the generic 500 handler must not leak its text.
    # (A dedicated client that surfaces the 500 response rather than re-raising
    # the app exception, which is what the ASGI transport does by default.)
    secure_container = container.clone()
    secure_container[AuthorizationPolicy] = make_composite_policy(container)
    app = await create_api_app(secure_container)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as client:
        session_response = await client.post(
            "/sessions",
            json={"agent_id": agent_id},
            headers={"Authorization": f"Bearer {ADMIN_API_KEY}"},
        )
        session_id = session_response.json()["id"]

        response = await client.get(
            f"/sessions/{session_id}/events",
            params={"kinds": "definitely-not-a-kind"},
            headers={"Authorization": f"Bearer {ADMIN_API_KEY}"},
        )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    # The raw exception text (which echoes the caller's input and can carry
    # driver/provider internals) must not appear in the client-visible body.
    assert "definitely-not-a-kind" not in response.text
    assert "EventKindDTO" not in response.text


async def test_that_a_non_admin_agent_read_returns_only_id_and_name(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    # An anonymous caller may read an agent (the widget needs its name), but
    # must not receive its configuration — description carries the agent's
    # instructions, and models/moderation/review reveal deployment internals.
    response = await secure_client.get(f"/agents/{agent_id}")

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert set(body.keys()) == {"id", "name"}
    assert body["id"] == agent_id


@mark.parametrize("agent_name", ["docs", "redoc", "chat"])
async def test_that_an_agent_whose_id_looks_like_a_docs_or_ui_path_can_be_read_like_any_other(
    container: Container,
    secure_client: httpx.AsyncClient,
    agent_name: str,
) -> None:
    # The API docs and the integrated UI are gated by path. An agent's own path,
    # such as /agents/docs, must not fall under that gate just for containing it.
    agent = await container[AgentStore].create_agent(name="Docs", id=AgentId(agent_name))

    response = await secure_client.get(f"/agents/{agent.id}")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"id": agent_name, "name": "Docs"}


async def test_that_behind_a_url_prefix_the_docs_stay_gated_and_an_agent_named_docs_stays_readable(
    container: Container,
) -> None:
    # A reverse proxy may serve the API under a prefix (ROOT_PATH), so the gate
    # must match paths relative to it, not anywhere in the URL.
    secure_container = container.clone()
    secure_container[AuthorizationPolicy] = make_composite_policy(container)
    app = await create_api_app(secure_container)
    agent = await container[AgentStore].create_agent(name="Docs", id=AgentId("docs"))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, root_path="/ask3Sn2jas"),
        base_url="http://testserver",
    ) as client:
        docs_response = await client.get("/ask3Sn2jas/docs")
        agent_response = await client.get(f"/ask3Sn2jas/agents/{agent.id}")

    assert docs_response.status_code == status.HTTP_403_FORBIDDEN
    assert agent_response.status_code == status.HTTP_200_OK
    assert agent_response.json() == {"id": "docs", "name": "Docs"}


async def test_that_an_admin_agent_read_returns_only_identity(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    response = await secure_client.get(
        f"/agents/{agent_id}",
        headers={"Authorization": f"Bearer {ADMIN_API_KEY}"},
    )

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    # Agent configuration is code; even an admin gets only identity over REST.
    assert set(body.keys()) == {"id", "name"}
    assert body["id"] == agent_id


async def test_that_the_gated_ui_and_docs_return_403_rather_than_500_for_anonymous_callers(
    secure_client: httpx.AsyncClient,
) -> None:
    # These paths are authorized inside a middleware, where a raised
    # AuthorizationException bypasses the app's exception handlers; it must
    # still surface as a clean 403, never an internal server error.
    docs_response = await secure_client.get("/docs")
    assert docs_response.status_code == status.HTTP_403_FORBIDDEN
    assert docs_response.json() == {"detail": "Forbidden"}

    chat_response = await secure_client.get("/chat/index.html")
    assert chat_response.status_code == status.HTTP_403_FORBIDDEN
    assert chat_response.json() == {"detail": "Forbidden"}


class _GateWithoutPersonaSessions(PermissionGate):
    """A real gate that withholds exactly one capability.

    An admin passes every operation, so an admin cannot demonstrate that the
    persona capability is consulted at all. A customer principal is denied it
    but can only ever target its own id, and a persona's token is refused at
    login — so no stock principal can reach the check either.

    That is the point of the check rather than an argument against it: it is
    the second lock, and it holds even if the login rule is later loosened.
    Testing it needs a caller that is allowed ordinary customer sessions and
    denied persona ones, which is this gate.
    """

    @override
    async def check(self, principal: Principal, operation: Operation) -> bool:
        return operation is not Operation.CREATE_PERSONA_SESSION


@fixture
async def persona_restricted_client(container: Container) -> AsyncIterator[httpx.AsyncClient]:
    restricted_container = container.clone()
    restricted_container[AuthorizationPolicy] = CompositeAuthorizationPolicy(
        authenticator=TokenAuthenticator(
            guest_token_secret="test-guest-secret-0123456789abcdef",
            admin_api_key=ADMIN_API_KEY,
            customer_jwt_secret=CUSTOMER_JWT_SECRET,
        ),
        permission_gate=_GateWithoutPersonaSessions(),
        customer_store=container[CustomerStore],
    )

    app = await create_api_app(restricted_container)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


@fixture
async def persona_customer(container: Container) -> CustomerId:
    customer = await container[CustomerStore].create_customer(
        name="Test Persona",
        id=CustomerId("cust_persona"),
        kind=CustomerKind.PERSONA,
    )
    return customer.id


async def test_that_starting_a_session_with_a_persona_requires_the_persona_permission(
    persona_restricted_client: httpx.AsyncClient,
    agent_id: AgentId,
    persona_customer: CustomerId,
) -> None:
    response = await persona_restricted_client.post(
        "/sessions",
        json={"agent_id": agent_id, "customer_id": persona_customer},
        headers=bearer(ADMIN_API_KEY),
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN


async def test_that_starting_a_session_with_a_real_customer_does_not_require_the_persona_permission(
    persona_restricted_client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_alpha: CustomerId,
) -> None:
    response = await persona_restricted_client.post(
        "/sessions",
        json={"agent_id": agent_id, "customer_id": customer_alpha},
        headers=bearer(ADMIN_API_KEY),
    )

    assert response.status_code == status.HTTP_201_CREATED


async def test_that_starting_a_session_with_an_unknown_customer_id_still_works(
    persona_restricted_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    """Session creation does not require the customer to exist, and must not start.

    Only the agent is validated today, so a session can be opened for a
    customer id that was never stored, and roughly four dozen call sites in
    this suite rely on it. Reading the customer to learn its kind must treat a
    miss as a real customer: an id nobody stored is not a persona, so the
    lookup failing is not grounds for demanding the persona permission, nor
    for rejecting the request.
    """
    response = await persona_restricted_client.post(
        "/sessions",
        json={"agent_id": agent_id, "customer_id": "cust_never_stored"},
        headers=bearer(ADMIN_API_KEY),
    )

    assert response.status_code == status.HTTP_201_CREATED


async def test_that_an_admin_can_start_a_session_with_a_persona(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
    persona_customer: CustomerId,
) -> None:
    response = await secure_client.post(
        "/sessions",
        json={"agent_id": agent_id, "customer_id": persona_customer},
        headers=bearer(ADMIN_API_KEY),
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["customer_id"] == persona_customer


async def test_that_a_persona_token_cannot_create_a_session(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
    persona_customer: CustomerId,
) -> None:
    """The first lock, seen from the outside: a persona cannot authenticate."""
    response = await secure_client.post(
        "/sessions",
        json={"agent_id": agent_id},
        headers=bearer(mint_customer_jwt(persona_customer)),
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


# Guest token renewal
###############################################################################

DAY = 24 * 60 * 60


def age_guest_token(guest_token: str, *, issued_days_ago: float, expires_in_days: float) -> str:
    """The same guest's token, re-minted as if it had been issued days ago."""
    claims = jwt.decode(guest_token, options={"verify_signature": False})
    now = int(time.time())
    issued_at = now - int(issued_days_ago * DAY)

    return jwt.encode(
        {
            **claims,
            "iat": issued_at,
            "identity_iat": issued_at,
            "exp": now + int(expires_in_days * DAY),
        },
        "test-guest-secret-0123456789abcdef",
        algorithm="HS256",
    )


async def test_that_listing_events_with_an_aging_guest_token_returns_a_renewed_token_in_the_guest_token_header(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    session_id, guest_token = await create_guest_session(secure_client, agent_id)
    aging_token = age_guest_token(guest_token, issued_days_ago=4, expires_in_days=3)

    response = await secure_client.get(
        f"/sessions/{session_id}/events",
        params={"wait_for_data": 0},
        headers=bearer(aging_token),
    )

    assert response.status_code == status.HTTP_200_OK
    renewed_token = response.headers.get(GUEST_TOKEN_HEADER)
    assert renewed_token
    assert renewed_token != aging_token


async def test_that_a_renewed_guest_token_reads_the_guests_existing_sessions(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    session_id, guest_token = await create_guest_session(secure_client, agent_id)
    aging_token = age_guest_token(guest_token, issued_days_ago=4, expires_in_days=3)

    renewing_response = await secure_client.get(
        f"/sessions/{session_id}", headers=bearer(aging_token)
    )
    renewed_token = renewing_response.headers[GUEST_TOKEN_HEADER]

    events_response = await secure_client.get(
        f"/sessions/{session_id}/events",
        params={"wait_for_data": 0},
        headers=bearer(renewed_token),
    )
    assert events_response.status_code == status.HTTP_200_OK


async def test_that_a_fresh_guest_token_gets_no_renewal_header(
    secure_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    session_id, guest_token = await create_guest_session(secure_client, agent_id)

    response = await secure_client.get(
        f"/sessions/{session_id}/events",
        params={"wait_for_data": 0},
        headers=bearer(guest_token),
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.headers.get(GUEST_TOKEN_HEADER) is None
