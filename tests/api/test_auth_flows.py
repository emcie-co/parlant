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
from typing import AsyncIterator

import httpx
import jwt
from fastapi import status
from lagom import Container
from pytest import fixture

from parlant.api.app import create_api_app
from parlant.api.authentication import TokenAuthenticator
from parlant.api.authorization import AuthorizationPolicy, CompositeAuthorizationPolicy
from parlant.core.agents import AgentId
from parlant.core.customers import CustomerId, CustomerStore

from tests.test_utilities import create_agent


GUEST_TOKEN_HEADER = "X-Parlant-Guest-Token"
ADMIN_API_KEY = "test-admin-api-key"
CUSTOMER_JWT_SECRET = "integrator-shared-secret-0123456789abcdef"


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
    create_agent_response = await secure_client.post(
        "/agents",
        json={"name": "admin-created-agent"},
        headers=bearer(ADMIN_API_KEY),
    )
    assert create_agent_response.status_code == status.HTTP_201_CREATED

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
    response = await secure_client.post("/agents", json={"name": "sneaky-agent"})

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

    agents_response = await async_client.post("/agents", json={"name": "dev-agent"})
    assert agents_response.status_code == status.HTTP_201_CREATED
