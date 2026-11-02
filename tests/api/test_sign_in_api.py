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

"""The API side of signing in: message scopes and reserved sign-in events.

Each customer message records, under the reserved ``__parlant__`` metadata
namespace, the scopes of the token it was sent with. Scopes belong to that
message only: a later message sent with a narrower token has narrower scopes.
They are honored only from short-lived tokens, so a long-lived or non-expiring
token can still send messages but grants no scopes.

The sign-in event types are emitted only by Parlant itself, so a client can't
post one to fake a sign-in prompt or a completed sign-in.
"""

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import jwt
from fastapi import status

from parlant.api.app import create_api_app
from parlant.api.authentication import TokenAuthenticator
from parlant.api.authorization import AuthorizationPolicy, CompositeAuthorizationPolicy
from parlant.core import metadata as metadata_utils
from parlant.core.agents import AgentId
from parlant.core.application import Application
from parlant.core.container import Container
from parlant.core.customers import CustomerId, CustomerStore
from parlant.core.sessions import EventKind, EventSource, SessionId, SessionStore
from pytest import fixture

from tests.test_utilities import create_agent


GUEST_TOKEN_HEADER = "X-Parlant-Guest-Token"
ADMIN_API_KEY = "test-admin-api-key"
CUSTOMER_JWT_SECRET = "integrator-shared-secret-0123456789abcdef"


def mint_customer_jwt(
    customer_id: str,
    *,
    lifetime_seconds: int | None = 300,
    **claims: Any,
) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {"sub": customer_id, "iat": now, **claims}

    if lifetime_seconds is not None:
        payload["exp"] = now + lifetime_seconds

    return jwt.encode(payload, CUSTOMER_JWT_SECRET, algorithm="HS256")


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


ClientFactory = Callable[..., Any]


@fixture
def client_with(container: Container) -> ClientFactory:
    """A secured API client whose authenticator takes the given options."""

    @asynccontextmanager
    async def client(**authenticator_options: Any) -> AsyncIterator[httpx.AsyncClient]:
        secure_container = container.clone()
        secure_container[AuthorizationPolicy] = CompositeAuthorizationPolicy(
            authenticator=TokenAuthenticator(
                guest_token_secret="test-guest-secret-0123456789abcdef",
                admin_api_key=ADMIN_API_KEY,
                customer_jwt_secret=CUSTOMER_JWT_SECRET,
                **authenticator_options,
            ),
            customer_store=container[CustomerStore],
        )

        app = await create_api_app(secure_container)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as api_client:
            yield api_client

    return client


@fixture
async def agent_id(container: Container) -> AgentId:
    agent = await create_agent(container, name="message-scopes-test-agent")
    return agent.id


@fixture
async def customer_id(container: Container) -> CustomerId:
    customer = await container[CustomerStore].create_customer(
        name="Customer Alpha", id=CustomerId("cust_alpha")
    )
    return customer.id


async def create_session(
    client: httpx.AsyncClient,
    agent_id: AgentId,
    token: str | None = None,
) -> tuple[str, str]:
    """Creates a session, returning (session_id, token) — a guest token when
    none was given."""
    response = await client.post(
        "/sessions",
        json={"agent_id": agent_id},
        headers=bearer(token) if token else {},
    )

    assert response.status_code == status.HTTP_201_CREATED

    session_token = token or response.headers.get(GUEST_TOKEN_HEADER)
    assert session_token

    return response.json()["id"], session_token


async def post_message(
    client: httpx.AsyncClient,
    session_id: str,
    token: str,
    message: str = "hello",
    metadata: dict[str, Any] | None = None,
) -> httpx.Response:
    body: dict[str, Any] = {"kind": "message", "source": "customer", "message": message}

    if metadata is not None:
        body["metadata"] = metadata

    return await client.post(
        f"/sessions/{session_id}/events",
        json=body,
        headers=bearer(token),
    )


async def recorded_scopes(container: Container, session_id: str) -> list[Any]:
    """The recorded scopes of each customer message in the session, in order
    (``None`` for a message that recorded none)."""
    events = await container[SessionStore].list_events(
        SessionId(session_id),
        source=EventSource.CUSTOMER,
        kinds=[EventKind.MESSAGE],
    )

    return [metadata_utils.internal_value(e.metadata, ("auth", "scopes")) for e in events]


async def test_that_a_customer_message_records_the_scopes_of_the_token_it_was_sent_with(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        token = mint_customer_jwt(customer_id, scope="flights:book profile")
        session_id, _ = await create_session(client, agent_id, token)

        response = await post_message(client, session_id, token)

    assert response.status_code == status.HTTP_201_CREATED
    assert await recorded_scopes(container, session_id) == [["flights:book", "profile"]]


async def test_that_scopes_are_read_from_the_configured_claim(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with(customer_scope_claim="scp") as client:
        token = mint_customer_jwt(
            customer_id,
            scope="ignored",
            scp=["flights:book", "profile"],
        )
        session_id, _ = await create_session(client, agent_id, token)

        await post_message(client, session_id, token)

    assert await recorded_scopes(container, session_id) == [["flights:book", "profile"]]


async def test_that_scopes_from_a_token_without_expiry_are_not_honored(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        token = mint_customer_jwt(customer_id, lifetime_seconds=None, scope="flights:book")
        session_id, _ = await create_session(client, agent_id, token)

        response = await post_message(client, session_id, token)

    assert response.status_code == status.HTTP_201_CREATED
    assert await recorded_scopes(container, session_id) == [None]


async def test_that_scopes_from_a_long_lived_token_are_not_honored(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with(max_scope_token_lifetime_seconds=3600) as client:
        long_lived = mint_customer_jwt(customer_id, lifetime_seconds=7200, scope="flights:book")
        session_id, _ = await create_session(client, agent_id, long_lived)

        response = await post_message(client, session_id, long_lived)

    assert response.status_code == status.HTTP_201_CREATED
    assert await recorded_scopes(container, session_id) == [None]


async def test_that_each_message_records_only_the_scopes_it_was_sent_with(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        broad = mint_customer_jwt(customer_id, scope="flights:book profile")
        narrow = mint_customer_jwt(customer_id, scope="profile")
        session_id, _ = await create_session(client, agent_id, broad)

        await post_message(client, session_id, broad, "book it")
        await post_message(client, session_id, narrow, "and what's my name?")

    assert await recorded_scopes(container, session_id) == [
        ["flights:book", "profile"],
        ["profile"],
    ]


async def test_that_guest_messages_carry_no_scopes(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
) -> None:
    async with client_with() as client:
        session_id, guest_token = await create_session(client, agent_id)

        response = await post_message(client, session_id, guest_token)

    assert response.status_code == status.HTTP_201_CREATED
    assert await recorded_scopes(container, session_id) == [None]


async def test_that_a_client_cannot_plant_scopes_through_message_metadata(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        token = mint_customer_jwt(customer_id)
        session_id, _ = await create_session(client, agent_id, token)

        await post_message(
            client,
            session_id,
            token,
            metadata={"__parlant__": {"auth": {"scopes": ["flights:book"]}}},
        )

    assert all(scopes is None for scopes in await recorded_scopes(container, session_id))


async def test_that_not_even_an_admin_can_post_a_reserved_custom_event_type(
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        token = mint_customer_jwt(customer_id)
        session_id, _ = await create_session(client, agent_id, token)

        responses = [
            await client.post(
                f"/sessions/{session_id}/events",
                json={"kind": "custom", "source": "customer_ui", "data": {"type": reserved}},
                headers=bearer(ADMIN_API_KEY),
            )
            for reserved in ("authorization_required", "customer.signed_in")
        ]

    assert [r.status_code for r in responses] == [status.HTTP_422_UNPROCESSABLE_CONTENT] * 2


###############################################################################
# Claiming a guest session
###############################################################################


@fixture
async def other_customer_id(container: Container) -> CustomerId:
    customer = await container[CustomerStore].create_customer(
        name="Customer Beta", id=CustomerId("cust_beta")
    )
    return customer.id


async def claim(
    client: httpx.AsyncClient,
    session_id: str,
    customer_token: str,
    guest_token: str,
) -> httpx.Response:
    return await client.post(
        f"/sessions/{session_id}/claim",
        json={"guest_token": guest_token},
        headers=bearer(customer_token),
    )


async def signed_in_events(container: Container, session_id: str) -> list[Any]:
    events = await container[SessionStore].list_events(
        SessionId(session_id),
        source=EventSource.SYSTEM,
        kinds=[EventKind.CUSTOM],
    )
    return [
        e for e in events if isinstance(e.data, dict) and e.data.get("type") == "customer.signed_in"
    ]


async def test_that_a_customer_can_claim_their_guest_session(
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        session_id, guest_token = await create_session(client, agent_id)
        customer_token = mint_customer_jwt(customer_id)

        response = await claim(client, session_id, customer_token, guest_token)

        as_customer = await client.get(f"/sessions/{session_id}", headers=bearer(customer_token))
        as_guest = await client.get(f"/sessions/{session_id}", headers=bearer(guest_token))

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["customer_id"] == customer_id
    assert as_customer.status_code == status.HTTP_200_OK
    assert as_guest.status_code == status.HTTP_403_FORBIDDEN


async def test_that_claiming_needs_the_token_of_the_guest_who_owns_the_session(
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        session_id, _ = await create_session(client, agent_id)
        _, another_guest_token = await create_session(client, agent_id)

        response = await claim(
            client, session_id, mint_customer_jwt(customer_id), another_guest_token
        )

    assert response.status_code == status.HTTP_403_FORBIDDEN


async def test_that_a_customers_session_cannot_be_claimed(
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
    other_customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        _, guest_token = await create_session(client, agent_id)
        session_id, _ = await create_session(client, agent_id, mint_customer_jwt(customer_id))

        response = await claim(
            client, session_id, mint_customer_jwt(other_customer_id), guest_token
        )

    assert response.status_code in (status.HTTP_403_FORBIDDEN, status.HTTP_409_CONFLICT)


async def test_that_only_one_of_two_concurrent_claims_wins(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
    other_customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        session_id, guest_token = await create_session(client, agent_id)

        responses = await asyncio.gather(
            claim(client, session_id, mint_customer_jwt(customer_id), guest_token),
            claim(client, session_id, mint_customer_jwt(other_customer_id), guest_token),
        )

    codes = sorted(r.status_code for r in responses)
    winner = next(r for r in responses if r.status_code == status.HTTP_200_OK)
    session = await container[SessionStore].read_session(SessionId(session_id))

    assert codes == [status.HTTP_200_OK, status.HTTP_409_CONFLICT]
    assert session.customer_id == winner.json()["customer_id"]
    assert len(await signed_in_events(container, session_id)) == 1


async def test_that_claiming_without_a_customer_token_is_rejected(
    client_with: ClientFactory,
    agent_id: AgentId,
) -> None:
    async with client_with() as client:
        session_id, guest_token = await create_session(client, agent_id)

        response = await claim(client, session_id, guest_token, guest_token)

    assert response.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)


async def test_that_a_claim_records_that_the_customer_signed_in(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        session_id, guest_token = await create_session(client, agent_id)

        await claim(client, session_id, mint_customer_jwt(customer_id), guest_token)

    events = await signed_in_events(container, session_id)

    assert [e.data for e in events] == [
        {"type": "customer.signed_in", "customer_id": customer_id, "name": "Customer Alpha"}
    ]


async def test_that_a_claim_records_the_scopes_of_its_token(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        session_id, guest_token = await create_session(client, agent_id)

        await claim(
            client,
            session_id,
            mint_customer_jwt(customer_id, scope="flights:book"),
            guest_token,
        )

    [event] = await signed_in_events(container, session_id)

    assert metadata_utils.internal_value(event.metadata, ("auth", "scopes")) == ["flights:book"]


###############################################################################
# Allowing more, once signed in (step-up)
###############################################################################


async def authorize(
    client: httpx.AsyncClient,
    session_id: str,
    customer_token: str,
) -> httpx.Response:
    return await client.post(
        f"/sessions/{session_id}/authorize",
        headers=bearer(customer_token),
    )


async def authorized_events(container: Container, session_id: str) -> list[Any]:
    events = await container[SessionStore].list_events(
        SessionId(session_id),
        source=EventSource.SYSTEM,
        kinds=[EventKind.CUSTOM],
    )
    return [
        e
        for e in events
        if isinstance(e.data, dict) and e.data.get("type") == "customer.authorized"
    ]


async def test_that_a_signed_in_customer_can_allow_more(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        session_id, _ = await create_session(
            client, agent_id, mint_customer_jwt(customer_id, scope="orders:read")
        )

        response = await authorize(
            client,
            session_id,
            mint_customer_jwt(customer_id, scope="orders:read orders:dispute"),
        )

    assert response.status_code == status.HTTP_200_OK

    [event] = await authorized_events(container, session_id)

    assert event.data == {"type": "customer.authorized", "customer_id": customer_id}
    assert metadata_utils.internal_value(event.metadata, ("auth", "scopes")) == [
        "orders:read",
        "orders:dispute",
    ]


async def test_that_only_the_sessions_own_customer_can_allow_more(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
    other_customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        session_id, _ = await create_session(client, agent_id, mint_customer_jwt(customer_id))
        guest_session_id, guest_token = await create_session(client, agent_id)

        as_other_customer = await authorize(
            client, session_id, mint_customer_jwt(other_customer_id, scope="orders:read")
        )
        as_guest = await authorize(client, guest_session_id, guest_token)
        on_a_guests_session = await authorize(
            client, guest_session_id, mint_customer_jwt(customer_id, scope="orders:read")
        )

    assert as_other_customer.status_code == status.HTTP_403_FORBIDDEN
    assert as_guest.status_code == status.HTTP_403_FORBIDDEN
    assert on_a_guests_session.status_code == status.HTTP_403_FORBIDDEN
    assert await authorized_events(container, session_id) == []


async def test_that_customer_authorized_is_a_reserved_event_type(
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        session_id, _ = await create_session(client, agent_id, mint_customer_jwt(customer_id))

        response = await client.post(
            f"/sessions/{session_id}/events",
            json={
                "kind": "custom",
                "source": "customer_ui",
                "data": {"type": "customer.authorized"},
            },
            headers=bearer(ADMIN_API_KEY),
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


###############################################################################
# Claiming with a prepared claim secret
#
# A session the platform creates for an AI agent has no guest owner, so there
# is no guest token to prove ownership with. The platform prepares a claim
# instead: Parlant keeps only the hash of a one-time secret and its expiry, and
# whoever holds the secret, with the customer's token, may claim the session.
###############################################################################


def secret_sha256(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


async def create_platform_session(client: httpx.AsyncClient, agent_id: AgentId) -> str:
    """A session as the platform creates one for an AI agent: no guest owner."""
    response = await client.post(
        "/sessions",
        json={"agent_id": agent_id, "customer_represented_by_an_agent": True},
        headers=bearer(ADMIN_API_KEY),
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert GUEST_TOKEN_HEADER not in response.headers

    return str(response.json()["id"])


async def prepare_claim(
    container: Container,
    session_id: str,
    secret: str,
    expires_in: timedelta = timedelta(minutes=15),
) -> None:
    await container[Application].sessions.prepare_claim(
        SessionId(session_id),
        secret_sha256=secret_sha256(secret),
        expires_at=datetime.now(timezone.utc) + expires_in,
    )


async def claim_with_secret(
    client: httpx.AsyncClient,
    session_id: str,
    customer_token: str,
    claim_secret: str,
) -> httpx.Response:
    return await client.post(
        f"/sessions/{session_id}/claim",
        json={"claim_secret": claim_secret},
        headers=bearer(customer_token),
    )


async def test_that_a_customer_can_claim_a_session_with_a_prepared_claim_secret(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        session_id = await create_platform_session(client, agent_id)
        await prepare_claim(container, session_id, "s3cret-one-time")

        response = await claim_with_secret(
            client, session_id, mint_customer_jwt(customer_id), "s3cret-one-time"
        )

    session = await container[SessionStore].read_session(SessionId(session_id))

    assert response.status_code == status.HTTP_200_OK
    assert session.customer_id == customer_id
    assert [e.data for e in await signed_in_events(container, session_id)] == [
        {"type": "customer.signed_in", "customer_id": customer_id, "name": "Customer Alpha"}
    ]
    # Spent: nothing of the claim is left to use again.
    assert metadata_utils.internal_value(session.metadata, ("auth", "claim")) is None


async def test_that_a_claim_secret_cannot_claim_the_session_again(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
    other_customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        session_id = await create_platform_session(client, agent_id)
        await prepare_claim(container, session_id, "s3cret-one-time")
        await claim_with_secret(
            client, session_id, mint_customer_jwt(customer_id), "s3cret-one-time"
        )

        again = await claim_with_secret(
            client, session_id, mint_customer_jwt(other_customer_id), "s3cret-one-time"
        )

    session = await container[SessionStore].read_session(SessionId(session_id))

    assert again.status_code == status.HTTP_409_CONFLICT
    assert session.customer_id == customer_id


async def test_that_a_wrong_claim_secret_is_rejected(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        session_id = await create_platform_session(client, agent_id)
        await prepare_claim(container, session_id, "s3cret-one-time")

        response = await claim_with_secret(
            client, session_id, mint_customer_jwt(customer_id), "a-guessed-secret"
        )

    session = await container[SessionStore].read_session(SessionId(session_id))

    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert session.customer_id == CustomerStore.GUEST_ID


async def test_that_an_expired_claim_secret_is_rejected(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        session_id = await create_platform_session(client, agent_id)
        await prepare_claim(
            container, session_id, "s3cret-one-time", expires_in=timedelta(seconds=-1)
        )

        response = await claim_with_secret(
            client, session_id, mint_customer_jwt(customer_id), "s3cret-one-time"
        )

    assert response.status_code == status.HTTP_403_FORBIDDEN


async def test_that_preparing_a_new_claim_replaces_the_old_secret(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        session_id = await create_platform_session(client, agent_id)
        await prepare_claim(container, session_id, "first-secret")
        await prepare_claim(container, session_id, "second-secret")

        with_old = await claim_with_secret(
            client, session_id, mint_customer_jwt(customer_id), "first-secret"
        )
        with_new = await claim_with_secret(
            client, session_id, mint_customer_jwt(customer_id), "second-secret"
        )

    assert with_old.status_code == status.HTTP_403_FORBIDDEN
    assert with_new.status_code == status.HTTP_200_OK


async def test_that_a_session_without_a_prepared_claim_cannot_be_claimed_with_a_secret(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        platform_session_id = await create_platform_session(client, agent_id)
        guest_session_id, _ = await create_session(client, agent_id)

        responses = [
            await claim_with_secret(client, session_id, mint_customer_jwt(customer_id), "anything")
            for session_id in (platform_session_id, guest_session_id)
        ]

    assert [r.status_code for r in responses] == [status.HTTP_403_FORBIDDEN] * 2


async def test_that_a_claim_needs_exactly_one_of_a_guest_token_and_a_claim_secret(
    container: Container,
    client_with: ClientFactory,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    async with client_with() as client:
        session_id, guest_token = await create_session(client, agent_id)
        await prepare_claim(container, session_id, "s3cret-one-time")

        bodies: list[dict[str, Any]] = [
            {},
            {"guest_token": guest_token, "claim_secret": "s3cret-one-time"},
        ]
        responses = [
            await client.post(
                f"/sessions/{session_id}/claim",
                json=body,
                headers=bearer(mint_customer_jwt(customer_id)),
            )
            for body in bodies
        ]

    assert [r.status_code for r in responses] == [status.HTTP_422_UNPROCESSABLE_CONTENT] * 2
