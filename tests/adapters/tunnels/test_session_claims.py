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

"""Claims the platform prepares over the tunnel.

A session the platform creates for an AI agent has no guest owner, so the
person behind the agent can't prove ownership with a guest token. When they
need to sign in, the platform prepares a claim: it sends only the hash of a
one-time secret and when it expires, and gives the secret itself to the person
through the store's sign-in page. Parlant keeps the hash under its own
``__parlant__`` namespace, where no public metadata write can reach it, and
never hands it back over the tunnel.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

from pytest import fixture

from parlant.core import metadata as metadata_utils
from parlant.core.application import Application
from parlant.core.container import Container
from parlant.core.customers import CustomerStore
from parlant.core.health import HealthReporter
from parlant.core.loggers import Logger
from parlant.core.sessions import SessionId, SessionStore
from parlant.core.tunnels import TunnelRequest, TunnelRequestDispatcher


@fixture
async def dispatcher(container: Container) -> TunnelRequestDispatcher:
    return TunnelRequestDispatcher(
        application=container[Application],
        health_reporter=container[HealthReporter],
        logger=container[Logger],
    )


async def call(dispatcher: TunnelRequestDispatcher, method: str, params: dict[str, Any]) -> Any:
    response = await dispatcher.dispatch(
        TunnelRequest(request_id="req-1", method=method, params=params)
    )
    assert response.error is None, response.error
    return response.result


async def call_expecting_error(
    dispatcher: TunnelRequestDispatcher, method: str, params: dict[str, Any]
) -> str:
    response = await dispatcher.dispatch(
        TunnelRequest(request_id="req-1", method=method, params=params)
    )
    assert response.error is not None
    assert isinstance(response.error, str)
    return response.error


async def platform_session(dispatcher: TunnelRequestDispatcher, **params: Any) -> str:
    agent = (await call(dispatcher, "agents.create", {"name": "Store Agent"}))["agent"]
    created = await call(
        dispatcher,
        "sessions.create",
        {"agent_id": agent["id"], "customer_represented_by_an_agent": True, **params},
    )
    return str(created["session_id"])


def claim_params(session_id: str, secret: str = "s3cret-one-time") -> dict[str, Any]:
    return {
        "session_id": session_id,
        "secret_sha256": hashlib.sha256(secret.encode()).hexdigest(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(),
    }


async def test_that_the_tunnel_prepares_a_claim_storing_only_the_secrets_hash_and_expiry(
    container: Container,
    dispatcher: TunnelRequestDispatcher,
) -> None:
    session_id = await platform_session(dispatcher)
    params = claim_params(session_id)

    await call(dispatcher, "sessions.prepare_claim", params)

    session = await container[SessionStore].read_session(SessionId(session_id))
    stored = metadata_utils.internal_value(session.metadata, ("auth", "claim"))

    assert stored == {"secret_sha256": params["secret_sha256"], "expires_at": params["expires_at"]}
    assert "s3cret-one-time" not in str(session.metadata)


async def test_that_preparing_a_claim_on_a_customers_session_is_refused(
    container: Container,
    dispatcher: TunnelRequestDispatcher,
) -> None:
    customer = await container[CustomerStore].create_customer(name="Signed In Already")
    session_id = await platform_session(dispatcher, customer_id=customer.id)

    error = await call_expecting_error(
        dispatcher, "sessions.prepare_claim", claim_params(session_id)
    )

    session = await container[SessionStore].read_session(SessionId(session_id))

    assert "already" in error.lower()
    assert metadata_utils.internal_value(session.metadata, ("auth", "claim")) is None


async def test_that_the_tunnel_never_returns_a_prepared_claim(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    session_id = await platform_session(dispatcher)
    params = claim_params(session_id)
    await call(dispatcher, "sessions.prepare_claim", params)

    read = await call(dispatcher, "sessions.read", {"session_id": session_id})
    listed = await call(dispatcher, "sessions.list", {})

    assert params["secret_sha256"] not in str(read)
    assert params["secret_sha256"] not in str(listed)
