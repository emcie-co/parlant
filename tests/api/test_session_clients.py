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

"""Recording where a browser-created session came from.

A session created over the REST API records, under the reserved ``__parlant__``
metadata namespace, the originating client's IP address (resolved through the
configured trusted proxies), the page origin it was opened from, and an
approximate location from a local GeoIP database when one is configured.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import status
from pytest import MonkeyPatch, fixture

from parlant.adapters.geolocation.mmdb import MmdbGeoLocator
from parlant.adapters.modules.parlant_dashboard.auth import (
    ParlantDashboardAuthorizationPolicy,
    build_default_inner_policy,
)
from parlant.api.app import create_api_app
from parlant.api.authentication import TokenAuthenticator
from parlant.api.authorization import AuthorizationPolicy, CompositeAuthorizationPolicy
from parlant.api.client_address import ClientAddressResolver
from parlant.core import metadata as metadata_utils
from parlant.core.agents import AgentId
from parlant.core.container import Container
from parlant.core.customers import CustomerId, CustomerStore
from parlant.core.geolocation import GeoLocator, NullGeoLocator
from parlant.core.session_clients import SessionClientRecorder
from parlant.core.sessions import SessionId, SessionStore
from parlant.core.common import JSONSerializable

from tests.api.test_auth_flows import (
    ADMIN_API_KEY,
    CUSTOMER_JWT_SECRET,
    bearer,
    mint_customer_jwt,
)
from tests.mmdb_utilities import city_record, write_city_database
from tests.test_utilities import create_agent

BROWSER = "203.0.113.7"
UNLOCATED_BROWSER = "198.51.100.20"
LOCAL_PROXY = "10.0.0.5"
CLOUDFLARE_EDGE_RANGE = "173.245.48.0/20"
CLOUDFLARE_EDGE = "173.245.48.10"

ClientFactory = Callable[..., Any]


@fixture
async def agent_id(container: Container) -> AgentId:
    agent = await create_agent(container, name="session-client-test-agent")
    return agent.id


@fixture
def geoip_database(tmp_path: Path) -> Path:
    return write_city_database(
        tmp_path / "city.mmdb",
        {
            "203.0.113.0/24": city_record("GB", "United Kingdom", "England", "London"),
            "2001:db8::/32": city_record("DE", "Germany", "Berlin", "Berlin"),
        },
    )


@fixture
def make_client(container: Container) -> ClientFactory:
    """Build an API client whose requests arrive from ``peer``."""

    @asynccontextmanager
    async def factory(
        *,
        peer: str = BROWSER,
        trusted_proxies: frozenset[str] = frozenset(),
        geo_locator: GeoLocator | None = None,
        enabled: bool = True,
    ) -> AsyncIterator[httpx.AsyncClient]:
        configured = container.clone()
        configured[AuthorizationPolicy] = CompositeAuthorizationPolicy(
            authenticator=TokenAuthenticator(
                guest_token_secret="test-guest-secret-0123456789abcdef",
                admin_api_key=ADMIN_API_KEY,
                customer_jwt_secret=CUSTOMER_JWT_SECRET,
            ),
            customer_store=container[CustomerStore],
            trusted_proxies=trusted_proxies,
        )
        configured[ClientAddressResolver] = ClientAddressResolver(trusted_proxies=trusted_proxies)
        configured[SessionClientRecorder] = SessionClientRecorder(
            geo_locator=geo_locator or NullGeoLocator(),
            enabled=enabled,
        )

        app = await create_api_app(configured)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=(peer, 50000)),
            base_url="http://testserver",
        ) as client:
            yield client

    return factory


async def create_session(
    client: httpx.AsyncClient,
    agent_id: AgentId,
    headers: dict[str, str] | None = None,
) -> str:
    response = await client.post("/sessions", json={"agent_id": agent_id}, headers=headers or {})
    assert response.status_code == status.HTTP_201_CREATED
    return str(response.json()["id"])


async def recorded_client(container: Container, session_id: str) -> JSONSerializable:
    session = await container[SessionStore].read_session(SessionId(session_id))
    return metadata_utils.internal_value(session.metadata, ("client",))


async def test_that_a_guest_session_records_the_originating_client_ip(
    container: Container,
    make_client: ClientFactory,
    agent_id: AgentId,
) -> None:
    async with make_client() as client:
        session_id = await create_session(client, agent_id)

    assert await recorded_client(container, session_id) == {"ip": BROWSER}


async def test_that_a_customer_session_records_the_originating_client_ip(
    container: Container,
    make_client: ClientFactory,
    agent_id: AgentId,
) -> None:
    customer = await container[CustomerStore].create_customer(
        name="Customer Alpha", id=CustomerId("cust_alpha")
    )

    async with make_client() as client:
        session_id = await create_session(
            client, agent_id, headers=bearer(mint_customer_jwt(customer.id))
        )

    assert await recorded_client(container, session_id) == {"ip": BROWSER}


async def test_that_an_admin_key_session_records_the_client_only_when_the_request_has_an_origin(
    container: Container,
    make_client: ClientFactory,
    agent_id: AgentId,
) -> None:
    async with make_client() as client:
        server_to_server = await create_session(client, agent_id, headers=bearer(ADMIN_API_KEY))
        from_a_browser = await create_session(
            client,
            agent_id,
            headers={**bearer(ADMIN_API_KEY), "Origin": "https://shop.example"},
        )

    assert await recorded_client(container, server_to_server) is None
    assert await recorded_client(container, from_a_browser) == {
        "ip": BROWSER,
        "origin": "https://shop.example",
    }


async def test_that_a_session_behind_trusted_proxies_records_the_client_and_not_a_proxy(
    container: Container,
    make_client: ClientFactory,
    agent_id: AgentId,
) -> None:
    async with make_client(
        peer=LOCAL_PROXY,
        trusted_proxies=frozenset({LOCAL_PROXY, CLOUDFLARE_EDGE_RANGE}),
    ) as client:
        session_id = await create_session(
            client,
            agent_id,
            headers={"X-Forwarded-For": f"198.51.100.66, {BROWSER}, {CLOUDFLARE_EDGE}"},
        )

    assert await recorded_client(container, session_id) == {"ip": BROWSER}


async def test_that_forwarded_for_from_an_untrusted_peer_is_not_recorded(
    container: Container,
    make_client: ClientFactory,
    agent_id: AgentId,
) -> None:
    async with make_client(peer=UNLOCATED_BROWSER) as client:
        session_id = await create_session(client, agent_id, headers={"X-Forwarded-For": BROWSER})

    assert await recorded_client(container, session_id) == {"ip": UNLOCATED_BROWSER}


async def test_that_a_session_records_the_page_origin_from_the_origin_header(
    container: Container,
    make_client: ClientFactory,
    agent_id: AgentId,
) -> None:
    async with make_client() as client:
        session_id = await create_session(
            client, agent_id, headers={"Origin": "https://shop.example"}
        )

    assert await recorded_client(container, session_id) == {
        "ip": BROWSER,
        "origin": "https://shop.example",
    }


async def test_that_the_page_origin_falls_back_to_the_referer_without_path_or_query(
    container: Container,
    make_client: ClientFactory,
    agent_id: AgentId,
) -> None:
    async with make_client() as client:
        session_id = await create_session(
            client,
            agent_id,
            headers={"Referer": "https://shop.example:8443/checkout?email=someone@example.com"},
        )

    assert await recorded_client(container, session_id) == {
        "ip": BROWSER,
        "origin": "https://shop.example:8443",
    }


async def test_that_an_opaque_or_non_web_origin_is_not_recorded(
    container: Container,
    make_client: ClientFactory,
    agent_id: AgentId,
) -> None:
    async with make_client() as client:
        opaque = await create_session(client, agent_id, headers={"Origin": "null"})
        non_web = await create_session(
            client, agent_id, headers={"Origin": "chrome-extension://abcdef"}
        )

    assert await recorded_client(container, opaque) == {"ip": BROWSER}
    assert await recorded_client(container, non_web) == {"ip": BROWSER}


async def test_that_a_session_records_approximate_location_from_the_local_geoip_database(
    container: Container,
    make_client: ClientFactory,
    agent_id: AgentId,
    geoip_database: Path,
) -> None:
    async with make_client(geo_locator=MmdbGeoLocator(geoip_database)) as client:
        session_id = await create_session(
            client, agent_id, headers={"Origin": "https://shop.example"}
        )

    assert await recorded_client(container, session_id) == {
        "ip": BROWSER,
        "origin": "https://shop.example",
        "location": {
            "country_code": "GB",
            "country": "United Kingdom",
            "region": "England",
            "city": "London",
            "source": "DBIP-City-Lite",
        },
    }


async def test_that_an_ipv6_client_is_recorded_and_located(
    container: Container,
    make_client: ClientFactory,
    agent_id: AgentId,
    geoip_database: Path,
) -> None:
    async with make_client(
        peer="2001:db8::7",
        geo_locator=MmdbGeoLocator(geoip_database),
    ) as client:
        session_id = await create_session(client, agent_id)

    recorded = await recorded_client(container, session_id)

    assert isinstance(recorded, dict)
    assert recorded["ip"] == "2001:db8::7"
    assert isinstance(recorded["location"], dict)
    assert recorded["location"]["country_code"] == "DE"


async def test_that_the_ip_is_recorded_without_location_when_the_database_has_no_entry(
    container: Container,
    make_client: ClientFactory,
    agent_id: AgentId,
    geoip_database: Path,
) -> None:
    async with make_client(
        peer=UNLOCATED_BROWSER,
        geo_locator=MmdbGeoLocator(geoip_database),
    ) as client:
        session_id = await create_session(client, agent_id)

    assert await recorded_client(container, session_id) == {"ip": UNLOCATED_BROWSER}


async def test_that_session_creation_succeeds_when_the_geoip_database_cannot_be_read(
    container: Container,
    make_client: ClientFactory,
    agent_id: AgentId,
    tmp_path: Path,
) -> None:
    async with make_client(geo_locator=MmdbGeoLocator(tmp_path / "missing.mmdb")) as client:
        session_id = await create_session(client, agent_id)

    assert await recorded_client(container, session_id) == {"ip": BROWSER}


async def test_that_client_recording_can_be_switched_off(
    container: Container,
    make_client: ClientFactory,
    agent_id: AgentId,
    geoip_database: Path,
) -> None:
    async with make_client(
        geo_locator=MmdbGeoLocator(geoip_database),
        enabled=False,
    ) as client:
        session_id = await create_session(
            client, agent_id, headers={"Origin": "https://shop.example"}
        )

    assert await recorded_client(container, session_id) is None


async def test_that_the_recorded_client_is_not_returned_by_the_sessions_api(
    container: Container,
    make_client: ClientFactory,
    agent_id: AgentId,
) -> None:
    async with make_client() as client:
        session_id = await create_session(
            client, agent_id, headers={"Origin": "https://shop.example"}
        )
        response = await client.get(f"/sessions/{session_id}", headers=bearer(ADMIN_API_KEY))

    assert await recorded_client(container, session_id) is not None
    assert response.status_code == status.HTTP_200_OK
    assert metadata_utils.PARLANT_METADATA_KEY not in response.json()["metadata"]


async def test_that_the_production_policy_wiring_resolves_the_client_behind_trusted_proxies(
    container: Container,
    agent_id: AgentId,
    monkeypatch: MonkeyPatch,
) -> None:
    # As a managed runtime is configured: PARLANT_TRUSTED_PROXIES reaches the
    # Dashboard policy's inner composite policy, and nothing binds a resolver.
    monkeypatch.setenv("PARLANT_TRUSTED_PROXIES", f"10.30.0.0/24,{CLOUDFLARE_EDGE_RANGE}")

    configured = container.clone()
    configured[AuthorizationPolicy] = ParlantDashboardAuthorizationPolicy(
        project_token="project-token-for-tests",
        inner=build_default_inner_policy(customer_store=container[CustomerStore]),
    )
    app = await create_api_app(configured)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("10.30.0.89", 50000)),
        base_url="http://testserver",
    ) as client:
        session_id = await create_session(
            client,
            agent_id,
            headers={"X-Forwarded-For": f"198.51.100.66, {BROWSER}, {CLOUDFLARE_EDGE}"},
        )

    recorded = await recorded_client(container, session_id)

    assert isinstance(recorded, dict)
    assert recorded["ip"] == BROWSER
