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

"""Sessions the platform creates over the tunnel on behalf of a browser.

The platform (its playground) creates sessions through the authenticated
tunnel, so the runtime never sees the browser. The platform resolves the
browser's address itself and passes it as ``client``; the runtime validates it
and records it exactly as it would for a REST-created session.
"""

from pathlib import Path
from typing import Any

from pytest import fixture

from parlant.adapters.geolocation.mmdb import MmdbGeoLocator
from parlant.core import metadata as metadata_utils
from parlant.core.agents import AgentId
from parlant.core.application import Application
from parlant.core.common import JSONSerializable
from parlant.core.container import Container
from parlant.core.geolocation import NullGeoLocator
from parlant.core.session_clients import SessionClientRecorder
from parlant.core.sessions import SessionId, SessionStore
from parlant.core.tunnel_handlers.sessions import SessionTunnelHandlers

from tests.mmdb_utilities import city_record, write_city_database
from tests.test_utilities import create_agent


@fixture
async def agent_id(container: Container) -> AgentId:
    agent = await create_agent(container, name="tunnel-session-client-agent")
    return agent.id


def handlers(container: Container, recorder: SessionClientRecorder) -> SessionTunnelHandlers:
    return SessionTunnelHandlers(
        container[Application].sessions,
        session_client_recorder=recorder,
    )


async def create(
    container: Container,
    recorder: SessionClientRecorder,
    params: dict[str, Any],
) -> JSONSerializable:
    create_session = handlers(container, recorder).handler_set().unary["sessions.create"]
    result = await create_session(params)

    session = await container[SessionStore].read_session(SessionId(result["session_id"]))
    return metadata_utils.internal_value(session.metadata, ("client",))


async def test_that_tunnel_session_creation_records_the_client_supplied_by_the_platform(
    container: Container,
    agent_id: AgentId,
) -> None:
    recorded = await create(
        container,
        SessionClientRecorder(geo_locator=NullGeoLocator()),
        {
            "agent_id": agent_id,
            "client": {"ip": "203.0.113.7", "origin": "https://platform.example"},
        },
    )

    assert recorded == {"ip": "203.0.113.7", "origin": "https://platform.example"}


async def test_that_tunnel_session_creation_locates_the_supplied_client_ip(
    container: Container,
    agent_id: AgentId,
    tmp_path: Path,
) -> None:
    database = write_city_database(
        tmp_path / "city.mmdb",
        {"203.0.113.0/24": city_record("GB", "United Kingdom", "England", "London")},
    )

    recorded = await create(
        container,
        SessionClientRecorder(geo_locator=MmdbGeoLocator(database)),
        {"agent_id": agent_id, "client": {"ip": "203.0.113.7"}},
    )

    assert isinstance(recorded, dict)
    assert recorded["location"] == {
        "country_code": "GB",
        "country": "United Kingdom",
        "region": "England",
        "city": "London",
        "source": "DBIP-City-Lite",
    }


async def test_that_tunnel_session_creation_drops_an_invalid_client_address_and_origin(
    container: Container,
    agent_id: AgentId,
) -> None:
    recorded = await create(
        container,
        SessionClientRecorder(geo_locator=NullGeoLocator()),
        {
            "agent_id": agent_id,
            "client": {"ip": "not-an-address", "origin": "javascript:alert(1)"},
        },
    )

    assert recorded is None


async def test_that_tunnel_session_creation_without_a_client_records_nothing(
    container: Container,
    agent_id: AgentId,
) -> None:
    recorded = await create(
        container,
        SessionClientRecorder(geo_locator=NullGeoLocator()),
        {"agent_id": agent_id},
    )

    assert recorded is None
