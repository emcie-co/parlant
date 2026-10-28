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

import asyncio
import time
from contextlib import suppress

import httpx
import pytest
from pymongo.uri_parser import parse_uri

import parlant.sdk as p
from parlant.sdk import (
    SDKError,
    _ensure_same_mongo_connection,
    _parse_mongo_connection_settings,
)

from tests.test_utilities import get_random_port


def test_that_mongo_connection_settings_take_the_database_name_from_the_uri() -> None:
    settings = _parse_mongo_connection_settings(
        "mongodb://localhost:27017/project_6a69b209e02b15f68a23dd36",
        parse_uri,
    )

    assert settings.database_name == "project_6a69b209e02b15f68a23dd36"


def test_that_mongo_connection_settings_fall_back_to_the_default_database_when_the_uri_names_none() -> (
    None
):
    settings = _parse_mongo_connection_settings("mongodb://localhost:27017", parse_uri)

    assert settings.database_name == "parlant"


def test_that_matching_mongo_connections_are_accepted() -> None:
    uri = "mongodb://localhost:27017/project_a"

    _ensure_same_mongo_connection(
        _parse_mongo_connection_settings(uri, parse_uri),
        _parse_mongo_connection_settings(uri, parse_uri),
    )


@pytest.mark.parametrize(
    "second_uri",
    [
        "mongodb://localhost:27017/project_b",
        "mongodb://otherhost:27017/project_a",
    ],
)
def test_that_mismatched_mongo_connections_are_rejected(second_uri: str) -> None:
    with pytest.raises(SDKError):
        _ensure_same_mongo_connection(
            _parse_mongo_connection_settings("mongodb://localhost:27017/project_a", parse_uri),
            _parse_mongo_connection_settings(second_uri, parse_uri),
        )


def test_that_mongo_connection_settings_repr_hides_the_uri() -> None:
    settings = _parse_mongo_connection_settings(
        "mongodb://user:hunter2-password@localhost:27017/project_a",
        parse_uri,
    )

    assert "hunter2-password" not in repr(settings)


###############################################################################
# Live wiring tests (need a running MongoDB; skipped otherwise)
###############################################################################


def mongo_uri_with_database(server_uri: str, database_name: str) -> str:
    base, question_mark, query = server_uri.partition("?")
    return f"{base.rstrip('/')}/{database_name}{question_mark}{query}"


@pytest.fixture
def test_mongo_server(mongo_server_uri: str) -> str:
    return mongo_server_uri


async def _run_server_until_cancelled(server: "p.Server") -> None:
    # Entering the server starts it; exiting RUNS it until externally
    # stopped, so live tests hold it in a task and cancel when done
    # (mirroring the SDKTest harness).
    async with server:
        pass


async def _wait_for_healthz(port: int, timeout_seconds: float = 60.0) -> None:
    start = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start < timeout_seconds:
            try:
                response = await client.get(f"http://localhost:{port}/healthz")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError("Server did not become healthy in time")


async def test_that_a_mongo_backed_server_creates_its_collections_in_the_uri_database(
    test_mongo_server: str,
) -> None:
    from pymongo import AsyncMongoClient

    database_name = "parlant_test_uri_database"
    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(test_mongo_server)
    port = get_random_port()

    server = p.Server(
        port=port,
        tool_service_port=get_random_port(),
        log_level=p.LogLevel.WARNING,
        session_store=mongo_uri_with_database(test_mongo_server, database_name),
        customer_store=mongo_uri_with_database(test_mongo_server, database_name),
        models={"*": "gemini"},
        hot_reload=False,
    )
    server_task = asyncio.create_task(_run_server_until_cancelled(server))

    try:
        await client.drop_database(database_name)
        await _wait_for_healthz(port)

        assert await client[database_name].list_collection_names()
    finally:
        server_task.cancel()
        with suppress(asyncio.CancelledError):
            await server_task
        await client.drop_database(database_name)
        await client.close()


async def test_that_stores_on_different_mongo_databases_are_rejected_at_startup(
    test_mongo_server: str,
) -> None:
    server = p.Server(
        port=get_random_port(),
        tool_service_port=get_random_port(),
        log_level=p.LogLevel.WARNING,
        session_store=mongo_uri_with_database(test_mongo_server, "parlant_test_db_a"),
        customer_store=mongo_uri_with_database(test_mongo_server, "parlant_test_db_b"),
        models={"*": "gemini"},
        hot_reload=False,
    )

    # Startup errors are fatal by design: the server prints the SDKError and
    # exits rather than propagating it.
    with pytest.raises(SystemExit):
        await server.__aenter__()
