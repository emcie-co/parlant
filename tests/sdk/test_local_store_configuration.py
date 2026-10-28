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

"""Live wiring tests for the 'local' (SQLite-backed) store configuration.

The 'local' spec persists to SQLite files under PARLANT_HOME_DIR, and is the
default for the session and customer stores. PARLANT_HOME_DIR is resolved at
import time, so these tests run against the real home directory and remove the
store files they create around each test.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import suppress

import httpx
import pytest

import parlant.sdk as p
from parlant.bin.server import PARLANT_HOME_DIR

from tests.test_utilities import get_random_port


_LOCAL_STORE_FILES = [
    PARLANT_HOME_DIR / "sessions.db",
    PARLANT_HOME_DIR / "customers.db",
    PARLANT_HOME_DIR / "variables.db",
    # Pre-SQLite servers wrote these; clear them so the no-JSON assertions
    # prove that THIS run didn't create them.
    PARLANT_HOME_DIR / "sessions.json",
    PARLANT_HOME_DIR / "customers.json",
    PARLANT_HOME_DIR / "variables.json",
]


@pytest.fixture(autouse=True)
async def clean_local_store_files() -> AsyncIterator[None]:
    for file in _LOCAL_STORE_FILES:
        file.unlink(missing_ok=True)
    yield
    for file in _LOCAL_STORE_FILES:
        file.unlink(missing_ok=True)


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


def _local_server(port: int, **store_kwargs: str) -> p.Server:
    return p.Server(
        port=port,
        tool_service_port=get_random_port(),
        log_level=p.LogLevel.WARNING,
        models={"*": "gemini"},
        hot_reload=False,
        **store_kwargs,  # type: ignore[arg-type]
    )


async def test_that_a_local_session_store_persists_sessions_in_a_sqlite_file_across_restarts() -> (
    None
):
    port = get_random_port()
    session_id: str

    server = _local_server(port, session_store="local", customer_store="transient")
    server_task = asyncio.create_task(_run_server_until_cancelled(server))
    try:
        await _wait_for_healthz(port)

        async with httpx.AsyncClient(base_url=f"http://localhost:{port}") as client:
            agent = (await client.post("/agents", json={"name": "Local Store Agent"})).json()
            response = await client.post("/sessions", json={"agent_id": agent["id"]})
            assert response.status_code == 201
            session_id = response.json()["id"]

        assert (PARLANT_HOME_DIR / "sessions.db").exists()
        assert not (PARLANT_HOME_DIR / "sessions.json").exists()
    finally:
        server_task.cancel()
        with suppress(asyncio.CancelledError):
            await server_task

    port = get_random_port()
    server = _local_server(port, session_store="local", customer_store="transient")
    server_task = asyncio.create_task(_run_server_until_cancelled(server))
    try:
        await _wait_for_healthz(port)

        async with httpx.AsyncClient(base_url=f"http://localhost:{port}") as client:
            response = await client.get(f"/sessions/{session_id}")
            assert response.status_code == 200
    finally:
        server_task.cancel()
        with suppress(asyncio.CancelledError):
            await server_task


async def test_that_a_local_customer_store_persists_customers_in_a_sqlite_file_across_restarts() -> (
    None
):
    port = get_random_port()
    customer_id: str

    server = _local_server(port, session_store="transient", customer_store="local")
    server_task = asyncio.create_task(_run_server_until_cancelled(server))
    try:
        await _wait_for_healthz(port)

        async with httpx.AsyncClient(base_url=f"http://localhost:{port}") as client:
            response = await client.post("/customers", json={"name": "Local Store Customer"})
            assert response.status_code == 201
            customer_id = response.json()["id"]

        assert (PARLANT_HOME_DIR / "customers.db").exists()
        assert not (PARLANT_HOME_DIR / "customers.json").exists()
    finally:
        server_task.cancel()
        with suppress(asyncio.CancelledError):
            await server_task

    port = get_random_port()
    server = _local_server(port, session_store="transient", customer_store="local")
    server_task = asyncio.create_task(_run_server_until_cancelled(server))
    try:
        await _wait_for_healthz(port)

        async with httpx.AsyncClient(base_url=f"http://localhost:{port}") as client:
            response = await client.get(f"/customers/{customer_id}")
            assert response.status_code == 200
            assert response.json()["name"] == "Local Store Customer"
    finally:
        server_task.cancel()
        with suppress(asyncio.CancelledError):
            await server_task


async def test_that_session_and_customer_stores_default_to_local() -> None:
    port = get_random_port()

    server = _local_server(port)
    server_task = asyncio.create_task(_run_server_until_cancelled(server))
    try:
        await _wait_for_healthz(port)

        async with httpx.AsyncClient(base_url=f"http://localhost:{port}") as client:
            agent = (await client.post("/agents", json={"name": "Default Store Agent"})).json()
            assert (
                await client.post("/sessions", json={"agent_id": agent["id"]})
            ).status_code == 201
            assert (
                await client.post("/customers", json={"name": "Default Store Customer"})
            ).status_code == 201

        assert (PARLANT_HOME_DIR / "sessions.db").exists()
        assert (PARLANT_HOME_DIR / "customers.db").exists()
    finally:
        server_task.cancel()
        with suppress(asyncio.CancelledError):
            await server_task
