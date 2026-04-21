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
from fastapi.testclient import TestClient
from parlant.api.app import ASGIApplication
from lagom import Container
import pytest

from parlant.adapters.loggers.websocket import WebSocketLogger
from parlant.core.loggers import LogLevel
from parlant.core.tracer import LocalTracer, Tracer


@pytest.fixture
def test_client(api_app: ASGIApplication) -> TestClient:
    return TestClient(api_app)


async def test_that_websocket_logger_sends_messages(
    container: Container,
    test_client: TestClient,
) -> None:
    ws_logger = container[WebSocketLogger]
    tracer = container[Tracer]

    with test_client.websocket_connect("/logs") as ws:
        ws_logger.info("Hello from test!")
        await asyncio.sleep(1)

        data = ws.receive_json()

        assert "Hello from test!" in data["message"]
        assert data["level"] == "INFO"
        assert data["trace_id"] == tracer.trace_id


async def test_that_websocket_reconnects_and_receives_messages(
    container: Container,
    test_client: TestClient,
) -> None:
    ws_logger = container[WebSocketLogger]
    tracer = container[Tracer]

    with test_client.websocket_connect("/logs") as ws1:
        ws_logger.info("First connection test")
        await asyncio.sleep(1)

        data1 = ws1.receive_json()
        assert "First connection test" in data1["message"]
        assert data1["level"] == "INFO"
        assert data1["trace_id"] == tracer.trace_id

    with test_client.websocket_connect("/logs") as ws2:
        ws_logger.info("Second connection test")
        await asyncio.sleep(1)

        data2 = ws2.receive_json()
        assert "Second connection test" in data2["message"]
        assert data2["level"] == "INFO"
        assert data2["trace_id"] == tracer.trace_id


async def test_that_draining_queued_messages_without_subscribers_does_not_starve_event_loop() -> (
    None
):
    NUM_MESSAGES = 200_000
    MAX_STALL_SECONDS = 0.05

    tracer = LocalTracer()
    logger = WebSocketLogger(tracer, LogLevel.INFO)

    for i in range(NUM_MESSAGES):
        logger.info(f"Message {i}")

    drain_task = asyncio.create_task(logger.start())

    t0 = asyncio.get_event_loop().time()
    await asyncio.sleep(0)
    stall = asyncio.get_event_loop().time() - t0

    drain_task.cancel()
    try:
        await drain_task
    except asyncio.CancelledError:
        pass

    assert stall < MAX_STALL_SECONDS, (
        f"Event loop was starved for {stall:.3f}s while draining {NUM_MESSAGES} messages "
        f"(threshold: {MAX_STALL_SECONDS}s)"
    )
