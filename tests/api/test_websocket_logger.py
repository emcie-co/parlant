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
from parlant.core.tracer import Tracer


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
