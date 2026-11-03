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
from contextlib import suppress
from typing import Callable, Iterator
from urllib.parse import urlencode
from uuid import uuid4

import pytest
from moto.server import ThreadedMotoServer

import parlant.sdk as p
from parlant.core.application_context import ApplicationContext
from parlant.core.engines.compass.hooks import CompassHooks
from tests.sdk.utils import Context, SDKTest
from tests.test_utilities import get_random_port


@pytest.fixture
def s3_endpoint(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    try:
        host, port = server.get_host_and_port()
        yield f"http://{host}:{port}"
    finally:
        server.stop()


class _S3SDKRun(SDKTest):
    def __init__(self, store: str, session_id: str | None = None) -> None:
        self.store = store
        self.session_id = session_id

    async def _create_server_task(self, port: int) -> asyncio.Task[None]:
        async def serve() -> None:
            self.server, self.get_container = await self.create_server(port)
            try:
                async with self.server:
                    await self.setup(self.server)
            except SystemExit as error:
                # SDK startup reports configuration errors as a process exit.
                # Keep that real failure observable without aborting pytest's
                # event loop before it can inspect the diagnostic.
                raise RuntimeError("SDK server exited during startup") from error

        self.active_server_task = asyncio.create_task(serve(), name="S3 SDK Server Task")
        return self.active_server_task

    async def _wait_for_startup(self, port: int) -> None:
        # Follow the existing replay test helper: surface startup errors, rather
        # than waiting a minute for a server that has already failed.
        ready = asyncio.create_task(super()._wait_for_startup(port))
        try:
            await asyncio.wait(
                (ready, self.active_server_task), return_when=asyncio.FIRST_COMPLETED
            )
            if self.active_server_task.done():
                await self.active_server_task
                raise AssertionError("SDK server exited before becoming ready")
            await ready
        finally:
            ready.cancel()
            with suppress(asyncio.CancelledError):
                await ready

    async def create_server(self, port: int) -> tuple[p.Server, Callable[[], p.Container]]:
        container_for_test = p.Container()

        async def configure(container: p.Container) -> p.Container:
            nonlocal container_for_test
            container[ApplicationContext] = ApplicationContext(instance_id=self.INSTANCE_ID)
            container[CompassHooks] = await self.configure_hooks(container[CompassHooks])
            container_for_test = await self.configure_container(container)
            return container_for_test

        return p.Server(
            port=port,
            tool_service_port=get_random_port(),
            models=dict(self.MODELS),
            configure_container=configure,
            configure_api=self.configure_api,
            hot_reload=False,
            log_level=p.LogLevel.WARNING,
            session_store=self.store,
            customer_store="transient",
        ), lambda: container_for_test

    async def setup(self, server: p.Server) -> None:
        if self.session_id is None:
            self.agent = await server.create_agent(
                id="s3-storage-agent",
                name="Storage agent",
                instructions="Test session persistence.",
            )

    async def run(self, context: Context) -> None:
        if self.session_id is None:
            session = await context.client.sessions.create(
                agent_id=self.agent.id, allow_greeting=False
            )
            self.session_id = session.id
        else:
            session = await context.client.sessions.retrieve(session_id=self.session_id)
            assert session.id == self.session_id
            assert session.agent_id == "s3-storage-agent"


async def test_that_an_s3_session_store_persists_a_session_across_server_restarts(
    s3_endpoint: str,
) -> None:
    query = urlencode({"region": "us-east-1", "endpoint": s3_endpoint})
    url = f"s3://parlant-{uuid4().hex}?{query}"
    first = _S3SDKRun(url)
    await first.test_run()
    assert first.session_id is not None
    await _S3SDKRun(url, first.session_id).test_run()


async def test_that_invalid_s3_store_urls_do_not_expose_credentials(
    s3_endpoint: str, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "not-a-real-secret-sensitive"
    query = urlencode({"endpoint": s3_endpoint})
    with pytest.raises(RuntimeError) as failure:
        await _S3SDKRun(f"s3://testing:{secret}@/?{query}").test_run()
    diagnostic = capsys.readouterr()
    assert secret not in str(failure.value) + diagnostic.out + diagnostic.err
