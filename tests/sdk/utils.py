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
from dataclasses import dataclass
from datetime import timedelta
import os
import time
from typing import Callable, cast

from parlant.client import AsyncParlantClient as Client
from parlant.client.types.event import Event as ClientEvent

from fastapi import FastAPI
import httpx

from parlant.adapters.nlp.emcie_service import EmcieService
from parlant.core.application_context import ApplicationContext
from parlant.core.health import HealthReporter
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.sessions import Session
from parlant.core.tracer import Tracer
import parlant.sdk as p

from parlant.core.engines.alpha.perceived_performance_policy import (
    NullPerceivedPerformancePolicy,
    PerceivedPerformancePolicy,
)

from tests.test_utilities import get_random_port


def get_message(event: ClientEvent) -> str:
    if message := event.model_dump().get("data", {}).get("message", ""):
        return cast(str, message)

    raise ValueError("Event does not contain a message in its data.")


@dataclass
class Context:
    server: p.Server
    client: Client
    container: p.Container
    _session_id: str | None = None

    async def get_session(self) -> Session:
        if self._session_id is None:
            raise ValueError("No session has been created yet")

        session_store = self.container[p.SessionStore]
        return await session_store.read_session(p.SessionId(self._session_id))

    async def send_and_receive_message_event(
        self,
        customer_message: str,
        recipient: p.Agent,
        sender: p.Customer | None = None,
        reuse_session: bool = False,
    ) -> ClientEvent:
        if (not self._session_id) or (not reuse_session):
            self._session_id = (
                await self.client.sessions.create(
                    agent_id=recipient.id,
                    customer_id=sender.id if sender else None,
                    allow_greeting=False,
                )
            ).id

        event = await self.client.sessions.create_event(
            session_id=self._session_id,
            kind="message",
            source="customer",
            message=customer_message,
        )

        agent_messages = await self.client.sessions.list_events(
            session_id=self._session_id,
            min_offset=event.offset,
            source="ai_agent",
            kinds="message",
            wait_for_data=30,
        )

        assert len(agent_messages) >= 1

        agent_message = agent_messages[0]

        # For streaming mode, wait for the message to be complete
        # (chunks array ends with null terminator)
        if self._is_streaming_in_progress(agent_message):
            agent_message = await self._wait_for_streaming_completion(
                session_id=self._session_id,
                event_id=agent_message.id,
                min_offset=event.offset,
            )

        return agent_message

    def _is_streaming_in_progress(self, event: ClientEvent) -> bool:
        """Check if the event is still streaming (chunks property exists and not yet terminated with null)."""
        event_data = event.model_dump().get("data", {})
        chunks = event_data.get("chunks")
        # If chunks property doesn't exist, this is block mode - not streaming
        if chunks is None:
            return False
        # If chunks exists but is empty, streaming has started but no chunks yet - still in progress
        if len(chunks) == 0:
            return True
        # If chunks has content, check if the last element is None (completion marker)
        return chunks[-1] is not None

    async def _wait_for_streaming_completion(
        self,
        session_id: str,
        event_id: str,
        min_offset: int,
        timeout: float = 60.0,
    ) -> ClientEvent:
        """Wait for a streaming message to complete."""
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Streaming message did not complete within {timeout} seconds")

            events = await self.client.sessions.list_events(
                session_id=session_id,
                source="ai_agent",
                kinds="message",
                min_offset=min_offset,
                wait_for_data=10,
            )

            for event in events:
                if event.id == event_id:
                    if not self._is_streaming_in_progress(event):
                        return event
                    break

            await asyncio.sleep(0.1)

    async def receive_message_events(
        self,
        min_offset: int,
        wait_for_data: int = 30,
    ) -> list[ClientEvent]:
        """Receive agent message events from the current session starting at the given offset."""
        if self._session_id is None:
            raise ValueError("No session has been created yet")

        events = await self.client.sessions.list_events(
            session_id=self._session_id,
            min_offset=min_offset,
            source="ai_agent",
            kinds="message",
            wait_for_data=wait_for_data,
        )

        result: list[ClientEvent] = []
        for event in events:
            if self._is_streaming_in_progress(event):
                completed = await self._wait_for_streaming_completion(
                    session_id=self._session_id,
                    event_id=event.id,
                    min_offset=min_offset,
                )
                result.append(completed)
            else:
                result.append(event)

        return result

    async def send_and_receive_message(
        self,
        customer_message: str,
        recipient: p.Agent,
        sender: p.Customer | None = None,
        reuse_session: bool = False,
    ) -> str:
        agent_message = await self.send_and_receive_message_event(
            customer_message=customer_message,
            recipient=recipient,
            sender=sender,
            reuse_session=reuse_session,
        )

        return get_message(agent_message)


class SDKTest:
    STARTUP_TIMEOUT = 60
    INSTANCE_ID = "sdk-test-instance"

    async def test_run(self) -> None:
        port = get_random_port()

        server_task = await self._create_server_task(port)
        client = Client(base_url=f"http://localhost:{port}")

        try:
            await self._wait_for_startup(port)
            await self.run(Context(self.server, client, self.get_container()))
        finally:
            server_task.cancel()

            try:
                await server_task
            except asyncio.CancelledError:
                pass

    async def _create_server_task(self, port: int) -> asyncio.Task[None]:
        async def server_task() -> None:
            self.server, self.get_container = await self.create_server(port)

            async with self.server:
                try:
                    await self.setup(self.server)
                except BaseException:
                    raise

        task = asyncio.create_task(server_task(), name="SDK Server Task")
        return task

    async def _wait_for_startup(self, port: int) -> None:
        start_time = time.time()
        url = f"http://localhost:{port}/healthz"

        async with httpx.AsyncClient() as http:
            while True:
                try:
                    response = await http.get(url, timeout=5.0)
                    if response.status_code == 200:
                        data = response.json()
                        assert data["instance_id"] == self.INSTANCE_ID
                        return
                except (httpx.RequestError, httpx.TimeoutException):
                    pass

                if time.time() >= (start_time + self.STARTUP_TIMEOUT):
                    raise RuntimeError("Server did not start in time")

                await asyncio.sleep(0.25)

    async def configure_hooks(self, hooks: p.EngineHooks) -> p.EngineHooks:
        return hooks

    async def configure_api(self, app: FastAPI) -> None:
        """Override to customize the FastAPI app (add routes/middleware)."""

    async def configure_container(self, container: p.Container) -> p.Container:
        """Override to bind/replace dependencies in the test container.

        May return the same container (most cases) or a different one.
        """
        return container

    async def create_server(self, port: int) -> tuple[p.Server, Callable[[], p.Container]]:
        test_container: p.Container = p.Container()

        async def _configure(container: p.Container) -> p.Container:
            nonlocal test_container
            test_container = container.clone()

            # Apply test defaults
            container[ApplicationContext] = ApplicationContext(instance_id=self.INSTANCE_ID)
            container[PerceivedPerformancePolicy] = NullPerceivedPerformancePolicy()
            # Tests need to observe newly reported health data immediately rather
            # than waiting for the production cache TTL to elapse.
            container[HealthReporter]._snapshot_cache_ttl = timedelta(0)

            # Apply test-specific overrides
            test_container = await self.configure_container(test_container)

            return test_container

        return p.Server(
            port=port,
            tool_service_port=get_random_port(),
            log_level=p.LogLevel.TRACE,
            configure_container=_configure,
            configure_hooks=self.configure_hooks,
            configure_api=self.configure_api,
            nlp_service=lambda c: EmcieService(
                c[Logger],
                c[Tracer],
                c[Meter],
                c[HealthReporter],
                model_tier=os.environ.get("EMCIE_MODEL_TIER", "jackal"),  # type: ignore
                model_role=os.environ.get("EMCIE_MODEL_ROLE", "teacher"),  # type: ignore
            ),
        ), lambda: test_container

    async def setup(self, server: p.Server) -> None: ...
    async def run(self, ctx: Context) -> None: ...
