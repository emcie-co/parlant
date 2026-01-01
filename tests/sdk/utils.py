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
import os
import time
from typing import Callable, cast

from parlant.client import AsyncParlantClient as Client
from parlant.client.types.event import Event as ClientEvent

from parlant.adapters.nlp.emcie_service import EmcieService
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
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

    async def send_and_receive_message_event(
        self,
        customer_message: str,
        recipient: p.Agent,
        reuse_session: bool = False,
    ) -> ClientEvent:
        if (not self._session_id) or (not reuse_session):
            self._session_id = (
                await self.client.sessions.create(
                    agent_id=recipient.id,
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

        return agent_messages[0]

    async def send_and_receive_message(
        self,
        customer_message: str,
        recipient: p.Agent,
        reuse_session: bool = False,
    ) -> str:
        agent_message = await self.send_and_receive_message_event(
            customer_message=customer_message,
            recipient=recipient,
            reuse_session=reuse_session,
        )

        return get_message(agent_message)


class SDKTest:
    STARTUP_TIMEOUT = 60

    async def test_run(self) -> None:
        port = get_random_port()

        server_task = await self._create_server_task(port)
        client = Client(base_url=f"http://localhost:{port}")

        try:
            await self._wait_for_startup(client)
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

    async def _wait_for_startup(self, client: Client) -> None:
        start_time = time.time()

        while True:
            try:
                await client.agents.list()
                return
            except Exception:
                if time.time() >= (start_time + self.STARTUP_TIMEOUT):
                    raise RuntimeError("Server did not start in time")

                await asyncio.sleep(0.25)

    async def configure_hooks(self, hooks: p.EngineHooks) -> p.EngineHooks:
        return hooks

    async def create_server(self, port: int) -> tuple[p.Server, Callable[[], p.Container]]:
        test_container: p.Container = p.Container()

        async def configure_container(container: p.Container) -> p.Container:
            nonlocal test_container
            test_container = container.clone()
            test_container[PerceivedPerformancePolicy] = NullPerceivedPerformancePolicy()
            return test_container

        return p.Server(
            port=port,
            tool_service_port=get_random_port(),
            log_level=p.LogLevel.TRACE,
            configure_container=configure_container,
            configure_hooks=self.configure_hooks,
            nlp_service=lambda c: EmcieService(
                c[Logger],
                c[Tracer],
                c[Meter],
                model_tier=os.environ.get("EMCIE_MODEL_TIER", "jackal"),  # type: ignore
                model_role=os.environ.get("EMCIE_MODEL_ROLE", "teacher"),  # type: ignore
            ),
        ), lambda: test_container

    async def setup(self, server: p.Server) -> None: ...
    async def run(self, ctx: Context) -> None: ...
