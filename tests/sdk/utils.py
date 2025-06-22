import asyncio
from dataclasses import dataclass
from typing import Callable, cast

from parlant.client import AsyncParlantClient as Client
from parlant.client.types.event import Event as ClientEvent

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
    client: Client
    container: p.Container


class SDKTest:
    async def test_run(self) -> None:
        port = get_random_port()

        server_task = await self._create_server_task(port)
        client = Client(base_url=f"http://localhost:{port}")

        try:
            await self._wait_for_startup(client)
            await self.run(Context(client, self.get_container()))
        finally:
            server_task.cancel()
            await server_task

    async def _create_server_task(self, port: int) -> asyncio.Task[None]:
        async def server_task() -> None:
            server, self.get_container = await self.create_server(port)

            async with server:
                await self.setup(server)

        task = asyncio.create_task(server_task())
        return task

    async def _wait_for_startup(self, client: Client) -> None:
        attempts = 0

        while True:
            try:
                await client.agents.list()
                return
            except Exception:
                attempts += 1

                if attempts > 10:
                    raise RuntimeError("Server did not start in time")

                await asyncio.sleep(0.333)

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
            log_level=p.LogLevel.DEBUG,
            configure_container=configure_container,
        ), lambda: test_container

    async def setup(self, server: p.Server) -> None: ...
    async def run(self, ctx: Context) -> None: ...
