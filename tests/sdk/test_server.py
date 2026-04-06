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

import time
from typing import Awaitable, Callable
from fastapi import FastAPI, Request, Response
import httpx
import pytest

import parlant.sdk as p

from tests.sdk.utils import Context, SDKTest
from tests.test_utilities import get_random_port


class Test_that_server_exposes_api_property_with_fastapi_app(SDKTest):
    async def setup(self, server: p.Server) -> None:
        pass

    async def run(self, ctx: Context) -> None:
        # Verify that server.api returns a FastAPI instance
        assert isinstance(ctx.server.api, FastAPI)
        assert ctx.server.api.title == "Parlant API"


class Test_that_configure_api_hook_is_called_with_fastapi_app(SDKTest):
    configure_api_was_called = False
    received_app: FastAPI | None = None

    async def create_server(self, port: int) -> tuple[p.Server, Callable[[], p.Container]]:
        test_container: p.Container = p.Container()

        async def configure_container(container: p.Container) -> p.Container:
            nonlocal test_container
            test_container = container.clone()
            return test_container

        async def configure_api(app: FastAPI) -> None:
            self.configure_api_was_called = True
            self.received_app = app

        return p.Server(
            port=port,
            tool_service_port=get_random_port(),
            log_level=p.LogLevel.TRACE,
            configure_container=configure_container,
            configure_api=configure_api,
        ), lambda: test_container

    async def setup(self, server: p.Server) -> None:
        pass

    async def run(self, ctx: Context) -> None:
        # Verify that configure_api was called with FastAPI app
        assert self.configure_api_was_called
        assert isinstance(self.received_app, FastAPI)
        assert self.received_app is ctx.server.api


class Test_that_custom_routes_added_via_configure_api_are_accessible(SDKTest):
    async def create_server(self, port: int) -> tuple[p.Server, Callable[[], p.Container]]:
        test_container: p.Container = p.Container()

        async def configure_container(container: p.Container) -> p.Container:
            nonlocal test_container
            test_container = container.clone()
            return test_container

        async def configure_api(app: FastAPI) -> None:
            @app.get("/custom-endpoint")
            async def custom_endpoint() -> dict[str, str]:
                return {"message": "custom response"}

        return p.Server(
            port=port,
            tool_service_port=get_random_port(),
            log_level=p.LogLevel.TRACE,
            configure_api=configure_api,
        ), lambda: test_container

    async def setup(self, server: p.Server) -> None:
        pass

    async def run(self, ctx: Context) -> None:
        # Make HTTP request to custom endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://localhost:{ctx.server.port}/custom-endpoint")
            assert response.status_code == 200
            assert response.json() == {"message": "custom response"}


class Test_that_configure_api_can_add_middleware(SDKTest):
    middleware_was_called = False

    async def create_server(self, port: int) -> tuple[p.Server, Callable[[], p.Container]]:
        test_container: p.Container = p.Container()

        async def configure_container(container: p.Container) -> p.Container:
            nonlocal test_container
            test_container = container.clone()
            return test_container

        async def configure_api(app: FastAPI) -> None:
            @app.middleware("http")
            async def custom_middleware(
                request: Request, call_next: Callable[[Request], Awaitable[Response]]
            ) -> Response:
                self.middleware_was_called = True
                response = await call_next(request)
                response.headers["X-Custom-Header"] = "test-value"
                return response

        return p.Server(
            port=port,
            tool_service_port=get_random_port(),
            log_level=p.LogLevel.TRACE,
            configure_container=configure_container,
            configure_api=configure_api,
        ), lambda: test_container

    async def setup(self, server: p.Server) -> None:
        pass

    async def run(self, ctx: Context) -> None:
        # Make HTTP request to verify middleware was applied
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://localhost:{ctx.server.port}/healthz")
            assert response.status_code == 200
            assert "X-Custom-Header" in response.headers
            assert response.headers["X-Custom-Header"] == "test-value"
            assert self.middleware_was_called


class Test_that_get_tag_returns_tag_by_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tag = await server.create_tag("test-tag")

    async def run(self, ctx: Context) -> None:
        retrieved = await ctx.server.get_tag(id=self.tag.id)
        assert retrieved.id == self.tag.id
        assert retrieved.name == self.tag.name


class Test_that_get_tag_returns_tag_by_name(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tag = await server.create_tag("test-tag")

    async def run(self, ctx: Context) -> None:
        retrieved = await ctx.server.get_tag(name="test-tag")
        assert retrieved.id == self.tag.id
        assert retrieved.name == self.tag.name


class Test_that_get_tag_raises_when_both_id_and_name_are_provided(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.tag = await server.create_tag("test-tag")

    async def run(self, ctx: Context) -> None:
        with pytest.raises(p.SDKError):
            await ctx.server.get_tag(id=self.tag.id, name="test-tag")


class Test_that_get_tag_raises_when_name_does_not_exist(SDKTest):
    async def setup(self, server: p.Server) -> None:
        pass

    async def run(self, ctx: Context) -> None:
        with pytest.raises(p.SDKError, match="not found"):
            await ctx.server.get_tag(name="nonexistent")


class Test_that_get_tag_raises_when_neither_id_nor_name_is_provided(SDKTest):
    async def setup(self, server: p.Server) -> None:
        pass

    async def run(self, ctx: Context) -> None:
        with pytest.raises(p.SDKError):
            await ctx.server.get_tag()


class Test_that_server_works_without_configure_api(SDKTest):
    async def create_server(self, port: int) -> tuple[p.Server, Callable[[], p.Container]]:
        test_container: p.Container = p.Container()

        async def configure_container(container: p.Container) -> p.Container:
            nonlocal test_container
            test_container = container.clone()
            return test_container

        # Create server without configure_api parameter
        return p.Server(
            port=port,
            tool_service_port=get_random_port(),
            log_level=p.LogLevel.TRACE,
            configure_container=configure_container,
        ), lambda: test_container

    async def setup(self, server: p.Server) -> None:
        pass

    async def run(self, ctx: Context) -> None:
        # Verify server works normally without configure_api
        assert isinstance(ctx.server.api, FastAPI)

        # Verify health endpoint still works
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://localhost:{ctx.server.port}/healthz")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ("healthy", "degraded", "unhealthy")


class Test_that_healthz_detects_event_loop_blocking_from_synchronous_tool(SDKTest):
    async def setup(self, server: p.Server) -> None:

        self.agent = await server.create_agent(
            name="Blocking Tool Agent",
            description="Agent for testing event loop health detection",
        )

        @p.tool
        async def slow_lookup(context: p.ToolContext, account_id: str) -> p.ToolResult:
            # Simulate a badly-written tool that blocks the event loop
            time.sleep(3)
            return p.ToolResult(data={"account_id": account_id, "status": "found"})

        await self.agent.create_observation(
            condition="the customer asks to look up their account",
            tools=[slow_lookup],
        )

    async def run(self, ctx: Context) -> None:
        # Send a message that triggers the blocking tool
        response = await ctx.send_and_receive_message(
            customer_message="Please look up my account, ID is 12345",
            recipient=self.agent,
        )

        # The agent should still respond despite the blocking tool
        assert len(response) > 0

        # The event loop monitor should have detected the blocking
        async with httpx.AsyncClient() as client:
            health_response = await client.get(f"http://localhost:{ctx.server.port}/healthz")
            assert health_response.status_code == 200

            data = health_response.json()
            assert data["status"] == "unhealthy", (
                f"Expected unhealthy after 3s blocking tool, got {data}"
            )
            assert data["checks"]["event_loop"]["latency_ms"] >= 500


class Test_that_healthz_reports_healthy_after_well_behaved_tool(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Good Tool Agent",
            description="Agent for testing event loop stays healthy with async tools",
        )

        @p.tool
        async def fast_lookup(context: p.ToolContext, account_id: str) -> p.ToolResult:
            return p.ToolResult(data={"account_id": account_id, "status": "found"})

        await self.agent.create_observation(
            condition="the customer asks to look up their account",
            tools=[fast_lookup],
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Please look up my account, ID is 12345",
            recipient=self.agent,
        )

        assert len(response) > 0

        async with httpx.AsyncClient() as client:
            health_response = await client.get(f"http://localhost:{ctx.server.port}/healthz")
            assert health_response.status_code == 200

            data = health_response.json()
            assert data["status"] == "healthy", (
                f"Expected healthy after well-behaved tool, got {data}"
            )
            assert data["checks"]["event_loop"]["latency_ms"] < 200
