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

from typing import Awaitable, Callable
from fastapi import FastAPI, Request, Response
import httpx

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
            assert response.json() == {"status": "ok"}
