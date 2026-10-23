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
from contextlib import asynccontextmanager
import time
from typing import AsyncIterator, Awaitable, Callable
from fastapi import FastAPI, Request, Response
import httpx
import pytest

import parlant.sdk as p

from parlant.core.health import ENGINE_TURN_KIND, HealthReporter
from tests.sdk.utils import Context, SDKTest


def test_that_server_defaults_to_openrouter() -> None:
    server = p.Server(hot_reload=False)

    assert server._models == {"*": "openrouter"}


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

    async def configure_api(self, app: FastAPI) -> None:
        self.configure_api_was_called = True
        self.received_app = app

    async def setup(self, server: p.Server) -> None:
        pass

    async def run(self, ctx: Context) -> None:
        assert self.configure_api_was_called
        assert isinstance(self.received_app, FastAPI)
        assert self.received_app is ctx.server.api


class Test_that_custom_routes_added_via_configure_api_are_accessible(SDKTest):
    async def configure_api(self, app: FastAPI) -> None:
        @app.get("/custom-endpoint")
        async def custom_endpoint() -> dict[str, str]:
            return {"message": "custom response"}

    async def setup(self, server: p.Server) -> None:
        pass

    async def run(self, ctx: Context) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://localhost:{ctx.server.port}/custom-endpoint")
            assert response.status_code == 200
            assert response.json() == {"message": "custom response"}


class Test_that_configure_api_can_add_middleware(SDKTest):
    middleware_was_called = False

    async def configure_api(self, app: FastAPI) -> None:
        @app.middleware("http")
        async def custom_middleware(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            self.middleware_was_called = True
            response = await call_next(request)
            response.headers["X-Custom-Header"] = "test-value"
            return response

    async def setup(self, server: p.Server) -> None:
        pass

    async def run(self, ctx: Context) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://localhost:{ctx.server.port}/healthz")
            assert response.status_code == 200
            assert "X-Custom-Header" in response.headers
            assert response.headers["X-Custom-Header"] == "test-value"
            assert self.middleware_was_called


class Test_that_get_group_returns_group_by_id(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.group = await server.create_group("test-group")

    async def run(self, ctx: Context) -> None:
        retrieved = await ctx.server.get_group(id=self.group.id)
        assert retrieved.id == self.group.id
        assert retrieved.name == self.group.name


class Test_that_get_group_returns_group_by_name(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.group = await server.create_group("test-group")

    async def run(self, ctx: Context) -> None:
        retrieved = await ctx.server.get_group(name="test-group")
        assert retrieved.id == self.group.id
        assert retrieved.name == self.group.name


class Test_that_get_group_raises_when_both_id_and_name_are_provided(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.group = await server.create_group("test-group")

    async def run(self, ctx: Context) -> None:
        with pytest.raises(p.SDKError):
            await ctx.server.get_group(id=self.group.id, name="test-group")


class Test_that_get_group_raises_when_name_does_not_exist(SDKTest):
    async def setup(self, server: p.Server) -> None:
        pass

    async def run(self, ctx: Context) -> None:
        with pytest.raises(p.SDKError, match="not found"):
            await ctx.server.get_group(name="nonexistent")


class Test_that_get_group_raises_when_neither_id_nor_name_is_provided(SDKTest):
    async def setup(self, server: p.Server) -> None:
        pass

    async def run(self, ctx: Context) -> None:
        with pytest.raises(p.SDKError):
            await ctx.server.get_group()


class Test_that_setup_blocking_does_not_poison_event_loop_health(SDKTest):
    async def setup(self, server: p.Server) -> None:
        # Establish a pending monitor tick before simulating synchronous setup work.
        await asyncio.sleep(0.15)
        time.sleep(0.7)
        await asyncio.sleep(0.15)

    async def run(self, ctx: Context) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ctx.base_url}/healthz")

        response.raise_for_status()
        event_loop = response.json()["checks"]["event_loop"]
        assert event_loop["status"] == "healthy", event_loop
        assert event_loop["latency_ms"] < 200, event_loop


class Test_that_lifespan_startup_blocking_does_not_poison_event_loop_health(SDKTest):
    async def configure_api(self, app: FastAPI) -> None:
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application: FastAPI) -> AsyncIterator[None]:
            async with original_lifespan(application):
                await asyncio.sleep(0.15)
                time.sleep(0.7)
                await asyncio.sleep(0.15)
                yield

        app.router.lifespan_context = lifespan

    async def setup(self, server: p.Server) -> None:
        pass

    async def run(self, ctx: Context) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ctx.base_url}/healthz")

        response.raise_for_status()
        event_loop = response.json()["checks"]["event_loop"]
        assert event_loop["status"] == "healthy", event_loop
        assert event_loop["latency_ms"] < 200, event_loop


class Test_that_runtime_blocking_is_detected_after_clean_startup(SDKTest):
    async def configure_container(self, container: p.Container) -> p.Container:
        # One 0.7s stall is far under 1% of five minutes; a zero share makes
        # it count, so this checks the monitor is measuring, not the share.
        container[p.HealthThresholds] = p.HealthThresholds(
            event_loop=p.EventLoopHealthThresholds(lag_share=0.0),
        )
        return container

    async def setup(self, server: p.Server) -> None:
        pass

    async def run(self, ctx: Context) -> None:
        async with httpx.AsyncClient() as client:
            before = await client.get(f"{ctx.base_url}/healthz")
            before.raise_for_status()
            assert before.json()["checks"]["event_loop"]["status"] == "healthy"

            await asyncio.sleep(0.15)
            time.sleep(0.7)
            await asyncio.sleep(0.15)

            after = await client.get(f"{ctx.base_url}/healthz")

        after.raise_for_status()
        event_loop = after.json()["checks"]["event_loop"]
        assert event_loop["status"] == "unhealthy", event_loop
        assert event_loop["latency_ms"] >= 500, event_loop


class Test_that_healthz_detects_event_loop_blocking_from_synchronous_tool(SDKTest):
    async def setup(self, server: p.Server) -> None:

        self.agent = await server.create_agent(
            id="server-agent-2",
            name="Blocking Tool Agent",
            instructions="Agent for testing event loop health detection",
        )

        @p.tool
        async def slow_lookup(context: p.ToolContext, account_id: str) -> p.ToolResult:
            # Simulate a badly-written tool that blocks the event loop: 3.5s
            # is over 1% of the five-minute window, with room to spare.
            time.sleep(3.5)
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
                f"Expected unhealthy after 3.5s blocking tool, got {data}"
            )
            assert data["checks"]["event_loop"]["latency_ms"] >= 500


class Test_that_healthz_reports_healthy_after_well_behaved_tool(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="server-agent-3",
            name="Good Tool Agent",
            instructions="Agent for testing event loop stays healthy with async tools",
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


class Test_that_healthz_nlp_section_starts_empty_before_any_message(SDKTest):
    async def setup(self, server: p.Server) -> None:
        pass

    async def run(self, ctx: Context) -> None:
        async with httpx.AsyncClient() as client:
            health_response = await client.get(f"http://localhost:{ctx.server.port}/healthz")
            assert health_response.status_code == 200

            data = health_response.json()
            assert "nlp" in data["checks"], f"Expected nlp section in /healthz, got {data}"
            assert data["checks"]["nlp"]["status"] == "healthy"
            assert data["checks"]["nlp"]["sample_count"] == 0


class Test_that_healthz_reports_nlp_section_after_message_exchange(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="server-agent-4",
            name="NLP Health Agent",
            instructions="Agent for testing NLP health reporting",
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Hello, how are you?",
            recipient=self.agent,
        )
        assert len(response) > 0

        async with httpx.AsyncClient() as client:
            health_response = await client.get(f"http://localhost:{ctx.server.port}/healthz")
            assert health_response.status_code == 200

            data = health_response.json()
            assert "nlp" in data["checks"], f"Expected nlp section in /healthz, got {data}"

            nlp_section = data["checks"]["nlp"]
            assert nlp_section["sample_count"] > 0, (
                f"Expected nlp.sample_count > 0 after a message exchange, got {nlp_section}"
            )
            assert nlp_section["status"] == "healthy", (
                f"Expected nlp.status healthy after a successful exchange, got {nlp_section}"
            )
            assert isinstance(nlp_section.get("routes"), dict)
            assert "react" in nlp_section["routes"], (
                f"Expected the react route in nlp.routes after a message exchange, got {nlp_section}"
            )
            react_route = nlp_section["routes"]["react"]
            assert self.agent.id in react_route["agents"], (
                f"Expected the responding agent in react.agents, got {react_route}"
            )


class Test_that_healthz_reports_token_and_request_rates_after_message_exchange(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="server-agent-5",
            name="Token Rate Agent",
            instructions="Agent for testing token/request rate reporting",
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Hello, how are you?",
            recipient=self.agent,
        )
        assert len(response) > 0

        async with httpx.AsyncClient() as client:
            health_response = await client.get(f"http://localhost:{ctx.server.port}/healthz")
            assert health_response.status_code == 200

            data = health_response.json()
            nlp_section = data["checks"]["nlp"]

            for block_name in ("tokens_per_minute", "requests_per_minute"):
                assert block_name in nlp_section, (
                    f"Expected '{block_name}' in nlp section, got {nlp_section}"
                )
                block = nlp_section[block_name]
                for window in ("1m", "5m", "1h", "1d"):
                    assert window in block, (
                        f"Expected window '{window}' in {block_name}, got {block}"
                    )
                    assert block[window] > 0, (
                        f"Expected positive {block_name}[{window}] after a message exchange, got {block}"
                    )


class Test_that_healthz_reports_engine_section_after_message_exchange(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="server-agent-6",
            name="Engine Health Agent",
            instructions="Agent for testing engine health reporting",
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Hello there",
            recipient=self.agent,
        )
        assert len(response) > 0

        async with httpx.AsyncClient() as client:
            health_response = await client.get(f"http://localhost:{ctx.server.port}/healthz")
            assert health_response.status_code == 200

            data = health_response.json()
            assert "engine" in data["checks"], f"Expected 'engine' section in /healthz, got {data}"

            engine = data["checks"]["engine"]
            assert engine["sample_count"] > 0, f"Expected positive sample_count, got {engine}"
            assert engine["success_rate"] == 1.0, (
                f"Expected success_rate 1.0 after a clean turn, got {engine}"
            )
            for field in ("p50_latency_ms", "p95_latency_ms", "p50_ttfm_ms", "p95_ttfm_ms"):
                assert field in engine, f"Expected '{field}' in engine section, got {engine}"
                assert engine[field] > 0, f"Expected positive {field} after a turn, got {engine}"

            assert "turns_per_minute" in engine, (
                f"Expected 'turns_per_minute' in engine section, got {engine}"
            )
            for window in ("1m", "5m", "1h", "1d"):
                assert window in engine["turns_per_minute"], (
                    f"Expected window '{window}' in turns_per_minute, got {engine['turns_per_minute']}"
                )
                assert engine["turns_per_minute"][window] > 0, (
                    f"Expected positive turns_per_minute[{window}], got {engine['turns_per_minute']}"
                )


class Test_that_healthz_lists_recent_diagnostic_report_links(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="server-agent-healthz-reports",
            name="Report Links Agent",
            instructions="Agent for testing diagnostic report links in healthz",
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Hello there",
            recipient=self.agent,
        )
        assert len(response) > 0

        async with httpx.AsyncClient() as client:
            health_response = await client.get(f"http://localhost:{ctx.server.port}/healthz")
            assert health_response.status_code == 200

            data = health_response.json()
            reports = data.get("diagnostic_reports")
            assert isinstance(reports, list), (
                f"Expected diagnostic_reports in /healthz after a turn, got {data.keys()}"
            )
            assert len(reports) >= 1
            entry = reports[0]
            assert entry["report_uri"].startswith("/debug/reports/")
            assert entry["trace_id"]
            assert entry["outcome"] in ("success", "error")


class Test_that_health_thresholds_bound_in_configure_container_change_the_healthz_status(SDKTest):
    """Binding ``HealthThresholds`` is the one way to tune ``/healthz``. A turn
    that is healthy under the defaults must read as degraded under a tighter
    engine threshold bound through ``configure_container``."""

    async def configure_container(self, container: p.Container) -> p.Container:
        container[p.HealthThresholds] = p.HealthThresholds(
            engine=p.EngineHealthThresholds(degraded_p95_ms=5.0, unhealthy_p95_ms=1_000.0),
        )
        return container

    async def run(self, ctx: Context) -> None:
        reporter = ctx.container[HealthReporter]
        reporter.report(
            ENGINE_TURN_KIND,
            {"success": True, "latency_ms": 10.0, "error_class": None},
        )

        async with httpx.AsyncClient() as client:
            health_response = await client.get(f"http://localhost:{ctx.server.port}/healthz")
            assert health_response.status_code == 200

            data = health_response.json()
            assert data["status"] == "degraded", data
            assert data["checks"]["engine"]["reasons"][0]["threshold"] == 5.0


class Test_that_event_loop_thresholds_bound_in_configure_container_change_the_healthz_status(
    SDKTest
):
    """The event loop monitor reads its lag thresholds from the same binding."""

    async def configure_container(self, container: p.Container) -> p.Container:
        container[p.HealthThresholds] = p.HealthThresholds(
            # The 0.15s stall below is 0.05% of the five-minute window.
            event_loop=p.EventLoopHealthThresholds(
                degraded_lag_ms=10.0,
                unhealthy_lag_ms=50.0,
                lag_share=0.0001,
            ),
        )
        return container

    async def run(self, ctx: Context) -> None:
        # The server shares this event loop, so stalling it here is what a
        # blocking tool would do inside the server.
        time.sleep(0.15)
        await asyncio.sleep(0.3)

        async with httpx.AsyncClient() as client:
            health_response = await client.get(f"http://localhost:{ctx.server.port}/healthz")
            assert health_response.status_code == 200

            data = health_response.json()
            assert data["checks"]["event_loop"]["status"] == "unhealthy", data
            assert data["status"] == "unhealthy"
