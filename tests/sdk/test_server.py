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
from typing import Awaitable, Callable, Mapping
from fastapi import FastAPI, Request, Response
import httpx
import pytest
from typing_extensions import override

import parlant.sdk as p
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.services.indexing.indexer import IndexRequest, Indexer

from tests.sdk.utils import Context, SDKTest


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


class Test_that_indexer_runs_during_server_startup_with_full_payload(SDKTest):
    captured_payload: Mapping[str, Mapping[str, IndexRequest]] | None = None
    call_count: int = 0

    async def configure_container(self, container: p.Container) -> p.Container:
        outer = self

        class _RecordingIndexer(Indexer):
            @override
            async def index(
                self,
                payload: Mapping[str, Mapping[str, IndexRequest]],
                progress_report: ProgressReport,
            ) -> None:
                outer.captured_payload = payload
                outer.call_count += 1
                total = sum(len(bucket) for bucket in payload.values())
                if total > 0:
                    await progress_report.increment(total)

        container[Indexer] = _RecordingIndexer
        return container

    async def setup(self, server: p.Server) -> None:
        agent = await server.create_agent(
            name="Indexed Agent",
            description="An agent for the indexer test",
        )
        await agent.create_guideline(
            condition="customer says hi",
            action="reply with a greeting",
        )

    async def run(self, ctx: Context) -> None:
        assert self.call_count == 1
        assert self.captured_payload is not None

        payload = self.captured_payload
        assert {
            "agents",
            "guidelines",
            "journeys",
            "relationships",
            "glossary",
            "context_variables",
            "canned_responses",
            "tools",
        }.issubset(payload.keys())
        assert len(payload["agents"]) == 1
        assert len(payload["guidelines"]) >= 1


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
            name="NLP Health Agent",
            description="Agent for testing NLP health reporting",
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
            assert isinstance(nlp_section.get("schemas"), dict)
            assert len(nlp_section["schemas"]) > 0, (
                f"Expected at least one schema in nlp.schemas, got {nlp_section}"
            )


class Test_that_healthz_reports_token_and_request_rates_after_message_exchange(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Token Rate Agent",
            description="Agent for testing token/request rate reporting",
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
            name="Engine Health Agent",
            description="Agent for testing engine health reporting",
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
