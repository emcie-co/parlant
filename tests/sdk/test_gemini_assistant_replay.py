"""Real SDK conversations must preserve native Gemini history across turns.

These tests call the actual provider and use real HTTP, emitters and stores.
Protocol edge cases belong in test_gemini_interactions.py instead.
"""

import asyncio
import os
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx
import pytest

import parlant.sdk as p
from parlant.adapters.db.sqlite import SQLiteDocumentDatabase
from parlant.core import metadata
from parlant.core.application_context import ApplicationContext
from parlant.core.engines.compass.hooks import CompassHookResult, CompassHooks
from parlant.core.loggers import StdoutLogger
from parlant.core.nlp.model_routing import ModelRouter
from parlant.core.sessions import Event, EventKind, EventSource, SessionDocumentStore
from parlant.core.tracer import LocalTracer
from tests.sdk.utils import Client, Context, SDKTest
from tests.test_utilities import get_random_port

pytestmark = pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="requires live Gemini")


class _ReplaySDKTest(SDKTest):
    server: p.Server
    get_container: Callable[[], p.Container]
    MODELS = {"*": "gemini"}
    output_mode = p.OutputMode.STREAM
    store_override: str | p.SessionStore = "transient"

    async def _create_server_task(self, port: int) -> asyncio.Task[None]:
        self.active_server_task = await super()._create_server_task(port)
        return self.active_server_task

    async def _wait_for_startup(self, port: int) -> None:
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
            models={**dict(self.MODELS), "react": p.ModelConfig("gemini:gemini-4-argon")},
            configure_container=configure,
            hot_reload=False,
            log_level=p.LogLevel.WARNING,
            session_store=self.store_override,
            customer_store="transient",
        ), lambda: container_for_test

    async def configure_hooks(self, hooks: CompassHooks) -> CompassHooks:
        self.routes: list[str] = []

        async def record(
            context: p.EngineContext, step: int, exc: Exception | None
        ) -> CompassHookResult:
            generator = await self.get_container()[ModelRouter].get_react_generator(
                context=context, route="react", qualifiers=context.state.model_route_qualifiers
            )
            self.routes.append(generator.resolve_model({}))
            return CompassHookResult.CALL_NEXT

        hooks.on_step_completed.append(record)
        return hooks

    async def setup(self, server: p.Server) -> None:
        self.knowledge = "The Cedar project's colour is purple."
        self.agent = await server.create_agent(
            id="native-assistant-replay",
            name="Project assistant",
            instructions="Answer the customer's project questions using the supplied knowledge and the conversation history.",
            output_mode=self.output_mode,
        )

        async def project_knowledge(context: p.RetrieverContext) -> p.RetrieverResult:
            return p.RetrieverResult(data=self.knowledge)

        await self.agent.attach_retriever(project_knowledge)

    async def turn(self, ctx: Context, session: str, question: str) -> Event:
        async with httpx.AsyncClient(base_url=ctx.base_url, timeout=20) as http:
            sent = await http.post(
                f"/sessions/{session}/events",
                json={"kind": "message", "source": "customer", "message": question},
            )
            sent.raise_for_status()
            offset = int(sent.json()["offset"]) + 1
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                response = await http.get(
                    f"/sessions/{session}/events", params={"min_offset": offset, "wait_for_data": 5}
                )
                if response.status_code == 504:
                    continue
                response.raise_for_status()
                for event in response.json():
                    offset = max(offset, int(event["offset"]) + 1)
                    if event["kind"] == "status":
                        assert event["data"].get("status") != "error", (
                            "real Gemini turn failed; inspect the server log for the provider rejection"
                        )
                        if event["data"].get("status") == "ready":
                            events = await ctx.container[p.SessionStore].list_events(
                                p.SessionId(session),
                                source=EventSource.AI_AGENT,
                                kinds=[EventKind.MESSAGE],
                            )
                            assert events
                            return events[-1]
        raise TimeoutError("real SDK turn did not complete")

    def assert_saved_replay(self, event: Event) -> dict[str, Any]:
        replay = metadata.internal_value(event.metadata, ("assistant_replay",), None)
        assert isinstance(replay, dict), (
            "ordinary assistant reply lost native replay artifacts at persistence"
        )
        assert replay.get("provider") == "gemini"
        # Provider artifacts must be internal metadata, never user-visible text.
        assert "signature" not in str(event.data)
        return replay


class Test_that_argon_can_answer_followup_retrieval_after_a_text_only_reply(_ReplaySDKTest):
    @pytest.fixture(params=[p.OutputMode.STREAM, p.OutputMode.BLOCK], autouse=True)
    def output_mode_for_test(self, request: pytest.FixtureRequest) -> None:
        self.output_mode = request.param

    async def run(self, ctx: Context) -> None:
        session = (await ctx.client.sessions.create(agent_id=self.agent.id)).id
        first = await self.turn(ctx, session, "What colour is the Cedar project?")
        assert "purple" in str(first.data).lower()
        self.knowledge = "The Cedar project ships on Monday."
        second = await self.turn(
            ctx, session, "What colour did we establish, and when does it ship?"
        )
        assert "purple" in str(second.data).lower()
        assert "monday" in str(second.data).lower()
        self.assert_saved_replay(first)


class Test_that_flash_argon_flash_routing_preserves_native_assistant_history(_ReplaySDKTest):
    async def run(self, ctx: Context) -> None:
        session = (await ctx.client.sessions.create(agent_id=self.agent.id)).id
        for model, question, fact in [
            ("gemini-3.8-flash", "What colour is the Cedar project?", "purple"),
            ("gemini-4-argon", "What colour did we establish, and when does it ship?", "monday"),
            ("gemini-3.8-flash", "Remind me of the project colour and shipping day.", "purple"),
        ]:
            await self.agent.update(models={"react": p.ModelConfig(f"gemini:{model}")})
            if model == "gemini-4-argon":
                self.knowledge = "The Cedar project ships on Monday."
            event = await self.turn(ctx, session, question)
            assert fact in str(event.data).lower()
            assert self.routes[-1].endswith(model)
            self.assert_saved_replay(event)


class Test_that_native_assistant_history_survives_reopening_the_session_store(_ReplaySDKTest):
    async def test_run(self) -> None:
        logger = StdoutLogger(LocalTracer())
        session: str | None = None
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.db"
            for generation in range(2):
                async with SQLiteDocumentDatabase(logger, path) as database:
                    async with SessionDocumentStore(database) as store:
                        self.store_override = store
                        port = get_random_port()
                        task = await self._create_server_task(port)
                        try:
                            await self._wait_for_startup(port)
                            ctx = Context(
                                self.server,
                                Client(f"http://localhost:{port}"),
                                self.get_container(),
                                base_url=f"http://localhost:{port}",
                            )
                            if generation == 0:
                                session = (
                                    await ctx.client.sessions.create(agent_id=self.agent.id)
                                ).id
                                first = await self.turn(
                                    ctx, session, "What colour is the Cedar project?"
                                )
                                self.assert_saved_replay(first)
                            else:
                                assert session is not None
                                saved = await store.list_events(
                                    p.SessionId(session),
                                    source=EventSource.AI_AGENT,
                                    kinds=[EventKind.MESSAGE],
                                )
                                self.assert_saved_replay(saved[-1])
                                self.knowledge = "The Cedar project ships on Monday."
                                answer = await self.turn(
                                    ctx, session, "What was its colour, and when does it ship?"
                                )
                                assert "purple" in str(answer.data).lower()
                                assert "monday" in str(answer.data).lower()
                        finally:
                            task.cancel()
                            with suppress(asyncio.CancelledError):
                                await task
