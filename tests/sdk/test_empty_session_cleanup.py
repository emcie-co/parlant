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

"""Empty-session lifecycle through real SDK servers, stores and greeting hooks.

The cleanup service is resolved inside a test, not during collection, so its
absence produces a useful RED result after the real SDK fixture has started.
Explicit sweep times test the boundary without changing the system clock.
"""

import asyncio
from contextlib import AsyncExitStack
from datetime import datetime, timedelta, timezone
from importlib import import_module
from typing import Any, Awaitable, Callable, Coroutine, Protocol, cast
from typing_extensions import override

import httpx
import pytest

import parlant.sdk as p
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.agents import AgentId
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.common import ItemNotFoundError
from parlant.core.cost_control import CostContext, CostControlPolicy, CostVerdict, WorkKind
from parlant.core.customers import CustomerStore
from parlant.core.application import Application
from parlant.core.engines.compass.hooks import CompassHooks, CompassHookResult
from parlant.core.engines.types import UtteranceRationale, UtteranceRequest
from parlant.core.loggers import Logger
from parlant.core.sessions import (
    CompositeSessionStore,
    Event,
    EventKind,
    EventSource,
    Session,
    SessionDocumentStore,
    SessionId,
    SessionStore,
)
from parlant.core.store_provider import StoreProvider, StoreProviderHints, StoreT
from tests.sdk.utils import Context, SDKTest


_FLAG = "PARLANT_EMPTY_SESSION_CLEANUP_INCLUDE_GREETINGS"
_ENABLED_FLAG = "PARLANT_EMPTY_SESSION_CLEANUP_ENABLED"
_AGENT = AgentId("empty-session-cleanup-sdk-test")


class _Cleanup(Protocol):
    async def run_once(self, *, now: datetime, limit: int = 100) -> object: ...


class _ConditionalCleanup(Protocol):
    async def delete_session_if_empty(
        self,
        session_id: SessionId,
        *,
        created_before: datetime,
        include_greetings: bool,
    ) -> bool: ...


def _cleanup(ctx: Context) -> _Cleanup:
    module = import_module("parlant.core.empty_session_cleanup")
    service_type = cast(type[_Cleanup], getattr(module, "EmptySessionCleanup"))
    assert service_type in ctx.container.defined_types, "SDK must register the cleanup service"
    return ctx.container[service_type]


def _runtime(ctx: Context) -> SessionStore:
    return ctx.container[StoreProvider].get_store(SessionStore, {"call_site": "app"})


def _sweep_time() -> datetime:
    # All boundary fixtures are younger than the real worker's cutoff. Only
    # the explicitly evaluated sweep advances them across that boundary.
    return datetime.now(timezone.utc) + timedelta(minutes=2)


async def _seed(store: SessionStore, at: datetime) -> Session:
    return await store.create_session(
        customer_id=CustomerStore.GUEST_ID,
        agent_id=_AGENT,
        creation_utc=at,
    )


async def _message(
    store: SessionStore, session: Session, source: EventSource, *, trace_id: str = "fixture-input"
) -> Event:
    return await store.create_event(
        session.id,
        source=source,
        kind=EventKind.MESSAGE,
        trace_id=trace_id,
        data={
            "message": "Please help with my booking",
            "participant": {"display_name": "Customer"},
        },
    )


async def _exists(store: SessionStore, session_id: SessionId) -> bool:
    try:
        await store.read_session(session_id)
        return True
    except ItemNotFoundError:
        return False


async def _wait_absent(store: SessionStore, session_id: SessionId) -> None:
    async def wait() -> None:
        while await _exists(store, session_id):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), timeout=2)


class _WarmupPolicy(CostControlPolicy):
    """Real cost-policy extension: forbid paid warming, optionally pause its gate."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.session_id: SessionId | None = None
        self.pause_turn = False
        self.turn_entered = asyncio.Event()
        self.turn_release = asyncio.Event()

    async def check(self, context: CostContext, work: WorkKind) -> CostVerdict:
        if work is WorkKind.BACKGROUND:
            self.session_id = context.session_id
            self.entered.set()
            await self.release.wait()
            return CostVerdict(allowed=False, reason="No paid cache warming in lifecycle tests")
        if work is WorkKind.TURN and self.pause_turn:
            self.turn_entered.set()
            await self.turn_release.wait()
            return CostVerdict(allowed=False, reason="No paid response in lifecycle tests")
        return CostVerdict(allowed=True)


class _CleanupSDKTest(SDKTest):
    greeting_flag: str | None = None
    enabled_flag: str | None = None
    greeting_messages = 1

    async def test_run(self) -> None:
        with pytest.MonkeyPatch.context() as environment:
            environment.delenv(_FLAG, raising=False)
            environment.delenv(_ENABLED_FLAG, raising=False)
            if self.enabled_flag is not None:
                environment.setenv(_ENABLED_FLAG, self.enabled_flag)
            if self.greeting_flag is not None:
                environment.setenv(_FLAG, self.greeting_flag)
            await super().test_run()

    async def configure_container(self, container: p.Container) -> p.Container:
        self.warmup_policy = _WarmupPolicy()
        container[CostControlPolicy] = self.warmup_policy
        self.greeting_entered = asyncio.Event()
        self.greeting_release = asyncio.Event()
        self.greeting_release.set()
        return container

    async def configure_hooks(self, hooks: CompassHooks) -> CompassHooks:
        async def greet(
            context: p.EngineContext, payload: object, exception: Exception | None
        ) -> CompassHookResult:
            # A real application hook emits the deterministic greeting. The
            # engine, coordinator and event emitter all execute normally.
            self.greeting_entered.set()
            await self.greeting_release.wait()
            for _ in range(self.greeting_messages):
                await context.session_event_emitter.emit_message_event(
                    trace_id=context.tracer.trace_id,
                    data="Hello! How can I help?",
                )
            return CompassHookResult.BAIL

        hooks.on_acknowledged.append(greet)
        return hooks

    async def create_server(self, port: int) -> tuple[p.Server, Callable[[], p.Container]]:
        server, container = await super().create_server(port)
        server.diagnostic_reports = False
        return server, container

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id=str(_AGENT),
            name="Cleanup lifecycle test",
            instructions="Test agent with an application-provided greeting.",
        )

    async def greeting(self, ctx: Context) -> Session:
        created = await ctx.client.sessions.create(agent_id=self.agent.id, allow_greeting=True)
        await asyncio.wait_for(self.greeting_entered.wait(), timeout=5)
        tasks = ctx.container[BackgroundTaskService]._tasks
        task = tasks.get(f"process-session({created.id})")
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        session = await _runtime(ctx).read_session(SessionId(created.id))
        messages = await _runtime(ctx).list_events(session.id, kinds=[EventKind.MESSAGE])
        assert len(messages) == self.greeting_messages
        assert all(message.source is EventSource.AI_AGENT for message in messages)
        return session

    async def backdate(self, store: SessionStore, session: Session, at: datetime) -> None:
        while isinstance(store, CompositeSessionStore):
            store = store._writable_store
        assert isinstance(store, SessionDocumentStore)
        await store._session_collection.update_one(
            {"id": {"$eq": session.id}},
            {"creation_utc": at.isoformat()},
        )


class Test_that_empty_session_cleanup_respects_the_thirty_minute_boundary(_CleanupSDKTest):
    async def run(self, ctx: Context) -> None:
        store, now = _runtime(ctx), _sweep_time()
        before = await _seed(store, now - timedelta(minutes=30) + timedelta(microseconds=1))
        boundary = await _seed(store, now - timedelta(minutes=30))
        after = await _seed(store, now - timedelta(minutes=30) - timedelta(microseconds=1))

        await _cleanup(ctx).run_once(now=now)

        assert await _exists(store, before.id)
        assert not await _exists(store, boundary.id)
        assert not await _exists(store, after.id)
        async with httpx.AsyncClient(base_url=ctx.base_url) as http:
            assert (await http.get(f"/sessions/{after.id}")).status_code == 404


@pytest.mark.parametrize("configured_greeting_flag", [None, "false"], indirect=True)
class Test_that_agent_greetings_are_preserved_by_default(_CleanupSDKTest):
    @pytest.fixture(autouse=True)
    def configured_greeting_flag(self, request: pytest.FixtureRequest) -> None:
        self.greeting_flag = cast(str | None, request.param)

    async def run(self, ctx: Context) -> None:
        store, now = _runtime(ctx), _sweep_time()
        session = await self.greeting(ctx)
        await self.backdate(store, session, now - timedelta(minutes=31))

        await _cleanup(ctx).run_once(now=now)

        assert await _exists(store, session.id)
        assert len(await store.list_events(session.id, kinds=[EventKind.MESSAGE])) == 1


class Test_that_the_environment_flag_includes_initial_greeting_only_sessions(_CleanupSDKTest):
    greeting_flag = "true"

    async def run(self, ctx: Context) -> None:
        store, now = _runtime(ctx), _sweep_time()
        greeting = await self.greeting(ctx)
        await self.backdate(store, greeting, now - timedelta(minutes=31))
        unknown = await _seed(store, now - timedelta(minutes=31))
        await _message(store, unknown, EventSource.AI_AGENT, trace_id="unattributed-proactive-turn")

        await _cleanup(ctx).run_once(now=now)

        assert not await _exists(store, greeting.id)
        assert await _exists(store, unknown.id), (
            "Unknown agent work is not a proven initial greeting"
        )


class Test_that_customer_and_human_participation_prevent_empty_cleanup(_CleanupSDKTest):
    greeting_flag = "true"

    async def run(self, ctx: Context) -> None:
        store, now = _runtime(ctx), _sweep_time()
        protected = []
        for source in (
            EventSource.CUSTOMER,
            EventSource.HUMAN_AGENT,
            EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT,
        ):
            session = await _seed(store, now - timedelta(minutes=31))
            await _message(store, session, source)
            protected.append(session.id)
        ui = await _seed(store, now - timedelta(minutes=31))
        await store.create_event(
            ui.id,
            source=EventSource.CUSTOMER_UI,
            kind=EventKind.CUSTOM,
            trace_id="customer-action",
            data={"type": "button_clicked", "action": "request_human"},
        )
        protected.append(ui.id)

        await _cleanup(ctx).run_once(now=now)

        for session_id in protected:
            assert await _exists(store, session_id)


class Test_that_deleted_customer_messages_do_not_make_a_conversation_empty(_CleanupSDKTest):
    greeting_flag = "true"

    async def run(self, ctx: Context) -> None:
        store, now = _runtime(ctx), _sweep_time()
        session = await _seed(store, now - timedelta(minutes=31))
        message = await _message(store, session, EventSource.CUSTOMER)
        await store.delete_event(message.id)
        assert not await store.list_events(session.id)

        await _cleanup(ctx).run_once(now=now)

        assert await _exists(store, session.id)
        history = await store.list_events(session.id, exclude_deleted=False)
        assert len(history) == 1 and history[0].deleted


class Test_that_initialization_and_status_updates_do_not_keep_an_empty_session_alive(
    _CleanupSDKTest
):
    async def run(self, ctx: Context) -> None:
        store, now = _runtime(ctx), _sweep_time()
        session = await _seed(store, now - timedelta(minutes=31))
        await store.update_session(session.id, {"metadata": {"cache_warmed": True}})
        await store.create_event(
            session.id,
            source=EventSource.AI_AGENT,
            kind=EventKind.STATUS,
            trace_id="finished-initialization",
            data={"status": "ready", "data": {}},
        )
        await store.create_event(
            session.id,
            source=EventSource.SYSTEM,
            kind=EventKind.CUSTOM,
            trace_id="finished-initialization",
            data={"type": "cache_warmup_finished"},
        )

        await _cleanup(ctx).run_once(now=now)

        assert not await _exists(store, session.id)


class _PausedDeletionStore(SessionDocumentStore):
    """Pause before the real conditional operation, retaining all real writes."""

    async def delete_session_if_empty(
        self, session_id: SessionId, *, created_before: datetime, include_greetings: bool
    ) -> bool:
        if session_id == self.pause_session:
            self.deleting.set()
            await self.continue_deleting.wait()
        return await cast(_ConditionalCleanup, super()).delete_session_if_empty(
            session_id, created_before=created_before, include_greetings=include_greetings
        )

    pause_session: SessionId | None = None
    deleting: asyncio.Event
    continue_deleting: asyncio.Event


class _RuntimeProvider(StoreProvider):
    def __init__(self, base: StoreProvider, runtime: SessionStore) -> None:
        self.base, self.runtime = base, runtime

    def get_store(
        self, store_type: Callable[..., StoreT], hints: StoreProviderHints = {}
    ) -> StoreT:
        if store_type is SessionStore and hints.get("call_site") in {"app", "engine"}:
            return cast(StoreT, self.runtime)
        return self.base.get_store(store_type, hints)


class Test_that_cleanup_and_the_first_customer_message_do_not_lose_or_orphan_the_message(
    _CleanupSDKTest
):
    async def test_run(self) -> None:
        async with _PausedDeletionStore(TransientDocumentDatabase()) as store:
            self.race_store = store
            store.deleting, store.continue_deleting = asyncio.Event(), asyncio.Event()
            await super().test_run()

    async def configure_container(self, container: p.Container) -> p.Container:
        container = await super().configure_container(container)
        container[StoreProvider] = _RuntimeProvider(container[StoreProvider], self.race_store)
        return container

    async def run(self, ctx: Context) -> None:
        store, now = self.race_store, _sweep_time()
        incoming = await _seed(store, now - timedelta(minutes=31))
        store.pause_session = incoming.id
        service = _cleanup(ctx)
        sweep = asyncio.create_task(service.run_once(now=now))
        try:
            await asyncio.wait_for(store.deleting.wait(), timeout=2)
            first_message = await _message(store, incoming, EventSource.CUSTOMER)
        finally:
            store.continue_deleting.set()
            await asyncio.wait_for(sweep, timeout=2)
        assert await _exists(store, incoming.id)
        saved_message = await store.read_event(incoming.id, first_message.id)
        assert saved_message.source is EventSource.CUSTOMER
        assert isinstance(saved_message.data, dict)
        assert saved_message.data["message"] == "Please help with my booking"

        # Reverse ordering: cleanup wins before a first message is accepted.
        removed = await _seed(store, now - timedelta(minutes=31))
        await service.run_once(now=now)
        assert not await _exists(store, removed.id)
        with pytest.raises(ItemNotFoundError):
            await _message(store, removed, EventSource.CUSTOMER)
        assert not await store._event_collection.find({"session_id": {"$eq": removed.id}})


class Test_that_restart_cleanup_is_bounded_and_idempotent(_CleanupSDKTest):
    async def test_run(self) -> None:
        database = TransientDocumentDatabase()
        async with SessionDocumentStore(database) as original:
            self.old_after_restart = await _seed(
                original, datetime.now(timezone.utc) - timedelta(hours=1)
            )
        async with SessionDocumentStore(database) as reopened:
            self.reopened_store = reopened
            await super().test_run()

    async def configure_container(self, container: p.Container) -> p.Container:
        container = await super().configure_container(container)
        container[StoreProvider] = _RuntimeProvider(container[StoreProvider], self.reopened_store)
        return container

    async def run(self, ctx: Context) -> None:
        store = _runtime(ctx)
        await _wait_absent(store, self.old_after_restart.id)
        now = _sweep_time()
        sessions = [await _seed(store, now - timedelta(minutes=31)) for _ in range(5)]
        service = _cleanup(ctx)

        await service.run_once(now=now, limit=2)
        remaining = [s.id for s in sessions if await _exists(store, s.id)]
        assert len(remaining) == 3
        await asyncio.gather(service.run_once(now=now, limit=2), service.run_once(now=now, limit=2))
        await service.run_once(now=now, limit=2)
        await service.run_once(now=now, limit=2)
        assert not any([await _exists(store, session.id) for session in sessions])


class Test_that_cleanup_uses_the_writable_runtime_store_and_preserves_read_only_sessions(
    _CleanupSDKTest
):
    async def test_run(self) -> None:
        async with AsyncExitStack() as stack:
            self.runtime_store = await stack.enter_async_context(
                SessionDocumentStore(TransientDocumentDatabase())
            )
            self.read_only_store = await stack.enter_async_context(
                SessionDocumentStore(TransientDocumentDatabase())
            )
            await super().test_run()

    async def configure_container(self, container: p.Container) -> p.Container:
        container = await super().configure_container(container)
        composite = CompositeSessionStore(self.runtime_store, [self.read_only_store])
        container[StoreProvider] = _RuntimeProvider(container[StoreProvider], composite)
        return container

    async def run(self, ctx: Context) -> None:
        now = _sweep_time()
        self.declaration_store = ctx.container[SessionStore]
        writable = await _seed(self.runtime_store, now - timedelta(minutes=31))
        read_only = [await _seed(self.read_only_store, now - timedelta(hours=2)) for _ in range(3)]
        declared = await _seed(self.declaration_store, now - timedelta(minutes=31))

        await _cleanup(ctx).run_once(now=now, limit=1)

        assert not await _exists(self.runtime_store, writable.id)
        for session in read_only:
            assert await _exists(self.read_only_store, session.id)
        assert await _exists(self.declaration_store, declared.id)


class Test_that_the_cleanup_worker_stops_with_the_server(_CleanupSDKTest):
    async def test_run(self) -> None:
        await super().test_run()
        assert self.worker.done(), "Cleanup worker must not survive SDK shutdown"

    async def run(self, ctx: Context) -> None:
        tasks = ctx.container[BackgroundTaskService]._tasks
        assert "empty-session-cleanup" in tasks, "SDK must start its cleanup worker"
        self.worker = tasks["empty-session-cleanup"]
        assert not self.worker.done()


class Test_that_cleanup_preserves_in_flight_initialization_and_greeting(_CleanupSDKTest):
    greeting_flag = "true"

    async def run(self, ctx: Context) -> None:
        store, now = _runtime(ctx), _sweep_time()
        policy = self.warmup_policy
        policy.entered.clear()
        policy.release.clear()
        creating = asyncio.create_task(ctx.client.sessions.create(agent_id=self.agent.id))
        try:
            await asyncio.wait_for(policy.entered.wait(), timeout=5)
            assert policy.session_id is not None
            initializing = await store.read_session(policy.session_id)
            await self.backdate(store, initializing, now - timedelta(minutes=31))
            service = _cleanup(ctx)
            await service.run_once(now=now)
            assert await _exists(store, initializing.id)
        finally:
            policy.release.set()
            await asyncio.wait_for(creating, timeout=5)
        await service.run_once(now=now)
        assert not await _exists(store, initializing.id)

        self.greeting_entered.clear()
        self.greeting_release.clear()
        try:
            created = await ctx.client.sessions.create(agent_id=self.agent.id, allow_greeting=True)
            await asyncio.wait_for(self.greeting_entered.wait(), timeout=5)
            greeting = await store.read_session(SessionId(created.id))
            await self.backdate(store, greeting, now - timedelta(minutes=31))
            await service.run_once(now=now)
            assert await _exists(store, greeting.id)
            task = ctx.container[BackgroundTaskService]._tasks[f"process-session({greeting.id})"]
        finally:
            self.greeting_release.set()
        await asyncio.wait_for(asyncio.shield(task), timeout=5)
        await service.run_once(now=now)
        assert not await _exists(store, greeting.id)


@pytest.mark.parametrize("include_greetings", ["false", "true"], indirect=True)
class Test_that_disabling_cleanup_preserves_old_sessions_and_starts_no_worker(_CleanupSDKTest):
    enabled_flag = "false"

    @pytest.fixture(autouse=True)
    def include_greetings(self, request: pytest.FixtureRequest) -> None:
        self.greeting_flag = cast(str, request.param)

    async def test_run(self) -> None:
        async with SessionDocumentStore(TransientDocumentDatabase()) as store:
            self.runtime_store = store
            self.old_session = await _seed(store, datetime.now(timezone.utc) - timedelta(hours=1))
            await super().test_run()

    async def configure_container(self, container: p.Container) -> p.Container:
        container = await super().configure_container(container)
        container[StoreProvider] = _RuntimeProvider(container[StoreProvider], self.runtime_store)
        return container

    async def run(self, ctx: Context) -> None:
        assert "empty-session-cleanup" not in ctx.container[BackgroundTaskService]._tasks
        assert await _exists(self.runtime_store, self.old_session.id)
        assert await _cleanup(ctx).run_once(now=_sweep_time()) == 0
        assert await _exists(self.runtime_store, self.old_session.id)


class Test_that_an_invalid_cleanup_toggle_gives_an_actionable_configuration_error(_CleanupSDKTest):
    async def run(self, ctx: Context) -> None:
        module = import_module("parlant.core.empty_session_cleanup")
        with pytest.MonkeyPatch.context() as environment:
            environment.setenv(_ENABLED_FLAG, "flase")
            with pytest.raises(ValueError, match=f"{_ENABLED_FLAG}.*true.*false"):
                module.EmptySessionCleanup(ctx.container[StoreProvider], ctx.container[Logger])


class Test_that_passive_page_views_do_not_count_as_customer_participation(_CleanupSDKTest):
    greeting_flag = "true"

    async def run(self, ctx: Context) -> None:
        store, now = _runtime(ctx), _sweep_time()
        page, action = [await _seed(store, now - timedelta(minutes=31)) for _ in range(2)]
        for session, data in (
            (page, {"type": "page_view", "page": "/pricing"}),
            (action, {"type": "button_clicked", "action": "request_human"}),
        ):
            await store.create_event(
                session.id, EventSource.CUSTOMER_UI, EventKind.CUSTOM, "client-event", data
            )
        await _cleanup(ctx).run_once(now=now)
        assert not await _exists(store, page.id)
        assert await _exists(store, action.id)


class Test_that_all_messages_in_the_initial_greeting_follow_the_same_cleanup_rule(_CleanupSDKTest):
    greeting_flag = "true"
    greeting_messages = 2

    async def run(self, ctx: Context) -> None:
        store, now = _runtime(ctx), _sweep_time()
        session = await self.greeting(ctx)
        await self.backdate(store, session, now - timedelta(minutes=31))
        await _cleanup(ctx).run_once(now=now)
        assert not await _exists(store, session.id)


class Test_that_protected_old_sessions_do_not_starve_later_empty_candidates(_CleanupSDKTest):
    greeting_flag = "true"

    async def run(self, ctx: Context) -> None:
        store, now = _runtime(ctx), _sweep_time()
        protected = [await _seed(store, now - timedelta(hours=2)) for _ in range(3)]
        for session in protected:
            await _message(store, session, EventSource.AI_AGENT, trace_id="unknown-agent-work")
        empty = [await _seed(store, now - timedelta(minutes=31)) for _ in range(2)]
        for _ in range(6):
            await _cleanup(ctx).run_once(now=now, limit=1)
        assert all([await _exists(store, session.id) for session in protected])
        assert not any([await _exists(store, session.id) for session in empty])


class Test_that_cancelling_a_greeting_releases_its_cleanup_protection(_CleanupSDKTest):
    greeting_flag = "true"

    async def run(self, ctx: Context) -> None:
        store, now = _runtime(ctx), _sweep_time()
        self.greeting_release.clear()
        created = await ctx.client.sessions.create(agent_id=self.agent.id, allow_greeting=True)
        await asyncio.wait_for(self.greeting_entered.wait(), timeout=5)
        session = await store.read_session(SessionId(created.id))
        await self.backdate(store, session, now - timedelta(minutes=31))
        task = ctx.container[BackgroundTaskService]._tasks[f"process-session({session.id})"]
        try:
            await _cleanup(ctx).run_once(now=now)
            assert await _exists(store, session.id)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self.greeting_release.set()
        await _cleanup(ctx).run_once(now=now)
        assert not await _exists(store, session.id)


class _DispatchFailureTasks(BackgroundTaskService):
    @override
    async def restart(self, f: Coroutine[Any, Any, None], /, *, tag: str) -> asyncio.Task[None]:
        if tag.startswith("process-session("):
            raise OSError("Task dispatch unavailable")
        return await super().restart(f, tag=tag)


class Test_that_failed_greeting_dispatch_releases_its_cleanup_protection(_CleanupSDKTest):
    greeting_flag = "true"

    async def configure_container(self, container: p.Container) -> p.Container:
        container = await super().configure_container(container)
        container[BackgroundTaskService] = _DispatchFailureTasks(container[Logger])
        return container

    async def run(self, ctx: Context) -> None:
        async with httpx.AsyncClient(base_url=ctx.base_url) as client:
            failed = await client.post(
                "/sessions?allow_greeting=true",
                json={
                    "agent_id": self.agent.id,
                    "allow_greeting": True,
                },
            )
        assert failed.status_code == 500
        store, now = _runtime(ctx), _sweep_time()
        [session] = (await store.list_sessions()).items
        await self.backdate(store, session, now - timedelta(minutes=31))
        await _cleanup(ctx).run_once(now=now)
        assert not await _exists(store, session.id)


@pytest.mark.parametrize("use_queue", [False, True], indirect=True)
class Test_that_cleanup_preserves_explicit_or_queued_utter_work(_CleanupSDKTest):
    @pytest.fixture(autouse=True)
    def use_queue(self, request: pytest.FixtureRequest) -> None:
        self.queued = bool(request.param)

    async def run(self, ctx: Context) -> None:
        store, now = _runtime(ctx), _sweep_time()
        created = await ctx.client.sessions.create(agent_id=self.agent.id)
        session = await store.read_session(SessionId(created.id))
        await self.backdate(store, session, now - timedelta(minutes=31))
        policy = self.warmup_policy
        policy.pause_turn = True
        module = ctx.container[Application].sessions
        work: Awaitable[object]
        if self.queued:
            await module.queue_response(session.id)
            work = ctx.container[BackgroundTaskService]._tasks[
                module.response_coordinator.task_tag(session.id)
            ]
        else:
            work = asyncio.create_task(
                module.utter(
                    session.id,
                    [
                        UtteranceRequest(
                            action="Follow up about the pending request.",
                            rationale=UtteranceRationale.FOLLOW_UP,
                        )
                    ],
                )
            )
        try:
            await asyncio.wait_for(policy.turn_entered.wait(), timeout=5)
            await _cleanup(ctx).run_once(now=now)
            assert await _exists(store, session.id)
        finally:
            policy.turn_release.set()
            await asyncio.wait_for(asyncio.gather(work, return_exceptions=True), timeout=5)
        await _cleanup(ctx).run_once(now=now)
        assert not await _exists(store, session.id)
