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

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest

from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.loggers import Logger
from parlant.core.sessions import SessionId

if TYPE_CHECKING:
    from parlant.core.response_coordination import ResponseRun


SID = SessionId("queued-session")


class Harness:
    def __init__(self, background: BackgroundTaskService, logger: Logger) -> None:
        from parlant.core.response_coordination import LocalResponseCoordinator

        self.inputs: frozenset[str] = frozenset({"customer:first"})
        self.delivered: list[frozenset[str]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.started_runs: list[ResponseRun] = []
        self.coordinator = LocalResponseCoordinator(
            background_tasks=background,
            logger=logger,
            read_inputs=self.read_inputs,
            render=self.render,
        )

    async def read_inputs(self, session_id: SessionId) -> frozenset[str]:
        assert session_id == SID
        return self.inputs

    async def render(self, run: ResponseRun) -> bool:
        self.started_runs.append(run)
        snapshot = self.inputs
        await run.observe_inputs(snapshot)
        self.started.set()
        await self.release.wait()

        async def deliver() -> None:
            self.delivered.append(snapshot)

        await run.publish(deliver, visible=True)
        return True


@asynccontextmanager
async def harness(logger: Logger) -> AsyncIterator[Harness]:
    async with BackgroundTaskService(logger) as background:
        yield Harness(background, logger)


async def idle(h: Harness) -> None:
    await asyncio.wait_for(h.coordinator.wait_idle(SID), timeout=2)


async def test_that_concurrent_requests_share_one_pending_response(logger: Logger) -> None:
    async with harness(logger) as h:
        await asyncio.gather(*(h.coordinator.request(SID) for _ in range(3)))
        await idle(h)
        assert h.delivered == [frozenset({"customer:first"})]


async def test_that_registered_work_delays_only_the_deferred_response(logger: Logger) -> None:
    async with harness(logger) as h:
        a, b = asyncio.Event(), asyncio.Event()
        saved = asyncio.Event()

        async def work_a() -> None:
            await a.wait()
            h.inputs |= {"result:A"}
            saved.set()

        async def work_b() -> None:
            await b.wait()
            h.inputs |= {"result:B"}

        await h.coordinator.request(SID, after=work_a())
        await h.coordinator.request(SID, after=work_b())
        a.set()
        await asyncio.wait_for(saved.wait(), timeout=2)
        # A foreground response can still run, but has not seen A yet.
        foreground = await h.coordinator.begin_foreground(SID)
        await foreground.observe_inputs(frozenset({"customer:first"}))

        async def acknowledge() -> None:
            pass

        await foreground.publish(acknowledge, visible=True)
        await h.coordinator.finish_foreground(foreground, succeeded=True)
        assert h.delivered == []
        b.set()
        await idle(h)
        assert h.delivered == [frozenset({"customer:first", "result:A", "result:B"})]


async def test_that_a_foreground_reply_consumes_only_inputs_it_saw(logger: Logger) -> None:
    async with harness(logger) as h:
        foreground = await h.coordinator.begin_foreground(SID)
        await foreground.observe_inputs(h.inputs)
        await h.coordinator.request(SID)

        async def publish() -> None:
            pass

        await foreground.publish(publish, visible=True)
        h.inputs |= {"result:late"}
        await h.coordinator.request(SID)
        await h.coordinator.finish_foreground(foreground, succeeded=True)
        await idle(h)
        assert h.delivered == [frozenset({"customer:first", "result:late"})]


async def test_that_an_active_foreground_reply_absorbs_requests_for_seen_inputs(
    logger: Logger,
) -> None:
    async with harness(logger) as h:
        foreground = await h.coordinator.begin_foreground(SID)
        await foreground.observe_inputs(h.inputs)
        await h.coordinator.request(SID)

        async def publish() -> None:
            pass

        await foreground.publish(publish, visible=True)
        await h.coordinator.finish_foreground(foreground, succeeded=True)
        await idle(h)
        assert h.delivered == []


async def test_that_new_inputs_before_publication_refresh_the_deferred_reply(
    logger: Logger,
) -> None:
    async with harness(logger) as h:
        h.release.clear()
        await h.coordinator.request(SID)
        await asyncio.wait_for(h.started.wait(), timeout=2)
        h.inputs |= {"result:late"}
        await h.coordinator.request(SID)
        h.release.set()
        await idle(h)
        assert h.delivered == [frozenset({"customer:first", "result:late"})]
        assert len(h.started_runs) == 2


async def test_that_new_inputs_after_visible_output_leave_one_pending_successor(
    logger: Logger,
) -> None:
    async with harness(logger) as h:
        visible, finish = asyncio.Event(), asyncio.Event()
        original = h.coordinator._render

        async def paused_render(run: ResponseRun) -> bool:
            await run.observe_inputs(h.inputs)

            async def publish() -> None:
                h.delivered.append(run.inputs)

            await run.publish(publish, visible=True)
            visible.set()
            await finish.wait()
            return True

        h.coordinator._render = paused_render
        await h.coordinator.request(SID)
        await asyncio.wait_for(visible.wait(), timeout=2)
        h.inputs |= {"result:new"}
        await asyncio.gather(*(h.coordinator.request(SID) for _ in range(3)))
        h.coordinator._render = original
        finish.set()
        await idle(h)
        assert h.delivered == [
            frozenset({"customer:first"}),
            frozenset({"customer:first", "result:new"}),
        ]


async def test_that_customer_interruption_preserves_uncovered_requests(logger: Logger) -> None:
    from parlant.core.response_coordination import ResponseSuperseded

    async with harness(logger) as h:
        h.release.clear()
        await h.coordinator.request(SID)
        await asyncio.wait_for(h.started.wait(), timeout=2)
        foreground = await h.coordinator.begin_foreground(SID)
        h.inputs |= {"customer:second"}
        await foreground.observe_inputs(h.inputs)
        h.inputs |= {"result:after-snapshot"}
        await h.coordinator.request(SID)

        async def publish() -> None:
            pass

        # The stale foreground cannot publish a reply that omits ready new data.
        with pytest.raises(ResponseSuperseded):
            await foreground.publish(publish, visible=True)
        await h.coordinator.finish_foreground(foreground, succeeded=False)
        h.release.set()
        await idle(h)
        assert h.delivered == [h.inputs]


async def test_that_failed_jobs_release_the_batch_without_losing_other_results(
    logger: Logger,
) -> None:
    async with harness(logger) as h:
        release = asyncio.Event()

        async def fail() -> None:
            await release.wait()
            raise RuntimeError("expected background work failure")

        async def succeed() -> None:
            await release.wait()
            h.inputs |= {"result:success"}

        await h.coordinator.request(SID, after=fail())
        await h.coordinator.request(SID, after=succeed())
        release.set()
        await idle(h)
        assert h.delivered == [frozenset({"customer:first", "result:success"})]
        assert h.coordinator.session_count == 0


async def test_that_independent_sessions_do_not_block_each_other(logger: Logger) -> None:
    from parlant.core.response_coordination import LocalResponseCoordinator

    slow_started, fast_done, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async with BackgroundTaskService(logger) as background:

        async def read_inputs(session_id: SessionId) -> frozenset[str]:
            return frozenset({str(session_id)})

        async def render(run: ResponseRun) -> bool:
            await run.observe_inputs(frozenset({str(run.session_id)}))
            if run.session_id == SID:
                slow_started.set()
                await release.wait()

            async def publish() -> None:
                if run.session_id != SID:
                    fast_done.set()

            await run.publish(publish, visible=True)
            return True

        coordinator = LocalResponseCoordinator(
            background_tasks=background, logger=logger, read_inputs=read_inputs, render=render
        )
        await coordinator.request(SID)
        await asyncio.wait_for(slow_started.wait(), timeout=2)
        await coordinator.request(SessionId("another-session"))
        await asyncio.wait_for(fast_done.wait(), timeout=2)
        release.set()
        await asyncio.wait_for(coordinator.wait_idle(SID), timeout=2)


async def test_that_shutdown_cancels_registered_work_and_releases_session_state(
    logger: Logger,
) -> None:
    started, unwound = asyncio.Event(), asyncio.Event()
    async with BackgroundTaskService(logger) as background:
        h = Harness(background, logger)

        async def work() -> None:
            try:
                started.set()
                await asyncio.Future[None]()
            finally:
                unwound.set()

        await h.coordinator.request(SID, after=work())
        await asyncio.wait_for(started.wait(), timeout=2)
    assert unwound.is_set()
    assert h.coordinator.session_count == 0


async def test_that_a_waiting_request_survives_foreground_state_retirement(logger: Logger) -> None:
    async with harness(logger) as h:
        foreground = await h.coordinator.begin_foreground(SID)
        await foreground.observe_inputs(h.inputs)
        entered, release = asyncio.Event(), asyncio.Event()

        async def publish() -> None:
            entered.set()
            await release.wait()

        publishing = asyncio.create_task(foreground.publish(publish, visible=True))
        await asyncio.wait_for(entered.wait(), timeout=2)
        finishing = asyncio.create_task(h.coordinator.finish_foreground(foreground, succeeded=True))
        incoming = asyncio.create_task(h.coordinator.request(SID))
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(publishing, finishing, incoming)
        await idle(h)
        assert h.delivered == [h.inputs]
        assert h.coordinator.session_count == 0


async def test_that_streaming_updates_cannot_publish_an_unseen_result(logger: Logger) -> None:
    from parlant.core.emission.event_buffer import EventBuffer
    from parlant.core.response_coordination import CoordinatedEventEmitter, ResponseSuperseded
    from tests.core.stable.engines.compass.matching.utils import create_agent

    async with harness(logger) as h:
        foreground = await h.coordinator.begin_foreground(SID)
        await foreground.observe_inputs(h.inputs)
        buffer = EventBuffer(create_agent())
        emitter = CoordinatedEventEmitter(buffer, foreground)
        handle = await emitter.emit_message_event("trace", "")
        assert not foreground.visible
        h.inputs |= {"result:new"}
        await h.coordinator.request(SID)
        with pytest.raises(ResponseSuperseded):
            await handle.update({"message": "stale text", "participant": {"display_name": "Agent"}})
        assert isinstance(buffer.events[0].data, Mapping)
        assert buffer.events[0].data["message"] == ""
        await h.coordinator.finish_foreground(foreground, succeeded=False)
        await idle(h)


async def test_that_streaming_handles_keep_the_guard_after_each_update(logger: Logger) -> None:
    from parlant.core.emission.event_buffer import EventBuffer
    from parlant.core.response_coordination import CoordinatedEventEmitter, ResponseSuperseded
    from tests.core.stable.engines.compass.matching.utils import create_agent

    async with harness(logger) as h:
        first = await h.coordinator.begin_foreground(SID)
        await first.observe_inputs(h.inputs)
        buffer = EventBuffer(create_agent())
        emitter = CoordinatedEventEmitter(buffer, first)
        handle = await emitter.emit_message_event("trace", "")
        handle = await handle.update(
            {"message": "visible text", "participant": {"display_name": "Agent"}}
        )
        assert first.visible
        successor = await h.coordinator.begin_foreground(SID)
        with pytest.raises(ResponseSuperseded):
            await handle.update(
                {"message": "stale continuation", "participant": {"display_name": "Agent"}}
            )
        await emitter.emit_status_event("trace", {"status": "cancelled"})
        assert len(buffer.events) == 2
        assert isinstance(buffer.events[0].data, Mapping)
        assert buffer.events[0].data["message"] == "visible text"
        await h.coordinator.finish_foreground(successor, succeeded=True)


async def test_that_work_already_answered_by_a_completed_customer_turn_does_not_repeat(
    logger: Logger,
) -> None:
    async with harness(logger) as h:
        saved, release = asyncio.Event(), asyncio.Event()

        async def work() -> None:
            h.inputs |= {"result:A"}
            saved.set()
            await release.wait()

        await h.coordinator.request(SID, after=work())
        await asyncio.wait_for(saved.wait(), timeout=2)
        foreground = await h.coordinator.begin_foreground(SID)
        await foreground.observe_inputs(h.inputs)

        async def publish() -> None:
            pass

        await foreground.publish(publish, visible=True)
        await h.coordinator.finish_foreground(foreground, succeeded=True)
        release.set()
        await idle(h)
        assert h.delivered == []


async def test_that_a_failed_generation_releases_pending_state_without_retrying_forever(
    logger: Logger,
) -> None:
    async with harness(logger) as h:

        async def fail(run: ResponseRun) -> bool:
            raise RuntimeError("model unavailable")

        h.coordinator._render = fail
        await h.coordinator.request(SID)
        await idle(h)
        assert h.coordinator.session_count == 0


async def test_that_a_renderer_without_snapshot_reporting_does_not_repeat_forever(
    logger: Logger,
) -> None:
    async with harness(logger) as h:
        calls = 0

        async def render_without_coverage(run: ResponseRun) -> bool:
            nonlocal calls
            calls += 1

            async def publish() -> None:
                pass

            await run.publish(publish, visible=True)
            return True

        h.coordinator._render = render_without_coverage
        await h.coordinator.request(SID)
        await idle(h)
        assert calls == 1
        assert h.coordinator.session_count == 0
