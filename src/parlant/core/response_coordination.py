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

"""Process-local coordination of deferred replies and ordinary customer turns.

Intents and registrations are deliberately ephemeral. Session affinity is needed;
this module does not claim distributed exclusion or restart recovery.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import Context, ContextVar, copy_context
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Literal, Mapping, TypeVar, cast
from typing_extensions import override

from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.common import JSONSerializable
from parlant.core.emissions import (
    EmittedEvent,
    EventEmitter,
    MessageEventHandle,
    StatusEventHandle,
)
from parlant.core.loggers import Logger
from parlant.core.sessions import (
    Event,
    EventKind,
    EventSource,
    MessageEventData,
    SessionId,
    StatusEventData,
    ToolEventData,
)

T = TypeVar("T")
Inputs = frozenset[str]


class ResponseSuperseded(asyncio.CancelledError):
    """An unpublished reply no longer covers the pending session inputs."""


def tool_input(trace_id: str, data: object) -> str:
    return (
        "tool:"
        + hashlib.sha256(
            (trace_id + json.dumps(data, sort_keys=True, ensure_ascii=False)).encode()
        ).hexdigest()
    )


def response_inputs(events: Sequence[Event]) -> Inputs:
    inputs: set[str] = set()
    for event in events:
        if event.deleted:
            continue
        if event.kind == EventKind.MESSAGE and event.source in (
            EventSource.CUSTOMER,
            EventSource.CUSTOMER_UI,
        ):
            inputs.add(f"customer:{event.id}")
        elif event.kind == EventKind.TOOL:
            inputs.add(tool_input(event.trace_id, event.data))
    return frozenset(inputs)


@dataclass(frozen=True)
class _Request:
    inputs: Inputs
    context: Context


def _retain_live_inputs(state: _SessionState, inputs: Inputs) -> None:
    state.requests = {
        seq: _Request(request.inputs & inputs, request.context)
        for seq, request in state.requests.items()
        if not request.inputs or request.inputs & inputs
    }


@dataclass
class _SessionState:
    session_id: SessionId
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    idle: asyncio.Event = field(default_factory=asyncio.Event)
    generation: int = 0
    sequence: int = 0
    requests: dict[int, _Request] = field(default_factory=dict)
    jobs: dict[int, asyncio.Task[None] | None] = field(default_factory=dict)
    foreground: ResponseRun | None = None
    queued: ResponseRun | None = None
    rendering: asyncio.Task[bool] | None = None
    worker_installed: bool = False
    installation: asyncio.Task[None] | None = None
    closed: bool = False
    users: int = 0
    answered_inputs: Inputs | None = None


class ResponseRun:
    def __init__(
        self,
        state: _SessionState,
        kind: Literal["foreground", "queued"],
        read_inputs: Callable[[SessionId], Awaitable[Inputs]],
    ) -> None:
        self._state = state
        self._read_inputs = read_inputs
        self.session_id = state.session_id
        self.kind = kind
        self.generation = state.generation
        self.inputs: Inputs = frozenset()
        self.snapshot_loaded = False
        self.visible = False

    async def observe_inputs(self, inputs: Inputs) -> None:
        async with self._state.lock:
            self.inputs |= inputs
            self.snapshot_loaded = True

    async def publish(self, operation: Callable[[], Awaitable[T]], *, visible: bool) -> T:
        state = self._state
        async with state.lock:
            if state.closed or self.generation != state.generation:
                raise ResponseSuperseded("A newer customer turn superseded this reply")
            if self.kind == "queued" and state.foreground is not None:
                raise ResponseSuperseded("A customer turn is handling this session")
            if not self.visible and state.requests:
                # Read under the publication lock: an older model snapshot is
                # not evidence that a newly registered result was deleted.
                current = await self._read_inputs(self.session_id)
                _retain_live_inputs(state, current)
                if self.snapshot_loaded and not self.inputs <= current:
                    raise ResponseSuperseded("Conversation history changed during generation")
            if not self.visible and self.snapshot_loaded:
                unseen = any(
                    not request.inputs <= self.inputs for request in state.requests.values()
                )
                if self.kind == "queued" and state.jobs:
                    raise ResponseSuperseded("Registered background work is still pending")
                if unseen and (self.kind == "queued" or not state.jobs):
                    raise ResponseSuperseded("New information arrived after the response snapshot")
            # The same local lock protects incoming intents and publication. No
            # stale check followed by an unguarded await/write in this process.
            result = await operation()
            if visible:
                self.visible = True
            return result


_current_run: ContextVar[ResponseRun | None] = ContextVar("queued_response_run", default=None)


@contextmanager
def response_run_scope(run: ResponseRun | None) -> Iterator[None]:
    token = _current_run.set(run)
    try:
        yield
    finally:
        _current_run.reset(token)


async def observe_response_inputs(session_id: SessionId, events: Sequence[Event]) -> None:
    run = _current_run.get()
    if run is not None and run.session_id == session_id:
        await run.observe_inputs(response_inputs(events))


class CoordinatedEventEmitter(EventEmitter):
    """Guard initial text and streaming updates at their actual persistence boundary."""

    def __init__(
        self,
        emitter: EventEmitter,
        run: ResponseRun,
        *,
        may_publish: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        self._emitter = emitter
        self._run = run
        self._may_publish = may_publish
        self._open_messages: dict[object, tuple[MessageEventHandle, MessageEventData]] = {}

    async def _publish(self, operation: Callable[[], Awaitable[T]], *, visible: bool) -> T:
        async def guarded() -> T:
            if self._may_publish is not None and not await self._may_publish():
                raise ResponseSuperseded("Automatic replies are no longer allowed")
            return await operation()

        return await self._run.publish(guarded, visible=visible)

    def _handle(self, handle: MessageEventHandle, key: object | None = None) -> MessageEventHandle:
        key = key if key is not None else object()
        data = cast(MessageEventData, handle.event.data)
        chunks = data.get("chunks")
        if chunks and chunks[-1] is not None:
            # The loop mutates its chunk buffer before the next write. Freeze
            # only what this handle actually persisted, not that live buffer.
            self._open_messages[key] = (handle, {**data, "chunks": list(chunks)})
        else:
            self._open_messages.pop(key, None)

        async def update(
            data: MessageEventData,
            *,
            internal_metadata: Mapping[str, JSONSerializable] | None = None,
        ) -> MessageEventHandle:
            result = await self._publish(
                lambda: handle.update(data, internal_metadata=internal_metadata),
                visible=bool(data["message"].strip()),
            )
            return self._handle(result, key)

        return MessageEventHandle(event=handle.event, update=update)

    async def close_interrupted_messages(self) -> None:
        async with self._run._state.lock:
            for key, (handle, data) in list(self._open_messages.items()):
                # Closing a committed prefix is allowed after supersession or
                # handoff; generating or appending new text is still fenced.
                await handle.update({**data, "chunks": [*data["chunks"], None]})
                self._open_messages.pop(key, None)

    @override
    async def emit_message_event(
        self,
        trace_id: str,
        data: str | MessageEventData,
        metadata: Mapping[str, JSONSerializable] | None = None,
        *,
        source: EventSource = EventSource.AI_AGENT,
    ) -> MessageEventHandle:
        result = await self._publish(
            lambda: self._emitter.emit_message_event(trace_id, data, metadata, source=source),
            visible=bool((data if isinstance(data, str) else data["message"]).strip()),
        )
        return self._handle(result)

    @override
    async def emit_status_event(
        self,
        trace_id: str,
        data: StatusEventData,
        metadata: Mapping[str, JSONSerializable] | None = None,
    ) -> StatusEventHandle:
        # Cancellation/completion must still reach listeners after supersession.
        return await self._emitter.emit_status_event(trace_id, data, metadata)

    @override
    async def emit_tool_event(
        self,
        trace_id: str,
        data: ToolEventData,
        metadata: Mapping[str, JSONSerializable] | None = None,
    ) -> EmittedEvent:
        result = await self._emitter.emit_tool_event(trace_id, data, metadata)
        # These are this generation's own tool results. Background SDK emissions
        # use their own publisher and are observed only on a later history load.
        await self._run.observe_inputs(frozenset({tool_input(trace_id, data)}))
        return result

    @override
    async def emit_custom_event(
        self,
        trace_id: str,
        data: JSONSerializable,
        metadata: Mapping[str, JSONSerializable] | None = None,
        *,
        source: EventSource = EventSource.AI_AGENT,
    ) -> EmittedEvent:
        return await self._emitter.emit_custom_event(trace_id, data, metadata, source=source)


class LocalResponseCoordinator:
    def __init__(
        self,
        *,
        background_tasks: BackgroundTaskService,
        logger: Logger,
        read_inputs: Callable[[SessionId], Awaitable[Inputs]],
        render: Callable[[ResponseRun], Coroutine[Any, Any, bool]],
    ) -> None:
        self._background = background_tasks
        self._logger = logger
        self._read_inputs = read_inputs
        self._render = render
        self._states: dict[SessionId, _SessionState] = {}

    @property
    def session_count(self) -> int:
        return len(self._states)

    def task_tag(self, session_id: SessionId) -> str:
        return f"queued-response({session_id})"

    def _state(self, session_id: SessionId) -> _SessionState:
        if session_id not in self._states:
            self._states[session_id] = _SessionState(session_id)
        return self._states[session_id]

    def _discard_idle(self, state: _SessionState) -> None:
        if (
            not state.worker_installed
            and not state.users
            and not state.foreground
            and not state.requests
            and not state.jobs
        ):
            if self._states.get(state.session_id) is state:
                del self._states[state.session_id]
            state.idle.set()

    @asynccontextmanager
    async def _access(self, session_id: SessionId) -> AsyncIterator[_SessionState]:
        state = self._state(session_id)
        # Pin before awaiting the publication lock: an idle predecessor must
        # not remove a state that already has an incoming caller waiting on it.
        state.users += 1
        try:
            async with state.lock:
                if state.closed:
                    raise RuntimeError("The queued response is shutting down")
                yield state
        finally:
            state.users -= 1
            self._discard_idle(state)

    async def begin_foreground(self, session_id: SessionId) -> ResponseRun:
        async with self._access(session_id) as state:
            state.generation += 1
            run = ResponseRun(state, "foreground", self._read_inputs)
            state.foreground = run
            if state.rendering is not None:
                state.rendering.cancel("A new customer turn superseded the deferred reply")
            state.changed.set()
            return run

    async def finish_foreground(self, run: ResponseRun, *, succeeded: bool) -> None:
        state = run._state
        async with state.lock:
            if state.foreground is run:
                if succeeded and run.visible:
                    self._consume(state, run)
                state.foreground = None
                state.changed.set()
                self._discard_idle(state)

    def _consume(self, state: _SessionState, run: ResponseRun) -> None:
        if run.snapshot_loaded:
            state.answered_inputs = run.inputs
            state.requests = {
                seq: request
                for seq, request in state.requests.items()
                if not request.inputs <= run.inputs
            }

    async def request(
        self, session_id: SessionId, *, after: Coroutine[Any, Any, None] | None = None
    ) -> None:
        if after is not None:
            await self._register_work(session_id, after)
            return
        await self._request_ready(session_id)

    async def _request_ready(self, session_id: SessionId, *, registered_work: bool = False) -> None:
        inputs = await self._read_inputs(session_id)
        async with self._access(session_id) as state:
            if (
                registered_work
                and state.answered_inputs is not None
                and inputs <= state.answered_inputs
            ):
                # A foreground reply may have included these results while the
                # application coroutine was finishing other bookkeeping.
                return
            state.sequence += 1
            state.requests[state.sequence] = _Request(inputs, copy_context())
            if (
                state.queued is not None
                and not state.queued.visible
                and state.queued.snapshot_loaded
                and not inputs <= state.queued.inputs
                and state.rendering is not asyncio.current_task()
                and state.rendering is not None
            ):
                state.rendering.cancel("New information superseded the unpublished reply")
            state.changed.set()
        await self._ensure_worker(state)

    async def _register_work(self, session_id: SessionId, work: Coroutine[Any, Any, None]) -> None:
        try:
            async with self._access(session_id) as state:
                state.sequence += 1
                registration = state.sequence
                state.jobs[registration] = None
                if (
                    state.queued is not None
                    and not state.queued.visible
                    and state.rendering is not None
                ):
                    state.rendering.cancel("More background work was registered")
                state.changed.set()
        except BaseException:
            work.close()
            raise
        try:
            await self._ensure_worker(state)
            async with state.lock:
                if state.closed:
                    raise RuntimeError("The queued response is shutting down")
                task = asyncio.create_task(
                    self._work(state, registration, work),
                    name=f"queued-response-work:{session_id}:{registration}",
                )
                # A task canceled before its first step never enters _work's
                # try/finally; close the unstarted application coroutine too.
                task.add_done_callback(lambda _: work.close())
                state.jobs[registration] = task
        except BaseException:
            work.close()
            async with state.lock:
                state.jobs.pop(registration, None)
                state.changed.set()
                self._discard_idle(state)
            raise

    async def _work(
        self, state: _SessionState, registration: int, work: Coroutine[Any, Any, None]
    ) -> None:
        try:
            await work
            await self._request_ready(state.session_id, registered_work=True)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # Application work owns its failure result. Do not manufacture an
            # answer or retry consequential work. Other completed jobs proceed.
            self._logger.error(f"Queued response work failed: {type(error).__name__}")
        finally:
            async with state.lock:
                state.jobs.pop(registration, None)
                state.changed.set()

    async def _ensure_worker(self, state: _SessionState) -> None:
        async with state.lock:
            if not state.worker_installed:
                state.worker_installed = True
                state.installation = asyncio.create_task(self._install_worker(state))
            installation = state.installation
            assert installation is not None

        # Every registering caller waits for the same owner to be installed;
        # none may launch application work under a canceled pre-start worker.
        await asyncio.shield(installation)

    async def _install_worker(self, state: _SessionState) -> None:
        worker = self._worker(state)
        owner: asyncio.Task[None] | None = None
        try:
            owner = await self._background.start(worker, tag=self.task_tag(state.session_id))
            if owner.cancelled() or owner.cancelling():
                raise asyncio.CancelledError(
                    "Queued response owner was canceled during installation"
                )
        except BaseException:
            if owner is None:
                worker.close()
            else:
                await asyncio.gather(owner, return_exceptions=True)
            async with state.lock:
                state.requests.clear()
                state.worker_installed = False
                self._discard_idle(state)
            raise

    async def _worker(self, state: _SessionState) -> None:
        exhausted = False
        try:
            while True:
                async with state.lock:
                    if not state.requests and not state.jobs:
                        state.worker_installed = False
                        exhausted = True
                        self._discard_idle(state)
                        return
                    if state.foreground is not None or state.jobs:
                        state.changed.clear()
                        selected = None
                    else:
                        current = await self._read_inputs(state.session_id)
                        _retain_live_inputs(state, current)
                        if not state.requests:
                            continue
                        selected = max(state.requests)
                        attempt = frozenset(state.requests)
                        context = state.requests[selected].context
                        run = ResponseRun(state, "queued", self._read_inputs)
                        state.queued = run
                        state.rendering = asyncio.create_task(self._render(run), context=context)
                        rendering = state.rendering
                if selected is None:
                    await state.changed.wait()
                    continue
                succeeded: bool | None = None
                try:
                    succeeded = await rendering
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        raise
                except Exception as error:
                    self._logger.error(f"Queued response generation failed: {type(error).__name__}")
                    succeeded = False
                async with state.lock:
                    if succeeded and not run.snapshot_loaded:
                        # Custom engines must report their loaded history to
                        # coalesce safely. Do not repeatedly generate a reply
                        # when an integration omits that coverage boundary.
                        self._logger.error("Queued renderer did not report its input snapshot")
                        succeeded = False
                    if succeeded is True:
                        self._consume(state, run)
                    elif succeeded is False:
                        # Failed attempts do not spin/retry forever. Their data
                        # stays in history; a new request can retry explicitly.
                        for seq in attempt:
                            state.requests.pop(seq, None)
                    state.queued = None
                    state.rendering = None
        finally:
            # No awaits between retiring an exhausted worker and its task's
            # return: a new caller can safely install the same service tag.
            if not exhausted:
                async with state.lock:
                    state.closed = True
                    tasks: list[asyncio.Task[None] | asyncio.Task[bool]] = [
                        task for task in state.jobs.values() if task is not None
                    ]
                    if state.rendering is not None:
                        tasks.append(state.rendering)
                    for task in tasks:
                        task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                async with state.lock:
                    state.jobs.clear()
                    state.requests.clear()
                    state.queued = None
                    state.rendering = None
                    state.worker_installed = False
                    # A foreground run may still be completing after consuming all
                    # requests. Its finish callback owns the remaining state.
                    state.closed = False
                    self._discard_idle(state)

    async def wait_idle(self, session_id: SessionId) -> None:
        while (state := self._states.get(session_id)) is not None:
            await state.idle.wait()
