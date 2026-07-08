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
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Coroutine, Optional, TypeAlias
from typing_extensions import Self

from parlant.core.loggers import Logger


Task: TypeAlias = asyncio.Task[None]


class BackgroundTaskService:
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

        self._last_garbage_collection = 0.0
        self._garbage_collection_interval = 5.0
        self._tasks = dict[str, Task]()

        # Guards the _tasks and _tag_locks dicts. CRITICAL INVARIANT: this lock is
        # only ever held for synchronous dict access — never across `await <a
        # task>`. Awaiting a slow (or cancellation-shielded) task while holding it
        # would serialize every tag's start/restart/collect behind that one task,
        # freezing all session dispatch behind a single stuck turn.
        self._lock = asyncio.Lock()

        # Per-tag guards serialize the cancel->unwind->install swap for a single
        # tag (so a tag never ends up with two live tasks), WITHOUT holding the
        # global lock across the displaced task's unwind. Refcounted so the map is
        # bounded to tags with an in-flight start/restart.
        self._tag_locks = dict[str, tuple[asyncio.Lock, int]]()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        if exc_value:
            await self.cancel_all(reason="Shutting down")

        self._logger.info(f"{type(self).__name__}: Shutting down")

        await self.collect(force=True)

        return False

    @asynccontextmanager
    async def _tag_guard(self, tag: str) -> AsyncIterator[None]:
        """Serialize start/restart for a single ``tag`` without holding the global
        lock across the displaced task's unwind. Refcounted so the per-tag lock is
        dropped once no start/restart is using it (the count is incremented under
        the global lock *before* acquiring, so cleanup can't drop a lock that has
        pending users)."""
        async with self._lock:
            entry = self._tag_locks.get(tag)
            if entry is None:
                lock = asyncio.Lock()
                self._tag_locks[tag] = (lock, 1)
            else:
                lock, count = entry
                self._tag_locks[tag] = (lock, count + 1)

        try:
            async with lock:
                yield
        finally:
            async with self._lock:
                lock, count = self._tag_locks[tag]
                if count <= 1:
                    del self._tag_locks[tag]
                else:
                    self._tag_locks[tag] = (lock, count - 1)

    async def cancel(self, *, tag: str, reason: str = "(not given)") -> bool:
        async with self._lock:
            task = self._tasks.get(tag)

        if task is not None and not task.done():
            task.cancel(f"Forced cancellation by {type(self).__name__} [reason: {reason}]")
            cancelled = True
        else:
            cancelled = False

        await self.collect()

        return cancelled

    async def cancel_all(self, *, reason: str = "(not given)") -> None:
        async with self._lock:
            self._logger.info(
                f"{type(self).__name__}: Cancelling all remaining tasks ({len(self._tasks)})"
            )
            tasks = list(self._tasks.values())

        for task in tasks:
            if not task.done():
                task.cancel(f"Forced cancellation by {type(self).__name__} [reason: {reason}]")

        await self.collect()

    async def start(self, f: Coroutine[Any, Any, None], /, *, tag: str) -> Task:
        await self.collect()

        async with self._tag_guard(tag):
            async with self._lock:
                existing_task = self._tasks.get(tag)

            if existing_task is not None and not existing_task.done():
                raise Exception(
                    f"Task '{tag}' is already running; consider calling restart() instead"
                )

            self._logger.trace(f"{type(self).__name__}: Starting task '{tag}'")
            task = asyncio.create_task(f)

            async with self._lock:
                self._tasks[tag] = task

            return task

    async def restart(self, f: Coroutine[Any, Any, None], /, *, tag: str) -> Task:
        await self.collect()

        async with self._tag_guard(tag):
            async with self._lock:
                existing_task = self._tasks.get(tag)

            # Cancel + await the previous task's unwind under THIS tag's guard but
            # NOT the global lock, so other tags keep dispatching while a slow or
            # cancellation-shielded turn winds down. The tag guard still preserves
            # the swap's atomicity, so a tag never has two live tasks.
            if existing_task is not None and not existing_task.done():
                existing_task.cancel(f"Restarting task '{tag}'")
                await self._await_task(existing_task)

            self._logger.trace(f"{type(self).__name__}: Starting task '{tag}'")
            task = asyncio.create_task(f)

            async with self._lock:
                self._tasks[tag] = task

            return task

    async def collect(self, *, force: bool = False) -> None:
        now = asyncio.get_event_loop().time()

        if not force:
            if (now - self._last_garbage_collection) < self._garbage_collection_interval:
                return

        # Snapshot which tasks to reap under the lock, then await their unwind
        # OUTSIDE the lock so reaping (especially a forced shutdown wait on a stuck
        # task) never blocks other tags' start/restart/collect.
        async with self._lock:
            reaped = [(tag, task) for tag, task in self._tasks.items() if force or task.done()]
            self._tasks = {
                tag: task for tag, task in self._tasks.items() if not (force or task.done())
            }

        for tag, task in reaped:
            if not task.done():
                self._logger.info(f"{type(self).__name__}: Waiting for task '{tag}' to finish")

            await self._await_task(task)

        self._last_garbage_collection = now

    async def _await_task(self, task: Task) -> None:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._logger.warning(
                f"{type(self).__name__}: Awaited task raised an exception: {traceback.format_exception(exc)}"
            )
