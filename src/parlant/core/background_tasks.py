# Copyright 2025 Emcie Co Ltd.
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
from typing import Any, Coroutine, Optional, TypeAlias
from typing_extensions import Self

from parlant.core.loggers import Logger


Task: TypeAlias = asyncio.Task[None]


class BackgroundTaskService:
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

        self._last_garbage_collection = 0.0
        self._garbage_collection_interval = 5.0
        self._tasks = dict[str, Task]()
        self._lock = asyncio.Lock()

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

    async def cancel(self, *, tag: str, reason: str = "(not given)") -> None:
        async with self._lock:
            if task := self._tasks.get(tag):
                if not task.done():
                    task.cancel(f"Forced cancellation by {type(self).__name__} [reason: {reason}]")

        await self.collect()

    async def cancel_all(self, *, reason: str = "(not given)") -> None:
        async with self._lock:
            self._logger.info(
                f"{type(self).__name__}: Cancelling all remaining tasks ({len(self._tasks)})"
            )

            for task in self._tasks.values():
                if not task.done():
                    task.cancel(f"Forced cancellation by {type(self).__name__} [reason: {reason}]")

        await self.collect()

    async def start(self, f: Coroutine[Any, Any, None], /, *, tag: str) -> Task:
        await self.collect()

        async with self._lock:
            if existing_task := self._tasks.get(tag):
                if not existing_task.done():
                    raise Exception(
                        f"Task '{tag}' is already running; consider calling restart() instead"
                    )

            self._logger.trace(f"{type(self).__name__}: Starting task '{tag}'")
            task = asyncio.create_task(f)
            self._tasks[tag] = task
            return task

    async def restart(self, f: Coroutine[Any, Any, None], /, *, tag: str) -> Task:
        await self.collect()

        async with self._lock:
            if existing_task := self._tasks.get(tag):
                if not existing_task.done():
                    existing_task.cancel(f"Restarting task '{tag}'")
                    await self._await_task(existing_task)

            self._logger.trace(f"{type(self).__name__}: Starting task '{tag}'")
            task = asyncio.create_task(f)
            self._tasks[tag] = task
            return task

    async def collect(self, *, force: bool = False) -> None:
        now = asyncio.get_event_loop().time()

        if not force:
            if (now - self._last_garbage_collection) < self._garbage_collection_interval:
                return

        async with self._lock:
            new_tasks_dict = {}

            for tag, task in self._tasks.items():
                if task.done() or force:
                    if not task.done():
                        self._logger.info(
                            f"{type(self).__name__}: Waiting for task '{tag}' to finish"
                        )

                    await self._await_task(task)
                else:
                    # Task is still running; leave it there
                    new_tasks_dict[tag] = task

            self._tasks = new_tasks_dict

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
