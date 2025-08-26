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

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    TypeVar,
    overload,
    AsyncContextManager,
)
import asyncio
import math
import aiorwlock

from parlant.core.loggers import Logger


class Timeout:
    @staticmethod
    def none() -> Timeout:
        return Timeout(0)

    @staticmethod
    def infinite() -> Timeout:
        return Timeout(math.inf)

    def __init__(self, seconds: float) -> None:
        # We want to avoid calling _now() on a static level, because
        # it requires running within an event loop.
        self._creation = self._now() if seconds not in [0, math.inf] else 0
        self._expiration = self._creation + seconds

    def expired(self) -> bool:
        return self.remaining() == 0

    def remaining(self) -> float:
        return max(0, self._expiration - self._now())

    def afford_up_to(self, seconds: float) -> Timeout:
        return Timeout(min(self.remaining(), seconds))

    async def wait(self) -> None:
        await asyncio.sleep(self.remaining())

    async def wait_up_to(self, seconds: float) -> bool:
        await asyncio.sleep(self.afford_up_to(seconds).remaining())
        return self.expired()

    def __bool__(self) -> bool:
        return not self.expired()

    def _now(self) -> float:
        return asyncio.get_event_loop().time()


_TResult0 = TypeVar("_TResult0")
_TResult1 = TypeVar("_TResult1")
_TResult2 = TypeVar("_TResult2")
_TResult3 = TypeVar("_TResult3")


@overload
async def safe_gather(
    coros_or_future_0: asyncio.Future[_TResult0]
    | asyncio.Task[_TResult0]
    | Coroutine[Any, Any, _TResult0]
    | Awaitable[_TResult0],
) -> tuple[_TResult0]: ...


@overload
async def safe_gather(
    coros_or_future_0: asyncio.Future[_TResult0]
    | asyncio.Task[_TResult0]
    | Coroutine[Any, Any, _TResult0]
    | Awaitable[_TResult0],
    coros_or_future_1: asyncio.Future[_TResult1]
    | asyncio.Task[_TResult1]
    | Coroutine[Any, Any, _TResult1]
    | Awaitable[_TResult1],
) -> tuple[_TResult0, _TResult1]: ...


@overload
async def safe_gather(
    coros_or_future_0: asyncio.Future[_TResult0]
    | asyncio.Task[_TResult0]
    | Coroutine[Any, Any, _TResult0]
    | Awaitable[_TResult0],
    coros_or_future_1: asyncio.Future[_TResult1]
    | asyncio.Task[_TResult1]
    | Coroutine[Any, Any, _TResult1]
    | Awaitable[_TResult1],
    coros_or_future_2: asyncio.Future[_TResult2]
    | asyncio.Task[_TResult2]
    | Coroutine[Any, Any, _TResult2]
    | Awaitable[_TResult2],
) -> tuple[_TResult0, _TResult2]: ...


@overload
async def safe_gather(
    coros_or_future_0: asyncio.Future[_TResult0]
    | asyncio.Task[_TResult0]
    | Coroutine[Any, Any, _TResult0]
    | Awaitable[_TResult0],
    coros_or_future_1: asyncio.Future[_TResult1]
    | asyncio.Task[_TResult1]
    | Coroutine[Any, Any, _TResult1]
    | Awaitable[_TResult1],
    coros_or_future_2: asyncio.Future[_TResult2]
    | asyncio.Task[_TResult2]
    | Coroutine[Any, Any, _TResult2]
    | Awaitable[_TResult2],
    coros_or_future_3: asyncio.Future[_TResult3]
    | asyncio.Task[_TResult3]
    | Coroutine[Any, Any, _TResult3]
    | Awaitable[_TResult3],
) -> tuple[_TResult0, _TResult3]: ...


async def safe_gather(  # type: ignore[misc]
    *coros_or_futures: asyncio.Future[_TResult0]
    | asyncio.Task[_TResult0]
    | Coroutine[Any, Any, _TResult0]
    | Awaitable[_TResult0],
) -> Iterable[_TResult0]:
    futures = [asyncio.ensure_future(x) for x in coros_or_futures]

    try:
        return await asyncio.gather(
            *futures,
            return_exceptions=False,
        )
    except asyncio.CancelledError:
        for future in futures:
            future.add_done_callback(default_done_callback())
            future.cancel()

        raise


async def with_timeout(
    coro_or_future: asyncio.Future[_TResult0]
    | asyncio.Task[_TResult0]
    | Coroutine[Any, Any, _TResult0],
    timeout: Timeout,
) -> _TResult0:
    fut = asyncio.ensure_future(coro_or_future)

    try:
        return await asyncio.wait_for(coro_or_future, timeout.remaining())
    except asyncio.TimeoutError:
        fut.add_done_callback(default_done_callback())
        fut.cancel()
        raise


@overload
def completed_task() -> asyncio.Task[None]:
    """
    Returns a completed asyncio Task with no value.
    """
    ...


@overload
def completed_task(value: _TResult0) -> asyncio.Task[_TResult0]:
    """
    Returns a completed asyncio Task with the given value.
    """
    ...


def completed_task(value: _TResult0 | None = None) -> asyncio.Task[_TResult0 | None]:
    async def return_value() -> _TResult0 | None:
        return value

    return asyncio.create_task(return_value())


def default_done_callback(
    logger: Logger | None = None,
) -> Callable[[asyncio.Task[_TResult0]], object]:
    def done_callback(task: asyncio.Task[_TResult0]) -> object:
        try:
            return task.result()
        except asyncio.CancelledError:
            return None
        except Exception as e:
            if logger:
                logger.error(f"Exception encountered in background task {task.get_name()}: {e}")
            return None

    return done_callback


class ReaderWriterLock:
    def __init__(self) -> None:
        _lock = aiorwlock.RWLock()
        self._reader_lock = _lock.reader
        self._writer_lock = _lock.writer

    @property
    def reader_lock(self) -> AsyncContextManager[None]:
        @asynccontextmanager
        async def _reader_cm() -> AsyncIterator[None]:
            async with self._reader_lock:
                yield

        return _reader_cm()

    @property
    def writer_lock(self) -> AsyncContextManager[None]:
        @asynccontextmanager
        async def _writer_cm() -> AsyncIterator[None]:
            async with self._writer_lock:
                yield

        return _writer_cm()
