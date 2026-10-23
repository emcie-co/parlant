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
import traceback
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Generic,
    Iterable,
    TypeVar,
    overload,
    AsyncContextManager,
)
import asyncio
import time
import math
import aiorwlock

from parlant.core.loggers import Logger


def _now() -> float:
    return time.monotonic()


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
        self._creation = _now() if seconds not in [0, math.inf] else 0
        self._expiration = self._creation + seconds

    def expired(self) -> bool:
        return self.remaining() == 0

    def remaining(self) -> float:
        return max(0, self._expiration - _now())

    def afford_up_to(self, seconds: float) -> Timeout:
        return Timeout(min(self.remaining(), seconds))

    async def wait(self) -> None:
        await asyncio.sleep(self.remaining())

    async def wait_up_to(self, seconds: float) -> bool:
        await asyncio.sleep(self.afford_up_to(seconds).remaining())
        return self.expired()

    def __bool__(self) -> bool:
        return not self.expired()


class Stopwatch:
    @staticmethod
    def start() -> Stopwatch:
        return Stopwatch(_now())

    def __init__(self, start_time: float) -> None:
        self._start = start_time

    @property
    def elapsed(self) -> float:
        return _now() - self._start

    @property
    def start_time(self) -> float:
        return self._start


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
) -> tuple[_TResult0, _TResult1, _TResult2]: ...


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
) -> tuple[_TResult0, _TResult1, _TResult2, _TResult3]: ...


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


_TASK_TRACEBACK_NOTE_PREFIX = (
    "Full task traceback (frames above truncate at the async task boundary):"
)


def _preserve_task_traceback(exc: BaseException) -> None:
    """Attach the exception's current (still-complete) traceback as a note so it survives the
    truncation that a task done-callback's result retrieval causes. Idempotent across nested
    latches."""
    existing = getattr(exc, "__notes__", None) or []
    if any(note.startswith(_TASK_TRACEBACK_NOTE_PREFIX) for note in existing):
        return
    exc.add_note(_TASK_TRACEBACK_NOTE_PREFIX + "\n" + "".join(traceback.format_exception(exc)))


def default_done_callback(
    logger: Logger | None = None,
) -> Callable[[asyncio.Future[_TResult0]], object]:
    def done_callback(fut: asyncio.Future[_TResult0]) -> object:
        try:
            return fut.result()
        except asyncio.CancelledError:
            return None
        except Exception as e:
            if logger:
                logger.error(f"Exception encountered in background task: {e}")
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
        async def _reader_cm() -> AsyncGenerator[None, None]:
            async with self._reader_lock:
                yield

        return _reader_cm()

    @property
    def writer_lock(self) -> AsyncContextManager[None]:
        @asynccontextmanager
        async def _writer_cm() -> AsyncGenerator[None, None]:
            async with self._writer_lock:
                yield

        return _writer_cm()


class CancellationSuppressionLatch(Generic[_TResult0]):
    def __init__(
        self, func: Callable[[CancellationSuppressionLatch[_TResult0]], Awaitable[_TResult0]]
    ) -> None:
        self._func: Callable[[CancellationSuppressionLatch[_TResult0]], Awaitable[_TResult0]] = func
        self._unshielded_task: asyncio.Future[None]
        self._shielded_task: asyncio.Future[None] | None = None
        self._cancellation_error: asyncio.CancelledError | None = None
        self._exception: BaseException | None = None
        self._enabled = False
        self._done = asyncio.Event()
        self._result: _TResult0

    async def __aenter__(self) -> CancellationSuppressionLatch[_TResult0]:
        async def unshielded_shim() -> None:
            try:
                self._result = await self._func(self)
            except BaseException as exc:
                # The done-callback below retrieves the task result first, which truncates
                # this exception's __traceback__ down to the callback frame — erasing the real
                # raise site by the time it surfaces. Preserve the full in-task traceback as an
                # exception note (included verbatim by traceback formatting / our reports).
                _preserve_task_traceback(exc)
                raise

        self._unshielded_task = asyncio.create_task(unshielded_shim())
        self._unshielded_task.add_done_callback(default_done_callback())

        async def task_shim() -> None:
            try:
                await self._unshielded_task
            except (Exception, asyncio.CancelledError) as exc:
                self._exception = exc
            finally:
                self._done.set()

        self._shielded_task = asyncio.shield(task_shim())
        self._shielded_task.add_done_callback(default_done_callback())

        try:
            await self._shielded_task
        except asyncio.CancelledError as exc:
            self._cancellation_error = exc

            if not self._enabled:
                self._unshielded_task.cancel()

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException],
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> None:
        assert self._shielded_task is not None

        if self._cancellation_error and not self._enabled:
            await self._shielded_task
            raise asyncio.CancelledError() from self._cancellation_error

    @property
    def cancellation_pending(self) -> bool:
        """True once the shielded caller was cancelled while suppression was
        enabled — the protected code can poll this to abandon voluntarily
        before committing anything user-visible."""
        return self._cancellation_error is not None

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def _raise_exception_if_any(self) -> None:
        if self._exception is None:
            return

        if isinstance(self._exception, asyncio.CancelledError):
            # The protected code abandoned itself (e.g. on a pending
            # supersede-cancellation): surface it as a genuine cancellation.
            raise self._exception

        if isinstance(self._exception, Exception):
            raise Exception("Task failed") from self._exception
        elif isinstance(self._exception, BaseException):
            raise BaseException("Task failed") from self._exception
        else:
            raise self._exception

    def _raise_deferred_cancellation_if_unsuppressed(self) -> None:
        if self._cancellation_error and not self._enabled:
            raise asyncio.CancelledError() from self._cancellation_error

    async def _get_result(self) -> _TResult0:
        await self._done.wait()
        self._raise_exception_if_any()
        self._raise_deferred_cancellation_if_unsuppressed()
        return self._result


async def latched_shield(
    func: Callable[[CancellationSuppressionLatch[_TResult0]], Awaitable[_TResult0]],
) -> _TResult0:
    async with CancellationSuppressionLatch(func) as latch:
        pass

    return await latch._get_result()


async def delay(
    delay: float,
    coro: asyncio.Future[_TResult0] | asyncio.Task[_TResult0] | Coroutine[Any, Any, _TResult0],
) -> _TResult0:
    await asyncio.sleep(delay)
    return await coro
