from __future__ import annotations
from typing import Any, Coroutine, Iterable, TypeVar, overload
import asyncio
import math


class Timeout:
    @staticmethod
    def none() -> Timeout:
        return Timeout(0)

    @staticmethod
    def infinite() -> Timeout:
        return Timeout(math.inf)

    def __init__(self, seconds: float) -> None:
        self._creation = self._now()
        self._expiration = self._creation + seconds

    def expired(self) -> bool:
        return self.remaining() == 0

    def remaining(self) -> float:
        return max(0, self._expiration - self._now())

    def afford_up_to(self, seconds: float) -> Timeout:
        return Timeout(min(self.remaining(), seconds))

    async def wait(self) -> None:
        await asyncio.sleep(self.remaining())

    async def wait_up_to(self, seconds: float) -> None:
        await asyncio.sleep(self.afford_up_to(seconds).remaining())

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
    | Coroutine[Any, Any, _TResult0],
) -> tuple[_TResult0]: ...


@overload
async def safe_gather(
    coros_or_future_0: asyncio.Future[_TResult0]
    | asyncio.Task[_TResult0]
    | Coroutine[Any, Any, _TResult0],
    coros_or_future_1: asyncio.Future[_TResult1]
    | asyncio.Task[_TResult1]
    | Coroutine[Any, Any, _TResult1],
) -> tuple[_TResult0, _TResult1]: ...


@overload
async def safe_gather(
    coros_or_future_0: asyncio.Future[_TResult0]
    | asyncio.Task[_TResult0]
    | Coroutine[Any, Any, _TResult0],
    coros_or_future_1: asyncio.Future[_TResult1]
    | asyncio.Task[_TResult1]
    | Coroutine[Any, Any, _TResult1],
    coros_or_future_2: asyncio.Future[_TResult2]
    | asyncio.Task[_TResult2]
    | Coroutine[Any, Any, _TResult2],
) -> tuple[_TResult0, _TResult2]: ...


@overload
async def safe_gather(
    coros_or_future_0: asyncio.Future[_TResult0]
    | asyncio.Task[_TResult0]
    | Coroutine[Any, Any, _TResult0],
    coros_or_future_1: asyncio.Future[_TResult1]
    | asyncio.Task[_TResult1]
    | Coroutine[Any, Any, _TResult1],
    coros_or_future_2: asyncio.Future[_TResult2]
    | asyncio.Task[_TResult2]
    | Coroutine[Any, Any, _TResult2],
    coros_or_future_3: asyncio.Future[_TResult3]
    | asyncio.Task[_TResult3]
    | Coroutine[Any, Any, _TResult3],
) -> tuple[_TResult0, _TResult3]: ...


async def safe_gather(  # type: ignore[misc]
    *coros_or_futures: asyncio.Future[_TResult0]
    | asyncio.Task[_TResult0]
    | Coroutine[Any, Any, _TResult0],
) -> Iterable[_TResult0]:
    coros_or_futures_list = list(coros_or_futures)

    try:
        return await asyncio.gather(
            *coros_or_futures_list,
            return_exceptions=False,
        )
    except asyncio.CancelledError:
        for coro_or_future in coros_or_futures_list:
            if asyncio.isfuture(coro_or_future):
                coro_or_future.cancel()
        raise
