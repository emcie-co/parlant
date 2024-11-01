from __future__ import annotations
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
