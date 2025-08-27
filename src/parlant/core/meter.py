from abc import ABC, abstractmethod
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Mapping

from parlant.core.common import AttributeValue


class Meter(ABC):
    @abstractmethod
    async def record_counter(
        self,
        name: str,
        value: int = 1,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None: ...

    @abstractmethod
    async def record_histogram(
        self,
        name: str,
        value: float,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None: ...

    @asynccontextmanager
    async def measure_duration(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> AsyncGenerator[None, None]:
        """
        Measure the duration of a block of code.
        Usage:
            async with meter.measure_duration("my_duration"):
                # Code to measure
        """
        start_time = asyncio.get_event_loop().time()
        try:
            yield
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            await self.record_histogram(name, duration, attributes)


class NullMeter(Meter):
    async def record_counter(
        self,
        name: str,
        value: int = 1,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        pass

    async def record_histogram(
        self,
        name: str,
        value: float,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        pass
