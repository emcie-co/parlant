from abc import ABC, abstractmethod
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Mapping
from typing_extensions import override

from parlant.core.loggers import Logger


class Histogram(ABC):
    @abstractmethod
    async def record(
        self,
        value: float,
        attributes: Mapping[str, str] | None = None,
    ) -> None: ...


class DurationHistogram(Histogram):
    """
    A histogram that records durations in milliseconds.
    """

    @abstractmethod
    @asynccontextmanager
    async def measure(
        self,
        attributes: Mapping[str, str] | None = None,
    ) -> AsyncGenerator[None, None]:
        yield


class Counter(ABC):
    @abstractmethod
    async def increment(
        self,
        value: int,
        attributes: Mapping[str, str] | None = None,
    ) -> None: ...


class Meter(ABC):
    @abstractmethod
    def create_counter(
        self,
        name: str,
        description: str,
    ) -> Counter: ...

    @abstractmethod
    def create_custom_histogram(
        self,
        name: str,
        description: str,
        unit: str,
    ) -> Histogram: ...

    @abstractmethod
    def create_duration_histogram(
        self,
        name: str,
        description: str,
    ) -> DurationHistogram: ...


class NullCounter(Counter):
    @override
    async def increment(
        self,
        value: int,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        pass


class LocalHistogram(DurationHistogram):
    def __init__(self, name: str, logger: Logger) -> None:
        self._name = name
        self._logger = logger

    @override
    async def record(
        self,
        value: float,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        attrs = f" attributes={attributes}" if attributes else ""
        self._logger.trace(f"Histogram '{self._name}' recorded duration={value:.6f}{attrs}")

    @override
    @asynccontextmanager
    async def measure(
        self,
        attributes: Mapping[str, str] | None = None,
    ) -> AsyncGenerator[None, None]:
        start_time = asyncio.get_running_loop().time()
        try:
            yield
        finally:
            duration = asyncio.get_running_loop().time() - start_time
            await self.record(duration, attributes)


class LocalMeter(Meter):
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    @override
    def create_counter(
        self,
        name: str,
        description: str,
    ) -> Counter:
        return NullCounter()

    @override
    def create_custom_histogram(
        self,
        name: str,
        description: str,
        unit: str,
    ) -> DurationHistogram:
        return LocalHistogram(name, self._logger)

    @override
    def create_duration_histogram(
        self,
        name: str,
        description: str,
    ) -> DurationHistogram:
        return LocalHistogram(name, self._logger)
