from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Mapping
from typing_extensions import override


class Histogram(ABC):
    @abstractmethod
    async def record(
        self,
        value: float,
        attributes: Mapping[str, str] | None = None,
    ) -> None: ...

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
    def create_histogram(
        self,
        name: str,
        description: str,
        unit: str = "ms",
    ) -> Histogram: ...


class NullCounter(Counter):
    @override
    async def increment(
        self,
        value: int,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        pass


class NullHistogram(Histogram):
    @override
    async def record(
        self,
        value: float,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        pass

    @override
    @asynccontextmanager
    async def measure(
        self,
        attributes: Mapping[str, str] | None = None,
    ) -> AsyncGenerator[None, None]:
        yield


class NullMeter(Meter):
    @override
    def create_counter(
        self,
        name: str,
        description: str,
    ) -> Counter:
        return NullCounter()

    @override
    def create_histogram(
        self,
        name: str,
        description: str,
        unit: str = "ms",
    ) -> Histogram:
        return NullHistogram()
