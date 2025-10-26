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


class NullHistogram(DurationHistogram):
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
    def create_custom_histogram(
        self,
        name: str,
        description: str,
        unit: str,
    ) -> DurationHistogram:
        return NullHistogram()

    @override
    def create_duration_histogram(
        self,
        name: str,
        description: str,
    ) -> DurationHistogram:
        return NullHistogram()
