from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Mapping
from typing_extensions import override

from parlant.core.tracer import AttributeValue


class Meter(ABC):
    @abstractmethod
    async def increment(
        self,
        name: str,
        value: int = 1,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None: ...

    @abstractmethod
    async def record(
        self,
        name: str,
        value: float,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None: ...

    @abstractmethod
    @asynccontextmanager
    async def measure(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> AsyncGenerator[None, None]:
        yield


class NullMeter(Meter):
    @override
    async def increment(
        self,
        name: str,
        value: int = 1,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        pass

    @override
    async def record(
        self,
        name: str,
        value: float,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        pass

    @override
    @asynccontextmanager
    async def measure(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> AsyncGenerator[None, None]:
        yield
