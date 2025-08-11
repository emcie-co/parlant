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

from abc import ABC, abstractmethod
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Mapping, Sequence, TypeAlias, Union
from typing_extensions import override


AttributeValue: TypeAlias = Union[
    str,
    bool,
    int,
    float,
    Sequence[str],
    Sequence[bool],
    Sequence[int],
    Sequence[float],
]


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


class Span(ABC):
    @abstractmethod
    async def set_attribute(
        self,
        key: str,
        value: AttributeValue,
    ) -> None: ...

    @abstractmethod
    async def add_event(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None: ...

    @abstractmethod
    async def end(
        self,
        error: Exception | None = None,
    ) -> None: ...


class Tracer(ABC):
    @abstractmethod
    async def start_span(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> Span: ...

    @abstractmethod
    async def end_span(
        self,
        error: Exception | None = None,
    ) -> None: ...

    @property
    @abstractmethod
    def meter(self) -> Meter: ...

    @asynccontextmanager
    async def span(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> AsyncGenerator[Span, None]:
        try:
            yield await self.start_span(name, attributes)
        except Exception as e:
            await self.end_span(error=e)
            raise
        else:
            await self.end_span()


class NoOpSpan(Span):
    @override
    async def set_attribute(
        self,
        key: str,
        value: AttributeValue,
    ) -> None:
        pass

    @override
    async def add_event(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        pass

    @override
    async def end(
        self,
        error: Exception | None = None,
    ) -> None:
        pass


class NoOpMeter(Meter):
    @override
    async def record_counter(
        self,
        name: str,
        value: int = 1,  # keep default to match base signature
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        pass

    @override
    async def record_histogram(
        self,
        name: str,
        value: float,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        pass


class NoOpTracer(Tracer):
    @override
    async def start_span(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> Span:
        return NoOpSpan()

    @override
    async def end_span(
        self,
        error: Exception | None = None,
    ) -> None:
        pass
