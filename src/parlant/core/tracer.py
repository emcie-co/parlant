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

from abc import ABC, abstractmethod
from contextlib import contextmanager
import contextvars
from typing import Iterator, Mapping, Union, Sequence
from typing_extensions import deprecated, override

from parlant.core.common import generate_id

_UNINITIALIZED = 0xC0FFEE

AttributeValue = Union[
    str,
    bool,
    int,
    float,
    Sequence[str],
    Sequence[bool],
    Sequence[int],
    Sequence[float],
]


class Tracer(ABC):
    @contextmanager
    @abstractmethod
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Iterator[None]: ...

    @contextmanager
    @abstractmethod
    def attributes(
        self,
        attributes: Mapping[str, AttributeValue],
    ) -> Iterator[None]: ...

    @property
    @abstractmethod
    def trace_id(self) -> str: ...

    @property
    @deprecated("Use trace_id instead")
    def correlation_id(self) -> str:
        return self.trace_id

    @property
    @abstractmethod
    def span_id(self) -> str: ...

    @abstractmethod
    def get_attribute(self, name: str) -> AttributeValue | None: ...

    @abstractmethod
    def set_attribute(self, name: str, value: AttributeValue) -> None: ...

    @abstractmethod
    def add_event(self, name: str, attributes: Mapping[str, AttributeValue] = {}) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...


class LocalTracer(Tracer):
    def __init__(self) -> None:
        self._spans = contextvars.ContextVar[str](
            "tracer_spans",
            default="",
        )

        self._attributes = contextvars.ContextVar[Mapping[str, AttributeValue]](
            "tracer_attributes",
            default={},
        )

        self._trace_id = contextvars.ContextVar[str](
            "tracer_trace_id",
            default="",
        )

    @contextmanager
    @override
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Iterator[None]:
        current_spans = self._spans.get()

        if not current_spans:
            new_trace_id = generate_id({"strategy": "uuid4"})
            new_spans = span_id
            trace_id_reset_token = self._trace_id.set(new_trace_id)
        else:
            new_spans = current_spans + f"::{span_id}"
            trace_id_reset_token = None

        current_attributes = self._attributes.get()
        new_attributes = {**current_attributes, **attributes}

        spans_reset_token = self._spans.set(new_spans)
        attributes_reset_token = self._attributes.set(new_attributes)

        yield

        self._spans.reset(spans_reset_token)
        self._attributes.reset(attributes_reset_token)
        if trace_id_reset_token is not None:
            self._trace_id.reset(trace_id_reset_token)

    @contextmanager
    @override
    def attributes(
        self,
        attributes: Mapping[str, AttributeValue],
    ) -> Iterator[None]:
        current_attributes = self._attributes.get()
        new_attributes = {**current_attributes, **attributes}

        attributes_reset_token = self._attributes.set(new_attributes)

        yield

        self._attributes.reset(attributes_reset_token)

    @property
    @override
    def trace_id(self) -> str:
        if trace_id := self._trace_id.get():
            return trace_id

        return "<main>"

    @property
    @override
    def span_id(self) -> str:
        if spans := self._spans.get():
            return spans

        return "<main>"

    @override
    def get_attribute(
        self,
        name: str,
    ) -> AttributeValue | None:
        attributes = self._attributes.get()
        return attributes.get(name, None)

    @override
    def set_attribute(
        self,
        name: str,
        value: AttributeValue,
    ) -> None:
        current_attributes = self._attributes.get()
        new_attributes = {**current_attributes, name: value}
        self._attributes.set(new_attributes)

    @override
    def add_event(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> None:
        pass

    @override
    def flush(self) -> None:
        pass


@deprecated("Please use the Tracer class instead of ContextualCorrelator")
class ContextualCorrelator(Tracer):
    pass
