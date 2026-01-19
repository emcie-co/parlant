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
from contextlib import contextmanager, ExitStack
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

    def _generate_trace_id(self) -> str:
        return str(generate_id({"strategy": "uuid4"}))


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
            new_trace_id = self._generate_trace_id()
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


class CompositeTracer(Tracer):
    """A tracer that combines multiple tracers into one."""

    def __init__(self, tracers: Sequence[Tracer]) -> None:
        self._tracers = list(tracers)
        # Context variable to track shared trace_id across all tracers
        self._shared_trace_id = contextvars.ContextVar[str](
            "composite_tracer_shared_trace_id",
            default="",
        )

    def append(self, tracer: Tracer) -> None:
        self._tracers.append(tracer)

    @contextmanager
    @override
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Iterator[None]:
        # Generate shared trace_id if this is a root span
        current_trace_id = self._shared_trace_id.get()
        if not current_trace_id:
            shared_trace_id = self._generate_trace_id()
            trace_id_reset_token = self._shared_trace_id.set(shared_trace_id)

            # Set the shared trace_id in all tracers before creating spans
            for tracer in self._tracers:
                if hasattr(tracer, "_trace_id"):
                    tracer._trace_id.set(shared_trace_id)
        else:
            trace_id_reset_token = None

        with ExitStack() as stack:
            for context in [tracer.span(span_id, attributes) for tracer in self._tracers]:
                stack.enter_context(context)
            try:
                yield
            finally:
                if trace_id_reset_token is not None:
                    self._shared_trace_id.reset(trace_id_reset_token)

    @contextmanager
    @override
    def attributes(
        self,
        attributes: Mapping[str, AttributeValue],
    ) -> Iterator[None]:
        with ExitStack() as stack:
            for context in [tracer.attributes(attributes) for tracer in self._tracers]:
                stack.enter_context(context)
            yield

    @property
    @override
    def trace_id(self) -> str:
        if shared_trace_id := self._shared_trace_id.get():
            return shared_trace_id
        if self._tracers:
            return self._tracers[0].trace_id
        return "<main>"

    @property
    @override
    def span_id(self) -> str:
        if self._tracers:
            return self._tracers[0].span_id
        return "<main>"

    @override
    def get_attribute(
        self,
        name: str,
    ) -> AttributeValue | None:
        if self._tracers:
            return self._tracers[0].get_attribute(name)
        return None

    @override
    def set_attribute(
        self,
        name: str,
        value: AttributeValue,
    ) -> None:
        for tracer in self._tracers:
            tracer.set_attribute(name, value)

    @override
    def add_event(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> None:
        for tracer in self._tracers:
            tracer.add_event(name, attributes)

    @override
    def flush(self) -> None:
        for tracer in self._tracers:
            tracer.flush()


@deprecated("Please use the Tracer class instead of ContextualCorrelator")
class ContextualCorrelator(Tracer):
    pass
