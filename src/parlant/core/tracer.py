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
from collections.abc import Generator
from dataclasses import dataclass
import contextvars
from typing import Mapping, Sequence, Union, cast

from typing_extensions import override

from parlant.core.common import generate_id

NO_TRACE_ID = "<main>"
"""What ``Tracer.trace_id`` reports when no span is open: there is no trace."""

SESSION_OPERATION_ATTRIBUTE = "parlant.session.operation"
"""Marks a trace that a session operation started on its own, outside any
request: an agent speaking unprompted (``utter``) or an event emitted into a
session (``emit``). Observability exports these like request traces."""

TURN_TRACE_ID_ATTRIBUTE = "parlant.turn_trace_id"
"""The originating engine trace shared by independently accounted follow-ups."""


@dataclass(frozen=True)
class TraceContext:
    """A native span identity for a causal link, not a Parlant scope name."""

    trace_id: str
    span_id: str


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
    def fresh_trace(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
        *,
        links: Sequence[TraceContext] = (),
        trace_id: str | None = None,
    ) -> Generator[None, None, None]:
        """Open an independent root and restore inherited tracing on exit.

        Custom tracers must implement this scope to support session follow-ups.
        ``trace_id`` lets a composite share one identity across its exporters.
        Links carry native span contexts; local-only tracers may ignore them.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement fresh_trace() for session follow-ups"
        )
        yield  # pragma: no cover

    @property
    def current_span_context(self) -> TraceContext | None:
        """Return a native link target when this tracer has one."""
        return None

    @contextmanager
    @abstractmethod
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Generator[None, None, None]: ...

    @contextmanager
    @abstractmethod
    def attributes(
        self,
        attributes: Mapping[str, AttributeValue],
    ) -> Generator[None, None, None]: ...

    @property
    @abstractmethod
    def trace_id(self) -> str: ...

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
    def fresh_trace(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
        *,
        links: Sequence[TraceContext] = (),
        trace_id: str | None = None,
    ) -> Generator[None, None, None]:
        with ExitStack() as stack:
            stack.callback(
                self._trace_id.reset, self._trace_id.set(trace_id or self._generate_trace_id())
            )
            stack.callback(self._spans.reset, self._spans.set(""))
            stack.callback(self._attributes.reset, self._attributes.set({}))
            with self.span(span_id, attributes):
                yield

    @contextmanager
    @override
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Generator[None, None, None]:
        current_spans = self._spans.get()

        if not current_spans:
            new_spans = span_id

            # A trace id may already have been injected from outside (e.g. by a
            # CompositeTracer sharing one id across its children); only generate
            # a fresh one when this root span truly starts a new trace.
            if self._trace_id.get():
                trace_id_reset_token = None
            else:
                trace_id_reset_token = self._trace_id.set(self._generate_trace_id())
        else:
            new_spans = current_spans + f"::{span_id}"
            trace_id_reset_token = None

        current_attributes = self._attributes.get()
        new_attributes = {**current_attributes, **attributes}

        spans_reset_token = self._spans.set(new_spans)
        attributes_reset_token = self._attributes.set(new_attributes)

        try:
            yield
        finally:
            self._spans.reset(spans_reset_token)
            self._attributes.reset(attributes_reset_token)
            if trace_id_reset_token is not None:
                self._trace_id.reset(trace_id_reset_token)

    @contextmanager
    @override
    def attributes(
        self,
        attributes: Mapping[str, AttributeValue],
    ) -> Generator[None, None, None]:
        current_attributes = self._attributes.get()
        new_attributes = {**current_attributes, **attributes}

        attributes_reset_token = self._attributes.set(new_attributes)

        try:
            yield
        finally:
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
    def fresh_trace(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
        *,
        links: Sequence[TraceContext] = (),
        trace_id: str | None = None,
    ) -> Generator[None, None, None]:
        shared_id = trace_id or self._generate_trace_id()
        with ExitStack() as stack:
            stack.callback(self._shared_trace_id.reset, self._shared_trace_id.set(shared_id))
            for tracer in self._tracers:
                stack.enter_context(
                    tracer.fresh_trace(span_id, attributes, links=links, trace_id=shared_id)
                )
            yield

    @property
    @override
    def current_span_context(self) -> TraceContext | None:
        return next(
            (
                context
                for tracer in self._tracers
                if (context := tracer.current_span_context) is not None
            ),
            None,
        )

    @contextmanager
    @override
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Generator[None, None, None]:
        with ExitStack() as stack:
            # Generate shared trace_id if this is a root span
            current_trace_id = self._shared_trace_id.get()
            if not current_trace_id:
                shared_trace_id = self._generate_trace_id()
                trace_id_reset_token = self._shared_trace_id.set(shared_trace_id)
                stack.callback(self._shared_trace_id.reset, trace_id_reset_token)

                # Set the shared trace_id in all tracers before creating spans
                for tracer in self._tracers:
                    trace_id_context = getattr(tracer, "_trace_id", None)
                    if isinstance(trace_id_context, contextvars.ContextVar):
                        child_trace_id = cast(contextvars.ContextVar[str], trace_id_context)
                        child_reset_token = child_trace_id.set(shared_trace_id)
                        stack.callback(child_trace_id.reset, child_reset_token)

            for tracer in self._tracers:
                stack.enter_context(tracer.span(span_id, attributes))
            yield

    @contextmanager
    @override
    def attributes(
        self,
        attributes: Mapping[str, AttributeValue],
    ) -> Generator[None, None, None]:
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


@contextmanager
def session_operation_span(
    tracer: Tracer,
    span_id: str,
    session_id: str,
    operation: str,
) -> Generator[None, None, None]:
    """Capture this session's origin before replacing inherited trace state.

    Never infer an origin from the latest session message or an unrelated
    request. Engine turns establish the explicit attribute; copied asyncio
    contexts retain it even after the original span has closed.
    """
    origin = tracer.get_attribute(TURN_TRACE_ID_ATTRIBUTE)
    same_session = tracer.get_attribute("session_id") == session_id
    origin = origin if same_session and isinstance(origin, str) and origin.strip() else None
    native = tracer.current_span_context if origin else None
    with tracer.fresh_trace(
        span_id,
        {"session_id": session_id, SESSION_OPERATION_ATTRIBUTE: operation},
        links=[native] if native else [],
    ):
        tracer.set_attribute(TURN_TRACE_ID_ATTRIBUTE, origin or tracer.trace_id)
        yield
