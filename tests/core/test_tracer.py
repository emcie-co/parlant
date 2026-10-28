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

import pytest
import asyncio

from parlant.core.tracer import (
    CompositeTracer,
    LocalTracer,
    TURN_TRACE_ID_ATTRIBUTE,
    session_operation_span,
)


@pytest.mark.parametrize(
    "caller_session,has_origin", [("same", True), ("different", True), ("same", False)]
)
def test_that_session_operations_only_link_explicit_origins_from_the_same_session(
    caller_session: str,
    has_origin: bool,
) -> None:
    tracer = LocalTracer()
    with tracer.span("request", {"session_id": caller_session}):
        origin = tracer.trace_id
        if has_origin:
            tracer.set_attribute(TURN_TRACE_ID_ATTRIBUTE, origin)
        with session_operation_span(tracer, "sessions.utter", "same", "utter"):
            assert tracer.trace_id != origin
            expected = origin if has_origin and caller_session == "same" else tracer.trace_id
            assert tracer.get_attribute(TURN_TRACE_ID_ATTRIBUTE) == expected
        assert tracer.trace_id == origin


async def test_that_cancelling_a_fresh_scope_restores_inherited_context() -> None:
    tracer = LocalTracer()
    started = asyncio.Event()
    with tracer.span("origin"):
        origin = tracer.trace_id

        async def run() -> None:
            try:
                with tracer.fresh_trace("follow_up"):
                    assert tracer.trace_id != origin
                    started.set()
                    await asyncio.Future[None]()
            finally:
                assert tracer.trace_id == origin
                assert tracer.span_id == "origin"

        task = asyncio.create_task(run())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert tracer.trace_id == origin


def test_that_composite_native_trace_links_use_the_shared_trace_identity() -> None:
    from tests.core.stable.engines.compass.test_tracing import _bounded_otel_tracer

    native, exporter = _bounded_otel_tracer()
    tracer = CompositeTracer([LocalTracer(), native])
    try:
        with tracer.span("origin"):
            origin = tracer.current_span_context
            assert origin is not None and origin.trace_id == tracer.trace_id
            with tracer.fresh_trace("follow_up", links=[origin]):
                current = tracer.current_span_context
                assert current is not None and current.trace_id == tracer.trace_id
                assert current.trace_id != origin.trace_id
            assert tracer.current_span_context == origin
        spans = exporter.get_finished_spans()
        follow = next(span for span in spans if span.name == "follow_up")
        assert follow.links[0].context.trace_id == int(origin.trace_id, 16)
        assert follow.links[0].context.span_id == int(origin.span_id, 16)
    finally:
        native._tracer_provider.shutdown()


@pytest.mark.parametrize("fails", [False, True])
def test_that_a_fresh_trace_restores_its_parent_after_success_or_failure(fails: bool) -> None:
    tracer = LocalTracer()
    with tracer.span("parent", {"old": "parent"}):
        parent_id = tracer.trace_id
        try:
            with tracer.fresh_trace("follow_up", {TURN_TRACE_ID_ATTRIBUTE: parent_id}):
                assert tracer.trace_id != parent_id
                assert tracer.span_id == "follow_up"
                assert tracer.get_attribute("old") is None
                assert tracer.get_attribute(TURN_TRACE_ID_ATTRIBUTE) == parent_id
                if fails:
                    raise ValueError("failed follow-up")
        except ValueError:
            assert fails
        assert tracer.trace_id == parent_id
        assert tracer.span_id == "parent"
        assert tracer.get_attribute("old") == "parent"
        assert tracer.get_attribute(TURN_TRACE_ID_ATTRIBUTE) is None
    assert tracer.trace_id == "<main>"


def test_that_composite_tracers_share_the_new_trace_without_reusing_the_parent() -> None:
    children = [LocalTracer(), LocalTracer()]
    tracer = CompositeTracer(children)
    with tracer.span("parent"):
        parent = tracer.trace_id
        with tracer.fresh_trace("follow_up"):
            fresh = tracer.trace_id
            assert fresh != parent
            assert all(child.trace_id == fresh for child in children)
            assert all(child.span_id == "follow_up" for child in children)
            with tracer.span("nested"):
                assert all(child.trace_id == fresh for child in children)
        assert tracer.trace_id == parent
        assert all(child.trace_id == parent for child in children)
    assert all(child.trace_id == "<main>" for child in children)


def test_that_local_tracer_span_state_is_reset_when_the_body_raises() -> None:
    tracer = LocalTracer()

    with pytest.raises(ValueError):
        with tracer.span("outer"):
            raise ValueError("body failure")

    assert tracer.trace_id == "<main>"
    assert tracer.span_id == "<main>"


def test_that_local_tracer_attributes_are_reset_when_the_body_raises() -> None:
    tracer = LocalTracer()

    with pytest.raises(ValueError):
        with tracer.attributes({"key": "value"}):
            raise ValueError("body failure")

    assert tracer.get_attribute("key") is None


def test_that_composite_tracer_resets_child_trace_ids_after_a_root_span() -> None:
    child = LocalTracer()
    composite = CompositeTracer([child])

    with composite.span("root"):
        assert child.trace_id != "<main>"

    assert child.trace_id == "<main>"


def test_that_composite_tracer_state_is_reset_when_the_body_raises() -> None:
    child = LocalTracer()
    composite = CompositeTracer([child])

    with pytest.raises(ValueError):
        with composite.span("root"):
            raise ValueError("body failure")

    assert child.trace_id == "<main>"
    assert child.span_id == "<main>"


def test_that_composite_tracer_shares_one_trace_id_across_children() -> None:
    first = LocalTracer()
    second = LocalTracer()
    composite = CompositeTracer([first, second])

    with composite.span("root"):
        assert first.trace_id == second.trace_id != "<main>"

        with composite.span("nested"):
            assert first.trace_id == second.trace_id != "<main>"
