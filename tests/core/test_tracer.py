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

from parlant.core.tracer import CompositeTracer, LocalTracer


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
