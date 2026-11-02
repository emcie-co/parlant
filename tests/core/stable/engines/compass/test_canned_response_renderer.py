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

from dataclasses import replace

from parlant.core.canned_responses import CannedResponse
from parlant.core.engines.compass.canned_response_renderer import CannedResponseRenderer
from parlant.core.engines.compass.context import EngineContext, ResponseState
from parlant.core.loggers import StdoutLogger
from parlant.core.sessions import EventSource
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.matching.utils import create_engine_context


def _renderer() -> CannedResponseRenderer:
    return CannedResponseRenderer(logger=StdoutLogger(LocalTracer()))


def _context() -> EngineContext:
    context = create_engine_context(
        conversation=[(EventSource.CUSTOMER, "Hi there.")],
    )
    context.state = ResponseState()
    return context


def _canned_response(value: str) -> CannedResponse:
    return CannedResponse.create_transient(value=value)


def test_that_a_template_without_fields_renders_verbatim() -> None:
    rendered = _renderer().render(_canned_response("Hello, how can I help?"), _context())

    assert rendered == "Hello, how can I help?"


def test_that_standard_customer_and_agent_fields_are_resolved() -> None:
    context = _context()

    rendered = _renderer().render(
        _canned_response("Hi {{ std.customer.name }}, I'm {{ std.agent.name }}."),
        context,
    )

    assert rendered == "Hi Test Customer, I'm Test Agent."


def test_that_context_variables_are_resolved_under_std() -> None:
    from types import SimpleNamespace

    context = _context()
    context.state.context_variables = [
        (SimpleNamespace(name="plan"), SimpleNamespace(data="gold")),  # type: ignore[list-item]
    ]

    rendered = _renderer().render(
        _canned_response("Your plan is {{ std.variables.plan }}."),
        context,
    )

    assert rendered == "Your plan is gold."


def test_that_additional_fields_are_resolved() -> None:
    context = _context()
    context.state.additional_canned_response_fields = {"order_status": "shipped"}

    rendered = _renderer().render(
        _canned_response("Your order is {{ order_status }}."),
        context,
    )

    assert rendered == "Your order is shipped."


def test_that_unresolved_generative_field_is_kept_as_a_placeholder() -> None:
    rendered = _renderer().render(
        _canned_response("Summary: {{ generative.summary }}."),
        _context(),
    )

    assert rendered == "Summary: {{ generative.summary }}."


def test_that_unresolved_flat_field_is_kept_as_a_placeholder() -> None:
    rendered = _renderer().render(
        _canned_response("Your balance is {{ balance }}."),
        _context(),
    )

    assert rendered == "Your balance is {{ balance }}."


def test_that_a_mix_of_resolved_and_unresolved_fields_renders_correctly() -> None:
    context = _context()
    context.state.additional_canned_response_fields = {"order_status": "ready"}

    rendered = _renderer().render(
        _canned_response(
            "Hi {{ std.customer.name }}, order is {{ order_status }}; {{ generative.note }}."
        ),
        context,
    )

    assert rendered == "Hi Test Customer, order is ready; {{ generative.note }}."


def test_that_a_malformed_template_falls_back_to_the_raw_value() -> None:
    malformed = "Hello {{ unclosed "

    rendered = _renderer().render(_canned_response(malformed), _context())

    assert rendered == malformed


def _canned_response_with_dependencies(value: str, dependencies: list[str]) -> CannedResponse:
    return replace(CannedResponse.create_transient(value=value), field_dependencies=dependencies)


def test_that_a_response_without_field_dependencies_is_satisfiable() -> None:
    assert _renderer().can_satisfy(_canned_response("Your order is ready."), _context())


def test_that_an_unmet_field_dependency_disqualifies_a_response() -> None:
    # Nothing in the context provides an "order" field — the response's data
    # premise doesn't hold, so it must not be offered as a style reference.
    response = _canned_response_with_dependencies("Your order is ready.", ["order"])

    assert not _renderer().can_satisfy(response, _context())


def test_that_a_rule_provided_field_satisfies_a_dependency() -> None:
    response = _canned_response_with_dependencies("Your order is ready.", ["order"])
    context = _context()
    context.state.additional_canned_response_fields = {"order": {"status": "ready"}}

    assert _renderer().can_satisfy(response, context)


def test_that_all_field_dependencies_must_be_met() -> None:
    response = _canned_response_with_dependencies("Details inside.", ["order", "invoice"])
    context = _context()
    context.state.additional_canned_response_fields = {"order": {"status": "ready"}}

    assert not _renderer().can_satisfy(response, context)


def test_that_the_std_bundle_satisfies_a_std_dependency() -> None:
    assert _renderer().can_satisfy(
        _canned_response_with_dependencies("Hello {{std.customer.name}}.", ["std"]),
        _context(),
    )


def test_that_a_template_cannot_reach_python_internals_through_attribute_traversal() -> None:
    # A sandbox-escape payload must not execute; the sandbox raises, so the
    # renderer falls back to the raw template (its malformed-template behavior)
    # rather than substituting the result of running code. The payload text
    # itself contains the marker, so we assert the template is returned
    # unchanged (no execution/substitution occurred).
    payload = "{{ cycler.__init__.__globals__.os.popen('echo SSTI_PWNED').read() }}"

    rendered = _renderer().render(_canned_response(payload), _context())

    assert rendered == payload


def test_that_a_template_cannot_call_mutating_methods_on_resolved_values() -> None:
    context = _context()
    context.state.additional_canned_response_fields = {"items": [1, 2, 3]}

    _renderer().render(
        _canned_response("{{ items.append(4) }}"),
        context,
    )

    # The immutable sandbox forbids the mutating call, so the append never
    # happens and the underlying list is unchanged.
    assert context.state.additional_canned_response_fields["items"] == [1, 2, 3]


def test_that_rendering_the_same_template_twice_reuses_the_compiled_template() -> None:
    from parlant.core.engines.compass import canned_response_renderer as module

    module._compile_template.cache_clear()

    renderer = _renderer()
    template = _canned_response("Hi {{ std.customer.name }}.")

    renderer.render(template, _context())
    renderer.render(template, _context())

    info = module._compile_template.cache_info()
    assert info.misses == 1
    assert info.hits >= 1
