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

"""Deterministic rendering of canned responses for the compass reviewer.

Unlike the alpha generator, this renderer performs NO generative (LLM) field
extraction — it resolves only the fields that are deterministically available in
the engine context (the standard ``std`` bundle, tool-result fields, and
rule-provided additional fields) and leaves any other field as a readable
``{{ placeholder }}``. It is therefore a pure, side-effect-free function of the
canned response and the context: no model calls, fully unit-testable, and safe to
run once per relevant canned response on the message-review path.

The rendered text is only ever shown to the reviewer as style/compliance
guidance; it is never emitted to the user, so an unresolved field surfacing as a
placeholder is harmless and, if anything, more informative to the reviewer than a
blanked value would be.
"""

from dataclasses import replace
from functools import lru_cache
from typing import Any, cast

import jinja2
import jinja2.meta
from jinja2.sandbox import ImmutableSandboxedEnvironment

from parlant.core.canned_responses import CannedResponse
from parlant.core.common import JSONSerializable
from parlant.core.engines.compass.context import EngineContext
from parlant.core.loggers import Logger
from parlant.core.sessions import EventKind, ToolCall, ToolEventData


class _PlaceholderUndefined(jinja2.Undefined):
    """A Jinja ``Undefined`` that renders an unresolved field back as its
    ``{{ path }}`` placeholder (instead of blanking or raising) and tolerates
    attribute/item access along the path, so nested references like
    ``{{ generative.summary }}`` survive intact."""

    __slots__ = ()

    def _placeholder(self) -> str:
        return "{{ " + (self._undefined_name or "") + " }}"

    def __str__(self) -> str:
        return self._placeholder()

    def __html__(self) -> str:
        return self._placeholder()

    def __getattr__(self, name: str) -> "_PlaceholderUndefined":
        if name.startswith("__"):
            raise AttributeError(name)
        base = self._undefined_name or ""
        return _PlaceholderUndefined(name=f"{base}.{name}" if base else name)

    def __getitem__(self, key: Any, /) -> Any:
        base = self._undefined_name or ""
        return _PlaceholderUndefined(name=f"{base}.{key}" if base else str(key))


# A sandboxed environment: canned-response templates can originate from
# semi-trusted sources (tool servers and retrievers supply them verbatim via
# ``ToolResult.canned_responses``), so rendering must never grant access to
# Python internals. The immutable variant additionally forbids mutating calls
# on resolved values. ``_PlaceholderUndefined`` preserves the readable
# ``{{ placeholder }}`` behavior for unresolved fields.
_ENVIRONMENT = ImmutableSandboxedEnvironment(undefined=_PlaceholderUndefined)


@lru_cache(maxsize=1024)
def _compile_template(value: str) -> jinja2.Template:
    """Compile a template once and reuse it. Compilation dominates render cost
    for the short templates canned responses use, and the same values recur
    across turns, so caching is a large win over the sandbox's small overhead."""
    return _ENVIRONMENT.from_string(value)


class CannedResponseRenderer:
    """Renders a canned response's value template against the compass engine
    context, using deterministic field extractors only."""

    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    def render(self, canned_response: CannedResponse, context: EngineContext) -> str:
        with self._logger.scope("CannedResponseRenderer"):
            return self._render(canned_response, context)

    def _render(self, canned_response: CannedResponse, context: EngineContext) -> str:
        try:
            args: dict[str, JSONSerializable] = {}

            for field_name in _template_fields(canned_response.value):
                resolved, value = self._resolve_field(field_name, context)

                if resolved:
                    args[field_name] = value

            return _compile_template(canned_response.value).render(**args)
        except jinja2.TemplateError as exc:
            # A malformed template (or a render failure) must never break the review
            # pass; fall back to the raw value so the reviewer still sees the content.
            self._logger.warning(
                f"Failed to render canned response {canned_response.id} for review "
                f"({exc}); using the raw template."
            )
            return canned_response.value

    def can_satisfy(self, canned_response: CannedResponse, context: EngineContext) -> bool:
        """Whether all of the response's declared ``field_dependencies`` are
        resolvable in the current context (std bundle, tool-result fields, or
        rule-provided additional fields).

        A response with an unmet dependency is disqualified from selection
        entirely — its data premise doesn't hold, so offering it as a style
        reference (or review guidance) would invite the model to assert
        facts nobody staged."""
        return all(
            self._resolve_field(field_name, context)[0]
            for field_name in canned_response.field_dependencies
        )

    def _resolve_field(
        self,
        field_name: str,
        context: EngineContext,
    ) -> tuple[bool, JSONSerializable]:
        if field_name == "std":
            return True, self._standard_fields(context)

        if (tool_value := self._tool_field(field_name, context)) is not None:
            return True, tool_value

        additional = context.state.additional_canned_response_fields
        if field_name in additional:
            return True, cast(JSONSerializable, additional[field_name])

        return False, None

    def _standard_fields(self, context: EngineContext) -> JSONSerializable:
        return {
            "customer": {"name": context.customer.name},
            "agent": {"name": context.agent.name},
            "variables": {
                variable.name: value.data for variable, value in context.state.context_variables
            },
            "glossary": {term.name: term.description for term in context.state.glossary_terms},
        }

    def _tool_field(self, field_name: str, context: EngineContext) -> JSONSerializable | None:
        # Staged (this-turn) tool events first, then the most recent historical ones.
        tool_calls: list[ToolCall] = []

        tool_calls.extend(
            tc
            for e in context.state.tool_events
            if e.kind == EventKind.TOOL
            for tc in cast(ToolEventData, e.data)["tool_calls"]
        )

        tool_calls.extend(
            tc
            for e in reversed(context.interaction.events)
            if e.kind == EventKind.TOOL
            for tc in cast(ToolEventData, e.data)["tool_calls"]
        )

        for tool_call in tool_calls:
            value = tool_call["result"].get("canned_response_fields", {}).get(field_name)
            if value is not None:
                return cast(JSONSerializable, value)

        return None


@lru_cache(maxsize=1024)
def _template_fields(template: str) -> frozenset[str]:
    return frozenset(jinja2.meta.find_undeclared_variables(_ENVIRONMENT.parse(template)))


def tool_provided_canned_responses(context: EngineContext) -> list[CannedResponse]:
    """Literal canned-response texts returned by this turn's tools and
    retrievers (``ToolResult.canned_responses`` / staged retriever results).

    These form the highest-precedence selection tier: the tool supplied them
    for THIS turn specifically, so they skip relevance ranking entirely. The
    entries are treated as literal response texts (not store ids) and wrapped
    as transient canned responses."""
    seen: set[str] = set()
    result: list[CannedResponse] = []

    for event in context.state.tool_events:
        if event.kind != EventKind.TOOL:
            continue

        for call in cast(ToolEventData, event.data)["tool_calls"]:
            for text in call["result"].get("canned_responses", []) or []:
                if isinstance(text, str) and text.strip() and text not in seen:
                    seen.add(text)
                    # The style-reference framing promises that an example's
                    # Note is honored — use that to carry the verbatim-use
                    # contract these tool-supplied texts imply.
                    result.append(
                        replace(
                            CannedResponse.create_transient(value=text),
                            note=(
                                "This exact response was supplied specifically for this "
                                "reply. Prefer it over composing your own, and reproduce "
                                "it word-for-word, exactly as written."
                            ),
                        )
                    )

    return result
