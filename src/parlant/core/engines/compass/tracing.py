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

import asyncio
import json
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from collections.abc import Generator
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import chain
from typing import Mapping, Protocol, TypeGuard

from parlant.core.agents import Effort
from parlant.core.canned_responses import CannedResponse
from parlant.core.common import JSONSerializable
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.glossary import Term
from parlant.core.groups import GroupId
from parlant.core import metadata
from parlant.core.nlp.react import ToolCallPart
from parlant.core.rules import Rule
from parlant.core.sessions import MessageEventData, ToolEventData
from parlant.core.tools import Tool, ToolId, ToolResult
from parlant.core.tracer import AttributeValue, Tracer


_tool_call_id: ContextVar[str | None] = ContextVar("compass_tool_call_id", default=None)


@dataclass(frozen=True)
class RuleDecision:
    """A staged authoritative per-rule verdict, emitted as rule.selected /
    rule.ruled_out only after the turn's behavioral state commits."""

    rule: Rule
    selected: bool
    rationale: str


@dataclass
class _ProcessTraceState:
    reply_emitted: bool = False
    outcome: str | None = None
    stop_reason: str | None = None


_process_trace_state: ContextVar[_ProcessTraceState | None] = ContextVar(
    "compass_process_trace_state",
    default=None,
)

_COMPONENT_EVENT_CHUNK_SIZE = 100


class _ScoredRuleTraceResult(Protocol):
    @property
    def rule(self) -> Rule: ...

    @property
    def is_relevant(self) -> bool: ...

    @property
    def score(self) -> float | None: ...


class _RankedRuleTraceResult(_ScoredRuleTraceResult, Protocol):
    @property
    def reasoning(self) -> str: ...


class _ToolReviewTraceResult(Protocol):
    @property
    def todo(self) -> str | None: ...

    @property
    def adjusted_reasoning(self) -> str | None: ...


def format_json_attr(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _is_attribute_value(value: object) -> TypeGuard[AttributeValue]:
    return isinstance(value, (str, bool, int, float))


def _is_attribute_value_sequence(value: object) -> TypeGuard[AttributeValue]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return False

    if not value:
        return True

    return (
        all(type(item) is str for item in value)
        or all(type(item) is bool for item in value)
        or all(type(item) is int for item in value)
        or all(type(item) is float for item in value)
    )


def normalize_attrs(attributes: Mapping[str, object]) -> dict[str, AttributeValue]:
    normalized_attributes: dict[str, AttributeValue] = {}

    for key, value in attributes.items():
        if value is None:
            continue

        if _is_attribute_value(value) or _is_attribute_value_sequence(value):
            normalized_attributes[key] = value
        else:
            normalized_attributes[key] = format_json_attr(value)

    return normalized_attributes


def _root_cause(error: BaseException) -> BaseException:
    while error.__cause__ is not None:
        error = error.__cause__
    return error


class CompassTracer:
    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def event(self, name: str, attributes: Mapping[str, object] | None = None) -> None:
        try:
            self._tracer.add_event(name, attributes=normalize_attrs(attributes or {}))
        except (Exception, asyncio.CancelledError):
            pass

    @contextmanager
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, object] | None = None,
    ) -> Generator[None, None, None]:
        span_context = None
        try:
            span_context = self._tracer.span(
                span_id,
                attributes=normalize_attrs(attributes or {}),
            )
            span_context.__enter__()
        except (Exception, asyncio.CancelledError):
            span_context = None

        if span_context is None:
            yield
            return

        try:
            yield
        except BaseException as engine_error:
            try:
                span_context.__exit__(
                    type(engine_error),
                    engine_error,
                    engine_error.__traceback__,
                )
            except (Exception, asyncio.CancelledError):
                pass
            raise
        else:
            try:
                span_context.__exit__(None, None, None)
            except (Exception, asyncio.CancelledError):
                pass

    @contextmanager
    def attributes(self, attributes: Mapping[str, object]) -> Generator[None, None, None]:
        attributes_context = None
        try:
            attributes_context = self._tracer.attributes(normalize_attrs(attributes))
            attributes_context.__enter__()
        except (Exception, asyncio.CancelledError):
            attributes_context = None

        if attributes_context is None:
            yield
            return

        try:
            yield
        except BaseException as engine_error:
            try:
                attributes_context.__exit__(
                    type(engine_error),
                    engine_error,
                    engine_error.__traceback__,
                )
            except (Exception, asyncio.CancelledError):
                pass
            raise
        else:
            try:
                attributes_context.__exit__(None, None, None)
            except (Exception, asyncio.CancelledError):
                pass

    @contextmanager
    def process_scope(self) -> Generator[None, None, None]:
        token = _process_trace_state.set(_ProcessTraceState())
        try:
            yield
        finally:
            _process_trace_state.reset(token)

    def mark_reply_emitted(self) -> None:
        if state := _process_trace_state.get():
            state.reply_emitted = True

    def mark_stopped(self, reason: str) -> None:
        if state := _process_trace_state.get():
            state.outcome = "stopped"
            state.stop_reason = reason

    def mark_gave_up(self) -> None:
        if state := _process_trace_state.get():
            state.outcome = "gave_up"

    def traced_outcome(self) -> tuple[str | None, str | None]:
        if state := _process_trace_state.get():
            return state.outcome, state.stop_reason
        return None, None

    def has_emitted_reply(self) -> bool:
        if state := _process_trace_state.get():
            return state.reply_emitted
        return False

    @contextmanager
    def tool_call_scope(self, tool_call_id: str) -> Generator[None, None, None]:
        token = _tool_call_id.set(tool_call_id)
        try:
            yield
        finally:
            _tool_call_id.reset(token)

    def current_tool_call_id(self) -> str | None:
        return _tool_call_id.get()

    def _set_attributes(self, attributes: Mapping[str, object]) -> None:
        try:
            for name, value in normalize_attrs(attributes).items():
                self._tracer.set_attribute(name, value)
        except (Exception, asyncio.CancelledError):
            pass

    def process_resolved(
        self,
        outcome: str,
        stage: str,
        reply_emitted: bool,
        stop_reason: str | None = None,
    ) -> None:
        self._set_attributes(
            {
                "outcome": outcome,
                "stage": stage,
                "reply_emitted": reply_emitted,
                "stop_reason": stop_reason,
            }
        )

    def process_failed(
        self,
        error: BaseException,
        stage: str,
        reply_emitted: bool,
    ) -> None:
        cause = _root_cause(error)
        self.event(
            "process.failed",
            {
                "error_type": type(cause).__name__,
                "stage": stage,
                "reply_emitted": reply_emitted,
                "terminal": not reply_emitted,
            },
        )

    def turn_interrupted(self, cause: str, session_id: str, agent_id: str) -> None:
        self.event(
            "turn.interrupted",
            {
                "cause": cause,
                "session_id": session_id,
                "agent_id": agent_id,
            },
        )

    def compaction_checked(
        self,
        needed: bool,
        effort: Effort,
        threshold: int,
        token_count: int | None = None,
        reason: str | None = None,
    ) -> None:
        self.event(
            "compaction.checked.yes" if needed else "compaction.checked.no",
            {
                "reason": reason,
                "effort": effort.value,
                "token_count": token_count,
                "threshold": threshold,
            },
        )

    def compaction_compacted(self, model: str, summary: str) -> None:
        self.event(
            "compaction.compacted",
            {
                "model": model,
                "summary_length": len(summary),
            },
        )

    def compaction_failed(self, error: BaseException) -> None:
        self.event("compaction.failed", {"error_type": type(error).__name__})

    def component_loaded(
        self,
        *,
        entity_type: str,
        parlant_id: str,
        last_modified_utc: str,
    ) -> None:
        self.event(
            "loaded.entity",
            {
                "entity_type": entity_type,
                "parlant_id": parlant_id,
                "last_modified_utc": last_modified_utc,
            },
        )

    def _components_loaded(
        self,
        references: Iterable[tuple[str, str, str]],
    ) -> None:
        unique_references: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for reference in references:
            if reference in seen:
                continue
            seen.add(reference)
            unique_references.append(reference)

        for offset in range(0, len(unique_references), _COMPONENT_EVENT_CHUNK_SIZE):
            chunk = unique_references[offset : offset + _COMPONENT_EVENT_CHUNK_SIZE]
            with self.span("compass.engine.load_components"):
                for entity_type, parlant_id, last_modified_utc in chunk:
                    self.component_loaded(
                        entity_type=entity_type,
                        parlant_id=parlant_id,
                        last_modified_utc=last_modified_utc,
                    )

    def groups_loaded(self, group_ids: Iterable[GroupId]) -> None:
        self._components_loaded(("group", str(group_id), "") for group_id in group_ids)

    def context_variables_loaded(
        self,
        loaded_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
    ) -> None:
        self._components_loaded(
            chain(
                (
                    (
                        "context_variable",
                        str(variable.id),
                        variable.modified_utc.isoformat(),
                    )
                    for variable, _ in loaded_variables
                ),
                (
                    ("group", str(group_id), "")
                    for variable, _ in loaded_variables
                    for group_id in variable.groups
                ),
            )
        )

        for variable, value in loaded_variables:
            self.event(
                "loaded.variable",
                {
                    "variable_id": variable.id,
                    "name": variable.name,
                    "value_type": type(value.data).__name__,
                    "value_size_chars": len(format_json_attr(value.data)),
                    "value": value.data,
                    "value_is_null": True if value.data is None else None,
                    "modified_utc": value.modified_utc.isoformat(),
                },
            )

    def glossary_loaded(self, terms: Iterable[Term]) -> None:
        sorted_terms = sorted(terms, key=lambda t: (t.name, str(t.id)))
        self._components_loaded(
            chain(
                (("term", str(term.id), term.modified_utc.isoformat()) for term in sorted_terms),
                (("group", str(group_id), "") for term in sorted_terms for group_id in term.groups),
            )
        )

        for term in sorted_terms:
            self.event(
                "loaded.glossary",
                {
                    "term_id": str(term.id),
                    "name": term.name,
                    "last_modified": term.modified_utc.isoformat(),
                },
            )

    def rules_loaded(self, rules: Sequence[Rule]) -> None:
        unique_rules: list[Rule] = []
        seen: set[tuple[str, str]] = set()
        for rule in rules:
            reference = (str(rule.id), rule.modified_utc.isoformat())
            if reference in seen:
                continue
            seen.add(reference)
            unique_rules.append(rule)

        self._components_loaded(
            chain(
                (("rule", str(rule.id), rule.modified_utc.isoformat()) for rule in unique_rules),
                (("group", str(group_id), "") for rule in unique_rules for group_id in rule.groups),
            )
        )

        for rule in unique_rules:
            self.event(
                "loaded.rule",
                {
                    "rule_id": str(rule.id),
                    "title": rule.title or "",
                    "last_modified": rule.modified_utc.isoformat(),
                    "weight": rule.weight.value,
                    "effort": rule.effort.value if rule.effort else "",
                },
            )

    def canned_responses_loaded(self, canned_responses: Iterable[CannedResponse]) -> None:
        loaded_canned_responses = list(canned_responses)
        self._components_loaded(
            chain(
                (
                    (
                        "canned_response",
                        str(canned_response.id),
                        canned_response.modified_utc.isoformat(),
                    )
                    for canned_response in loaded_canned_responses
                ),
                (
                    ("group", str(group_id), "")
                    for canned_response in loaded_canned_responses
                    for group_id in canned_response.groups
                ),
            )
        )

    def rule_function_matched(
        self,
        rule: Rule,
        matched: bool,
        error_type: str | None,
    ) -> None:
        self.event(
            "match.function.yes" if matched else "match.function.no",
            {
                "rule_id": str(rule.id),
                "title": rule.title or "",
                "error_type": error_type,
            },
        )

    def rules_recalled(self, recalled_rules: Iterable[_ScoredRuleTraceResult]) -> None:
        for recalled_rule in recalled_rules:
            self.event(
                "match.recall.yes" if recalled_rule.is_relevant else "match.recall.no",
                {
                    "rule_id": str(recalled_rule.rule.id),
                    "title": recalled_rule.rule.title or "",
                    "score": recalled_rule.score,
                },
            )

    def rules_ranked(self, ranked_rules: Iterable[_RankedRuleTraceResult]) -> None:
        for ranked_rule in ranked_rules:
            self.event(
                "match.rank.yes" if ranked_rule.is_relevant else "match.rank.no",
                {
                    "rule_id": str(ranked_rule.rule.id),
                    "title": ranked_rule.rule.title or "",
                    "score": ranked_rule.score,
                    "reasoning": ranked_rule.reasoning,
                },
            )

    def rule_decision(self, rule: Rule, *, selected: bool, rationale: str) -> None:
        self.event(
            "rule.selected" if selected else "rule.ruled_out",
            {
                "rule_id": str(rule.id),
                "title": rule.title or "",
                "last_modified": rule.modified_utc.isoformat(),
                "rationale": rationale,
            },
        )

    def effort_raised(
        self,
        from_effort: Effort,
        to_effort: Effort,
        rule_ids: Sequence[str],
    ) -> None:
        self.event(
            "action.raise_effort",
            {
                "from_effort": from_effort.value,
                "to_effort": to_effort.value,
                "rule_ids": sorted(rule_ids),
            },
        )

    def labels_added(self, labels: Iterable[str], rule_ids: Iterable[str]) -> None:
        self.event(
            "action.add_label",
            {
                "labels": sorted(labels),
                "rule_ids": sorted(rule_ids),
            },
        )

    def tool_recall_scored(self, candidate_count: int, scored_count: int) -> None:
        self._set_attributes(
            {
                "candidate_count": candidate_count,
                "scored_count": scored_count,
            }
        )

    def tools_selected(self, tools: Sequence[Tool]) -> None:
        self._set_attributes({"selected_count": len(tools)})

    def loop_reasoning_started(self) -> None:
        self.event("loop.reasoning.started")

    def loop_step_resolved(self, outcome: str, retry_count: int) -> None:
        self._set_attributes(
            {
                "outcome": outcome,
                "retry_count": retry_count,
            }
        )

    def loop_reasoning_finished(
        self,
        chunk_count: int | None = None,
        *,
        outcome: str = "completed",
    ) -> None:
        self.event(
            "loop.reasoning.finished",
            {"chunk_count": chunk_count, "outcome": outcome},
        )

    def loop_message_started(self, mode: str) -> None:
        self.event("loop.message.started", {"mode": mode})

    def loop_message_finished(
        self,
        mode: str | None = None,
        emitted: bool | None = None,
        *,
        outcome: str = "completed",
    ) -> None:
        self.event(
            "loop.message.finished",
            {"mode": mode, "emitted": emitted, "outcome": outcome},
        )

    def loop_tools_started(self, tool_count: int | None = None) -> None:
        self.event("loop.tools.started", {"tool_count": tool_count})

    def loop_tools_finished(
        self,
        tool_count: int | None = None,
        *,
        outcome: str = "completed",
    ) -> None:
        self.event(
            "loop.tools.finished",
            {"tool_count": tool_count, "outcome": outcome},
        )

    def loop_reasoning(
        self,
        reasoning: str,
        chunk_count: int,
        *,
        replaces_previous: bool = False,
    ) -> None:
        attributes: dict[str, object] = {
            "reasoning": reasoning,
            "chunk_count": chunk_count,
        }
        if replaces_previous:
            attributes["replaces_previous"] = True

        self.event(
            "loop.reasoning",
            attributes,
        )

    def loop_message(self, data: MessageEventData, mode: str, output_ref: str) -> None:
        data_mapping = dict(data)
        chunks = data_mapping.get("chunks")
        self.event(
            "loop.message",
            {
                "mode": mode,
                "message": data_mapping.get("message"),
                "chunk_count": len(chunks) if isinstance(chunks, list) else None,
                "output_ref": output_ref,
            },
        )

    def loop_tool_transient(
        self,
        data: ToolEventData,
        tool_call_ids: Sequence[str],
    ) -> None:
        self._loop_tool_event("loop.tool.transient", data, tool_call_ids)

    def loop_tool_persistent(
        self,
        data: ToolEventData,
        tool_call_ids: Sequence[str],
    ) -> None:
        self._loop_tool_event("loop.tool.persistent", data, tool_call_ids)

    def _loop_tool_event(
        self,
        name: str,
        data: ToolEventData,
        tool_call_ids: Sequence[str],
    ) -> None:
        calls = data["tool_calls"]
        calls_with_ids = [
            {**call, "tool_call_id": tool_call_id}
            for call, tool_call_id in zip(calls, tool_call_ids, strict=True)
        ]
        self.event(
            name,
            {
                "tool_count": len(calls),
                "tool_names": [call["tool_id"] for call in calls],
                "tool_call_ids": list(tool_call_ids),
                "tool_calls": calls_with_ids,
            },
        )

    def loop_give_up(self, reason: str, iteration_count: int) -> None:
        self.event(
            "loop.give_up",
            {
                "reason": reason,
                "iteration_count": iteration_count,
            },
        )

    def tool_calls_requested(self, tool_calls: Sequence[ToolCallPart]) -> None:
        for tool_call in tool_calls:
            self.event(
                "tool.requested",
                {
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "arguments": tool_call.args,
                },
            )

    def tool_called(
        self,
        tool: ToolId,
        tool_call_id: str,
        arguments: Mapping[str, JSONSerializable],
    ) -> None:
        self.event(
            "tool.called",
            {
                "tool_id": tool.to_string(),
                "tool_call_id": tool_call_id,
                "tool_name": tool.tool_name,
                "service_name": tool.service_name,
                "arguments": arguments,
            },
        )

    def tool_result(
        self,
        tool: ToolId,
        tool_call_id: str,
        result: ToolResult,
    ) -> None:
        self.event(
            "tool.result",
            {
                "tool_id": tool.to_string(),
                "tool_call_id": tool_call_id,
                "tool_name": tool.tool_name,
                "service_name": tool.service_name,
                "is_error": metadata.internal_value(result.metadata, ("error_details",))
                is not None,
                "result": result.data,
                "result_is_null": True if result.data is None else None,
                "metadata": result.metadata,
                "control": result.control,
                "costs": {category: format(value, "f") for category, value in result.costs.items()},
            },
        )

    def tool_error(
        self,
        tool: ToolId,
        tool_call_id: str,
        result: ToolResult,
        error_type: str,
        timed_out: bool,
    ) -> None:
        self.event(
            "tool.error",
            {
                "tool_id": tool.to_string(),
                "tool_call_id": tool_call_id,
                "tool_name": tool.tool_name,
                "service_name": tool.service_name,
                "error_type": error_type,
                "error_details": metadata.internal_value(result.metadata, ("error_details",)),
                "timed_out": timed_out,
            },
        )

    def tool_call_error(
        self,
        tool_call: ToolCallPart,
        error_type: str,
        message: str,
    ) -> None:
        self.event(
            "tool.error",
            {
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.name,
                "error_type": error_type,
                "message": message,
            },
        )

    def tool_review_started(
        self,
        review_ref: str,
        tool_calls: Sequence[ToolCallPart],
    ) -> None:
        self.event(
            "review.started",
            {
                "review_ref": review_ref,
                "tool_count": len(tool_calls),
                "tool_names": [tool_call.name for tool_call in tool_calls],
            },
        )

    def tool_review(
        self,
        review_ref: str,
        result: _ToolReviewTraceResult,
        tool_calls: Sequence[ToolCallPart],
        available_tools: Sequence[Tool],
        rejected: bool,
    ) -> None:
        review_status = "rejected" if rejected else "passed"
        approved = not rejected
        has_todo = bool(result.todo)
        has_adjusted_reasoning = bool(result.adjusted_reasoning)
        consequential_by_name = {tool.name: tool.consequential for tool in available_tools}
        self.event(
            "review.rejected" if rejected else "review.passed",
            {
                "review_ref": review_ref,
                "tool_count": len(tool_calls),
                "tool_names": [tool_call.name for tool_call in tool_calls],
                "approved": approved,
                "has_todo": has_todo,
                "has_adjusted_reasoning": has_adjusted_reasoning,
            },
        )

        for tool_call in tool_calls:
            self.event(
                "tool.reviewed",
                {
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "arguments": tool_call.args,
                    "review_status": review_status,
                    "approved": approved,
                    "consequential": consequential_by_name.get(tool_call.name, False),
                    "has_todo": has_todo,
                    "has_adjusted_reasoning": has_adjusted_reasoning,
                },
            )
