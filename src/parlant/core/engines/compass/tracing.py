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

import json
from collections.abc import Iterable, Sequence
from typing import Mapping, Protocol, TypeGuard

from parlant.core.agents import Effort
from parlant.core.tracer import AttributeValue, Tracer
from parlant.core.common import JSONSerializable
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.glossary import Term
from parlant.core.nlp.react import ToolCallPart
from parlant.core.rules import Rule
from parlant.core.sessions import MessageEventData, ToolEventData
from parlant.core.tools import ToolId, ToolResult


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


class _DistilledRuleTraceResult(Protocol):
    @property
    def rule(self) -> Rule: ...

    @property
    def reasoning(self) -> str: ...

    @property
    def is_relevant(self) -> bool: ...

    @property
    def highlights(self) -> Sequence[str]: ...


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


class CompassTracer:
    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def event(self, name: str, attributes: Mapping[str, object] | None = None) -> None:
        self._tracer.add_event(name, attributes=normalize_attrs(attributes or {}))

    def process_failed(self, error: BaseException) -> None:
        self.event("process.failed", {"error_type": type(error).__name__})

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

    def context_variables_loaded(
        self,
        loaded_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
    ) -> None:
        for variable, value in loaded_variables:
            self.event(
                "loaded.variable",
                {
                    "variable_id": variable.id,
                    "name": variable.name,
                    "value_type": type(value.data).__name__,
                    "value_size_chars": len(format_json_attr(value.data)),
                    "value": value.data,
                    "modified_utc": value.modified_utc.isoformat(),
                },
            )

    def glossary_loaded(self, terms: Iterable[Term]) -> None:
        for term in sorted(terms, key=lambda t: (t.name, str(t.id))):
            self.event(
                "loaded.glossary",
                {
                    "term_id": str(term.id),
                    "name": term.name,
                    "last_modified": term.modified_utc.isoformat(),
                },
            )

    def rules_loaded(self, rules: Sequence[Rule]) -> None:
        for rule in rules:
            self.event(
                "loaded.rule",
                {
                    "rule_id": str(rule.id),
                    "title": rule.title or "",
                    "last_modified": rule.modified_utc.isoformat(),
                    "criticality": rule.weight.value,
                    "effort": rule.effort_lift.value if rule.effort_lift else "",
                },
            )

    def rule_function_matched(
        self,
        rule: Rule,
        matched: bool,
        error_type: str | None,
    ) -> None:
        self.event(
            "matched.function.yes" if matched else "matched.function.no",
            {
                "rule_id": str(rule.id),
                "title": rule.title or "",
                "error_type": error_type,
            },
        )

    def rules_recalled(self, recalled_rules: Iterable[_ScoredRuleTraceResult]) -> None:
        for recalled_rule in recalled_rules:
            self.event(
                "matched.recall.yes" if recalled_rule.is_relevant else "matched.recall.no",
                {
                    "rule_id": str(recalled_rule.rule.id),
                    "title": recalled_rule.rule.title or "",
                    "score": recalled_rule.score,
                },
            )

    def rules_ranked(self, ranked_rules: Iterable[_RankedRuleTraceResult]) -> None:
        for ranked_rule in ranked_rules:
            self.event(
                "matched.rank.yes" if ranked_rule.is_relevant else "matched.rank.no",
                {
                    "rule_id": str(ranked_rule.rule.id),
                    "title": ranked_rule.rule.title or "",
                    "score": ranked_rule.score,
                    "reasoning": ranked_rule.reasoning,
                },
            )

    def rules_distilled(self, distilled_rules: Iterable[_DistilledRuleTraceResult]) -> None:
        for distilled_rule in distilled_rules:
            self.event(
                "matched.distill.yes" if distilled_rule.is_relevant else "matched.distill.no",
                {
                    "rule_id": str(distilled_rule.rule.id),
                    "title": distilled_rule.rule.title or "",
                    "reasoning": distilled_rule.reasoning,
                    "relevant_points_count": len(distilled_rule.highlights),
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

    def loop_reasoning_started(self) -> None:
        self.event("loop.reasoning.started")

    def loop_reasoning_finished(self, chunk_count: int | None = None) -> None:
        self.event("loop.reasoning.finished", {"chunk_count": chunk_count})

    def loop_message_started(self, mode: str) -> None:
        self.event("loop.message.started", {"mode": mode})

    def loop_message_finished(self, mode: str | None = None, emitted: bool | None = None) -> None:
        self.event("loop.message.finished", {"mode": mode, "emitted": emitted})

    def loop_tools_started(self, tool_count: int | None = None) -> None:
        self.event("loop.tools.started", {"tool_count": tool_count})

    def loop_tools_finished(self, tool_count: int | None = None) -> None:
        self.event("loop.tools.finished", {"tool_count": tool_count})

    def loop_reasoning(self, reasoning: str, chunk_count: int) -> None:
        self.event(
            "loop.reasoning",
            {
                "reasoning": reasoning,
                "chunk_count": chunk_count,
            },
        )

    def loop_message(self, data: MessageEventData, mode: str) -> None:
        data_mapping = dict(data)
        chunks = data_mapping.get("chunks")
        self.event(
            "loop.message",
            {
                "mode": mode,
                "message": data_mapping.get("message"),
                "chunk_count": len(chunks) if isinstance(chunks, list) else None,
            },
        )

    def loop_tool_transient(self, data: ToolEventData) -> None:
        self._loop_tool_event("loop.tool.transient", data)

    def loop_tool_persistent(self, data: ToolEventData) -> None:
        self._loop_tool_event("loop.tool.persistent", data)

    def _loop_tool_event(self, name: str, data: ToolEventData) -> None:
        calls = data["tool_calls"]
        self.event(
            name,
            {
                "tool_count": len(calls),
                "tool_names": [call["tool_id"] for call in calls],
                "tool_calls": calls,
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
        arguments: Mapping[str, JSONSerializable],
    ) -> None:
        self.event(
            "tool.called",
            {
                "tool_id": tool.to_string(),
                "tool_name": tool.tool_name,
                "service_name": tool.service_name,
                "arguments": arguments,
            },
        )

    def tool_result(self, tool: ToolId, result: ToolResult) -> None:
        self.event(
            "tool.result",
            {
                "tool_id": tool.to_string(),
                "tool_name": tool.tool_name,
                "service_name": tool.service_name,
                "is_error": "error_details" in result.metadata,
                "result": result.data,
                "metadata": result.metadata,
                "control": result.control,
            },
        )

    def tool_error(self, tool: ToolId, result: ToolResult) -> None:
        self.event(
            "tool.error",
            {
                "tool_id": tool.to_string(),
                "tool_name": tool.tool_name,
                "service_name": tool.service_name,
                "error_details": result.metadata.get("error_details"),
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

    def tool_review(
        self,
        result: _ToolReviewTraceResult,
        tool_calls: Sequence[ToolCallPart],
        rejected: bool,
    ) -> None:
        review_status = "rejected" if rejected else "passed"
        self.event(
            "review.rejected" if rejected else "review.passed",
            {
                "tool_count": len(tool_calls),
                "tool_names": [tool_call.name for tool_call in tool_calls],
                "todo": result.todo,
                "adjusted_reasoning": result.adjusted_reasoning,
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
                    "todo": result.todo,
                    "adjusted_reasoning": result.adjusted_reasoning,
                },
            )
