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

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from parlant.client.types.event import Event

if TYPE_CHECKING:
    from parlant.testing.suite import Suite


@dataclass(frozen=True)
class ToolCall:
    """Represents a tool call extracted from a tool event."""

    tool_id: str
    arguments: Mapping[str, Any]
    result: Any


@dataclass(frozen=True)
class Should:
    """A weighted assertion condition.

    Args:
        value: The condition string to evaluate.
        weight: The weight of this condition for scoring (default 1.0).
    """

    value: str
    weight: float = 1.0


class Response:
    """Wraps all events from a single agent turn (same trace_id).

    Provides both raw access to events and processed accessors for
    messages, tool calls, and status events.
    """

    def __init__(
        self,
        events: List[Event],
        trace_id: str,
        suite: "Suite",
        test_name: Any = None,
        listener: Any = None,
        conversation: Optional[List[tuple[str, str]]] = None,
    ) -> None:
        self._events = events
        self._trace_id = trace_id
        self._suite = suite
        self._test_name = test_name
        self._listener = listener
        self._conversation = conversation or []  # (role, message) pairs

    @property
    def trace_id(self) -> str:
        """The trace ID for this response."""
        return self._trace_id

    @property
    def events(self) -> List[Event]:
        """All events with this trace_id."""
        return self._events

    def __iter__(self) -> Iterator[Event]:
        """Iterate over raw Event objects."""
        return iter(self._events)

    @property
    def message_events(self) -> List[Event]:
        """Only kind=MESSAGE events."""
        return [e for e in self._events if e.kind == "message"]

    @property
    def messages(self) -> List[str]:
        """Message content strings from all message events."""
        result: List[str] = []
        for event in self.message_events:
            data = event.data
            if isinstance(data, dict) and "message" in data:
                result.append(str(data["message"]))
        return result

    @property
    def message(self) -> str:
        """Concatenated message content from all message events, separated by newlines."""
        return "\n\n".join(self.messages)

    @property
    def tool_events(self) -> List[Event]:
        """Only kind=TOOL events."""
        return [e for e in self._events if e.kind == "tool"]

    @property
    def tool_calls(self) -> List[ToolCall]:
        """Extracted tool calls from all tool events."""
        result: List[ToolCall] = []
        for event in self.tool_events:
            data = event.data
            if isinstance(data, dict) and "tool_calls" in data:
                for tc in data["tool_calls"]:
                    if isinstance(tc, dict):
                        result.append(
                            ToolCall(
                                tool_id=tc.get("tool_id", ""),
                                arguments=tc.get("arguments", {}),
                                result=tc.get("result"),
                            )
                        )
        return result

    @property
    def status_events(self) -> List[Event]:
        """Only kind=STATUS events."""
        return [e for e in self._events if e.kind == "status"]

    async def should(self, condition: Union[str, Should, Sequence[Union[str, Should]]]) -> float:
        """Assert condition(s) on response.message and return a normalized score.

        Formats condition as "The message should {condition}" and runs nlp_test.
        For multiple conditions, runs all in parallel with safe_gather.
        Returns a score from 0-100 based on weighted conditions.
        Raises AssertionError on failure.

        Args:
            condition: Single condition (str or Should) or sequence of conditions.
                       Use Should(value, weight) for weighted assertions.
                       Unweighted strings default to weight=1.

        Returns:
            A normalized score from 0 to 100.
        """
        from parlant.core.async_utils import safe_gather

        # Normalize input to list of Should objects
        if isinstance(condition, str):
            conditions = [Should(value=condition, weight=1.0)]
        elif isinstance(condition, Should):
            conditions = [condition]
        else:
            conditions = [
                c if isinstance(c, Should) else Should(value=c, weight=1.0) for c in condition
            ]

        # Extract condition strings for listener
        condition_strs = [c.value for c in conditions]

        # Notify listener that we're evaluating (with conditions)
        if self._listener and self._test_name:
            await self._listener.on_evaluating(self._test_name, condition_strs)

        # Build full conversation context including the agent's response
        context_parts = []
        for role, msg in self._conversation:
            context_parts.append(f"{role}: {msg}")
        # Add the agent's response being evaluated
        context_parts.append(f"Agent: {self.message}")
        full_context = "\n".join(context_parts)

        async def check_condition(cond: Should) -> tuple[Should, bool, str]:
            formatted = f"The message should {cond.value}"
            result, reasoning = await self._suite.nlp_test(full_context, formatted)
            # Notify listener of individual condition result
            if self._listener and self._test_name:
                await self._listener.on_condition_result(self._test_name, cond.value, result)
            return cond, result, reasoning

        results = await safe_gather(*[check_condition(c) for c in conditions])

        # Calculate weighted score
        total_weight = sum(c.weight for c in conditions)
        earned_weight = 0.0
        failures: List[tuple[str, float, str]] = []

        for cond, passed, reasoning in results:
            if passed:
                earned_weight += cond.weight
            else:
                failures.append((cond.value, cond.weight, reasoning))

        # Normalize to 0-100
        score = (earned_weight / total_weight) * 100 if total_weight > 0 else 0.0

        # Notify listener of the assertion score
        if self._listener and self._test_name:
            await self._listener.on_assertion_score(self._test_name, score)

        if failures:
            failure_details = "\n".join(
                f"  - '{cond}' ({(weight / total_weight * 100):.0f}%): {reasoning}"
                for cond, weight, reasoning in failures
            )
            raise AssertionError(
                f"Response assertion failed (score: {score:.1f}/100):\n"
                f"Actual message: {self.message!r}\n"
                f"Failed conditions:\n{failure_details}"
            )

        return score
