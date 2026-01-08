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
    ) -> None:
        self._events = events
        self._trace_id = trace_id
        self._suite = suite
        self._test_name = test_name
        self._listener = listener

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

    async def should(self, condition: Union[str, Sequence[str]]) -> None:
        """Assert condition(s) on response.message.

        Formats condition as "The message should {condition}" and runs nlp_test.
        For multiple conditions, runs all in parallel with safe_gather.
        Raises AssertionError on failure.

        Args:
            condition: Single condition string or sequence of conditions.
        """
        from parlant.core.async_utils import safe_gather

        if isinstance(condition, str):
            conditions = [condition]
        else:
            conditions = list(condition)

        # Notify listener that we're evaluating (with conditions)
        if self._listener and self._test_name:
            await self._listener.on_evaluating(self._test_name, conditions)

        async def check_condition(cond: str) -> tuple[str, bool, str]:
            formatted = f"The message should {cond}"
            result, reasoning = await self._suite.nlp_test(self.message, formatted)
            # Notify listener of individual condition result
            if self._listener and self._test_name:
                await self._listener.on_condition_result(self._test_name, cond, result)
            return cond, result, reasoning

        results = await safe_gather(*[check_condition(c) for c in conditions])

        failures: List[tuple[str, str]] = []
        for cond, passed, reasoning in results:
            if not passed:
                failures.append((cond, reasoning))

        if failures:
            failure_details = "\n".join(
                f"  - '{cond}': {reasoning}" for cond, reasoning in failures
            )
            raise AssertionError(
                f"Response assertion failed:\n"
                f"Actual message: {self.message!r}\n"
                f"Failed conditions:\n{failure_details}"
            )
