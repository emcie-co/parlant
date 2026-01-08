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

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

from parlant.client.types.event import Event
from parlant.testing.builder import InteractionBuilder, PrefabEvent
from parlant.testing.response import Response
from parlant.testing.steps import AgentMessage, CustomerMessage, Step

if TYPE_CHECKING:
    from parlant.testing.suite import Suite


@dataclass
class SubTestResult:
    """Result of a single sub-test within unfold()."""

    name: str  # e.g., "test_conversation[step_1]"
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    actual: Optional[str] = None
    expected: Optional[str] = None


class UnfoldResults(Exception):
    """Exception raised by unfold() to communicate sub-test results to runner."""

    def __init__(self, results: List[SubTestResult]) -> None:
        self.results = results
        super().__init__(f"unfold() completed with {len(results)} sub-tests")


class Session:
    """Context manager for managing a test session with a Parlant agent.

    Creates a session on the server, provides methods to send messages
    and receive responses, and cleans up on exit if transient.
    """

    def __init__(
        self,
        suite: "Suite",
        agent_id: str,
        customer_id: Optional[str],
        transient: bool = True,
        test_name: Optional[str] = None,
        listener: Optional[Any] = None,
    ) -> None:
        self._suite = suite
        self._agent_id = agent_id
        self._customer_id = customer_id
        self._transient = transient
        self._session_id: Optional[str] = None
        self._current_offset: int = 0
        # Test context - passed in for parallel safety
        self._test_name = test_name
        self._listener = listener

    @property
    def id(self) -> str:
        """The server-side session ID."""
        if self._session_id is None:
            raise RuntimeError("Session not initialized. Use 'async with' context.")
        return self._session_id

    async def __aenter__(self) -> "Session":
        """Create the session on the server."""
        client = await self._suite._get_client()
        session = await client.sessions.create(
            agent_id=self._agent_id,
            customer_id=self._customer_id,
            allow_greeting=False,
            metadata={"__emcie__": {"createdById": "TEST"}},
        )
        self._session_id = session.id
        self._current_offset = 0
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Queue session for deletion if transient."""
        if self._transient and self._session_id:
            self._suite._queue_session_for_deletion(self._session_id)

    async def send(
        self,
        message: str,
        timeout: Optional[int] = None,
        _skip_ui_notification: bool = False,
    ) -> Response:
        """Send a customer message and wait for agent response.

        Args:
            message: The customer message to send.
            timeout: Timeout in seconds. Defaults to suite's response_timeout.
            _skip_ui_notification: Internal flag to skip customer message UI notification
                (used when message was already shown via add_events).

        Returns:
            Response object wrapping all agent events for this turn.

        Raises:
            TimeoutError: If agent doesn't respond within timeout.
        """
        if self._session_id is None:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        effective_timeout = timeout if timeout is not None else self._suite.response_timeout
        client = await self._suite._get_client()

        test_name = self._test_name
        listener = self._listener

        # Notify listener about customer message (unless already shown)
        if listener and test_name and not _skip_ui_notification:
            await listener.on_message_sent(test_name, "Customer", message)

        # Send customer message
        event = await client.sessions.create_event(
            session_id=self._session_id,
            kind="message",
            source="customer",
            message=message,
        )

        trace_id = event.trace_id
        self._current_offset = event.offset

        # Notify listener that we're waiting for agent (unless already shown)
        if listener and test_name and not _skip_ui_notification:
            await listener.on_waiting_for_agent(test_name)

        # Wait for agent response with "ready" status
        response_events = await self._wait_for_response(
            trace_id=trace_id,
            timeout=effective_timeout,
        )

        # Notify listener about agent response
        response = Response(
            events=response_events,
            trace_id=trace_id,
            suite=self._suite,
            test_name=test_name,
            listener=listener,
        )
        if listener and test_name and response.message:
            await listener.on_message_received(test_name, "Agent", response.message)

        return response

    async def _wait_for_response(
        self,
        trace_id: str,
        timeout: int,
    ) -> List[Event]:
        """Wait for agent response events until ready status is received."""
        if self._session_id is None:
            raise RuntimeError("Session not initialized.")

        client = await self._suite._get_client()
        collected_events: List[Event] = []
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            remaining = timeout - elapsed

            if remaining <= 0:
                raise TimeoutError(f"Timeout waiting for agent response after {timeout}s")

            try:
                # Use short polling intervals (100ms) to allow UI to refresh
                wait_ms = min(int(remaining * 1000), 100)

                # Run list_events in a task so we can yield control periodically
                task = asyncio.create_task(
                    client.sessions.list_events(
                        session_id=self._session_id,
                        min_offset=self._current_offset + 1,
                        source="ai_agent",
                        trace_id=trace_id,
                        wait_for_data=wait_ms,
                    )
                )

                # Wait for task with periodic yields to allow UI refresh
                while not task.done():
                    await asyncio.sleep(0.05)  # Yield every 50ms for UI refresh

                events = await task

                for event in events:
                    if event.offset > self._current_offset:
                        self._current_offset = event.offset
                        collected_events.append(event)

                        # Check for ready status with stage="completed"
                        if event.kind == "status":
                            data = event.data
                            if isinstance(data, dict) and data.get("status") == "ready":
                                inner_data = data.get("data", {})
                                if (
                                    isinstance(inner_data, dict)
                                    and inner_data.get("stage") == "completed"
                                ):
                                    return collected_events

            except Exception as e:
                # Handle timeout from long polling (504 Gateway Timeout)
                if "504" in str(e) or "timeout" in str(e).lower():
                    continue
                raise

        return collected_events

    async def add_events(
        self,
        events: List[PrefabEvent],
        next_customer_message: Optional[str] = None,
    ) -> None:
        """Add prefab events to the session.

        Creates real events on the server with the session's agent/customer IDs.
        Also notifies the listener about message events for UI display.

        Args:
            events: List of PrefabEvent objects to create.
            next_customer_message: Optional upcoming customer message to show in UI
                immediately after prefab history (before slow server calls).
        """
        if self._session_id is None:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        client = await self._suite._get_client()
        test_name = self._test_name
        listener = self._listener

        # Notify listener about all message events FIRST for immediate UI display
        if listener and test_name:
            for prefab in events:
                if prefab.kind == "message":
                    message = prefab.data.get("message", "")
                    if prefab.source == "customer":
                        await listener.on_message_sent(test_name, "Customer", message)
                    else:
                        # Agent messages (from prefab history)
                        await listener.on_message_received(test_name, "Agent", message)

            # Show upcoming customer message immediately (before slow server calls)
            if next_customer_message:
                await listener.on_message_sent(test_name, "Customer", next_customer_message)
                await listener.on_waiting_for_agent(test_name)

        # Switch to manual mode to prevent trigger_processing on prefab events
        await client.sessions.update(session_id=self._session_id, mode="manual")

        # Then create events on server (this can be slow)
        for prefab in events:
            event = await client.sessions.create_event(
                session_id=self._session_id,
                kind=prefab.kind,
                source=prefab.source,
                message=prefab.data.get("message") if prefab.kind == "message" else None,
                status=prefab.data.get("status") if prefab.kind == "status" else None,
                data=prefab.data if prefab.kind not in ("message", "status") else None,
            )
            self._current_offset = max(self._current_offset, event.offset)

        # Switch back to auto mode so the next customer message triggers processing
        await client.sessions.update(session_id=self._session_id, mode="auto")

    async def unfold(self, steps: Sequence[Step]) -> None:
        """Execute a multi-step conversation test with parallel sub-tests.

        Each AgentMessage generates a separate sub-test that runs in parallel:
        1. Creates NEW session with prefab history of all prior steps
        2. Sends the CustomerMessage before this AgentMessage
        3. Waits for real agent response
        4. Asserts the 'should' condition(s)

        All sub-tests run concurrently and report results independently.
        Raises UnfoldResults with all sub-test results for the runner to process.

        Args:
            steps: Sequence of CustomerMessage and AgentMessage steps.
        """
        # Only AgentMessages with should conditions create tests
        agent_indices = [
            i
            for i, step in enumerate(steps)
            if isinstance(step, AgentMessage) and step.should is not None
        ]

        if not agent_indices:
            return

        base_name = self._test_name or "unfold"
        total_steps = len(agent_indices)

        async def run_subtest(
            step_num: int,
            agent_idx: int,
            customer_idx: int,
            customer_step: CustomerMessage,
            agent_step: AgentMessage,
        ) -> SubTestResult:
            """Run a single sub-test."""
            subtest_name = f"{base_name} ({step_num}/{total_steps})"
            start_time = time.time()

            try:
                # Create NEW session for this test
                async with Session(
                    suite=self._suite,
                    agent_id=self._agent_id,
                    customer_id=self._customer_id,
                    transient=True,
                    test_name=subtest_name,
                    listener=self._listener,
                ) as test_session:
                    # Add prefab history (all steps before customer_idx)
                    has_prefab = customer_idx > 0
                    if has_prefab:
                        prefab_steps = list(steps[:customer_idx])
                        builder = InteractionBuilder.from_steps(prefab_steps)
                        await test_session.add_events(
                            builder.build(),
                            next_customer_message=customer_step.message,
                        )

                    # Send customer message and get response
                    # Skip UI notification if we already showed it in add_events
                    response = await test_session.send(
                        customer_step.message,
                        _skip_ui_notification=has_prefab,
                    )

                    # Assert (should is guaranteed non-None by agent_indices filter)
                    assert agent_step.should is not None
                    await response.should(agent_step.should)

                    # Success - notify listener immediately for real-time UI update
                    duration_ms = (time.time() - start_time) * 1000
                    if self._listener:
                        await self._listener.on_test_passed(subtest_name, duration_ms)
                    return SubTestResult(
                        name=subtest_name,
                        passed=True,
                        duration_ms=duration_ms,
                    )

            except AssertionError as e:
                duration_ms = (time.time() - start_time) * 1000
                error_str = str(e)
                result = SubTestResult(
                    name=subtest_name,
                    passed=False,
                    duration_ms=duration_ms,
                    error=error_str,
                )
                # Extract actual/expected from error message
                if "Actual message:" in error_str:
                    try:
                        start_idx = error_str.index("Actual message:") + len("Actual message:")
                        end_idx = error_str.index("Failed conditions:", start_idx)
                        result.actual = error_str[start_idx:end_idx].strip().strip("'")
                    except ValueError:
                        pass
                if "Failed conditions:" in error_str:
                    try:
                        start_idx = error_str.index("Failed conditions:") + len(
                            "Failed conditions:"
                        )
                        result.expected = error_str[start_idx:].strip()
                    except ValueError:
                        pass
                # Notify listener immediately for real-time UI update
                if self._listener:
                    await self._listener.on_test_failed(
                        subtest_name,
                        duration_ms,
                        error_str,
                        {"expected": result.expected, "actual": result.actual},
                    )
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                error_str = str(e)
                # Notify listener immediately for real-time UI update
                if self._listener:
                    await self._listener.on_test_failed(subtest_name, duration_ms, error_str, None)
                return SubTestResult(
                    name=subtest_name,
                    passed=False,
                    duration_ms=duration_ms,
                    error=error_str,
                )

        # Build list of sub-tests to run
        subtests: List[tuple[int, int, int, CustomerMessage, AgentMessage]] = []
        for step_num, agent_idx in enumerate(agent_indices, start=1):
            # Find preceding CustomerMessage
            customer_idx = agent_idx - 1
            while customer_idx >= 0 and not isinstance(steps[customer_idx], CustomerMessage):
                customer_idx -= 1

            if customer_idx < 0:
                raise ValueError(
                    f"AgentMessage at index {agent_idx} has no preceding CustomerMessage"
                )

            customer_step = steps[customer_idx]
            assert isinstance(customer_step, CustomerMessage)
            agent_step = steps[agent_idx]
            assert isinstance(agent_step, AgentMessage)

            subtests.append((step_num, agent_idx, customer_idx, customer_step, agent_step))

        # Run sub-tests sequentially
        results: List[SubTestResult] = []
        for subtest in subtests:
            step_num = subtest[0]
            subtest_name = f"{base_name} ({step_num}/{total_steps})"
            # Notify listener about this sub-test starting
            if self._listener:
                await self._listener.on_test_start(subtest_name)
            # Run the sub-test
            result = await run_subtest(*subtest)
            results.append(result)

        # Raise UnfoldResults so runner can process sub-tests
        raise UnfoldResults(list(results))
