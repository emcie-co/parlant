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
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rich.console import Console, Group

from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from parlant.testing.runner import TestReport, TestStatus


@dataclass
class ConversationMessage:
    """A message in a test conversation."""

    role: str  # "Customer" or "Agent"
    content: str


@dataclass
class TestPanel:
    """State for a test panel in the Rich UI."""

    test_name: str
    messages: List[ConversationMessage] = field(default_factory=list)
    waiting_for_agent: bool = False
    waiting_since: Optional[float] = None  # timestamp when waiting started
    evaluating: bool = False
    # Conditions: list of (condition_text, status) where status is "pending", "passed", "failed"
    evaluating_conditions: List[tuple[str, str]] = field(default_factory=list)
    status: Optional[str] = None  # "PASSED", "FAILED: ...", or None (running)
    status_style: str = "white"  # "green", "red", "white"
    flash_until: float = 0  # Timestamp until which to flash
    spinner: Spinner = field(default_factory=lambda: Spinner("point"))
    eval_spinner: Spinner = field(default_factory=lambda: Spinner("star2"))


class RichReporter:
    """Rich terminal UI reporter for test execution.

    Displays N panels for N parallel tests, showing conversations
    unfolding in real-time with pass/fail indicators.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        panel_count: int = 1,
        panel_height: int = 15,
    ) -> None:
        self._console = console or Console()
        self._panel_count = panel_count
        self._panel_height = panel_height
        self._panels: Dict[str, TestPanel] = {}
        self._active_panels: List[str] = []  # Test names in active panel slots
        self._test_queue: List[str] = []
        self._live: Optional[Live] = None
        self._refresh_task: Optional[asyncio.Task[None]] = None
        self._total_tests: int = 0
        self._started_tests: int = 0  # Dynamically tracks tests as they start
        self._completed_tests: int = 0

    async def on_suite_start(self, suite_name: str, total_tests: int) -> None:
        """Called when test suite starts."""
        self._total_tests = total_tests
        self._started_tests = 0
        self._completed_tests = 0
        self._panels.clear()
        self._active_panels = [""] * self._panel_count
        self._test_queue.clear()
        # Start background refresh task for spinner animation
        if self._refresh_task is None:
            self._refresh_task = asyncio.create_task(self._background_refresh())

    async def on_test_start(self, test_name: str) -> None:
        """Called when a test starts."""
        self._started_tests += 1
        self._panels[test_name] = TestPanel(test_name=test_name)

        # Check if this is a sub-test (e.g., "test_conversation (1/3)")
        # If so, remove the parent panel from active slots
        if " (" in test_name and "/" in test_name:
            parent_name = test_name.rsplit(" (", 1)[0]
            for i, active in enumerate(self._active_panels):
                if active == parent_name:
                    # Remove parent panel, this slot becomes available
                    self._active_panels[i] = ""
                    if parent_name in self._panels:
                        del self._panels[parent_name]
                    break

        # Find an empty panel slot or queue the test
        for i, active in enumerate(self._active_panels):
            if not active:
                self._active_panels[i] = test_name
                break
        else:
            self._test_queue.append(test_name)

        self._refresh_display()

    async def on_message_sent(self, test_name: str, role: str, content: str) -> None:
        """Called when a message is sent in a test."""
        if test_name in self._panels:
            self._panels[test_name].messages.append(ConversationMessage(role=role, content=content))
        self._refresh_display()

    async def on_waiting_for_agent(self, test_name: str) -> None:
        """Called when waiting for agent response."""
        if test_name in self._panels:
            self._panels[test_name].waiting_for_agent = True
            self._panels[test_name].waiting_since = time.time()
            self._refresh_display()

    async def on_message_received(self, test_name: str, role: str, content: str) -> None:
        """Called when a message is received."""
        if test_name in self._panels:
            panel = self._panels[test_name]
            panel.waiting_for_agent = False
            panel.messages.append(ConversationMessage(role=role, content=content))
            self._refresh_display()

    async def on_evaluating(self, test_name: str, conditions: List[str]) -> None:
        """Called when evaluating the response."""
        if test_name in self._panels:
            panel = self._panels[test_name]
            panel.evaluating = True
            panel.evaluating_conditions = [(c, "pending") for c in conditions]
            self._refresh_display()

    async def on_condition_result(self, test_name: str, condition: str, passed: bool) -> None:
        """Called when a single condition evaluation completes."""
        if test_name in self._panels:
            panel = self._panels[test_name]
            # Update the status of the matching condition
            status = "passed" if passed else "failed"
            panel.evaluating_conditions = [
                (c, status) if c == condition else (c, s) for c, s in panel.evaluating_conditions
            ]
            self._refresh_display()

    async def on_test_passed(self, test_name: str, duration_ms: float) -> None:
        """Called when a test passes."""
        if test_name in self._panels:
            panel = self._panels[test_name]
            panel.status = "PASSED"
            panel.status_style = "green"
            panel.flash_until = asyncio.get_event_loop().time() + 0.5  # 500ms flash

        self._completed_tests += 1
        self._refresh_display()

        # Wait for flash duration then move to next test
        await asyncio.sleep(0.5)
        self._move_to_next_test(test_name)

    async def on_test_failed(
        self, test_name: str, duration_ms: float, error: str, details: Optional[Dict[str, Any]]
    ) -> None:
        """Called when a test fails."""
        if test_name in self._panels:
            panel = self._panels[test_name]
            # Truncate error for display
            short_error = error[:80] + "..." if len(error) > 80 else error
            panel.status = f"FAILED: {short_error}"
            panel.status_style = "red"
            panel.flash_until = asyncio.get_event_loop().time() + 3.0  # 3000ms flash

        self._completed_tests += 1
        self._refresh_display()

        # Wait for flash duration then move to next test
        await asyncio.sleep(3.0)
        self._move_to_next_test(test_name)

    async def on_suite_end(self, report: TestReport) -> None:
        """Called when test suite ends."""
        # Sync counts with final report (handles unfold parent test being replaced)
        self._completed_tests = report.total
        self._started_tests = report.total
        self._refresh_display()

    def _move_to_next_test(self, completed_test: str) -> None:
        """Move panel slot to next queued test."""
        # Find the slot with the completed test
        for i, active in enumerate(self._active_panels):
            if active == completed_test:
                # Get next from queue or clear slot
                if self._test_queue:
                    self._active_panels[i] = self._test_queue.pop(0)
                else:
                    self._active_panels[i] = ""
                break

        self._refresh_display()

    def _refresh_display(self) -> None:
        """Refresh the Rich display."""
        if self._live:
            self._live.update(self._build_display())

    def _build_display(self) -> Group:
        """Build the Rich display group with 2-column layout."""
        panels: List[Panel] = []

        for test_name in self._active_panels:
            if test_name and test_name in self._panels:
                panels.append(self._build_panel(self._panels[test_name]))

        # Arrange panels in 2-column grid
        grid: Group | Text | Table
        if len(panels) <= 1:
            # Single panel - no table needed
            grid = Group(*panels) if panels else Text("")
        else:
            # Create a table with 2 columns
            table = Table.grid(expand=True)
            table.add_column(ratio=1)
            table.add_column(ratio=1)

            # Add panels in rows of 2
            for i in range(0, len(panels), 2):
                if i + 1 < len(panels):
                    table.add_row(panels[i], panels[i + 1])
                else:
                    # Odd number of panels - last one gets full row
                    table.add_row(panels[i], Text(""))

            grid = table

        # Add progress line
        running = self._started_tests - self._completed_tests
        if running > 0:
            progress = Text(
                f"\nProgress: {self._completed_tests} completed, {running} running",
                style="bold",
            )
        else:
            progress = Text(
                f"\nProgress: {self._completed_tests} tests completed",
                style="bold",
            )

        return Group(grid, progress)

    def _build_panel(self, panel: TestPanel) -> Panel:
        """Build a single test panel."""
        # Build conversation content
        lines: List[Text] = []

        # Calculate space needed for evaluating conditions
        eval_lines_needed = 0
        if panel.evaluating and panel.evaluating_conditions:
            eval_lines_needed = len(panel.evaluating_conditions) + 1  # +1 for empty line

        # Reserve space for waiting spinner
        waiting_lines_needed = 1 if panel.waiting_for_agent else 0

        # Show last N messages that fit in panel (reserve space for eval/waiting)
        # When evaluating or waiting, be aggressive - only show last 3 messages to ensure status fits
        if panel.evaluating and panel.evaluating_conditions:
            max_messages = min(3, self._panel_height - 4 - eval_lines_needed)
        elif panel.waiting_for_agent:
            max_messages = min(3, self._panel_height - 4 - waiting_lines_needed)
        else:
            max_messages = self._panel_height - 4
        max_messages = max(1, max_messages)  # Always show at least 1 message
        visible_messages = panel.messages[-max_messages:]

        for msg in visible_messages:
            style = "white" if msg.role == "Customer" else "cyan"
            lines.append(Text(f"{msg.role}: ", style=f"bold {style}") + Text(msg.content))

        # Add waiting spinner with elapsed time
        if panel.waiting_for_agent:
            spinner_frame = panel.spinner.render(time.time())
            # Calculate elapsed time
            elapsed = ""
            if panel.waiting_since:
                elapsed_secs = time.time() - panel.waiting_since
                elapsed = f" {elapsed_secs:.1f}s"
            # spinner.render() returns Text, so we can concatenate directly
            agent_waiting = (
                Text("Agent: ", style="bold cyan") + spinner_frame + Text(elapsed, style="dim")
            )  # type: ignore[operator]
            lines.append(agent_waiting)

        # Add evaluating conditions with status-based colors
        if panel.evaluating and panel.evaluating_conditions:
            lines.append(Text(""))  # Empty line before conditions
            for cond, status in panel.evaluating_conditions:
                if status == "pending":
                    spinner_frame = panel.eval_spinner.render(time.time())
                    # spinner.render() returns Text, concatenation works
                    condition_line = spinner_frame + Text(f" It should {cond}", style="yellow")  # type: ignore[operator, assignment]
                elif status == "passed":
                    condition_line = Text("✓ ", style="green") + Text(
                        f"It should {cond}", style="green"
                    )
                else:  # failed
                    condition_line = Text("✗ ", style="red") + Text(
                        f"It should {cond}", style="red"
                    )
                lines.append(condition_line)  # type: ignore[arg-type]

        # Combine lines
        content = Text("\n").join(lines) if lines else Text("Starting...", style="dim")

        # Determine status text and style for title
        if panel.status == "PASSED":
            status_str = "PASSED"
            status_style = "green"
            border_style = "green"
        elif panel.status and panel.status.startswith("FAILED"):
            status_str = "FAILED"
            status_style = "red"
            border_style = "red"
        elif panel.evaluating:
            status_str = "EVALUATING"
            status_style = "yellow"
            border_style = "yellow"
        else:
            status_str = "RUNNING"
            status_style = "white"
            border_style = "white"

        title = f"[bold]{panel.test_name}[/bold] [{status_style}]({status_str})[/{status_style}]"

        return Panel(
            content,
            title=title,
            border_style=border_style,
            height=self._panel_height,
            width=100,
        )

    def start_live(self) -> None:
        """Start the Live display."""
        self._live = Live(
            self._build_display(),
            console=self._console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()

    async def _background_refresh(self) -> None:
        """Background task that periodically refreshes the display for animations."""
        while True:
            await asyncio.sleep(0.05)  # Refresh every 50ms
            if self._live:
                self._live.update(self._build_display())

    def stop_live(self) -> None:
        """Stop the Live display."""
        if self._refresh_task:
            self._refresh_task.cancel()
            self._refresh_task = None
        if self._live:
            self._live.stop()
            self._live = None


def print_summary(console: Console, report: TestReport) -> None:
    """Print the final summary to console."""
    console.print()
    console.print("[bold]parlant-test[/bold]")
    console.print()

    # Summary line
    console.print(
        f"Results: [bold green]{report.passed} passed[/bold green], "
        f"[bold red]{report.failed} failed[/bold red]"
    )
    console.print(f"Duration: {report.duration_ms / 1000:.2f}s")
    console.print()

    # Print failures
    if report.failed > 0:
        console.print("[bold red]Failures:[/bold red]")
        for scenario in report.scenarios:
            for test in scenario.tests:
                if test.status == TestStatus.FAILED:
                    console.print(f"  [bold]{test.name}[/bold]")
                    if test.expected:
                        console.print(f"    Expected: {test.expected}")
                    if test.actual:
                        console.print(f"    Actual: [red]{test.actual}[/red]")
                    if test.reasoning:
                        console.print(f"    Reasoning: {test.reasoning}")
                    if test.error and not test.expected:
                        console.print(f"    Error: {test.error}")
                    console.print()


def generate_json_report(report: TestReport) -> str:
    """Generate JSON report string."""
    scenarios_list: List[Dict[str, Any]] = []
    result: Dict[str, Any] = {
        "summary": {
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "skipped": report.skipped,
            "errors": report.errors,
            "duration_ms": report.duration_ms,
        },
        "scenarios": scenarios_list,
    }

    for scenario in report.scenarios:
        tests_list: List[Dict[str, Any]] = []
        scenario_data: Dict[str, Any] = {
            "name": scenario.name,
            "tests": tests_list,
        }

        for test in scenario.tests:
            test_data: Dict[str, Any] = {
                "name": test.name,
                "status": test.status.value,
                "duration_ms": test.duration_ms,
            }

            # Only include details for failed tests
            if test.status != TestStatus.PASSED:
                if test.expected:
                    test_data["expected"] = test.expected
                if test.actual:
                    test_data["actual"] = test.actual
                if test.reasoning:
                    test_data["reasoning"] = test.reasoning
                if test.error:
                    test_data["error"] = test.error
                if test.traceback:
                    test_data["traceback"] = test.traceback

            tests_list.append(test_data)

        scenarios_list.append(scenario_data)

    return json.dumps(result, indent=2)
