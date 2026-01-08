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
import re
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
)

from parlant.testing.session import UnfoldResults
from parlant.testing.suite import Scenario, Suite, reset_test_context, set_test_context


class TestStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Result of a single test execution."""

    name: str
    status: TestStatus
    duration_ms: float
    error: Optional[str] = None
    traceback: Optional[str] = None
    expected: Optional[str] = None
    actual: Optional[str] = None
    reasoning: Optional[str] = None


@dataclass
class ScenarioResult:
    """Result of a scenario (may contain multiple test results from unfold)."""

    name: str
    tests: List[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(t.status == TestStatus.PASSED for t in self.tests)


@dataclass
class TestReport:
    """Complete report of test execution."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_ms: float = 0
    scenarios: List[ScenarioResult] = field(default_factory=list)


class TestEventListener(Protocol):
    """Protocol for test event listeners (used by Rich UI)."""

    async def on_suite_start(self, suite_name: str, total_tests: int) -> None: ...
    async def on_test_start(self, test_name: str) -> None: ...
    async def on_message_sent(self, test_name: str, role: str, content: str) -> None: ...
    async def on_waiting_for_agent(self, test_name: str) -> None: ...
    async def on_message_received(self, test_name: str, role: str, content: str) -> None: ...
    async def on_evaluating(self, test_name: str, conditions: List[str]) -> None: ...
    async def on_condition_result(self, test_name: str, condition: str, passed: bool) -> None: ...
    async def on_test_passed(self, test_name: str, duration_ms: float) -> None: ...
    async def on_test_failed(
        self, test_name: str, duration_ms: float, error: str, details: Optional[Dict[str, Any]]
    ) -> None: ...
    async def on_suite_end(self, report: TestReport) -> None: ...


class NullEventListener:
    """Default no-op event listener."""

    async def on_suite_start(self, suite_name: str, total_tests: int) -> None:
        pass

    async def on_test_start(self, test_name: str) -> None:
        pass

    async def on_message_sent(self, test_name: str, role: str, content: str) -> None:
        pass

    async def on_waiting_for_agent(self, test_name: str) -> None:
        pass

    async def on_message_received(self, test_name: str, role: str, content: str) -> None:
        pass

    async def on_evaluating(self, test_name: str, conditions: List[str]) -> None:
        pass

    async def on_condition_result(self, test_name: str, condition: str, passed: bool) -> None:
        pass

    async def on_test_passed(self, test_name: str, duration_ms: float) -> None:
        pass

    async def on_test_failed(
        self, test_name: str, duration_ms: float, error: str, details: Optional[Dict[str, Any]]
    ) -> None:
        pass

    async def on_suite_end(self, report: TestReport) -> None:
        pass


@dataclass
class ExpandedTest:
    """A single test to execute (may be from scenario or unfold expansion)."""

    name: str
    suite: Suite
    scenario: Scenario
    repetition: int = 1
    total_repetitions: int = 1


class TestRunner:
    """Executes test scenarios with parallel support."""

    def __init__(
        self,
        listener: Optional[TestEventListener] = None,
    ) -> None:
        self._listener = listener or NullEventListener()

    async def run(
        self,
        suites: List[Suite],
        parallel: int = 1,
        pattern: Optional[str] = None,
        fail_fast: bool = False,
    ) -> TestReport:
        """Run all scenarios from the given suites.

        Args:
            suites: List of test suites to run.
            parallel: Number of tests to run concurrently.
            pattern: Regex pattern to filter scenario names.
            fail_fast: Stop on first failure.

        Returns:
            TestReport with all results.
        """
        report = TestReport()
        start_time = time.time()

        # Collect and expand all tests
        expanded_tests = self._expand_tests(suites, pattern)
        report.total = len(expanded_tests)

        await self._listener.on_suite_start("parlant-test", report.total)

        # Run before_all hooks for each suite
        suite_set = {t.suite for t in expanded_tests}
        for suite in suite_set:
            for hook in suite.get_hooks().before_all:
                await hook()

        # Create semaphore for parallelism
        semaphore = asyncio.Semaphore(parallel)
        failed = False

        async def run_test(test: ExpandedTest) -> List[TestResult]:
            nonlocal failed
            if fail_fast and failed:
                return [
                    TestResult(
                        name=test.name,
                        status=TestStatus.SKIPPED,
                        duration_ms=0,
                        error="Skipped due to fail-fast",
                    )
                ]

            async with semaphore:
                return await self._run_single_test(test)

        # Run tests
        if parallel == 1:
            # Sequential execution
            for test in expanded_tests:
                if fail_fast and failed:
                    results = [
                        TestResult(
                            name=test.name,
                            status=TestStatus.SKIPPED,
                            duration_ms=0,
                            error="Skipped due to fail-fast",
                        )
                    ]
                else:
                    results = await self._run_single_test(test)
                    if any(r.status == TestStatus.FAILED for r in results):
                        failed = True

                for result in results:
                    self._add_result_to_report(report, test, result)
        else:
            # Parallel execution
            tasks = [asyncio.create_task(run_test(test)) for test in expanded_tests]
            all_results = await asyncio.gather(*tasks)

            for test, results in zip(expanded_tests, all_results):
                if any(r.status == TestStatus.FAILED for r in results):
                    failed = True
                for result in results:
                    self._add_result_to_report(report, test, result)

        # Run after_all hooks
        for suite in suite_set:
            for hook in suite.get_hooks().after_all:
                await hook()

        # Recalculate total from actual results (unfold can expand tests)
        report.total = report.passed + report.failed + report.skipped + report.errors

        report.duration_ms = (time.time() - start_time) * 1000
        await self._listener.on_suite_end(report)

        return report

    def _expand_tests(self, suites: List[Suite], pattern: Optional[str]) -> List[ExpandedTest]:
        """Expand scenarios with repetitions into individual tests."""
        expanded: List[ExpandedTest] = []
        compiled_pattern = re.compile(pattern) if pattern else None

        for suite in suites:
            for scenario in suite.get_scenarios():
                # Filter by pattern
                if compiled_pattern and not compiled_pattern.search(scenario.name):
                    continue

                # Expand repetitions
                for rep in range(1, scenario.repetitions + 1):
                    name = scenario.name
                    if scenario.repetitions > 1:
                        name = f"{scenario.name}[rep_{rep}/{scenario.repetitions}]"

                    expanded.append(
                        ExpandedTest(
                            name=name,
                            suite=suite,
                            scenario=scenario,
                            repetition=rep,
                            total_repetitions=scenario.repetitions,
                        )
                    )

        return expanded

    async def _run_single_test(self, test: ExpandedTest) -> List[TestResult]:
        """Run a single test and return the result(s).

        Returns a list because unfold() can expand into multiple sub-tests.
        """
        start_time = time.time()
        await self._listener.on_test_start(test.name)

        # Set test context using contextvars for coroutine-safe access
        context_tokens = set_test_context(test.name, self._listener)

        try:
            # Run before_each hooks
            for hook in test.suite.get_hooks().before_each:
                try:
                    await hook(test.name)
                except Exception as e:
                    duration = (time.time() - start_time) * 1000
                    result = TestResult(
                        name=test.name,
                        status=TestStatus.FAILED,
                        duration_ms=duration,
                        error=f"before_each hook failed: {e}",
                        traceback=traceback.format_exc(),
                    )
                    await self._run_after_each(test, result)
                    return [result]

            # Run the actual test
            try:
                await test.scenario.func()
                duration = (time.time() - start_time) * 1000
                result = TestResult(
                    name=test.name,
                    status=TestStatus.PASSED,
                    duration_ms=duration,
                )
                await self._listener.on_test_passed(test.name, duration)
                results = [result]

            except UnfoldResults as unfold:
                # Convert sub-test results to TestResults
                # Note: on_test_start and on_test_passed/failed were already called
                # by unfold() for real-time UI updates
                results = []
                for sub in unfold.results:
                    if sub.passed:
                        result = TestResult(
                            name=sub.name,
                            status=TestStatus.PASSED,
                            duration_ms=sub.duration_ms,
                        )
                    else:
                        result = TestResult(
                            name=sub.name,
                            status=TestStatus.FAILED,
                            duration_ms=sub.duration_ms,
                            error=sub.error,
                            actual=sub.actual,
                            expected=sub.expected,
                        )
                    results.append(result)

            except AssertionError as e:
                duration = (time.time() - start_time) * 1000
                error_str = str(e)
                result = TestResult(
                    name=test.name,
                    status=TestStatus.FAILED,
                    duration_ms=duration,
                    error=error_str,
                    traceback=traceback.format_exc(),
                )
                # Parse assertion error for details
                if "Actual message:" in error_str:
                    result.actual = self._extract_between(
                        error_str, "Actual message:", "Failed conditions:"
                    )
                if "Failed conditions:" in error_str:
                    result.expected = self._extract_after(error_str, "Failed conditions:")

                await self._listener.on_test_failed(
                    test.name,
                    duration,
                    error_str,
                    {"expected": result.expected, "actual": result.actual},
                )
                results = [result]

            except TimeoutError as e:
                duration = (time.time() - start_time) * 1000
                result = TestResult(
                    name=test.name,
                    status=TestStatus.FAILED,
                    duration_ms=duration,
                    error=str(e),
                    traceback=traceback.format_exc(),
                )
                await self._listener.on_test_failed(test.name, duration, str(e), None)
                results = [result]

            except Exception as e:
                duration = (time.time() - start_time) * 1000
                result = TestResult(
                    name=test.name,
                    status=TestStatus.ERROR,
                    duration_ms=duration,
                    error=str(e),
                    traceback=traceback.format_exc(),
                )
                await self._listener.on_test_failed(test.name, duration, str(e), None)
                results = [result]

            # Run after_each hooks (use first result for pass/fail status)
            await self._run_after_each(test, results[0] if results else None)

            return results
        finally:
            # Always reset test context
            reset_test_context(context_tokens)

    async def _run_after_each(self, test: ExpandedTest, result: Optional[TestResult]) -> None:
        """Run after_each hooks."""
        error: Optional[str]
        if result is None:
            passed = False
            error = "No result"
        else:
            passed = result.status == TestStatus.PASSED
            error = result.error if not passed else None

        for hook in test.suite.get_hooks().after_each:
            try:
                await hook(test.name, passed, error)
            except Exception:
                pass  # Log but don't affect test result

    def _add_result_to_report(
        self, report: TestReport, test: ExpandedTest, result: TestResult
    ) -> None:
        """Add a test result to the report."""
        # Find or create scenario result
        scenario_result = next((s for s in report.scenarios if s.name == test.scenario.name), None)
        if not scenario_result:
            scenario_result = ScenarioResult(name=test.scenario.name)
            report.scenarios.append(scenario_result)

        scenario_result.tests.append(result)

        # Update counts
        if result.status == TestStatus.PASSED:
            report.passed += 1
        elif result.status == TestStatus.FAILED:
            report.failed += 1
        elif result.status == TestStatus.SKIPPED:
            report.skipped += 1
        elif result.status == TestStatus.ERROR:
            report.errors += 1

    @staticmethod
    def _extract_between(text: str, start: str, end: str) -> Optional[str]:
        """Extract text between two markers."""
        try:
            s = text.index(start) + len(start)
            e = text.index(end, s)
            return text[s:e].strip()
        except ValueError:
            return None

    @staticmethod
    def _extract_after(text: str, marker: str) -> Optional[str]:
        """Extract text after a marker."""
        try:
            s = text.index(marker) + len(marker)
            return text[s:].strip()
        except ValueError:
            return None
