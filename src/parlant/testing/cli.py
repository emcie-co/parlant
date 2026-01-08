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
import sys
from pathlib import Path
from typing import Optional, Tuple

import click
from rich.console import Console

from parlant.testing.discovery import DiscoveryError, discover_suites, list_tests
from parlant.testing.reporter import (
    RichReporter,
    generate_json_report,
    print_summary,
)
from parlant.testing.runner import TestReport, TestRunner


@click.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--pattern",
    "-p",
    default=None,
    help="Regex pattern to filter scenario names",
)
@click.option(
    "--parallel",
    "-n",
    default=1,
    type=int,
    help="Number of tests to run concurrently",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Write JSON report to file",
)
@click.option(
    "--fail-fast",
    "-x",
    is_flag=True,
    default=False,
    help="Stop on first failure",
)
@click.option(
    "--list",
    "list_only",
    is_flag=True,
    default=False,
    help="Show discovered tests without running",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level (-v, -vv, -vvv)",
)
def main(
    paths: Tuple[str, ...],
    pattern: Optional[str],
    parallel: int,
    output: Optional[str],
    fail_fast: bool,
    list_only: bool,
    verbose: int,
) -> None:
    """Run Parlant agent tests.

    PATHS: One or more files or directories containing test files.

    Examples:

        parlant-test tests/

        parlant-test tests/test_greeting.py --pattern "greet"

        parlant-test tests/ --parallel 4 --output results.json
    """
    console = Console()

    exit_code = 0
    try:
        # Discover test suites
        console.print(f"[dim]Discovering tests in: {', '.join(paths)}[/dim]")
        suites = discover_suites(list(paths))

        if not suites:
            console.print("[yellow]No test suites found.[/yellow]")
            sys.exit(0)

        # Count scenarios
        total_scenarios = sum(len(s.get_scenarios()) for s in suites)
        console.print(f"[dim]Found {len(suites)} suite(s) with {total_scenarios} scenario(s)[/dim]")

        # List mode
        if list_only:
            tests = list_tests(suites)
            if pattern:
                import re

                compiled = re.compile(pattern)
                tests = [t for t in tests if compiled.search(t)]

            console.print()
            console.print("[bold]Discovered tests:[/bold]")
            for test in tests:
                # Escape square brackets for Rich
                escaped = test.replace("[", "\\[")
                console.print(f"  {escaped}")
            console.print()
            console.print(f"Total: {len(tests)} test(s)")
            sys.exit(0)

        # Create reporter and runner
        reporter = RichReporter(
            console=console,
            panel_count=parallel,
            panel_height=12 if parallel > 2 else 15,
        )

        runner = TestRunner(listener=reporter)

        # Run tests
        console.print()
        reporter.start_live()

        async def run_and_cleanup() -> TestReport:
            try:
                return await runner.run(
                    suites=suites,
                    parallel=parallel,
                    pattern=pattern,
                    fail_fast=fail_fast,
                )
            finally:
                # Cleanup sessions from all suites
                has_sessions = any(s.has_sessions_to_cleanup() for s in suites)
                if has_sessions:
                    # Wait 10 seconds with countdown spinner
                    cleanup_seconds = 10
                    with console.status("") as status:
                        for remaining in range(cleanup_seconds, 0, -1):
                            status.update(f"[dim]Cleaning up sessions... ({remaining}s)[/dim]")
                            await asyncio.sleep(1)
                        status.update("[dim]Deleting sessions...[/dim]")
                        # Delete sessions from all suites
                        for suite in suites:
                            await suite.delete_queued_sessions()

        try:
            report = asyncio.run(run_and_cleanup())
        finally:
            reporter.stop_live()

        # Print summary
        print_summary(console, report)

        # Write JSON output
        if output:
            json_report = generate_json_report(report)
            Path(output).write_text(json_report)
            console.print(f"[dim]JSON report written to: {output}[/dim]")

        exit_code = 0 if report.failed == 0 else 1

    except DiscoveryError as e:
        console.print(f"[bold red]Discovery error:[/bold red] {e}")
        exit_code = 2
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        exit_code = 130
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose >= 2:
            import traceback

            console.print(traceback.format_exc())
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
