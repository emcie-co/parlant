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

"""``parlant dashboard deployment`` — a managed project's deployments.

Each deployment attempt builds one commit with the settings saved when it
started. Its output is read per channel: `build` (the build), `deployment`
(startup, readiness and going live) and `runtime` (the live app).
"""

from __future__ import annotations

from typing import Any, Mapping

import click
import httpx

from parlant.bin.dashboard.auth import authenticate
from parlant.bin.dashboard.client import deployments_path, refusal_code, request, require_ok
from parlant.bin.dashboard.commands import dashboard
from parlant.bin.dashboard.commands._shared import (
    print_fields,
    print_table,
    require_list,
    require_mapping,
    require_project,
    runtime_for,
    text,
    timestamp,
)
from parlant.bin.dashboard.deployments import overview
from parlant.bin.dashboard.output import write_stream_line
from parlant.bin.dashboard.runtime import DashboardRuntime
from parlant.bin.help import examples

LOG_POLL_INTERVAL_SECONDS = 1.0

CHANNELS = ("build", "deployment", "runtime")


@dashboard.group()
def deployment() -> None:
    """Inspect, follow and stop a managed project's deployments."""


@deployment.command(
    "list",
    epilog=examples(
        "parlant dashboard deployment list",
        "parlant dashboard deployment list --project PROJECT_ID",
    ),
)
@click.option("--cursor", help="Continue from where a previous page ended.")
@click.pass_context
def deployments_list(ctx: click.Context, cursor: str | None) -> None:
    """List deployments, newest first."""
    runtime = runtime_for(ctx)
    project = require_project(runtime)
    access_token = authenticate(runtime)

    serving = overview(runtime, project, access_token).get("servingId")
    page = require_mapping(
        require_ok(
            request(
                runtime,
                "GET",
                deployments_path(project, "attempts"),
                access_token=access_token,
                query={"cursor": cursor} if cursor else None,
            )
        ),
        "deployment list",
    )
    entries = require_list(page.get("items"), "deployment list")

    if not entries:
        click.echo("This project has no deployments yet.")
        return

    print_table(
        ("ID", "Status", "Commit", "Trigger", "Started"),
        (
            (
                text(e, "id"),
                _status(e) + (" (live)" if serving and text(e, "id") == serving else ""),
                _commit(e),
                text(e, "trigger"),
                timestamp(e, "startedAt"),
            )
            for e in entries
        ),
    )

    next_cursor = page.get("nextCursor")
    if isinstance(next_cursor, str) and next_cursor:
        click.echo(f"More: `parlant dashboard deployment list --cursor {next_cursor}`")


@deployment.command(
    "view",
    epilog=examples(
        "parlant dashboard deployment view ATTEMPT_ID",
    ),
)
@click.argument("attempt_id")
@click.pass_context
def deployments_view(ctx: click.Context, attempt_id: str) -> None:
    """Show one deployment.

    ATTEMPT_ID identifies one deployment attempt, as `deployment list` prints
    it. It is not the project id.
    """
    runtime = runtime_for(ctx)
    project = require_project(runtime)

    attempt = _attempt(runtime, project, authenticate(runtime), attempt_id)
    source = attempt.get("source")
    failure = attempt.get("failure")

    print_fields(
        [
            ("ID", text(attempt, "id")),
            ("STATUS", _status(attempt)),
            ("TRIGGER", text(attempt, "trigger")),
            ("SOURCE", f"{text(source, 'repository')}@{text(source, 'branch')}"),
            ("COMMIT", _commit(attempt, full=True)),
            ("STARTED", text(attempt, "startedAt")),
            ("FINISHED", text(attempt, "finishedAt")),
            (
                "FAILURE",
                f"{text(failure, 'kind')}: {text(failure, 'message')}"
                if isinstance(failure, Mapping)
                else "-",
            ),
        ]
    )


@deployment.command(
    "stop",
    epilog=examples(
        "parlant dashboard deployment stop ATTEMPT_ID",
        "parlant dashboard deployment stop ATTEMPT_ID --yes",
    ),
)
@click.argument("attempt_id")
@click.option("--yes", "-y", is_flag=True, help="Do not prompt for confirmation.")
@click.pass_context
def deployments_stop(ctx: click.Context, attempt_id: str, yes: bool) -> None:
    """Stop a running deployment. The live deployment keeps serving.

    ATTEMPT_ID is the attempt to stop, as `deployment list` prints it. Only an
    attempt still building or deploying can be stopped.
    """
    runtime = runtime_for(ctx)
    project = require_project(runtime)
    access_token = authenticate(runtime)

    if not yes:
        click.confirm(f"Stop deployment {attempt_id}?", abort=True)

    attempt = require_mapping(
        require_ok(
            request(
                runtime,
                "POST",
                deployments_path(project, "attempts", attempt_id, "stop"),
                access_token=access_token,
            )
        ),
        "deployment",
    )

    if text(attempt, "status") == "in-progress":
        click.echo(
            f"Stop requested for {attempt_id}; it is stopped once the platform confirms. "
            f"Check with `parlant dashboard deployment view {attempt_id}`."
        )
    else:
        click.echo(f"Deployment {attempt_id} is {_status(attempt)}; nothing left to stop.")


@deployment.command(
    "logs",
    epilog=examples(
        "parlant dashboard deployment logs ATTEMPT_ID",
        "parlant dashboard deployment logs ATTEMPT_ID --channel runtime",
        "parlant dashboard deployment logs ATTEMPT_ID --channel build --follow",
    ),
)
@click.argument("attempt_id")
@click.option(
    "--channel",
    type=click.Choice(CHANNELS),
    default="build",
    show_default=True,
    help="build: the build · deployment: startup and going live · runtime: the live app.",
)
@click.option("--follow", "-f", is_flag=True, help="Keep printing new output until it ends.")
@click.pass_context
def deployments_logs(ctx: click.Context, attempt_id: str, channel: str, follow: bool) -> None:
    """Print a deployment's retained output.

    ATTEMPT_ID is the attempt whose output to read, as `deployment list`
    prints it. Pick which output with --channel.
    """
    runtime = runtime_for(ctx)
    project = require_project(runtime)
    access_token = authenticate(runtime)

    try:
        _print_logs(runtime, project, access_token, attempt_id, channel, follow)
    except KeyboardInterrupt:
        click.echo("")


def _print_logs(
    runtime: DashboardRuntime,
    project: str,
    access_token: str,
    attempt_id: str,
    channel: str,
    follow: bool,
) -> None:
    cursor: str | None = None
    printed = -1
    first = True

    while True:
        response = _read_logs(runtime, project, access_token, attempt_id, channel, cursor)

        if response is None:
            click.echo("[Earlier output is no longer retained; continuing from the latest.]")
            cursor = None
            continue

        page = require_mapping(response, "log page")
        omitted = page.get("omittedLines")

        if first and isinstance(omitted, int) and omitted > 0:
            click.echo(f"[{omitted} earlier lines are no longer retained.]")
        first = False

        for line in require_list(page.get("lines"), "log page"):
            sequence = _sequence(line)
            if sequence is not None and sequence <= printed:
                continue
            printed = sequence if sequence is not None else printed
            write_stream_line(text(line, "text") if text(line, "text") != "-" else "")

        state = text(page, "state")
        next_cursor = page.get("cursor")
        cursor = next_cursor if isinstance(next_cursor, str) else cursor

        if state == "expired":
            click.echo("This output is no longer retained.")
            return

        if state == "complete":
            return

        if not follow:
            click.echo(f"[Still running. Add --follow to keep watching {attempt_id}.]")
            return

        runtime.sleep(LOG_POLL_INTERVAL_SECONDS)


def _read_logs(
    runtime: DashboardRuntime,
    project: str,
    access_token: str,
    attempt_id: str,
    channel: str,
    cursor: str | None,
) -> Any:
    """One log page, or None when the cursor fell behind retention."""
    response: httpx.Response = request(
        runtime,
        "GET",
        deployments_path(project, "attempts", attempt_id, "logs"),
        access_token=access_token,
        query={"channel": channel, **({"cursor": cursor} if cursor else {})},
    )

    if cursor is not None and refusal_code(response) == "cursor-expired":
        return None

    return require_ok(response)


def _sequence(line: Any) -> int | None:
    identifier = text(line, "id")
    _, _, tail = identifier.rpartition(":")

    return int(tail) if tail.isdigit() else None


def _attempt(
    runtime: DashboardRuntime, project: str, access_token: str, attempt_id: str
) -> Mapping[str, Any]:
    return require_mapping(
        require_ok(
            request(
                runtime,
                "GET",
                deployments_path(project, "attempts", attempt_id),
                access_token=access_token,
            )
        ),
        "deployment",
    )


def _status(attempt: Any) -> str:
    status = text(attempt, "status")
    phase = text(attempt, "phase")

    if status == "in-progress" and phase != "-":
        return phase

    failure = attempt.get("failure") if isinstance(attempt, Mapping) else None
    if status == "failed" and text(failure, "kind") == "stopped":
        return "stopped"

    return status


def _commit(attempt: Any, *, full: bool = False) -> str:
    commit = attempt.get("commit") if isinstance(attempt, Mapping) else None
    sha = text(commit, "sha")

    if sha == "-":
        return "-"

    if full:
        message = text(commit, "message")
        return sha if message == "-" else f"{sha} {message}"

    return sha[:12]
