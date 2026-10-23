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

"""``parlant dashboard project`` — discovering and inspecting projects."""

from __future__ import annotations

from typing import Any, Mapping

import click
from rich import box
from rich.console import Console
from rich.table import Table

from parlant.bin.dashboard.auth import authenticate
from parlant.bin.dashboard.client import PROJECTS_PATH, project_path, request, require_ok
from parlant.bin.dashboard.commands import dashboard
from parlant.bin.dashboard.commands._shared import (
    fetch_project,
    print_fields,
    project_kind,
    require_mapping,
    require_project,
    runtime_for,
    text,
)
from parlant.bin.dashboard.errors import DashboardError
from parlant.bin.dashboard.output import emit_json, json_mode, json_option
from parlant.bin.dashboard.runtime import DashboardRuntime
from parlant.bin.help import examples


@dashboard.group()
def project() -> None:
    """Inspect your dashboard projects."""


@project.command(
    "list",
    epilog=examples(
        "parlant dashboard project list",
        "parlant dashboard project list --json",
    ),
)
@json_option
@click.pass_context
def projects_list(ctx: click.Context) -> None:
    """List the projects in your workspace."""
    runtime = runtime_for(ctx)

    response = request(
        runtime,
        "GET",
        PROJECTS_PATH,
        access_token=authenticate(runtime),
    )

    entries = require_ok(response)

    if not isinstance(entries, list):
        raise DashboardError(
            "The Parlant dashboard returned an unexpected project list. Upgrading Parlant may help."
        )

    if json_mode():
        emit_json([project_summary(entry) for entry in entries if isinstance(entry, dict)])
        return

    if not entries:
        click.echo("No projects in this workspace yet.")
        return

    console = Console()

    table = Table(box=box.SIMPLE, pad_edge=False)
    table.add_column("Name", style="bold", no_wrap=True)
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Kind", no_wrap=True)
    table.add_column("Hosting", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("URL", style="dim")

    for entry in entries:
        table.add_row(
            text(entry, "name"),
            text(entry, "id"),
            text(entry, "projectKind"),
            _hosting_mode(entry),
            _state(entry),
            text(entry, "parlantUrl"),
        )

    console.print(table)


def project_summary(entry: Mapping[str, Any]) -> dict[str, Any]:
    """A project as `--json` prints it."""
    setup = entry.get("setup")
    # A managed project deploys a GitHub repository and branch.
    deployment = entry.get("deployment")
    deployed = deployment if isinstance(deployment, Mapping) else {}
    return {
        "id": entry.get("id"),
        "name": entry.get("name"),
        "projectKind": entry.get("projectKind"),
        "state": entry.get("lifecycleState"),
        "setupStatus": setup.get("status") if isinstance(setup, Mapping) else None,
        "repository": deployed.get("githubRepoFullName") or None,
        "branch": deployed.get("branch") or None,
    }


def _state(entry: Any) -> str:
    """Read the lifecycle state, falling back to setup status."""
    lifecycle = text(entry, "lifecycleState")

    if lifecycle != "-":
        return lifecycle

    setup = entry.get("setup") if isinstance(entry, dict) else None

    return text(setup, "status") if isinstance(setup, dict) else "-"


def _hosting_mode(entry: Any) -> str:
    """Where the project runs, from the project or its deployment."""
    hosting = text(entry, "hostingMode")

    if hosting != "-":
        return hosting

    deployment = entry.get("deployment") if isinstance(entry, dict) else None

    return text(deployment, "hostingMode") if isinstance(deployment, dict) else "-"


@project.command(
    "describe",
    epilog=examples(
        "parlant dashboard project describe PROJECT_ID",
    ),
)
@click.argument("project_id", required=False)
@click.pass_context
def projects_describe(ctx: click.Context, project_id: str | None) -> None:
    """Show one project in full, including its live status when managed.

    PROJECT_ID is the project to show, as `project list` prints it.
    """
    runtime = runtime_for(ctx)
    identifier = project_id or require_project(runtime)
    access_token = authenticate(runtime)

    project = fetch_project(runtime, identifier, access_token)

    print_fields(
        [
            ("ID", text(project, "id")),
            ("PUBLIC ID", text(project, "publicId")),
            ("NAME", text(project, "name")),
            ("WORKSPACE", text(project, "tenantId")),
            ("KIND", text(project, "projectKind")),
            ("HOSTING", _hosting_mode(project)),
            ("STATE", _state(project)),
            ("URL", text(project, "parlantUrl")),
            ("CREATED", text(project, "createdAt")),
            ("UPDATED", text(project, "updatedAt")),
        ]
    )

    if project_kind(project) == "managed":
        _describe_managed_runtime(runtime, identifier, access_token)


def _describe_managed_runtime(
    runtime: DashboardRuntime, identifier: str, access_token: str
) -> None:
    """Append the build/deploy status and health of a managed project."""
    status = require_ok(
        request(
            runtime,
            "GET",
            project_path(identifier, "status"),
            access_token=access_token,
        )
    )

    click.echo("")

    if status is None:
        click.echo("This project has not been built or deployed yet.")
    else:
        _print_status(require_mapping(status, "project status"))

    _print_health(runtime, identifier, access_token)


def _print_status(status: Mapping[str, Any]) -> None:
    print_fields(
        [
            ("STATUS", text(status, "status")),
            ("VERSION", text(status, "currentVersion")),
            ("ACTIVE BUILD", _stage(status, "activeBuild")),
            ("ACTIVE DEPLOY", _stage(status, "activeDeploy")),
            ("LAST BUILD", _stage(status, "lastBuild")),
            ("LAST DEPLOY", _stage(status, "lastDeploy")),
        ]
    )


def _print_health(runtime: DashboardRuntime, identifier: str, access_token: str) -> None:
    health = require_ok(
        request(
            runtime,
            "GET",
            project_path(identifier, "health"),
            access_token=access_token,
        )
    )

    if isinstance(health, dict):
        click.echo("")
        print_fields(
            [
                ("HEALTH", text(health, "status")),
                ("CONNECTED", text(health, "connected")),
                ("PARLANT VERSION", text(health, "compatibleVersion")),
            ]
        )

        _print_failing_checks(health)


def _stage(status: Any, key: str) -> str:
    """Format a build or deployment as an ID and status, or a dash if absent."""
    stage = status.get(key) if isinstance(status, dict) else None

    if not isinstance(stage, dict):
        return "-"

    identifier = text(stage, "id")
    state = text(stage, "status")

    return f"{identifier} ({state})" if state != "-" else identifier


HEALTHY_CHECK_STATUSES = frozenset({"healthy", "connected", "ok", "up", "ready"})


def _print_failing_checks(health: Any) -> None:
    """Print failing health checks and their measurements."""
    parlant = health.get("parlant") if isinstance(health, dict) else None
    checks = parlant.get("checks") if isinstance(parlant, dict) else None

    if not isinstance(checks, dict):
        return

    failing = [
        (name, check)
        for name, check in checks.items()
        if isinstance(check, dict) and str(check.get("status")) not in HEALTHY_CHECK_STATUSES
    ]

    if not failing:
        return

    click.echo("")

    for name, check in failing:
        measures = ", ".join(
            f"{key}={value}"
            for key, value in check.items()
            if key not in ("status", "routes") and not isinstance(value, (dict, list))
        )
        detail = f" ({measures})" if measures else ""
        click.echo(f"  FAILING CHECK  {name}: {check.get('status')}{detail}")
