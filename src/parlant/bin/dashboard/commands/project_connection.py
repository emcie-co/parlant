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

"""Connecting a local project's server: ``parlant dashboard project connect|check|confirm``.

A local project runs its own server, which reaches the dashboard through its
project token. `connect` puts the token in the project's `.env`, `check` asks
whether the dashboard sees the server, and `confirm` completes the project's
setup once it does, importing any sessions the server already holds."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from parlant.bin.dashboard.pricing import echo_pricing_notice

import click
import httpx

from parlant.bin.dashboard.auth import authenticate
from parlant.bin.dashboard.client import project_path, request, require_ok
from parlant.bin.dashboard.commands._shared import require_mapping, require_project, runtime_for
from parlant.bin.dashboard.commands.projects import project
from parlant.bin.dashboard.env_file import set_env_values
from parlant.bin.dashboard.errors import DashboardError
from parlant.bin.dashboard.output import emit_json, json_mode, json_option, say
from parlant.bin.dashboard.runtime import DashboardRuntime
from parlant.bin.help import examples

TOKEN_ENV_VAR = "PARLANT_DASHBOARD_PROJECT_TOKEN"

# Coding agents often stop a command after about two minutes; stop waiting well before.
SYNC_WAIT_SECONDS = 90.0


@project.command(
    "connect",
    epilog=examples(
        "parlant dashboard project connect .",
        "parlant dashboard project connect ./my-agent --project PROJECT_ID",
    ),
)
@click.argument(
    "folder",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=Path("."),
)
@json_option
@click.pass_context
def project_connect(ctx: click.Context, folder: Path) -> None:
    """Put the project token in FOLDER/.env so its server reaches the dashboard."""
    runtime = runtime_for(ctx)
    result = connect(runtime, require_project(runtime), folder)

    if json_mode():
        emit_json(result)
        return
    click.echo(f"Wrote {result['envVarName']} to {result['envFile']}.")


@project.command(
    "check",
    epilog=examples(
        "parlant dashboard project check",
        "parlant dashboard project check --project PROJECT_ID",
    ),
)
@json_option
@click.pass_context
def project_check(ctx: click.Context) -> None:
    """Check whether the dashboard sees this project's server."""
    runtime = runtime_for(ctx)
    project_id = require_project(runtime)
    status = require_mapping(
        require_ok(
            request(
                runtime,
                "GET",
                project_path(project_id, "project-connection"),
                access_token=authenticate(runtime),
            )
        ),
        "connection status",
    )
    report = connection_report(project_id, status)

    if json_mode():
        emit_json(report)
        return
    if report["connected"]:
        click.echo(f"Connected: Parlant {report['parlantVersion'] or '(unknown version)'}.")
    else:
        click.echo(str(report["message"]))
    click.echo(f"Setup: {report['setupStatus'] or '-'}")
    if report["connected"]:
        echo_pricing_notice(status)


@project.command(
    "confirm",
    epilog=examples(
        "parlant dashboard project confirm",
        "parlant dashboard project confirm --project PROJECT_ID",
    ),
)
@json_option
@click.pass_context
def project_confirm(ctx: click.Context) -> None:
    """Complete this project's setup once the dashboard sees its server.

    If the server already holds sessions, such as from chatting with the agent
    to try it, they are imported into the dashboard first, as its Sync does.
    A long import may outlast one run: run this again to keep waiting for it."""
    runtime = runtime_for(ctx)
    project_id = require_project(runtime)
    access_token = authenticate(runtime)
    response = _complete_setup(runtime, project_id, access_token)
    imported: int | None = None

    # Backend refuses to complete setup until the server's sessions are imported.
    # Another conflict, such as a project being deleted, is refused again on the
    # retry and reported as it is.
    if response.status_code == 409:
        sync_status = _sync_status(runtime, project_id, access_token)
        if _is_imported(sync_status):
            imported = _import_progress(sync_status)[0]
        else:
            imported = _import_sessions(runtime, project_id, access_token, sync_status)
        response = _complete_setup(runtime, project_id, access_token)

    status = require_mapping(require_ok(response), "setup completion")

    # The dashboard answers either way, and completes only a connected project.
    if _setup_status(status) != "completed":
        report = connection_report(project_id, status)
        raise DashboardError(
            f"Setup is not complete. {report.get('message') or ''}".strip(),
            code="INCOMPATIBLE_VERSION" if report["phase"] == "incompatible" else "NOT_CONNECTED",
        )

    if json_mode():
        emit_json(
            {
                "projectId": project_id,
                "setupComplete": True,
                **({"importedSessions": imported} if imported is not None else {}),
            }
        )
        return
    click.echo("Setup complete.")
    echo_pricing_notice(status)


def _complete_setup(
    runtime: DashboardRuntime, project_id: str, access_token: str
) -> httpx.Response:
    return request(
        runtime,
        "POST",
        project_path(project_id, "setup", "complete"),
        access_token=access_token,
    )


def _import_sessions(
    runtime: DashboardRuntime,
    project_id: str,
    access_token: str,
    sync_status: Mapping[str, Any],
) -> int:
    """Import the server's sessions, as the dashboard's Sync does, and wait for it.

    An import already running is waited for, not started again. Returns how many
    sessions were imported; raises when it fails or outlasts this run."""
    if sync_status.get("status") != "importing":
        require_ok(
            request(
                runtime,
                "POST",
                project_path(project_id, "sessions", "sync"),
                access_token=access_token,
            )
        )

    waited = 0.0
    while True:
        sync_status = _sync_status(runtime, project_id, access_token)
        state = sync_status.get("status")
        imported, total = _import_progress(sync_status)
        if _is_imported(sync_status):
            return imported
        if state in ("failed", "offline"):
            reason = sync_status.get("lastError") or "The dashboard could not read the server."
            raise DashboardError(
                f"Importing the server's sessions failed: {reason} Make sure the server is "
                "running, then run `parlant dashboard project confirm` again.",
                code="SYNC_FAILED",
            )
        if waited >= SYNC_WAIT_SECONDS:
            raise DashboardError(
                f"Still importing the server's sessions ({imported} of {total}). "
                "Run `parlant dashboard project confirm` again to keep waiting; "
                "it continues the same import.",
                code="SYNC_IN_PROGRESS",
            )
        say(f"Importing the server's sessions: {imported} of {total}...")
        interval = sync_status.get("pollIntervalSeconds")
        pause = min(max(float(interval), 1.0), 10.0) if isinstance(interval, (int, float)) else 5.0
        runtime.sleep(pause)
        waited += pause


def _sync_status(
    runtime: DashboardRuntime, project_id: str, access_token: str
) -> Mapping[str, Any]:
    return require_mapping(
        require_ok(
            request(
                runtime,
                "GET",
                project_path(project_id, "sessions", "sync-status"),
                access_token=access_token,
            )
        ),
        "session import status",
    )


def _is_imported(sync_status: Mapping[str, Any]) -> bool:
    """Backend completes setup once the import has synced or moved on to live sessions."""
    return sync_status.get("status") == "synced" or sync_status.get("phase") == "live"


def _import_progress(sync_status: Mapping[str, Any]) -> tuple[int, int]:
    progress = sync_status.get("progress")
    progress = progress if isinstance(progress, Mapping) else {}
    imported = progress.get("importedSessions")
    total = progress.get("totalSessions")
    if not isinstance(total, int):
        total = sync_status.get("availableSessions")
    return (
        imported if isinstance(imported, int) else 0,
        total if isinstance(total, int) else 0,
    )


def connect(runtime: DashboardRuntime, project_id: str, folder: Path) -> dict[str, str]:
    """Fetch the project token and write it to `folder/.env`. The token is never printed."""
    setup = require_mapping(
        require_ok(
            request(
                runtime,
                "POST",
                project_path(project_id, "self-hosted", "auth-setup"),
                json_body={},
                access_token=authenticate(runtime),
            )
        ),
        "project token setup",
    )
    token = setup.get("projectToken")
    name = setup.get("envVarName") or TOKEN_ENV_VAR
    if not isinstance(token, str) or not token or not isinstance(name, str):
        raise DashboardError(
            "The Parlant dashboard did not return a project token. Upgrading Parlant may help."
        )

    env_file = set_env_values(folder, {name: token})
    return {"projectId": project_id, "envFile": str(env_file), "envVarName": name}


def connection_report(project_id: str, status: Mapping[str, Any]) -> dict[str, Any]:
    """The dashboard's connection status as `check --json` prints it."""
    connected = status.get("connected") is True
    compatible = status.get("compatibleVersion")
    report: dict[str, Any] = {
        "projectId": project_id,
        "phase": "connected",
        "connected": connected,
        "parlantVersion": status.get("parlantVersion"),
        "setupStatus": _setup_status(status),
    }
    if isinstance(status.get("pricing"), Mapping):
        report["pricing"] = status["pricing"]
    if not connected:
        report["phase"] = "unreachable"
        report["message"] = (
            "The dashboard doesn't see this project's server yet. Start it (./run.sh) "
            f"with {TOKEN_ENV_VAR} set in its .env, then check again."
        )
    elif compatible is False:
        report["phase"] = "incompatible"
        report["message"] = (
            f"The server's Parlant version ({status.get('parlantVersion') or 'unknown'}) "
            "is not supported by the dashboard. Upgrade Parlant in the project."
        )
    return report


def _setup_status(status: Mapping[str, Any]) -> str | None:
    setup = status.get("setup")
    if isinstance(setup, Mapping) and isinstance(setup.get("status"), str):
        return str(setup["status"])
    return "pending" if status.get("setupStatus") == "pending" else None
