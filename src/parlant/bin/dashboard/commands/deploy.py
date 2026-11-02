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

"""``parlant dashboard deploy`` — build and deploy a managed project.

Deploys the configured branch's latest commit, or an exact commit, with the
project's saved settings and secrets, then follows the deployment until it is
live or has failed. One deployment runs at a time per project.
"""

from __future__ import annotations

import re
import time
from typing import Any, Mapping

import click

from parlant.bin.dashboard.auth import authenticate
from parlant.bin.dashboard.client import deployments_path, project_path, request, require_ok
from parlant.bin.dashboard.pricing import echo_pricing_notice
from parlant.bin.dashboard.commands import dashboard
from parlant.bin.dashboard.commands._shared import (
    echo_target,
    require_managed,
    require_mapping,
    require_project,
    runtime_for,
    text,
)
from parlant.bin.dashboard.deployments import Refused, change, overview
from parlant.bin.dashboard.errors import DashboardError, not_connected
from parlant.bin.dashboard.runtime import DashboardRuntime
from parlant.bin.help import examples

POLL_INTERVAL_SECONDS = 2.0

DEPLOYMENT_DEADLINE_SECONDS = 14 * 60

PHASES = {"building": "Building", "starting": "Starting", "promoting": "Promoting"}


@dashboard.command(
    "deploy",
    epilog=examples(
        "parlant dashboard deploy",
        "parlant dashboard deploy --commit COMMIT_SHA",
        "parlant dashboard deploy --project PROJECT_ID",
    ),
)
@click.option(
    "--commit",
    "commit",
    metavar="SHA",
    help="Deploy this exact commit (full 40-character SHA) instead of the branch's latest.",
)
@click.pass_context
def deploy(ctx: click.Context, commit: str | None) -> None:
    """Build and deploy the managed project with its saved settings and secrets."""
    runtime = runtime_for(ctx)
    project = require_project(runtime)

    if commit is not None:
        commit = commit.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise DashboardError("--commit takes a full 40-character commit SHA.")

    access_token = authenticate(runtime)
    managed = require_managed(runtime, project, access_token)

    settings = overview(runtime, project, access_token).get("settings")
    source = settings.get("source") if isinstance(settings, Mapping) else None
    if not isinstance(source, Mapping) or not source.get("repository"):
        raise not_connected()

    echo_target(managed)

    try:
        attempt = require_mapping(
            change(
                runtime,
                project,
                access_token,
                method="POST",
                segments=("attempts",),
                body_for=lambda _: {"commit": commit} if commit else {},
                accepted_as="attempt",
            ),
            "deployment",
        )
    except Refused as refusal:
        if refusal.code == "busy":
            raise _busy(runtime, project, access_token) from refusal
        raise

    attempt_id = text(attempt, "id")
    target = f"commit {commit[:12]}" if commit else f"branch {text(source, 'branch')}"
    click.echo(f"Started deployment {attempt_id} of {text(source, 'repository')} {target}.")

    try:
        _watch(runtime, project, access_token, attempt_id, attempt)
    except KeyboardInterrupt:
        click.echo("")
        click.echo("Stopped watching. The deployment is still running on the platform.")
        click.echo(f"Check on it with `parlant dashboard deployment view {attempt_id}`.")
        raise SystemExit(1)


def _busy(runtime: DashboardRuntime, project: str, access_token: str) -> DashboardError:
    active = overview(runtime, project, access_token).get("activeId")

    if not isinstance(active, str) or not active:
        return DashboardError(
            "Another deployment or a repository check is running for this project. "
            "Wait for it to finish, then deploy again."
        )

    return DashboardError(
        f"Deployment {active} is still running for this project. Watch it with "
        f"`parlant dashboard deployment view {active}`, or stop it with "
        f"`parlant dashboard deployment stop {active}`, then deploy again."
    )


def _watch(
    runtime: DashboardRuntime,
    project: str,
    access_token: str,
    attempt_id: str,
    attempt: Mapping[str, Any],
) -> None:
    deadline = time.monotonic() + DEPLOYMENT_DEADLINE_SECONDS
    last_phase = ""

    while True:
        phase = PHASES.get(text(attempt, "phase"), "")
        if phase and phase != last_phase:
            click.echo(f"{phase}…")
            last_phase = phase

        status = text(attempt, "status")

        if status == "completed":
            click.echo(f"Deployed. {attempt_id} is live.")
            # A single fresh read after promotion; an older or temporarily
            # unavailable coverage endpoint cannot turn a live deploy into failure.
            try:
                response = request(
                    runtime,
                    "GET",
                    project_path(project, "project-connection"),
                    access_token=access_token,
                )
                if response.status_code == 200:
                    status_body = response.json()
                    if isinstance(status_body, Mapping):
                        echo_pricing_notice(status_body)
            except (DashboardError, ValueError):
                pass
            return

        if status == "failed":
            raise _failed(attempt_id, attempt)

        if time.monotonic() >= deadline:
            raise DashboardError(
                f"The deployment has not finished after {DEPLOYMENT_DEADLINE_SECONDS // 60} "
                "minutes, so this command stopped waiting. The platform is still working "
                f"on it — this is not a cancellation. Check it with "
                f"`parlant dashboard deployment view {attempt_id}`."
            )

        runtime.sleep(POLL_INTERVAL_SECONDS)
        attempt = require_mapping(
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


def _failed(attempt_id: str, attempt: Mapping[str, Any]) -> DashboardError:
    failure = attempt.get("failure")
    kind = text(failure, "kind")
    reason = text(failure, "message")
    logs = f"`parlant dashboard deployment logs {attempt_id}`"

    if kind == "stopped":
        return DashboardError(f"Deployment {attempt_id} was stopped.")

    outcomes = {"build": "failed to build", "startup": "failed to start", "timeout": "timed out"}
    headline = f"Deployment {attempt_id} {outcomes.get(kind, 'failed')}"

    detail = f": {reason}" if reason != "-" else ""

    return DashboardError(f"{headline}{detail}. Its logs will say more: {logs}.")
