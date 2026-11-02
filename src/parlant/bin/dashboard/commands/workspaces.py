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
"""``parlant dashboard workspace`` — the workspaces you can reach, and which one commands use."""

from __future__ import annotations

from typing import Any

import click

from parlant.bin.dashboard.auth import authenticate, identify_legacy_sign_in
from parlant.bin.dashboard.client import WORKSPACES_PATH, request, require_ok
from parlant.bin.dashboard.commands import dashboard
from parlant.bin.dashboard.commands._shared import print_table, require_list, runtime_for, text
from parlant.bin.dashboard.errors import not_signed_in
from parlant.bin.dashboard.output import emit_json, json_mode, json_option
from parlant.bin.dashboard.runtime import DashboardRuntime
from parlant.bin.help import examples


@dashboard.group()
def workspace() -> None:
    """List your workspaces and choose the one commands act in."""


@workspace.command(
    "list",
    epilog=examples(
        "parlant dashboard workspace list",
        "parlant dashboard workspace list --json",
    ),
)
@json_option
@click.pass_context
def workspaces_list(ctx: click.Context) -> None:
    """List your workspaces, and which ones this machine is signed in to."""
    runtime = runtime_for(ctx)

    _identify_legacy(runtime)
    entries = require_list(
        require_ok(request(runtime, "GET", WORKSPACES_PATH, access_token=authenticate(runtime))),
        "workspace list",
    )
    signed_in = set(runtime.config.signed_in_workspaces(runtime.host))
    active = runtime.config.active_workspace(runtime.host)

    if json_mode():
        emit_json(
            [
                {
                    "id": e.get("tenantId"),
                    "name": e.get("name"),
                    "role": e.get("role"),
                    "signedIn": e.get("tenantId") in signed_in,
                    "active": e.get("tenantId") == active,
                }
                for e in entries
                if isinstance(e, dict)
            ]
        )
        return

    if not entries:
        click.echo("You do not belong to any workspaces yet.")
        return

    print_table(
        ("Name", "Workspace", "Role", "Access", "Signed in"),
        (
            (
                text(e, "name"),
                text(e, "tenantId"),
                text(e, "role"),
                text(e, "accessSource"),
                _signed_in(e, signed_in, active),
            )
            for e in entries
        ),
    )


@workspace.command(
    "use",
    epilog=examples(
        "parlant dashboard workspace use WORKSPACE_ID",
    ),
)
@click.argument("workspace_id")
@json_option
@click.pass_context
def workspaces_use(ctx: click.Context, workspace_id: str) -> None:
    """Make WORKSPACE_ID the workspace commands act in by default."""
    runtime = runtime_for(ctx)

    _identify_legacy(runtime)
    if workspace_id not in runtime.config.signed_in_workspaces(runtime.host):
        raise not_signed_in(workspace_id)

    runtime.config.set_active_workspace(runtime.host, workspace_id)

    if json_mode():
        emit_json({"workspace": workspace_id, "active": True})
        return
    click.echo(f"Active workspace: {workspace_id}.")


def _identify_legacy(runtime: DashboardRuntime) -> None:
    if runtime.credentials.get(runtime.host) is not None:
        identify_legacy_sign_in(runtime)


def _signed_in(entry: Any, signed_in: set[str], active: str | None) -> str:
    tenant = entry.get("tenantId") if isinstance(entry, dict) else None
    if tenant is not None and tenant == active:
        return "active"
    return "yes" if tenant in signed_in else "-"
