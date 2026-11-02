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

"""``parlant dashboard secret`` — the deployment secrets of a managed project.

Secrets belong to managed projects only: the platform stores them and gives
them to the running app, so a self-hosted or local project — which runs its own
server — has no platform secret to set. Secrets are write-only: their names are
listed, their values never come back. Setting or removing a secret saves the
project's deployment settings; the running deployment keeps what it started
with until the next deployment, which is why every change here points at
`parlant dashboard deploy`.
"""

from __future__ import annotations

import re
import sys
from typing import Mapping

import click

from parlant.bin.dashboard.auth import authenticate
from parlant.bin.dashboard.commands import dashboard
from parlant.bin.dashboard.commands._shared import (
    echo_target,
    print_table,
    require_list,
    require_managed,
    require_project,
    runtime_for,
)
from parlant.bin.dashboard.deployments import change, overview
from parlant.bin.dashboard.errors import DashboardError
from parlant.bin.dashboard.runtime import DashboardRuntime
from parlant.bin.help import examples

MAX_SECRET_VALUE_BYTES = 64 * 1024

DEPLOY_REMINDER = (
    "The running deployment keeps its secrets until the next deployment. "
    "Apply this change now with `parlant dashboard deploy`."
)


@dashboard.group()
def secret() -> None:
    """Manage a managed project's deployment secrets."""


@secret.command(
    "list",
    epilog=examples(
        "parlant dashboard secret list",
        "parlant dashboard secret list --project PROJECT_ID",
    ),
)
@click.pass_context
def secrets_list(ctx: click.Context) -> None:
    """List secret keys. Values are never returned."""
    runtime = runtime_for(ctx)
    project = require_project(runtime)
    access_token = authenticate(runtime)
    require_managed(runtime, project, access_token)

    settings = overview(runtime, project, access_token).get("settings")
    names = require_list(
        settings.get("secretNames") if isinstance(settings, Mapping) else None, "secret list"
    )

    if not names:
        click.echo("This project has no deployment secrets.")
        return

    print_table(("Key",), ((str(name),) for name in sorted(names)))


@secret.command(
    "set",
    epilog=examples(
        "parlant dashboard secret set STRIPE_API_KEY=value",
        'printf %s "$STRIPE_API_KEY" | parlant dashboard secret set STRIPE_API_KEY',
    ),
)
@click.argument("assignment", metavar="KEY=VALUE")
@click.pass_context
def secrets_set(ctx: click.Context, assignment: str) -> None:
    """Set a deployment secret.

    ASSIGNMENT is `KEY=VALUE`.

    Give KEY on its own to read the value from stdin instead, which keeps it out
    of your shell history and the process list:
    `printf %s "$SECRET" | parlant dashboard secret set KEY`.

    Set secrets yourself rather than sharing their values with anyone: run this
    command, or add the secret from the project's page in the dashboard. The
    value is sent only to the Parlant dashboard and is never printed back. Run
    `parlant dashboard deploy` afterwards to apply it.
    """
    runtime = runtime_for(ctx)
    project = require_project(runtime)
    access_token = authenticate(runtime)
    managed = require_managed(runtime, project, access_token)

    key, separator, inline = assignment.partition("=")
    key = key.strip()

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise DashboardError(
            "A secret name must be a valid environment variable name: letters, "
            "digits and underscores, not starting with a digit."
        )

    if separator:
        value = inline
    else:
        if sys.stdin.isatty():
            raise DashboardError(
                "No value given. Pass it as KEY=VALUE, or give KEY alone and pipe the "
                'value in: `printf %s "$SECRET" | parlant dashboard secret set KEY`.'
            )
        value = sys.stdin.read()

    value = value.rstrip("\n")

    if not value:
        raise DashboardError("Refusing to store an empty secret value.")

    encoded = value.encode()

    if len(encoded) > MAX_SECRET_VALUE_BYTES:
        raise DashboardError(
            f"That value is {len(encoded)} bytes, above the "
            f"{MAX_SECRET_VALUE_BYTES}-byte limit for a deployment secret."
        )

    echo_target(managed)

    _save(runtime, project, access_token, {"kind": "set", "name": key, "value": value})

    click.echo(f"Set {key}. {DEPLOY_REMINDER}")


@secret.command(
    "unset",
    epilog=examples(
        "parlant dashboard secret unset STRIPE_API_KEY",
        "parlant dashboard secret unset STRIPE_API_KEY --yes",
    ),
)
@click.argument("key")
@click.option("--yes", "-y", is_flag=True, help="Do not prompt for confirmation.")
@click.pass_context
def secrets_unset(ctx: click.Context, key: str, yes: bool) -> None:
    """Remove a secret from the project's deployment configuration.

    KEY is the secret's name, as `secret list` prints it. Run
    `parlant dashboard deploy` afterwards to apply its removal.
    """
    runtime = runtime_for(ctx)
    project = require_project(runtime)
    access_token = authenticate(runtime)
    managed = require_managed(runtime, project, access_token)

    echo_target(managed)

    if not yes:
        click.confirm(f"Remove secret {key} from this project?", abort=True)

    _save(runtime, project, access_token, {"kind": "remove", "name": key})

    click.echo(f"Removed {key}. {DEPLOY_REMINDER}")


def _save(
    runtime: DashboardRuntime,
    project: str,
    access_token: str,
    secret: dict[str, str],
) -> None:
    """Save one secret change, sending the saved settings back as they are."""
    change(
        runtime,
        project,
        access_token,
        method="PUT",
        segments=("settings",),
        body_for=lambda current: {"settings": current.get("settings"), "secrets": [secret]},
        accepted_as="overview",
    )
