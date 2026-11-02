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

"""The ``parlant dashboard`` command group."""

from __future__ import annotations

import click

from parlant.bin.dashboard.output import reset_json_mode
from parlant.bin.dashboard.runtime import DashboardRuntime, build_runtime


@click.group()
@click.option("--project", default=None, help="Parlant project identifier.")
@click.option("--workspace", default=None, help="Parlant workspace identifier.")
@click.pass_context
def dashboard(ctx: click.Context, project: str | None, workspace: str | None) -> None:
    """Work with the Parlant dashboard.

    Sign in once, then build and inspect your agents from here.
    """
    reset_json_mode()
    if ctx.obj is None and ctx.invoked_subcommand is not None:
        ctx.obj = build_runtime()

    if isinstance(ctx.obj, DashboardRuntime):
        if project:
            ctx.obj.project = project
        if workspace:
            ctx.obj.workspace = workspace


from parlant.bin.dashboard.commands import (  # noqa: E402
    agents,
    analytics,
    auth,
    deployments,
    logs,
    projects,
    project_creation,
    project_connection,
    deploy,
    secrets,
    sessions,
    traces,
    workspaces,
)

__all__ = [
    "dashboard",
    "agents",
    "analytics",
    "auth",
    "deployments",
    "logs",
    "projects",
    "project_creation",
    "project_connection",
    "deploy",
    "secrets",
    "sessions",
    "traces",
    "workspaces",
]


# gcloud treats --project as a global flag that works after any subcommand, not
# only on the top group. Bind it (and --workspace) on every project-scoped
# subcommand too, so `... deploy --project X` reads the same as
# `... --project X deploy`. Auth and workspace selection are not scoped to a
# project, so they are left alone.
_UNSCOPED_GROUPS = frozenset({"auth", "workspace"})


def _apply_scope(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
    if value and isinstance(ctx.obj, DashboardRuntime):
        setattr(ctx.obj, str(param.name), value)
    return value


def _install_scope_options(group: click.Group, *, skip: frozenset[str]) -> None:
    for name, command in group.commands.items():
        if name in skip:
            continue

        if isinstance(command, click.Group):
            _install_scope_options(command, skip=frozenset())
            continue

        present = {p.name for p in command.params}

        for option, help_text in (
            ("project", "Parlant project identifier."),
            ("workspace", "Parlant workspace identifier."),
        ):
            if option not in present:
                command.params.append(
                    click.Option(
                        [f"--{option}"],
                        expose_value=False,
                        callback=_apply_scope,
                        help=help_text,
                    )
                )


_install_scope_options(dashboard, skip=_UNSCOPED_GROUPS)
