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

"""Shared command context, validation, and output helpers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import click
from rich import box
from rich.console import Console
from rich.table import Table

from parlant.bin.dashboard.client import project_path, request, require_ok
from parlant.bin.dashboard.errors import DashboardError, no_project_context, not_managed
from parlant.bin.dashboard.runtime import DashboardRuntime, build_runtime


def runtime_for(ctx: click.Context) -> DashboardRuntime:
    """Return the invocation runtime, initializing it if needed."""
    if ctx.obj is None:
        ctx.obj = build_runtime()

    assert isinstance(ctx.obj, DashboardRuntime)

    return ctx.obj


def require_project(runtime: DashboardRuntime) -> str:
    """Return the selected project ID or raise a usage error."""
    if runtime.project:
        return runtime.project

    raise no_project_context()


def fetch_project(
    runtime: DashboardRuntime, project_id: str, access_token: str
) -> Mapping[str, Any]:
    """Fetch one project, or raise if the response is not a project record."""
    return require_mapping(
        require_ok(request(runtime, "GET", project_path(project_id), access_token=access_token)),
        "project",
    )


def project_kind(project: Mapping[str, Any]) -> str:
    """The hosting kind of a project, from the project or its deployment."""
    kind = text(project, "projectKind")

    if kind != "-":
        return kind

    deployment = project.get("deployment") if isinstance(project, Mapping) else None

    return text(deployment, "hostingMode") if isinstance(deployment, Mapping) else "-"


def require_managed(
    runtime: DashboardRuntime, project_id: str, access_token: str
) -> Mapping[str, Any]:
    """Fetch a project and confirm it is managed-hosted, or raise NOT_MANAGED."""
    project = fetch_project(runtime, project_id, access_token)

    if project_kind(project) != "managed":
        raise not_managed(text(project, "name"))

    return project


def echo_target(project: Mapping[str, Any], workspace: str | None = None) -> None:
    """Print the resolved target of a mutation so it never runs blind."""
    name = text(project, "name")
    identifier = text(project, "id")
    tenant = workspace or text(project, "tenantId")

    click.echo(f"→ {name} ({identifier}) · workspace {tenant}")


def text(entry: Any, key: str) -> str:
    """Format a field for display, using a dash for missing values."""
    if not isinstance(entry, Mapping):
        return "-"

    value = entry.get(key)

    if value is None or value == "":
        return "-"

    if isinstance(value, bool):
        return "yes" if value else "no"

    return str(value)


def nested(entry: Any, *keys: str) -> str:
    """Like text(), for a value one level down."""
    if not isinstance(entry, Mapping):
        return "-"

    inner = entry.get(keys[0]) if keys else None

    return text(inner, keys[1]) if len(keys) > 1 else text(entry, keys[0])


def require_list(body: Any, what: str) -> list[Any]:
    if not isinstance(body, list):
        raise DashboardError(
            f"The Parlant dashboard returned an unexpected {what}. Upgrading Parlant may help."
        )
    return body


def require_mapping(body: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(body, Mapping):
        raise DashboardError(
            f"The Parlant dashboard returned an unexpected {what}. Upgrading Parlant may help."
        )
    return body


IDENTIFIER_COLUMNS = frozenset(
    {"id", "build", "deploy", "turn", "session", "workspace", "agent", "trace", "key"}
)


def print_table(columns: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    """Print a table with unwrapped identifiers and wrapping descriptive columns."""
    table = Table(box=box.SIMPLE, pad_edge=False)

    for index, column in enumerate(columns):
        identifier = column.strip().lower() in IDENTIFIER_COLUMNS
        table.add_column(
            column,
            style="bold" if index == 0 else None,
            no_wrap=identifier,
            overflow="fold",
        )

    for row in rows:
        table.add_row(*row)

    Console().print(table)


def timestamp(entry: Any, key: str) -> str:
    """Format a timestamp to minute precision."""
    raw = text(entry, key)

    if raw == "-" or "T" not in raw:
        return raw

    date, _, rest = raw.partition("T")

    return f"{date} {rest[:5]}"


def print_fields(pairs: Sequence[tuple[str, str]]) -> None:
    """Print a record as aligned label/value pairs."""
    width = max((len(label) for label, _ in pairs), default=0)

    for label, value in pairs:
        click.echo(f"  {label.ljust(width)}  {value}")
