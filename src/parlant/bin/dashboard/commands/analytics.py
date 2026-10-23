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

"""``parlant dashboard analytics`` — activity across a project's sessions."""

from __future__ import annotations

from typing import Any, Mapping

import click

from parlant.bin.dashboard.auth import authenticate
from parlant.bin.dashboard.client import project_path, request, require_ok
from parlant.bin.dashboard.commands import dashboard
from parlant.bin.dashboard.commands._shared import (
    print_fields,
    print_table,
    require_list,
    require_mapping,
    require_project,
    runtime_for,
    text,
)
from parlant.bin.help import examples

RANGES = ("last-24-hours", "last-7-days", "last-30-days")

SESSION_TYPES = ("all", "live", "simulation")


@dashboard.group()
def analytics() -> None:
    """See how much a project is being used."""


@analytics.command(
    "view",
    epilog=examples(
        "parlant dashboard analytics view",
        "parlant dashboard analytics view --range last-7-days",
        "parlant dashboard analytics view --range last-30-days --agent AGENT_ID --type live",
    ),
)
@click.option("--range", "time_range", type=click.Choice(RANGES), required=True, help="Time range.")
@click.option("--agent", default=None, help="Filter by agent ID.")
@click.option(
    "--type",
    "session_type",
    type=click.Choice(SESSION_TYPES),
    default=None,
    help="Session type filter.",
)
@click.pass_context
def analytics_view(
    ctx: click.Context, time_range: str, agent: str | None, session_type: str | None
) -> None:
    """Show session counts and activity for a time range."""
    runtime = runtime_for(ctx)
    project = require_project(runtime)

    query: dict[str, Any] = {"range": time_range}

    if agent:
        query["agentId"] = agent
    if session_type:
        query["type"] = session_type

    general = require_mapping(
        require_ok(
            request(
                runtime,
                "GET",
                project_path(project, "dashboard", "general"),
                access_token=authenticate(runtime),
                query=query,
            )
        ),
        "dashboard summary",
    )

    counts = general.get("sessionCount")

    if isinstance(counts, Mapping):
        print_fields(
            [
                ("RANGE", time_range),
                ("SESSIONS", text(counts, "current")),
                ("PREVIOUS", text(counts, "previous")),
                ("CHANGE", _percentage(counts.get("changePercent"))),
            ]
        )

    _print_busiest_hour(general)
    _print_agent_activity(general)


def _print_busiest_hour(general: Mapping[str, Any]) -> None:
    hourly = general.get("hourlyActivity")
    buckets = hourly.get("buckets") if isinstance(hourly, Mapping) else None

    if not isinstance(buckets, list) or not buckets:
        return

    busiest = max(buckets, key=_count_of)
    total = _count_of(busiest)

    if total <= 0:
        return

    hour = busiest.get("hour") if isinstance(busiest, Mapping) else None
    label = f"{hour:02d}:00" if isinstance(hour, int) else text(busiest, "from")

    click.echo("")
    print_fields([("BUSIEST HOUR", f"{label} ({total})")])


def _print_agent_activity(general: Mapping[str, Any]) -> None:
    activity = general.get("activity")
    series = activity.get("series") if isinstance(activity, Mapping) else None

    if not isinstance(series, list) or not series:
        return

    click.echo("")
    print_table(
        ("Agent", "Sessions"),
        (
            (
                text(s, "displayName") if text(s, "displayName") != "-" else text(s, "key"),
                text(s, "total"),
            )
            for s in series
        ),
    )


@analytics.command(
    "labels",
    epilog=examples(
        "parlant dashboard analytics labels",
        "parlant dashboard analytics labels --range last-30-days --limit 20",
    ),
)
@click.option("--range", "time_range", type=click.Choice(RANGES), required=True, help="Time range.")
@click.option("--agent", default=None, help="Filter by agent ID.")
@click.option(
    "--type",
    "session_type",
    type=click.Choice(SESSION_TYPES),
    default=None,
    help="Session type filter.",
)
@click.option("--search", default=None, help="Filter labels by search text.")
@click.option("--limit", default="50", show_default=True, help="Maximum labels (1-100).")
@click.option("--offset", default="0", show_default=True, help="Labels to skip.")
@click.pass_context
def analytics_labels(
    ctx: click.Context,
    time_range: str,
    agent: str | None,
    session_type: str | None,
    search: str | None,
    limit: str,
    offset: str,
) -> None:
    """Show which labels sessions are being tagged with."""
    runtime = runtime_for(ctx)
    project = require_project(runtime)

    query: dict[str, Any] = {"range": time_range, "limit": limit, "offset": offset}

    for name, value in (("agentId", agent), ("type", session_type), ("search", search)):
        if value:
            query[name] = value

    body = require_ok(
        request(
            runtime,
            "GET",
            project_path(project, "dashboard", "labels"),
            access_token=authenticate(runtime),
            query=query,
        )
    )

    entries = require_list(body.get("items") if isinstance(body, Mapping) else body, "label list")

    if not entries:
        click.echo("No labels in this range.")
        return

    print_table(("Label", "Sessions"), ((text(e, "name"), text(e, "count")) for e in entries))


def _count_of(bucket: Any) -> int:
    """Sum session counts in an activity bucket."""
    values = bucket.get("values") if isinstance(bucket, Mapping) else None

    if not isinstance(values, Mapping):
        return 0

    return sum(count for count in values.values() if isinstance(count, int))


def _percentage(value: Any) -> str:
    return f"{value:.2f}%" if isinstance(value, (int, float)) else "-"
