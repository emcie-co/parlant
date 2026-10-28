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

"""``parlant dashboard session`` — reading conversations and what drove them."""

from __future__ import annotations

from typing import Any, Mapping

import click

from parlant.bin.dashboard.auth import authenticate
from parlant.bin.dashboard.client import request, require_ok, sessions_path
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
from parlant.bin.help import examples

EVENT_KINDS = ("message", "tool", "status", "custom")

EVENT_SOURCES = (
    "customer",
    "customer_ui",
    "human_agent",
    "human_agent_on_behalf_of_ai_agent",
    "ai_agent",
    "system",
)

SESSION_TYPES = ("live", "simulation")


@dashboard.group()
def session() -> None:
    """Read a project's sessions."""


@session.command(
    "list",
    epilog=examples(
        "parlant dashboard session list",
        "parlant dashboard session list --agent AGENT_ID --limit 20",
        "parlant dashboard session list --label premium --sort asc",
    ),
)
@click.option("--search", default=None, help="Search sessions.")
@click.option("--agent", default=None, help="Filter by agent.")
@click.option("--customer", default=None, help="Filter by customer.")
@click.option("--label", multiple=True, help="Require a label (repeatable).")
@click.option(
    "--type",
    "session_type",
    type=click.Choice(SESSION_TYPES),
    default=None,
    help="Filter by session type.",
)
@click.option("--created-from", default=None, help="Filter by earliest creation time.")
@click.option("--created-to", default=None, help="Filter by latest creation time.")
@click.option(
    "--sort",
    type=click.Choice(("asc", "desc")),
    default="desc",
    show_default=True,
    help="Sort by creation time.",
)
@click.option("--limit", default="25", show_default=True, help="Maximum sessions.")
@click.option("--offset", default="0", show_default=True, help="Sessions to skip.")
@click.pass_context
def sessions_list(
    ctx: click.Context,
    search: str | None,
    agent: str | None,
    customer: str | None,
    label: tuple[str, ...],
    session_type: str | None,
    created_from: str | None,
    created_to: str | None,
    sort: str,
    limit: str,
    offset: str,
) -> None:
    """List sessions in the project."""
    runtime = runtime_for(ctx)
    project = require_project(runtime)

    query: dict[str, Any] = {"sort": sort, "limit": limit, "offset": offset}

    for name, value in (
        ("search", search),
        ("agentId", agent),
        ("customerId", customer),
        ("type", session_type),
        ("createdFrom", created_from),
        ("createdTo", created_to),
    ):
        if value:
            query[name] = value

    if label:
        query["label"] = list(label)

    body = require_ok(
        request(
            runtime,
            "GET",
            sessions_path(project),
            access_token=authenticate(runtime),
            query=query,
        )
    )

    entries = require_list(_items(body, "sessions"), "session list")

    if not entries:
        click.echo("No sessions matched.")
        return

    print_table(
        ("ID", "Agent", "Customer", "Type", "Created"),
        (
            (
                text(e, "id"),
                text(e, "agentId"),
                text(e, "customerId"),
                _session_type(e),
                timestamp(e, "createdAt"),
            )
            for e in entries
        ),
    )

    if isinstance(body, Mapping) and text(body, "total") != "-":
        click.echo("")
        click.echo(f"  {len(entries)} of {text(body, 'total')} sessions")


@session.command(
    "view",
    epilog=examples(
        "parlant dashboard session view SESSION_ID",
    ),
)
@click.argument("session_id")
@click.pass_context
def sessions_view(ctx: click.Context, session_id: str) -> None:
    """Show a session and its turns.

    SESSION_ID is the session to show, as `session list` prints it.
    """
    runtime = runtime_for(ctx)
    project = require_project(runtime)

    body = require_mapping(
        require_ok(
            request(
                runtime,
                "GET",
                sessions_path(project, session_id, "view"),
                access_token=authenticate(runtime),
            )
        ),
        "session",
    )

    session = body.get("session")
    session = session if isinstance(session, Mapping) else body

    print_fields(
        [
            ("ID", text(session, "id")),
            ("AGENT", text(session, "agentId")),
            ("CUSTOMER", text(session, "customerId")),
            ("TYPE", _session_type(session)),
            ("CREATED", text(session, "createdAt")),
        ]
    )

    turns = body.get("turns")

    if isinstance(turns, list) and turns:
        click.echo("")
        print_table(
            ("Turn", "#", "State", "Inspectable", "Started"),
            (
                (
                    text(t, "turnId"),
                    text(t, "ordinal"),
                    text(t, "state"),
                    text(t, "inspectable"),
                    timestamp(t, "startedAt"),
                )
                for t in turns
            ),
        )


@session.command(
    "events",
    epilog=examples(
        "parlant dashboard session events SESSION_ID",
        "parlant dashboard session events SESSION_ID --kind tool",
        "parlant dashboard session events SESSION_ID --kind message --source customer",
    ),
)
@click.argument("session_id")
@click.option(
    "--kind",
    multiple=True,
    type=click.Choice(EVENT_KINDS),
    help="Filter by event kind (repeatable).",
)
@click.option(
    "--source", type=click.Choice(EVENT_SOURCES), default=None, help="Filter by event source."
)
@click.option("--trace", default=None, help="Filter by trace id.")
@click.option("--min-offset", default=None, help="Only events at or after this offset.")
@click.pass_context
def sessions_events(
    ctx: click.Context,
    session_id: str,
    kind: tuple[str, ...],
    source: str | None,
    trace: str | None,
    min_offset: str | None,
) -> None:
    """List the raw events in a session.

    SESSION_ID is the session to read, as `session list` prints it.
    """
    runtime = runtime_for(ctx)
    project = require_project(runtime)

    query: dict[str, Any] = {}

    if kind:
        query["kind"] = list(kind)
    if source:
        query["source"] = source
    if trace:
        query["traceId"] = trace
    if min_offset:
        query["minOffset"] = min_offset

    body = require_ok(
        request(
            runtime,
            "GET",
            sessions_path(project, session_id, "events"),
            access_token=authenticate(runtime),
            query=query,
        )
    )

    events = require_list(_items(body, "events"), "event list")

    if not events:
        click.echo("No events matched.")
        return

    print_table(
        ("Offset", "Kind", "Source", "Created", "Summary"),
        (
            (
                text(e, "offset"),
                text(e, "kind"),
                text(e, "source"),
                timestamp(e, "createdAt"),
                _event_summary(e),
            )
            for e in events
        ),
    )


@session.command(
    "summary",
    epilog=examples(
        "parlant dashboard session summary SESSION_ID",
    ),
)
@click.argument("session_id")
@click.pass_context
def sessions_summary(ctx: click.Context, session_id: str) -> None:
    """Summarize a session.

    SESSION_ID is the session to summarize, as `session list` prints it.
    """
    runtime = runtime_for(ctx)
    project = require_project(runtime)

    body = require_mapping(
        require_ok(
            request(
                runtime,
                "POST",
                sessions_path(project, session_id, "summary"),
                access_token=authenticate(runtime),
            )
        ),
        "session summary",
    )

    summary = text(body, "summary")

    if summary == "-":
        click.echo("No summary has been generated for this session yet.")
        return

    click.echo(summary)


@session.command(
    "inspect",
    epilog=examples(
        "parlant dashboard session inspect SESSION_ID --turn TURN_ID",
        "parlant dashboard session inspect SESSION_ID --turn TURN_ID --include-logs",
    ),
)
@click.argument("session_id")
@click.option("--turn", required=True, help="Opaque turn identifier.")
@click.option("--include-logs", is_flag=True, help="Include bounded log evidence.")
@click.pass_context
def sessions_inspect(ctx: click.Context, session_id: str, turn: str, include_logs: bool) -> None:
    """Show why one turn produced what it did.

    SESSION_ID is the session holding the turn, as `session list` prints it,
    and --turn says which turn within it.
    """
    runtime = runtime_for(ctx)
    project = require_project(runtime)

    # Omit false flags; this query option is enabled by its presence.
    query = {"includeLogs": "true"} if include_logs else {}

    body = require_mapping(
        require_ok(
            request(
                runtime,
                "GET",
                sessions_path(project, session_id, "turns", turn, "inspection"),
                access_token=authenticate(runtime),
                query=query,
            )
        ),
        "turn inspection",
    )

    turn_info = body.get("turn")
    turn_info = turn_info if isinstance(turn_info, Mapping) else {}

    trace = body.get("inspection")
    trace = trace.get("trace") if isinstance(trace, Mapping) else None
    trace = trace if isinstance(trace, Mapping) else {}

    print_fields(
        [
            ("TURN", text(turn_info, "turnId")),
            ("STATE", text(turn_info, "state")),
            ("TRACE", text(trace, "traceId")),
            ("STARTED", text(turn_info, "startedAt")),
            ("COMPLETED", text(turn_info, "completedAt")),
        ]
    )

    completeness = body.get("completeness")

    if isinstance(completeness, Mapping) and completeness.get("complete") is False:
        truncated = completeness.get("truncatedFields")
        listed = ", ".join(str(f) for f in truncated) if isinstance(truncated, list) else ""
        click.echo("")
        click.echo(f"  (evidence truncated: {listed or 'some fields'})")

    if trace:
        click.echo("")
        for name, value in trace.items():
            if isinstance(value, list) and value:
                click.echo(f"  {name}: {len(value)} entr{'y' if len(value) == 1 else 'ies'}")


def _items(body: Any, key: str) -> Any:
    """Extract records from a list or a paginated response."""
    if isinstance(body, Mapping):
        for candidate in (key, "items", "data"):
            value = body.get(candidate)
            if isinstance(value, list):
                return value
    return body


def _session_type(session: Any) -> str:
    """Format the session type as live or simulation."""
    if not isinstance(session, Mapping):
        return "-"
    flag = session.get("isSimulation")
    return "simulation" if flag else ("live" if flag is False else "-")


def _event_summary(event: Any) -> str:
    if not isinstance(event, Mapping):
        return "-"

    data = event.get("data")

    if isinstance(data, Mapping):
        for key in ("message", "text", "status", "toolName"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value

    return text(event, "kind")
