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

"""``parlant dashboard trace`` — what the engine did for one trace."""

from __future__ import annotations

from typing import Mapping

import click

from parlant.bin.dashboard.auth import authenticate
from parlant.bin.dashboard.client import project_path, request, require_ok
from parlant.bin.dashboard.commands import dashboard
from parlant.bin.dashboard.commands._shared import (
    print_fields,
    require_mapping,
    require_project,
    runtime_for,
    text,
)
from parlant.bin.help import examples


@dashboard.group()
def trace() -> None:
    """Inspect engine traces."""


@trace.command(
    "view",
    epilog=examples(
        "parlant dashboard trace view TRACE_ID",
        "parlant dashboard trace view TRACE_ID --project PROJECT_ID",
    ),
)
@click.argument("trace_id")
@click.pass_context
def traces_view(ctx: click.Context, trace_id: str) -> None:
    """Show one trace.

    TRACE_ID is the trace carried by a session's events; `session events`
    prints it, and its --trace option filters by it.
    """
    runtime = runtime_for(ctx)
    project = require_project(runtime)

    body = require_mapping(
        require_ok(
            request(
                runtime,
                "GET",
                project_path(project, "inspect", trace_id),
                access_token=authenticate(runtime),
            )
        ),
        "trace",
    )

    trace = body.get("trace")
    trace = trace if isinstance(trace, Mapping) else body

    print_fields(
        [
            ("TRACE", text(trace, "traceId") if text(trace, "traceId") != "-" else trace_id),
            ("AGENT", text(trace, "agentId")),
            ("STATUS", text(trace, "status")),
            ("STARTED", text(trace, "timestamp")),
            ("UPDATED", text(trace, "updatedAt")),
        ]
    )

    click.echo("")

    for name, value in trace.items():
        if isinstance(value, list) and value:
            click.echo(f"  {name}: {len(value)} entr{'y' if len(value) == 1 else 'ies'}")
