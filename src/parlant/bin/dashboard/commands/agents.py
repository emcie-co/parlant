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
"""``parlant dashboard agent`` — the agents in a project."""

from __future__ import annotations

from typing import Any, Mapping

import click

from parlant.bin.dashboard.auth import authenticate
from parlant.bin.dashboard.client import agents_path, request, require_ok
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


@dashboard.group()
def agent() -> None:
    """Inspect the agents in a project."""


@agent.command(
    "list",
    epilog=examples(
        "parlant dashboard agent list",
        "parlant dashboard agent list --project PROJECT_ID",
    ),
)
@click.pass_context
def agents_list(ctx: click.Context) -> None:
    """List the project's agents."""
    runtime = runtime_for(ctx)
    project = require_project(runtime)

    entries = require_list(
        require_ok(
            request(runtime, "GET", agents_path(project), access_token=authenticate(runtime))
        ),
        "agent list",
    )

    if not entries:
        click.echo("This project has no agents yet.")
        return

    print_table(
        ("Name", "ID", "Composition", "Output", "Sessions"),
        (
            (
                text(e, "name"),
                text(e, "id"),
                text(e, "compositionMode"),
                text(e, "messageOutputMode"),
                text(e, "sessionsCount"),
            )
            for e in entries
        ),
    )


@agent.command(
    "view",
    epilog=examples(
        "parlant dashboard agent view AGENT_ID",
        "parlant dashboard agent view AGENT_ID --project PROJECT_ID",
    ),
)
@click.argument("agent_id")
@click.pass_context
def agents_view(ctx: click.Context, agent_id: str) -> None:
    """Show one agent in full.

    AGENT_ID is the agent's id, as `agent list` prints it.
    """
    runtime = runtime_for(ctx)
    project = require_project(runtime)

    agent = require_mapping(
        require_ok(
            request(
                runtime,
                "GET",
                agents_path(project, agent_id),
                access_token=authenticate(runtime),
            )
        ),
        "agent",
    )

    print_fields(
        [
            ("ID", text(agent, "id")),
            ("NAME", text(agent, "name")),
            ("DESCRIPTION", text(agent, "description")),
            ("COMPOSITION", text(agent, "compositionMode")),
            ("OUTPUT MODE", text(agent, "messageOutputMode")),
            ("MAX STEPS", _max_steps(agent)),
            ("EFFORT", text(agent, "effort")),
            ("MODERATION", text(agent, "moderation")),
            ("MESSAGE REVIEW", text(agent, "messageReview")),
            ("TOOL REVIEW", text(agent, "toolReview")),
            ("TAGS", _tags(agent)),
        ]
    )


def _max_steps(agent: Any) -> str:
    """Read the step limit from either supported field name."""
    steps = text(agent, "maxSteps")

    return steps if steps != "-" else text(agent, "maxEngineIterations")


def _tags(agent: Any) -> str:
    tags = agent.get("tags") if isinstance(agent, Mapping) else None

    if not isinstance(tags, list) or not tags:
        return "-"

    return ", ".join(str(tag) for tag in tags)
