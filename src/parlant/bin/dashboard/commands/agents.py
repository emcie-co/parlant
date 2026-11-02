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

import json
from typing import Any, Mapping

import click
import httpx

from parlant.bin.dashboard.auth import authenticate
from parlant.bin.dashboard.client import (
    a2a_endpoint_path,
    a2a_endpoints_path,
    agents_path,
    refusal_code,
    request,
    require_ok,
    server_message,
)
from parlant.bin.dashboard.commands import dashboard
from parlant.bin.dashboard.commands._a2a import (
    SETTINGS_OPTIONS,
    a2a_entry,
    a2a_state,
    fetch_a2a_settings,
    no_address,
    print_a2a_settings,
    read_skills,
    settings_body,
)
from parlant.bin.dashboard.errors import DashboardError
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
    access_token = authenticate(runtime)

    agent = require_mapping(
        require_ok(
            request(
                runtime,
                "GET",
                agents_path(project, agent_id),
                access_token=access_token,
            )
        ),
        "agent",
    )
    a2a = a2a_state(fetch_a2a_settings(runtime, project, access_token), agent_id)

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
            ("A2A", a2a.summary),
            ("A2A ADDRESS", a2a.address),
            ("AGENT CARD", a2a.card),
        ]
    )


@agent.command(
    "card",
    epilog=examples(
        "parlant dashboard agent card AGENT_ID",
        "parlant dashboard agent card AGENT_ID > public/.well-known/agent-card.json",
    ),
)
@click.argument("agent_id")
@click.pass_context
def agents_card(ctx: click.Context, agent_id: str) -> None:
    """Print the agent's A2A card, the JSON personal AI agents read.

    AGENT_ID is the agent's id, as `agent list` prints it. Only the card is printed,
    so redirecting it to a file gives one your site can serve at
    /.well-known/agent-card.json, for hosts that can't redirect there instead.
    Fetch it again whenever the agent's settings change.
    """
    runtime = runtime_for(ctx)
    project = require_project(runtime)
    access_token = authenticate(runtime)

    settings = require_mapping(
        require_ok(request(runtime, "GET", a2a_endpoints_path(project), access_token=access_token)),
        "A2A settings",
    )
    card_url = a2a_state(settings, agent_id).card

    if card_url == "-" or a2a_entry(settings, agent_id) is None:
        raise no_address(agent_id)

    # The card is public: the A2A gateway serves it, not the Dashboard API.
    card = require_ok(request(runtime, "GET", card_url))
    click.echo(json.dumps(card, indent=2))


@agent.group("a2a")
def agent_a2a() -> None:
    """Read and change how personal AI agents see an agent over A2A.

    These are the settings in the Dashboard's Agent settings (the gear next to the
    agent's name): its site, and the description and organization its A2A card
    gives the AI agents that read it.
    """


@agent_a2a.command(
    "get",
    epilog=examples(
        "parlant dashboard agent a2a get AGENT_ID",
        "parlant dashboard agent a2a get AGENT_ID --project PROJECT_ID",
    ),
)
@click.argument("agent_id")
@click.pass_context
def agent_a2a_get(ctx: click.Context, agent_id: str) -> None:
    """Show the agent's A2A settings.

    AGENT_ID is the agent's id, as `agent list` prints it. A description or
    organization left unset shows what the card says instead.
    """
    runtime = runtime_for(ctx)
    project = require_project(runtime)
    settings, entry = _a2a_settings_of(runtime, project, authenticate(runtime), agent_id)

    print_a2a_settings(settings, entry)


@agent_a2a.command(
    "set",
    epilog=examples(
        "parlant dashboard agent a2a set AGENT_ID --site acme.com",
        'parlant dashboard agent a2a set AGENT_ID --organization "Acme" '
        '--description "Answers questions about Acme\'s orders and returns."',
        'parlant dashboard agent a2a set AGENT_ID --site ""',
        "parlant dashboard agent a2a set AGENT_ID --skills skills.json",
    ),
)
@click.argument("agent_id")
@click.option(
    "--site",
    help="Your site's bare domain, such as acme.com. The card links to it as the agent's provider. Empty removes it.",
)
@click.option(
    "--description",
    help="What the agent does, as the card tells AI agents. Empty restores the default.",
)
@click.option(
    "--organization",
    help="Who provides the agent, as the card tells AI agents. Empty uses the agent's name.",
)
@click.option(
    "--skills",
    "skills_file",
    help=(
        "A JSON file listing the skills the card tells AI agents about: up to 10 of "
        '{"id", "name", "description", "tags", and optionally "examples"}, as '
        'in [{"id": "orders", "name": "Orders", "description": "Places and tracks '
        'orders.", "tags": ["orders"]}]. "-" reads stdin; empty removes them, and the '
        "card falls back to one Conversation skill."
    ),
)
@click.option(
    "--sign-in-url",
    "sign_in_url",
    help=(
        "Your sign-in page, where the person behind an AI agent signs in so the agent "
        "can act on their account. A full https address. Empty removes it."
    ),
)
@click.pass_context
def agent_a2a_set(
    ctx: click.Context,
    agent_id: str,
    site: str | None,
    description: str | None,
    organization: str | None,
    sign_in_url: str | None,
    skills_file: str | None,
) -> None:
    """Change the agent's A2A settings, and print them as saved.

    AGENT_ID is the agent's id, as `agent list` prints it. Only the options you
    give change; an empty value ("") clears that setting. The agent must already
    have an A2A address: turn AI agents on in its Agent settings first.
    """
    given = {
        "site": site,
        "description": description,
        "organization": organization,
        "signInUrl": sign_in_url,
    }
    changes: dict[str, Any] = {
        key: value.strip() for key, value in given.items() if value is not None
    }
    if skills_file is not None:
        changes["skills"] = read_skills(skills_file.strip())

    if not changes:
        raise click.UsageError("Give at least one of " + ", ".join(SETTINGS_OPTIONS.values()) + ".")

    runtime = runtime_for(ctx)
    project = require_project(runtime)
    access_token = authenticate(runtime)
    settings, entry = _a2a_settings_of(runtime, project, access_token, agent_id)
    endpoint = entry.get("endpoint")

    if not isinstance(endpoint, Mapping):
        raise no_address(agent_id)

    response = request(
        runtime,
        "PUT",
        a2a_endpoint_path(project, agent_id),
        json_body=settings_body(endpoint, changes),
        access_token=access_token,
    )

    if response.status_code == 400:
        field = _refused_field(response)
        option = SETTINGS_OPTIONS.get(field or "")
        message = server_message(response) or "The Dashboard refused these settings."
        raise DashboardError(
            f"{message} Check {option}." if option else message,
            code=refusal_code(response) or None,
        )

    saved = require_mapping(require_ok(response), "A2A settings")
    print_a2a_settings(settings, saved)


def _a2a_settings_of(
    runtime: Any, project: str, access_token: str, agent_id: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """The project's A2A settings and the agent's entry in them."""
    settings = require_mapping(
        require_ok(request(runtime, "GET", a2a_endpoints_path(project), access_token=access_token)),
        "A2A settings",
    )
    entry = a2a_entry(settings, agent_id)

    if entry is None:
        raise DashboardError(
            f"Agent {agent_id} isn't in this project. Run `parlant dashboard agent list` "
            "to see its agents' ids.",
            code="AGENT_NOT_FOUND",
        )

    return settings, entry


def _refused_field(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None

    field = body.get("field") if isinstance(body, Mapping) else None

    return field if isinstance(field, str) else None


def _max_steps(agent: Any) -> str:
    """Read the step limit from either supported field name."""
    steps = text(agent, "maxSteps")

    return steps if steps != "-" else text(agent, "maxEngineIterations")


def _tags(agent: Any) -> str:
    tags = agent.get("tags") if isinstance(agent, Mapping) else None

    if not isinstance(tags, list) or not tags:
        return "-"

    return ", ".join(str(tag) for tag in tags)
