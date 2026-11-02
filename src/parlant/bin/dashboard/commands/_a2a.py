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
"""How personal agents reach a project's agents over A2A, read from the Dashboard."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import click

from parlant.bin.dashboard.client import a2a_endpoints_path, request
from parlant.bin.dashboard.commands._shared import print_fields
from parlant.bin.dashboard.errors import DashboardError
from parlant.bin.dashboard.runtime import DashboardRuntime

CALLERS = {
    "verified": "verified agents only",
    "verified-and-guests": "verified agents, and others as guests",
    "guests": "everyone as a guest",
}


@dataclass(frozen=True)
class A2AState:
    """One agent's A2A state, as `agent view` and `project describe` show it."""

    # Full sentence-like state, e.g. "on: verified agents, and others as guests".
    summary: str
    # One word for a table column, e.g. "on" or "not-in-plan".
    short: str
    address: str
    card: str


UNKNOWN = A2AState(summary="-", short="-", address="-", card="-")

# What the card says when settings give no description (Backend A2A_DEFAULT_CARD_DESCRIPTION).
DEFAULT_DESCRIPTION = "Answers questions."

# `agent a2a set` options, by the settings field each one writes.
SETTINGS_OPTIONS = {
    "site": "--site",
    "description": "--description",
    "organization": "--organization",
    "signInUrl": "--sign-in-url",
    "skills": "--skills",
}

SKILL_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
MAX_SKILLS = 10


def read_skills(value: str) -> list[dict[str, Any]] | None:
    """The skills in the JSON file at ``value`` (``-`` reads stdin), or None for "" to
    remove them. Checked here against the Dashboard's own rules, so a bad file fails
    before anything is sent."""
    if value == "":
        return None

    def refuse(problem: str) -> click.UsageError:
        return click.UsageError(f"--skills: {problem}")

    try:
        text = sys.stdin.read() if value == "-" else Path(value).read_text()
    except OSError as error:
        raise refuse(f"can't read {value} ({error.strerror}).")

    try:
        skills = json.loads(text)
    except ValueError:
        raise refuse("the file isn't valid JSON.")

    if not isinstance(skills, list) or not 1 <= len(skills) <= MAX_SKILLS:
        raise refuse(
            f"expected a JSON list of 1 to {MAX_SKILLS} skills, each "
            '{"id", "name", "description", "tags", and optionally "examples"}.'
        )

    for number, skill in enumerate(skills, start=1):
        if not isinstance(skill, dict):
            raise refuse(f"skill {number} isn't an object.")
        if not isinstance(skill.get("id"), str) or not SKILL_ID.match(skill["id"]):
            raise refuse(
                f"skill {number}'s id must be lowercase letters, digits and dashes, "
                "starting with a letter or digit."
            )
        for field in ("name", "description"):
            if not isinstance(skill.get(field), str) or not skill[field].strip():
                raise refuse(f"skill {number} has no {field}.")
        for field, required in (("tags", True), ("examples", False)):
            items = skill.get(field)
            if items is None and not required:
                continue
            if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
                raise refuse(f"skill {number}'s {field} must be a list of strings.")
        unknown = set(skill) - {"id", "name", "description", "tags", "examples"}
        if unknown:
            raise refuse(
                f"skill {number} has fields the Dashboard doesn't take: {', '.join(sorted(unknown))}."
            )

    return skills


def no_address(agent_id: str) -> DashboardError:
    return DashboardError(
        f"Agent {agent_id} has no A2A address, so it has no card yet. "
        "Turn on AI agents in its Agent settings in the Dashboard (the gear next "
        "to its name); a managed agent gets an address with its project's next deploy.",
        code="NO_A2A_ADDRESS",
    )


def settings_body(endpoint: Mapping[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    """The whole record to save: the saved settings with only `changes` applied.

    The route replaces every field, so everything else is sent back as saved; an
    empty change clears its field. The address and callers are sent for Dashboards
    that still require them; newer ones ignore both.
    """
    card = {
        key: value
        for key, value in dict(endpoint.get("card") or {"mode": "auto"}).items()
        if key not in changes
    }
    for key in ("description", "organization", "skills"):
        if changes.get(key):
            card[key] = changes[key]

    site = changes["site"] if "site" in changes else endpoint.get("site")
    # Sent only when changed or already saved: older Dashboards refuse it.
    sign_in: dict[str, Any] = (
        {"signInUrl": changes["signInUrl"] or None}
        if "signInUrl" in changes
        else {"signInUrl": endpoint["signInUrl"]}
        if endpoint.get("signInUrl")
        else {}
    )

    return {
        "enabled": endpoint.get("enabled") is True,
        "slug": endpoint.get("slug"),
        "callers": endpoint.get("callers"),
        "site": site or None,
        **sign_in,
        "card": card,
        "guestLimits": dict(endpoint.get("guestLimits") or {}),
        "limits": dict(endpoint.get("limits") or {}),
        "replyTimeoutSeconds": endpoint.get("replyTimeoutSeconds"),
    }


def print_a2a_settings(settings: Mapping[str, Any], entry: Mapping[str, Any]) -> None:
    """One agent's A2A settings, as `agent a2a get` and `set` print them."""
    agent_id = str(entry.get("agentId"))
    endpoint = _mapping(entry.get("endpoint"))
    card = _mapping(endpoint.get("card"))
    effective = _mapping(endpoint.get("effective"))
    state = a2a_state(settings, agent_id)

    description = card.get("description") or f"{DEFAULT_DESCRIPTION} (the default)"
    name = entry.get("agentName") or agent_id
    organization = card.get("organization") or f"{name} (the agent's name)"
    wait = effective.get("replyTimeoutSeconds")
    reply_wait = f"{wait} seconds" if isinstance(wait, int) else "-"
    if isinstance(wait, int) and not isinstance(endpoint.get("replyTimeoutSeconds"), int):
        reply_wait += " (the default)"

    print_fields(
        [
            ("A2A", state.summary),
            ("SITE", str(endpoint.get("site") or "-")),
            ("SIGN-IN URL", str(endpoint.get("signInUrl") or "-")),
            ("SKILLS", _skill_names(card)),
            ("DESCRIPTION", str(description)),
            ("ORGANIZATION", str(organization)),
            ("REPLY WAIT", reply_wait),
            ("A2A ADDRESS", state.address),
            ("AGENT CARD", state.card),
        ]
    )


def fetch_a2a_settings(
    runtime: DashboardRuntime, project_id: str, access_token: str
) -> Mapping[str, Any] | None:
    """The project's A2A settings, or None when the Dashboard can't give them.

    An older Dashboard has no such route, and a failure here mustn't hide the rest
    of what a command shows.
    """
    try:
        response = request(
            runtime, "GET", a2a_endpoints_path(project_id), access_token=access_token
        )
    except DashboardError:
        return None

    if not response.is_success:
        return None

    body = response.json()

    return body if isinstance(body, Mapping) else None


def a2a_entry(settings: Mapping[str, Any], agent_id: str) -> Mapping[str, Any] | None:
    agents = settings.get("agents")

    if not isinstance(agents, list):
        return None

    return next(
        (
            entry
            for entry in agents
            if isinstance(entry, Mapping) and entry.get("agentId") == agent_id
        ),
        None,
    )


def a2a_state(settings: Mapping[str, Any] | None, agent_id: str) -> A2AState:
    if settings is None or settings.get("projectCanUse") is False:
        return UNKNOWN

    entry = a2a_entry(settings, agent_id)
    endpoint = entry.get("endpoint") if entry else None
    urls = endpoint.get("urls") if isinstance(endpoint, Mapping) else None
    address = _url(urls, "interface")
    card = _url(urls, "card")
    plan = settings.get("plan")

    if isinstance(plan, Mapping) and plan.get("enabled") is False:
        return A2AState(summary="not in your plan", short="not-in-plan", address=address, card=card)

    if isinstance(endpoint, Mapping) and endpoint.get("enabled") is True:
        callers = CALLERS.get(str(endpoint.get("callers")), str(endpoint.get("callers")))
        return A2AState(summary=f"on: {callers}", short="on", address=address, card=card)

    if endpoint is None and entry is not None and entry.get("turnsOnAtNextDeploy") is True:
        return A2AState(
            summary="off; turns on with the next deploy",
            short="on-next-deploy",
            address="-",
            card="-",
        )

    return A2AState(summary="off", short="off", address=address, card=card)


def _skill_names(card: Mapping[str, Any]) -> str:
    """The written skills' names, or the one the card lists when none is written."""
    skills = card.get("skills")
    names = [
        str(skill.get("name"))
        for skill in (skills if isinstance(skills, list) else [])
        if isinstance(skill, Mapping)
    ]
    return ", ".join(names) if names else "Conversation (the default)"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _url(urls: Any, key: str) -> str:
    value = urls.get(key) if isinstance(urls, Mapping) else None

    return value if isinstance(value, str) and value else "-"
