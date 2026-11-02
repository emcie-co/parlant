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

"""CLI project selection. Deploy and secrets live in test_dashboard_deployments.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, cast

import httpx
import pytest
from click.testing import CliRunner

from parlant.bin.cli import cli
from parlant.bin.dashboard.runtime import DashboardRuntime

Handler = Callable[[httpx.Request], httpx.Response]


def runtime(handler: Handler, project: str | None = "p1") -> DashboardRuntime:
    return DashboardRuntime(
        host="https://dashboard.test",
        http=httpx.Client(
            base_url="https://dashboard.test", transport=httpx.MockTransport(handler)
        ),
        credentials=cast(Any, None),
        config=cast(Any, None),
        open_browser=lambda url: False,
        sleep=lambda seconds: None,
        project=project,
    )


def _no_auth(monkeypatch: pytest.MonkeyPatch, *modules: str) -> None:
    for module in modules:
        mod = __import__(f"parlant.bin.dashboard.commands.{module}", fromlist=["authenticate"])
        monkeypatch.setattr(mod, "authenticate", lambda r: "token")


def _project(kind: str, *, repo: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"id": "p1", "name": "Acme", "tenantId": "Emcie", "projectKind": kind}
    if repo is not None:
        body["deployment"] = {"hostingMode": "managed", "githubRepoFullName": repo}
    return body


def test_no_project_context_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_auth(monkeypatch, "secrets")
    result = CliRunner().invoke(
        cli,
        ["dashboard", "secret", "list"],
        obj=runtime(lambda req: httpx.Response(500), project=None),
    )
    assert result.exit_code != 0
    assert "No project selected" in result.output
    assert "--project" in result.output


def test_project_flag_binds_on_a_subcommand_not_only_the_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_auth(monkeypatch, "secrets")

    def respond(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/projects/pX":
            return httpx.Response(200, json={**_project("managed", repo="a/b"), "id": "pX"})
        if req.url.path == "/projects/pX/deployments":
            return httpx.Response(200, json={"revision": "r1", "settings": {"secretNames": []}})
        return httpx.Response(500, json={"message": req.url.path})

    # --project given AFTER the subcommand (was previously rejected).
    result = CliRunner().invoke(
        cli, ["dashboard", "secret", "list", "--project", "pX"], obj=runtime(respond, project=None)
    )

    assert result.exit_code == 0, result.output
    assert "no deployment secrets" in result.output


def test_auth_login_has_no_project_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    result = CliRunner().invoke(cli, ["dashboard", "auth", "login", "--project", "pX"])
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


# --- A2A: how personal agents reach an agent --------------------------------

A2A_BASE = "https://agents.parlant.io/a2a"

BASIC_A2A_PLAN: dict[str, Any] = {
    "enabled": True,
    "messagesPerConversation": 5,
    "messagesPerEndpointPerDay": 200,
    "guestConversationsPerAddressPerMinute": 60,
    "requestsPerPersonalAgentUserPerMinute": 10,
    "maxReplyTimeoutSeconds": 90,
}


def _a2a_endpoint(
    slug: str, *, enabled: bool = True, callers: str = "verified-and-guests"
) -> dict[str, Any]:
    base = f"{A2A_BASE}/{slug}"
    return {
        "slug": slug,
        "enabled": enabled,
        "callers": callers,
        "urls": {
            "card": f"{base}/.well-known/agent-card.json",
            "interface": base,
            "howTo": f"{base}/how-to",
        },
    }


def _a2a_agent(
    agent_id: str,
    name: str,
    endpoint: dict[str, Any] | None,
    *,
    turns_on_at_next_deploy: bool = False,
) -> dict[str, Any]:
    return {
        "agentId": agent_id,
        "agentName": name,
        "turnsOnAtNextDeploy": turns_on_at_next_deploy,
        "endpoint": endpoint,
    }


def _a2a_settings(
    agents: list[dict[str, Any]],
    *,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The Dashboard's GET /projects/{id}/a2a/endpoints, as Backend serves it."""
    return {
        "plan": BASIC_A2A_PLAN if plan is None else plan,
        "projectCanUse": True,
        "agentsUnavailable": False,
        "publicBaseUrl": A2A_BASE,
        "agents": agents,
    }


def _agent_view(monkeypatch: pytest.MonkeyPatch, a2a: httpx.Response) -> Any:
    _no_auth(monkeypatch, "agents")

    def respond(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/projects/p1/agents/home":
            return httpx.Response(200, json={"id": "home", "name": "Home"})
        if req.url.path == "/projects/p1/a2a/endpoints":
            return a2a
        return httpx.Response(500, json={"message": req.url.path})

    return CliRunner().invoke(cli, ["dashboard", "agent", "view", "home"], obj=runtime(respond))


def _field(output: str, label: str) -> str:
    """The value printed beside a label, as print_fields lays it out."""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(label + " "):
            return stripped[len(label) :].strip()
    raise AssertionError(f"no {label} line in:\n{output}")


def test_that_agent_view_prints_the_agents_a2a_address_and_card_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _agent_view(
        monkeypatch,
        httpx.Response(
            200,
            json=_a2a_settings([_a2a_agent("home", "Home", _a2a_endpoint("acme-store-home"))]),
        ),
    )

    assert result.exit_code == 0, result.output
    assert _field(result.output, "NAME") == "Home"
    assert _field(result.output, "A2A") == "on: verified agents, and others as guests"
    assert _field(result.output, "A2A ADDRESS") == f"{A2A_BASE}/acme-store-home"
    assert (
        _field(result.output, "AGENT CARD")
        == f"{A2A_BASE}/acme-store-home/.well-known/agent-card.json"
    )


def test_that_agent_view_says_a_managed_agents_a2a_turns_on_with_the_next_deploy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _agent_view(
        monkeypatch,
        httpx.Response(
            200,
            json=_a2a_settings([_a2a_agent("home", "Home", None, turns_on_at_next_deploy=True)]),
        ),
    )

    assert result.exit_code == 0, result.output
    assert _field(result.output, "A2A") == "off; turns on with the next deploy"
    assert _field(result.output, "A2A ADDRESS") == "-"


def test_that_agent_view_says_a2a_is_off_when_the_agent_has_no_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    switched_off = _agent_view(
        monkeypatch,
        httpx.Response(
            200,
            json=_a2a_settings(
                [_a2a_agent("home", "Home", _a2a_endpoint("acme-store-home", enabled=False))]
            ),
        ),
    )
    never_on = _agent_view(
        monkeypatch, httpx.Response(200, json=_a2a_settings([_a2a_agent("home", "Home", None)]))
    )

    assert switched_off.exit_code == 0, switched_off.output
    assert _field(switched_off.output, "A2A") == "off"
    # The address stays reserved while switched off, so it is still worth showing.
    assert _field(switched_off.output, "A2A ADDRESS") == f"{A2A_BASE}/acme-store-home"
    assert never_on.exit_code == 0, never_on.output
    assert _field(never_on.output, "A2A") == "off"
    assert _field(never_on.output, "A2A ADDRESS") == "-"
    assert _field(never_on.output, "AGENT CARD") == "-"


def test_that_agent_view_says_a2a_is_not_in_the_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _agent_view(
        monkeypatch,
        httpx.Response(
            200,
            json=_a2a_settings([_a2a_agent("home", "Home", None)], plan={"enabled": False}),
        ),
    )

    assert result.exit_code == 0, result.output
    assert _field(result.output, "A2A") == "not in your plan"


def test_that_agent_view_still_shows_the_agent_when_its_a2a_settings_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An older Dashboard without the route, or one that can't answer it right now.
    for failure in (httpx.Response(404, json={"message": "Not found"}), httpx.Response(503)):
        result = _agent_view(monkeypatch, failure)

        assert result.exit_code == 0, result.output
        assert _field(result.output, "NAME") == "Home"
        assert _field(result.output, "A2A") == "-"
        assert _field(result.output, "A2A ADDRESS") == "-"


def test_that_project_describe_lists_each_agents_a2a_state_and_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_auth(monkeypatch, "projects")
    settings = _a2a_settings(
        [
            _a2a_agent("home", "Home", _a2a_endpoint("acme-store-home")),
            _a2a_agent("billing", "Billing", None),
        ]
    )

    def respond(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/projects/p1":
            return httpx.Response(200, json=_project("self-hosted"))
        if req.url.path == "/projects/p1/a2a/endpoints":
            return httpx.Response(200, json=settings)
        return httpx.Response(500, json={"message": req.url.path})

    result = CliRunner().invoke(
        cli, ["dashboard", "project", "describe", "p1"], obj=runtime(respond)
    )

    assert result.exit_code == 0, result.output
    rows = {
        line.split()[0]: line.split()[1:]
        for line in result.output.splitlines()
        if line.strip().startswith(("home", "billing"))
    }
    assert rows["home"] == ["on", f"{A2A_BASE}/acme-store-home"]
    assert rows["billing"] == ["off", "-"]
    assert "AGENT" in result.output and "ADDRESS" in result.output


def test_that_agent_card_prints_the_live_card_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_auth(monkeypatch, "agents")
    card = {"name": "Home", "supportedInterfaces": [{"url": f"{A2A_BASE}/acme-store-home"}]}
    settings = _a2a_settings([_a2a_agent("home", "Home", _a2a_endpoint("acme-store-home"))])

    def respond(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/projects/p1/a2a/endpoints":
            return httpx.Response(200, json=settings)
        # The card is public, served by the A2A gateway rather than the Dashboard API.
        if (
            req.url.host == "agents.parlant.io"
            and req.url.path == "/a2a/acme-store-home/.well-known/agent-card.json"
        ):
            return httpx.Response(200, json=card)
        return httpx.Response(500, json={"message": str(req.url)})

    result = CliRunner().invoke(cli, ["dashboard", "agent", "card", "home"], obj=runtime(respond))

    assert result.exit_code == 0, result.output
    # Only the card, so `> agent-card.json` writes a file a site can serve as is.
    assert json.loads(result.output) == card


def test_that_agent_card_says_how_to_turn_a2a_on_when_the_agent_has_no_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_auth(monkeypatch, "agents")
    settings = _a2a_settings([_a2a_agent("home", "Home", None)])

    def respond(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/projects/p1/a2a/endpoints":
            return httpx.Response(200, json=settings)
        return httpx.Response(500, json={"message": str(req.url)})

    result = CliRunner().invoke(cli, ["dashboard", "agent", "card", "home"], obj=runtime(respond))

    assert result.exit_code != 0
    assert "has no A2A address" in result.output
    assert "Agent settings" in result.output


# --- A2A: reading and writing an agent's A2A settings -----------------------


def _a2a_saved_endpoint(**overrides: Any) -> dict[str, Any]:
    """An endpoint as the settings route returns it, with what an Owner set."""
    return {
        **_a2a_endpoint("k3v9q2m7x1p8d4ta"),
        "site": "acme.com",
        "expiresAt": None,
        "card": {
            "mode": "auto",
            "description": "Answers questions about Acme.",
            "organization": "Acme",
        },
        "guestLimits": {"conversationsPerAddressPerMinute": 20, "idleExpiryHours": 48},
        "limits": {"messagesPerDay": 150},
        "replyTimeoutSeconds": 90,
        "effective": {
            "guestLimits": {
                "messagesPerConversation": 5,
                "idleExpiryHours": 48,
                "conversationsPerAddressPerMinute": 20,
            },
            "limits": {"requestsPerPersonalAgentUserPerMinute": 10, "messagesPerDay": 150},
            "replyTimeoutSeconds": 90,
        },
        **overrides,
    }


class _A2ADashboard:
    """The Dashboard's A2A settings routes, keeping what each PUT sent."""

    def __init__(self, endpoint: dict[str, Any] | None, refusal: httpx.Response | None = None):
        self.endpoint = endpoint
        self.refusal = refusal
        self.puts: list[dict[str, Any]] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path == "/projects/p1/a2a/endpoints":
            return httpx.Response(
                200, json=_a2a_settings([_a2a_agent("home", "Home", self.endpoint)])
            )
        if req.method == "PUT" and req.url.path == "/projects/p1/a2a/endpoints/home":
            body = json.loads(req.content)
            self.puts.append(body)
            if self.refusal is not None:
                return self.refusal
            assert self.endpoint is not None
            self.endpoint = {
                **self.endpoint,
                "site": body.get("site"),
                "card": body["card"],
                "guestLimits": body.get("guestLimits", {}),
                "limits": body.get("limits", {}),
                "replyTimeoutSeconds": body.get("replyTimeoutSeconds"),
            }
            return httpx.Response(200, json=_a2a_agent("home", "Home", self.endpoint))
        return httpx.Response(500, json={"message": f"{req.method} {req.url}"})


def _a2a(monkeypatch: pytest.MonkeyPatch, dashboard: _A2ADashboard, *args: str) -> Any:
    _no_auth(monkeypatch, "agents")
    return CliRunner().invoke(cli, ["dashboard", "agent", "a2a", *args], obj=runtime(dashboard))


def test_that_agent_a2a_get_prints_the_site_description_and_organization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _a2a(monkeypatch, _A2ADashboard(_a2a_saved_endpoint()), "get", "home")

    assert result.exit_code == 0, result.output
    assert _field(result.output, "A2A") == "on: verified agents, and others as guests"
    assert _field(result.output, "SITE") == "acme.com"
    assert _field(result.output, "DESCRIPTION") == "Answers questions about Acme."
    assert _field(result.output, "ORGANIZATION") == "Acme"
    assert _field(result.output, "REPLY WAIT") == "90 seconds"
    assert (
        _field(result.output, "AGENT CARD")
        == f"{A2A_BASE}/k3v9q2m7x1p8d4ta/.well-known/agent-card.json"
    )


def test_that_agent_a2a_get_shows_the_default_description_and_the_agents_name_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _a2a_saved_endpoint(site=None, card={"mode": "auto"}, replyTimeoutSeconds=None)
    endpoint["effective"]["replyTimeoutSeconds"] = 60

    result = _a2a(monkeypatch, _A2ADashboard(endpoint), "get", "home")

    assert result.exit_code == 0, result.output
    assert _field(result.output, "SITE") == "-"
    assert _field(result.output, "DESCRIPTION") == "Answers questions. (the default)"
    assert _field(result.output, "ORGANIZATION") == "Home (the agent's name)"
    assert _field(result.output, "REPLY WAIT") == "60 seconds (the default)"


def test_that_agent_a2a_set_changes_only_the_given_fields_and_keeps_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = _A2ADashboard(_a2a_saved_endpoint())

    result = _a2a(monkeypatch, dashboard, "set", "home", "--site", "shop.acme.com")

    assert result.exit_code == 0, result.output
    assert dashboard.puts == [
        {
            "enabled": True,
            "slug": "k3v9q2m7x1p8d4ta",
            "callers": "verified-and-guests",
            "site": "shop.acme.com",
            "card": {
                "mode": "auto",
                "description": "Answers questions about Acme.",
                "organization": "Acme",
            },
            "guestLimits": {"conversationsPerAddressPerMinute": 20, "idleExpiryHours": 48},
            "limits": {"messagesPerDay": 150},
            "replyTimeoutSeconds": 90,
        }
    ]
    # It prints the settings as saved.
    assert _field(result.output, "SITE") == "shop.acme.com"
    assert _field(result.output, "ORGANIZATION") == "Acme"


def test_that_agent_a2a_set_with_an_empty_value_clears_the_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = _A2ADashboard(_a2a_saved_endpoint())

    result = _a2a(
        monkeypatch,
        dashboard,
        "set",
        "home",
        "--site",
        "",
        "--description",
        "",
        "--organization",
        "",
    )

    assert result.exit_code == 0, result.output
    assert dashboard.puts[0]["site"] is None
    assert dashboard.puts[0]["card"] == {"mode": "auto"}
    assert _field(result.output, "DESCRIPTION") == "Answers questions. (the default)"
    assert _field(result.output, "ORGANIZATION") == "Home (the agent's name)"


def test_that_agent_a2a_set_says_how_to_turn_a2a_on_when_the_agent_has_no_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = _A2ADashboard(None)

    result = _a2a(monkeypatch, dashboard, "set", "home", "--organization", "Acme")

    assert result.exit_code != 0
    assert dashboard.puts == []
    assert "has no A2A address" in result.output
    assert "Agent settings" in result.output


def test_that_agent_a2a_set_prints_the_dashboards_refusal_for_an_invalid_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = _A2ADashboard(
        _a2a_saved_endpoint(),
        refusal=httpx.Response(
            400,
            json={
                "code": "invalid",
                "message": "Enter your site's domain, such as example.com.",
                "field": "site",
            },
        ),
    )

    result = _a2a(monkeypatch, dashboard, "set", "home", "--site", "https://acme.com/")

    assert result.exit_code != 0
    assert "Enter your site's domain, such as example.com." in result.output
    assert "--site" in result.output


def test_that_agent_a2a_set_without_any_option_says_which_options_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = _A2ADashboard(_a2a_saved_endpoint())

    result = _a2a(monkeypatch, dashboard, "set", "home")

    assert result.exit_code != 0
    assert dashboard.puts == []
    for option in ("--site", "--description", "--organization"):
        assert option in result.output


# --- A2A: the sign-in page for the person behind an AI agent ----------------

SIGN_IN_PAGE = "https://acme.com/a2a-sign-in"


def test_that_agent_a2a_get_prints_the_sign_in_url(monkeypatch: pytest.MonkeyPatch) -> None:
    with_page = _a2a(
        monkeypatch, _A2ADashboard(_a2a_saved_endpoint(signInUrl=SIGN_IN_PAGE)), "get", "home"
    )
    without = _a2a(monkeypatch, _A2ADashboard(_a2a_saved_endpoint()), "get", "home")

    assert with_page.exit_code == 0, with_page.output
    assert _field(with_page.output, "SIGN-IN URL") == SIGN_IN_PAGE
    assert _field(without.output, "SIGN-IN URL") == "-"


def test_that_agent_a2a_set_changes_the_sign_in_url_and_keeps_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = _A2ADashboard(_a2a_saved_endpoint())

    result = _a2a(monkeypatch, dashboard, "set", "home", "--sign-in-url", SIGN_IN_PAGE)

    assert result.exit_code == 0, result.output
    assert dashboard.puts == [
        {
            "enabled": True,
            "slug": "k3v9q2m7x1p8d4ta",
            "callers": "verified-and-guests",
            "site": "acme.com",
            "signInUrl": SIGN_IN_PAGE,
            "card": {
                "mode": "auto",
                "description": "Answers questions about Acme.",
                "organization": "Acme",
            },
            "guestLimits": {"conversationsPerAddressPerMinute": 20, "idleExpiryHours": 48},
            "limits": {"messagesPerDay": 150},
            "replyTimeoutSeconds": 90,
        }
    ]


def test_that_agent_a2a_set_with_an_empty_sign_in_url_clears_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = _A2ADashboard(_a2a_saved_endpoint(signInUrl=SIGN_IN_PAGE))

    result = _a2a(monkeypatch, dashboard, "set", "home", "--sign-in-url", "")

    assert result.exit_code == 0, result.output
    assert dashboard.puts[0]["signInUrl"] is None


def test_that_agent_a2a_set_names_the_sign_in_url_option_when_the_dashboard_refuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = _A2ADashboard(
        _a2a_saved_endpoint(),
        refusal=httpx.Response(
            400,
            json={
                "code": "invalid",
                "message": "Enter your sign-in page's full https address, such as https://example.com/sign-in.",
                "field": "signInUrl",
            },
        ),
    )

    result = _a2a(monkeypatch, dashboard, "set", "home", "--sign-in-url", "http://acme.com/x")

    assert result.exit_code != 0
    assert "full https address" in result.output
    assert "--sign-in-url" in result.output


# --- A2A: the card's skills (2026-11-01) ------------------------------------

ORDERS_SKILL: dict[str, Any] = {
    "id": "orders",
    "name": "Orders",
    "description": "Places and tracks orders.",
    "tags": ["orders"],
    "examples": ["Where is my order?"],
}


def test_that_agent_a2a_set_writes_skills_from_a_json_file_and_keeps_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    skills_file = tmp_path / "skills.json"
    skills_file.write_text(json.dumps([ORDERS_SKILL]))
    dashboard = _A2ADashboard(_a2a_saved_endpoint())

    result = _a2a(monkeypatch, dashboard, "set", "home", "--skills", str(skills_file))

    assert result.exit_code == 0, result.output
    assert dashboard.puts[0]["card"] == {
        "mode": "auto",
        "description": "Answers questions about Acme.",
        "organization": "Acme",
        "skills": [ORDERS_SKILL],
    }
    assert dashboard.puts[0]["site"] == "acme.com"
    assert dashboard.puts[0]["limits"] == {"messagesPerDay": 150}


def test_that_agent_a2a_set_with_an_empty_skills_value_removes_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _a2a_saved_endpoint()
    endpoint["card"] = {**endpoint["card"], "skills": [ORDERS_SKILL]}
    dashboard = _A2ADashboard(endpoint)

    result = _a2a(monkeypatch, dashboard, "set", "home", "--skills", "")

    assert result.exit_code == 0, result.output
    assert "skills" not in dashboard.puts[0]["card"]


@pytest.mark.parametrize(
    ("content", "complaint"),
    [
        ("not json", "JSON"),
        (json.dumps({"id": "orders"}), "list"),
        (json.dumps([{"id": "orders", "name": "Orders", "tags": []}]), "description"),
    ],
)
def test_that_agent_a2a_set_says_what_is_wrong_with_a_skills_file_before_saving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str, complaint: str
) -> None:
    skills_file = tmp_path / "skills.json"
    skills_file.write_text(content)
    dashboard = _A2ADashboard(_a2a_saved_endpoint())

    result = _a2a(monkeypatch, dashboard, "set", "home", "--skills", str(skills_file))

    assert result.exit_code != 0
    assert complaint in result.output
    assert "--skills" in result.output
    assert dashboard.puts == []


def test_that_agent_a2a_get_lists_the_skills_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint = _a2a_saved_endpoint()
    endpoint["card"] = {
        **endpoint["card"],
        "skills": [ORDERS_SKILL, {**ORDERS_SKILL, "id": "returns", "name": "Returns"}],
    }

    with_skills = _a2a(monkeypatch, _A2ADashboard(endpoint), "get", "home")
    without = _a2a(monkeypatch, _A2ADashboard(_a2a_saved_endpoint()), "get", "home")

    assert with_skills.exit_code == 0, with_skills.output
    assert _field(with_skills.output, "SKILLS") == "Orders, Returns"
    assert _field(without.output, "SKILLS") == "Conversation (the default)"
