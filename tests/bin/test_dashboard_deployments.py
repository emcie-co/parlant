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

"""Deploying, watching, stopping and configuring a managed project through the
Backend deployments API (`/projects/{id}/deployments`).

`FakeDeployments` is a small working implementation of that API: it keeps a
configuration revision, one work slot, attempts that progress when read, an
operation-id receipt per request (so a repeated id replays), write-only secret
names and cursor-paged logs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, cast

import httpx
import pytest
from click.testing import CliRunner

from parlant.bin.cli import cli
from parlant.bin.dashboard.runtime import DashboardRuntime

BASE = "/projects/p1/deployments"
SHA = "a" * 40
LEGACY = re.compile(r"^/projects/[^/]+/(builds|deploys)(/|$)|^/projects/[^/]+/deployment(/|$)")


@dataclass
class FakeDeployments:
    kind: str = "managed"
    repository: str = "acme/agent"
    revision: int = 1
    secret_names: list[str] = field(default_factory=list)
    attempts: dict[str, dict[str, Any]] = field(default_factory=dict)
    progress: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    active: str | None = None
    serving: str | None = None
    receipts: dict[str, tuple[str, Any, dict[str, Any]]] = field(default_factory=dict)
    logs: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    producer_done: set[tuple[str, str]] = field(default_factory=set)
    retained_from: dict[tuple[str, str], int] = field(default_factory=dict)
    requests: list[str] = field(default_factory=list)
    secret_values: dict[str, str] = field(default_factory=dict)
    lose_next_mutation_response: bool = False
    next_mutation_outcome_unknown: bool = False
    revision_moves_before_next_mutation: bool = False
    stopped: list[str] = field(default_factory=list)
    log_channels: list[str] = field(default_factory=list)

    # -- helpers for tests -------------------------------------------------

    def next_states(self, attempt_id: str, *states: dict[str, Any]) -> None:
        self.progress[attempt_id] = list(states)

    def paths(self) -> list[str]:
        return [r.split(" ", 1)[1] for r in self.requests]

    # -- the API -------------------------------------------------------------

    def settings(self) -> dict[str, Any]:
        return {
            "source": {"installationId": "i1", "repository": self.repository, "branch": "main"},
            "buildCommand": "",
            "runCommand": "python app.py" if self.repository else "",
            "buildEnv": [],
            "runEnv": [{"name": "MODE", "value": "prod"}],
            "secretNames": sorted(self.secret_names),
        }

    def overview(self) -> dict[str, Any]:
        return {
            "projectName": "Acme",
            "revision": f"rev-{self.revision}",
            "settings": self.settings(),
            "servingId": self.serving,
            "activeId": self.active,
            "canUpdate": True,
            "settingsDifferFromLive": False,
        }

    def __call__(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        self.requests.append(f"{req.method} {path}")
        if LEGACY.match(path):
            return httpx.Response(404, json={"message": "Route not found"})
        if path == "/projects/p1":
            return httpx.Response(
                200,
                json={"id": "p1", "name": "Acme", "tenantId": "Emcie", "projectKind": self.kind},
            )
        if not path.startswith(BASE):
            return httpx.Response(500, json={"message": f"unexpected {path}"})
        if self.kind != "managed":
            return refusal(501, "unavailable", "Only managed projects have deployments.")
        rest = path[len(BASE) :]
        body = json.loads(req.content) if req.content else None

        if req.method == "GET" and rest == "":
            return httpx.Response(200, json=self.overview())
        if req.method == "POST" and rest == "/attempts":
            return self.mutate("deploy", body, self.deploy)
        if req.method == "PUT" and rest == "/settings":
            return self.mutate("save", body, self.save)
        if req.method == "GET" and rest == "/attempts":
            items = sorted(self.attempts.values(), key=lambda a: a["startedAt"], reverse=True)
            return httpx.Response(200, json={"items": items, "nextCursor": None})
        match = re.fullmatch(r"/attempts/([^/]+)(/stop|/logs)?", rest)
        if match:
            attempt_id, action = match.groups()
            if attempt_id not in self.attempts:
                return refusal(404, "not-found", "This deployment could not be found.")
            if action == "/stop" and req.method == "POST":
                return self.stop(attempt_id)
            if action == "/logs" and req.method == "GET":
                return self.read_logs(attempt_id, req)
            if action is None and req.method == "GET":
                return httpx.Response(200, json=self.read_attempt(attempt_id))
        match = re.fullmatch(r"/operations/([^/]+)", rest)
        if match and req.method == "GET":
            receipt = self.receipts.get(match.group(1))
            if receipt is None:
                return httpx.Response(200, json={"status": "not-found"})
            return httpx.Response(200, json=receipt[2])
        return httpx.Response(500, json={"message": f"unexpected {req.method} {path}"})

    def mutate(
        self,
        method: str,
        body: Any,
        admit: Callable[[dict[str, Any]], tuple[httpx.Response, dict[str, Any] | None]],
    ) -> httpx.Response:
        operation_id = body["operationId"]
        payload = {k: v for k, v in body.items() if k != "operationId"}
        if operation_id in self.receipts:
            stored_method, stored_payload, stored = self.receipts[operation_id]
            if (stored_method, stored_payload) != (method, payload):
                return refusal(409, "conflict", "This operation ID was already used.")
            return httpx.Response(200, json=stored["attempt" if method == "deploy" else "overview"])
        if self.revision_moves_before_next_mutation:
            self.revision_moves_before_next_mutation = False
            self.revision += 1
        if body["expectedRevision"] != f"rev-{self.revision}":
            return refusal(409, "conflict", "Settings changed in another window.")
        if self.active is not None:
            return refusal(409, "busy", "Wait for the active deployment to finish.")
        response, result = admit(body)
        if result is not None:
            self.receipts[operation_id] = (method, payload, result)
        if self.next_mutation_outcome_unknown and result is not None:
            self.next_mutation_outcome_unknown = False
            return refusal(409, "unknown-outcome", "Check the original operation's outcome.")
        if self.lose_next_mutation_response and result is not None:
            self.lose_next_mutation_response = False
            raise httpx.ConnectError("connection reset")
        return response

    def deploy(self, body: dict[str, Any]) -> tuple[httpx.Response, dict[str, Any] | None]:
        commit = body.get("commit")
        if commit is not None and not re.fullmatch(r"[0-9a-f]{40}", commit):
            return refusal(400, "validation", "A full 40-character commit SHA is required."), None
        if not self.repository:
            return refusal(422, "source-unavailable", "Connect a repository first."), None
        attempt_id = f"att{len(self.attempts) + 1}"
        attempt = {
            "id": attempt_id,
            "commit": {"sha": commit, "message": "Fix greeting"} if commit else None,
            "source": self.settings()["source"],
            "trigger": "commit" if commit else "manual",
            "status": "in-progress",
            "phase": "building",
            "failure": None,
            "startedAt": f"2026-09-28T10:0{len(self.attempts)}:00.000Z",
            "finishedAt": None,
            "capturedSettings": None,
        }
        self.attempts[attempt_id] = attempt
        self.active = attempt_id
        return httpx.Response(200, json=attempt), {"status": "deployed", "attempt": attempt}

    def save(self, body: dict[str, Any]) -> tuple[httpx.Response, dict[str, Any] | None]:
        if body["settings"]["source"] != self.settings()["source"]:
            return refusal(400, "validation", "The source must be sent unchanged."), None
        for change in body["secrets"]:
            if change["kind"] == "set":
                self.secret_values[change["name"]] = change["value"]
                if change["name"] not in self.secret_names:
                    self.secret_names.append(change["name"])
            elif change["name"] in self.secret_names:
                self.secret_names.remove(change["name"])
                self.secret_values.pop(change["name"], None)
        self.revision += 1
        overview = self.overview()
        return httpx.Response(200, json=overview), {"status": "saved", "overview": overview}

    def read_attempt(self, attempt_id: str) -> dict[str, Any]:
        attempt = self.attempts[attempt_id]
        queue = self.progress.get(attempt_id)
        if queue and attempt["status"] == "in-progress":
            attempt.update(queue.pop(0))
            if attempt["status"] != "in-progress":
                attempt["finishedAt"] = "2026-09-28T10:09:00.000Z"
                if self.active == attempt_id:
                    self.active = None
                if attempt["status"] == "completed":
                    self.serving = attempt_id
        return attempt

    def stop(self, attempt_id: str) -> httpx.Response:
        attempt = self.attempts[attempt_id]
        self.stopped.append(attempt_id)
        if attempt["status"] == "in-progress":
            self.next_states(
                attempt_id,
                {
                    "status": "failed",
                    "phase": None,
                    "failure": {"kind": "stopped", "message": "Stopped by a user."},
                },
            )
        return httpx.Response(200, json=attempt)

    def read_logs(self, attempt_id: str, req: httpx.Request) -> httpx.Response:
        channel = req.url.params.get("channel", "")
        self.log_channels.append(channel)
        if channel == "runtime":
            return refusal(501, "unavailable", "Runtime logs are not available here yet.")
        key = (attempt_id, channel)
        lines = self.logs.get(key, [])
        retained = self.retained_from.get(key, 0)
        cursor = req.url.params.get("cursor")
        if cursor is None:
            start = max(retained, len(lines) - 1000)
            omitted = start
        else:
            position, omitted = (int(part) for part in cursor.split(":"))
            if position + 1 < retained:
                return refusal(410, "cursor-expired", "These logs moved on.")
            start = position + 1
        window = list(enumerate(lines))[start:]
        last = window[-1][0] if window else start - 1
        done = key in self.producer_done
        return httpx.Response(
            200,
            json={
                "lines": [{"id": f"{attempt_id}:{channel}:{i}", "text": t} for i, t in window],
                "cursor": f"{last}:{omitted}",
                "state": "complete" if done else "streaming",
                "omittedLines": omitted,
            },
        )


def refusal(status: int, code: str, message: str) -> httpx.Response:
    return httpx.Response(status, json={"code": code, "message": message})


def invoke(api: FakeDeployments, *args: str, input: str | None = None) -> Any:
    runtime = DashboardRuntime(
        host="https://dashboard.test",
        http=httpx.Client(base_url="https://dashboard.test", transport=httpx.MockTransport(api)),
        credentials=cast(Any, None),
        config=cast(Any, None),
        open_browser=lambda url: False,
        sleep=lambda seconds: None,
        project="p1",
    )
    return CliRunner().invoke(cli, ["dashboard", *args], obj=runtime, input=input)


@pytest.fixture(autouse=True)
def signed_in(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in ("deploy", "deployments", "secrets"):
        mod = __import__(f"parlant.bin.dashboard.commands.{module}", fromlist=["authenticate"])
        monkeypatch.setattr(mod, "authenticate", lambda r: "token")


def assert_no_legacy_calls(api: FakeDeployments) -> None:
    assert not [p for p in api.paths() if LEGACY.match(p)], api.requests


# -- deploy -------------------------------------------------------------------


def test_that_deploy_admits_an_attempt_at_the_current_revision_and_waits_until_it_is_live() -> None:
    api = FakeDeployments()
    api.progress["att1"] = [
        {"phase": "starting"},
        {"phase": "promoting"},
        {"status": "completed", "phase": None},
    ]

    result = invoke(api, "deploy")

    assert result.exit_code == 0, result.output
    posted = [r for r in api.requests if r == f"POST {BASE}/attempts"]
    assert len(posted) == 1
    assert "Acme (p1)" in result.output
    assert "att1" in result.output
    assert "Starting" in result.output and "Promoting" in result.output
    assert "live" in result.output.lower()
    assert api.serving == "att1"
    assert_no_legacy_calls(api)


def test_that_deploy_with_a_commit_deploys_exactly_that_commit() -> None:
    api = FakeDeployments()
    api.progress["att1"] = [{"status": "completed", "phase": None}]

    result = invoke(api, "deploy", "--commit", SHA.upper())

    assert result.exit_code == 0, result.output
    assert api.attempts["att1"]["commit"]["sha"] == SHA
    assert api.attempts["att1"]["trigger"] == "commit"


def test_that_deploy_refuses_a_commit_that_is_not_a_full_sha_without_calling_the_platform() -> None:
    api = FakeDeployments()

    result = invoke(api, "deploy", "--commit", "abc123")

    assert result.exit_code != 0
    assert "40-character" in result.output
    assert f"POST {BASE}/attempts" not in api.requests


def test_that_deploy_while_another_attempt_runs_names_it_and_how_to_watch_or_stop_it() -> None:
    api = FakeDeployments()
    api.attempts["att9"] = {"id": "att9", "status": "in-progress", "startedAt": "x"}
    api.active = "att9"

    result = invoke(api, "deploy")

    assert result.exit_code != 0
    assert "att9" in result.output
    assert "parlant dashboard deployment view att9" in result.output
    assert "parlant dashboard deployment stop att9" in result.output
    assert len(api.attempts) == 1


def test_that_deploy_rereads_the_revision_when_settings_changed_meanwhile() -> None:
    api = FakeDeployments(revision_moves_before_next_mutation=True)
    api.progress["att1"] = [{"status": "completed", "phase": None}]

    result = invoke(api, "deploy")

    assert result.exit_code == 0, result.output
    assert len(api.attempts) == 1
    assert api.requests.count(f"GET {BASE}") >= 2


def test_that_a_lost_deploy_response_is_retried_with_the_same_operation_and_deploys_once() -> None:
    api = FakeDeployments(lose_next_mutation_response=True)
    api.progress["att1"] = [{"status": "completed", "phase": None}]

    result = invoke(api, "deploy")

    assert result.exit_code == 0, result.output
    assert len(api.attempts) == 1
    assert len(api.receipts) == 1
    assert api.requests.count(f"POST {BASE}/attempts") == 2


def test_that_an_unknown_deploy_outcome_is_resolved_by_looking_the_operation_up() -> None:
    api = FakeDeployments(next_mutation_outcome_unknown=True)
    api.progress["att1"] = [{"status": "completed", "phase": None}]

    result = invoke(api, "deploy")

    assert result.exit_code == 0, result.output
    assert len(api.attempts) == 1
    assert [p for p in api.paths() if p.startswith(f"{BASE}/operations/")]
    assert "att1" in result.output


def test_that_a_failed_attempt_reports_why_and_where_its_logs_are() -> None:
    api = FakeDeployments()
    api.progress["att1"] = [
        {
            "status": "failed",
            "phase": None,
            "failure": {"kind": "build", "message": "pip install failed"},
        }
    ]

    result = invoke(api, "deploy")

    assert result.exit_code != 0
    assert "pip install failed" in result.output
    assert "parlant dashboard deployment logs att1" in result.output


def test_that_deploy_refuses_a_project_that_is_not_managed() -> None:
    api = FakeDeployments(kind="local")

    result = invoke(api, "deploy")

    assert result.exit_code != 0
    assert "not managed-hosted" in result.output
    assert not [r for r in api.requests if r.startswith("POST")]


def test_that_deploy_refuses_a_managed_project_without_a_repository() -> None:
    api = FakeDeployments(repository="")

    result = invoke(api, "deploy")

    assert result.exit_code != 0
    assert "no connected GitHub repository" in result.output
    assert not [r for r in api.requests if r.startswith("POST")]


# -- deployments ----------------------------------------------------------------


def test_that_deployments_list_shows_attempts_and_marks_the_live_one() -> None:
    api = FakeDeployments()
    api.attempts["att1"] = {
        "id": "att1",
        "status": "completed",
        "trigger": "manual",
        "commit": {"sha": SHA, "message": "Fix"},
        "startedAt": "2026-09-28T10:00:00.000Z",
    }
    api.serving = "att1"

    result = invoke(api, "deployment", "list")

    assert result.exit_code == 0, result.output
    assert "att1" in result.output
    assert "live" in result.output
    assert "aaaaaaaaaaaa" in result.output


def test_that_deployments_view_shows_the_attempt_and_its_failure() -> None:
    api = FakeDeployments()
    api.attempts["att2"] = {
        "id": "att2",
        "status": "failed",
        "phase": None,
        "trigger": "push",
        "source": {"installationId": "i1", "repository": "acme/agent", "branch": "main"},
        "commit": {"sha": SHA, "message": "Fix"},
        "failure": {"kind": "startup", "message": "Port never opened"},
        "startedAt": "2026-09-28T10:00:00.000Z",
        "finishedAt": "2026-09-28T10:04:00.000Z",
    }

    result = invoke(api, "deployment", "view", "att2")

    assert result.exit_code == 0, result.output
    assert "failed" in result.output
    assert "startup: Port never opened" in result.output
    assert "acme/agent@main" in result.output


def test_that_deployments_stop_confirms_then_stops_exactly_that_attempt() -> None:
    api = FakeDeployments()
    api.attempts["att3"] = {"id": "att3", "status": "in-progress", "startedAt": "x"}
    api.active = "att3"

    aborted = invoke(api, "deployment", "stop", "att3", input="n\n")
    assert aborted.exit_code != 0
    assert api.stopped == []

    result = invoke(api, "deployment", "stop", "att3", "-y")
    assert result.exit_code == 0, result.output
    assert api.stopped == ["att3"]
    assert "stopped" in result.output.lower()


def test_that_deployments_logs_prints_the_retained_tail_of_the_build_channel() -> None:
    api = FakeDeployments()
    api.attempts["att1"] = {"id": "att1", "status": "completed", "startedAt": "x"}
    api.logs[("att1", "build")] = ["one", "two", "three"]
    api.producer_done.add(("att1", "build"))

    result = invoke(api, "deployment", "logs", "att1")

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-3:] == ["one", "two", "three"]
    assert api.log_channels == ["build"]


def test_that_following_logs_polls_the_cursor_until_the_producer_is_done() -> None:
    api = FakeDeployments()
    api.attempts["att1"] = {"id": "att1", "status": "in-progress", "startedAt": "x"}
    key = ("att1", "deployment")
    api.logs[key] = ["starting"]
    reads = {"count": 0}
    original = api.read_logs

    def growing(attempt_id: str, req: httpx.Request) -> httpx.Response:
        reads["count"] += 1
        if reads["count"] == 2:
            api.logs[key].append("healthy")
        if reads["count"] == 3:
            api.logs[key].append("live")
            api.producer_done.add(key)
        return original(attempt_id, req)

    api.read_logs = growing  # type: ignore[method-assign]

    result = invoke(api, "deployment", "logs", "att1", "--channel", "deployment", "--follow")

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-3:] == ["starting", "healthy", "live"]
    assert reads["count"] == 3


def test_that_an_expired_log_cursor_resumes_from_the_retained_tail_and_says_output_was_lost() -> (
    None
):
    api = FakeDeployments()
    api.attempts["att1"] = {"id": "att1", "status": "in-progress", "startedAt": "x"}
    key = ("att1", "build")
    api.logs[key] = ["l0", "l1"]
    reads = {"count": 0}
    original = api.read_logs

    def pruning(attempt_id: str, req: httpx.Request) -> httpx.Response:
        reads["count"] += 1
        if reads["count"] == 2:
            api.logs[key] += ["l2", "l3", "l4"]
            api.retained_from[key] = 4
        if reads["count"] == 3:
            api.producer_done.add(key)
        return original(attempt_id, req)

    api.read_logs = pruning  # type: ignore[method-assign]

    result = invoke(api, "deployment", "logs", "att1", "--follow")

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[:2] == ["l0", "l1"]
    assert lines[-1] == "l4"
    assert any("no longer retained" in line for line in lines)
    assert "l0" not in lines[2:]


# -- secrets ------------------------------------------------------------------


def test_that_secrets_set_saves_a_secret_change_with_the_settings_unchanged() -> None:
    api = FakeDeployments()

    result = invoke(api, "secret", "set", "KEY=s3cr3t")

    assert result.exit_code == 0, result.output
    assert api.secret_values == {"KEY": "s3cr3t"}
    assert "s3cr3t" not in result.output
    assert "`parlant dashboard deploy`" in result.output
    assert api.requests.count(f"PUT {BASE}/settings") == 1
    assert_no_legacy_calls(api)


def test_that_secrets_set_reads_a_key_only_value_from_stdin() -> None:
    api = FakeDeployments()

    result = invoke(api, "secret", "set", "TOKEN", input="piped-value\n")

    assert result.exit_code == 0, result.output
    assert api.secret_values == {"TOKEN": "piped-value"}


def test_that_secrets_unset_confirms_before_removing() -> None:
    api = FakeDeployments(secret_names=["KEY"])

    aborted = invoke(api, "secret", "unset", "KEY", input="n\n")
    assert aborted.exit_code != 0
    assert api.secret_names == ["KEY"]

    confirmed = invoke(api, "secret", "unset", "KEY", "-y")
    assert confirmed.exit_code == 0, confirmed.output
    assert api.secret_names == []
    assert "`parlant dashboard deploy`" in confirmed.output
    assert_no_legacy_calls(api)


def test_that_secrets_list_shows_names_from_the_deployment_settings() -> None:
    api = FakeDeployments(secret_names=["B_KEY", "A_KEY"])

    result = invoke(api, "secret", "list")

    assert result.exit_code == 0, result.output
    assert "A_KEY" in result.output and "B_KEY" in result.output
    assert_no_legacy_calls(api)


def test_that_secrets_set_refuses_a_project_that_is_not_managed() -> None:
    api = FakeDeployments(kind="local")

    result = invoke(api, "secret", "set", "KEY=value")

    assert result.exit_code != 0
    assert "not managed-hosted" in result.output
    assert not [r for r in api.requests if r.startswith("PUT")]


def test_that_secrets_set_rejects_an_invalid_key_name() -> None:
    api = FakeDeployments()

    result = invoke(api, "secret", "set", "1BAD=x")

    assert result.exit_code != 0
    assert "valid environment variable name" in result.output


class PricingDeployments(FakeDeployments):
    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.url.path == "/projects/p1/project-connection":
            self.requests.append(f"{req.method} {req.url.path}")
            return httpx.Response(
                200,
                json={
                    "connected": True,
                    "pricing": {
                        "price_table": False,
                        "currency": None,
                        "models": [
                            {
                                "key": "gemini:needs-price",
                                "price": "missing",
                                "source": "configured",
                            },
                        ],
                    },
                },
            )
        return super().__call__(req)


def test_that_a_live_managed_deploy_checks_prices_once_and_explains_the_gap() -> None:
    api = PricingDeployments()
    api.progress["att1"] = [{"status": "completed", "phase": None}]
    result = invoke(api, "deploy")
    assert result.exit_code == 0, result.output
    assert "gemini:needs-price" in result.output
    assert "set_pricing" in result.output
    assert api.paths().count("/projects/p1/project-connection") == 1


def test_that_a_failed_deploy_does_not_ask_an_old_runtime_for_prices() -> None:
    api = PricingDeployments()
    api.progress["att1"] = [{"status": "failed", "phase": None, "error": {"message": "Failed"}}]
    result = invoke(api, "deploy")
    assert result.exit_code != 0
    assert "/projects/p1/project-connection" not in api.paths()
