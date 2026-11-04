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

"""The commands scripts and coding agents drive: `--json` output, and connecting a local project.

The flow they serve: create the project on the dashboard, write the repository,
put the project token in its `.env`, check that the server reached the dashboard,
then confirm creation. Every command here signs in for real against the fake
dashboard, so the refresh-token rotation runs as it does in production.

Each sign-in is approved for one workspace. A machine can hold one per
workspace; commands act in the active workspace unless given `--workspace`."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner, Result

from parlant.bin.cli import cli
from parlant.bin.dashboard.config import FileConfigStore
from parlant.bin.dashboard.credentials import Credential, CredentialStore
from parlant.bin.dashboard.runtime import DashboardRuntime, resolve_host

HOST = "https://dashboard.test"
PROJECT_TOKEN = "ppt_do-not-print-this"


class MemoryCredentialStore(CredentialStore):
    def __init__(self, credential: Credential | None = None) -> None:
        self.credentials: dict[str, Credential] = {}
        if credential is not None:
            self.credentials[HOST] = credential

    def get(self, host: str) -> Credential | None:
        return self.credentials.get(host)

    def save(self, host: str, credential: Credential) -> None:
        self.credentials[host] = credential

    def delete(self, host: str) -> None:
        self.credentials.pop(host, None)


class FakeDashboard:
    """Answers like Backend's CLI sign-in, workspace, project and local-connection routes.

    Each device sign-in is approved for the next entry in `approvals`, and its
    tokens reach only that workspace. An entry that lists several workspaces is
    one approval for all of them, answered in the multi-workspace format:
    `{"grants": [...]}`, one sign-in per workspace in the order they were
    ticked. A single workspace is answered the way Backend did before. A local project starts with setup
    pending. Its server counts as connected once `connect_server()` is called,
    and setup completes only while it is.

    A server that already holds sessions (`server_sessions`) must be imported
    first, as Backend requires: setup completion is refused with a 409 until a
    sync started through `POST sessions/sync` has finished. The import finishes
    after `import_polls` reads of its status, or fails with `import_failure`."""

    WORKSPACES = {"Emcie": "owner", "Other": "builder"}

    def __init__(
        self,
        *,
        approvals: list[str | list[str]] | None = None,
        pending_polls: int = 1,
        create_refusal: httpx.Response | None = None,
        watch: Path | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.approvals: list[str | list[str]] = list(approvals or ["Emcie"])
        self.pending_polls = pending_polls
        self.create_refusal = create_refusal
        self.watch = watch
        self.files_when_created: list[str] | None = None
        self.projects: dict[str, dict[str, Any]] = {}
        self.connected: set[str] = set()
        self.logins = 0
        self.rotations = 0
        # refresh token -> (workspace, session id); access token -> workspace.
        self.refresh_tokens: dict[str, tuple[str, str]] = {}
        self.access_tokens: dict[str, tuple[str, str]] = {}
        self.revoked: list[str] = []
        self.server_sessions: dict[str, int] = {}
        self.import_polls = 0
        self.import_failure: str | None = None
        # project id -> status reads left before the import finishes.
        self.importing: dict[str, int] = {}
        self.imported: set[str] = set()

    def issue(self, workspace: str) -> str:
        """A sign-in made before this test, as a refresh token for `workspace`."""
        self.logins += 1
        refresh_token = f"refresh-{workspace}-{self.logins}-0"
        self.refresh_tokens[refresh_token] = (workspace, f"session-{self.logins}")
        return refresh_token

    def connect_server(self, project_id: str) -> None:
        self.connected.add(project_id)

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        path, method = req.url.path, req.method
        body = json.loads(req.content) if req.content else {}

        if method == "POST" and path == "/v1/cli/auth/device":
            return httpx.Response(
                200,
                json={
                    "device_code": f"device-{self.logins + 1}",
                    "user_code": "WXYZ-1234",
                    "verification_uri": f"{HOST}/device",
                    "verification_uri_complete": f"{HOST}/device?code=WXYZ-1234",
                    "expires_in": 600,
                    "interval": 5,
                },
            )
        if method == "POST" and path == "/v1/cli/auth/token":
            if body["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code":
                if self.pending_polls > 0:
                    self.pending_polls -= 1
                    return httpx.Response(400, json={"error": "authorization_pending"})
                approved = self.approvals.pop(0)
                if isinstance(approved, str):
                    return self._tokens(self.issue(approved))
                return httpx.Response(
                    200,
                    json={
                        "grants": [
                            {**self._tokens(self.issue(workspace)).json(), "tenant_id": workspace}
                            for workspace in approved
                        ]
                    },
                )
            if body["refresh_token"] not in self.refresh_tokens:
                return httpx.Response(400, json={"error": "invalid_grant"})
            return self._tokens(body["refresh_token"])
        if method == "POST" and path == "/v1/cli/auth/revoke":
            if self.refresh_tokens.pop(body["refresh_token"], None) is not None:
                self.revoked.append(body["refresh_token"])
            return httpx.Response(200, json={})

        signed_in = self.access_tokens.get(req.headers.get("authorization", "")[len("Bearer ") :])
        if signed_in is None:
            return httpx.Response(401, json={"message": "Unauthorized"})
        workspace, session_id = signed_in

        if method == "GET" and path == "/v1/cli/auth/session":
            return httpx.Response(
                200,
                json={
                    "user_id": "user-1",
                    "tenant_id": workspace,
                    "session_id": session_id,
                    "scope": "workspace.read projects.read projects.create projects.update",
                },
            )
        if method == "GET" and path == "/users/self/workspaces":
            # Membership, not sign-in: every workspace the user belongs to.
            return httpx.Response(
                200,
                json=[
                    {"tenantId": name, "name": name, "role": role, "accessSource": "member"}
                    for name, role in self.WORKSPACES.items()
                ],
            )
        if method == "GET" and path == "/projects/creation-readiness":
            return httpx.Response(200, json={"ready": True, "blocker": None})
        if method == "GET" and path == "/projects":
            return httpx.Response(
                200, json=[p for p in self.projects.values() if p["tenantId"] == workspace]
            )
        if method == "POST" and path == "/projects":
            if self.watch is not None:
                self.files_when_created = sorted(
                    str(p.relative_to(self.watch)) for p in self.watch.rglob("*") if p.is_file()
                )
            if self.create_refusal is not None:
                return self.create_refusal
            self.projects[body["id"]] = {
                "id": body["id"],
                "name": body["name"],
                "tenantId": workspace,
                "projectKind": body["projectKind"],
                "lifecycleState": "active",
                "setup": {"status": "pending", "completedAt": None},
            }
            return httpx.Response(201, json=self.projects[body["id"]])

        parts = path.strip("/").split("/")
        if parts[0] != "projects" or len(parts) < 2:
            raise AssertionError(f"Unexpected {method} {path}")
        project = self.projects.get(parts[1])
        if project is None or project["tenantId"] != workspace:
            return httpx.Response(404, json={"message": "Project not found"})
        rest = parts[2:]
        if method == "GET" and not rest:
            return httpx.Response(200, json=project)
        if method == "POST" and rest == ["self-hosted", "auth-setup"]:
            return httpx.Response(
                200,
                json={
                    "hostingMode": "remote",
                    "envVarName": "PARLANT_DASHBOARD_PROJECT_TOKEN",
                    "headerName": "X-Parlant-Project-Token",
                    "projectToken": PROJECT_TOKEN,
                    "instructions": [],
                },
            )
        if method == "GET" and rest == ["project-connection"]:
            return httpx.Response(200, json=self._connection(project))
        if method == "POST" and rest == ["setup", "complete"]:
            # Backend answers 200 either way, and completes only a connected project.
            if project["id"] in self.connected:
                if self.server_sessions.get(project["id"]) and project["id"] not in self.imported:
                    return httpx.Response(
                        409,
                        json={
                            "statusCode": 409,
                            "error": "Conflict",
                            "message": "Sync your project before opening it.",
                        },
                    )
                project["setup"] = {"status": "completed", "completedAt": "2026-10-17T10:00:00Z"}
            return httpx.Response(200, json=self._connection(project))
        if method == "POST" and rest == ["sessions", "sync"]:
            self.importing[project["id"]] = self.import_polls
            return httpx.Response(
                200, json={"triggered": True, "status": self._sync_status(project["id"])}
            )
        if method == "GET" and rest == ["sessions", "sync-status"]:
            if project["id"] in self.importing:
                if self.importing[project["id"]] == 0 and self.import_failure is None:
                    del self.importing[project["id"]]
                    self.imported.add(project["id"])
                else:
                    self.importing[project["id"]] = max(self.importing[project["id"]] - 1, 0)
            return httpx.Response(200, json=self._sync_status(project["id"]))
        raise AssertionError(f"Unexpected {method} {path}")

    def _sync_status(self, project_id: str) -> dict[str, Any]:
        """The Ingestor's view of the import, as `GET sessions/sync-status` answers it."""
        total = self.server_sessions.get(project_id, 0)
        status, phase, imported = "pending", None, 0
        if project_id in self.imported:
            status, phase, imported = "synced", "live", total
        elif project_id in self.importing:
            failed = self.import_failure is not None and self.importing[project_id] == 0
            status, phase = ("failed", "historical") if failed else ("importing", "historical")
            imported = total // 2
        return {
            "status": status,
            "phase": phase,
            "progress": {
                "totalSessions": total,
                "processedSessions": imported,
                "importedSessions": imported,
                "failedSessions": 0,
                "batchSize": 50,
            },
            "availableSessions": total,
            "lastError": self.import_failure if status == "failed" else None,
            "pollIntervalSeconds": 5,
        }

    def _tokens(self, refresh_token: str) -> httpx.Response:
        """Rotate: the presented refresh token is spent and a new pair issued."""
        workspace, session_id = self.refresh_tokens.pop(refresh_token)
        self.rotations += 1
        rotated = f"{refresh_token.rsplit('-', 1)[0]}-{self.rotations}"
        access = f"access-{workspace}-{self.rotations}"
        self.refresh_tokens[rotated] = (workspace, session_id)
        self.access_tokens[access] = (workspace, session_id)
        return httpx.Response(
            200,
            json={"access_token": access, "refresh_token": rotated, "session_id": session_id},
        )

    def _connection(self, project: dict[str, Any]) -> dict[str, Any]:
        connected = project["id"] in self.connected
        return {
            "projectId": project["id"],
            "setup": project["setup"],
            "connected": connected,
            "connection": {
                "status": "connected" if connected else "disconnected",
                "observedAt": "2026-10-17T10:00:00Z",
            },
            "parlantVersion": "4.0.0a1" if connected else None,
            "compatibleVersion": True if connected else None,
        }

    def signed_in_workspaces(self) -> set[str]:
        return {workspace for workspace, _ in self.refresh_tokens.values()}

    def sent(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path == path]


def runtime(
    dashboard: FakeDashboard,
    tmp_path: Path,
    *,
    credentials: MemoryCredentialStore | None = None,
    opened: list[str] | None = None,
) -> DashboardRuntime:
    def open_browser(url: str) -> bool:
        if opened is not None:
            opened.append(url)
        return True

    return DashboardRuntime(
        host=HOST,
        http=httpx.Client(base_url=HOST, transport=httpx.MockTransport(dashboard)),
        credentials=credentials or MemoryCredentialStore(),
        config=FileConfigStore(tmp_path / "cli-config.json"),
        open_browser=open_browser,
        sleep=lambda seconds: None,
    )


def invoke(rt: DashboardRuntime, args: list[str]) -> Result:
    # Each invocation is a fresh process: only the keychain and config file carry over.
    rt.project = None
    rt.workspace = None
    return CliRunner().invoke(cli, args, obj=rt)


def signed_in(dashboard: FakeDashboard, tmp_path: Path, *workspaces: str) -> DashboardRuntime:
    """A machine that signed in once per workspace, in order, through `auth login`."""
    rt = runtime(dashboard, tmp_path)
    dashboard.approvals = list(workspaces or ["Emcie"])
    for _ in dashboard.approvals[:]:
        dashboard.pending_polls = 0
        result = invoke(rt, ["dashboard", "auth", "login"])
        assert result.exit_code == 0, result.output
    return rt


def json_lines(result: Result) -> list[Any]:
    """Every stdout line, each of which must be a JSON document."""
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def only_json(result: Result) -> Any:
    [document] = json_lines(result)
    return document


def local_project(
    dashboard: FakeDashboard, project_id: str = "a" * 24, workspace: str = "Emcie"
) -> str:
    dashboard.projects[project_id] = {
        "id": project_id,
        "name": "Support Bot",
        "tenantId": workspace,
        "projectKind": "local",
        "lifecycleState": "active",
        "setup": {"status": "pending", "completedAt": None},
    }
    return project_id


CREATE_LOCAL = ["project", "create", "--kind", "local", "--name", "Support Bot", "--yes"]
INIT_WITH_DASHBOARD = [
    "--package-manager",
    "uv",
    "--no-git",
    "--dashboard",
    "--name",
    "Support Bot",
    "--yes",
]


def test_that_auth_status_with_json_prints_the_signed_in_user_and_host(tmp_path: Path) -> None:
    rt = signed_in(FakeDashboard(), tmp_path)

    result = invoke(rt, ["dashboard", "auth", "status", "--json"])

    assert result.exit_code == 0, result.output
    assert only_json(result) == {
        "signedIn": True,
        "host": HOST,
        "workspace": "Emcie",
        "user": {"id": "user-1"},
    }


def test_that_auth_status_with_json_when_signed_out_prints_an_error_envelope_and_fails(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()

    result = invoke(runtime(dashboard, tmp_path), ["dashboard", "auth", "status", "--json"])

    assert result.exit_code != 0
    envelope = only_json(result)
    assert envelope["error"]["code"] == "NOT_SIGNED_IN"
    assert "parlant dashboard auth login" in envelope["error"]["message"]
    assert dashboard.requests == []


@pytest.mark.parametrize(
    ("no_browser", "browser_available"),
    [(False, True), (False, False), (True, True)],
    ids=["browser-opens", "browser-unavailable", "no-browser"],
)
def test_that_auth_login_prints_the_complete_approval_link(
    tmp_path: Path, no_browser: bool, browser_available: bool
) -> None:
    dashboard = FakeDashboard()
    opened: list[str] = []
    rt = runtime(dashboard, tmp_path)

    def open_browser(url: str) -> bool:
        opened.append(url)
        return browser_available

    rt.open_browser = open_browser
    args = ["dashboard", "auth", "login"]
    if no_browser:
        args.append("--no-browser")

    result = invoke(rt, args)

    assert result.exit_code == 0, result.output
    assert f"    Approve at: {HOST}/device?code=WXYZ-1234" in result.output.splitlines()
    assert "Your code:  WXYZ-1234" in result.output
    assert opened == ([] if no_browser else [f"{HOST}/device?code=WXYZ-1234"])


def test_that_auth_login_with_json_streams_the_verification_event_then_the_signed_in_event_without_opening_a_browser(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard(pending_polls=2)
    opened: list[str] = []
    rt = runtime(dashboard, tmp_path, opened=opened)

    result = invoke(rt, ["dashboard", "auth", "login", "--json", "--no-browser"])

    assert result.exit_code == 0, result.output
    assert json_lines(result) == [
        {
            "event": "verification",
            "userCode": "WXYZ-1234",
            "verificationUri": f"{HOST}/device",
            "verificationUriComplete": f"{HOST}/device?code=WXYZ-1234",
            "expiresIn": 600,
        },
        {
            "event": "signed-in",
            "signedIn": True,
            "host": HOST,
            "workspace": "Emcie",
            "workspaces": ["Emcie"],
            "user": {"id": "user-1"},
        },
    ]
    assert opened == []
    assert dashboard.signed_in_workspaces() == {"Emcie"}
    for refresh_token in dashboard.refresh_tokens:
        assert refresh_token not in result.output


def test_that_signing_in_to_a_second_workspace_keeps_the_first_and_makes_the_new_one_active(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path, "Emcie", "Other")

    status = invoke(rt, ["dashboard", "auth", "status", "--json"])
    workspaces = invoke(rt, ["dashboard", "workspace", "list", "--json"])

    assert status.exit_code == 0, status.output
    assert only_json(status)["workspace"] == "Other"
    assert workspaces.exit_code == 0, workspaces.output
    assert only_json(workspaces) == [
        {"id": "Emcie", "name": "Emcie", "role": "owner", "signedIn": True, "active": False},
        {"id": "Other", "name": "Other", "role": "builder", "signedIn": True, "active": True},
    ]
    assert dashboard.signed_in_workspaces() == {"Emcie", "Other"}
    assert dashboard.revoked == []


def test_that_workspace_use_switches_which_workspace_later_commands_act_in(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path, "Emcie", "Other")
    ours = local_project(dashboard, "a" * 24, workspace="Emcie")
    local_project(dashboard, "c" * 24, workspace="Other")

    use = invoke(rt, ["dashboard", "workspace", "use", "Emcie"])
    projects = invoke(rt, ["dashboard", "project", "list", "--json"])

    assert use.exit_code == 0, use.output
    assert "Emcie" in use.output
    assert projects.exit_code == 0, projects.output
    assert [p["id"] for p in only_json(projects)] == [ours]


def test_that_workspace_use_refuses_a_workspace_this_machine_is_not_signed_in_to(
    tmp_path: Path,
) -> None:
    rt = signed_in(FakeDashboard(), tmp_path, "Emcie")

    result = invoke(rt, ["dashboard", "workspace", "use", "Other", "--json"])

    assert result.exit_code != 0
    envelope = only_json(result)
    assert envelope["error"]["code"] == "NOT_SIGNED_IN"
    assert "Other" in envelope["error"]["message"]
    assert "parlant dashboard auth login" in envelope["error"]["message"]
    assert only_json(invoke(rt, ["dashboard", "auth", "status", "--json"]))["workspace"] == "Emcie"


def test_that_the_workspace_flag_acts_in_that_workspace_without_changing_the_active_one(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path, "Other", "Emcie")

    result = invoke(rt, ["dashboard", "--workspace", "Other", *CREATE_LOCAL, "--json"])

    assert result.exit_code == 0, result.output
    [project] = dashboard.projects.values()
    assert project["tenantId"] == "Other"
    assert only_json(invoke(rt, ["dashboard", "auth", "status", "--json"]))["workspace"] == "Emcie"


def test_that_a_command_for_a_workspace_without_a_sign_in_says_how_to_sign_in_to_it(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path, "Emcie")
    before = len(dashboard.requests)

    result = invoke(rt, ["dashboard", "--workspace", "Other", *CREATE_LOCAL, "--json"])

    assert result.exit_code != 0
    envelope = only_json(result)
    assert envelope["error"]["code"] == "NOT_SIGNED_IN"
    assert "Other" in envelope["error"]["message"]
    assert "parlant dashboard auth login" in envelope["error"]["message"]
    # Emcie's sign-in is never used for Other.
    assert dashboard.requests[before:] == []
    assert dashboard.projects == {}


def test_that_signing_in_again_to_the_same_workspace_replaces_its_sign_in_and_revokes_the_old_one(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path, "Emcie", "Other")
    [old_emcie] = [t for t, (w, _) in dashboard.refresh_tokens.items() if w == "Emcie"]

    dashboard.approvals = ["Emcie"]
    result = invoke(rt, ["dashboard", "auth", "login"])

    assert result.exit_code == 0, result.output
    assert dashboard.revoked == [old_emcie]
    assert dashboard.signed_in_workspaces() == {"Emcie", "Other"}
    assert invoke(rt, ["dashboard", "--workspace", "Other", "project", "list"]).exit_code == 0
    assert invoke(rt, ["dashboard", "--workspace", "Emcie", "project", "list"]).exit_code == 0


def test_that_auth_logout_signs_out_of_the_active_workspace_only(tmp_path: Path) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path, "Emcie", "Other")

    result = invoke(rt, ["dashboard", "auth", "logout"])

    assert result.exit_code == 0, result.output
    assert "Other" in result.output
    assert dashboard.signed_in_workspaces() == {"Emcie"}
    # With one sign-in left, it becomes the active workspace.
    assert only_json(invoke(rt, ["dashboard", "auth", "status", "--json"]))["workspace"] == "Emcie"


def test_that_auth_logout_all_signs_out_of_every_workspace(tmp_path: Path) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path, "Emcie", "Other")

    result = invoke(rt, ["dashboard", "auth", "logout", "--all"])

    assert result.exit_code == 0, result.output
    assert dashboard.signed_in_workspaces() == set()
    status = invoke(rt, ["dashboard", "auth", "status", "--json"])
    assert only_json(status)["error"]["code"] == "NOT_SIGNED_IN"


def test_that_a_sign_in_saved_before_workspaces_were_tracked_keeps_working_as_its_workspace(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()
    # The keychain entry today's CLI writes: one credential under the host.
    legacy = dashboard.issue("Emcie")
    rt = runtime(
        dashboard,
        tmp_path,
        credentials=MemoryCredentialStore(Credential(refresh_token=legacy, session_id="session-1")),
    )

    status = invoke(rt, ["dashboard", "auth", "status", "--json"])
    dashboard.approvals = ["Other"]
    login = invoke(rt, ["dashboard", "auth", "login"])

    assert status.exit_code == 0, status.output
    assert only_json(status)["workspace"] == "Emcie"
    assert login.exit_code == 0, login.output
    assert dashboard.signed_in_workspaces() == {"Emcie", "Other"}
    assert dashboard.revoked == []


def test_that_workspace_list_and_project_list_with_json_print_one_json_array(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path)
    project_id = local_project(dashboard)

    workspaces = invoke(rt, ["dashboard", "workspace", "list", "--json"])
    projects = invoke(rt, ["dashboard", "project", "list", "--json"])

    assert workspaces.exit_code == 0, workspaces.output
    assert only_json(workspaces) == [
        {"id": "Emcie", "name": "Emcie", "role": "owner", "signedIn": True, "active": True},
        {"id": "Other", "name": "Other", "role": "builder", "signedIn": False, "active": False},
    ]
    assert projects.exit_code == 0, projects.output
    assert only_json(projects) == [
        {
            "id": project_id,
            "name": "Support Bot",
            "projectKind": "local",
            "state": "active",
            "setupStatus": "pending",
            "repository": None,
            "branch": None,
        }
    ]


def test_that_project_list_with_json_prints_an_empty_array_for_an_empty_workspace(
    tmp_path: Path,
) -> None:
    rt = signed_in(FakeDashboard(), tmp_path)

    result = invoke(rt, ["dashboard", "project", "list", "--json"])

    assert result.exit_code == 0, result.output
    assert only_json(result) == []


def test_that_project_create_with_json_prints_only_the_created_project(tmp_path: Path) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path)
    request_id = "b" * 24

    result = invoke(rt, ["dashboard", *CREATE_LOCAL, "--request-id", request_id, "--json"])

    assert result.exit_code == 0, result.output
    assert only_json(result) == {
        "id": request_id,
        "name": "Support Bot",
        "projectKind": "local",
        "setupStatus": "pending",
        "nextSteps": [
            f"parlant dashboard --project {request_id} project connect PROJECT_FOLDER",
            "Start the project's server (./run.sh in a scaffolded project, run.cmd on Windows).",
            f"parlant dashboard --project {request_id} project check",
            f"parlant dashboard --project {request_id} project confirm",
        ],
    }
    assert len(dashboard.sent("POST", "/projects")) == 1


def test_that_init_with_json_reports_the_written_files_without_the_next_steps_panel(
    tmp_path: Path,
) -> None:
    target = tmp_path / "support-bot"

    result = CliRunner().invoke(
        cli, ["init", str(target), "--package-manager", "uv", "--no-git", "--json"]
    )

    assert result.exit_code == 0, result.output
    report = only_json(result)
    assert report["dir"] == str(target)
    assert "main.py" in report["files"] and ".env.example" in report["files"]
    assert all((target / name).is_file() for name in report["files"])
    assert "Next" not in result.stdout


def test_that_project_connect_writes_the_token_into_env_created_from_env_example_and_never_prints_it(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path)
    project_id = local_project(dashboard)
    folder = tmp_path / "support-bot"
    folder.mkdir()
    (folder / ".env.example").write_text(
        "# Copy to .env and fill in.\nOPENROUTER_API_KEY=\nPARLANT_ENV=development\n"
    )

    result = invoke(
        rt, ["dashboard", "--project", project_id, "project", "connect", str(folder), "--json"]
    )

    assert result.exit_code == 0, result.output
    assert only_json(result) == {
        "projectId": project_id,
        "envFile": str(folder / ".env"),
        "envVarName": "PARLANT_DASHBOARD_PROJECT_TOKEN",
    }
    assert (folder / ".env").read_text() == (
        "# Copy to .env and fill in.\n"
        "OPENROUTER_API_KEY=\n"
        "PARLANT_ENV=development\n"
        f"PARLANT_DASHBOARD_PROJECT_TOKEN={PROJECT_TOKEN}\n"
    )
    assert stat.S_IMODE((folder / ".env").stat().st_mode) == 0o600
    assert PROJECT_TOKEN not in result.output
    assert len(dashboard.sent("POST", f"/projects/{project_id}/self-hosted/auth-setup")) == 1


def test_that_project_connect_replaces_an_existing_token_line_and_keeps_every_other_line(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path)
    project_id = local_project(dashboard)
    folder = tmp_path / "support-bot"
    folder.mkdir()
    (folder / ".env.example").write_text("OPENROUTER_API_KEY=\n")
    (folder / ".env").write_text(
        "# mine\nOPENROUTER_API_KEY=sk-kept\nPARLANT_DASHBOARD_PROJECT_TOKEN=ppt_old\n\nEXTRA=1\n"
    )

    result = invoke(rt, ["dashboard", "--project", project_id, "project", "connect", str(folder)])

    assert result.exit_code == 0, result.output
    assert (folder / ".env").read_text() == (
        "# mine\n"
        "OPENROUTER_API_KEY=sk-kept\n"
        f"PARLANT_DASHBOARD_PROJECT_TOKEN={PROJECT_TOKEN}\n"
        "\n"
        "EXTRA=1\n"
    )
    assert PROJECT_TOKEN not in result.output
    assert "PARLANT_DASHBOARD_PROJECT_TOKEN" in result.output


def test_that_project_check_reports_connected_when_the_dashboard_sees_the_server(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path)
    project_id = local_project(dashboard)
    dashboard.connect_server(project_id)

    result = invoke(rt, ["dashboard", "--project", project_id, "project", "check", "--json"])

    assert result.exit_code == 0, result.output
    assert only_json(result) == {
        "projectId": project_id,
        "phase": "connected",
        "connected": True,
        "parlantVersion": "4.0.0a1",
        "setupStatus": "pending",
    }


def test_that_project_check_reports_unreachable_when_the_server_has_not_connected(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path)
    project_id = local_project(dashboard)

    result = invoke(rt, ["dashboard", "--project", project_id, "project", "check", "--json"])

    # Not being connected yet is an answer, not a failure: the app polls on it.
    assert result.exit_code == 0, result.output
    report = only_json(result)
    assert report["phase"] == "unreachable"
    assert report["connected"] is False
    assert report["message"]


def test_that_project_confirm_completes_the_pending_setup(tmp_path: Path) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path)
    project_id = local_project(dashboard)
    dashboard.connect_server(project_id)

    result = invoke(rt, ["dashboard", "--project", project_id, "project", "confirm", "--json"])

    assert result.exit_code == 0, result.output
    assert only_json(result) == {"projectId": project_id, "setupComplete": True}
    assert dashboard.projects[project_id]["setup"]["status"] == "completed"
    # A server without sessions has nothing to import.
    assert not dashboard.sent("POST", f"/projects/{project_id}/sessions/sync")


def test_that_project_confirm_fails_when_the_dashboard_has_not_seen_the_server(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path)
    project_id = local_project(dashboard)
    dashboard.server_sessions[project_id] = 3

    result = invoke(rt, ["dashboard", "--project", project_id, "project", "confirm", "--json"])

    assert result.exit_code != 0
    assert only_json(result)["error"]["code"] == "NOT_CONNECTED"
    assert dashboard.projects[project_id]["setup"]["status"] == "pending"
    assert not dashboard.sent("POST", f"/projects/{project_id}/sessions/sync")


def test_that_project_confirm_syncs_existing_sessions_and_then_completes_the_setup(
    tmp_path: Path,
) -> None:
    # A coding agent usually chats with the agent before confirming, so its
    # server already holds sessions. Confirming imports them, as the
    # dashboard's Sync would, instead of leaving setup to the dashboard.
    dashboard = FakeDashboard()
    dashboard.import_polls = 2
    rt = signed_in(dashboard, tmp_path)
    project_id = local_project(dashboard)
    dashboard.connect_server(project_id)
    dashboard.server_sessions[project_id] = 4

    result = invoke(rt, ["dashboard", "--project", project_id, "project", "confirm", "--json"])

    assert result.exit_code == 0, result.output
    assert only_json(result) == {
        "projectId": project_id,
        "setupComplete": True,
        "importedSessions": 4,
    }
    assert dashboard.projects[project_id]["setup"]["status"] == "completed"
    assert len(dashboard.sent("POST", f"/projects/{project_id}/sessions/sync")) == 1
    # Progress goes to stderr, so stdout stays one JSON document.
    assert "Importing" in result.stderr


def test_that_project_confirm_reports_sync_in_progress_and_a_rerun_completes_the_setup(
    tmp_path: Path,
) -> None:
    # Coding agents often stop a command after about two minutes, so confirm
    # stops waiting before that and says to run it again.
    dashboard = FakeDashboard()
    dashboard.import_polls = 1_000
    rt = signed_in(dashboard, tmp_path)
    project_id = local_project(dashboard)
    dashboard.connect_server(project_id)
    dashboard.server_sessions[project_id] = 400

    first = invoke(rt, ["dashboard", "--project", project_id, "project", "confirm", "--json"])

    assert first.exit_code != 0
    error = only_json(first)["error"]
    assert error["code"] == "SYNC_IN_PROGRESS"
    assert "200 of 400" in error["message"]
    assert "project confirm" in error["message"]
    assert dashboard.projects[project_id]["setup"]["status"] == "pending"

    dashboard.importing[project_id] = 0
    second = invoke(rt, ["dashboard", "--project", project_id, "project", "confirm", "--json"])

    assert second.exit_code == 0, second.output
    assert only_json(second)["setupComplete"] is True
    # The rerun waited for the running import instead of starting another.
    assert len(dashboard.sent("POST", f"/projects/{project_id}/sessions/sync")) == 1


def test_that_project_confirm_fails_with_the_import_error_when_the_sync_fails(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()
    dashboard.import_failure = "The server stopped answering session reads."
    rt = signed_in(dashboard, tmp_path)
    project_id = local_project(dashboard)
    dashboard.connect_server(project_id)
    dashboard.server_sessions[project_id] = 4

    result = invoke(rt, ["dashboard", "--project", project_id, "project", "confirm", "--json"])

    assert result.exit_code != 0
    error = only_json(result)["error"]
    assert error["code"] == "SYNC_FAILED"
    assert "The server stopped answering session reads." in error["message"]
    assert "project confirm" in error["message"]
    assert dashboard.projects[project_id]["setup"]["status"] == "pending"


def test_that_init_with_dashboard_creates_the_project_before_writing_any_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "support-bot"
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path)
    dashboard.watch = target

    result = invoke(rt, ["init", str(target), *INIT_WITH_DASHBOARD])

    assert result.exit_code == 0, result.output
    # The dashboard admitted the project while the target folder was still empty.
    assert dashboard.files_when_created == []
    [project] = dashboard.projects.values()
    assert project["projectKind"] == "local" and project["name"] == "Support Bot"
    assert (target / "main.py").is_file()
    assert f"PARLANT_DASHBOARD_PROJECT_TOKEN={PROJECT_TOKEN}\n" in (target / ".env").read_text()
    assert PROJECT_TOKEN not in result.output
    assert "./run.sh" in result.output
    assert f"parlant dashboard --project {project['id']} project confirm" in result.output
    # Confirming waits for the server; init never completes setup on its own.
    assert project["setup"]["status"] == "pending"


def test_that_init_with_dashboard_json_lists_the_next_steps_ending_in_confirm(
    tmp_path: Path,
) -> None:
    # A coding agent reads only the JSON; the human panel is not printed there.
    target = tmp_path / "support-bot"
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path)

    result = invoke(rt, ["init", str(target), *INIT_WITH_DASHBOARD, "--json"])

    assert result.exit_code == 0, result.output
    [project] = dashboard.projects.values()
    report = only_json(result)
    assert report["projectId"] == project["id"]
    assert report["nextSteps"] == [
        f"cd {target}",
        "uv sync",
        "Set OPENROUTER_API_KEY in .env.",
        "./run.sh (run.cmd on Windows) starts the server; it connects to the dashboard on its own.",
        f"parlant dashboard --project {project['id']} project check",
        f"parlant dashboard --project {project['id']} project confirm",
    ]


def test_that_init_with_dashboard_creates_the_project_in_the_given_workspace(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path, "Other", "Emcie")

    result = invoke(
        rt, ["init", str(tmp_path / "support-bot"), *INIT_WITH_DASHBOARD, "--workspace", "Other"]
    )

    assert result.exit_code == 0, result.output
    [project] = dashboard.projects.values()
    assert project["tenantId"] == "Other"


def test_that_init_with_dashboard_writes_nothing_when_the_dashboard_refuses_the_project(
    tmp_path: Path,
) -> None:
    target = tmp_path / "support-bot"
    dashboard = FakeDashboard(
        create_refusal=httpx.Response(
            409,
            json={"code": "PROJECT_LIMIT_REACHED", "message": "No free local project slot."},
        )
    )
    rt = signed_in(dashboard, tmp_path)

    result = invoke(rt, ["init", str(target), *INIT_WITH_DASHBOARD])

    assert result.exit_code != 0
    assert "No free local project slot." in result.output
    assert not target.exists()


def test_that_any_failing_command_with_json_prints_an_error_envelope_with_its_code(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard(
        create_refusal=httpx.Response(
            409,
            json={"code": "PROJECT_LIMIT_REACHED", "message": "No free local project slot."},
        )
    )
    rt = signed_in(dashboard, tmp_path)

    result = invoke(rt, ["dashboard", *CREATE_LOCAL, "--json"])

    assert result.exit_code != 0
    assert only_json(result) == {
        "error": {"code": "PROJECT_LIMIT_REACHED", "message": "No free local project slot."}
    }


def test_that_the_cli_signs_in_to_the_production_dashboard_api_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PARLANT_DASHBOARD_BASE_URL", raising=False)

    # dashboard.parlant.io is the web app, whose CDN answers API paths with the page.
    assert resolve_host() == "https://api.parlant.io"


def test_that_one_sign_in_approved_for_several_workspaces_saves_a_sign_in_for_each_and_makes_the_first_active(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard(approvals=[["Other", "Emcie"]], pending_polls=0)
    rt = runtime(dashboard, tmp_path)

    login = invoke(rt, ["dashboard", "auth", "login"])

    assert login.exit_code == 0, login.output
    assert dashboard.signed_in_workspaces() == {"Other", "Emcie"}
    assert only_json(invoke(rt, ["dashboard", "workspace", "list", "--json"])) == [
        {"id": "Emcie", "name": "Emcie", "role": "owner", "signedIn": True, "active": False},
        {"id": "Other", "name": "Other", "role": "builder", "signedIn": True, "active": True},
    ]
    # Each workspace's own sign-in is used for it.
    local_project(dashboard, "a" * 24, workspace="Emcie")
    projects = invoke(rt, ["dashboard", "--workspace", "Emcie", "project", "list", "--json"])
    assert [p["id"] for p in only_json(projects)] == ["a" * 24]


def test_that_signing_in_again_for_several_workspaces_replaces_and_revokes_only_those_workspaces_old_sign_ins(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path, "Emcie", "Other")
    old = {w: t for t, (w, _) in dashboard.refresh_tokens.items()}

    dashboard.approvals = [["Other"]]
    dashboard.pending_polls = 0
    result = invoke(rt, ["dashboard", "auth", "login"])

    assert result.exit_code == 0, result.output
    assert dashboard.revoked == [old["Other"]]
    assert old["Emcie"] in dashboard.refresh_tokens
    assert dashboard.signed_in_workspaces() == {"Emcie", "Other"}


def test_that_auth_login_with_json_reports_every_workspace_it_signed_in_to(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard(approvals=[["Other", "Emcie"]], pending_polls=0)

    result = invoke(
        runtime(dashboard, tmp_path), ["dashboard", "auth", "login", "--json", "--no-browser"]
    )

    assert result.exit_code == 0, result.output
    signed_in_event = json_lines(result)[-1]
    assert signed_in_event == {
        "event": "signed-in",
        "signedIn": True,
        "host": HOST,
        "workspace": "Other",
        "workspaces": ["Other", "Emcie"],
        "user": {"id": "user-1"},
    }


def test_that_a_dashboard_answering_with_a_single_sign_in_still_works(tmp_path: Path) -> None:
    # Backends without multi-workspace approval answer with one token, not grants.
    dashboard = FakeDashboard(approvals=["Emcie"], pending_polls=0)
    rt = runtime(dashboard, tmp_path)

    result = invoke(rt, ["dashboard", "auth", "login", "--json", "--no-browser"])

    assert result.exit_code == 0, result.output
    assert json_lines(result)[-1]["workspaces"] == ["Emcie"]
    assert dashboard.signed_in_workspaces() == {"Emcie"}
    assert only_json(invoke(rt, ["dashboard", "auth", "status", "--json"]))["workspace"] == "Emcie"


def test_that_project_list_with_json_includes_a_managed_projects_repository_and_branch(
    tmp_path: Path,
) -> None:
    dashboard = FakeDashboard()
    rt = signed_in(dashboard, tmp_path)
    dashboard.projects["b" * 24] = {
        "id": "b" * 24,
        "name": "Support Bot",
        "tenantId": "Emcie",
        "projectKind": "managed",
        "lifecycleState": "active",
        "setup": {"status": "completed", "completedAt": "2026-10-17T10:00:00Z"},
        "deployment": {
            "hostingMode": "managed",
            "githubRepoFullName": "acme/support-bot",
            "branch": "main",
        },
    }
    local_id = local_project(dashboard)

    result = invoke(rt, ["dashboard", "project", "list", "--json"])

    assert result.exit_code == 0, result.output
    by_id = {p["id"]: p for p in only_json(result)}
    assert by_id["b" * 24]["repository"] == "acme/support-bot"
    assert by_id["b" * 24]["branch"] == "main"
    assert by_id[local_id]["repository"] is None
    assert by_id[local_id]["branch"] is None


class PricingDashboard(FakeDashboard):
    def _connection(self, project: dict[str, Any]) -> dict[str, Any]:
        return {
            **super()._connection(project),
            "pricing": {
                "price_table": False,
                "currency": None,
                "models": [
                    {"key": "gemini:needs-price", "price": "missing", "source": "configured"},
                    {
                        "key": "openrouter:typesafe/jev",
                        "price": "provider_reported",
                        "source": "configured",
                    },
                ],
            },
        }


@pytest.mark.parametrize("command", ["check", "confirm"])
def test_that_local_connection_commands_explain_missing_prices(
    tmp_path: Path, command: str
) -> None:
    dashboard = PricingDashboard()
    rt = signed_in(dashboard, tmp_path)
    project_id = local_project(dashboard)
    dashboard.connect_server(project_id)
    result = invoke(rt, ["dashboard", "--project", project_id, "project", command])
    assert result.exit_code == 0, result.output
    assert "gemini:needs-price" in result.output
    assert "set_pricing" in result.output
    assert "https://parlant.io/docs/" in result.output
    assert "typesafe/jev" not in result.output


def test_that_json_connection_keeps_structured_pricing_without_mixing_human_text(
    tmp_path: Path,
) -> None:
    dashboard = PricingDashboard()
    rt = signed_in(dashboard, tmp_path)
    project_id = local_project(dashboard)
    dashboard.connect_server(project_id)
    result = invoke(rt, ["dashboard", "--project", project_id, "project", "check", "--json"])
    assert result.exit_code == 0, result.output
    report = only_json(result)
    assert report["pricing"] == dashboard._connection(dashboard.projects[project_id])["pricing"]
