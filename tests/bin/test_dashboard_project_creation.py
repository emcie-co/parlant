import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from click.testing import CliRunner

from parlant.bin.cli import cli
from parlant.bin.dashboard.runtime import DashboardRuntime


def runtime(handler: Any) -> DashboardRuntime:
    return DashboardRuntime(
        host="https://dashboard.test",
        http=httpx.Client(
            base_url="https://dashboard.test", transport=httpx.MockTransport(handler)
        ),
        credentials=cast(Any, None),
        config=cast(Any, None),
        open_browser=lambda url: False,
        sleep=lambda seconds: None,
    )


class FakeBackend:
    """Answers like Backend's project, GitHub and managed-setup routes.

    A managed project is created from a repository and a branch; its setup check
    starts on its own and finishes after `check_polls` reads of the setup."""

    def __init__(
        self,
        *,
        check: dict[str, Any] | None = None,
        check_polls: int = 1,
        found: dict[str, Any] | None = None,
        creates: list[httpx.Response] | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.creates = creates
        self.check = check or done_check("standard", "Looks like a standard Parlant project.")
        self.check_polls = check_polls
        self.found = found or {"settings": {"runCommand": "parlant-server run"}}
        self.projects: dict[str, dict[str, Any]] = {}
        self.setups: dict[str, dict[str, Any]] = {}
        self.secret_values: dict[str, str] = {}

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        path, method = req.url.path, req.method
        if path == "/projects/creation-readiness":
            return httpx.Response(200, json={"ready": True, "blocker": None})
        if path == "/github/installations":
            return httpx.Response(
                200, json=[{"id": "installation", "repositories": [{"fullName": "example/agent"}]}]
            )
        if path.endswith("/branches"):
            return httpx.Response(200, json=[{"name": "main"}])
        if method == "POST" and path == "/projects":
            return self._create(json.loads(req.content))
        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[0] == "projects" and parts[2] == "managed-setup":
            setup = self.setups.get(parts[1])
            if setup is None:
                return httpx.Response(
                    404,
                    json={"code": "not-found", "message": "We couldn't find this project's setup."},
                )
            return self._setup(req, setup, parts[3:])
        if method == "GET" and len(parts) == 2 and parts[0] == "projects":
            if parts[1] in self.projects:
                return httpx.Response(200, json=self.projects[parts[1]])
            return httpx.Response(404, json={"message": "Project not found"})
        return httpx.Response(
            404,
            json={
                "message": f"Route {method}:{path} not found",
                "error": "Not Found",
                "statusCode": 404,
            },
        )

    def _create(self, body: dict[str, Any]) -> httpx.Response:
        if self.creates:
            response = self.creates.pop(0)
            if response.status_code not in (200, 201):
                return response
        if body.get("projectKind") != "managed":
            if "managedDeployment" in body:
                return httpx.Response(400, json={"message": "body must NOT be valid"})
            project_id = body.get("id") or "f" * 24
            self.projects[project_id] = {
                "id": project_id,
                "name": body.get("name"),
                "projectKind": body["projectKind"],
            }
            return httpx.Response(201, json=self.projects[project_id])
        if "id" not in body or "managedDeployment" not in body:
            return httpx.Response(
                400, json={"message": "body must have required property 'managedDeployment'"}
            )
        deployment = body["managedDeployment"]
        assert set(deployment) <= {
            "githubInstallationId",
            "githubRepoFullName",
            "branch",
            "runCommand",
            "buildCommand",
            "buildEnvVars",
            "runEnvVars",
            "managedHostSecrets",
        }
        self.projects[body["id"]] = {
            "id": body["id"],
            "name": body["name"],
            "projectKind": "managed",
        }
        self.setups[body["id"]] = {
            "projectId": body["id"],
            "projectName": body["name"],
            "revision": 1,
            "repository": {
                "fullName": deployment["githubRepoFullName"],
                "branch": deployment["branch"],
            },
            "check": {
                "status": "running",
                "checkId": "check-1",
                "commit": None,
                "stage": None,
                "startedAt": "2026-10-01T10:00:00.000Z",
                "first": True,
            },
            "deploy": {"status": "none"},
            "settings": {"buildCommand": "", "runCommand": "", "buildEnv": [], "runEnv": []},
            "origins": {
                "buildCommand": "empty",
                "runCommand": "empty",
                "buildEnv": "empty",
                "runEnv": "empty",
            },
            "runCommandSuggestions": [],
            "suggestions": [],
            "savedSecrets": [],
            "expectedSecrets": [],
            "newCommitsSinceCheck": 0,
        }
        return httpx.Response(201, json={"id": body["id"], "name": body["name"]})

    def _setup(self, req: httpx.Request, setup: dict[str, Any], rest: list[str]) -> httpx.Response:
        if req.method == "GET" and not rest:
            if setup["check"]["status"] == "running":
                self.check_polls -= 1
                if self.check_polls <= 0:
                    self._finish_check(setup)
            return httpx.Response(200, json=setup)
        if req.method == "PATCH" and rest == ["settings"]:
            patch = json.loads(req.content)
            if not set(patch) <= {"buildCommand", "runCommand", "buildEnv", "runEnv"}:
                return httpx.Response(
                    400, json={"code": "invalid", "message": "Enter valid managed setup input."}
                )
            secrets = {row["name"] for row in setup["savedSecrets"]}
            for field in ("buildEnv", "runEnv"):
                names: set[str] = set()
                for row in patch.get(field, []):
                    message = None
                    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", row["name"]):
                        message = "Use letters, digits and underscores."
                    elif row["name"] in names:
                        message = "Use each variable name only once."
                    elif field == "runEnv" and row["name"] in secrets:
                        message = "This name is already a secret. Keep it in one place."
                    if message:
                        return httpx.Response(
                            400,
                            json={
                                "code": "invalid",
                                "message": message,
                                "field": field,
                                "name": row["name"],
                            },
                        )
                    names.add(row["name"])
            for field, value in patch.items():
                setup["settings"][field] = value
                setup["origins"][field] = "user"
            setup["revision"] += 1
            return httpx.Response(200, json=setup)
        if req.method == "PUT" and len(rest) == 2 and rest[0] == "secrets":
            value = json.loads(req.content).get("value")
            if not value:
                return httpx.Response(
                    400, json={"code": "invalid", "message": "Enter a valid secret value."}
                )
            if any(row["name"] == rest[1] for row in setup["settings"]["runEnv"]):
                return httpx.Response(
                    400,
                    json={
                        "code": "invalid",
                        "message": "This name is already a runtime variable. Keep it in one place.",
                    },
                )
            self.secret_values[rest[1]] = value
            setup["savedSecrets"] = sorted(
                [row for row in setup["savedSecrets"] if row["name"] != rest[1]]
                + [{"name": rest[1], "updatedAt": "2026-10-01T10:01:00.000Z"}],
                key=lambda row: row["name"],
            )
            setup["revision"] += 1
            return httpx.Response(200, json=setup)
        raise AssertionError(f"Unexpected {req.method} {req.url.path}")

    def _finish_check(self, setup: dict[str, Any]) -> None:
        setup["check"] = self.check
        for field, value in self.found.get("settings", {}).items():
            if setup["origins"][field] != "user":
                setup["settings"][field] = value
                setup["origins"][field] = "check"
        for key in ("runCommandSuggestions", "suggestions", "expectedSecrets"):
            setup[key] = self.found.get(key, [])
        setup["revision"] += 1

    def sent(self, method: str, path_end: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path.endswith(path_end)]


def done_check(
    verdict: str, summary: str, review_fields: list[str] | None = None
) -> dict[str, Any]:
    return {
        "status": "done",
        "checkId": "check-1",
        "commit": {"sha": "a" * 40, "message": "Initial agent", "branch": "main"},
        "verdict": verdict,
        "summary": summary,
        "reviewFields": review_fields or [],
        "finishedAt": "2026-10-01T10:00:30.000Z",
    }


def test_managed_creation_creates_first_then_saves_settings_through_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorize_creation(monkeypatch)
    monkeypatch.setenv("PROJECT_TEST_SECRET", "do-not-print-this")
    backend = FakeBackend(check_polls=3)

    result = CliRunner().invoke(
        cli,
        [
            "dashboard",
            "project",
            "create",
            "--kind",
            "managed",
            "--name",
            "Test Project",
            "--repo",
            "example/agent",
            "--branch",
            "main",
            "--run-command",
            "python main.py",
            "--build-env",
            "BUILD=yes",
            "--run-env",
            "MODE=dev",
            "--secret",
            "API_KEY=PROJECT_TEST_SECRET",
            "--yes",
        ],
        obj=runtime(backend),
    )

    assert result.exit_code == 0, result.output
    creates = backend.sent("POST", "/projects")
    assert len(creates) == 1
    created = json.loads(creates[0].content)
    assert created["managedDeployment"] == {
        "githubInstallationId": "installation",
        "githubRepoFullName": "example/agent",
        "branch": "main",
    }
    assert not any("repository-inspections" in r.url.path for r in backend.requests)
    assert len(backend.sent("GET", "/managed-setup")) == 3
    [patch] = backend.sent("PATCH", "/managed-setup/settings")
    assert json.loads(patch.content) == {
        "runCommand": "python main.py",
        "buildEnv": [{"name": "BUILD", "value": "yes"}],
        "runEnv": [{"name": "MODE", "value": "dev"}],
    }
    assert backend.secret_values == {"API_KEY": "do-not-print-this"}
    assert "Looks like a standard Parlant project" in result.output
    assert "Run command: python main.py" in result.output
    assert "do-not-print-this" not in result.output
    project_id = created["id"]
    assert f"parlant dashboard --project {project_id} deploy" in result.output


def test_check_result_shows_detected_run_command_and_warnings_without_saving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorize_creation(monkeypatch)
    backend = FakeBackend(
        check=done_check(
            "review",
            "This repository starts its server its own way.",
            review_fields=["runCommand"],
        ),
        found={
            "settings": {"runCommand": "python -m agent", "buildCommand": "uv sync"},
            "suggestions": [
                {
                    "id": "s1",
                    "kind": "add-env",
                    "scope": "run",
                    "name": "LOG_LEVEL",
                    "value": "info",
                    "reason": "The server reads LOG_LEVEL at start.",
                }
            ],
            "expectedSecrets": [{"name": "OPENAI_API_KEY", "reason": "Read by the agent."}],
        },
    )
    args = [a for a in CREATE_ARGS if a not in ("--run-command", "python main.py")]

    result = CliRunner().invoke(cli, args + ["--yes"], obj=runtime(backend))

    assert result.exit_code == 0, result.output
    assert "review the marked fields" in result.output
    assert "This repository starts its server its own way." in result.output
    assert "Run command: python -m agent" in result.output
    assert "Build command: uv sync" in result.output
    assert "Review the run command" in result.output
    assert "LOG_LEVEL" in result.output and "The server reads LOG_LEVEL at start." in result.output
    assert "OPENAI_API_KEY" in result.output
    assert not backend.sent("PATCH", "/managed-setup/settings")
    assert not backend.sent("PUT", "/managed-setup/secrets/OPENAI_API_KEY")


def test_failed_check_keeps_the_project_and_says_how_to_proceed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorize_creation(monkeypatch)
    backend = FakeBackend(
        check={
            "status": "failed",
            "checkId": "check-1",
            "commit": None,
            "finishedAt": "2026-10-01T10:00:30.000Z",
        },
        found={},
    )

    result = CliRunner().invoke(cli, CREATE_ARGS + ["--yes"], obj=runtime(backend))

    assert result.exit_code == 0, result.output
    project_id = json.loads(backend.sent("POST", "/projects")[0].content)["id"]
    assert f"Project created: {project_id}" in result.output
    assert "setup check failed" in result.output.lower()
    assert "setup page" in result.output
    [patch] = backend.sent("PATCH", "/managed-setup/settings")
    assert json.loads(patch.content) == {"runCommand": "python main.py"}


def test_check_still_running_at_the_deadline_saves_settings_and_points_to_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    authorize_creation(monkeypatch)
    clock = iter(range(0, 10_000, 30))
    monkeypatch.setattr(
        "parlant.bin.dashboard.commands.project_creation.time.monotonic", lambda: float(next(clock))
    )
    backend = FakeBackend(check_polls=1_000)

    result = CliRunner().invoke(cli, CREATE_ARGS + ["--yes"], obj=runtime(backend))

    assert result.exit_code == 0, result.output
    assert "still running" in result.output
    assert "setup page" in result.output
    [patch] = backend.sent("PATCH", "/managed-setup/settings")
    assert json.loads(patch.content) == {"runCommand": "python main.py"}


def test_interactive_creation_offers_the_checks_run_commands_after_creating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorize_creation(monkeypatch)
    backend = FakeBackend(
        check=done_check("unknown", "Two servers could start this agent."),
        found={
            "runCommandSuggestions": [
                {"command": "python main.py", "likely": True},
                {"command": "python other.py", "reason": "Also starts a server.", "likely": False},
            ]
        },
    )
    args = [a for a in CREATE_ARGS if a not in ("--run-command", "python main.py")]

    result = CliRunner().invoke(cli, args, input="y\n2\n\n\n\n\n\n", obj=runtime(backend))

    assert result.exit_code == 0, result.output
    assert result.output.index("Project created") < result.output.index("python other.py")
    [patch] = backend.sent("PATCH", "/managed-setup/settings")
    assert json.loads(patch.content) == {"runCommand": "python other.py"}


def test_skip_inspection_is_no_longer_an_option(monkeypatch: pytest.MonkeyPatch) -> None:
    authorize_creation(monkeypatch)
    backend = FakeBackend()

    result = CliRunner().invoke(
        cli, CREATE_ARGS + ["--skip-inspection", "--yes"], obj=runtime(backend)
    )

    assert result.exit_code == 2
    assert "--skip-inspection" in result.output
    assert not backend.requests


def test_missing_scope_stops_before_setup_or_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    import parlant.bin.dashboard.commands.project_creation as creation

    monkeypatch.setattr(creation, "authenticate", lambda r: "token")
    monkeypatch.setattr(
        creation, "fetch_session", lambda r, t: SimpleNamespace(scope="projects.read")
    )
    result = CliRunner().invoke(
        cli,
        ["dashboard", "project", "create"],
        obj=runtime(lambda req: pytest.fail("No setup request permitted")),
    )
    assert result.exit_code != 0
    assert "auth login --reauthorize" in result.output


@pytest.mark.parametrize("succeeds", [False, True])
def test_reauthorization_preserves_previous_login_until_new_grant_is_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, succeeds: bool
) -> None:
    import parlant.bin.dashboard.commands.auth as auth_commands
    from parlant.bin.dashboard.config import FileConfigStore
    from parlant.bin.dashboard.credentials import Credential, credential_key
    from parlant.bin.dashboard.errors import DashboardError

    previous = Credential(refresh_token="previous-token", session_id="previous-session")
    key = credential_key("https://dashboard.test", "Emcie")
    saved = [previous]
    revoked = []
    r = runtime(lambda req: pytest.fail("Unexpected HTTP"))
    r.config = FileConfigStore(tmp_path / "config.json")
    r.config.add_signed_in_workspace(r.host, "Emcie")
    r.config.set_active_workspace(r.host, "Emcie")
    r.credentials = cast(
        Any,
        SimpleNamespace(
            get=lambda k: saved[-1] if k == key else None,
            save=lambda k, credential: saved.append(credential) if k == key else pytest.fail(k),
        ),
    )
    monkeypatch.setattr(
        auth_commands,
        "start_device_authorization",
        lambda r: SimpleNamespace(
            user_code="123456",
            verification_uri="https://dashboard.test/authorize",
            verification_uri_complete="https://dashboard.test/authorize?code=123456",
        ),
    )

    def poll(r: Any) -> Any:
        assert saved == [previous]
        if not succeeds:
            raise DashboardError("Approval expired")
        return [
            SimpleNamespace(
                access_token="new-access", refresh_token="new-token", session_id="new-session"
            )
        ]

    monkeypatch.setattr(auth_commands, "poll_for_token", lambda r, device: poll(r))
    monkeypatch.setattr(
        auth_commands,
        "fetch_session",
        lambda r, access_token: SimpleNamespace(tenant_id="Emcie", user_id="user-1"),
    )
    monkeypatch.setattr(
        auth_commands, "revoke_session", lambda r, credential: revoked.append(credential)
    )
    result = CliRunner().invoke(cli, ["dashboard", "auth", "login", "--reauthorize"], obj=r)
    assert result.exit_code == (0 if succeeds else 1), result.output
    assert len(saved) == (2 if succeeds else 1)
    assert revoked == ([previous] if succeeds else [])
    assert "previous-token" not in result.output
    assert "new-token" not in result.output


def authorize_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    import parlant.bin.dashboard.commands.project_creation as creation

    monkeypatch.setattr(creation, "authenticate", lambda r: "token")
    monkeypatch.setattr(
        creation,
        "fetch_session",
        lambda r, t: SimpleNamespace(scope="projects.create projects.update"),
    )


CREATE_ARGS = [
    "dashboard",
    "project",
    "create",
    "--kind",
    "managed",
    "--name",
    "Test Project",
    "--repo",
    "example/agent",
    "--branch",
    "main",
    "--run-command",
    "python main.py",
]


def test_canceling_final_review_never_creates_a_project(monkeypatch: pytest.MonkeyPatch) -> None:
    authorize_creation(monkeypatch)
    backend = FakeBackend()
    result = CliRunner().invoke(cli, CREATE_ARGS, input="n\n", obj=runtime(backend))
    assert result.exit_code == 0, result.output
    assert "Canceled. No project was created." in result.output
    assert not backend.sent("POST", "/projects")


def test_in_progress_retry_reuses_identical_complete_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorize_creation(monkeypatch)
    backend = FakeBackend(
        creates=[
            httpx.Response(202, json={"status": "in_progress", "retryAfterSeconds": 1}),
            httpx.Response(200, json={"id": "123456789012345678901234"}),
        ]
    )
    result = CliRunner().invoke(
        cli,
        CREATE_ARGS + ["--yes", "--request-id", "123456789012345678901234"],
        obj=runtime(backend),
    )
    assert result.exit_code == 0, result.output
    creates = backend.sent("POST", "/projects")
    assert len(creates) == 2
    assert creates[0].content == creates[1].content


def test_resume_recovers_committed_project_without_collecting_or_resending_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorize_creation(monkeypatch)
    requests: list[httpx.Request] = []

    def respond(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        assert req.method == "GET" and req.url.path == "/projects/123456789012345678901234"
        return httpx.Response(200, json={"id": "123456789012345678901234"})

    result = CliRunner().invoke(
        cli,
        ["dashboard", "project", "create", "--request-id", "123456789012345678901234"],
        obj=runtime(respond),
    )
    assert result.exit_code == 0, result.output
    assert len(requests) == 1


def test_resume_with_setup_options_saves_them_on_the_recovered_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorize_creation(monkeypatch)
    monkeypatch.setenv("PROJECT_TEST_SECRET", "do-not-print-this")
    backend = FakeBackend()
    identifier = "123456789012345678901234"
    backend(
        httpx.Request(
            "POST",
            "https://dashboard.test/projects",
            json={
                "id": identifier,
                "name": "Test Project",
                "projectKind": "managed",
                "managedDeployment": {
                    "githubInstallationId": "installation",
                    "githubRepoFullName": "example/agent",
                    "branch": "main",
                },
            },
        )
    )
    backend.requests.clear()

    def resume() -> Any:
        return CliRunner().invoke(
            cli,
            CREATE_ARGS
            + ["--secret", "API_KEY=PROJECT_TEST_SECRET", "--yes", "--request-id", identifier],
            obj=runtime(backend),
        )

    first, second = resume(), resume()

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert not backend.sent("POST", "/projects")
    assert [json.loads(r.content) for r in backend.sent("PATCH", "/managed-setup/settings")] == [
        {"runCommand": "python main.py"}
    ]
    assert len(backend.sent("PUT", "/managed-setup/secrets/API_KEY")) == 2
    assert [row["name"] for row in backend.setups[identifier]["savedSecrets"]] == ["API_KEY"]
    assert "do-not-print-this" not in first.output + second.output


def test_a_secret_named_like_a_detected_run_variable_moves_out_of_the_run_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorize_creation(monkeypatch)
    monkeypatch.setenv("PROJECT_TEST_SECRET", "postgres://secret")
    backend = FakeBackend(
        found={
            "settings": {
                "runCommand": "python main.py",
                "runEnv": [
                    {"name": "DATABASE_URL", "value": ""},
                    {"name": "PORT", "value": "8800"},
                ],
            }
        }
    )

    result = CliRunner().invoke(
        cli,
        CREATE_ARGS + ["--secret", "DATABASE_URL=PROJECT_TEST_SECRET", "--yes"],
        obj=runtime(backend),
    )

    assert result.exit_code == 0, result.output
    [project_id] = backend.setups
    assert backend.setups[project_id]["settings"]["runEnv"] == [{"name": "PORT", "value": "8800"}]
    assert backend.secret_values == {"DATABASE_URL": "postgres://secret"}
    assert "postgres://secret" not in result.output


def test_managed_creation_requires_the_update_grant_before_anything_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorize_creation(monkeypatch)
    import parlant.bin.dashboard.commands.project_creation as creation

    monkeypatch.setattr(
        creation, "fetch_session", lambda r, t: SimpleNamespace(scope="projects.create")
    )
    backend = FakeBackend()

    result = CliRunner().invoke(cli, CREATE_ARGS + ["--yes"], obj=runtime(backend))

    assert result.exit_code != 0
    assert "projects.update" in result.output
    assert "auth login --reauthorize" in result.output
    assert not backend.requests


def test_a_setup_failure_after_creation_says_the_project_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorize_creation(monkeypatch)
    backend = FakeBackend()

    def respond(req: httpx.Request) -> httpx.Response:
        if req.method == "PATCH":
            backend.requests.append(req)
            return httpx.Response(
                503,
                json={
                    "code": "unavailable",
                    "message": "Managed setup is temporarily unavailable. Try again.",
                },
            )
        return backend(req)

    result = CliRunner().invoke(cli, CREATE_ARGS + ["--yes"], obj=runtime(respond))

    assert result.exit_code == 1
    [project_id] = backend.setups
    assert "temporarily unavailable" in result.output
    assert f"Project {project_id} was created; finish its setup in the dashboard" in result.output
    assert f"parlant dashboard --project {project_id} deploy" in result.output


def test_changed_payload_conflict_does_not_retry_with_a_new_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorize_creation(monkeypatch)
    backend = FakeBackend(
        creates=[httpx.Response(409, json={"message": "project_creation_request_conflict"})]
    )
    result = CliRunner().invoke(
        cli,
        CREATE_ARGS + ["--yes", "--request-id", "123456789012345678901234"],
        obj=runtime(backend),
    )
    assert result.exit_code != 0
    assert "123456789012345678901234" in result.output
    assert len(backend.sent("POST", "/projects")) == 1
    assert not backend.sent("GET", "/managed-setup")


@pytest.mark.parametrize("case", ["empty-branches", "literal-secret", "secret-and-run-env"])
def test_invalid_setup_stops_before_creation_without_exposing_secrets(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    authorize_creation(monkeypatch)
    monkeypatch.setenv("PROJECT_TEST_SECRET", "SYNTHETIC-SECRET-VALUE")
    backend = FakeBackend()

    def respond(req: httpx.Request) -> httpx.Response:
        if case == "empty-branches" and req.url.path.endswith("/branches"):
            backend.requests.append(req)
            return httpx.Response(200, json=[])
        return backend(req)

    args = list(CREATE_ARGS)
    if case == "empty-branches":
        index = args.index("--branch")
        del args[index : index + 2]
    elif case == "literal-secret":
        args.extend(["--yes", "--secret", "TOKEN=SYNTHETIC-SECRET-VALUE"])
    else:
        args.extend(["--yes", "--secret", "TOKEN=PROJECT_TEST_SECRET", "--run-env", "TOKEN=x"])
    result = CliRunner().invoke(cli, args, input="main\ny\n", obj=runtime(respond))
    assert result.exit_code != 0, result.output
    assert "SYNTHETIC-SECRET-VALUE" not in result.output
    assert not backend.sent("POST", "/projects")
    if case == "empty-branches":
        assert "No branch options" in result.output


def test_connecting_github_and_selecting_repository_precedes_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorize_creation(monkeypatch)
    opened: list[str] = []
    backend = FakeBackend()
    import parlant.bin.dashboard.commands.project_creation as creation

    monkeypatch.setattr(creation, "authenticate", lambda r: "refreshed" if opened else "initial")

    def respond(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/github/installations" and not opened:
            return httpx.Response(200, json=[])
        if req.url.path == "/github/install-url":
            return httpx.Response(
                200, json={"url": "https://github.com/apps/test/installations/new"}
            )
        if opened:
            assert req.headers["authorization"] == "Bearer refreshed"
        return backend(req)

    r = runtime(respond)

    def record_opened(url: str) -> bool:
        opened.append(url)
        return True

    r.open_browser = record_opened
    args = list(CREATE_ARGS)
    index = args.index("--repo")
    del args[index : index + 2]
    result = CliRunner().invoke(cli, args, input="y\n\n1\ny\n\n\n\n\n\n", obj=r)
    assert result.exit_code == 0, result.output
    assert opened == ["https://github.com/apps/test/installations/new"]
    [create] = backend.sent("POST", "/projects")
    assert json.loads(create.content)["managedDeployment"]["githubRepoFullName"] == "example/agent"


def test_managed_create_explains_how_to_grant_repo_access(monkeypatch: pytest.MonkeyPatch) -> None:
    authorize_creation(monkeypatch)

    def respond(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/projects/creation-readiness":
            return httpx.Response(200, json={"ready": True})
        if req.url.path == "/github/installations":
            return httpx.Response(200, json=[{"id": "inst", "repositories": []}])
        return httpx.Response(500, json={"message": f"unexpected {req.url.path}"})

    result = CliRunner().invoke(
        cli,
        [
            "dashboard",
            "project",
            "create",
            "--kind",
            "managed",
            "--name",
            "X",
            "--repo",
            "acme/agent",
            "--yes",
        ],
        obj=runtime(respond),
    )

    assert result.exit_code != 0
    assert "not available to this workspace" in result.output
    assert "owner may need to approve" in result.output
