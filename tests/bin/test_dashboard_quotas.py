"""Quota failures through the dashboard HTTP client and real Click commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
import pytest
from click.testing import CliRunner

from parlant.bin.cli import cli
from parlant.bin.dashboard.client import failure_for, request, require_ok, stream
from parlant.bin.dashboard.config import FileConfigStore
from parlant.bin.dashboard.credentials import Credential, CredentialStore
from parlant.bin.dashboard.errors import CredentialRejected, DashboardError
from parlant.bin.dashboard.runtime import DashboardRuntime


class MemoryCredentials(CredentialStore):
    def __init__(self) -> None:
        self.credential: Credential | None = Credential("refresh", "session")
        self.deletions = 0

    def get(self, host: str) -> Credential | None:
        return self.credential

    def save(self, host: str, credential: Credential) -> None:
        self.credential = credential

    def delete(self, host: str) -> None:
        self.deletions += 1
        self.credential = None


def runtime(tmp_path: Path, handler: Callable[[httpx.Request], httpx.Response]) -> DashboardRuntime:
    def unexpected_sleep(seconds: float) -> None:
        pytest.fail("Quota failure must not retry or sleep")

    return DashboardRuntime(
        host="https://dashboard.test",
        http=httpx.Client(
            base_url="https://dashboard.test", transport=httpx.MockTransport(handler)
        ),
        credentials=MemoryCredentials(),
        config=FileConfigStore(tmp_path / "config.json"),
        open_browser=lambda url: False,
        sleep=unexpected_sleep,
        project="p1",
    )


def quota_body(
    key: str = "cliTraceReads", reset: str | None = "2026-10-01T00:00:00Z"
) -> dict[str, Any]:
    return {
        "statusCode": 429,
        "error": "Too Many Requests",
        "code": "cli_quota_exceeded",
        "message": "untrusted \x1b]52;c;c2VjcmV0\a\nmessage\u202e",
        "quotaKey": key,
        "limit": 37,
        "remaining": 0,
        "resetAt": reset,
    }


def invoke_client(runtime: DashboardRuntime, streaming: bool) -> None:
    if streaming:
        list(stream(runtime, "/logs", access_token="access"))
    else:
        require_ok(request(runtime, "GET", "/trace", access_token="access"))


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "key,label",
    [
        ("cliTraceReads", "trace-read"),
        ("cliLogReads", "log-read"),
        ("cliInspections", "inspection"),
    ],
)
@pytest.mark.parametrize("reset", ["2026-10-01T00:00:00Z", None])
def test_that_quota_errors_use_safe_canonical_facts_on_both_transports(
    tmp_path: Path, streaming: bool, key: str, label: str, reset: str | None
) -> None:
    calls = []

    def respond(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(429, json=quota_body(key, reset), headers={"Retry-After": "60"})

    env = runtime(tmp_path, respond)
    with env.http, pytest.raises(DashboardError) as caught:
        invoke_client(env, streaming)
    error = caught.value
    assert error.exit_code == 7
    assert getattr(error, "code", None) == "CLI_QUOTA_EXCEEDED"
    assert getattr(error, "quota_key", None) == key
    assert getattr(error, "limit", None) == 37
    assert getattr(error, "reset_at", "absent") == reset
    assert str(error) == (
        f"Your workspace has reached its CLI {label} allowance. "
        f"Quota: {key}; limit: 37; remaining: 0; resets at: {reset or 'unknown'}."
    )
    assert not isinstance(error, CredentialRejected)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("statusCode", 400),
        ("error", "Bad Request"),
        ("quotaKey", "unknown"),
        ("limit", -1),
        ("limit", True),
        ("limit", 1.5),
        ("limit", "37"),
        ("limit", 9007199254740992),
        ("remaining", 1),
        ("remaining", False),
        ("resetAt", "tomorrow"),
        ("resetAt", "2026-10-01"),
        ("resetAt", 0),
        ("message", ""),
        ("message", "x" * 257),
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_that_malformed_quota_fields_fall_back_without_rendering_server_text(
    tmp_path: Path, field: str, value: Any, streaming: bool
) -> None:
    body = quota_body()
    body[field] = value
    env = runtime(tmp_path, lambda req: httpx.Response(429, json=body))
    with env.http, pytest.raises(DashboardError) as caught:
        invoke_client(env, streaming)
    assert caught.value.exit_code == 1
    assert "HTTP 429" in str(caught.value)
    assert "untrusted" not in str(caught.value)
    assert "\x1b" not in str(caught.value)


@pytest.mark.parametrize("field", list(quota_body()))
def test_that_missing_quota_fields_do_not_create_a_quota_error(field: str) -> None:
    body = quota_body()
    del body[field]
    error = failure_for(httpx.Response(429, json=body))
    assert error.exit_code == 1
    assert getattr(error, "code", None) != "CLI_QUOTA_EXCEEDED"


@pytest.mark.parametrize("body", [b"not JSON", b"[]", b"null", b'{"code":"cli_quota_exceeded"}'])
def test_that_invalid_quota_documents_remain_generic_failures(body: bytes) -> None:
    error = failure_for(httpx.Response(429, content=body))
    assert error.exit_code == 1
    assert "HTTP 429" in str(error)


class OversizedBody(httpx.SyncByteStream):
    def __init__(self) -> None:
        self.chunks_read = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for _ in range(1000):
            self.chunks_read += 1
            yield b"x" * 1024

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("declared_length", [None, "1000000", "1"])
def test_that_429_bodies_are_bounded_and_closed_even_with_wrong_content_length(
    tmp_path: Path, streaming: bool, declared_length: str | None
) -> None:
    body = OversizedBody()
    headers = {"Content-Length": declared_length} if declared_length else {}
    env = runtime(tmp_path, lambda req: httpx.Response(429, stream=body, headers=headers))
    with env.http, pytest.raises(DashboardError):
        invoke_client(env, streaming)
    assert body.chunks_read <= 65
    assert body.closed


COMMANDS = [
    (["trace", "view", "trace1"], "/projects/p1/inspect/trace1", "cliTraceReads"),
    (
        ["session", "inspect", "session1", "--turn", "turn1", "--include-logs"],
        "/projects/p1/sessions/session1/turns/turn1/inspection",
        "cliInspections",
    ),
    (
        ["deployment", "logs", "attempt1"],
        "/projects/p1/deployments/attempts/attempt1/logs",
        "cliLogReads",
    ),
    (["logs"], "/projects/p1/runtime-logs/stream", "cliLogReads"),
]

STREAMED = [command for command in COMMANDS if command[1].endswith("/stream")]


@pytest.mark.parametrize("command,path,key", COMMANDS)
def test_that_real_commands_keep_credentials_and_do_not_retry_quota_failures(
    tmp_path: Path, command: list[str], path: str, key: str
) -> None:
    calls = []

    def respond(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if req.url.path == "/v1/cli/auth/token":
            return httpx.Response(
                200,
                json={
                    "access_token": "access",
                    "refresh_token": "rotated",
                    "session_id": "session",
                },
            )
        assert req.url.path == path
        assert req.headers["authorization"] == "Bearer access"
        return httpx.Response(429, json=quota_body(key))

    env = runtime(tmp_path, respond)
    with env.http:
        result = CliRunner().invoke(cli, ["dashboard", *command], obj=env)
    assert result.exit_code == 7, result.output
    assert f"Quota: {key}; limit: 37" in result.stderr
    assert result.stdout == ""
    assert "untrusted" not in result.output
    assert calls == ["/v1/cli/auth/token", path]
    assert env.credentials.get(env.host) == Credential("rotated", "session")
    assert isinstance(env.credentials, MemoryCredentials)
    assert env.credentials.deletions == 0


@pytest.mark.parametrize("command,path,key", STREAMED)
def test_that_log_commands_fail_canonically_at_the_stream_limit_without_reconnecting(
    tmp_path: Path, command: list[str], path: str, key: str
) -> None:
    calls = []

    def respond(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if req.url.path == "/v1/cli/auth/token":
            return httpx.Response(
                200,
                json={
                    "access_token": "access",
                    "refresh_token": "rotated",
                    "session_id": "session",
                },
            )
        assert req.url.path == path
        body = json.dumps({"code": "cli_stream_limit_reached", "message": "untrusted \x1b[31m"})
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=f"data: before limit\n\nevent: error\ndata: {body}\n\ndata: after limit\n\nevent: done\ndata: done\n\n",
        )

    env = runtime(tmp_path, respond)
    with env.http:
        result = CliRunner().invoke(cli, ["dashboard", *command], obj=env)
    assert result.exit_code == 7, result.output
    assert (
        result.stderr
        == "Error: CLI log-stream time limit reached. Start a new log request to continue.\n"
    )
    assert "before limit" in result.stdout
    assert "after limit" not in result.stdout
    assert "untrusted" not in result.output
    assert calls == ["/v1/cli/auth/token", path]
    assert env.credentials.get(env.host) == Credential("rotated", "session")


@pytest.mark.parametrize("status", [400, 404, 429, 503])
def test_that_generic_http_errors_keep_existing_messages(status: int) -> None:
    error = failure_for(httpx.Response(status, json={"message": "Existing error"}))
    assert str(error) == "Existing error"
    assert error.exit_code == 1


def test_that_an_unauthorized_response_means_the_sign_in_was_rejected() -> None:
    assert isinstance(failure_for(httpx.Response(401)), CredentialRejected)


PLAN_LIMIT_MESSAGE = (
    "Your plan's Local project limit has been reached. "
    "Upgrade your plan to create more Local projects."
)
PLAN_LIMIT_BODY = {
    "statusCode": 403,
    "code": "localProjects_limit_reached",
    "error": "Forbidden",
    "message": PLAN_LIMIT_MESSAGE,
}


def test_that_a_forbidden_response_shows_the_dashboards_own_message() -> None:
    error = failure_for(httpx.Response(403, json=PLAN_LIMIT_BODY))
    assert not isinstance(error, CredentialRejected)
    assert str(error) == PLAN_LIMIT_MESSAGE
    assert error.code == "localProjects_limit_reached"
    assert error.exit_code == 1


def test_that_a_forbidden_permission_denial_names_the_missing_permission() -> None:
    error = failure_for(
        httpx.Response(
            403,
            json={
                "statusCode": 403,
                "error": "Forbidden",
                "message": "authorization_permission_denied",
                "requiredPermission": "projects.create",
                "activeTenantId": "t1",
            },
        )
    )
    assert not isinstance(error, CredentialRejected)
    assert str(error) == (
        "Your role in this workspace doesn't include `projects.create`. "
        "Ask a workspace Owner or Admin to grant it; signing in again won't help."
    )


def test_that_a_bare_forbidden_response_says_signing_in_again_will_not_help() -> None:
    error = failure_for(httpx.Response(403))
    assert not isinstance(error, CredentialRejected)
    assert str(error) == (
        "The Parlant dashboard refused that request (HTTP 403). Your sign-in is still "
        "valid, so signing in again won't help: check your role in this workspace "
        "and your plan's limits."
    )


def test_that_a_forbidden_response_never_clears_the_stored_sign_in(tmp_path: Path) -> None:
    command, path, _ = COMMANDS[0]

    def respond(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/cli/auth/token":
            return httpx.Response(
                200,
                json={
                    "access_token": "access",
                    "refresh_token": "rotated",
                    "session_id": "session",
                },
            )
        assert req.url.path == path
        return httpx.Response(403, json=PLAN_LIMIT_BODY)

    env = runtime(tmp_path, respond)
    with env.http:
        result = CliRunner().invoke(cli, ["dashboard", *command], obj=env)
    assert result.exit_code == 1, result.output
    assert PLAN_LIMIT_MESSAGE in result.stderr
    assert "sign-in has expired" not in result.output
    assert env.credentials.get(env.host) == Credential("rotated", "session")
    assert isinstance(env.credentials, MemoryCredentials)
    assert env.credentials.deletions == 0


def test_that_generic_stream_errors_keep_existing_behavior(tmp_path: Path) -> None:
    env = runtime(
        tmp_path,
        lambda req: httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text='event: error\ndata: {"message":"Existing stream error"}\n\n',
        ),
    )
    with env.http, pytest.raises(DashboardError, match="Existing stream error") as caught:
        invoke_client(env, True)
    assert caught.value.exit_code == 1


def checked_setup() -> dict[str, Any]:
    """A managed setup snapshot, as Backend returns it, whose automatic check is done."""
    return {
        "projectId": "p1",
        "projectName": "Agent",
        "revision": 2,
        "repository": {"fullName": "example/agent", "branch": "main"},
        "check": {
            "status": "done",
            "checkId": "check-1",
            "commit": {"sha": "a" * 40, "message": "Initial agent", "branch": "main"},
            "verdict": "standard",
            "summary": "Looks like a standard Parlant project.",
            "reviewFields": [],
            "finishedAt": "2026-10-01T10:00:30.000Z",
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


def test_that_an_exhausted_inspection_allowance_does_not_stop_managed_project_creation(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []

    def respond(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path))
        if req.url.path == "/v1/cli/auth/token":
            return httpx.Response(
                200,
                json={
                    "access_token": "access",
                    "refresh_token": "rotated",
                    "session_id": "session",
                },
            )
        if req.url.path == "/v1/cli/auth/session":
            return httpx.Response(
                200,
                json={
                    "user_id": "user",
                    "tenant_id": "workspace",
                    "session_id": "session",
                    "scope": "projects.create projects.update",
                },
            )
        if req.url.path == "/projects/creation-readiness":
            return httpx.Response(200, json={"ready": True})
        if req.url.path == "/github/installations":
            return httpx.Response(
                200, json=[{"id": "installation", "repositories": [{"fullName": "example/agent"}]}]
            )
        if req.url.path.endswith("/branches"):
            return httpx.Response(200, json=[{"name": "main"}])
        if (req.method, req.url.path) == ("POST", "/projects"):
            created = json.loads(req.content)
            return httpx.Response(201, json={"id": created["id"], "name": created["name"]})
        if req.url.path.endswith("/managed-setup/checks"):
            return httpx.Response(429, json=quota_body("cliInspections"))
        if req.method == "GET" and req.url.path.endswith("/managed-setup"):
            return httpx.Response(200, json=checked_setup())
        if req.method == "PATCH" and req.url.path.endswith("/managed-setup/settings"):
            setup = checked_setup()
            setup["settings"].update(json.loads(req.content))
            return httpx.Response(200, json=setup)
        raise AssertionError(f"Unexpected {req.method} {req.url.path}")

    env = runtime(tmp_path, respond)
    with env.http:
        result = CliRunner().invoke(
            cli,
            [
                "dashboard",
                "project",
                "create",
                "--kind",
                "managed",
                "--name",
                "Agent",
                "--repo",
                "example/agent",
                "--branch",
                "main",
                "--run-command",
                "python agent.py",
                "--yes",
            ],
            obj=env,
        )
    assert result.exit_code == 0, result.output
    assert calls.count(("POST", "/projects")) == 1
    assert not any(path.endswith("/managed-setup/checks") for _, path in calls)
    assert "Classification: standard" in result.output
    assert env.credentials.get(env.host) == Credential("rotated", "session")
    assert isinstance(env.credentials, MemoryCredentials)
    assert env.credentials.deletions == 0


@pytest.mark.parametrize("streaming", [False, True])
def test_that_compressed_quota_responses_are_not_decoded_twice(
    tmp_path: Path, streaming: bool
) -> None:
    import gzip

    body = gzip.compress(json.dumps(quota_body()).encode())
    env = runtime(
        tmp_path,
        lambda req: httpx.Response(
            429, headers={"Content-Encoding": "gzip"}, stream=httpx.ByteStream(body)
        ),
    )
    with env.http, pytest.raises(DashboardError) as caught:
        invoke_client(env, streaming)
    assert caught.value.exit_code == 7
    assert getattr(caught.value, "quota_key", None) == "cliTraceReads"


@pytest.mark.parametrize("value", [None, "", 123, "x" * 257])
def test_that_malformed_stream_limit_messages_are_never_displayed(
    tmp_path: Path, value: Any
) -> None:
    body = json.dumps({"code": "cli_stream_limit_reached", "message": value})
    env = runtime(
        tmp_path,
        lambda req: httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=f"event: error\ndata: {body}\n\n",
        ),
    )
    with env.http, pytest.raises(DashboardError) as caught:
        invoke_client(env, True)
    assert caught.value.exit_code == 1
    assert str(caught.value) == "The stream reported an invalid CLI log-stream limit response."


@pytest.mark.parametrize("streaming", [False, True])
def test_that_deeply_nested_quota_documents_remain_generic_cli_errors(
    tmp_path: Path, streaming: bool
) -> None:
    body = b'{"code":"cli_quota_exceeded","padding":' + b"[" * 15000 + b"0" + b"]" * 15000 + b"}"
    assert len(body) < 64 * 1024
    # Establish that this fixture actually exceeds the cloud interpreter's
    # parser limit, instead of only failing quota-field validation afterward.
    with pytest.raises(RecursionError):
        json.loads(body)
    env = runtime(tmp_path, lambda req: httpx.Response(429, content=body))
    with env.http, pytest.raises(DashboardError) as caught:
        invoke_client(env, streaming)
    assert caught.value.exit_code == 1
    assert "HTTP 429" in str(caught.value)


def test_that_deeply_nested_stream_errors_remain_safe_cli_errors(tmp_path: Path) -> None:
    body = (
        '{"code":"cli_stream_limit_reached","message":"untrusted","padding":'
        + "[" * 15000
        + "0"
        + "]" * 15000
        + "}"
    )
    assert len(body) < 64 * 1024
    with pytest.raises(RecursionError):
        json.loads(body)
    env = runtime(
        tmp_path,
        lambda req: httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=f"event: error\ndata: {body}\n\n",
        ),
    )
    with env.http, pytest.raises(DashboardError) as caught:
        invoke_client(env, True)
    assert caught.value.exit_code == 1
    assert str(caught.value) == "The stream reported an invalid error response."
