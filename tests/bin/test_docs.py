"""Documentation queries through the real Click CLI and a local HTTP server."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

from parlant.bin.cli import cli
from parlant.bin.dashboard.config import FileConfigStore
from parlant.bin.dashboard.credentials import Credential, CredentialStore, credential_key
from parlant.bin.dashboard.runtime import DashboardRuntime


class DocsServer:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, object]]] = []
        self.connects: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.status = 200
        self.delay_seconds = 0.0
        self.body: object = {
            "request_id": "request-1",
            "status": "answered",
            "answer": "Use agent.create_policy with a condition and action.",
            "citations": [
                {
                    "document_id": "policies",
                    "title": "Policies",
                    "url": "https://parlant.io/docs/customizing-behavior/policies",
                }
            ],
            "warnings": [],
            "snapshot_revision": "snapshot-1",
            "source_commit": "commit-1",
            "model": "gemini-4-argon",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cached_tokens": 0,
                "thinking_tokens": 0,
            },
            "duration_ms": 12,
        }
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version: str = "HTTP/1.1"

            def do_CONNECT(self) -> None:
                # A real local HTTPS proxy observes the selected hostname, then
                # rejects the tunnel. The test never connects to production.
                owner.connects.append(self.path)
                self.send_response(502)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self) -> None:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.path, payload))
                owner.headers.append({key.lower(): value for key, value in self.headers.items()})
                if owner.delay_seconds:
                    time.sleep(owner.delay_seconds)
                body = owner.body
                if self.path == "/v1/cli/auth/token" and owner.status == 200:
                    body = {
                        "access_token": f"access-for-{payload['refresh_token']}",
                        "refresh_token": f"rotated-{payload['refresh_token']}",
                        "session_id": "docs-test-session",
                    }
                response = json.dumps(body).encode()
                self.send_response(owner.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                try:
                    self.wfile.write(response)
                except BrokenPipeError:
                    pass

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.http.server_port}"


@pytest.fixture
def docs_server() -> Iterator[DocsServer]:
    server = DocsServer()
    server.thread.start()
    try:
        yield server
    finally:
        server.http.shutdown()
        server.http.server_close()
        server.thread.join(timeout=3)


def test_that_docs_query_sends_question_context_and_installed_version(
    docs_server: DocsServer,
) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "docs",
            "query",
            "How do I add a policy?",
            "--context",
            "Existing retail agent",
            "--endpoint",
            docs_server.url,
        ],
    )
    assert result.exit_code == 0, result.output
    assert docs_server.requests == [
        (
            "/v1/query",
            {
                "query": "How do I add a policy?",
                "context": "Existing retail agent",
                "sdk_version": version("parlant"),
                "client": f"parlant-cli/{version('parlant')}",
            },
        )
    ]
    assert "agent.create_policy" in result.output
    assert "https://parlant.io/docs/customizing-behavior/policies" in result.output


def test_that_docs_query_returns_json_without_presentation_text(docs_server: DocsServer) -> None:
    result = CliRunner().invoke(
        cli, ["docs", "query", "How?", "--endpoint", docs_server.url, "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == docs_server.body


def test_that_explicit_endpoint_overrides_environment(docs_server: DocsServer) -> None:
    result = CliRunner().invoke(
        cli,
        ["docs", "query", "How?", "--endpoint", docs_server.url],
        env={"PARLANT_DOCS_URL": "http://127.0.0.1:1"},
    )
    assert result.exit_code == 0, result.output
    assert len(docs_server.requests) == 1


def test_that_environment_can_select_the_local_docs_service(docs_server: DocsServer) -> None:
    result = CliRunner().invoke(
        cli, ["docs", "query", "How?"], env={"PARLANT_DOCS_URL": docs_server.url}
    )
    assert result.exit_code == 0, result.output
    assert len(docs_server.requests) == 1


def test_that_docs_query_uses_production_by_default(
    docs_server: DocsServer, docs_runtime: DashboardRuntime
) -> None:
    result = CliRunner().invoke(
        cli,
        ["docs", "query", "How?", "--timeout", "1"],
        obj=docs_runtime,
        env={
            "PARLANT_DOCS_URL": "",
            "HTTPS_PROXY": docs_server.url,
            "ALL_PROXY": "",
            "NO_PROXY": "",
        },
    )
    assert docs_server.connects == ["lighthouse.parlant.io:443"]
    # The controlled proxy refuses the tunnel, rather than invoking the model.
    assert result.exit_code != 0
    assert not docs_server.requests


def test_that_docs_help_shows_the_production_default() -> None:
    result = CliRunner().invoke(cli, ["docs", "query", "--help"])
    assert result.exit_code == 0, result.output
    assert "https://lighthouse.parlant.io" in result.output


@pytest.mark.parametrize("status", [400, 401, 429, 500, 502, 503, 504])
def test_that_docs_http_errors_fail_without_guessing_an_answer(
    docs_server: DocsServer, status: int
) -> None:
    docs_server.status = status
    docs_server.body = {"detail": {"code": "service_busy", "private": "must-not-print"}}
    result = CliRunner().invoke(cli, ["docs", "query", "How?", "--endpoint", docs_server.url])
    assert result.exit_code != 0
    assert str(status) in result.output
    assert "must-not-print" not in result.output
    assert len(docs_server.requests) == 1


@pytest.mark.parametrize("body", [None, [], {"answer": "incomplete"}, {"status": "answered"}])
def test_that_malformed_docs_responses_are_not_reported_as_success(
    docs_server: DocsServer, body: object
) -> None:
    docs_server.body = body
    result = CliRunner().invoke(cli, ["docs", "query", "How?", "--endpoint", docs_server.url])
    assert result.exit_code != 0
    assert "response" in result.output.lower()


def test_that_unavailable_docs_endpoint_fails_clearly() -> None:
    result = CliRunner().invoke(
        cli, ["docs", "query", "How?", "--endpoint", "http://127.0.0.1:1", "--timeout", "0.1"]
    )
    assert result.exit_code != 0
    assert "connect" in result.output.lower() or "unavailable" in result.output.lower()


def test_that_timeout_option_rejects_nonpositive_values(docs_server: DocsServer) -> None:
    result = CliRunner().invoke(
        cli, ["docs", "query", "How?", "--endpoint", docs_server.url, "--timeout", "0"]
    )
    assert result.exit_code != 0
    assert "timeout" in result.output.lower()
    assert not docs_server.requests


def test_that_docs_request_timeout_exits_without_retry(docs_server: DocsServer) -> None:
    docs_server.delay_seconds = 0.2
    result = CliRunner().invoke(
        cli, ["docs", "query", "How?", "--endpoint", docs_server.url, "--timeout", "0.05"]
    )
    assert result.exit_code != 0
    assert "timed out" in result.output.lower()
    assert len(docs_server.requests) == 1


class DocsCredentials(CredentialStore):
    """Real state behind the credential interface; never touch the OS keychain."""

    def __init__(self) -> None:
        self.values: dict[str, Credential] = {}
        self.reads: list[str] = []

    def get(self, host: str) -> Credential | None:
        self.reads.append(host)
        return self.values.get(host)

    def save(self, host: str, credential: Credential) -> None:
        self.values[host] = credential

    def delete(self, host: str) -> None:
        self.values.pop(host, None)


@pytest.fixture
def dashboard_server() -> Iterator[DocsServer]:
    server = DocsServer()
    server.thread.start()
    try:
        yield server
    finally:
        server.http.shutdown()
        server.http.server_close()
        server.thread.join(timeout=3)


@pytest.fixture
def docs_runtime(tmp_path: Path, dashboard_server: DocsServer) -> Iterator[DashboardRuntime]:
    with httpx.Client(base_url=dashboard_server.url) as http:
        yield DashboardRuntime(
            host=dashboard_server.url,
            http=http,
            config=FileConfigStore(tmp_path / "docs-dashboard.json"),
            credentials=DocsCredentials(),
            open_browser=lambda url: False,
            sleep=lambda seconds: None,
        )


def save_workspace(runtime: DashboardRuntime, workspace: str) -> None:
    runtime.config.add_signed_in_workspace(runtime.host, workspace)
    runtime.credentials.save(
        credential_key(runtime.host, workspace), Credential(f"refresh-{workspace}", "session")
    )


def test_that_signed_in_docs_query_sends_only_the_selected_workspace_access_token(
    docs_server: DocsServer,
    dashboard_server: DocsServer,
    docs_runtime: DashboardRuntime,
) -> None:
    save_workspace(docs_runtime, "workspace-1")
    save_workspace(docs_runtime, "workspace-2")
    docs_runtime.config.set_active_workspace(docs_runtime.host, "workspace-2")
    result = CliRunner().invoke(
        cli, ["docs", "query", "How?", "--endpoint", docs_server.url], obj=docs_runtime
    )
    assert result.exit_code == 0, result.output
    assert (
        docs_server.headers[0].get("x-parlant-dashboard-authorization")
        == "Bearer access-for-refresh-workspace-2"
    )
    assert len(dashboard_server.requests) == 1
    assert dashboard_server.requests[0][1]["refresh_token"] == "refresh-workspace-2"
    assert "refresh-workspace-1" not in str(docs_server.requests)
    assert "refresh_token" not in str(docs_server.requests)
    assert "access-for-" not in result.output


def test_that_explicit_workspace_selects_its_own_docs_attribution(
    docs_server: DocsServer,
    dashboard_server: DocsServer,
    docs_runtime: DashboardRuntime,
) -> None:
    save_workspace(docs_runtime, "workspace-1")
    save_workspace(docs_runtime, "workspace-2")
    docs_runtime.config.set_active_workspace(docs_runtime.host, "workspace-1")
    result = CliRunner().invoke(
        cli,
        ["docs", "query", "How?", "--workspace", "workspace-2", "--endpoint", docs_server.url],
        obj=docs_runtime,
    )
    assert result.exit_code == 0, result.output
    assert (
        docs_server.headers[0].get("x-parlant-dashboard-authorization")
        == "Bearer access-for-refresh-workspace-2"
    )
    assert dashboard_server.requests[0][1]["refresh_token"] == "refresh-workspace-2"
    assert docs_runtime.config.active_workspace(docs_runtime.host) == "workspace-1"


def test_that_anonymous_docs_query_needs_no_dashboard_or_keychain_access(
    docs_server: DocsServer,
    dashboard_server: DocsServer,
    docs_runtime: DashboardRuntime,
) -> None:
    result = CliRunner().invoke(
        cli, ["docs", "query", "How?", "--endpoint", docs_server.url], obj=docs_runtime
    )
    assert result.exit_code == 0, result.output
    assert docs_server.headers[0].get("x-parlant-dashboard-authorization") is None
    assert not dashboard_server.requests
    assert isinstance(docs_runtime.credentials, DocsCredentials)
    assert not docs_runtime.credentials.reads


def test_that_ambiguous_workspace_does_not_guess_attribution(
    docs_server: DocsServer,
    dashboard_server: DocsServer,
    docs_runtime: DashboardRuntime,
) -> None:
    save_workspace(docs_runtime, "workspace-1")
    save_workspace(docs_runtime, "workspace-2")
    result = CliRunner().invoke(
        cli, ["docs", "query", "How?", "--endpoint", docs_server.url], obj=docs_runtime
    )
    assert result.exit_code == 0, result.output
    assert not dashboard_server.requests
    assert docs_server.headers[0].get("x-parlant-dashboard-authorization") is None
    assert "workspace" in result.stderr.lower() and "attribution" in result.stderr.lower()


def test_that_optional_auth_failure_keeps_docs_available_without_leaking_errors(
    docs_server: DocsServer,
    dashboard_server: DocsServer,
    docs_runtime: DashboardRuntime,
) -> None:
    save_workspace(docs_runtime, "workspace-1")
    dashboard_server.status = 503
    dashboard_server.body = {"private": "must-not-print"}
    result = CliRunner().invoke(
        cli, ["docs", "query", "How?", "--endpoint", docs_server.url], obj=docs_runtime
    )
    assert result.exit_code == 0, result.output
    assert len(dashboard_server.requests) == 1
    assert docs_server.headers[0].get("x-parlant-dashboard-authorization") is None
    assert "attribution" in result.stderr.lower()
    assert "must-not-print" not in result.output
    assert "refresh-workspace-1" not in result.output


@pytest.mark.parametrize(
    "dashboard_host",
    ["https://api.parlant.io", "https://api.emcie.xyz", "https://untrusted.example"],
)
def test_that_custom_docs_endpoints_never_receive_dashboard_credentials(
    docs_server: DocsServer,
    dashboard_server: DocsServer,
    docs_runtime: DashboardRuntime,
    dashboard_host: str,
) -> None:
    # HTTP still targets our local server if a bug tries to refresh: no network
    # connection or real credential can escape this test.
    docs_runtime.host = dashboard_host
    save_workspace(docs_runtime, "workspace-1")
    result = CliRunner().invoke(
        cli, ["docs", "query", "How?", "--endpoint", docs_server.url], obj=docs_runtime
    )
    assert result.exit_code == 0, result.output
    assert not dashboard_server.requests
    assert docs_server.headers[0].get("x-parlant-dashboard-authorization") is None
    assert isinstance(docs_runtime.credentials, DocsCredentials)
    assert not docs_runtime.credentials.reads


@pytest.mark.parametrize(
    "dashboard,endpoint,allowed",
    [
        ("https://api.parlant.io", "https://lighthouse.parlant.io", True),
        ("https://api.emcie.xyz", "https://lighthouse.emcie.xyz", True),
        ("https://api.parlant.io", "https://lighthouse.emcie.xyz", False),
        ("https://api.emcie.xyz", "https://lighthouse.parlant.io", False),
        ("https://api.parlant.io", "https://lighthouse.parlant.io.evil.example", False),
        ("https://api.parlant.io", "http://lighthouse.parlant.io", False),
        ("https://api.parlant.io", "https://lighthouse.parlant.io:8443", False),
        ("https://api.parlant.io", "https://lighthouse.parlant.io/custom", False),
    ],
)
def test_that_docs_credentials_require_matching_trusted_origins(
    dashboard: str, endpoint: str, allowed: bool
) -> None:
    from parlant.bin.docs_attribution import credentials_may_be_sent

    assert credentials_may_be_sent(dashboard, endpoint) is allowed
