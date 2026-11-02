"""Prior CLI connections are hints, sent over real HTTP without refresh secrets."""

from __future__ import annotations

import json
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import httpx
import pytest

from parlant.bin.dashboard.auth import start_device_authorization
from parlant.bin.dashboard.config import FileConfigStore
from parlant.bin.dashboard.credentials import Credential, CredentialStore, credential_key
from parlant.bin.dashboard.errors import DashboardError
from parlant.bin.dashboard.runtime import DashboardRuntime

FIRST_SESSION = "000000000000000000000001"
SECOND_SESSION = "000000000000000000000002"


class PriorCredentials(CredentialStore):
    def __init__(self) -> None:
        self.values: dict[str, Credential] = {}
        self.reads: list[str] = []
        self.unavailable = False

    def get(self, host: str) -> Credential | None:
        self.reads.append(host)
        if self.unavailable:
            raise DashboardError("private-keyring-details")
        return self.values.get(host)

    def save(self, host: str, credential: Credential) -> None:
        self.values[host] = credential

    def delete(self, host: str) -> None:
        self.values.pop(host, None)


@pytest.fixture
def device_server() -> Iterator[tuple[str, list[dict[str, object]]]]:
    bodies: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            assert self.path == "/v1/cli/auth/device"
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            origin = f"http://127.0.0.1:{server.server_port}"
            data = json.dumps(
                {
                    "device_code": "device-secret",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": f"{origin}/cli/authorize",
                    "verification_uri_complete": f"{origin}/cli/authorize?user_code=ABCD-EFGH",
                    "expires_in": 600,
                    "interval": 5,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", bodies
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def runtime(tmp_path: Path, host: str, credentials: PriorCredentials) -> DashboardRuntime:
    return DashboardRuntime(
        host=host,
        credentials=credentials,
        config=FileConfigStore(tmp_path / "prior-workspaces.json"),
        http=httpx.Client(base_url=host),
        open_browser=lambda _: False,
        sleep=lambda _: None,
    )


def test_that_device_authorization_sends_saved_session_references(
    tmp_path: Path,
    device_server: tuple[str, list[dict[str, object]]],
) -> None:
    host, bodies = device_server
    credentials = PriorCredentials()
    env = runtime(tmp_path, host, credentials)
    try:
        for workspace, session in [("workspace-1", FIRST_SESSION), ("workspace-2", SECOND_SESSION)]:
            env.config.add_signed_in_workspace(host, workspace)
            credentials.save(
                credential_key(host, workspace), Credential(f"refresh-{workspace}", session)
            )
        env.config.set_active_workspace(host, "workspace-2")
        assert start_device_authorization(env).user_code == "ABCD-EFGH"
        assert bodies[0]["existing_session_ids"] == [SECOND_SESSION, FIRST_SESSION]
    finally:
        env.http.close()


def test_that_device_authorization_never_sends_refresh_tokens_or_other_host_sign_ins(
    tmp_path: Path,
    device_server: tuple[str, list[dict[str, object]]],
) -> None:
    host, bodies = device_server
    credentials = PriorCredentials()
    env = runtime(tmp_path, host, credentials)
    try:
        env.config.add_signed_in_workspace(host, "workspace-1")
        env.config.add_signed_in_workspace("https://other.example", "workspace-other")
        credentials.save(
            credential_key(host, "workspace-1"), Credential("private-local-refresh", FIRST_SESSION)
        )
        credentials.save(
            credential_key("https://other.example", "workspace-other"),
            Credential("private-foreign-refresh", SECOND_SESSION),
        )
        start_device_authorization(env)
        assert bodies[0]["existing_session_ids"] == [FIRST_SESSION]
        assert SECOND_SESSION not in json.dumps(bodies)
        assert "private-local-refresh" not in json.dumps(bodies)
        assert "private-foreign-refresh" not in json.dumps(bodies)
        assert not any("other.example" in read for read in credentials.reads)
    finally:
        env.http.close()


def test_that_device_authorization_continues_when_prior_credentials_are_unavailable(
    tmp_path: Path,
    device_server: tuple[str, list[dict[str, object]]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    host, bodies = device_server
    credentials = PriorCredentials()
    credentials.unavailable = True
    env = runtime(tmp_path, host, credentials)
    try:
        env.config.add_signed_in_workspace(host, "workspace-1")
        assert start_device_authorization(env).user_code == "ABCD-EFGH"
        assert len(bodies) == 1 and "existing_session_ids" not in bodies[0]
        assert credentials.reads  # The prior-connection path was attempted.
        captured = capsys.readouterr()
        assert "private-keyring-details" not in captured.out + captured.err
    finally:
        env.http.close()
