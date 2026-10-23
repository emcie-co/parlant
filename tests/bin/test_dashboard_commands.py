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
