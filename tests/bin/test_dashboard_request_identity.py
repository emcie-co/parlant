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

"""What every dashboard request says about the CLI run that sent it.

The platform shows each person's CLI activity as the commands they ran, so
each request names its command and its run. It never names the arguments:
those can hold ids, search text and other things the activity log must not
keep.
"""

from __future__ import annotations

import uuid
from importlib import metadata

import httpx
import pytest
from click.testing import CliRunner

from parlant.bin.cli import cli
from parlant.bin.dashboard.runtime import DashboardRuntime

from tests.bin.test_dashboard_commands import _no_auth, runtime


def _recording(requests: list[httpx.Request]) -> DashboardRuntime:
    def respond(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        if req.url.path.endswith("/runtime-logs/stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"event: done\ndata: bye\n\n",
            )
        return httpx.Response(200, json={"id": "s1", "sessions": [], "events": []})

    return runtime(respond)


def _run(dashboard: DashboardRuntime, *args: str) -> None:
    result = CliRunner().invoke(cli, ["dashboard", *args], obj=dashboard)
    assert result.exit_code == 0, result.output


def test_that_dashboard_requests_identify_the_cli_and_its_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_auth(monkeypatch, "sessions")
    requests: list[httpx.Request] = []

    _run(_recording(requests), "session", "view", "s1")

    assert requests
    assert all(
        req.headers["user-agent"] == f"parlant-cli/{metadata.version('parlant')}"
        for req in requests
    )


def test_that_dashboard_requests_name_the_running_command_without_its_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_auth(monkeypatch, "sessions")
    requests: list[httpx.Request] = []

    _run(
        _recording(requests),
        "session",
        "list",
        "--search",
        "refund for jane@example.com",
        "--limit",
        "5",
    )

    assert requests
    assert {req.headers["x-parlant-cli-command"] for req in requests} == {
        "parlant dashboard session list"
    }


def test_that_requests_from_one_cli_process_share_an_invocation_id_and_another_process_gets_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_auth(monkeypatch, "sessions")
    first_process: list[httpx.Request] = []
    second_process: list[httpx.Request] = []

    one_process = _recording(first_process)
    _run(one_process, "session", "view", "s1")
    _run(one_process, "session", "list")
    _run(_recording(second_process), "session", "view", "s1")

    first_ids = {req.headers["x-parlant-cli-invocation"] for req in first_process}
    second_ids = {req.headers["x-parlant-cli-invocation"] for req in second_process}
    assert len(first_process) >= 2
    assert len(first_ids) == 1
    assert len(second_ids) == 1
    assert first_ids != second_ids
    assert all(uuid.UUID(value).version == 4 for value in first_ids | second_ids)


def test_that_streamed_requests_carry_the_same_identifying_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_auth(monkeypatch, "logs")
    requests: list[httpx.Request] = []

    _run(_recording(requests), "logs")

    assert len(requests) == 1
    headers = requests[0].headers
    assert headers["user-agent"] == f"parlant-cli/{metadata.version('parlant')}"
    assert headers["x-parlant-cli-command"] == "parlant dashboard logs"
    assert uuid.UUID(headers["x-parlant-cli-invocation"]).version == 4
