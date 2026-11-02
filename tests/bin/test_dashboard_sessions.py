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

"""Reading a project's sessions from the CLI."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from parlant.bin.cli import cli
from tests.bin.test_dashboard_commands import _no_auth, runtime

LONG_REPLY = (
    "The biggest difference comes down to what each is built for: LangChain is a "
    "general developer framework, while Parlant is purpose-built for customer-facing "
    "conversations and checks actions against your policies before executing them."
)


def _events(*messages: str) -> dict[str, Any]:
    return {
        "events": [
            {
                "offset": offset,
                "kind": "message",
                "source": "ai_agent",
                "createdAt": "2026-10-18T16:29:30Z",
                "data": {"message": message},
            }
            for offset, message in enumerate(messages)
        ]
    }


def _invoke(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]) -> str:
    _no_auth(monkeypatch, "sessions")

    def respond(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/projects/p1/sessions/s1/events":
            return httpx.Response(200, json=body)
        return httpx.Response(500, json={"message": req.url.path})

    result = CliRunner().invoke(
        cli,
        ["dashboard", "session", "events", "s1", "--kind", "message"],
        obj=runtime(respond),
        terminal_width=100,
    )
    assert result.exit_code == 0, result.output
    return result.output


def _shows(output: str, text: str) -> bool:
    """Whether the text appears in order, however the table wrapped it across rows."""
    words = [word for line in output.splitlines() for word in line.replace("│", " ").split()]
    wanted = text.split()
    return any(words[i : i + len(wanted)] == wanted for i in range(len(words)))


def test_that_session_events_prints_a_long_message_in_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _invoke(monkeypatch, _events(LONG_REPLY))

    assert "…" not in output
    assert _shows(output, LONG_REPLY), output


def test_that_session_events_keeps_every_line_of_a_multiline_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _invoke(
        monkeypatch,
        _events("We can definitely do that.\n\nCould you share your name and company?"),
    )

    assert _shows(output, "We can definitely do that."), output
    assert _shows(output, "Could you share your name and company?"), output
