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

"""Safe terminal output for remote text, and the machine-readable `--json` mode."""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any, Callable, TypeVar

import click

F = TypeVar("F", bound=Callable[..., Any])

# Set by `--json` for the rest of the invocation, so errors raised anywhere
# below a command render as the envelope instead of as text.
_json_mode: ContextVar[bool] = ContextVar("parlant_dashboard_json_mode", default=False)

_ESCAPED = (
    set(range(0x00, 0x09))
    | set(range(0x0A, 0x20))
    | {0x7F}
    | set(range(0x80, 0xA0))
    | {0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069}
)


def escape_terminal_text(text: str) -> str:
    """Escape terminal control characters while keeping their values visible."""
    return "".join(
        character if ord(character) not in _ESCAPED else _escape(character) for character in text
    )


def _escape(character: str) -> str:
    return f"\\u{ord(character):04x}"


def write_stream_line(text: str) -> None:
    """Print a stream line with terminal control characters escaped."""
    click.echo(escape_terminal_text(text))


def json_mode() -> bool:
    """Whether this invocation prints JSON."""
    return _json_mode.get()


def reset_json_mode() -> None:
    """Start an invocation in text mode; `--json` turns JSON on for it."""
    _json_mode.set(False)


def _enable_json(ctx: click.Context, param: click.Parameter, value: bool) -> bool:
    if value:
        _json_mode.set(True)
    return value


def json_option(command: F) -> F:
    """`--json`: stdout carries only JSON, and failures print an error envelope."""
    return click.option(
        "--json",
        "as_json",
        is_flag=True,
        is_eager=True,
        expose_value=False,
        callback=_enable_json,
        help="Print JSON on stdout, and nothing else.",
    )(command)


def emit_json(document: Any) -> None:
    """Print one JSON document as a single stdout line."""
    click.echo(json.dumps(document, separators=(",", ":")))


def say(text: str = "") -> None:
    """Human progress text: stdout normally, stderr in JSON mode."""
    click.echo(text, err=json_mode())
