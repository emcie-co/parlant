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

"""Setting values in a project's `.env` without disturbing the rest of it."""

from __future__ import annotations

import os
import re
from pathlib import Path

from parlant.bin.dashboard.errors import DashboardError

ENV_FILE = ".env"
ENV_EXAMPLE_FILE = ".env.example"

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PLAIN_VALUE = re.compile(r"[A-Za-z0-9_./:@+=,-]*")


def set_env_values(folder: Path, values: dict[str, str]) -> Path:
    """Set `values` in `folder/.env`, readable only by you.

    A missing `.env` starts from `.env.example`. Each name's existing line is
    replaced in place (later duplicates are dropped); new names are appended.
    Comments, blank lines, order and every other line are kept."""
    for name, value in values.items():
        if not _NAME.fullmatch(name):
            raise DashboardError(f"{name!r} is not a valid environment variable name.")
        if "\n" in value or "\r" in value:
            raise DashboardError(f"The value for {name} spans several lines; .env cannot hold it.")

    if not folder.is_dir():
        raise DashboardError(f"{folder} is not a folder.")

    path = folder / ENV_FILE
    example = folder / ENV_EXAMPLE_FILE
    if path.exists():
        text = path.read_text()
    elif example.exists():
        text = example.read_text()
    else:
        text = ""

    lines = text.splitlines()
    written: set[str] = set()
    kept: list[str] = []
    for line in lines:
        assigned = _assigned_name(line)
        if assigned is not None and assigned in values:
            if assigned not in written:
                kept.append(f"{assigned}={_quoted(values[assigned])}")
                written.add(assigned)
            continue
        kept.append(line)
    for name, value in values.items():
        if name not in written:
            kept.append(f"{name}={_quoted(value)}")

    _write_private(path, "\n".join(kept) + "\n")
    return path


def _assigned_name(line: str) -> str | None:
    match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
    return match.group(1) if match else None


def _quoted(value: str) -> str:
    if _PLAIN_VALUE.fullmatch(value):
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _write_private(path: Path, text: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(text)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
