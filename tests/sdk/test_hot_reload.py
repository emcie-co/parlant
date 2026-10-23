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

"""Hot reload: the server watches the entry script's tree and re-execs itself.

The end-to-end tests run a real agent script in a subprocess and observe
restarts through /healthz's per-boot instance id — no mocks, no fakes: the
observable is a genuinely rebooted server on the same port.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx

from parlant.sdk import _hot_reload_watch_root

from tests.test_utilities import get_random_port

_SCRIPT_TEMPLATE = """\
import asyncio

import parlant.sdk as p


async def main() -> None:
    async with p.Server(
        port={port},
        tool_service_port={tool_port},
        models={{"*": "gemini"}},
        log_level=p.LogLevel.WARNING,
        hot_reload={hot_reload},
        session_store="transient",
        customer_store="transient",
    ):
        pass


asyncio.run(main())
"""


def _write_script(directory: Path, port: int, tool_port: int, hot_reload: bool) -> Path:
    script = directory / "dev_agent.py"
    script.write_text(
        _SCRIPT_TEMPLATE.format(port=port, tool_port=tool_port, hot_reload=hot_reload)
    )
    return script


def _wait_for_instance(
    port: int, deadline_seconds: float, *, not_equal_to: str | None = None
) -> str:
    deadline = time.monotonic() + deadline_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://localhost:{port}/healthz", timeout=5.0)
            if response.status_code == 200:
                instance = str(response.json()["instance_id"])
                if not_equal_to is None or instance != not_equal_to:
                    return instance
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.25)

    raise TimeoutError(
        f"No {'new ' if not_equal_to else ''}server instance on port {port} "
        f"within {deadline_seconds}s (last error: {last_error})"
    )


def _touch(script: Path) -> None:
    script.write_text(script.read_text() + "\n# touched\n")


def _launch(script: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, str(script)],
        cwd=script.parent,
        env=dict(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _terminate(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def test_that_saving_the_entry_script_restarts_the_server_on_the_same_port(
    tmp_path: Path,
) -> None:
    port, tool_port = get_random_port(), get_random_port()
    script = _write_script(tmp_path, port, tool_port, hot_reload=True)
    process = _launch(script)
    try:
        first_instance = _wait_for_instance(port, 120)

        _touch(script)

        second_instance = _wait_for_instance(port, 120, not_equal_to=first_instance)
        assert second_instance != first_instance
        # os.execv replaces the process image in place: same PID, still running.
        assert process.poll() is None
    finally:
        _terminate(process)


def test_that_hot_reload_disabled_ignores_script_changes(tmp_path: Path) -> None:
    port, tool_port = get_random_port(), get_random_port()
    script = _write_script(tmp_path, port, tool_port, hot_reload=False)
    process = _launch(script)
    try:
        first_instance = _wait_for_instance(port, 120)

        _touch(script)
        time.sleep(8)

        current_instance = _wait_for_instance(port, 30)
        assert current_instance == first_instance
        assert process.poll() is None
    finally:
        _terminate(process)


# --- Watch-root resolution (pure) -------------------------------------------------


def test_that_the_watch_root_is_the_entry_scripts_directory() -> None:
    module = SimpleNamespace(__file__="/somewhere/project/dev_agent.py")

    assert _hot_reload_watch_root(module) == Path("/somewhere/project")


def test_that_hot_reload_disarms_without_a_main_script() -> None:
    # REPLs, notebooks, and embedded interpreters have no __main__ file —
    # there is nothing meaningful to watch.
    assert _hot_reload_watch_root(SimpleNamespace()) is None
