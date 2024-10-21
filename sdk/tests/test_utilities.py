from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Iterator


SERVER_PORT = 8089
SERVER_ADDRESS = f"http://localhost:{SERVER_PORT}"

DEFAULT_AGENT_NAME = "Default Agent"

LOGGER = logging.getLogger(__name__)


def get_package_path() -> Path:
    p = Path(__file__)

    while not (p / ".git").exists():
        p = p.parent
        assert p != Path("/"), "Failed to find repo path"

    package_path = p / "sdk"

    assert Path.cwd().is_relative_to(package_path), "Must run from within the package dir"

    return package_path


CLI_CLIENT_PATH = get_package_path() / "src/emcie/sdk/bin/emcie.py"
CLI_SERVER_PATH = get_package_path() / "../server/src/emcie/server/bin/server.py"


@dataclass(frozen=True)
class ContextOfTest:
    home_dir: Path


@contextmanager
def run_server(
    context: ContextOfTest,
    extra_args: list[str] = [],
) -> Iterator[subprocess.Popen[str]]:
    exec_args = [
        "poetry",
        "run",
        "python",
        CLI_SERVER_PATH.as_posix(),
        "run",
        "-p",
        str(SERVER_PORT),
    ]

    exec_args.extend(extra_args)

    caught_exception: Exception | None = None

    try:
        with subprocess.Popen(
            args=exec_args,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stdout,
            env={**os.environ, "EMCIE_HOME": context.home_dir.as_posix()},
        ) as process:
            try:
                yield process
            except Exception as exc:
                caught_exception = exc

            if process.poll() is not None:
                return

            process.send_signal(signal.SIGINT)

            for i in range(5):
                if process.poll() is not None:
                    return
                time.sleep(0.5)

            process.terminate()

            for i in range(5):
                if process.poll() is not None:
                    return
                time.sleep(0.5)

            LOGGER.error(
                "Server process had to be killed. stderr="
                + (process.stderr and process.stderr.read() or "None")
            )

            process.kill()
            process.wait()

    finally:
        if caught_exception:
            raise caught_exception
