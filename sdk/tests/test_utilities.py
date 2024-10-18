from contextlib import contextmanager
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterator, Literal, TypedDict, Union, cast


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
    index_file: Path


class _Agent(TypedDict):
    id: str
    name: str
    description: str


class _Guideline(TypedDict, total=False):
    id: str
    when: str
    then: str
    enabled_tools: list[str]


class _LocalService(TypedDict):
    type: Literal["local"]
    tools: list[Any]


class _PluginService(TypedDict):
    type: Literal["plugin"]
    name: str
    url: str


_Service = Union[_LocalService, _PluginService]


def read_guideline_config(
    config_file: Path,
    agent: str = DEFAULT_AGENT_NAME,
) -> list[_Guideline]:
    config = json.loads(config_file.read_text())
    assert agent in config["guidelines"]
    return cast(list[_Guideline], config["guidelines"][agent])


def write_guideline_config(
    new_guidelines: list[_Guideline],
    config_file: Path,
    agent: str = DEFAULT_AGENT_NAME,
) -> None:
    config = json.loads(config_file.read_text())
    assert agent in config["guidelines"]
    config["guidelines"][agent] = new_guidelines
    config_file.write_text(json.dumps(config))


def write_service_config(
    new_services: list[_Service],
    config_file: Path,
) -> None:
    config = json.loads(config_file.read_text())
    config["services"] = new_services
    config_file.write_text(json.dumps(config))


def find_guideline(guideline: _Guideline, within: list[_Guideline]) -> bool:
    return bool(
        [g for g in within if g["when"] == guideline["when"] and g["then"] == guideline["then"]]
    )


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
