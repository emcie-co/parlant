from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Iterator, TypedDict, cast
from loguru import logger


SERVER_PORT = 8089
SERVER_ADDRESS = f"http://localhost:{SERVER_PORT}"

DEFAULT_AGENT_NAME = "Default Agent"


def get_package_path() -> Path:
    p = Path(__file__)

    while not (p / ".git").exists():
        p = p.parent
        assert p != Path("/"), "Failed to find repo path"

    package_path = p / "server"

    assert Path.cwd().is_relative_to(package_path), "Must run from within the package dir"

    return package_path


CLI_PATH = get_package_path() / "bin/emcie-server"


@dataclass(frozen=True)
class _TestContext:
    home_dir: Path
    config_file: Path


class _Agent(TypedDict, total=False):
    id: str
    name: str
    description: str


class _Guideline(TypedDict, total=False):
    id: str
    when: str
    then: str
    enabled_tools: list[str]


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


def load_active_agents(home_dir: Path) -> list[_Agent]:
    agent_store = home_dir / "agents.json"

    agent_data = json.loads(agent_store.read_text())

    return [
        {
            "id": a["id"],
            "name": a["name"],
            "description": a.get("description"),
        }
        for a in agent_data["agents"]
    ]


def load_active_agent(home_dir: Path, agent_name: str) -> _Agent:
    agents = load_active_agents(home_dir)
    matching_agents = [a for a in agents if a["name"] == agent_name]
    assert len(matching_agents) == 1
    return matching_agents[0]


def read_loaded_guidelines(
    home_dir: Path,
    agent: str = DEFAULT_AGENT_NAME,
) -> list[_Guideline]:
    guideline_store = home_dir / "guidelines.json"
    guideline_data = json.loads(guideline_store.read_text())

    agent_id = load_active_agent(home_dir, agent)["id"]

    return [
        {
            "when": g["predicate"],
            "then": g["content"],
        }
        for g in guideline_data["guidelines"]
        if g["guideline_set"] == agent_id
    ]


def find_guideline(guideline: _Guideline, within: list[_Guideline]) -> bool:
    return bool(
        [g for g in within if g["when"] == guideline["when"] and g["then"] == guideline["then"]]
    )


@contextmanager
def run_server(context: _TestContext) -> Iterator[subprocess.Popen[str]]:
    exec_args = [
        "poetry",
        "run",
        "python",
        CLI_PATH.as_posix(),
        "-c",
        context.config_file.as_posix(),
        "run",
        "-p",
        str(SERVER_PORT),
    ]

    with subprocess.Popen(
        args=exec_args,
        text=True,
        stdout=sys.stdout,
        stderr=sys.stdout,
        env={**os.environ, "EMCIE_HOME": context.home_dir.as_posix()},
    ) as process:
        yield process

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

        logger.error(
            "Server process had to be killed. stderr="
            + (process.stderr and process.stderr.read() or "None")
        )

        process.kill()
        process.wait()
