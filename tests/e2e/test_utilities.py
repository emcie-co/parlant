from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterator, Optional, TypedDict, cast
from typing_extensions import Literal

import httpx


class _ServiceDTO(TypedDict):
    name: str
    kind: str
    url: str


class _TermDTO(TypedDict):
    id: str
    name: str
    description: str
    synonyms: Optional[list[str]]


class _FreshnessRulesDTO(TypedDict):
    months: Optional[list[int]]
    days_of_month: Optional[list[int]]
    days_of_week: Optional[
        list[
            Literal[
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
            ]
        ]
    ]
    hours: Optional[list[int]]
    minutes: Optional[list[int]]
    seconds: Optional[list[int]]


class _ContextVariableDTO(TypedDict):
    id: str
    name: str
    description: Optional[str]
    tool_id: Optional[str]
    freshness_rules: Optional[_FreshnessRulesDTO]


class _GuidelineDTO(TypedDict):
    id: str
    predicate: str
    action: str


class _ContextVariableValueDTO(TypedDict):
    id: str
    last_modified: str
    data: Any


SERVER_PORT = 8089
SERVER_ADDRESS = f"http://localhost:{SERVER_PORT}"

LOGGER = logging.getLogger(__name__)


def get_package_path() -> Path:
    p = Path(__file__)

    while not (p / ".git").exists():
        p = p.parent
        assert p != Path("/"), "Failed to find repo path"

    package_path = p / "emcie"

    assert Path.cwd().is_relative_to(package_path), "Must run from within the package dir"

    return package_path


CLI_CLIENT_PATH = get_package_path() / "src/emcie/server/bin/client.py"
CLI_SERVER_PATH = get_package_path() / "src/emcie/server/bin/server.py"


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


async def get_first_agent_id() -> str:
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(30),
    ) as client:
        agents_response = await client.get(
            f"{SERVER_ADDRESS}/agents/",
        )
        agents_response.raise_for_status()

        assert len(agents_response.json()["agents"]) > 0
        agent = agents_response.json()["agents"][0]
        return str(agent["id"])


async def get_term_list(agent_id: str) -> list[_TermDTO]:
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(30),
    ) as client:
        response = await client.get(
            f"{SERVER_ADDRESS}/agents/{agent_id}/terms/",
        )
        response.raise_for_status()

        return cast(list[_TermDTO], response.json()["terms"])


async def create_term(agent_id: str, term_name: str, description: str) -> _TermDTO:
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(30),
    ) as client:
        response = await client.post(
            f"{SERVER_ADDRESS}/agents/{agent_id}/terms/",
            json={"name": term_name, "description": description},
        )
        response.raise_for_status()

        return cast(_TermDTO, response.json()["term"])


async def list_guidelines(agent_id: str) -> list[_GuidelineDTO]:
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(30),
    ) as client:
        response = await client.get(
            f"{SERVER_ADDRESS}/agents/{agent_id}/guidelines/",
        )

        response.raise_for_status()

        return cast(list[_GuidelineDTO], response.json()["guidelines"])


async def create_guideline(
    agent_id: str,
    predicate: str,
    action: str,
    coherence_check: Optional[dict[str, Any]] = None,
    connection_propositions: Optional[dict[str, Any]] = None,
) -> _GuidelineDTO:
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(30),
    ) as client:
        response = await client.post(
            f"{SERVER_ADDRESS}/agents/{agent_id}/guidelines/",
            json={
                "invoices": [
                    {
                        "payload": {
                            "kind": "guideline",
                            "content": {
                                "predicate": predicate,
                                "action": action,
                            },
                            "operation": "add",
                            "coherence_check": True,
                            "connection_proposition": True,
                        },
                        "checksum": "checksum_value",
                        "approved": True if coherence_check is None else False,
                        "data": {
                            "coherence_checks": coherence_check if coherence_check else [],
                            "connection_propositions": connection_propositions
                            if connection_propositions
                            else None,
                        },
                        "error": None,
                    }
                ]
            },
        )

        response.raise_for_status()

        return cast(_GuidelineDTO, response.json()["items"][0]["guideline"])


async def create_context_variable(
    agent_id: str, name: str, description: str
) -> _ContextVariableDTO:
    async with httpx.AsyncClient(
        base_url=SERVER_ADDRESS,
        follow_redirects=True,
        timeout=httpx.Timeout(30),
    ) as client:
        response = await client.post(
            f"/agents/{agent_id}/context-variables",
            json={
                "name": name,
                "description": description,
            },
        )

        response.raise_for_status()

        return cast(_ContextVariableDTO, response.json()["context_variable"])


async def list_context_variables(agent_id: str) -> list[_ContextVariableDTO]:
    async with httpx.AsyncClient(
        base_url=SERVER_ADDRESS,
        follow_redirects=True,
        timeout=httpx.Timeout(30),
    ) as client:
        response = await client.get(f"/agents/{agent_id}/context-variables/")

        response.raise_for_status()

        return cast(list[_ContextVariableDTO], response.json()["context_variables"])


async def create_context_variable_value(
    agent_id: str,
    variable_id: str,
    key: str,
    data: Any,
) -> _ContextVariableValueDTO:
    async with httpx.AsyncClient(
        base_url=SERVER_ADDRESS,
        follow_redirects=True,
        timeout=httpx.Timeout(30),
    ) as client:
        response = await client.put(
            f"/agents/{agent_id}/context-variables/{variable_id}/{key}",
            json={
                "data": data,
            },
        )

        response.raise_for_status()

        return cast(_ContextVariableValueDTO, response.json()["context_variable_value"])


async def read_context_variable_value(
    agent_id: str, variable_id: str, key: str
) -> _ContextVariableValueDTO:
    async with httpx.AsyncClient(
        base_url=SERVER_ADDRESS,
        follow_redirects=True,
        timeout=httpx.Timeout(30),
    ) as client:
        response = await client.get(
            f"{SERVER_ADDRESS}/agents/{agent_id}/context-variables/{variable_id}/{key}",
        )

        response.raise_for_status()

        return cast(_ContextVariableValueDTO, response.json())


async def create_sdk_service(service_name: str, url: str) -> None:
    payload = {"kind": "sdk", "url": url}

    async with httpx.AsyncClient() as client:
        response = await client.put(f"{SERVER_ADDRESS}/services/{service_name}", json=payload)
        response.raise_for_status()


async def list_services() -> list[_ServiceDTO]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{SERVER_ADDRESS}/services/")
        response.raise_for_status()

    return cast(list[_ServiceDTO], response.json()["services"])
