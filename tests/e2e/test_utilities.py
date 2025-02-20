# Copyright 2024 Emcie Co Ltd.
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

from __future__ import annotations
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
import traceback
import httpx
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, AsyncIterator, Iterator, Optional, TypedDict, cast

from tests.test_utilities import SERVER_ADDRESS, SERVER_PORT


class _ServiceDTO(TypedDict):
    name: str
    kind: str
    url: str


class _TermDTO(TypedDict):
    id: str
    name: str
    description: str
    synonyms: Optional[list[str]]


class _ContextVariableDTO(TypedDict):
    id: str
    name: str
    description: Optional[str]
    tool_id: Optional[str]
    freshness_rules: Optional[str]


class _GuidelineDTO(TypedDict):
    id: str
    condition: str
    action: str


class _ContextVariableValueDTO(TypedDict):
    id: str
    last_modified: str
    data: Any


LOGGER = logging.getLogger(__name__)


def get_package_path() -> Path:
    p = Path(__file__)

    while not (p / ".git").exists():
        p = p.parent
        assert p != Path("/"), "Failed to find repo path"

    package_path = p / "."

    assert Path.cwd().is_relative_to(package_path), "Must run from within the package dir"

    return package_path


CLI_CLIENT_PATH = get_package_path() / "src/parlant/bin/client.py"
CLI_SERVER_PATH = get_package_path() / "src/parlant/bin/server.py"


@dataclass(frozen=True)
class ContextOfTest:
    home_dir: Path
    api: API


def is_server_responsive(port: int) -> bool:
    if _output_view := subprocess.getoutput(f"lsof -i:{port}"):
        print(_output_view)
        return True

    return False


@contextmanager
def run_server(
    context: ContextOfTest,
    extra_args: list[str] = [],
) -> Iterator[subprocess.Popen[str]]:
    if is_server_responsive(int(SERVER_PORT)):
        raise Exception(f"Server already running on chosen port {SERVER_PORT}")

    exec_args = [
        "poetry",
        "run",
        "python",
        CLI_SERVER_PATH.as_posix(),
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
            env={**os.environ, "PARLANT_HOME": context.home_dir.as_posix()},
        ) as process:
            try:
                yield process
            except Exception as exc:
                caught_exception = exc

            if process.poll() is not None:
                return

            process.send_signal(signal.SIGINT)

            for _ in range(5):
                if process.poll() is not None:
                    return
                time.sleep(0.5)

            process.terminate()

            for _ in range(5):
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


class API:
    def __init__(self, server_address: str = SERVER_ADDRESS) -> None:
        self.server_address = server_address

    @asynccontextmanager
    async def make_client(
        self,
    ) -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(
            base_url=self.server_address,
            follow_redirects=True,
            timeout=httpx.Timeout(60),
        ) as client:
            yield client

    async def get_first_agent(
        self,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.get("/agents")
            agent = response.raise_for_status().json()[0]
            return agent

    async def create_agent(
        self,
        name: str,
        description: Optional[str] = None,
        max_engine_iterations: Optional[int] = None,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.post(
                "/agents",
                json={
                    "name": name,
                    "description": description,
                    "max_engine_iterations": max_engine_iterations,
                },
            )

            return response.raise_for_status().json()

    async def list_agents(
        self,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.get("/agents")
            return response.raise_for_status().json()

    async def create_session(
        self,
        agent_id: str,
        customer_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.post(
                "/sessions",
                params={"allow_greeting": False},
                json={
                    "agent_id": agent_id,
                    **({"customer_id": customer_id} if customer_id else {}),
                    "title": title,
                },
            )

            return response.raise_for_status().json()

    async def read_session(self, session_id: str) -> Any:
        async with self.make_client() as client:
            response = await client.get(
                f"/sessions/{session_id}",
            )

            return response.raise_for_status().json()

    async def get_agent_reply(
        self,
        session_id: str,
        message: str,
    ) -> Any:
        return next(iter(await self.get_agent_replies(session_id, message, 1)))

    async def get_agent_replies(
        self,
        session_id: str,
        message: str,
        number_of_replies_to_expect: int,
    ) -> list[Any]:
        async with self.make_client() as client:
            try:
                customer_message_response = await client.post(
                    f"/sessions/{session_id}/events",
                    json={
                        "kind": "message",
                        "source": "customer",
                        "message": message,
                    },
                )
                customer_message_response.raise_for_status()
                customer_message_offset = int(customer_message_response.json()["offset"])

                last_known_offset = customer_message_offset

                replies: list[Any] = []
                start_time = time.time()
                timeout = 300

                while len(replies) < number_of_replies_to_expect:
                    response = await client.get(
                        f"/sessions/{session_id}/events",
                        params={
                            "min_offset": last_known_offset + 1,
                            "kinds": "message",
                        },
                    )
                    response.raise_for_status()
                    events = response.json()

                    if message_events := [e for e in events if e["kind"] == "message"]:
                        replies.append(message_events[0])

                    last_known_offset = events[-1]["offset"]

                    if (time.time() - start_time) >= timeout:
                        raise TimeoutError()

                return replies
            except:
                traceback.print_exc()
                raise

    async def create_term(
        self,
        agent_id: str,
        name: str,
        description: str,
        synonyms: str = "",
    ) -> Any:
        async with self.make_client() as client:
            response = await client.post(
                f"/agents/{agent_id}/terms/",
                json={
                    "name": name,
                    "description": description,
                    **({"synonyms": synonyms.split(",")} if synonyms else {}),
                },
            )

            return response.raise_for_status().json()

    async def list_terms(self, agent_id: str) -> Any:
        async with self.make_client() as client:
            response = await client.get(
                f"/agents/{agent_id}/terms/",
            )
            response.raise_for_status()

            return response.json()

    async def read_term(
        self,
        agent_id: str,
        term_id: str,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.get(
                f"/agents/{agent_id}/terms/{term_id}",
            )
            response.raise_for_status()

            return response.json()

    async def list_guidelines(self, agent_id: str) -> Any:
        async with self.make_client() as client:
            response = await client.get(
                f"/agents/{agent_id}/guidelines/",
            )

            response.raise_for_status()

            return response.json()

    async def read_guideline(
        self,
        agent_id: str,
        guideline_id: str,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.get(
                f"/agents/{agent_id}/guidelines/{guideline_id}",
            )

            response.raise_for_status()

            return response.json()

    async def create_guideline(
        self,
        agent_id: str,
        condition: str,
        action: str,
        coherence_check: Optional[dict[str, Any]] = None,
        connection_propositions: Optional[dict[str, Any]] = None,
        operation: str = "add",
        updated_id: Optional[str] = None,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.post(
                f"/agents/{agent_id}/guidelines",
                json={
                    "invoices": [
                        {
                            "payload": {
                                "kind": "guideline",
                                "guideline": {
                                    "content": {
                                        "condition": condition,
                                        "action": action,
                                    },
                                    "operation": operation,
                                    "updated_id": updated_id,
                                    "coherence_check": True,
                                    "connection_proposition": True,
                                },
                            },
                            "checksum": "checksum_value",
                            "approved": True if coherence_check is None else False,
                            "data": {
                                "guideline": {
                                    "coherence_checks": coherence_check if coherence_check else [],
                                    "connection_propositions": connection_propositions
                                    if connection_propositions
                                    else None,
                                }
                            },
                            "error": None,
                        }
                    ]
                },  # type: ignore
            )

            response.raise_for_status()

            return response.json()["items"][0]["guideline"]

    async def add_association(
        self,
        agent_id: str,
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.patch(
                f"/agents/{agent_id}/guidelines/{guideline_id}",
                json={
                    "tool_associations": {
                        "add": [
                            {
                                "service_name": service_name,
                                "tool_name": tool_name,
                            }
                        ]
                    }
                },
            )

            response.raise_for_status()

        return response.json()["tool_associations"]

    async def create_context_variable(
        self,
        agent_id: str,
        name: str,
        description: str,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.post(
                f"/agents/{agent_id}/context-variables",
                json={
                    "name": name,
                    "description": description,
                },
            )

            response.raise_for_status()

            return response.json()

    async def list_context_variables(self, agent_id: str) -> Any:
        async with self.make_client() as client:
            response = await client.get(f"/agents/{agent_id}/context-variables/")

            response.raise_for_status()

            return response.json()

    async def update_context_variable_value(
        self,
        agent_id: str,
        variable_id: str,
        key: str,
        value: Any,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.put(
                f"/agents/{agent_id}/context-variables/{variable_id}/{key}",
                json={"data": value},
            )
            response.raise_for_status()

    async def read_context_variable(
        self,
        agent_id: str,
        variable_id: str,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.get(
                f"/agents/{agent_id}/context-variables/{variable_id}",
            )

            response.raise_for_status()

            return response.json()

    async def read_context_variable_value(
        self,
        agent_id: str,
        variable_id: str,
        key: str,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.get(
                f"/agents/{agent_id}/context-variables/{variable_id}/{key}",
            )

            response.raise_for_status()

            return response.json()

    async def create_sdk_service(self, service_name: str, url: str) -> None:
        payload = {"kind": "sdk", "sdk": {"url": url}}

        async with self.make_client() as client:
            response = await client.put(f"/services/{service_name}", json=payload)
            response.raise_for_status()

    async def create_openapi_service(
        self,
        service_name: str,
        url: str,
    ) -> None:
        payload = {"kind": "openapi", "openapi": {"source": f"{url}/openapi.json", "url": url}}

        async with self.make_client() as client:
            response = await client.put(f"/services/{service_name}", json=payload)
            response.raise_for_status()

    async def list_services(
        self,
    ) -> list[_ServiceDTO]:
        async with self.make_client() as client:
            response = await client.get("/services")
            response.raise_for_status()

        return cast(list[_ServiceDTO], response.json())

    async def create_tag(self, name: str) -> Any:
        async with self.make_client() as client:
            response = await client.post("/tags", json={"name": name})
        return response.json()

    async def list_tags(
        self,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.get("/tags")
        return response.json()

    async def read_tag(self, id: str) -> Any:
        async with self.make_client() as client:
            response = await client.get(f"/tags/{id}")
        return response.json()

    async def create_customer(
        self,
        name: str,
        extra: Optional[dict[str, Any]] = {},
    ) -> Any:
        async with self.make_client() as client:
            respone = await client.post("/customers", json={"name": name, "extra": extra})
            respone.raise_for_status()

        return respone.json()

    async def list_customers(
        self,
    ) -> Any:
        async with self.make_client() as client:
            respone = await client.get("/customers")
            respone.raise_for_status()

        return respone.json()

    async def read_customer(self, id: str) -> Any:
        async with self.make_client() as client:
            respone = await client.get(f"/customers/{id}")
            respone.raise_for_status()

        return respone.json()

    async def add_customer_tag(self, id: str, tag_id: str) -> None:
        async with self.make_client() as client:
            respone = await client.patch(f"/customers/{id}", json={"tags": {"add": [tag_id]}})
            respone.raise_for_status()

    async def create_evaluation(self, agent_id: str, payloads: Any) -> Any:
        async with self.make_client() as client:
            evaluation_creation_response = await client.post(
                "/index/evaluations",
                json={"agent_id": agent_id, "payloads": payloads},
            )
            evaluation_creation_response.raise_for_status()
            return evaluation_creation_response.json()

    async def read_evaluation(self, evaluation_id: str) -> Any:
        async with self.make_client() as client:
            evaluation_response = await client.get(
                f"/index/evaluations/{evaluation_id}",
            )
            evaluation_response.raise_for_status()
            return evaluation_response.json()

    async def list_fragments(self) -> Any:
        async with self.make_client() as client:
            fragments_reponse = await client.get(
                "/fragments",
            )
            fragments_reponse.raise_for_status()
            return fragments_reponse.json()
