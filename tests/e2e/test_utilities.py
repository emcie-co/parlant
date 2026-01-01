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

from __future__ import annotations
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
import socket
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

from tests.test_utilities import SERVER_BASE_URL, get_random_port


class _ServiceDTO(TypedDict):
    name: str
    kind: str
    url: str


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


def _wait_for_port_ready(
    server_address: str,
    max_attempts: int = 40,
    initial_delay: float = 0.1,
) -> None:
    """Wait for the server port to be ready to accept connections."""
    # Parse the server address to get host and port
    host, port_str = server_address.rsplit(":", 1)
    if "://" in host:
        host = host.split("://")[1]
    port = int(port_str)

    delay = initial_delay

    for attempt in range(max_attempts):
        try:
            with socket.create_connection((host, port), timeout=5):
                return  # Server is ready
        except (socket.error, ConnectionRefusedError, OSError):
            pass  # Server not ready yet

        if attempt == max_attempts - 1:
            raise RuntimeError(f"Server failed to become ready after {max_attempts} attempts")

        time.sleep(delay)
        delay = min(delay * 1.3, 2.0)  # Exponential backoff with max 2s


@contextmanager
def run_server(
    context: ContextOfTest,
    extra_args: list[str] = [],
) -> Iterator[subprocess.Popen[str]]:
    exec_args = [
        "uv",
        "run",
        "python",
        CLI_SERVER_PATH.as_posix(),
        "run",
        "-p",
        str(context.api.get_port()),
    ]

    exec_args.extend(extra_args)

    caught_exception: Exception | None = None

    try:
        process = subprocess.Popen(
            args=exec_args,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stdout,
            env={**os.environ, "PARLANT_HOME": context.home_dir.as_posix()},
        )

        try:
            _wait_for_port_ready(context.api.server_address)
            yield process
        except Exception as exc:
            caught_exception = exc

    finally:
        if process is not None:
            try:
                # First try a graceful shutdown (SIGINT)
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)

                for _ in range(10):
                    if process.poll() is not None:
                        break
                    time.sleep(0.5)

                # If still running, try terminating (SIGTERM)
                if process.poll() is None:
                    process.terminate()

                for _ in range(5):
                    if process.poll() is not None:
                        break
                    time.sleep(0.5)

                # If still running, force kill
                if process.poll() is None:
                    LOGGER.error(
                        f"Server process had to be killed. stderr={process.stderr.read() if process.stderr else 'None'}"
                    )
                    process.kill()
                    process.wait(timeout=5)

            except Exception as e:
                LOGGER.error(f"Error while shutting down server process: {e}")
                # Make sure process is killed as a last resort
                try:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=1)
                except Exception:
                    pass

        if caught_exception:
            raise caught_exception


class API:
    def __init__(self) -> None:
        self.set_port(get_random_port(10000, 50000))

    def set_port(self, port: int) -> None:
        self.server_address = f"{SERVER_BASE_URL}:{port}"

    def get_port(self) -> int:
        return int(self.server_address.split(":")[-1])

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
        name: str,
        description: str,
        synonyms: str = "",
    ) -> Any:
        async with self.make_client() as client:
            response = await client.post(
                "/terms",
                json={
                    "name": name,
                    "description": description,
                    **({"synonyms": synonyms.split(",")} if synonyms else {}),
                },
            )

            return response.raise_for_status().json()

    async def list_terms(self) -> Any:
        async with self.make_client() as client:
            response = await client.get(
                "/terms",
            )
            response.raise_for_status()

            return response.json()

    async def read_term(
        self,
        term_id: str,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.get(
                f"/terms/{term_id}",
            )
            response.raise_for_status()

            return response.json()

    async def list_guidelines(self) -> Any:
        async with self.make_client() as client:
            response = await client.get(
                "/guidelines",
            )

            response.raise_for_status()

            return response.json()

    async def read_guideline(
        self,
        guideline_id: str,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.get(
                f"/guidelines/{guideline_id}",
            )

            response.raise_for_status()

            return response.json()

    async def create_guideline(
        self,
        condition: str,
        action: str,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.post(
                "/guidelines",
                json={
                    "condition": condition,
                    "action": action,
                },
            )

            response.raise_for_status()

            return response.json()

    async def update_guideline(
        self,
        guideline_id: str,
        enabled: bool,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.patch(
                f"/guidelines/{guideline_id}",
                json={"enabled": enabled},
            )

            response.raise_for_status()

            return response.json()["guideline"]

    async def add_association(
        self,
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.patch(
                f"/guidelines/{guideline_id}",
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
        name: str,
        description: str,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.post(
                "/context-variables",
                json={
                    "name": name,
                    "description": description,
                },
            )

            response.raise_for_status()

            return response.json()

    async def list_context_variables(self) -> Any:
        async with self.make_client() as client:
            response = await client.get("/context-variables")

            response.raise_for_status()

            return response.json()

    async def update_context_variable_value(
        self,
        variable_id: str,
        key: str,
        value: Any,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.put(
                f"/context-variables/{variable_id}/{key}",
                json={"data": value},
            )
            response.raise_for_status()

    async def read_context_variable(
        self,
        variable_id: str,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.get(
                f"/context-variables/{variable_id}",
            )

            response.raise_for_status()

            return response.json()

    async def read_context_variable_value(
        self,
        variable_id: str,
        key: str,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.get(
                f"/context-variables/{variable_id}/{key}",
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
            response = await client.post("/customers", json={"name": name, "extra": extra})
            response.raise_for_status()

        return response.json()

    async def list_customers(
        self,
    ) -> Any:
        async with self.make_client() as client:
            response = await client.get("/customers")
            response.raise_for_status()

        return response.json()

    async def read_customer(self, id: str) -> Any:
        async with self.make_client() as client:
            response = await client.get(f"/customers/{id}")
            response.raise_for_status()

        return response.json()

    async def add_customer_tag(self, id: str, tag_id: str) -> None:
        async with self.make_client() as client:
            response = await client.patch(f"/customers/{id}", json={"tags": {"add": [tag_id]}})
            response.raise_for_status()

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

    async def list_canned_responses(self) -> Any:
        async with self.make_client() as client:
            response = await client.get(
                "/canned_responses",
            )

            response.raise_for_status()
            return response.json()
