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

from typing import Optional
from lagom import Container
from enum import Enum

from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.services.tools.mcp_service import MCPToolServer, DEFAULT_MCP_PORT
from parlant.core.services.tools.service_registry import ServiceRegistry

server_instance: MCPToolServer | None = None


class ParrotSpecies(Enum):
    PARROT = "parrot"
    MACAW = "macaw"
    CONURE = "conure"
    LORIKEET = "lorikeet"
    NEELGAI = "neelgai"
    KAKA = "kaka"
    DRARA = "parakeet"


def parrot_numbers(my_bets: list[int], in_reality: list[float]) -> str:
    return f"Your bets on your grades were {my_bets} but in reality you got {in_reality}"


def parrot_bools(bools_high: list[bool], boolbool: Optional[bool] = False) -> str:
    return f"Bull's eye {bools_high} and boolbool  {boolbool}"


def parrot_enums(parrot_friends: list[ParrotSpecies]) -> str:
    return f"My friends species are {parrot_friends}"


async def configure_module(container: Container) -> Container:
    global server_instance
    _background_task_service = container[BackgroundTaskService]

    server = MCPToolServer(
        tools=[parrot_numbers, parrot_bools, parrot_enums],
        port=DEFAULT_MCP_PORT,
        host="0.0.0.0",
    )

    await _background_task_service.start(
        server.serve(),
        tag="Parrot service",
    )

    server_instance = server
    return container


async def initialize_module(container: Container) -> None:
    service_registry = container[ServiceRegistry]
    await service_registry.update_tool_service(
        name="parrot",
        kind="mcp",
        url=f"http://127.0.0.1:{DEFAULT_MCP_PORT}",
        transient=True,
    )


async def shutdown_module() -> None:
    global server_instance
