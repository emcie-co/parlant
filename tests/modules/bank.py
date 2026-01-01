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

from lagom import Container

from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.services.tools.plugins import PluginServer, tool
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tools import ToolContext, ToolResult


server_instance: PluginServer | None = None


@tool
def read_account_balance(context: ToolContext) -> ToolResult:
    return ToolResult(data="999", canned_response_fields={"balance": 999})


@tool
def get_account_details(context: ToolContext) -> ToolResult:
    return ToolResult({"name": "John Doe", "account_number": "1234567890"})


async def configure_module(container: Container) -> Container:
    global server_instance
    _background_task_service = container[BackgroundTaskService]

    server = PluginServer(
        tools=[read_account_balance, get_account_details],
        port=8094,
        host="127.0.0.1",
    )

    await _background_task_service.start(
        server.serve(),
        tag="Bank Plugin",
    )

    server_instance = server
    return container


async def initialize_module(container: Container) -> None:
    service_registry = container[ServiceRegistry]
    await service_registry.update_tool_service(
        name="bank",
        kind="sdk",
        url="http://127.0.0.1:8094",
        transient=True,
    )


async def shutdown_module() -> None:
    global server_instance

    if server_instance is not None:
        await server_instance.shutdown()
        server_instance = None
