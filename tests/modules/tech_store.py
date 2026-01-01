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

import httpx
from lagom import Container

from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.services.tools.plugins import PluginServer, tool
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tools import ToolContext, ToolResult


server_instance: PluginServer | None = None


@tool
def list_categories(context: ToolContext) -> ToolResult:
    return ToolResult(["laptops", "chairs"])


@tool
async def consult_expert(context: ToolContext, user_query: str) -> ToolResult:
    """
    This is an example for using the canned responses feature
    """

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        (
            await client.post(
                f"http://localhost:8800/sessions/{context.session_id}/events",
                json={
                    "kind": "message",
                    "source": "ai_agent",
                    "actions": [
                        {
                            "action": "Tell the user that you're thinking and will be right back with an answer",
                            "reason": "buy_time",
                        }
                    ],
                },
            )
        ).raise_for_status().json()

    return ToolResult(data="Best laptop is mac")


async def configure_module(container: Container) -> Container:
    global server_instance
    _background_task_service = container[BackgroundTaskService]

    server = PluginServer(
        tools=[list_categories, consult_expert],
        port=8095,
        host="127.0.0.1",
    )

    await _background_task_service.start(
        server.serve(),
        tag="Tech Store Plugin",
    )
    server_instance = server

    return container


async def initialize_module(container: Container) -> None:
    service_registry = container[ServiceRegistry]
    await service_registry.update_tool_service(
        name="tech-store",
        kind="sdk",
        url="http://127.0.0.1:8095",
        transient=True,
    )


async def shutdown_module() -> None:
    global server_instance

    if server_instance is not None:
        await server_instance.shutdown()
        server_instance = None
