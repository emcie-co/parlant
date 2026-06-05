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

from collections.abc import Mapping
import json

from parlant.core.common import JSONSerializable
from parlant.core.engines.sigma.response_state import EngineContext
from parlant.core.entity_cq import EntityQueries
from parlant.core.loggers import Logger
from parlant.core.tools import ToolContext, ToolId, ToolResult


class ToolRunner:
    """Runs a single tool against its service. Failures are captured into an
    error ToolResult rather than raised, so the loop can feed them back to the
    model like any other result."""

    def __init__(self, logger: Logger, entity_queries: EntityQueries) -> None:
        self._logger = logger
        self._entity_queries = entity_queries

    async def run_tool(
        self,
        context: EngineContext,
        tool: ToolId,
        arguments: Mapping[str, JSONSerializable],
    ) -> ToolResult:
        tool_context = ToolContext(
            agent_id=context.agent.id,
            session_id=context.session.id,
            customer_id=context.customer.id,
        )

        try:
            self._logger.debug(
                f"Running tool {tool.to_string()} with arguments {json.dumps(arguments, indent=2)}"
            )
            service = await self._entity_queries.read_tool_service(tool.service_name)
            return await service.call_tool(tool.tool_name, tool_context, arguments)
        except Exception as e:
            self._logger.error(f"Tool call failed ({tool.to_string()}): {e}")
            return ToolResult(data="Tool call error", metadata={"error_details": str(e)})
