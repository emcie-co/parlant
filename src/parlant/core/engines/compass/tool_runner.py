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

import asyncio
from collections.abc import Mapping
import json
import os

from parlant.core.common import DISABLE_WARNINGS, JSONSerializable
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.engines.compass.tracing import CompassTracer
from parlant.core.entity_cq import EntityQueries
from parlant.core.loggers import Logger
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.sessions import StatusEventData
from parlant.core.tools import ToolContext, ToolId, ToolResult, ToolService, pick_narration
from parlant.core.tracer import Tracer

# A tool call is given this long (seconds) to complete before it's abandoned and
# reported as an error. Overridable via the PARLANT_TOOL_TIMEOUT env var.
DEFAULT_TOOL_TIMEOUT = 300.0


class ToolRunner:
    """Runs a single tool against its service. Failures (including timeouts) are
    captured into an error ToolResult rather than raised, so the loop can feed them
    back to the model like any other result."""

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        entity_queries: EntityQueries,
        estimating_tokenizer: EstimatingTokenizer,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._compass_tracer = CompassTracer(tracer)
        self._entity_queries = entity_queries
        self._estimating_tokenizer = estimating_tokenizer

    async def run_tool(
        self,
        context: EngineContext,
        tool: ToolId,
        arguments: Mapping[str, JSONSerializable],
    ) -> ToolResult:
        with self._tracer.span(
            "tools.call",
            {
                "tool_id": tool.to_string(),
                "tool_name": tool.tool_name,
                "service_name": tool.service_name,
            },
        ):
            self._compass_tracer.tool_called(tool, arguments)

            result = await self._do_run_tool(context, tool, arguments)
            is_error = "error_details" in result.metadata

            self._compass_tracer.tool_result(tool, result)

            if is_error:
                self._compass_tracer.tool_error(tool, result)

            return result

    async def _do_run_tool(
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

        timeout = self._resolve_timeout()

        try:
            self._logger.debug(
                f"Running tool {tool.to_string()} with arguments {json.dumps(arguments, indent=2)}"
            )

            service = await self._entity_queries.read_tool_service(tool.service_name)

            await self._emit_narration(context, service, tool, tool_context)

            result = await asyncio.wait_for(
                service.call_tool(tool.tool_name, tool_context, arguments),
                timeout=timeout,
            )

            self._logger.debug(
                f"Tool {tool.to_string()} completed with result {json.dumps(result.data, indent=2)}"
            )

            await self._warn_if_result_is_large(tool, result)

            return result
        except asyncio.TimeoutError:
            self._logger.error(f"Tool call timed out after {timeout}s ({tool.to_string()})")
            return ToolResult(
                data="Tool call timed out",
                metadata={"error_details": f"Tool call timed out after {timeout} seconds"},
            )
        except Exception as e:
            self._logger.error(f"Tool call failed ({tool.to_string()}): {e}")
            return ToolResult(data="Tool call error", metadata={"error_details": str(e)})

    async def _emit_narration(
        self,
        context: EngineContext,
        service: ToolService,
        tool: ToolId,
        tool_context: ToolContext,
    ) -> None:
        """Show the tool's narration in the agent's "thinking" status before it runs.

        We resolve the tool (so a function-form narration is computed with the live
        ToolContext) and only emit when narration is present — otherwise the loop's
        generic "Running tool: X" status stands. Best-effort: a narration failure must
        never prevent the tool from running."""
        try:
            resolved = await service.resolve_tool(tool.tool_name, tool_context)

            message = pick_narration(resolved.narration)

            if message:
                await context.session_event_emitter.emit_status_event(
                    trace_id=context.tracer.trace_id,
                    data=StatusEventData(status="processing", message=message),
                )
        except Exception as e:
            self._logger.warning(f"Failed to resolve/emit narration for {tool.to_string()}: {e}")

    def _resolve_timeout(self) -> float:
        """Per-call tool timeout in seconds, from PARLANT_TOOL_TIMEOUT, falling back
        to the default (and on a malformed value, rather than breaking the call)."""
        raw = os.environ.get("PARLANT_TOOL_TIMEOUT")
        if not raw:
            return DEFAULT_TOOL_TIMEOUT
        try:
            return float(raw)
        except ValueError:
            self._logger.warning(
                f"Invalid PARLANT_TOOL_TIMEOUT={raw!r}; using default {DEFAULT_TOOL_TIMEOUT}s"
            )
            return DEFAULT_TOOL_TIMEOUT

    async def _warn_if_result_is_large(self, tool: ToolId, result: ToolResult) -> None:
        if DISABLE_WARNINGS:
            return

        lifespan = result.control.get("lifespan", "auto")
        threshold = 2_000 if lifespan in ("response", "auto") else 1_000

        token_count = await self._estimating_tokenizer.estimate_token_count(
            stringify_tool_result(result.data)
        )

        if token_count <= threshold:
            return

        if lifespan == "response":
            suggestion = "Consider compacting it."
        else:
            suggestion = 'Consider compacting it, or setting ToolResult(control={"lifespan": "response"}) to avoid excessive token accumulation throughout the session.'

        self._logger.warning(
            f"Tool {tool.to_string()} returned a {lifespan}-lifespan result with "
            f"{token_count} tokens. {suggestion}"
        )


def stringify_tool_result(output: object) -> str:
    try:
        return json.dumps(output, ensure_ascii=False, default=str)
    except TypeError:
        return str(output)
