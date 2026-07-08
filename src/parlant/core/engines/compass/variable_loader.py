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

from datetime import datetime, timezone
import json
from typing import Optional

from croniter import croniter

from parlant.core.async_utils import safe_gather
from parlant.core.agents import AgentId
from parlant.core.common import DISABLE_WARNINGS
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableStore,
    ContextVariableValue,
)
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.engines.compass.tracing import CompassTracer
from parlant.core.entity_cq import EntityCommands, EntityQueries
from parlant.core.loggers import Logger
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.sessions import Session
from parlant.core.groups import GroupIds
from parlant.core.tools import ToolContext


class VariableLoader:
    def __init__(
        self,
        logger: Logger,
        entity_queries: EntityQueries,
        entity_commands: EntityCommands,
        estimating_tokenizer: EstimatingTokenizer,
    ) -> None:
        self._logger = logger
        self._entity_queries = entity_queries
        self._entity_commands = entity_commands
        self._estimating_tokenizer = estimating_tokenizer

    async def load(
        self,
        context: EngineContext,
    ) -> list[tuple[ContextVariable, ContextVariableValue]]:
        with context.tracer.span("load.variables"):
            variables_supported_by_agent = (
                await self._entity_queries.find_context_variables_for_context(
                    agent_id=context.agent.id,
                )
            )

            keys_to_check_in_order_of_importance = (
                [context.customer.id]
                + [f"group:{group_id}" for group_id in context.customer.groups]
                + [GroupIds.for_agent_id(context.agent.id)]
                + [ContextVariableStore.GLOBAL_KEY]
            )

            async def load_variable(
                variable: ContextVariable,
            ) -> Optional[tuple[ContextVariable, ContextVariableValue]]:
                # Try keys in order of importance, stopping at and using
                # the first (and most important) set key for each variable.
                for key in keys_to_check_in_order_of_importance:
                    if value := await self._load_context_variable_value(context, variable, key):
                        return (variable, value)

                return None

            loaded_values = await safe_gather(
                *[load_variable(variable) for variable in variables_supported_by_agent]
            )

            loaded = [loaded for loaded in loaded_values if loaded]

            CompassTracer(context.tracer).context_variables_loaded(loaded)

            return loaded

    async def _load_context_variable_value(
        self,
        context: EngineContext,
        variable: ContextVariable,
        key: str,
    ) -> Optional[ContextVariableValue]:
        value = await load_fresh_context_variable_value(
            entity_queries=self._entity_queries,
            entity_commands=self._entity_commands,
            agent_id=context.agent.id,
            session=context.session,
            variable=variable,
            key=key,
        )

        if (not DISABLE_WARNINGS) and variable.tool_id and value:
            token_count = await self._estimating_tokenizer.estimate_token_count(
                stringify_variable_output(value.data)
            )

            if token_count > 1_000:
                self._logger.warning(
                    f"Tool-enabled context variable '{variable.name}' produced "
                    f"{token_count} tokens for key '{key}'. Consider compacting it "
                    "or using a tool with a response-lifetime result instead."
                )

        return value


def stringify_variable_output(output: object) -> str:
    try:
        return json.dumps(output, ensure_ascii=False, default=str)
    except TypeError:
        return str(output)


async def load_fresh_context_variable_value(
    entity_queries: EntityQueries,
    entity_commands: EntityCommands,
    agent_id: AgentId,
    session: Session,
    variable: ContextVariable,
    key: str,
    current_time: datetime = datetime.now(timezone.utc),
) -> Optional[ContextVariableValue]:
    # Load the existing value
    value = await entity_queries.read_context_variable_value(
        variable_id=variable.id,
        key=key,
    )

    # If there's no tool attached to this variable,
    # return the value we found for the key.
    # Note that this may be None here, which is okay.
    if not variable.tool_id:
        return value

    # So we do have a tool attached.
    # Do we already have a value, and is it sufficiently fresh?
    if value and variable.freshness_rules:
        cron_iterator = croniter(variable.freshness_rules, value.modified_utc)

        if cron_iterator.get_next(datetime) > current_time:
            # We already have a fresh value in store. Return it.
            return value

    # We don't have a sufficiently fresh value.
    # Get an updated one, utilizing the associated tool.

    tool_context = ToolContext(
        agent_id=agent_id,
        session_id=session.id,
        customer_id=session.customer_id,
    )

    tool_service = await entity_queries.read_tool_service(variable.tool_id.service_name)

    tool_result = await tool_service.call_tool(
        variable.tool_id.tool_name,
        context=tool_context,
        arguments={},
    )

    return await entity_commands.update_context_variable_value(
        variable_id=variable.id,
        key=key,
        data=tool_result.data,
    )
