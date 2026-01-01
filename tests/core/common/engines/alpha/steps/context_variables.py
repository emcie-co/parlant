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

from pytest_bdd import given, parsers

from parlant.core.agents import AgentId
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableStore,
    ContextVariableValue,
)
from parlant.core.customers import CustomerStore
from parlant.core.sessions import SessionId, SessionStore
from parlant.core.tags import Tag, TagStore
from parlant.core.tools import ToolId

from tests.core.common.engines.alpha.utils import step
from tests.core.common.utils import ContextOfTest


def get_or_create_variable(
    context: ContextOfTest,
    agent_id: AgentId,
    context_variable_store: ContextVariableStore,
    variable_name: str,
) -> ContextVariable:
    variables = context.sync_await(
        context_variable_store.list_variables(tags=[Tag.for_agent_id(agent_id)])
    )
    if variable := next(
        (variable for variable in variables if variable.name == variable_name), None
    ):
        return variable

    variable = context.sync_await(
        context_variable_store.create_variable(
            name=variable_name,
            description="",
            tool_id=None,
            freshness_rules=None,
        )
    )

    context.sync_await(
        context_variable_store.add_variable_tag(
            variable_id=variable.id,
            tag_id=Tag.for_agent_id(agent_id),
        )
    )
    return variable


@step(given, parsers.parse('a context variable "{variable_name}" set to "{variable_value}"'))
def given_a_context_variable(
    context: ContextOfTest,
    variable_name: str,
    variable_value: str,
    agent_id: AgentId,
    session_id: SessionId,
) -> ContextVariableValue:
    session_store = context.container[SessionStore]
    context_variable_store = context.container[ContextVariableStore]

    customer_id = context.sync_await(session_store.read_session(session_id)).customer_id

    variable = context.sync_await(
        context_variable_store.create_variable(
            name=variable_name,
            description="",
            tool_id=None,
            freshness_rules=None,
        )
    )

    context.sync_await(
        context_variable_store.add_variable_tag(
            variable_id=variable.id,
            tag_id=Tag.for_agent_id(agent_id),
        )
    )

    return context.sync_await(
        context_variable_store.update_value(
            key=customer_id,
            variable_id=variable.id,
            data=variable_value,
        )
    )


@step(
    given,
    parsers.parse(
        'a context variable "{variable_name}" set to "{variable_value}" for "{customer_name}"'
    ),
)
def given_a_context_variable_to_specific_customer(
    context: ContextOfTest,
    variable_name: str,
    variable_value: str,
    customer_name: str,
    agent_id: AgentId,
) -> ContextVariableValue:
    customer_store = context.container[CustomerStore]
    context_variable_store = context.container[ContextVariableStore]

    customers = context.sync_await(customer_store.list_customers())

    customer = next(c for c in customers if c.name == customer_name)

    variable = get_or_create_variable(context, agent_id, context_variable_store, variable_name)

    return context.sync_await(
        context_variable_store.update_value(
            key=customer.id,
            variable_id=variable.id,
            data=variable_value,
        )
    )


@step(
    given,
    parsers.parse(
        'a context variable "{variable_name}" set to "{variable_value}" for the tag "{name}"'
    ),
)
def given_a_context_variable_for_a_tag(
    context: ContextOfTest,
    variable_name: str,
    variable_value: str,
    agent_id: AgentId,
    name: str,
) -> ContextVariableValue:
    context_variable_store = context.container[ContextVariableStore]
    tag_store = context.container[TagStore]

    tag = next(t for t in context.sync_await(tag_store.list_tags()) if t.name == name)

    variable = context.sync_await(
        context_variable_store.create_variable(
            name=variable_name,
            description="",
            tool_id=None,
            freshness_rules=None,
        )
    )

    return context.sync_await(
        context_variable_store.update_value(
            key=f"tag:{tag.id}",
            variable_id=variable.id,
            data=variable_value,
        )
    )


@step(
    given,
    parsers.parse(
        'the context variable "{variable_name}" has freshness rules of "{freshness_rules}"'
    ),
)
def given_a_context_variable_with_freshness_rules(
    context: ContextOfTest,
    variable_name: str,
    freshness_rules: str,
    agent_id: AgentId,
) -> ContextVariable:
    context_variable_store = context.container[ContextVariableStore]

    variable = get_or_create_variable(context, agent_id, context_variable_store, variable_name)

    return context.sync_await(
        context_variable_store.update_variable(
            variable_id=variable.id,
            params={"freshness_rules": freshness_rules},
        )
    )


@step(
    given,
    parsers.parse('the context variable "{variable_name}" is connected to the tool "{tool_name}"'),
)
def given_a_context_variable_with_tool(
    context: ContextOfTest,
    variable_name: str,
    tool_name: str,
    agent_id: AgentId,
) -> ContextVariable:
    context_variable_store = context.container[ContextVariableStore]

    variable = get_or_create_variable(context, agent_id, context_variable_store, variable_name)

    return context.sync_await(
        context_variable_store.update_variable(
            variable_id=variable.id,
            params={"tool_id": ToolId(service_name="local", tool_name=tool_name)},
        )
    )
