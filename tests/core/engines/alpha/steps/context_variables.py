from pytest_bdd import given, parsers

from parlant.core.agents import AgentId
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableStore,
    ContextVariableValue,
)
from parlant.core.customers import CustomerStore
from parlant.core.sessions import SessionId, SessionStore

from parlant.core.tags import TagStore
from tests.core.engines.alpha.utils import ContextOfTest, step


def get_or_create_variable(
    context: ContextOfTest,
    agent_id: AgentId,
    context_variable_store: ContextVariableStore,
    variable_name: str,
) -> ContextVariable:
    variables = context.sync_await(context_variable_store.list_variables(agent_id))
    if variable := next(
        (variable for variable in variables if variable.name == variable_name), None
    ):
        return variable

    variable = context.sync_await(
        context_variable_store.create_variable(
            variable_set=agent_id,
            name=variable_name,
            description="",
            tool_id=None,
            freshness_rules=None,
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
            variable_set=agent_id,
            name=variable_name,
            description="",
            tool_id=None,
            freshness_rules=None,
        )
    )

    return context.sync_await(
        context_variable_store.update_value(
            variable_set=agent_id,
            key=customer_id,
            variable_id=variable.id,
            data=variable_value,
        )
    )


@step(
    given,
    parsers.parse(
        'a context variable "{variable_name}" set to "{variable_value}" to "{customer_name}"'
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
            variable_set=agent_id,
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
            variable_set=agent_id,
            name=variable_name,
            description="",
            tool_id=None,
            freshness_rules=None,
        )
    )

    return context.sync_await(
        context_variable_store.update_value(
            variable_set=agent_id,
            key=f"tag:{tag.id}",
            variable_id=variable.id,
            data=variable_value,
        )
    )
