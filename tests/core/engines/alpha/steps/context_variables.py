from pytest_bdd import given, parsers

from emcie.server.core.agents import AgentId
from emcie.server.core.context_variables import (
    ContextVariableStore,
    ContextVariableValue,
)
from emcie.server.core.sessions import SessionId, SessionStore

from tests.core.engines.alpha.utils import ContextOfTest, step


@step(
    given, parsers.parse('a context variable "{variable_name}" with a value of "{variable_value}"')
)
def given_a_context_variable(
    context: ContextOfTest,
    variable_name: str,
    variable_value: str,
    agent_id: AgentId,
    session_id: SessionId,
) -> ContextVariableValue:
    session_store = context.container[SessionStore]
    context_variable_store = context.container[ContextVariableStore]

    end_user_id = context.sync_await(session_store.read_session(session_id)).end_user_id

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
            key=end_user_id,
            variable_id=variable.id,
            data={variable_name: variable_value},
        )
    )
