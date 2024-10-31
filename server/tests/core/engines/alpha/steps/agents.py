from pytest_bdd import given, parsers

from emcie.server.core.agents import AgentId, AgentStore

from tests.core.engines.alpha.utils import ContextOfTest, step


@step(given, "an agent", target_fixture="agent_id")
def given_an_agent(
    agent_id: AgentId,
) -> AgentId:
    return agent_id


@step(given, parsers.parse("an agent whose job is {description}"), target_fixture="agent_id")
def given_an_agent_with_identity(
    context: ContextOfTest,
    description: str,
) -> AgentId:
    agent = context.sync_await(
        context.container[AgentStore].create_agent(
            name="test-agent",
            description=f"Your job is {description}",
        )
    )
    return agent.id
