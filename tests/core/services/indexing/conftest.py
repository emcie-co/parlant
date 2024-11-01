from lagom import Container
from pytest import fixture
from emcie.server.core.agents import Agent, AgentStore
from tests.test_utilities import SyncAwaiter


@fixture
def agent(
    container: Container,
    sync_await: SyncAwaiter,
) -> Agent:
    store = container[AgentStore]
    agent = sync_await(store.create_agent(name="test-agent"))
    return agent
