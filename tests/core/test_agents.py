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

from typing import AsyncIterator

from pytest import fixture

from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.common import IdGenerator
from parlant.core.agents import (
    AgentDocumentStore,
    AgentStore,
)
from parlant.core.playbooks import DisabledRuleRef


@fixture
async def agent_store(
    id_generator: IdGenerator,
) -> AsyncIterator[AgentStore]:
    async with AgentDocumentStore(
        id_generator=id_generator,
        database=TransientDocumentDatabase(),
    ) as store:
        yield store


@fixture
def id_generator() -> IdGenerator:
    return IdGenerator()


async def test_that_agent_has_empty_disabled_rules_by_default(
    agent_store: AgentStore,
) -> None:
    agent = await agent_store.create_agent(name="Test Agent")

    assert agent.disabled_rules == ()


async def test_that_disabled_rule_can_be_added_to_agent(
    agent_store: AgentStore,
) -> None:
    agent = await agent_store.create_agent(name="Test Agent")
    rule_ref = DisabledRuleRef("guideline:abc123")

    result = await agent_store.add_disabled_rule(agent.id, rule_ref)

    assert result is True

    updated_agent = await agent_store.read_agent(agent.id)
    assert rule_ref in updated_agent.disabled_rules


async def test_that_disabled_rule_can_be_removed_from_agent(
    agent_store: AgentStore,
) -> None:
    agent = await agent_store.create_agent(name="Test Agent")
    rule_ref = DisabledRuleRef("guideline:abc123")

    await agent_store.add_disabled_rule(agent.id, rule_ref)
    await agent_store.remove_disabled_rule(agent.id, rule_ref)

    updated_agent = await agent_store.read_agent(agent.id)
    assert rule_ref not in updated_agent.disabled_rules


async def test_that_adding_duplicate_disabled_rule_returns_false(
    agent_store: AgentStore,
) -> None:
    agent = await agent_store.create_agent(name="Test Agent")
    rule_ref = DisabledRuleRef("guideline:abc123")

    first_add = await agent_store.add_disabled_rule(agent.id, rule_ref)
    second_add = await agent_store.add_disabled_rule(agent.id, rule_ref)

    assert first_add is True
    assert second_add is False

    updated_agent = await agent_store.read_agent(agent.id)
    # Should only have one instance
    assert list(updated_agent.disabled_rules).count(rule_ref) == 1


async def test_that_multiple_disabled_rules_can_be_added_to_agent(
    agent_store: AgentStore,
) -> None:
    agent = await agent_store.create_agent(name="Test Agent")
    rule_ref1 = DisabledRuleRef("guideline:abc123")
    rule_ref2 = DisabledRuleRef("term:xyz789")
    rule_ref3 = DisabledRuleRef("capability:cap456")

    await agent_store.add_disabled_rule(agent.id, rule_ref1)
    await agent_store.add_disabled_rule(agent.id, rule_ref2)
    await agent_store.add_disabled_rule(agent.id, rule_ref3)

    updated_agent = await agent_store.read_agent(agent.id)
    assert rule_ref1 in updated_agent.disabled_rules
    assert rule_ref2 in updated_agent.disabled_rules
    assert rule_ref3 in updated_agent.disabled_rules


async def test_that_disabled_rules_are_deleted_when_agent_is_deleted(
    agent_store: AgentStore,
) -> None:
    agent = await agent_store.create_agent(name="Test Agent")
    rule_ref = DisabledRuleRef("guideline:abc123")

    await agent_store.add_disabled_rule(agent.id, rule_ref)

    # Delete the agent
    await agent_store.delete_agent(agent.id)

    # Create a new agent with the same ID should not have the old disabled rules
    new_agent = await agent_store.create_agent(name="New Agent", id=agent.id)
    assert new_agent.disabled_rules == ()
