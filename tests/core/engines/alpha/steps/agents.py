# Copyright 2024 Emcie Co Ltd.
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

from parlant.core.agents import AgentId, AgentStore

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


@step(
    given,
    parsers.parse('an agent named "{name}" whose job is {description}'),
    target_fixture="agent_id",
)
def given_an_agent_with_identity_and_name(
    context: ContextOfTest,
    description: str,
    name: str,
) -> AgentId:
    agent = context.sync_await(
        context.container[AgentStore].create_agent(
            name=name,
            description=f"Your job is {description}",
        )
    )
    return agent.id
