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

from parlant.core.agents import AgentId, AgentStore, Effort

from tests.core.common.engines.compass.utils import step
from tests.core.common.utils import ContextOfTest


@step(given, "an agent", target_fixture="agent_id")
def given_an_agent(
    agent_id: AgentId,
) -> AgentId:
    return agent_id


# On max_steps=20 (alpha used 2): an alpha iteration was a full
# match->batched-tool-calls->message cycle, while a compass iteration is one
# react step, which may serialize tool calls — so tool scenarios need headroom
# to run every obligated tool and still respond.


@step(given, parsers.parse("an agent whose job is {description}"), target_fixture="agent_id")
def given_an_agent_with_description(
    context: ContextOfTest,
    description: str,
) -> AgentId:
    agent = context.sync_await(
        context.container[AgentStore].create_agent(
            name="test-agent",
            instructions=f"Your job is {description}",
            max_steps=20,
            effort=Effort.HIGH,
        )
    )
    return agent.id


@step(
    given,
    parsers.parse('an agent named "{name}" whose job is {description}'),
    target_fixture="agent_id",
)
def given_an_agent_with_description_and_name(
    context: ContextOfTest,
    description: str,
    name: str,
) -> AgentId:
    agent = context.sync_await(
        context.container[AgentStore].create_agent(
            name=name,
            instructions=f"Your job is {description}",
            max_steps=20,
            effort=Effort.HIGH,
        )
    )
    return agent.id


@step(
    given,
    parsers.parse("an agent with max iteration of {max_steps}"),
    target_fixture="agent_id",
)
def given_an_agent_with_max_iteration(
    context: ContextOfTest,
    max_steps: str,
) -> AgentId:
    agent = context.sync_await(
        context.container[AgentStore].create_agent(
            name="test-agent",
            max_steps=int(max_steps),
            effort=Effort.HIGH,
        )
    )
    return agent.id
