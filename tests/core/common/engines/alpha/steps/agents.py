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

from typing import cast
from pytest_bdd import given, parsers

from parlant.core.agents import AgentId, AgentStore, CompositionMode

from tests.core.common.engines.alpha.utils import step
from tests.core.common.utils import ContextOfTest


@step(given, "an agent", target_fixture="agent_id")
def given_an_agent(
    agent_id: AgentId,
) -> AgentId:
    return agent_id


@step(given, parsers.parse("an agent whose job is {description}"), target_fixture="agent_id")
def given_an_agent_with_description(
    context: ContextOfTest,
    description: str,
) -> AgentId:
    agent = context.sync_await(
        context.container[AgentStore].create_agent(
            name="test-agent",
            description=f"Your job is {description}",
            max_engine_iterations=2,
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
            description=f"Your job is {description}",
            max_engine_iterations=2,
        )
    )
    return agent.id


@step(given, parsers.parse("that the agent uses the {mode} message composition mode"))
def given_that_the_agent_uses_a_message_composition(
    context: ContextOfTest,
    agent_id: AgentId,
    mode: str,
) -> None:
    context.sync_await(
        context.container[AgentStore].update_agent(
            agent_id,
            {"composition_mode": cast(CompositionMode, mode)},
        )
    )


@step(
    given,
    parsers.parse("an agent with max iteration of {max_engine_iterations}"),
    target_fixture="agent_id",
)
def given_an_agent_with_max_iteration(
    context: ContextOfTest,
    max_engine_iterations: str,
) -> AgentId:
    agent = context.sync_await(
        context.container[AgentStore].create_agent(
            name="test-agent",
            max_engine_iterations=int(max_engine_iterations),
        )
    )
    return agent.id
