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

import os

import pytest

from parlant.core import metadata
from parlant.core.agents import AgentStore
from parlant.core.application import Application
from parlant.core.application_context import ApplicationContext
from parlant.core.container import Container
from parlant.core.customers import CustomerStore
from parlant.core.environment import Environment, resolve_environment


async def test_that_a_session_records_the_environment_it_was_created_in(
    container: Container,
) -> None:
    agent = await container[AgentStore].create_agent(name="Env Agent")
    customer = await container[CustomerStore].create_customer(name="Env Customer")

    session = await container[Application].sessions.create(
        customer_id=customer.id,
        agent_id=agent.id,
    )

    recorded = metadata.internal_value(session.metadata, ("server", "environment"))

    assert recorded == container[ApplicationContext].environment.value


async def test_that_a_caller_cannot_forge_the_environment_stamp(
    container: Container,
) -> None:
    agent = await container[AgentStore].create_agent(name="Env Agent")
    customer = await container[CustomerStore].create_customer(name="Env Customer")

    # The reserved namespace is rejected outright, so a frontend cannot pass
    # itself off as production traffic.
    with pytest.raises(ValueError):
        await container[Application].sessions.create(
            customer_id=customer.id,
            agent_id=agent.id,
            metadata={metadata.PARLANT_METADATA_KEY: {"server": {"environment": "production"}}},
        )


def test_that_the_environment_is_resolved_from_the_parlant_env_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARLANT_ENV", "production")
    assert resolve_environment() is Environment.PRODUCTION

    monkeypatch.setenv("PARLANT_ENV", "test")
    assert resolve_environment() is Environment.TEST

    # Anything else — including an unset variable — is development, which is
    # the only mode safe to assume by default.
    monkeypatch.setenv("PARLANT_ENV", "something-else")
    assert resolve_environment() is Environment.DEVELOPMENT

    monkeypatch.delenv("PARLANT_ENV", raising=False)
    assert resolve_environment() is Environment.DEVELOPMENT

    assert os.environ.get("PARLANT_ENV") is None
