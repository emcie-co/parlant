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

from lagom import Container
from pytest import fixture

from parlant.core.agents import AgentId
from parlant.core.customers import CustomerId
from parlant.core.sessions import SessionId

from tests.test_utilities import create_agent, create_customer, create_session


@fixture
async def agent_id(container: Container) -> AgentId:
    agent = await create_agent(container, name="test-agent")
    return agent.id


@fixture
async def customer_id(container: Container) -> CustomerId:
    customer = await create_customer(container, "Test Customer")
    return customer.id


@fixture
async def session_id(container: Container, agent_id: AgentId) -> SessionId:
    session = await create_session(container, agent_id=agent_id)
    return session.id
