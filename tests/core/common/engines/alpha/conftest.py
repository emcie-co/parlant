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

from datetime import datetime, timezone

from lagom import Container
from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.customers import Customer, CustomerId, CustomerStore
from parlant.core.engines.alpha.tool_caller import ToolCaller, ToolCallInferenceSchema
from parlant.core.logging import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import Session, SessionStore
from parlant.core.tools import LocalToolService
from pytest import fixture

from tests.core.common.engines.alpha.utils import ContextOfTest
from tests.test_utilities import SyncAwaiter


@fixture
def context(sync_await: SyncAwaiter, container: Container) -> ContextOfTest:
    return ContextOfTest(
        sync_await,
        container,
        events=list(),
        guidelines=dict(),
        guideline_propositions=dict(),
        tools=dict(),
    )


@fixture
def agent(context: ContextOfTest) -> Agent:
    store = context.container[AgentStore]
    agent = context.sync_await(
        store.create_agent(
            name="test-agent",
        ),
    )
    return agent


@fixture
def agent_id(agent: Agent) -> AgentId:
    return agent.id


@fixture
def customer(context: ContextOfTest) -> Customer:
    store = context.container[CustomerStore]
    customer = context.sync_await(
        store.create_customer(
            name="Test Customer",
            extra={"email": "test@customer.com"},
        ),
    )
    return customer


@fixture
def customer_id(customer: Customer) -> CustomerId:
    return customer.id


@fixture
def new_session(
    context: ContextOfTest,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> Session:
    store = context.container[SessionStore]
    utc_now = datetime.now(timezone.utc)
    return context.sync_await(
        store.create_session(
            creation_utc=utc_now,
            customer_id=customer_id,
            agent_id=agent_id,
        )
    )


@fixture
def local_tool_service(container: Container) -> LocalToolService:
    return container[LocalToolService]


@fixture
def tool_caller(container: Container) -> ToolCaller:
    return ToolCaller(
        container[Logger],
        container[ServiceRegistry],
        container[SchematicGenerator[ToolCallInferenceSchema]],
    )
