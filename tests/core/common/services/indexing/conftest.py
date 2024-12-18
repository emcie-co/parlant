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

from dataclasses import dataclass
from lagom import Container
from pytest import fixture

from parlant.core.guidelines import GuidelineContent
from parlant.core.agents import AgentId, AgentStore
from tests.test_utilities import SyncAwaiter


@dataclass
class _CustomerTestContext:
    sync_await: SyncAwaiter
    container: Container
    agent_id: AgentId


@fixture
def agent_id(
    container: Container,
    sync_await: SyncAwaiter,
) -> AgentId:
    store = container[AgentStore]
    agent = sync_await(store.create_agent(name="test-agent"))
    return agent.id


@fixture
def customer_context(
    sync_await: SyncAwaiter,
    container: Container,
    agent_id: AgentId,
) -> _CustomerTestContext:
    return _CustomerTestContext(sync_await, container, agent_id)


@dataclass
class _TestContext:
    sync_await: SyncAwaiter
    container: Container


@fixture
def context(
    sync_await: SyncAwaiter,
    container: Container,
) -> _TestContext:
    return _TestContext(sync_await, container)


def _create_guideline_content(
    condition: str,
    action: str,
) -> GuidelineContent:
    return GuidelineContent(condition=condition, action=action)
