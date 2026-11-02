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

from pytest import fixture
from pytest_bdd import scenarios

from parlant.core.agents import Agent, AgentStore, Effort
from parlant.core.container import Container

from tests.core.common.engines.compass.utils import load_steps
from tests.test_utilities import SyncAwaiter


@fixture
def agent(
    container: Container,
    sync_await: SyncAwaiter,
) -> Agent:
    # BDD scenarios run the full engine live; directive-level nuances (e.g. a
    # previously-applied policy re-matching on customer pushback) need real
    # per-step reasoning, so the default BDD agent runs at HIGH effort.
    store = container[AgentStore]
    return sync_await(
        store.create_agent(
            name="test-agent",
            max_steps=20,
            effort=Effort.HIGH,
        )
    )


load_steps(
    "agents",
    "context_variables",
    "engines",
    "events",
    "policies",
    "procedures",
    "canned_responses",
    "sessions",
    "terms",
    "tools",
    "customers",
    "groups",
)

scenarios(
    *(
        f"core/stable/engines/compass/features/baseline/{feature}.feature"
        for feature in (
            "conversation",
            "glossary",
            "context_variables",
            "moderation",
            "errors",
            "proactivity",
            "relationships",
            "triggered_utterances",
            "tools",
            "supervision",
            "canned_responses",
            "procedures",
        )
    )
)
