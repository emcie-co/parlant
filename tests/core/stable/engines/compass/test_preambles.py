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

from parlant.core.agents import AgentId
from parlant.core.engines.compass.preambles import (
    PreambleConfiguration,
    PreambleConfigurationProvider,
)


def test_that_preamble_configurations_are_isolated_by_agent() -> None:
    provider = PreambleConfigurationProvider()
    encouraged = PreambleConfiguration.encourage(interval_seconds=4)
    discouraged = PreambleConfiguration.discourage()

    provider.set(AgentId("encouraged"), encouraged)
    provider.set(AgentId("discouraged"), discouraged)

    assert provider.get(AgentId("encouraged")) is encouraged
    assert provider.get(AgentId("discouraged")) is discouraged
    assert provider.get(AgentId("unconfigured")) is None


def test_that_an_explicit_default_is_distinct_from_no_configuration() -> None:
    provider = PreambleConfigurationProvider()
    agent_id = AgentId("agent")
    explicit_default = PreambleConfiguration.default()

    provider.set(agent_id, explicit_default)
    assert provider.get(agent_id) is explicit_default

    provider.clear(agent_id)
    assert provider.get(agent_id) is None
