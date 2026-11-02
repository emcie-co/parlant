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

import parlant.sdk as p

from tests.sdk.utils import Context, SDKTest


def test_that_the_sdk_exports_the_compass_preamble_types() -> None:
    assert p.PreambleConfiguration.__module__ == "parlant.core.engines.compass.preambles"
    assert p.PreambleConfigurationProvider.__module__ == ("parlant.core.engines.compass.preambles")


class Test_that_a_compass_agent_can_configure_preambles(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.configuration = p.PreambleConfiguration.encourage(interval_seconds=7)
        self.agent = await server.create_agent(
            id="preambles-agent-1",
            name="Preamble Agent",
            instructions="An agent with explicit preamble behavior",
            preamble_config=self.configuration,
        )

    async def run(self, ctx: Context) -> None:
        configured = ctx.container[p.PreambleConfigurationProvider].get(self.agent.id)

        assert configured is self.configuration
        assert configured.interval_seconds == 7
        assert configured.emission_policy == p.PreambleEmissionPolicy.INTERVAL


class Test_that_a_custom_preamble_provider_can_supply_deployment_defaults(SDKTest):
    deployment_default = p.PreambleConfiguration.discourage()

    async def configure_container(self, container: p.Container) -> p.Container:
        deployment_default = self.deployment_default

        class DeploymentPreambleProvider(p.PreambleConfigurationProvider):
            def get(self, agent_id: p.AgentId) -> p.PreambleConfiguration | None:
                return super().get(agent_id) or deployment_default

        container[p.PreambleConfigurationProvider] = DeploymentPreambleProvider()
        return container

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="preambles-agent-2",
            name="Deployment Default Agent",
            instructions="An agent that uses the deployment preamble default",
        )

    async def run(self, ctx: Context) -> None:
        configured = ctx.container[p.PreambleConfigurationProvider].get(self.agent.id)

        assert configured is self.deployment_default
