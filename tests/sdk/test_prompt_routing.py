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

from typing import Callable

import parlant.sdk as p
from parlant.core.nlp.prompt_routing import PromptConfigurationService

from tests.sdk.utils import Context, SDKTest


class _ServerWideBuilder(p.PromptBuilder):
    pass


class _AgentBuilder(p.PromptBuilder):
    pass


class Test_that_server_and_agent_prompt_routes_reach_the_configuration_service(SDKTest):
    async def create_server(self, port: int) -> tuple[p.Server, Callable[[], p.Container]]:
        server, get_container = await super().create_server(port)
        server.prompts = {"matching.*": _ServerWideBuilder}
        return server, get_container

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="prompt-routing-agent-1",
            name="Prompt Routing Agent",
            instructions="You are a helpful test agent.",
            prompts={"matching.rank": _AgentBuilder},
        )

    async def run(self, ctx: Context) -> None:
        service = ctx.container[PromptConfigurationService]

        # Agent scope beats the server-wide namespace wildcard...
        agent_builder = service.create_builder("matching.rank", agent_id=self.agent.id)
        assert type(agent_builder) is _AgentBuilder

        # ...which still covers this agent's other matching prompts...
        distill_builder = service.create_builder("matching.distill", agent_id=self.agent.id)
        assert type(distill_builder) is _ServerWideBuilder

        # ...and unconfigured routes fall back to the stock builder.
        compaction_builder = service.create_builder("compaction", agent_id=self.agent.id)
        assert type(compaction_builder) is p.PromptBuilder

        # The server scope alone (no agent) sees only the wildcard.
        server_builder = service.create_builder("matching.rank")
        assert type(server_builder) is _ServerWideBuilder


def test_that_the_sdk_aggregates_the_prompt_section_registries() -> None:
    # The aggregation namespace is the public registry — every compass
    # component's enum is reachable and enum-typed through it.
    for group in (
        "Ranker",
        "Pruner",
        "MatchingCommon",
        "Compacter",
        "Reviewer",
        "Responder",
    ):
        enum = getattr(p.PromptSections, group)
        members = list(enum)
        assert members, f"p.PromptSections.{group} is empty"
        assert all(isinstance(member, str) for member in members)

    assert p.PromptSections.Ranker.POLICY_CONTEXT.value == "ranker.policy-context"
    assert p.VETO is not None
