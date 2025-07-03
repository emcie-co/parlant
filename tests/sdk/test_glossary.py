# Copyright 2025 Emcie Co Ltd.
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

from parlant.core.glossary import GlossaryStore
import parlant.sdk as p
from tests.sdk.utils import Context, SDKTest


class Test_that_a_glossary_term_can_be_created(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rel Agent",
            description="Agent for guideline relationships",
        )

        self.term = await self.agent.create_term(
            name="Priority",
            description="Indicates something should be prioritized over another.",
            synonyms=["importance", "precedence"],
        )

    async def run(self, ctx: Context) -> None:
        glossary_store = ctx.container[GlossaryStore]

        term = await glossary_store.read_term(self.term.id)
        assert term.name == "Priority"
        assert term.description == "Indicates something should be prioritized over another."
        assert term.synonyms == ["importance", "precedence"]
        assert term.id == self.term.id
