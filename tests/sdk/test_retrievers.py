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

import parlant.sdk as p
from tests.sdk.utils import Context, SDKTest
from tests.test_utilities import nlp_test


class Test_that_a_custom_retriever_can_be_used_to_add_data_to_message_context(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Rel Agent",
            description="Agent for guideline relationships",
        )

        async def custom_retriever(ctx: p.RetrieverContext) -> str:
            assert ctx.interaction.last_customer_message is not None
            assert ctx.interaction.last_customer_message.content == "What is an orange eggplant?"
            return "An orange eggplant is actually a special type of tomato"

        await self.agent.attach_retriever(custom_retriever)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive(
            customer_message="What is an orange eggplant?",
            recipient=self.agent,
        )

        assert nlp_test(
            context=response,
            condition="It says that an orange  eggplant is a type of tomato",
        )
