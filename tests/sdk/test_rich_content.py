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


_ATTACHMENT = {
    "url": "https://cdn.example.com/u/9f3a.jpg",
    "media_type": "image/jpeg",
    "caption": "a photo of a cracked blue ceramic vase",
    "filename": "vase.jpg",
}


class Test_that_a_tool_reads_attachments_from_the_latest_customer_message(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.seen_urls: list[str] = []

        self.agent = await server.create_agent(
            id="rich-content-agent-1",
            name="Attachment Agent",
            instructions="Help the customer with files they attach.",
        )

        @p.tool
        async def inspect_attachment(context: p.ToolContext) -> p.ToolResult:
            last = p.Session.current.interaction.last_customer_message
            if last is not None:
                self.seen_urls.extend(a["url"] for a in last.attachments)
            return p.ToolResult(data="checked the attached file")

        await self.agent.create_observation(
            condition="the customer asks to check or handle a file they attached",
            tools=[inspect_attachment],
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            customer_message="Please check the file I attached.",
            recipient=self.agent,
            attachments=[_ATTACHMENT],
        )

        assert _ATTACHMENT["url"] in self.seen_urls, (
            f"Expected the tool to read the attachment URL, but saw: {self.seen_urls}"
        )


class Test_that_an_attachment_caption_is_visible_to_the_agent(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="rich-content-agent-2",
            name="Attachment Agent",
            instructions="Answer the customer's questions about what they send you.",
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="What did I just send you? Answer in one short sentence.",
            recipient=self.agent,
            attachments=[_ATTACHMENT],
        )

        assert "vase" in response.lower(), (
            f"Expected the agent to reference the attachment caption, but got: {response}"
        )
