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
from tests.test_utilities import nlp_test


class Test_that_a_custom_retriever_can_be_used_to_add_data_to_message_context(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy agent",
            description="Dummy agent",
        )

        async def custom_retriever(ctx: p.RetrieverContext) -> p.RetrieverResult:
            assert ctx.interaction.last_customer_message is not None
            assert ctx.interaction.last_customer_message.content == "What is an orange eggplant?"
            return p.RetrieverResult(data="An orange eggplant is actually a special type of tomato")

        await self.agent.attach_retriever(custom_retriever)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="What is an orange eggplant?",
            recipient=self.agent,
        )

        assert await nlp_test(
            context=response,
            condition="It says that an orange  eggplant is a type of tomato",
        )


class Test_that_multiple_custom_retrievers_can_be_used_to_add_data_to_message_context(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy agent",
            description="Dummy agent",
        )

        async def custom_retriever_1(ctx: p.RetrieverContext) -> p.RetrieverResult:
            return p.RetrieverResult(data="An orange eggplant is actually a special type of tomato")

        async def custom_retriever_2(ctx: p.RetrieverContext) -> p.RetrieverResult:
            return p.RetrieverResult(data="Parla loves orange eggplants")

        await self.agent.attach_retriever(custom_retriever_1)
        await self.agent.attach_retriever(custom_retriever_2)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="What's the name of he/she who is known to love tomatoes?",
            recipient=self.agent,
        )

        assert await nlp_test(
            context=response,
            condition="It mentions the name Parla",
        )


class Test_that_a_retriever_can_return_a_canned_response(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy agent",
            description="Dummy agent",
            composition_mode=p.CompositionMode.STRICT,
        )

        async def custom_retriever(ctx: p.RetrieverContext) -> p.RetrieverResult:
            return p.RetrieverResult(
                data="Hello", canned_responses=["Howdy Junior! How can I help?"]
            )

        await self.agent.attach_retriever(custom_retriever)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Hello",
            recipient=self.agent,
        )

        assert response == "Howdy Junior! How can I help?"


class Test_that_retriever_can_return_direct_result_immediately(SDKTest):
    """Test that existing behavior still works - retriever returns RetrieverResult directly."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy agent",
            description="Dummy agent",
        )

        async def custom_retriever(ctx: p.RetrieverContext) -> p.RetrieverResult:
            return p.RetrieverResult(
                data="Direct result: An orange eggplant is a tomato",
            )

        await self.agent.attach_retriever(custom_retriever)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="What is an orange eggplant?",
            recipient=self.agent,
        )

        assert await nlp_test(
            context=response,
            condition="It mentions that an orange eggplant is a tomato",
        )


class Test_that_retriever_can_return_deferred_callable_that_receives_engine_context(SDKTest):
    """Test that retriever can return a deferred callable which is called with EngineContext."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy agent",
            description="Dummy agent",
        )

        self.deferred_was_called = False
        self.engine_context_received = False

        async def custom_retriever(ctx: p.RetrieverContext) -> p.DeferredRetriever:
            # This runs during on_acknowledged
            async def deferred(engine_ctx: p.EngineContext) -> p.RetrieverResult:
                # This runs during on_generating_messages
                self.deferred_was_called = True
                self.engine_context_received = engine_ctx is not None
                return p.RetrieverResult(
                    data="Deferred result: A purple tomato is an eggplant",
                )

            return deferred

        await self.agent.attach_retriever(custom_retriever)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="What is a purple tomato?",
            recipient=self.agent,
        )

        assert self.deferred_was_called, "Deferred callable was not called"
        assert self.engine_context_received, "EngineContext was not received"

        assert await nlp_test(
            context=response,
            condition="It mentions that a purple tomato is an eggplant",
        )


class Test_that_deferred_retriever_receives_updated_engine_context_with_guidelines_and_tools(
    SDKTest
):
    """Test that the deferred callable receives the full EngineContext from on_generating_messages."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy agent",
            description="Dummy agent",
        )

        # Add a guideline that should be matched
        self.observation = await self.agent.create_observation(
            condition="the customer asks about Chongas",
        )

        async def custom_retriever(ctx: p.RetrieverContext) -> p.DeferredRetriever:
            async def deferred(engine_ctx: p.EngineContext) -> p.RetrieverResult:
                assert engine_ctx.state is not None

                assert len(engine_ctx.state.guidelines) == 1

                if engine_ctx.state.guidelines[0].id == self.observation.id:
                    return p.RetrieverResult(
                        data="Chongas are a tropical island fruit",
                    )
                else:
                    return p.RetrieverResult(None)

            return deferred

        await self.agent.attach_retriever(custom_retriever)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="What are chongas?",
            recipient=self.agent,
        )

        assert await nlp_test(
            context=response,
            condition="It says chongas are a fruit",
        )


class Test_that_deferred_retriever_can_return_none_based_on_engine_context(SDKTest):
    """Test that deferred callable can inspect engine context and return None."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Dummy agent",
            description="Dummy agent",
        )

        self.deferred_returned_none = False

        async def custom_retriever(ctx: p.RetrieverContext) -> p.DeferredRetriever:
            async def deferred(engine_ctx: p.EngineContext) -> p.RetrieverResult | None:
                # Simulate logic that decides not to return data based on context
                # For this test, we always return None
                self.deferred_returned_none = True
                return None

            return deferred

        await self.agent.attach_retriever(custom_retriever)

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            customer_message="Hello there",
            recipient=self.agent,
        )

        assert self.deferred_returned_none, "Deferred callable did not return None as expected"
        # The agent should still respond, just without retriever data
        assert len(response) > 0
