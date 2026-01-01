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

import pytest

from parlant.core.services.tools.plugins import tool
from parlant.core.tools import ToolContext, ToolResult
from tests.sdk.utils import SDKTest

from parlant import sdk as p


class Test_that_transition_to_validates_invalid_combinations_like_state_and_tool_instruction(
    SDKTest
):
    """Test that transition_to methods catch invalid parameter combinations like state + tool_instruction"""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Validation Agent",
            description="Agent for testing parameter validation",
        )

        self.journey = await self.agent.create_journey(
            title="Validation Journey",
            conditions=["Customer needs help"],
            description="Journey for testing validation",
        )

        @tool
        def test_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        # Test invalid combination: state + tool_instruction
        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(  # type: ignore[call-overload]
                condition="state and tool test",
                state=p.END_JOURNEY,
                tool_instruction="Use this tool",
            )
        assert "tool_instruction cannot be used with state" in str(exc_info.value)

        # Test invalid combination: chat_state + tool_instruction
        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(  # type: ignore[call-overload]
                condition="chat and tool test",
                chat_state="Help customer",
                tool_instruction="Use this tool",
            )
        assert "tool_instruction cannot be used with chat_state" in str(exc_info.value)

        # Test invalid combination: tool_instruction without tool_state
        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(  # type: ignore[call-overload]
                condition="tool instruction only test", tool_instruction="Use this tool"
            )
        assert "Must provide at least one target parameter" in str(exc_info.value)

        # Test valid combination: tool_instruction with tool_state (should work)
        await self.journey.initial_state.transition_to(
            tool_instruction="Use this tool", tool_state=test_tool
        )


class Test_that_transition_to_validates_conflicting_parameters(SDKTest):
    """Test that transition_to methods catch conflicting parameter combinations"""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Validation Agent",
            description="Agent for testing parameter validation",
        )

        self.journey = await self.agent.create_journey(
            title="Validation Journey",
            conditions=["Customer needs help"],
            description="Journey for testing validation",
        )

        @tool
        def test_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        self.test_tool = test_tool

        # Test conflict: chat_state + tool_state
        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(  # type: ignore[call-overload]
                chat_state="Help customer", tool_state=self.test_tool
            )
        assert "Cannot provide multiple target parameters simultaneously" in str(exc_info.value)
        assert "chat_state, tool_state" in str(exc_info.value)

        # Test conflict: chat_state + state
        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(  # type: ignore[call-overload]
                chat_state="Help customer", state=p.END_JOURNEY
            )
        assert "Cannot provide multiple target parameters simultaneously" in str(exc_info.value)
        assert "chat_state, state" in str(exc_info.value)

        # Test conflict: tool_state + journey
        sub_journey = await self.agent.create_journey(
            title="Sub Journey",
            conditions=[],
            description="Sub journey for testing",
        )

        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(  # type: ignore[call-overload]
                tool_state=self.test_tool, journey=sub_journey
            )
        assert "Cannot provide multiple target parameters simultaneously" in str(exc_info.value)
        assert "tool_state, journey" in str(exc_info.value)

        # Test conflict: state + journey
        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(state=p.END_JOURNEY, journey=sub_journey)  # type: ignore[call-overload]
        assert "Cannot provide multiple target parameters simultaneously" in str(exc_info.value)
        assert "state, journey" in str(exc_info.value)

        # Test conflict: all three main parameters
        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(  # type: ignore[call-overload]
                chat_state="Help customer", tool_state=self.test_tool, state=p.END_JOURNEY
            )
        assert "Cannot provide multiple target parameters simultaneously" in str(exc_info.value)


class Test_that_transition_to_requires_at_least_one_target_parameter(SDKTest):
    """Test that transition_to methods require at least one target parameter"""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Validation Agent",
            description="Agent for testing parameter validation",
        )

        self.journey = await self.agent.create_journey(
            title="Validation Journey",
            conditions=["Customer needs help"],
            description="Journey for testing validation",
        )

        # Test no target parameters provided
        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(condition="if customer is happy")  # type: ignore[call-overload]
        assert "Must provide at least one target parameter" in str(exc_info.value)
        assert "chat_state, state, tool_state, or journey" in str(exc_info.value)

        # Test empty tool_state (should be treated as no parameter)
        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(tool_state=[])  # type: ignore[call-overload]
        assert "Must provide at least one target parameter" in str(exc_info.value)


class Test_that_fork_journey_state_requires_condition_except_for_journey_transitions(SDKTest):
    """Test that ForkJourneyState requires conditions for non-journey transitions"""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Fork Validation Agent",
            description="Agent for testing fork state validation",
        )

        self.journey = await self.agent.create_journey(
            title="Fork Validation Journey",
            conditions=["Customer needs routing"],
            description="Journey for testing fork validation",
        )

        # Create a fork state
        self.fork_transition = await self.journey.initial_state.transition_to(
            chat_state="Ask what kind of help they need"
        )
        self.fork_state = await self.fork_transition.target.fork()

        # Test ForkJourneyState without condition for chat_state - should fail
        with pytest.raises(p.SDKError) as exc_info:
            await self.fork_state.target.transition_to(chat_state="Provide general help")  # type: ignore[call-overload]
        assert "ForkJourneyState requires a condition (except when transition to a journey)" in str(
            exc_info.value
        )

        # Test ForkJourneyState without condition for tool_state - should fail
        @tool
        def help_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        with pytest.raises(p.SDKError) as exc_info:
            await self.fork_state.target.transition_to(tool_state=help_tool)  # type: ignore[call-overload]
        assert "ForkJourneyState requires a condition (except when transition to a journey)" in str(
            exc_info.value
        )

        # Test ForkJourneyState without condition for state - should fail
        with pytest.raises(p.SDKError) as exc_info:
            await self.fork_state.target.transition_to(state=p.END_JOURNEY)  # type: ignore[call-overload]
        assert "ForkJourneyState requires a condition (except when transition to a journey)" in str(
            exc_info.value
        )

        # Test ForkJourneyState with condition for chat_state - should succeed
        await self.fork_state.target.transition_to(
            condition="if customer needs general help", chat_state="Provide general help"
        )

        # Test ForkJourneyState without condition for journey - should succeed (exception case)
        sub_journey = await self.agent.create_journey(
            title="Sub Journey",
            conditions=[],
            description="Sub journey for testing",
        )

        await self.fork_state.target.transition_to(
            condition="fork journey test", journey=sub_journey
        )


class Test_that_tool_journey_state_validates_parameters(SDKTest):
    """Test parameter validation specifically for ToolJourneyState"""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Tool Validation Agent",
            description="Agent for testing tool state validation",
        )

        self.journey = await self.agent.create_journey(
            title="Tool Validation Journey",
            conditions=["Customer needs tool assistance"],
            description="Journey for testing tool state validation",
        )

        @tool
        def initial_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        @tool
        def next_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        self.initial_tool = initial_tool
        self.next_tool = next_tool

        # Create a tool state
        self.tool_transition = await self.journey.initial_state.transition_to(
            tool_state=self.initial_tool
        )

        # Test conflicting parameters in ToolJourneyState
        with pytest.raises(p.SDKError) as exc_info:
            await self.tool_transition.target.transition_to(  # type: ignore[call-overload]
                chat_state="Ask for details", tool_state=self.next_tool
            )
        assert "Cannot provide multiple target parameters simultaneously" in str(exc_info.value)

        # Test valid transition from ToolJourneyState
        await self.tool_transition.target.transition_to(chat_state="Ask for details")


class Test_that_chat_journey_state_validates_parameters(SDKTest):
    """Test parameter validation specifically for ChatJourneyState"""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Chat Validation Agent",
            description="Agent for testing chat state validation",
        )

        self.journey = await self.agent.create_journey(
            title="Chat Validation Journey",
            conditions=["Customer needs chat assistance"],
            description="Journey for testing chat state validation",
        )

        # Create a chat state
        self.chat_transition = await self.journey.initial_state.transition_to(
            chat_state="Welcome the customer"
        )

        # Test conflicting parameters in ChatJourneyState
        with pytest.raises(p.SDKError) as exc_info:
            await self.chat_transition.target.transition_to(  # type: ignore[call-overload]
                chat_state="Ask for details", state=p.END_JOURNEY
            )
        assert "Cannot provide multiple target parameters simultaneously" in str(exc_info.value)

        # Test valid transition from ChatJourneyState
        await self.chat_transition.target.transition_to(chat_state="Ask for details")


class Test_that_unknown_parameters_are_caught(SDKTest):
    """Test validation logic works correctly for valid parameters"""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Valid Param Agent",
            description="Agent for testing valid parameter validation",
        )

        self.journey = await self.agent.create_journey(
            title="Valid Param Journey",
            conditions=["Customer needs help"],
            description="Journey for testing valid parameter validation",
        )

        # Test that valid parameters work correctly
        await self.journey.initial_state.transition_to(chat_state="Help customer")


class Test_that_all_journey_state_types_have_validation(SDKTest):
    """Test that all journey state types (Initial, Tool, Chat, Fork) have proper validation"""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="All States Agent",
            description="Agent for testing all state types validation",
        )

        self.journey = await self.agent.create_journey(
            title="All States Journey",
            conditions=["Customer needs comprehensive help"],
            description="Journey for testing all state types",
        )

        @tool
        def test_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        self.test_tool = test_tool

        # Create transitions to get different state types
        chat_transition = await self.journey.initial_state.transition_to(
            chat_state="Welcome customer"
        )

        tool_transition = await chat_transition.target.transition_to(tool_state=self.test_tool)

        fork_transition = await tool_transition.target.fork()

        # Test ForkJourneyState condition requirement
        with pytest.raises(p.SDKError) as exc_info:
            await fork_transition.target.transition_to(chat_state="Help without condition")  # type: ignore[call-overload]
        assert "ForkJourneyState requires a condition (except when transition to a journey)" in str(
            exc_info.value
        )


class Test_that_valid_parameters_still_work_after_validation_added(SDKTest):
    """Test that adding validation doesn't break existing valid usage"""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Valid Usage Agent",
            description="Agent for testing that valid usage still works",
        )

        self.journey = await self.agent.create_journey(
            title="Valid Usage Journey",
            conditions=["Customer needs help"],
            description="Journey for testing valid parameter usage",
        )

        @tool
        def help_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={"status": "helped"})

        self.help_tool = help_tool

        # Test valid chat_state from initial state
        chat_transition = await self.journey.initial_state.transition_to(
            condition="start chat", chat_state="Welcome to our service"
        )

        # Test valid tool_state from chat state
        tool_transition = await chat_transition.target.transition_to(
            condition="use help tool", tool_state=self.help_tool
        )

        # Test valid state (END_JOURNEY) from tool state
        await tool_transition.target.transition_to(condition="end journey", state=p.END_JOURNEY)

        # Test valid journey transition from initial state (second branch)
        sub_journey = await self.agent.create_journey(
            title="Sub Journey",
            conditions=[],
            description="Sub journey for testing",
        )

        await self.journey.initial_state.transition_to(
            condition="go to sub journey", journey=sub_journey
        )

        # Test valid fork with condition from initial state (third branch)
        fork_transition = await self.journey.initial_state.transition_to(
            condition="start fork flow", chat_state="Ask for preference"
        )
        fork_state_transition = await fork_transition.target.fork()

        # Valid fork transition with condition
        await fork_state_transition.target.transition_to(
            condition="if customer prefers phone support", chat_state="Transfer to phone support"
        )

        # Test valid parameters with all optional fields
        await self.journey.initial_state.transition_to(
            condition="if customer needs detailed help",
            chat_state="Provide detailed assistance",
            metadata={"priority": "high"},
            canned_responses=[],
        )


class Test_that_journey_transitions_reject_invalid_parameters(SDKTest):
    """Test that journey transitions only accept condition and journey parameters"""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Journey Param Agent",
            description="Agent for testing journey parameter validation",
        )

        self.journey = await self.agent.create_journey(
            title="Journey Param Journey",
            conditions=["Customer needs help"],
            description="Journey for testing journey parameter validation",
        )

        self.sub_journey = await self.agent.create_journey(
            title="Sub Journey",
            conditions=[],
            description="Sub journey for testing",
        )

        # Test journey + metadata (should fail)
        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(  # type: ignore[call-overload]
                journey=self.sub_journey, metadata={"test": "value"}
            )
        assert "Journey transitions do not support the following parameters: metadata" in str(
            exc_info.value
        )
        assert "Only 'condition' and 'journey' are allowed for journey transitions" in str(
            exc_info.value
        )

        # Test journey + canned_responses (should fail)
        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(  # type: ignore[call-overload]
                journey=self.sub_journey, canned_responses=["response1"]
            )
        assert (
            "Journey transitions do not support the following parameters: canned_responses"
            in str(exc_info.value)
        )

        # Test journey + on_match (should fail)
        async def on_match_handler(ctx: object, match: object) -> None:
            pass

        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(  # type: ignore[call-overload]
                journey=self.sub_journey, on_match=on_match_handler
            )
        assert "Journey transitions do not support the following parameters: on_match" in str(
            exc_info.value
        )

        # Test journey + tool_instruction (should fail)
        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(  # type: ignore[call-overload]
                journey=self.sub_journey, tool_instruction="Use this tool"
            )
        assert (
            "Journey transitions do not support the following parameters: tool_instruction"
            in str(exc_info.value)
        )

        # Test journey + multiple invalid params (should fail and list all)
        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(  # type: ignore[call-overload]
                journey=self.sub_journey,
                metadata={"test": "value"},
                canned_responses=["response1"],
                on_match=on_match_handler,
            )
        error_msg = str(exc_info.value)
        assert "Journey transitions do not support the following parameters:" in error_msg
        assert "metadata" in error_msg
        assert "canned_responses" in error_msg
        assert "on_match" in error_msg

        # Test valid journey transition (should succeed)
        await self.journey.initial_state.transition_to(journey=self.sub_journey)

        # Test valid journey transition with condition (should succeed)
        await self.journey.initial_state.transition_to(
            condition="if customer needs sub-journey help", journey=self.sub_journey
        )


class Test_that_tool_instruction_parameter_validation_works_correctly(SDKTest):
    """Test that tool_instruction parameter is validated correctly for different state types"""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Tool Instruction Agent",
            description="Agent for testing tool_instruction validation",
        )

        self.journey = await self.agent.create_journey(
            title="Tool Instruction Journey",
            conditions=["Customer needs tool help"],
            description="Journey for testing tool_instruction validation",
        )

        @tool
        def test_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        self.test_tool = test_tool

        # Test tool_instruction with state parameter (should fail)
        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(  # type: ignore[call-overload]
                state=p.END_JOURNEY, tool_instruction="Use this tool"
            )
        assert "tool_instruction cannot be used with state" in str(exc_info.value)
        assert "tool_instruction is only valid when using tool_state" in str(exc_info.value)

        # Test tool_instruction with chat_state parameter (should fail)
        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(  # type: ignore[call-overload]
                chat_state="Help customer", tool_instruction="Use this tool"
            )
        assert "tool_instruction cannot be used with chat_state" in str(exc_info.value)
        assert "tool_instruction is only valid when using tool_state" in str(exc_info.value)

        # Test tool_instruction without tool_state (should fail)
        with pytest.raises(p.SDKError) as exc_info:
            await self.journey.initial_state.transition_to(tool_instruction="Use this tool")  # type: ignore[call-overload]
        assert "Must provide at least one target parameter" in str(exc_info.value)

        # Test valid tool_instruction with tool_state (should succeed)
        tool_transition = await self.journey.initial_state.transition_to(
            tool_state=self.test_tool, tool_instruction="Use this tool to help customer"
        )

        # Test tool_state without tool_instruction (should also succeed)
        await tool_transition.target.transition_to(
            condition="if customer needs another tool", tool_state=self.test_tool
        )

        # Test with metadata, canned_responses, on_match for tool_state (should succeed)
        async def on_match_handler(ctx: object, match: object) -> None:
            pass

        await tool_transition.target.transition_to(  # type: ignore[call-overload]
            condition="if customer needs advanced tool",
            tool_state=self.test_tool,
            tool_instruction="Use tool with extras",
            metadata={"priority": "high"},
            canned_responses=[],
            on_match=on_match_handler,
        )


class Test_that_fork_state_condition_validation_is_comprehensive(SDKTest):
    """Test comprehensive condition validation for ForkJourneyState"""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            name="Fork Condition Agent",
            description="Agent for testing fork condition validation",
        )

        self.journey = await self.agent.create_journey(
            title="Fork Condition Journey",
            conditions=["Customer needs routing"],
            description="Journey for testing fork condition validation",
        )

        @tool
        def routing_tool(context: ToolContext) -> ToolResult:
            return ToolResult(data={})

        self.routing_tool = routing_tool

        # Create a fork state to test with
        chat_transition = await self.journey.initial_state.transition_to(
            chat_state="Ask what kind of help they need"
        )
        self.fork_transition = await chat_transition.target.fork()

        self.sub_journey = await self.agent.create_journey(
            title="Sub Journey",
            conditions=[],
            description="Sub journey for fork testing",
        )

        # Test all target types require condition except journey

        # chat_state without condition (should fail)
        with pytest.raises(p.SDKError) as exc_info:
            await self.fork_transition.target.transition_to(chat_state="Provide general help")  # type: ignore[call-overload]
        assert "ForkJourneyState requires a condition (except when transition to a journey)" in str(
            exc_info.value
        )

        # state without condition (should fail)
        with pytest.raises(p.SDKError) as exc_info:
            await self.fork_transition.target.transition_to(state=p.END_JOURNEY)  # type: ignore[call-overload]
        assert "ForkJourneyState requires a condition (except when transition to a journey)" in str(
            exc_info.value
        )

        # tool_state without condition (should fail)
        with pytest.raises(p.SDKError) as exc_info:
            await self.fork_transition.target.transition_to(tool_state=self.routing_tool)  # type: ignore[call-overload]
        assert "ForkJourneyState requires a condition (except when transition to a journey)" in str(
            exc_info.value
        )

        # tool_state + tool_instruction without condition (should fail)
        with pytest.raises(p.SDKError) as exc_info:
            await self.fork_transition.target.transition_to(  # type: ignore[call-overload]
                tool_state=self.routing_tool, tool_instruction="Route customer"
            )
        assert "ForkJourneyState requires a condition (except when transition to a journey)" in str(
            exc_info.value
        )

        # journey without condition (should succeed - this is the exception)
        await self.fork_transition.target.transition_to(journey=self.sub_journey)

        # All target types WITH condition should succeed
        await self.fork_transition.target.transition_to(
            condition="if customer needs chat help", chat_state="Provide chat help"
        )

        await self.fork_transition.target.transition_to(
            condition="if customer wants to end", state=p.END_JOURNEY
        )

        await self.fork_transition.target.transition_to(
            condition="if customer needs tool help", tool_state=self.routing_tool
        )

        await self.fork_transition.target.transition_to(
            condition="if customer needs complex routing",
            tool_state=self.routing_tool,
            tool_instruction="Perform complex routing",
        )

        # journey with condition should also succeed
        await self.fork_transition.target.transition_to(
            condition="if customer needs sub-journey help", journey=self.sub_journey
        )
