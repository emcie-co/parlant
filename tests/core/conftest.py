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

from datetime import datetime, timezone
from lagom import Container
import pytest
from pytest import fixture

from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.customers import Customer, CustomerId, CustomerStore
from parlant.core.sessions import Session, SessionStore
from tests.core.common.utils import ContextOfTest
from parlant.core.engines.alpha.reasoning_method import (
    TOOL_CALLER_REASONING_METHOD,
    GUIDELINE_PROPOSER_REASONING_METHOD,
    MESSAGE_GENERATOR_REASONING_METHOD,
)
from tests.test_utilities import SyncAwaiter
import csv
import os
import json


@fixture
def agent(
    container: Container,
    sync_await: SyncAwaiter,
) -> Agent:
    store = container[AgentStore]
    agent = sync_await(store.create_agent(name="test-agent", max_engine_iterations=2))
    return agent


@fixture
def agent_id(
    agent: Agent,
) -> AgentId:
    return agent.id


@fixture
def customer(context: ContextOfTest) -> Customer:
    store = context.container[CustomerStore]
    customer = context.sync_await(
        store.create_customer(
            name="Test Customer",
            extra={"email": "test@customer.com"},
        ),
    )
    return customer


@fixture
def customer_id(customer: Customer) -> CustomerId:
    return customer.id


@fixture
def context(
    sync_await: SyncAwaiter,
    container: Container,
) -> ContextOfTest:
    return ContextOfTest(
        sync_await,
        container,
        events=list(),
        guidelines=dict(),
        guideline_propositions=dict(),
        tools=dict(),
        actions=list(),
    )


@fixture
def new_session(
    context: ContextOfTest,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> Session:
    store = context.container[SessionStore]
    utc_now = datetime.now(timezone.utc)
    return context.sync_await(
        store.create_session(
            creation_utc=utc_now,
            customer_id=customer_id,
            agent_id=agent_id,
        )
    )


# pytest hooks and logging

CSV_LOG_PATH = "parlant_test_results.csv"
test_data = {}


def get_current_test():
    """Helper to find the currently running test"""
    for nodeid in test_data:
        if test_data[nodeid]["status"] == "running":
            return nodeid
    return None


def pytest_configure(config):
    """Set up the CSV file with headers when pytest starts"""
    # Create file with headers if it doesn't exist
    if not os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "test_name",
                    "timestamp",
                    "message_generator_mode" "message_generator_responses",
                    "message_generator_tokens",
                    "tool_caller_mode",
                    "tool_caller_responses",
                    "tool_caller_tokens",
                    "guideline_proposer_mode",
                    "guideline_proposer_responses",
                    "guideline_proposer_tokens",
                    "status",
                ]
            )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Initialize data collection for the test"""
    test_data[item.nodeid] = {
        "test_name": item.nodeid,
        "timestamp": datetime.now().isoformat(),
        "message_generator_responses": [],
        "message_generator_tokens": [],
        "tool_caller_responses": [],
        "tool_caller_tokens": [],
        "guideline_proposer_responses": [],
        "guideline_proposer_tokens": [],
        "status": "setup",
    }


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    """Record the test data to CSV after test completes"""
    if item.nodeid not in test_data:
        return

    with open(CSV_LOG_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        data = test_data[item.nodeid]

        # Convert lists to JSON strings for storage
        writer.writerow(
            [
                data["test_name"],
                data["timestamp"],
                MESSAGE_GENERATOR_REASONING_METHOD,
                json.dumps(data["message_generator_responses"]),
                json.dumps(data["message_generator_tokens"]),
                TOOL_CALLER_REASONING_METHOD,
                json.dumps(data["tool_caller_responses"]),
                json.dumps(data["tool_caller_tokens"]),
                GUIDELINE_PROPOSER_REASONING_METHOD,
                json.dumps(data["guideline_proposer_responses"]),
                json.dumps(data["guideline_proposer_tokens"]),
                data["status"],
            ]
        )

    # Clean up
    del test_data[item.nodeid]


def extract_token_count(info):
    """Helper to extract token count from GenerationInfo"""
    if hasattr(info, "usage") and info.usage:
        if hasattr(info.usage, "output_tokens"):
            return info.usage.output_tokens
    return None


def capture_fluid_message_generator(original_function):
    """Decorator to capture fluid message generator output"""

    async def wrapper(self, *args, **kwargs):
        result = await original_function(self, *args, **kwargs)

        current_test = get_current_test()
        if current_test and result:
            # Extract message content from the message compositions
            messages = []
            token_counts = []

            for composition in result:
                # Extract messages
                if hasattr(composition, "events") and composition.events:
                    for event in composition.events:
                        if event.kind == "message" and "message" in event.data:
                            messages.append(event.data["message"])

                # Extract token info
                if hasattr(composition, "info"):
                    token_count = extract_token_count(composition.info)
                    if token_count is not None:
                        token_counts.append(token_count)

            # Add to test data
            if messages:
                test_data[current_test]["message_generator_responses"].extend(messages)
            if token_counts:
                test_data[current_test]["message_generator_tokens"].extend(token_counts)

        return result

    return wrapper


def capture_tool_caller(original_function):
    """Decorator to capture tool caller output"""

    async def wrapper(self, *args, **kwargs):
        result = await original_function(self, *args, **kwargs)

        current_test = get_current_test()
        if current_test and result:
            # Extract tool call details
            tool_calls = []
            token_counts = []

            # Extract tool calls from batches
            if hasattr(result, "batches"):
                for batch in result.batches:
                    batch_calls = []
                    for tool_call in batch:
                        if hasattr(tool_call, "tool_id") and hasattr(tool_call, "arguments"):
                            batch_calls.append(
                                {
                                    "tool_id": str(tool_call.tool_id),
                                    "arguments": tool_call.arguments,
                                }
                            )
                    if batch_calls:
                        tool_calls.append(batch_calls)

            # Extract token info
            if hasattr(result, "batch_generations"):
                for generation_info in result.batch_generations:
                    token_count = extract_token_count(generation_info)
                    if token_count is not None:
                        token_counts.append(token_count)

            # Add to test data
            if tool_calls:
                test_data[current_test]["tool_caller_responses"].append(tool_calls)
            if token_counts:
                test_data[current_test]["tool_caller_tokens"].extend(token_counts)

        return result

    return wrapper


def capture_guideline_proposer(original_function):
    """Decorator to capture guideline proposer output"""

    async def wrapper(self, *args, **kwargs):
        result = await original_function(self, *args, **kwargs)

        current_test = get_current_test()
        if current_test and result:
            # Extract guideline propositions
            propositions = []
            token_counts = []

            # Process propositions from batches
            if hasattr(result, "batches"):
                for batch in result.batches:
                    batch_props = []
                    for prop in batch:
                        if hasattr(prop, "guideline") and hasattr(prop, "score"):
                            batch_props.append(
                                {
                                    "guideline_id": str(prop.guideline.id),
                                    "condition": prop.guideline.content.condition,
                                    "action": prop.guideline.content.action,
                                    "score": prop.score,
                                    "rationale": prop.rationale
                                    if hasattr(prop, "rationale")
                                    else None,
                                }
                            )
                    if batch_props:
                        propositions.append(batch_props)

            # Extract token info
            if hasattr(result, "batch_generations"):
                for generation_info in result.batch_generations:
                    token_count = extract_token_count(generation_info)
                    if token_count is not None:
                        token_counts.append(token_count)

            # Add to test data
            if propositions:
                test_data[current_test]["guideline_proposer_responses"].append(propositions)
            if token_counts:
                test_data[current_test]["guideline_proposer_tokens"].extend(token_counts)

        return result

    return wrapper


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    """Mark the test as running for data capture"""
    if item.nodeid in test_data:
        test_data[item.nodeid]["status"] = "running"


@pytest.hookimpl()
def pytest_runtest_protocol(item, nextitem):
    """Apply the monkey patches before running tests"""
    from parlant.core.engines.alpha.fluid_message_generator import FluidMessageGenerator
    from parlant.core.engines.alpha.tool_caller import ToolCaller
    from parlant.core.engines.alpha.guideline_proposer import GuidelineProposer

    # Patch FluidMessageGenerator
    if not hasattr(FluidMessageGenerator, "_original_generate_events"):
        FluidMessageGenerator._original_generate_events = FluidMessageGenerator.generate_events
        FluidMessageGenerator.generate_events = capture_fluid_message_generator(
            FluidMessageGenerator._original_generate_events
        )

    # Patch ToolCaller
    if not hasattr(ToolCaller, "_original_infer_tool_calls"):
        ToolCaller._original_infer_tool_calls = ToolCaller.infer_tool_calls
        ToolCaller.infer_tool_calls = capture_tool_caller(ToolCaller._original_infer_tool_calls)

    # Patch GuidelineProposer
    if not hasattr(GuidelineProposer, "_original_propose_guidelines"):
        GuidelineProposer._original_propose_guidelines = GuidelineProposer.propose_guidelines
        GuidelineProposer.propose_guidelines = capture_guideline_proposer(
            GuidelineProposer._original_propose_guidelines
        )

    # Continue with normal test execution
    return None


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Restore original functions when pytest session is done"""
    from parlant.core.engines.alpha.fluid_message_generator import FluidMessageGenerator
    from parlant.core.engines.alpha.tool_caller import ToolCaller
    from parlant.core.engines.alpha.guideline_proposer import GuidelineProposer

    # Restore original methods
    if hasattr(FluidMessageGenerator, "_original_generate_events"):
        FluidMessageGenerator.generate_events = FluidMessageGenerator._original_generate_events
        delattr(FluidMessageGenerator, "_original_generate_events")

    if hasattr(ToolCaller, "_original_infer_tool_calls"):
        ToolCaller.infer_tool_calls = ToolCaller._original_infer_tool_calls
        delattr(ToolCaller, "_original_infer_tool_calls")

    if hasattr(GuidelineProposer, "_original_propose_guidelines"):
        GuidelineProposer.propose_guidelines = GuidelineProposer._original_propose_guidelines
        delattr(GuidelineProposer, "_original_propose_guidelines")


print("\n*** conftest.py is being loaded! ***\n")
print(
    f"Available pytest hooks in this file: {[name for name in globals() if name.startswith('pytest_')]}"
)
