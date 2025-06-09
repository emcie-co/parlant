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

from datetime import datetime, timezone
import json
import os
from lagom import Container
import pytest
from pytest import fixture

from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.customers import Customer, CustomerId, CustomerStore
from parlant.core.sessions import Session, SessionStore

from tests.conftest import SERVICE_NAME
from tests.core.common.utils import ContextOfTest
from tests.test_utilities import SyncAwaiter

from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatcher,
)


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
        guideline_matches=dict(),
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

JSONL_LOG_PATH = "parlant_test_results.jsonl"
test_data = {}


def get_current_test():
    """Helper to find the currently running test"""
    for nodeid in test_data:
        if test_data[nodeid]["status"] == "running":
            return nodeid
    return None


def pytest_configure(config):
    if not os.path.exists(JSONL_LOG_PATH):
        open(JSONL_LOG_PATH, "w").close()


@pytest.hookimpl(tryfirst=True)  # type: ignore
def pytest_runtest_setup(item):
    """Initialize data collection for the test"""
    test_data[item.nodeid] = {
        "test_name": item.nodeid,
        "model_name": SERVICE_NAME,
        "timestamp": datetime.now().isoformat(),
        "guideline_matcher_responses": [],
        "guideline_matcher_tokens": [],
        "guideline_matcher_duration": -1.0,
        "status": "setup",
        "passed": False,
    }


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    """Capture test result information"""
    if report.when == "call":
        for nodeid in test_data:
            if nodeid == report.nodeid:
                # Update the result field
                if report.passed:
                    test_data[nodeid]["result"] = "passed"
                elif report.failed:
                    test_data[nodeid]["result"] = "failed"
                elif report.skipped:
                    test_data[nodeid]["result"] = "skipped"
                else:
                    test_data[nodeid]["result"] = "unknown"
                break


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    if item.nodeid not in test_data:
        return

    data_to_write = {
        "test_name": test_data[item.nodeid]["test_name"],
        "model_name": test_data[item.nodeid]["model_name"],
        "timestamp": test_data[item.nodeid]["timestamp"],
        "guideline_matcher_responses": test_data[item.nodeid]["guideline_matcher_responses"],
        "guideline_matcher_tokens": test_data[item.nodeid]["guideline_matcher_tokens"],
        "status": test_data[item.nodeid]["status"],
        "duration": test_data[item.nodeid]["guideline_matcher_duration"],
        "result": test_data[item.nodeid].get("result", "unknown"),
    }

    with open(JSONL_LOG_PATH, "a") as jsonl_file:
        json.dump(data_to_write, jsonl_file)
        jsonl_file.write("\n")

    del test_data[item.nodeid]


def extract_token_count(info):
    """Helper to extract token count from GenerationInfo"""
    if hasattr(info, "usage") and info.usage:
        if hasattr(info.usage, "output_tokens"):
            return info.usage.output_tokens
    return None


def capture_guideline_matching(original_function):
    """Decorator to capture guideline matcher output"""

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
            if hasattr(result, "batch_generations") and len(result.batch_generations) > 0:
                for generation_info in result.batch_generations:
                    token_count = extract_token_count(generation_info)
                    if token_count is not None:
                        token_counts.append(token_count)

            # Add to test data
            if propositions:
                test_data[current_test]["guideline_matcher_responses"].append(propositions)
            if token_counts:
                test_data[current_test]["guideline_matcher_tokens"].extend(token_counts)
            if result:
                test_data[current_test]["guideline_matcher_duration"] = result.total_duration

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

    if not hasattr(GuidelineMatcher, "_original_match_guidelines"):
        GuidelineMatcher._original_match_guidelines = GuidelineMatcher.match_guidelines  # type: ignore
        GuidelineMatcher.match_guidelines = capture_guideline_matching(
            GuidelineMatcher._original_match_guidelines  # type: ignore
        )

    return None


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Restore original functions when pytest session is done"""
    if hasattr(GuidelineMatcher, "_original_propose_guidelines"):
        GuidelineMatcher.match_guidelines = GuidelineMatcher._original_propose_guidelines  # type: ignore
        delattr(GuidelineMatcher, "_original_propose_guidelines")


print("\n*** conftest.py is being loaded! ***\n")
print(
    f"Available pytest hooks in this file: {[name for name in globals() if name.startswith('pytest_')]}"
)
