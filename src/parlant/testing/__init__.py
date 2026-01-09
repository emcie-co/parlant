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

"""Parlant Testing Framework.

A comprehensive testing framework for Parlant agents.

Example:
    from parlant.testing import Suite, CustomerMessage, AgentMessage
    from parlant.sdk import NLPServices

    suite = Suite(
        server_url="http://localhost:8000",
        nlp_service=NLPServices.emcie,
        agent_id="my_agent",
    )

    @suite.scenario
    async def test_greeting():
        async with suite.session() as session:
            response = await session.send("Hello!")
            await response.should("greet the customer")

    @suite.scenario(repetitions=3)
    async def test_conversation():
        async with suite.session() as session:
            await session.unfold([
                CustomerMessage(message="Hello"),
                AgentMessage(ideal="Hi!", should="greet the customer"),
            ])
"""

from parlant.testing.builder import InteractionBuilder, PrefabEvent
from parlant.testing.response import Response, ToolCall
from parlant.testing.runner import TestReport, TestResult, TestRunner, TestStatus
from parlant.testing.session import Session
from parlant.testing.steps import AgentMessage, CustomerMessage, StatusEvent, Step
from parlant.testing.suite import HookSet, Scenario, Suite

__all__ = [
    # Core classes
    "Suite",
    "Session",
    "Response",
    # Step definitions
    "CustomerMessage",
    "AgentMessage",
    "StatusEvent",
    "Step",
    # Builder
    "InteractionBuilder",
    "PrefabEvent",
    # Runner
    "TestRunner",
    "TestReport",
    "TestResult",
    "TestStatus",
    # Supporting types
    "ToolCall",
    "Scenario",
    "HookSet",
]
