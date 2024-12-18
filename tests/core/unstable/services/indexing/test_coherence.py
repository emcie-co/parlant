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

from parlant.core.agents import Agent
from parlant.core.guidelines import GuidelineContent
from parlant.core.services.indexing.coherence_checker import (
    CoherenceChecker,
    IncoherenceKind,
)

from tests.core.common.conftest import agent
from tests.core.common.services.indexing.conftest import (
    _CustomerTestContext,
    customer_context,
    agent_id,
)


def test_that_guidelines_with_many_incoherencies_are_detected(
    customer_context: _CustomerTestContext,
    agent: Agent,
) -> None:
    guidelines_with_incoherencies: list[GuidelineContent] = []

    for guideline_params in [
        {
            "condition": "Asked for UPS shipping",
            "action": "Send UPS a request and inform the client that the order would arrive in 5 business days",
        },
        {
            "condition": "Asked for express shipping",
            "action": "register the order and inform the client that the order would arrive tomorrow",
        },
        {
            "condition": "Asked for delayed shipping",
            "action": "Send UPS the request and inform the client that the order would arrive in the next calendar year",
        },
        {
            "condition": "The client stops responding mid-order",
            "action": "save the client's order in their cart and notify them that it has not been shipped yet",
        },
        {
            "condition": "The client refuses to give their address",
            "action": "cancel the order and notify the client",
        },
    ]:
        guidelines_with_incoherencies.append(
            GuidelineContent(
                condition=guideline_params["condition"], action=guideline_params["action"]
            )
        )

    coherence_checker = customer_context.container[CoherenceChecker]

    incoherence_results = list(
        customer_context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                guidelines_with_incoherencies,
            )
        )
    )

    n = len(guidelines_with_incoherencies)
    assert len(incoherence_results) == n * (n - 1) // 2

    # Tests that there's exactly 1 case of incoherence per pair of guidelines
    expected_pairs = {
        (g1, g2)
        for i, g1 in enumerate(guidelines_with_incoherencies)
        for g2 in guidelines_with_incoherencies[i + 1 :]
    }

    for c in incoherence_results:
        assert (c.guideline_a, c.guideline_b) in expected_pairs
        assert c.IncoherenceKind == IncoherenceKind.CONTINGENT
        expected_pairs.remove((c.guideline_a, c.guideline_b))

    assert len(expected_pairs) == 0
