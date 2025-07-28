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

from parlant.core.agents import Agent
from parlant.core.guidelines import GuidelineContent
from parlant.core.services.indexing.coherence_checker import (
    CoherenceChecker,
    IncoherenceKind,
)

from tests.core.common.utils import ContextOfTest
from tests.core.stable.services.indexing.test_coherence import incoherence_nlp_test


def test_that_guidelines_with_many_incoherencies_are_detected(
    context: ContextOfTest,
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

    coherence_checker = context.container[CoherenceChecker]

    incoherence_results = list(
        context.sync_await(
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


def test_that_contradictory_next_message_commands_are_detected_as_incoherencies(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    coherence_checker = context.container[CoherenceChecker]
    guidelines_to_evaluate = [
        GuidelineContent(
            condition="a document is being discussed",
            action="provide the full document to the customer",
        ),
        GuidelineContent(
            condition="a client asks to summarize a document",
            action="provide a summary of the document in 100 words or less",
        ),
        GuidelineContent(
            condition="the client asks for a summary of another customer's medical document",
            action="refuse to share the document or its summary",
        ),
    ]

    incoherence_results = list(
        context.sync_await(
            coherence_checker.propose_incoherencies(
                agent,
                guidelines_to_evaluate,
            )
        )
    )

    assert len(incoherence_results) == 3
    incoherencies_to_detect = {
        (guideline_a, guideline_b)
        for i, guideline_a in enumerate(guidelines_to_evaluate)
        for guideline_b in guidelines_to_evaluate[i + 1 :]
    }
    for c in incoherence_results:
        assert (c.guideline_a, c.guideline_b) in incoherencies_to_detect
        assert c.IncoherenceKind == IncoherenceKind.STRICT
        incoherencies_to_detect.remove((c.guideline_a, c.guideline_b))
    assert len(incoherencies_to_detect) == 0

    for incoherence in incoherence_results:
        assert context.sync_await(incoherence_nlp_test(context, agent, incoherence))
