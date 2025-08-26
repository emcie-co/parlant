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

from typing import Sequence

from parlant.core.agents import Agent
from parlant.core.guidelines import GuidelineContent
from parlant.core.services.indexing.guideline_connection_proposer import GuidelineConnectionProposer

from tests.core.common.utils import ContextOfTest


def test_that_entailment_due_to_the_sources_condition_is_detected(  # This test fails occasionally
    context: ContextOfTest,
    agent: Agent,
) -> None:
    connection_proposer = context.container[GuidelineConnectionProposer]

    source_guideline_content = GuidelineContent(
        condition="Planning trips to Brazil",
        action="Check if there are any festivals happening on the relevant days, and suggest them to the customer if they coincide with their plans",
    )
    target_guideline_content = GuidelineContent(
        condition="Suggesting activites in a non-English speaking country",
        action="Ask the customer if they speak the local language",
    )
    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(
                agent,
                [source_guideline_content, target_guideline_content],
            )
        )
    )
    assert len(connection_propositions) == 1
    assert connection_propositions[0].source == source_guideline_content
    assert connection_propositions[0].target == target_guideline_content


def test_that_connection_is_proposed_for_a_sequence_where_each_guideline_entails_the_next_one_2(
    context: ContextOfTest,
    agent: Agent,
) -> None:
    introduced_guidelines: Sequence[GuidelineContent] = [
        GuidelineContent(condition=i["condition"], action=i["action"])
        for i in [
            {
                "condition": "discussing sandwiches",
                "action": "recommend the daily specials",
            },
            {
                "condition": "listing the daily specials",
                "action": "consider mentioning ingredients that may cause allergic reactions",
            },
            {
                "condition": "discussing anything related to a food allergies",
                "action": "you may note that all dishes may contain peanut residues",
            },
        ]
    ]

    connection_proposer = context.container[GuidelineConnectionProposer]

    connection_propositions = list(
        context.sync_await(
            connection_proposer.propose_connections(agent, introduced_guidelines, [])
        )
    )

    assert len(connection_propositions) == 2
    assert connection_propositions[0].source == introduced_guidelines[0]
    assert connection_propositions[0].target == introduced_guidelines[1]
    assert connection_propositions[1].source == introduced_guidelines[1]
    assert connection_propositions[1].target == introduced_guidelines[2]
