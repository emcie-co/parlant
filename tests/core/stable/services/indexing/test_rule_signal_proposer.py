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

from parlant.core.services.indexing.rule_signal_proposer import RuleSignalProposer


def test_that_rule_signal_proposer_formats_full_rule() -> None:
    proposer = object.__new__(RuleSignalProposer)

    assert proposer._format_rule(
        "Refunds",
        content="Use the current refund rule.",
        condition="the customer asks for a refund",
    ) == (
        "# Refunds\n\n## Condition: the customer asks for a refund\n\nUse the current refund rule."
    )


def test_that_rule_signal_proposer_formats_partial_rules() -> None:
    proposer = object.__new__(RuleSignalProposer)

    assert (
        proposer._format_rule(
            "Observation",
            content=None,
            condition="the customer reports a lost card",
        )
        == "# Observation\n\n## Condition: the customer reports a lost card"
    )

    assert (
        proposer._format_rule(
            "Description Only",
            content="Speak concisely.",
            condition=None,
        )
        == "# Description Only\n\nSpeak concisely."
    )

    assert (
        proposer._format_rule(
            "Title Only",
            content=None,
            condition=None,
        )
        == "# Title Only"
    )
