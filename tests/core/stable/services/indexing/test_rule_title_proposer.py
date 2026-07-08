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

from parlant.core.rules import RuleContent
from parlant.core.services.indexing.rule_title_proposer import RuleTitleProposer


def test_that_rule_title_proposer_formats_rule_content() -> None:
    proposer = object.__new__(RuleTitleProposer)

    assert proposer._format_rule(
        RuleContent(
            condition="the customer requests a refund",
            action="review refund eligibility",
            description="Only issue refunds according to the refund policy.",
        )
    ) == (
        "Condition: the customer requests a refund\n\n"
        "Action: review refund eligibility\n\n"
        "Description: Only issue refunds according to the refund policy."
    )


def test_that_rule_title_proposer_normalizes_generated_title() -> None:
    proposer = object.__new__(RuleTitleProposer)

    assert proposer._normalize_title("  `Refund Eligibility:`  ") == "Refund Eligibility"


def test_that_rule_title_proposer_includes_few_shot_examples() -> None:
    proposer = object.__new__(RuleTitleProposer)

    examples = proposer._format_examples()

    assert "Compensation Eligibility" in examples
    assert "Handling Lost Cards" in examples
    assert "Handling Calculations Reliably with the Calculation Tool" in examples
