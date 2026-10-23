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

"""Procedural-policy catalog for the compass BDD suites.

Alpha expressed multi-step flows as journey node graphs; compass expresses
them as procedural policies - one policy whose content walks the flow (here
as mermaid flowcharts, per docs/policy-design.md). The catalog keys keep
alpha's journey titles so ported feature lines only change "journey" to
"procedure". Tool-bearing procedures get their tools associated and a
REEVALUATION relationship per tool, so the procedure is re-judged (and the
flow advances) right after a tool runs.
"""

from typing import Sequence

from pytest_bdd import given, parsers

from parlant.core.agents import AgentId
from parlant.core.common import JSONSerializable
from parlant.core import metadata as metadata_utils
from parlant.core.groups import GroupIds
from parlant.core.relationships import (
    RelationshipEntity,
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.rule_tool_associations import RuleToolAssociationStore
from parlant.core.rules import Rule, RuleStore
from parlant.core.indexing import PayloadOperation, RulePayload
from parlant.core.services.indexing.indexing_service import RuleIndexingProcessor
from parlant.core.tools import LocalToolService, ToolId

from tests.core.common.engines.compass.steps.tools import TOOLS
from tests.core.common.engines.compass.utils import step
from tests.core.common.utils import ContextOfTest


def _procedure(preamble: str, chart: str) -> str:
    return (
        f"{preamble}, follow this procedure step by step, tracking which step "
        f"the conversation is currently at:\n\n```mermaid\nflowchart TD\n{chart}\n```"
    )


# title -> (content, tool fixture keys)
PROCEDURES: dict[str, tuple[str, list[str]]] = {
    "Reset Password Journey": (
        _procedure(
            "When the customer wants to reset their password or can't remember it",
            """    s1["ask for their username"]
    s2["ask for their email address or phone number"]
    s3["wish them a good day"]
    s4["use the `reset_password` tool with the provided information"]
    s5["report the result to the customer"]
    s6["apologize and report that the password cannot be reset at this time"]
    s1 -->|"the customer provided their username"| s2
    s2 -->|"the customer provided their email address or phone number"| s3
    s3 -->|"the customer wished you a good day in return"| s4
    s3 -->|"the customer did not immediately wish you a good day in return"| s6
    s4 -->|"`reset_password` reported the password was successfully reset"| s5
    s4 -->|"`reset_password` reported failure"| s6""",
        ),
        ["reset_password"],
    ),
    "Book Flight": (
        _procedure(
            "When the customer wants to book a flight",
            """    s1["ask for the source and destination airport"]
    s2["ask for the dates of the departure and return flight"]
    s3["ask whether they want economy or business class"]
    s4["ask for the name of the traveler"]
    s5["book the flight using the `book_flight` tool and the provided details"]
    s1 --> s2
    s2 --> s3
    s3 --> s4
    s4 --> s5""",
        ),
        ["book_flight"],
    ),
    "Book Taxi Ride": (
        _procedure(
            "When the customer wants to book a taxi ride",
            """    s1["ask for the pickup location"]
    s2["ask for the drop-off location"]
    s3["ask for the desired pickup time"]
    s4["confirm all details with the customer before booking"]
    s1 --> s2
    s2 --> s3
    s3 --> s4""",
        ),
        [],
    ),
    "Place Food Order": (
        _procedure(
            "When the customer wants to order food",
            """    s1["ask if they'd like a salad or a sandwich"]
    s2["ask what kind of bread they'd like"]
    s3["ask what main filling they'd like from: peanut butter, jam or pesto"]
    s4["ask if they want any extras"]
    s5["ask what base greens they want"]
    s6["ask what toppings they'd like"]
    s7["ask what kind of dressing they prefer"]
    s8["confirm the full order before placing it"]
    s1 -->|"they choose a sandwich"| s2
    s2 --> s3
    s3 --> s4
    s4 --> s8
    s1 -->|"they choose a salad"| s5
    s5 --> s6
    s6 --> s7
    s7 --> s8""",
        ),
        [],
    ),
    "Book Hotel Journey": (
        _procedure(
            "When the customer wants to book a hotel",
            """    s1["ask which hotel the customer would like to stay in"]
    s2["ask what dates the customer would like to check in and check out"]
    s3["ask how many guests will be staying"]
    s4["ask if they need a specific type of room, like single, double, or suite"]
    s1 --> s2
    s2 --> s3
    s3 --> s4""",
        ),
        [],
    ),
    "Request Loan Journey": (
        _procedure(
            "When the customer is interested in applying for a loan",
            """    s1["ask what type of loan the customer is interested in"]
    s2["ask for the loan amount"]
    s3["ask for the purpose of the loan"]
    s4["ask for the account number for validation"]
    s5["validate the customer's eligibility with the `check_eligibility` tool"]
    s6["confirm eligibility with terms and ask to proceed with the application"]
    s7["explain that the loan request is denied due to ineligibility"]
    s1 -->|"the customer requested a personal loan"| s2
    s2 --> s3
    s3 --> s4
    s4 --> s5
    s5 -->|"the account is eligible"| s6
    s5 -->|"the account is not eligible"| s7""",
        ),
        ["check_eligibility"],
    ),
    "Change Credit Limits": (
        _procedure(
            "When the customer wants to change their credit limit, or says their current limit is too low",
            """    s1["ask for their account name"]
    s2["ask for the new desired credit limit"]
    s3["thank them and confirm the requested change"]
    s4["use the `get_credit_limit` tool with the provided account name to get the current limit"]
    s5["use the `change_credit_limit` tool with the provided account and desired limit"]
    s6["let the customer know the credit limit has been successfully updated"]
    s7["apologize and explain that the credit limit change cannot be done, and why"]
    s1 -->|"the customer provided their account name"| s2
    s2 -->|"the customer provided a desired credit limit"| s3
    s3 -->|"the customer confirmed the desired change"| s4
    s4 --> s5
    s5 -->|"`change_credit_limit` returned success"| s6
    s5 -->|"`change_credit_limit` returned that the limit cannot be changed"| s7""",
        ),
        ["get_credit_limit", "change_credit_limit"],
    ),
    "Lock Card Journey": (
        _procedure(
            "When the customer wants to lock their card",
            """    s1["use the `list_cards` tool to get the customer's cards"]
    s2["present the customer with their list of cards and ask which one to lock"]
    s3["ask for the reason for locking the card (e.g., lost, stolen, temporary lock)"]
    s4["ask them to call customer support at 123456789 to report the lost or stolen card"]
    s5["use the `lock_card` tool to lock the selected card"]
    s6["confirm whether or not the card has been locked successfully"]
    s1 --> s2
    s2 --> s3
    s3 -->|"the card is lost or stolen"| s4
    s3 -->|"otherwise"| s5
    s5 --> s6""",
        ),
        ["list_cards", "lock_card"],
    ),
    "Simple Lab Journey": (
        _procedure(
            "When the customer asks for their lab results",
            """    s1["use the `check_lab_results` tool to get the customer's lab results"]
    s2["congratulate the customer for their good results"]
    s3["tell the customer to contact the lab at 999-224-545 to get their results"]
    s1 -->|"the lab results are good (patient is healthy)"| s2
    s1 -->|"the lab results are negative"| s3""",
        ),
        ["check_lab_results"],
    ),
    "Complex Lab Journey": (
        _procedure(
            "When the customer requests blood test, plasma, or brain scan results",
            """    s0(["Start"])
    s1["use the `check_lab_results` tool to get the customer's blood test results"]
    s2["report the blood test results to the customer"]
    s3["ask the customer to call their personal doctor for the full results"]
    s4["tell the customer the results are not in yet, and ask them to check again later"]
    s0 -->|"the customer requested blood test results"| s1
    s1 --> s2
    s0 -->|"the customer requested plasma results"| s3
    s0 -->|"the customer requested brain scan results"| s4""",
        ),
        ["check_lab_results"],
    ),
}


def create_procedure_for_test(
    context: ContextOfTest,
    title: str,
    content: str,
    *,
    agent_id: AgentId,
    tool_keys: Sequence[str] = (),
) -> Rule:
    """Create a procedural policy the way production would: content-based with
    indexing propositions, tools associated, and a REEVALUATION relationship
    per tool so the flow advances right after its tools run."""
    metadata: dict[str, JSONSerializable] = {}
    signals: list[str] = []
    anti_signals: list[str] = []

    invoice = context.sync_await(
        context.container[RuleIndexingProcessor].process(
            payloads=[
                RulePayload(
                    content=content,
                    operation=PayloadOperation.ADD,
                    relativity_proposition=True,
                    signal_proposition=True,
                    agent_id=agent_id,
                )
            ],
        )
    )[0]

    if invoice.references_other_policies_proposition is not None:
        metadata = metadata_utils.set_internal_value(
            metadata,
            ("references_other_policies",),
            invoice.references_other_policies_proposition,
        )
    signals = list(invoice.signals_proposition or [])
    anti_signals = list(invoice.anti_signals_proposition or [])

    rule = context.sync_await(
        context.container[RuleStore].create_rule(
            content=content,
            title=title,
            metadata=metadata,
            signals=signals,
            anti_signals=anti_signals,
        )
    )
    context.sync_await(
        context.container[RuleStore].upsert_group(rule.id, GroupIds.for_agent_id(agent_id))
    )

    local_tool_service = context.container[LocalToolService]
    for key in tool_keys:
        tool = context.sync_await(local_tool_service.create_tool(**TOOLS[key]))
        tool_id = ToolId("local", tool.name)
        context.sync_await(
            context.container[RuleToolAssociationStore].create_association(
                rule_id=rule.id,
                tool_id=tool_id,
            )
        )
        context.sync_await(
            context.container[RelationshipStore].create_relationship(
                source=RelationshipEntity(id=rule.id, kind=RelationshipEntityKind.RULE),
                target=RelationshipEntity(id=tool_id, kind=RelationshipEntityKind.TOOL),
                kind=RelationshipKind.REEVALUATION,
            )
        )

    rule = context.sync_await(context.container[RuleStore].read_rule(rule.id))
    context.rules[title] = rule
    return rule


@step(given, parsers.parse('the procedure called "{title}"'))
def given_the_procedure_called(
    context: ContextOfTest,
    title: str,
    agent_id: AgentId,
) -> Rule:
    content, tool_keys = PROCEDURES[title]
    return create_procedure_for_test(
        context, title, content, agent_id=agent_id, tool_keys=tool_keys
    )


@step(
    given,
    parsers.parse('a procedure titled "{title}" to {do_something} when {a_condition_holds}'),
)
def given_a_procedure_to_when(
    context: ContextOfTest,
    title: str,
    do_something: str,
    a_condition_holds: str,
    agent_id: AgentId,
) -> Rule:
    """A minimal inline procedure - alpha's 'a journey titled X to Y when Z'."""
    content = f"When {a_condition_holds}, then {do_something}"
    return create_procedure_for_test(context, title, content, agent_id=agent_id)


@step(
    given,
    parsers.parse(
        'a dependency relationship between the policy "{policy_name}" and the "{procedure_title}" procedure'
    ),
)
def given_a_policy_procedure_dependency(
    context: ContextOfTest,
    policy_name: str,
    procedure_title: str,
) -> None:
    context.sync_await(
        context.container[RelationshipStore].create_relationship(
            source=RelationshipEntity(
                id=context.rules[policy_name].id,
                kind=RelationshipEntityKind.RULE,
            ),
            target=RelationshipEntity(
                id=context.rules[procedure_title].id,
                kind=RelationshipEntityKind.RULE,
            ),
            kind=RelationshipKind.DEPENDENCY,
        )
    )
