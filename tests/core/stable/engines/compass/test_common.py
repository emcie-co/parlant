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

from dataclasses import replace

from parlant.core.agents import Effort
from parlant.core.common import Weight
from parlant.core.engines.rule_match import RuleMatch
from parlant.core.engines.compass.response_state import ResponseState

from tests.core.stable.engines.compass.matching.utils import (
    create_agent,
    create_engine_context,
    create_rule,
)


def test_that_effort_values_are_ordered_by_processing_depth() -> None:
    assert Effort.MIN < Effort.LOW < Effort.MEDIUM < Effort.HIGH < Effort.MAX
    assert max([Effort.LOW, Effort.HIGH, Effort.MEDIUM]) == Effort.HIGH


def test_that_dynamic_effort_is_agent_effort_when_no_matched_rule_has_effort() -> None:
    context = create_engine_context(
        conversation=[],
        agent=create_agent(),
    )
    context.state = ResponseState(agent_effort=context.agent.effort)

    assert context.state.dynamic_effort_level == Effort.MEDIUM


def test_that_dynamic_effort_uses_maximum_matched_rule_effort() -> None:
    context = create_engine_context(
        conversation=[],
        agent=replace(create_agent(), effort=Effort.LOW),
    )
    high_effort_rule = replace(
        create_rule("the user requests a regulated action"), effort=Effort.HIGH
    )
    low_effort_rule = replace(create_rule("the user greets the agent"), effort=Effort.MIN)
    context.state = ResponseState(
        agent_effort=context.agent.effort,
        ordinary_rule_matches=[
            RuleMatch(rule=high_effort_rule, rationale="relevant"),
            RuleMatch(rule=low_effort_rule, rationale="also relevant"),
        ],
    )

    assert context.state.dynamic_effort_level == Effort.HIGH


def test_that_high_criticality_detection_matches_ordinary_rules() -> None:
    context = create_engine_context(
        conversation=[],
        agent=create_agent(),
    )
    high_criticality_rule = replace(
        create_rule("the user requests a regulated action"),
        criticality=Weight.HIGH,
    )
    context.state = ResponseState(
        ordinary_rule_matches=[
            RuleMatch(rule=high_criticality_rule, rationale="relevant"),
        ],
    )

    assert context.state.has_matched_high_criticality_rules


def test_that_high_criticality_detection_matches_tool_enabled_rules() -> None:
    context = create_engine_context(
        conversation=[],
        agent=create_agent(),
    )
    high_criticality_rule = replace(
        create_rule("the user requests a regulated action"),
        criticality=Weight.HIGH,
    )
    context.state = ResponseState(
        tool_enabled_rule_matches={
            RuleMatch(rule=high_criticality_rule, rationale="relevant"): [],
        },
    )

    assert context.state.has_matched_high_criticality_rules


def test_that_high_criticality_detection_ignores_lower_criticality_rules() -> None:
    context = create_engine_context(
        conversation=[],
        agent=create_agent(),
    )
    medium_criticality_rule = replace(
        create_rule("the user greets the agent"),
        criticality=Weight.MEDIUM,
    )
    context.state = ResponseState(
        ordinary_rule_matches=[
            RuleMatch(rule=medium_criticality_rule, rationale="relevant"),
        ],
    )

    assert not context.state.has_matched_high_criticality_rules


def test_that_rule_match_properties_are_computed_once_until_invalidated() -> None:
    low_effort_rule = replace(
        create_rule("the user requests a simple action"),
        effort=Effort.LOW,
        criticality=Weight.LOW,
    )
    high_effort_rule = replace(
        create_rule("the user requests a regulated action"),
        effort=Effort.HIGH,
        criticality=Weight.HIGH,
    )
    state = ResponseState(
        agent_effort=Effort.LOW,
        ordinary_rule_matches=[
            RuleMatch(rule=low_effort_rule, rationale="relevant"),
        ],
    )

    assert state.dynamic_effort_level == Effort.LOW
    assert not state.has_matched_high_criticality_rules

    state.ordinary_rule_matches.append(RuleMatch(rule=high_effort_rule, rationale="newly relevant"))

    assert state.dynamic_effort_level == Effort.LOW
    assert not state.has_matched_high_criticality_rules

    state.invalidate_cached_properties()

    # Read into a freshly-typed local: the asserts above narrow
    # `dynamic_effort_level` to Literal[Effort.LOW], and mypy can't see that
    # invalidation recomputes it to HIGH.
    recomputed_effort: Effort = state.dynamic_effort_level
    assert recomputed_effort == Effort.HIGH
    assert state.has_matched_high_criticality_rules
