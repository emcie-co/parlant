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

import json
from datetime import datetime, timezone

from parlant.core.common import Version, to_prompt_json
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableId,
    ContextVariableValue,
    ContextVariableValueId,
)
from parlant.core.engines.compass.prompt_builder import EventAdaptationFormat, PromptBuilder
from parlant.core.rule_representation import context_variables_to_json
from parlant.core.sessions import Event, EventId, EventKind, EventSource


def test_that_a_version_compares_greater_or_equal_against_another_version() -> None:
    # Version defines __eq__, __lt__ and __gt__; Python derives neither >= nor
    # <= from those, so without these the operator raises TypeError on the
    # first caller that reaches for it.
    assert Version.from_string("0.7.0") >= Version.from_string("0.6.0")
    assert Version.from_string("0.6.0") >= Version.from_string("0.6.0")
    assert not Version.from_string("0.5.0") >= Version.from_string("0.6.0")


def test_that_a_version_compares_less_or_equal_against_another_version() -> None:
    assert Version.from_string("0.5.0") <= Version.from_string("0.6.0")
    assert Version.from_string("0.6.0") <= Version.from_string("0.6.0")
    assert not Version.from_string("0.7.0") <= Version.from_string("0.6.0")


def test_that_prompt_json_preserves_unicode_without_indentation() -> None:
    value = {
        "中文": "客户要求退货",
        "hebrew": "בקשת החזרה",
        "quoted": 'say "hello"\nthen continue',
        "emoji": "🙂",
    }

    result = to_prompt_json(value)

    assert json.loads(result) == value
    assert "客户要求退货" in result
    assert "בקשת החזרה" in result
    assert "🙂" in result
    assert "\\u5ba2" not in result
    assert "\n" not in result


def test_that_prompt_json_can_stringify_non_json_values_without_escaping_unicode() -> None:
    result = to_prompt_json({"value": object()}, default=lambda _: "לא זמין")

    assert json.loads(result) == {"value": "לא זמין"}
    assert "לא זמין" in result


def test_that_compass_shared_event_prompts_preserve_unicode() -> None:
    now = datetime.now(timezone.utc)
    event = Event(
        id=EventId("event-1"),
        source=EventSource.CUSTOMER,
        kind=EventKind.MESSAGE,
        creation_utc=now,
        modified_utc=now,
        offset=0,
        trace_id="trace-1",
        data={
            "message": "客户要求退货",
            "participant": {"display_name": "客户"},
        },
        metadata={},
        deleted=False,
    )

    result = PromptBuilder.adapt_event(event, format=EventAdaptationFormat.JSON)

    assert json.loads(result)["data"]["message"] == "客户要求退货"
    assert "客户要求退货" in result
    assert "\\u5ba2" not in result


def test_that_compass_shared_context_variable_prompts_preserve_unicode() -> None:
    now = datetime.now(timezone.utc)
    variable = ContextVariable(
        id=ContextVariableId("variable-1"),
        name="客户资料",
        description="退货资格",
        creation_utc=now,
        modified_utc=now,
        tool_id=None,
        freshness_rules=None,
        groups=[],
    )
    value = ContextVariableValue(
        id=ContextVariableValueId("value-1"),
        modified_utc=now,
        data={"状态": "符合条件"},
    )

    result = context_variables_to_json([(variable, value)])

    assert json.loads(result)["客户资料"]["value"] == {"状态": "符合条件"}
    assert "客户资料" in result
    assert "\\u5ba2" not in result
