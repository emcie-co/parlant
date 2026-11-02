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

from typing import Optional

from parlant.core import metadata
from parlant.core.tools import ToolResult, cast_tool_argument


def test_that_a_native_bool_argument_is_passed_through() -> None:
    assert cast_tool_argument(bool, True) is True
    assert cast_tool_argument(bool, False) is False


def test_that_a_native_bool_argument_is_accepted_for_an_optional_bool_parameter() -> None:
    assert cast_tool_argument(Optional[bool], True) is True
    assert cast_tool_argument(Optional[bool], False) is False


def test_that_bool_strings_are_parsed_by_content_not_truthiness() -> None:
    assert cast_tool_argument(bool, "true") is True
    assert cast_tool_argument(bool, "True") is True
    assert cast_tool_argument(bool, "false") is False
    assert cast_tool_argument(bool, "False") is False


def test_that_bool_strings_are_accepted_for_an_optional_bool_parameter() -> None:
    assert cast_tool_argument(Optional[bool], "True") is True
    assert cast_tool_argument(Optional[bool], "false") is False


def test_that_an_authorization_required_result_tells_the_agent_and_carries_the_requirement() -> (
    None
):
    result = ToolResult.authorization_required(
        scopes=["flights:book"],
        reason="book your flight",
    )

    # The agent repeats what it's told, so it's told the reason in plain words;
    # scope names are for the app, not the customer.
    assert isinstance(result.data, str)
    assert "flights:book" not in result.data
    assert "book your flight" in result.data

    assert len(result.rules) == 1
    assert "sign in" in result.rules[0]["content"].lower()

    assert result.control.get("lifespan") == "response"

    assert metadata.internal_value(result.metadata, ("authorization_required",)) == {
        "scopes": ["flights:book"],
        "reason": "book your flight",
    }


def test_that_an_authorization_required_result_shows_the_agent_the_tools_own_message() -> None:
    result = ToolResult.authorization_required(
        scopes=["flights:book"],
        reason="book your flight",
        message="Booking needs the customer's account; their seat is held for 15 minutes.",
    )

    assert result.data == "Booking needs the customer's account; their seat is held for 15 minutes."
