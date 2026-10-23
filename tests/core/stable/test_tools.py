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

from parlant.core.tools import cast_tool_argument


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
