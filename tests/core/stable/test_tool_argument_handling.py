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

from enum import Enum

from parlant.core.tools import split_arg_list, cast_tool_argument


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Size(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


# ---- split_arg_list: already-a-list passthrough ----


def test_that_split_arg_list_returns_list_unchanged_when_argument_is_already_a_list() -> None:
    result = split_arg_list(["a", "b", "c"], str)
    assert result == ["a", "b", "c"]


# ---- split_arg_list: str item type with brackets ----


def test_that_split_arg_list_handles_bracketed_string_list() -> None:
    result = split_arg_list("['hello', 'world']", str)
    assert result == ["hello", "world"]


# ---- split_arg_list: str item type without brackets (LLM hallucination) ----


def test_that_split_arg_list_handles_bracketless_string_list() -> None:
    result = split_arg_list("hello, world", str)
    assert result == ["hello", "world"]


def test_that_split_arg_list_handles_bracketless_single_string_item() -> None:
    result = split_arg_list("hello", str)
    assert result == ["hello"]


# ---- split_arg_list: Enum item type with brackets ----


def test_that_split_arg_list_handles_bracketed_enum_list() -> None:
    result = split_arg_list("['red', 'green']", Color)
    assert result == ["red", "green"]


# ---- split_arg_list: Enum item type without brackets (LLM hallucination) ----


def test_that_split_arg_list_handles_bracketless_enum_list() -> None:
    result = split_arg_list("red, green", Color)
    assert result == ["red", "green"]


def test_that_split_arg_list_handles_bracketless_enum_list_without_spaces() -> None:
    result = split_arg_list("red,green,blue", Color)
    assert result == ["red", "green", "blue"]


# ---- split_arg_list: int item type with brackets ----


def test_that_split_arg_list_handles_bracketed_int_list() -> None:
    result = split_arg_list("[1, 2, 3]", int)
    assert result == ["1", "2", "3"]


# ---- split_arg_list: int item type without brackets (LLM hallucination) ----


def test_that_split_arg_list_handles_bracketless_int_list() -> None:
    result = split_arg_list("1, 2, 3", int)
    assert result == ["1", "2", "3"]


def test_that_split_arg_list_handles_bracketless_int_list_without_spaces() -> None:
    result = split_arg_list("1,2,3", int)
    assert result == ["1", "2", "3"]


# ---- cast_tool_argument: list[int] without brackets ----


def test_that_cast_tool_argument_handles_bracketless_int_list() -> None:
    result = cast_tool_argument(list[int], "1, 2, 3")
    assert result == [1, 2, 3]


# ---- cast_tool_argument: list[str] without brackets ----


def test_that_cast_tool_argument_handles_bracketless_string_list() -> None:
    result = cast_tool_argument(list[str], "hello, world")
    assert result == ["hello", "world"]


# ---- cast_tool_argument: list[Enum] without brackets ----


def test_that_cast_tool_argument_handles_bracketless_enum_list() -> None:
    result = cast_tool_argument(list[Color], "red, green")
    assert result == [Color.RED, Color.GREEN]
