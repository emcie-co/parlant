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

from dataclasses import dataclass
from typing import Sequence, Union


@dataclass(frozen=True)
class CustomerMessage:
    """Represents a customer message in a conversation step."""

    message: str


@dataclass(frozen=True)
class AgentMessage:
    """Represents an expected agent message in a conversation step.

    Attributes:
        ideal: The ideal/reference response. Used as history content in subsequent tests.
        should: Assertion condition(s). Formatted as "The message should {should}".
                Can be a single string or a sequence of strings (run in parallel).
    """

    ideal: str
    should: Union[str, Sequence[str]]


@dataclass(frozen=True)
class StatusEvent:
    """Represents a status event in a conversation step."""

    status: str  # "typing", "ready", "processing", etc.


# Type alias for any step type
Step = Union[CustomerMessage, AgentMessage, StatusEvent]
